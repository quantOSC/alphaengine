"""Public 2.0 coded-backtest adapter (`signal() -> target weights -> engine simulation`).

Implements `QOS-150` / `QOS-154` (unblocking hosted `QOS-117`, `QOS-121`, `QOS-123`):
- Researchers write `def signal(*, data_asof, params)` returning signed target
  portfolio weights. User code NEVER computes returns, fills, or P&L.
- `check_signal_source()` / `check_signal_module()` validate the AST contract
  and module import allowlist statically without executing user code (`0` trials
  recorded on static failure).
- `run_signal()` slices `data_asof` strictly up to bar `t` (`0 .. t`) at each
  rebalance step so look-ahead is structurally impossible, checks finite weights
  and gross/net/per-name leverage limits, and simulates through `run_backtest()`
  with next-open fills, mandatory positive transaction costs (`5.0` bps slippage,
  `1.0` bps commission defaults), short borrow accrual, and causal `adv_series`.
- Parameter grids are capped at `MAX_CODED_GRID = 400` combinations under family
  `coded:<idea_id>`. Every executed attempt (including runtime exceptions inside
  user `signal()`) receives a UUID `attempt_id` and counts as a trial (`QOS-153`).
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import math
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ._version import current_engine_version
from .core.backtest import run_backtest, score_backtest
from .core.conventions import (
    Verdict,
    build_annualization_spec,
    build_costs_model_spec,
    build_risk_free_spec,
)
from .core.performance import performance_report
from .core.validation import (
    deflated_sharpe,
    min_track_record_length,
    n_eff_participation,
    pbo_cscv,
)
from .manifest import build_run_manifest, hash_callable_or_module, hash_source_text
from .sweep.runner import Trial, _expand, _hash_data, _hash_params, _sharpe

MAX_CODED_GRID = 400
DEFAULT_CODED_SLIPPAGE_BPS = 5.0
DEFAULT_CODED_COMMISSION_BPS = 1.0
DEFAULT_CODED_BORROW_BPS = 50.0

ALLOWED_SIGNAL_IMPORTS: frozenset[str] = frozenset(
    {
        "__future__",
        "collections",
        "copy",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "fractions",
        "functools",
        "itertools",
        "math",
        "numbers",
        "operator",
        "re",
        "statistics",
        "types",
        "typing",
        "numpy",
        "scipy",
        "pandas",
        "statsmodels",
        "sklearn",
        "scikit_learn",
        "alphaengine",
    }
)

_ALLOWED_ALPHAENGINE_PREFIXES: tuple[str, ...] = (
    "alphaengine.core",
    "alphaengine.validation",
)

FORBIDDEN_BUILTINS: frozenset[str] = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "globals",
        "input",
        "locals",
        "open",
    }
)

__all__ = [
    "ALLOWED_SIGNAL_IMPORTS",
    "CodedSignalError",
    "DEFAULT_CODED_BORROW_BPS",
    "DEFAULT_CODED_COMMISSION_BPS",
    "DEFAULT_CODED_SLIPPAGE_BPS",
    "MAX_CODED_GRID",
    "Signal",
    "StaticCheckDiagnostic",
    "StaticCheckResult",
    "check_signal_module",
    "check_signal_source",
    "load_validated_signal",
    "run_signal",
]


class CodedSignalError(ValueError):
    """Raised when a coded signal fails static validation, weight invariants, or execution."""


@runtime_checkable
class Signal(Protocol):
    """Public 2.0 coded-backtest callable contract (`QOS-154`)."""

    def __call__(self, *, data_asof: Any, params: Mapping[str, Any]) -> Any: ...


@dataclass(frozen=True)
class StaticCheckDiagnostic:
    """Actionable static check diagnostic: what failed, where (line/col), and what to do (`QOS-121`)."""

    what: str
    line: int | None
    col: int | None
    remediation: str
    code: str = "contract_violation"

    def format(self) -> str:
        loc = (
            f"line {self.line}" + (f":{self.col}" if self.col is not None else "") if self.line else "module"
        )
        return f"{loc}: {self.what} What to do: {self.remediation}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "what": self.what,
            "line": self.line,
            "col": self.col,
            "remediation": self.remediation,
            "message": self.format(),
        }


@dataclass(frozen=True)
class StaticCheckResult:
    """Result of AST-only validation (`0` trials executed)."""

    ok: bool
    fn_name: str
    code_sha256: str | None
    diagnostics: list[StaticCheckDiagnostic] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "fn_name": self.fn_name,
            "code_sha256": self.code_sha256,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def _check_import_allowed(mod_name: str) -> tuple[bool, str]:
    clean = mod_name.strip()
    if not clean:
        return (
            False,
            "Relative imports without an explicit allowed package are not permitted in coded signals.",
        )
    top = clean.split(".", 1)[0]
    if top not in ALLOWED_SIGNAL_IMPORTS:
        return (
            False,
            f"Import `{clean}` is not allowed in coded signals "
            "(network, filesystem, and OS modules are blocked).",
        )
    if top == "alphaengine" and (
        clean in ("alphaengine", "alphaengine.auth")
        or not any(clean == p or clean.startswith(p + ".") for p in _ALLOWED_ALPHAENGINE_PREFIXES)
    ):
        return (
            False,
            (
                f"Import `{clean}` is blocked in coded signals. "
                "Bare `import alphaengine` and `alphaengine.auth` are forbidden "
                "(credentials security guard); import helpers from "
                "`alphaengine.core` or `alphaengine.validation` only."
            ),
        )
    return True, ""


def check_signal_source(
    source: str,
    *,
    fn_name: str = "signal",
    filename: str = "<signal>",
) -> StaticCheckResult:
    """Validate `source` statically via Python AST without executing user code (`0` trials)."""
    code_sha = hash_source_text(source)
    diags: list[StaticCheckDiagnostic] = []

    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        diags.append(
            StaticCheckDiagnostic(
                code="syntax_error",
                what=f"Python syntax error: {exc.msg}.",
                line=exc.lineno,
                col=exc.offset,
                remediation="Fix the syntax error at the indicated line before validating or running.",
            )
        )
        return StaticCheckResult(ok=False, fn_name=fn_name, code_sha256=code_sha, diagnostics=diags)

    fn_node: ast.FunctionDef | None = None
    for top_node in tree.body:
        if isinstance(top_node, ast.AsyncFunctionDef) and top_node.name == fn_name:
            diags.append(
                StaticCheckDiagnostic(
                    code="async_signal",
                    what=f"`{fn_name}` is declared `async def`, but coded signals must be synchronous.",
                    line=top_node.lineno,
                    col=top_node.col_offset,
                    remediation=(
                        f"Declare `def {fn_name}(*, data_asof, params):` as a regular synchronous function."
                    ),
                )
            )
        elif isinstance(top_node, ast.FunctionDef) and top_node.name == fn_name:
            fn_node = top_node

    if fn_node is None:
        diags.append(
            StaticCheckDiagnostic(
                code="missing_signal_fn",
                what=f"No top-level `def {fn_name}(*, data_asof, params)` function found.",
                line=1,
                col=0,
                remediation=(
                    f"Define `def {fn_name}(*, data_asof, params) -> dict[str, float]:` "
                    "returning signed target weights."
                ),
            )
        )
    else:
        args = fn_node.args
        kw_capable = {a.arg for a in args.args} | {a.arg for a in args.kwonlyargs}
        has_varkw = args.kwarg is not None
        missing_kw = [req for req in ("data_asof", "params") if req not in kw_capable and not has_varkw]
        if missing_kw:
            diags.append(
                StaticCheckDiagnostic(
                    code="invalid_signature",
                    what=f"`{fn_name}()` is missing required keyword parameter(s): {', '.join(missing_kw)}.",
                    line=fn_node.lineno,
                    col=fn_node.col_offset,
                    remediation=f"Change the signature to `def {fn_name}(*, data_asof, params):`.",
                )
            )
        required_pos = len(args.args) - len(args.defaults)
        extra_pos = [a.arg for a in args.args[:required_pos] if a.arg not in ("data_asof", "params")]
        if extra_pos or args.posonlyargs:
            diags.append(
                StaticCheckDiagnostic(
                    code="unexpected_positional_args",
                    what=f"`{fn_name}()` requires positional argument(s) the adapter does not pass.",
                    line=fn_node.lineno,
                    col=fn_node.col_offset,
                    remediation=(
                        f"Accept only `data_asof` and `params` (e.g. `def {fn_name}(*, data_asof, params):`)."
                    ),
                )
            )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                allowed, why = _check_import_allowed(alias.name)
                if not allowed:
                    diags.append(
                        StaticCheckDiagnostic(
                            code="forbidden_import",
                            what=why,
                            line=node.lineno,
                            col=node.col_offset,
                            remediation=(
                                "Remove the blocked import and rely only on `data_asof`, `params`, "
                                "`numpy`, `scipy`, `pandas`, or `alphaengine.core`."
                            ),
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                diags.append(
                    StaticCheckDiagnostic(
                        code="relative_import",
                        what="Relative imports are not permitted in standalone coded signals.",
                        line=node.lineno,
                        col=node.col_offset,
                        remediation="Keep signal logic self-contained and import only allowlisted libraries.",
                    )
                )
                continue
            base_mod = node.module or ""
            allowed, why = _check_import_allowed(base_mod)
            if not allowed:
                diags.append(
                    StaticCheckDiagnostic(
                        code="forbidden_import",
                        what=why,
                        line=node.lineno,
                        col=node.col_offset,
                        remediation=(
                            "Remove the blocked import and rely only on `data_asof`, `params`, "
                            "`numpy`, `scipy`, `pandas`, or `alphaengine.core`."
                        ),
                    )
                )
            elif base_mod == "alphaengine":
                for alias in node.names:
                    sub = f"alphaengine.{alias.name}"
                    sub_ok, sub_why = _check_import_allowed(sub)
                    if not sub_ok and alias.name not in ("Verdict",):
                        diags.append(
                            StaticCheckDiagnostic(
                                code="forbidden_import",
                                what=sub_why,
                                line=node.lineno,
                                col=node.col_offset,
                                remediation=(
                                    "Import math helpers from `alphaengine.core` "
                                    "or `alphaengine.validation` only."
                                ),
                            )
                        )
        elif isinstance(node, ast.Call):
            called_name: str | None = None
            if isinstance(node.func, ast.Name):
                called_name = node.func.id
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "builtins"
            ):
                called_name = node.func.attr
            if called_name in FORBIDDEN_BUILTINS:
                diags.append(
                    StaticCheckDiagnostic(
                        code="forbidden_builtin",
                        what=f"Call to `{called_name}()` is blocked in coded signals.",
                        line=node.lineno,
                        col=node.col_offset,
                        remediation=(
                            f"Remove `{called_name}()` - signals must be pure functions "
                            "of `data_asof` and `params`."
                        ),
                    )
                )
            elif called_name in (
                "read_csv",
                "read_table",
                "read_parquet",
                "read_json",
                "read_excel",
                "read_sql",
            ):
                for arg_node in list(node.args) + [kw.value for kw in node.keywords]:
                    if isinstance(arg_node, ast.Constant) and isinstance(arg_node.value, str):
                        s_val = arg_node.value.lower().strip()
                        if s_val.startswith(("http://", "https://", "ftp://", "s3://", "gs://")):
                            diags.append(
                                StaticCheckDiagnostic(
                                    code="forbidden_io_url",
                                    what=(
                                        f"Loading data from remote URL {arg_node.value!r} in "
                                        f"`{called_name}()` is blocked. "
                                        "Coded signals must be pure and offline."
                                    ),
                                    line=node.lineno,
                                    col=node.col_offset,
                                    remediation=(
                                        "Signals must be pure offline functions of `data_asof` and `params`."
                                    ),
                                )
                            )
                    elif isinstance(arg_node, ast.Name) and any(
                        u in arg_node.id.lower() for u in ("url", "http", "ftp")
                    ):
                        diags.append(
                            StaticCheckDiagnostic(
                                code="forbidden_io_url",
                                what=(
                                    f"URL variable `{arg_node.id}` in "
                                    f"`{called_name}()` is blocked in coded signals."
                                ),
                                line=node.lineno,
                                col=node.col_offset,
                                remediation=(
                                    "Signals must be pure offline functions of `data_asof` and `params`."
                                ),
                            )
                        )

    return StaticCheckResult(
        ok=len(diags) == 0,
        fn_name=fn_name,
        code_sha256=code_sha,
        diagnostics=diags,
    )


def _split_target_spec(target: str | Path) -> tuple[Path | None, str | None, str]:
    if isinstance(target, Path):
        return target.expanduser().resolve(), None, "signal"
    text = str(target).strip()
    fn_name = "signal"
    # Handle Windows drive letters (`C:\path\file.py:my_fn` vs `C:\path\file.py`)
    if ":" in text:
        head, tail = text.rsplit(":", 1)
        if tail.isidentifier() and not (len(head) == 1 and head.isalpha()):
            text = head.strip()
            fn_name = tail.strip()
    cand_path = Path(text).expanduser()
    if cand_path.suffix == ".py" or cand_path.is_file():
        return cand_path.resolve(), None, fn_name
    return None, text, fn_name


def _resolve_source_file(target: str | Path) -> tuple[Path, str]:
    file_path, mod_name, fn_name = _split_target_spec(target)
    if file_path is not None:
        if not file_path.is_file():
            raise FileNotFoundError(f"Signal file not found: {file_path}")
        return file_path, fn_name
    assert mod_name is not None
    spec = importlib.util.find_spec(mod_name)
    if spec is None or not spec.origin or not Path(spec.origin).is_file():
        raise ModuleNotFoundError(f"Cannot locate Python source file for module `{mod_name}`.")
    return Path(spec.origin).resolve(), fn_name


def check_signal_module(target: str | Path) -> StaticCheckResult:
    """Validate a `.py` file or dotted module path (`path.py[:fn]` / `pkg.mod[:fn]`) statically."""
    try:
        src_path, fn_name = _resolve_source_file(target)
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        diag = StaticCheckDiagnostic(
            code="source_not_found",
            what=str(exc),
            line=None,
            col=None,
            remediation="Pass an existing `.py` file path or importable module name (optionally `:fn_name`).",
        )
        return StaticCheckResult(ok=False, fn_name="signal", code_sha256=None, diagnostics=[diag])
    source = src_path.read_text(encoding="utf-8")
    return check_signal_source(source, fn_name=fn_name, filename=str(src_path))


def load_validated_signal(target: str | Path | Callable[..., Any]) -> tuple[Callable[..., Any], str | None]:
    """Validate and load a `Signal` callable, returning `(callable, code_sha256)`."""
    if callable(target) and not isinstance(target, (str, Path)):
        import inspect

        try:
            src = inspect.getsource(target)
        except (OSError, TypeError):
            src = None
        if src is not None:
            fn_name = getattr(target, "__name__", "signal")
            # Dedent nested functions before AST validation when defined inside a function/test
            import textwrap

            dedented = textwrap.dedent(src)
            check = check_signal_source(dedented, fn_name=fn_name)
            if not check.ok:
                msgs = "; ".join(d.format() for d in check.diagnostics)
                raise CodedSignalError(f"Signal static check failed: {msgs}")
            return target, check.code_sha256
        # Lambda or C/dynamic callable: verify keyword signature via inspect.signature
        sig = inspect.signature(target)
        kw_names = {
            p.name
            for p in sig.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if not has_varkw and not {"data_asof", "params"}.issubset(kw_names):
            raise CodedSignalError(
                "Signal callable must accept keyword arguments `data_asof` and `params` "
                "(define `def signal(*, data_asof, params):`)."
            )
        return target, hash_callable_or_module(target)

    check = check_signal_module(target)
    if not check.ok:
        msgs = "; ".join(d.format() for d in check.diagnostics)
        raise CodedSignalError(f"Signal static validation failed: {msgs}")

    src_path, fn_name = _resolve_source_file(target)
    mod_id = f"_alphaengine_coded_{check.code_sha256[:12] if check.code_sha256 else uuid.uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(mod_id, src_path)
    if spec is None or spec.loader is None:
        raise CodedSignalError(f"Could not create module spec from {src_path}.")
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, fn_name, None)
    if not callable(fn):
        raise CodedSignalError(f"`{fn_name}` in {src_path} is not callable.")
    return fn, check.code_sha256


def _extract_prices_and_dates(
    prices: Mapping[str, Any],
) -> tuple[list[str], int, list[str] | None, dict[str, list[float]]]:
    """Inspect `{ticker: series}` and return `(tickers, n_bars, aligned_dates, close_lists)`."""
    if not isinstance(prices, Mapping) or not prices:
        raise CodedSignalError("`prices` must be a non-empty mapping `{ticker: series}`.")
    tickers = sorted(str(k) for k in prices)
    lengths: list[int] = []
    date_sets: list[list[str]] = []
    close_lists: dict[str, list[float]] = {}

    for tk in tickers:
        raw = prices[tk]
        if isinstance(raw, Mapping):
            ks = sorted(str(d) for d in raw)
            vals: list[float] = []
            for d_key in ks:
                cell = raw[d_key]
                if isinstance(cell, Mapping):
                    c_val = cell.get("close", cell.get("Close", cell.get("price")))
                    vals.append(float(c_val) if c_val is not None else 0.0)
                else:
                    vals.append(float(cell))
            lengths.append(len(ks))
            date_sets.append(ks)
            close_lists[tk] = vals
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            seq = list(raw)
            lengths.append(len(seq))
            if seq and isinstance(seq[0], Mapping):
                if "date" in seq[0] or "Date" in seq[0]:
                    date_sets.append([str(row.get("date") or row.get("Date")) for row in seq])
                close_lists[tk] = [
                    float(row.get("close", row.get("Close", row.get("price", 0.0)))) for row in seq
                ]
            else:
                close_lists[tk] = [float(x) for x in seq]
        else:
            arr = np.asarray(raw, dtype=float)
            lengths.append(int(arr.size))
            close_lists[tk] = [float(x) for x in arr.reshape(-1).tolist()]

    n_bars = min(lengths) if lengths else 0
    if n_bars < 2:
        raise CodedSignalError(f"`prices` needs at least 2 aligned bars (got {n_bars}).")

    for tk in tickers:
        close_lists[tk] = close_lists[tk][:n_bars]

    dates: list[str] | None = None
    if date_sets and all(len(ds) >= n_bars for ds in date_sets):
        dates = date_sets[0][:n_bars]
    return tickers, n_bars, dates, close_lists


def _slice_value_asof(val: Any, end_inclusive: int, allowed_dates: set[str] | None) -> Any:
    """Slice a single series/panel strictly up to index `end_inclusive` (`0 .. end_inclusive`)."""
    cut = end_inclusive + 1
    if isinstance(val, Mapping):
        # Could be `{date: scalar}` or nested `{ticker: series}`
        sample_val = next(iter(val.values()), None)
        if isinstance(sample_val, (int, float)) or sample_val is None:
            sorted_keys = sorted(str(k) for k in val)
            if allowed_dates is not None:
                kept = [k for k in sorted_keys if k in allowed_dates][:cut]
            else:
                kept = sorted_keys[:cut]
            return {k: val[k] for k in kept if k in val}
        return {str(k): _slice_value_asof(v, end_inclusive, allowed_dates) for k, v in val.items()}
    if isinstance(val, np.ndarray):
        return val[:cut].copy()
    if isinstance(val, list):
        return list(val[:cut])
    if isinstance(val, tuple):
        return tuple(val[:cut])
    return val


def _extract_bar_weights(
    raw_out: Any,
    *,
    tickers: Sequence[str],
    bar_index: int,
    bar_date: str | None,
    max_abs_weight: float,
    max_gross_leverage: float,
    max_net_leverage: float,
) -> dict[str, float]:
    """Validate and extract `{ticker: signed_weight}` at `bar_index` from user `signal()` output."""
    if raw_out is None:
        raise CodedSignalError(
            f"`signal()` returned None at bar {bar_index}; expected signed target weights."
        )

    ticker_set = set(tickers)
    extracted: dict[str, float] = {}

    # Support pandas DataFrame / Series via duck typing without requiring pandas installed
    if hasattr(raw_out, "to_dict") and hasattr(raw_out, "columns"):
        # DataFrame[date x ticker]
        idx_list = [str(x) for x in list(raw_out.index)]
        if bar_date is not None and any(d > bar_date for d in idx_list):
            raise CodedSignalError(f"`signal()` returned future timestamp beyond as-of date {bar_date!r}.")
        if len(idx_list) == 0:
            extracted = {}
        else:
            last_row = raw_out.iloc[-1]
            row_map = dict(last_row.to_dict())
            for k, v in row_map.items():
                sk = str(k)
                if sk not in ticker_set:
                    allowed_t = sorted(ticker_set)
                    raise CodedSignalError(
                        f"Unknown ticker {sk!r} in signal output. Allowed universe tickers are: {allowed_t}"
                    )
                extracted[sk] = float(v)
    elif isinstance(raw_out, Mapping):
        for k, v in raw_out.items():
            sk = str(k)
            if sk not in ticker_set:
                allowed_t = sorted(ticker_set)
                raise CodedSignalError(
                    f"Unknown ticker {sk!r} in signal output. Allowed universe tickers are: {allowed_t}"
                )
            if isinstance(v, Mapping):
                # `{ticker: {date: weight}}`
                d_keys = sorted(str(dk) for dk in v)
                if bar_date is not None and any(dk > bar_date for dk in d_keys):
                    raise CodedSignalError(f"`signal()` emitted weight dated after {bar_date!r} for {sk}.")
                if d_keys:
                    chosen_d = bar_date if (bar_date is not None and bar_date in v) else d_keys[-1]
                    extracted[sk] = float(v[chosen_d])
            elif isinstance(v, (Sequence, np.ndarray)) and not isinstance(v, (str, bytes)):
                seq = list(v)
                if len(seq) > bar_index + 1:
                    raise CodedSignalError(
                        f"`signal()` returned {len(seq)} weights at bar {bar_index} "
                        f"(expected <= {bar_index + 1})."
                    )
                if seq:
                    extracted[sk] = float(seq[-1])
            else:
                extracted[sk] = float(v)
    else:
        raise CodedSignalError(
            f"`signal()` must return a dict of `{{ticker: weight}}` or dated weight panel, "
            f"got {type(raw_out).__name__}."
        )

    gross = 0.0
    net = 0.0
    out_weights: dict[str, float] = {tk: 0.0 for tk in tickers}
    for tk, w in extracted.items():
        if not math.isfinite(w):
            raise CodedSignalError(
                f"`signal()` returned non-finite weight {w!r} for {tk} at bar {bar_index}."
            )
        if abs(w) > max_abs_weight + 1e-9:
            raise CodedSignalError(
                f"`signal()` weight {w:.4f} for {tk} exceeds "
                f"`max_abs_weight={max_abs_weight}` at bar {bar_index}."
            )
        out_weights[tk] = float(w)
        gross += abs(float(w))
        net += float(w)

    if gross > max_gross_leverage + 1e-9:
        raise CodedSignalError(
            f"`signal()` gross leverage {gross:.4f} exceeds "
            f"`max_gross_leverage={max_gross_leverage}` at bar {bar_index}."
        )
    if abs(net) > max_net_leverage + 1e-9:
        raise CodedSignalError(
            f"`signal()` net exposure {abs(net):.4f} exceeds "
            f"`max_net_leverage={max_net_leverage}` at bar {bar_index}."
        )
    return out_weights


def _simulate_one_param_set(
    signal_fn: Callable[..., Any],
    *,
    prices: Mapping[str, Any],
    close_lists: Mapping[str, list[float]],
    aux_data: Mapping[str, Any] | None,
    tickers: list[str],
    n_bars: int,
    dates: list[str] | None,
    params: Mapping[str, Any],
    rebalance_every: int,
    slippage_bps: float,
    commission_bps: float,
    borrow_bps: float,
    initial_capital: float,
    adv: Mapping[str, float] | float | None,
    adv_series: Mapping[str, Any] | None,
    impact_coef: float,
    max_participation: float,
    max_abs_weight: float,
    max_gross_leverage: float,
    max_net_leverage: float,
    periods_per_year: int,
) -> dict[str, Any]:
    """Run causal `data_asof` weight generation + `run_backtest` for one parameter dict."""
    weight_panel: dict[str, list[dict[str, Any]]] = {tk: [] for tk in tickers}
    last_weights: dict[str, float] = {tk: 0.0 for tk in tickers}
    reb = max(int(rebalance_every), 1)

    for t in range(n_bars):
        bar_date = dates[t] if dates is not None else f"{t:06d}"
        if t % reb == 0:
            allowed_dates = set(dates[: t + 1]) if dates is not None else None
            prices_asof = {tk: list(close_lists[tk][: t + 1]) for tk in tickers}
            if aux_data:
                data_asof: dict[str, Any] = {
                    k: _slice_value_asof(v, t, allowed_dates) for k, v in aux_data.items() if k != "prices"
                }
                data_asof["prices"] = prices_asof
            else:
                data_asof = {"prices": prices_asof}
            data_asof["bars"] = _slice_value_asof(prices, t, allowed_dates)
            if dates is not None:
                data_asof["as_of"] = dates[t]
                data_asof["dates"] = list(dates[: t + 1])

            raw_w = signal_fn(data_asof=data_asof, params=dict(params))
            last_weights = _extract_bar_weights(
                raw_w,
                tickers=tickers,
                bar_index=t,
                bar_date=dates[t] if dates is not None else None,
                max_abs_weight=max_abs_weight,
                max_gross_leverage=max_gross_leverage,
                max_net_leverage=max_net_leverage,
            )
        for tk in tickers:
            weight_panel[tk].append({"date": bar_date, "target_weight": last_weights[tk]})

    adv_map: dict[str, float] | None = None
    if isinstance(adv, Mapping):
        adv_map = {str(k): float(v) for k, v in adv.items()}
    elif isinstance(adv, (int, float)) and not isinstance(adv, bool):
        adv_map = {tk: float(adv) for tk in tickers}

    bt_prices: dict[str, Any] = (
        dict(prices)
        if dates is not None
        else {tk: {f"{t:06d}": close_lists[tk][t] for t in range(n_bars)} for tk in tickers}
    )

    return run_backtest(
        weight_panel,
        bt_prices,
        slippage_bps=slippage_bps,
        commission_bps=commission_bps,
        borrow_bps=borrow_bps,
        fill_timing="next_open",
        initial_capital=initial_capital,
        max_position_pct=max_abs_weight,
        adv=adv_map,
        adv_series=dict(adv_series) if isinstance(adv_series, Mapping) else None,
        impact_coef=impact_coef,
        max_participation=max_participation,
        liquidate_final=True,
        periods_per_year=periods_per_year,
    )


def run_signal(
    signal_or_target: str | Path | Callable[..., Any],
    *,
    prices: Mapping[str, Any] | None = None,
    data: Mapping[str, Any] | None = None,
    params: Mapping[str, Any] | None = None,
    grid: Mapping[str, Iterable[Any]] | None = None,
    idea_id: str | None = None,
    family: str | None = None,
    rebalance_every: int = 1,
    slippage_bps: float = DEFAULT_CODED_SLIPPAGE_BPS,
    commission_bps: float = DEFAULT_CODED_COMMISSION_BPS,
    borrow_bps: float = DEFAULT_CODED_BORROW_BPS,
    initial_capital: float = 1_000_000.0,
    adv: Mapping[str, float] | float | None = None,
    adv_series: Mapping[str, Any] | None = None,
    impact_coef: float = 0.1,
    max_participation: float = 0.05,
    max_abs_weight: float = 1.0,
    max_gross_leverage: float = 2.0,
    max_net_leverage: float = 1.5,
    risk_free_rate: float | None = None,
    periods_per_year: int = 252,
    allow_zero_costs_unverified: bool = False,
    record_in_project: bool = True,
) -> dict[str, Any]:
    """Validate and simulate a coded `signal(*, data_asof, params)` strategy (`QOS-154`).

    - Static AST failures and grids exceeding `MAX_CODED_GRID = 400` raise
      `CodedSignalError` before execution with **`0` trials recorded**.
    - Transaction costs are mandatory (`slippage_bps > 0` and `commission_bps > 0`,
      defaults `5.0` and `1.0` bps) unless `allow_zero_costs_unverified=True`.
    - Once execution begins, every attempted configuration receives a UUID
      `attempt_id` and counts toward `n_trials` under family `coded:<idea_id>`
      (even when user `signal()` raises at runtime).
    """
    # 1. Pre-execution static & cost/grid gates (`0` trials on failure)
    if (float(slippage_bps) <= 0.0 or float(commission_bps) <= 0.0) and not allow_zero_costs_unverified:
        raise CodedSignalError(
            "Coded backtests require positive transaction costs (default 5.0 bps slippage, "
            "1.0 bps commission); running or saving with zero costs is refused."
        )

    resolved_prices: Mapping[str, Any] | None = prices
    aux_data: Mapping[str, Any] | None = data
    if resolved_prices is None and isinstance(data, Mapping):
        if isinstance(data.get("prices"), Mapping):
            resolved_prices = data["prices"]
        else:
            resolved_prices = data
            aux_data = None
    if resolved_prices is None:
        raise CodedSignalError(
            "`run_signal()` requires `prices={ticker: series}` (or `data={'prices': ...}`)."
        )

    tickers, n_bars, dates, close_lists = _extract_prices_and_dates(resolved_prices)

    combos: list[dict[str, Any]]
    is_grid = grid is not None
    if grid is not None:
        combos = _expand(grid)
        if not combos:
            raise CodedSignalError("Coded signal `grid` is empty: nothing to evaluate.")
        if len(combos) > MAX_CODED_GRID:
            raise CodedSignalError(
                f"Coded signal grid has {len(combos)} combinations, exceeding the v1 cap of "
                f"{MAX_CODED_GRID} (`QOS-154`). Narrow the parameter grid before running."
            )
    else:
        combos = [dict(params or {})]

    signal_fn, code_sha256 = load_validated_signal(signal_or_target)

    ppy = max(int(periods_per_year), 1)
    clean_idea = (idea_id or "").strip() or None
    resolved_family = (family or "").strip() or (f"coded:{clean_idea}" if clean_idea else "coded")
    data_hash = _hash_data(dict(resolved_prices))
    adv_source = "adv_series_asof" if adv_series is not None else ("static_adv" if adv is not None else None)
    costs_spec = build_costs_model_spec(
        mode="engine_simulated_weights" if not allow_zero_costs_unverified else "zero_costs_unverified",
        slippage_bps=float(slippage_bps),
        commission_bps=float(commission_bps),
        borrow_bps=float(borrow_bps),
        fill_timing="next_open",
        adv_source=adv_source,
        liquidate_final=True,
    )

    # 2. Execute each attempt (every attempt from here on counts as a trial!)
    trials: list[Trial] = []
    bt_results: list[dict[str, Any] | None] = []
    return_cols: list[np.ndarray[Any, Any]] = []

    for idx, combo_params in enumerate(combos):
        merged_params = {**dict(params or {}), **combo_params} if is_grid else dict(combo_params)
        aid = str(uuid.uuid4())
        exec_at = datetime.now(timezone.utc).isoformat()
        try:
            bt = _simulate_one_param_set(
                signal_fn,
                prices=resolved_prices,
                close_lists=close_lists,
                aux_data=aux_data,
                tickers=tickers,
                n_bars=n_bars,
                dates=dates,
                params=merged_params,
                rebalance_every=rebalance_every,
                slippage_bps=float(slippage_bps),
                commission_bps=float(commission_bps),
                borrow_bps=float(borrow_bps),
                initial_capital=float(initial_capital),
                adv=adv,
                adv_series=adv_series,
                impact_coef=float(impact_coef),
                max_participation=float(max_participation),
                max_abs_weight=float(max_abs_weight),
                max_gross_leverage=float(max_gross_leverage),
                max_net_leverage=float(max_net_leverage),
                periods_per_year=ppy,
            )
            r_arr = np.asarray(bt["returns"], dtype=float)
            sr = _sharpe(r_arr, risk_free_rate=risk_free_rate, periods_per_year=ppy)
            trials.append(
                Trial(
                    index=idx,
                    params=merged_params,
                    params_hash=_hash_params(merged_params),
                    n_obs=int(r_arr.size),
                    sharpe=round(sr, 6),
                    sharpe_annualized=round(sr * math.sqrt(ppy), 4),
                    total_return_pct=float(bt["total_return_pct"]),
                    attempt_id=aid,
                    executed_at=exec_at,
                )
            )
            bt_results.append(bt)
            return_cols.append(r_arr)
        except Exception as exc:  # noqa: BLE001
            trials.append(
                Trial(
                    index=idx,
                    params=merged_params,
                    params_hash=_hash_params(merged_params),
                    n_obs=0,
                    sharpe=0.0,
                    sharpe_annualized=0.0,
                    total_return_pct=0.0,
                    failed=f"{type(exc).__name__}: {exc}",
                    attempt_id=aid,
                    executed_at=exec_at,
                )
            )
            bt_results.append(None)

    trial_ledger = [
        {
            "attempt_id": t.attempt_id,
            "executed_at": t.executed_at,
            "index": t.index,
            "params_hash": t.params_hash,
            "status": "failed" if t.failed is not None else "ok",
            "n_obs": t.n_obs,
            "sharpe": None if t.failed is not None else t.sharpe,
            "sharpe_annualized": None if t.failed is not None else t.sharpe_annualized,
            "total_return_pct": None if t.failed is not None else t.total_return_pct,
            "failed": t.failed,
        }
        for t in trials
    ]

    total_attempted = len(trials)
    n_source = "derived_from_grid" if is_grid else "coded_single_run"
    manifest_params: dict[str, Any] = {
        "params": dict(params or {}),
        "grid": {str(k): list(v) for k, v in grid.items()} if grid is not None else None,
        "rebalance_every": int(rebalance_every),
        "slippage_bps": float(slippage_bps),
        "commission_bps": float(commission_bps),
        "borrow_bps": float(borrow_bps),
        "max_abs_weight": float(max_abs_weight),
        "max_gross_leverage": float(max_gross_leverage),
        "max_net_leverage": float(max_net_leverage),
    }
    manifest = build_run_manifest(
        params=manifest_params,
        dataset_hashes=[data_hash],
        code_sha256=code_sha256,
        as_of=dates[-1] if dates else None,
        horizon={
            "start": dates[0] if dates else 0,
            "end": dates[-1] if dates else (n_bars - 1),
            "n_bars": n_bars,
        },
        costs_model=costs_spec,
        periods_per_year=ppy,
        risk_free_rate=risk_free_rate,
    )

    ok_indices = [i for i, bt in enumerate(bt_results) if bt is not None]
    if not ok_indices:
        if record_in_project:
            _record_coded_in_project(
                command="coded_signal",
                params=manifest_params,
                dataset_hashes=[data_hash],
                n_trials=total_attempted,
                n_trials_source=n_source,
                figures={
                    "n_trials": total_attempted,
                    "run_n_trials": total_attempted,
                    "n_ok_trials": 0,
                    "n_failed_trials": total_attempted,
                    "n_trials_source": n_source,
                    "failed": trials[0].failed,
                },
                verdict=Verdict.UNVERIFIED.value,
                idea_id=clean_idea,
                family=resolved_family,
                repro_spec=None,
                returns_for_deflation=None,
                trials_sharpe_std=None,
                risk_free_rate=risk_free_rate,
                periods_per_year=ppy,
                manifest=manifest,
                trial_ledger=trial_ledger,
            )
        raise CodedSignalError(
            f"Coded signal execution failed across all {total_attempted} attempt(s). "
            f"First failure: {trials[0].failed}"
        )

    best_idx = max(ok_indices, key=lambda i: trials[i].sharpe)
    best_bt = bt_results[best_idx]
    assert best_bt is not None
    best_trial = trials[best_idx]
    best_rets = list(best_bt["returns"])

    trials_std: float | None = None
    if len(ok_indices) >= 2:
        disp = float(np.std([trials[i].sharpe for i in ok_indices], ddof=1))
        if math.isfinite(disp) and disp > 0:
            trials_std = disp

    scored = score_backtest(
        best_bt,
        n_trials=total_attempted,
        n_trials_source=n_source,
        initial_capital=float(initial_capital),
        risk_free_rate=risk_free_rate,
        periods_per_year=ppy,
    )
    dsr = deflated_sharpe(
        best_rets,
        n_trials=total_attempted,
        trials_sharpe_std=trials_std,
        risk_free_rate=risk_free_rate,
        periods_per_year=ppy,
    )
    perf = performance_report(best_rets, risk_free_rate=risk_free_rate, periods_per_year=ppy)
    mintrl = min_track_record_length(best_rets, risk_free_rate=risk_free_rate, periods_per_year=ppy)

    selection: dict[str, Any] | None = None
    n_eff: float | None = None
    if len(return_cols) >= 2:
        mat = np.column_stack(return_cols)
        n_eff = round(n_eff_participation(mat), 4)
        selection = pbo_cscv(mat.tolist())

    scored_sealable = bool(scored.get("sealable", True))
    validation = scored.get("validation") if isinstance(scored.get("validation"), dict) else {}
    scored_reason = validation.get("verdict_reason")
    borrow_withheld = (not scored_sealable) or (
        str(scored_reason or "") == "borrow not supplied for short positions"
    )
    if allow_zero_costs_unverified or borrow_withheld:
        final_verdict = Verdict.UNVERIFIED.value
    else:
        final_verdict = str(dsr.get("verdict") or validation.get("verdict") or Verdict.MARGINAL.value)
    can_seal = (not allow_zero_costs_unverified) and scored_sealable
    verdict_reason = (
        "borrow not supplied for short positions"
        if borrow_withheld
        else ("zero costs unverified" if allow_zero_costs_unverified else None)
    )
    deflation_applied = (
        False if (allow_zero_costs_unverified or borrow_withheld) else dsr.get("deflation_applied")
    )

    figures: dict[str, Any] = {
        "engine_verified": not allow_zero_costs_unverified,
        "can_seal": can_seal,
        "sealable": scored_sealable and not allow_zero_costs_unverified,
        "had_short_positions": bool(scored.get("had_short_positions")),
        "verdict_reason": verdict_reason,
        "n_trials": total_attempted,
        "run_n_trials": total_attempted,
        "n_ok_trials": len(ok_indices),
        "n_failed_trials": total_attempted - len(ok_indices),
        "n_trials_source": n_source,
        "deflation_applied": deflation_applied,
        "dsr_probability": dsr.get("dsr_probability"),
        "deflated_sharpe": dsr.get("deflated_sharpe"),
        "psr_vs_zero": dsr.get("psr_vs_zero"),
        "sr0_expected_max": dsr.get("sr0_expected_max"),
        "n_eff_participation": n_eff,
        "best_trial_index": best_trial.index,
        "best_params_hash": best_trial.params_hash,
        "n_bars": best_bt["n_bars"],
        "n_trades": best_bt["n_trades"],
        "final_equity": best_bt["final_equity"],
        "total_return_pct": best_bt["total_return_pct"],
        "cost_breakdown": best_bt["cost_breakdown"],
        "capacity": best_bt["capacity"],
        "performance": perf,
        "trade_summary": scored["trade_summary"],
        "costs": scored["costs"],
        "min_track_record_length": mintrl.get("min_track_record_length"),
        "pbo": selection.get("pbo") if selection is not None else None,
    }

    run_id: str | None = None
    cum_trials = total_attempted
    proj_trials = total_attempted
    if record_in_project:
        repro_target = str(signal_or_target) if isinstance(signal_or_target, (str, Path)) else None
        rec = _record_coded_in_project(
            command="coded_signal",
            params=manifest_params,
            dataset_hashes=[data_hash],
            n_trials=total_attempted,
            n_trials_source=n_source,
            figures=figures,
            verdict=final_verdict,
            idea_id=clean_idea,
            family=resolved_family,
            repro_spec=(
                {
                    "kind": "coded_signal",
                    "signal_target": repro_target,
                    "params": dict(params or {}),
                    "grid": {str(k): list(v) for k, v in grid.items()} if grid is not None else None,
                    "slippage_bps": float(slippage_bps),
                    "commission_bps": float(commission_bps),
                    "borrow_bps": float(borrow_bps),
                }
                if repro_target
                else None
            ),
            returns_for_deflation=None if (allow_zero_costs_unverified or borrow_withheld) else best_rets,
            trials_sharpe_std=trials_std,
            risk_free_rate=risk_free_rate,
            periods_per_year=ppy,
            manifest=manifest,
            trial_ledger=trial_ledger,
            raw_prices_for_repro=dict(resolved_prices),
        )
        if rec is not None:
            run_id = str(rec["run_id"])
            cum_trials = int(rec["cumulative_trials"])
            proj_trials = int(rec["project_trials"])
            if not allow_zero_costs_unverified and not borrow_withheld:
                final_verdict = str(rec.get("verdict") or final_verdict)
                figures = dict(rec.get("figures") or figures)

    return {
        "engine_version": current_engine_version(),
        "run_id": run_id,
        "idea_id": clean_idea,
        "family": resolved_family,
        "verdict": final_verdict,
        "verdict_reason": verdict_reason,
        "engine_verified": not allow_zero_costs_unverified,
        "can_seal": can_seal,
        "code_sha256": code_sha256,
        "dataset_hashes": [data_hash],
        "n_trials": cum_trials,
        "run_n_trials": total_attempted,
        "n_ok_trials": len(ok_indices),
        "n_failed_trials": total_attempted - len(ok_indices),
        "n_trials_source": n_source,
        "cumulative_trials": cum_trials,
        "project_trials": proj_trials,
        "backtest": best_bt,
        "score": scored,
        "figures": figures,
        "selection": selection,
        "trial_ledger": trial_ledger,
        "manifest": manifest.to_wire_summary(),
        "annualization": build_annualization_spec(ppy),
        "risk_free": build_risk_free_spec(risk_free_rate, ppy),
        "costs_model": costs_spec,
    }


def _record_coded_in_project(
    *,
    command: str,
    params: dict[str, Any],
    dataset_hashes: list[str],
    n_trials: int,
    n_trials_source: str,
    figures: dict[str, Any],
    verdict: str,
    idea_id: str | None,
    family: str,
    repro_spec: dict[str, Any] | None,
    returns_for_deflation: list[float] | None,
    trials_sharpe_std: float | None,
    risk_free_rate: float | None,
    periods_per_year: int,
    manifest: Any,
    trial_ledger: list[dict[str, Any]],
    raw_prices_for_repro: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    from .ledger import is_auto_sweep_suppressed, record_run, record_trials
    from .project import find_project

    if is_auto_sweep_suppressed():
        return None
    proj = find_project()
    if proj is None:
        return None
    final_repro = dict(repro_spec) if repro_spec is not None else None
    if final_repro is not None and raw_prices_for_repro is not None:
        with contextlib.suppress(Exception):
            final_repro["data_object_sha256"] = proj.put_object(raw_prices_for_repro)
    rec = record_run(
        proj,
        command=command,
        params=params,
        dataset_hashes=dataset_hashes,
        n_trials=n_trials,
        n_trials_source=n_trials_source,
        figures=figures,
        verdict=verdict,
        idea_id=idea_id,
        family=family,
        repro_spec=final_repro,
        returns_for_deflation=returns_for_deflation,
        trials_sharpe_std=trials_sharpe_std,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        manifest=manifest,
    )
    record_trials(
        proj,
        run_id=str(rec["run_id"]),
        trials=trial_ledger,
        idea_id=rec.get("idea_id"),
        family=str(rec["family"]),
        timestamp=str(rec["timestamp"]),
    )
    return rec
