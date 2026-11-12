"""Single source of truth for the 25 AlphaEngine MCP Tools (`QOS-172`, `QOS-174`).

Every tool is audited for:
- Deterministic vs account classification.
- Storage model: deterministic tools are 100% stateless (in-memory compute);
  account tools persist summaries/universes.
- Unified 2.0 conventions: engine_version stamp, pinned 4.21% cash rate,
  and canonical Verdict enum (`robust`, `marginal`, `likely_noise`, `unverified`).
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .._version import current_engine_version
from ..core import (
    combinatorial_oos_stability,
    compute_var_cvar,
    deflated_sharpe,
    denoise_cov,
    detone_cov,
    dgp_stress,
    ewma_cov,
    fama_macbeth,
    gbm_calibrate,
    harvey_liu_haircut,
    hrp_weights,
    jump_calibrate,
    min_track_record_length,
    ou_calibrate,
    pbo_cscv,
    performance_report,
    quantile_book,
    risk_parity_weights,
    run_backtest,
    score_backtest,
    screen_universe,
    vol_target,
)


@dataclass(frozen=True)
class MCPToolSpec:
    """Specification of an AlphaEngine MCP tool."""

    name: str
    kind: str  # "deterministic" | "account"
    inputs_summary: str
    outputs_summary: str
    storage: str
    description: str
    handler: Callable[..., Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "inputs": self.inputs_summary,
            "outputs": self.outputs_summary,
            "storage": self.storage,
            "description": self.description,
            "engine_version": current_engine_version(),
        }


def _jsonable(value: Any) -> Any:
    """NumPy arrays and scalars become plain lists and numbers; NaN/Inf become None."""
    import math

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return f if math.isfinite(f) else None
    return value


def _stamp_engine(result: Any) -> dict[str, Any]:
    out = _jsonable(result)
    if not isinstance(out, dict):
        out = {"result": out}
    if "engine_version" not in out:
        out["engine_version"] = current_engine_version()
    return out


def _call_core(
    fn: Callable[..., Any],
    tool: str,
    kwargs: dict[str, Any],
    *,
    aliases: dict[str, str] | None = None,
    ignored: tuple[str, ...] = (),
) -> Any:
    """Map documented/legacy aliases onto the core signature and refuse unknown inputs plainly.

    A tool called with its documented inputs must never die on a `TypeError` from
    a renamed keyword. Aliases are applied only when the canonical name is absent.
    Inputs that are documented but have no effect in the core function are dropped.
    """
    import inspect

    params = dict(kwargs)
    for alias, canonical in (aliases or {}).items():
        if alias in params:
            value = params.pop(alias)
            if canonical not in params and value is not None:
                params[canonical] = value
    for name in ignored:
        params.pop(name, None)
    sig = inspect.signature(fn)
    accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if not accepts_var_kw:
        unknown = sorted(k for k in params if k not in sig.parameters)
        if unknown:
            allowed = sorted(k for k in sig.parameters if k not in ("self",))
            raise ValueError(
                f"{tool} does not take {', '.join(unknown)}. Accepted inputs: {', '.join(allowed)}."
            )
    missing = [
        p.name
        for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.name not in params
    ]
    if missing:
        raise ValueError(f"{tool} needs {', '.join(missing)}.")
    return fn(**params)


def _handle_backtest(signals: dict[str, Any], prices: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return _stamp_engine(run_backtest(signals, prices, **kwargs))


def _handle_score_backtest(bt: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    if bt is None:
        bt = kwargs.pop("backtest", None)
    if not isinstance(bt, dict):
        raise ValueError("score_backtest needs bt, the dict returned by the backtest tool.")
    kwargs.pop("engine_version", None)
    return _stamp_engine(_call_core(score_backtest, "score_backtest", {"bt": bt, **kwargs}))


def _handle_deflated_sharpe(returns: list[float], n_trials: int | float, **kwargs: Any) -> dict[str, Any]:
    return _stamp_engine(deflated_sharpe(returns, n_trials=n_trials, **kwargs))


def _handle_pbo_cscv(pnl_matrix: list[list[float]], **kwargs: Any) -> dict[str, Any]:
    return _stamp_engine(pbo_cscv(pnl_matrix, **kwargs))


def _handle_combinatorial_oos(returns: list[float], **kwargs: Any) -> dict[str, Any]:
    return _stamp_engine(combinatorial_oos_stability(returns, **kwargs))


def _handle_min_track_record(returns: list[float], **kwargs: Any) -> dict[str, Any]:
    return _stamp_engine(min_track_record_length(returns, **kwargs))


def _handle_harvey_liu(
    sharpe_annualized: float, n_obs: int, n_trials: int | float, **kwargs: Any
) -> dict[str, Any]:
    return _stamp_engine(harvey_liu_haircut(sharpe_annualized, n_obs=n_obs, n_trials=n_trials, **kwargs))


def _handle_performance(returns: list[float], **kwargs: Any) -> dict[str, Any]:
    return _stamp_engine(performance_report(returns, **kwargs))


def _handle_var_cvar(**kwargs: Any) -> dict[str, Any]:
    alpha = kwargs.pop("alpha", None)
    if alpha is not None and "confidence" not in kwargs:
        a = float(alpha)
        if not 0.0 < a < 1.0:
            raise ValueError("var_cvar alpha must be strictly between 0 and 1 (e.g. 0.05).")
        kwargs["confidence"] = a if a >= 0.5 else 1.0 - a
    return _stamp_engine(
        _call_core(
            compute_var_cvar,
            "var_cvar",
            kwargs,
            aliases={"returns": "portfolio_returns"},
            ignored=("method",),
        )
    )


_SCREEN_RULES = {
    "momentum": "return_pct",
    "return": "return_pct",
    "returns": "return_pct",
    "volatility": "volatility_pct",
    "vol": "volatility_pct",
    "trend": "distance_from_sma_pct",
    "sma": "distance_from_sma_pct",
}


def _handle_screen_universe(
    prices: dict[str, Any] | None = None,
    universe: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    if prices is None:
        prices = kwargs.pop("panel", None)
    if not isinstance(prices, dict):
        raise ValueError("screen_universe needs prices as {ticker: closes}.")
    if universe is not None:
        wanted = {str(t).upper() for t in universe}
        prices = {k: v for k, v in prices.items() if str(k).upper() in wanted}
    rule = kwargs.pop("rule", None)
    if rule is not None and "rank_by" not in kwargs:
        kwargs["rank_by"] = _SCREEN_RULES.get(str(rule).lower(), str(rule))
    return _stamp_engine(_call_core(screen_universe, "screen_universe", {"prices": prices, **kwargs}))


def _handle_fama_macbeth(**kwargs: Any) -> dict[str, Any]:
    exp = kwargs.get("exposures")
    if isinstance(exp, dict) and len(exp) == 1:
        sole = next(iter(exp.values()))
        if isinstance(sole, dict):
            kwargs = dict(kwargs)
            kwargs["exposures"] = sole
    return _stamp_engine(
        _call_core(
            fama_macbeth,
            "fama_macbeth",
            kwargs,
            aliases={
                "signals": "signal",
                "panel": "signal",
                "exposures": "signal",
                "factors": "prices",
                "returns": "prices",
            },
        )
    )


def _handle_quantile_book(**kwargs: Any) -> dict[str, Any]:
    return _stamp_engine(
        _call_core(
            quantile_book,
            "quantile_book",
            kwargs,
            aliases={"signals": "signal", "returns": "prices", "n_quantiles": "quantiles"},
        )
    )


def _cov_input(kwargs: dict[str, Any], tool: str) -> np.ndarray[Any, Any]:
    raw = kwargs.pop("cov", None)
    alias = kwargs.pop("cov_matrix", None)
    if raw is None:
        raw = alias
    if raw is None:
        raise ValueError(f"{tool} needs cov_matrix (an N x N covariance).")
    arr = np.asarray(raw, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 1:
        raise ValueError(f"{tool} needs a square N x N covariance matrix.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{tool} covariance contains NaN or Inf.")
    return arr


def _handle_hrp(**kwargs: Any) -> dict[str, Any]:
    cov = _cov_input(kwargs, "hrp")
    return _stamp_engine(_call_core(hrp_weights, "hrp", {"cov": cov, **kwargs}, ignored=("method",)))


def _handle_risk_parity(**kwargs: Any) -> dict[str, Any]:
    cov = _cov_input(kwargs, "risk_parity")
    if kwargs.pop("target_risk_contributions", None) is not None:
        raise ValueError("risk_parity solves for equal risk contributions; custom targets are not supported.")
    return _stamp_engine(_call_core(risk_parity_weights, "risk_parity", {"cov": cov, **kwargs}))


def _handle_vol_target(**kwargs: Any) -> dict[str, Any]:
    max_lev = kwargs.pop("max_leverage", None)
    res = _call_core(vol_target, "vol_target", kwargs, aliases={"target_vol": "target"})
    out = _stamp_engine(res)
    if max_lev is not None:
        cap = float(max_lev)
        if not np.isfinite(cap) or cap <= 0:
            raise ValueError("vol_target max_leverage must be a positive number.")
        out["max_leverage"] = cap
        for key in ("leverages", "leverage", "scale", "scalar", "leverage_scalar", "mean_leverage"):
            val = out.get(key)
            if isinstance(val, list):
                out[key] = [min(cap, v) if isinstance(v, (int, float)) else v for v in val]
            elif isinstance(val, (int, float)) and not isinstance(val, bool):
                out[key] = min(cap, float(val))
    return out


def _handle_denoise_cov(**kwargs: Any) -> dict[str, Any]:
    cov = _cov_input(kwargs, "denoise_cov")
    q = kwargs.pop("q", None)
    kwargs.pop("method", None)
    n_obs = kwargs.pop("n_obs", None)
    if n_obs is None:
        n_obs = kwargs.pop("T", None)
    if n_obs is None:
        n_obs = int(round(float(q) * cov.shape[0])) if q is not None else 252
    n_obs = int(n_obs)
    if n_obs < 2:
        raise ValueError("denoise_cov needs n_obs >= 2 (or q = T/N with T >= 2).")
    return _stamp_engine(_call_core(denoise_cov, "denoise_cov", {"cov": cov, "n_obs": n_obs, **kwargs}))


def _handle_detone_cov(**kwargs: Any) -> dict[str, Any]:
    cov = _cov_input(kwargs, "detone_cov")
    modes = kwargs.pop("n_market_modes", 1)
    if modes not in (None, 1):
        raise ValueError("detone_cov removes exactly one market mode (n_market_modes=1).")
    res = _call_core(detone_cov, "detone_cov", {"cov": cov, **kwargs})
    return _stamp_engine({"detoned_cov": res})


def _handle_ewma_cov(**kwargs: Any) -> dict[str, Any]:
    raw = kwargs.pop("returns_matrix", None)
    if raw is None:
        raw = kwargs.pop("returns", None)
    if raw is None:
        raise ValueError("ewma_cov needs returns_matrix (T x N).")
    arr = np.asarray(raw, dtype=float)
    if arr.ndim != 2:
        raise ValueError("ewma_cov needs a 2-D T x N returns matrix.")
    res = _call_core(ewma_cov, "ewma_cov", {"returns": arr, **kwargs}, aliases={"lambda_param": "lam"})
    return _stamp_engine({"ewma_cov_matrix": res})


def _handle_dgp_stress(**kwargs: Any) -> dict[str, Any]:
    return _stamp_engine(
        _call_core(
            dgp_stress,
            "dgp_stress",
            kwargs,
            aliases={"returns": "values", "n_scenarios": "n_paths"},
            ignored=("horizon_bars",),
        )
    )


def _handle_ou_process(**kwargs: Any) -> dict[str, Any]:
    return _stamp_engine(
        _call_core(
            ou_calibrate, "ou_process", kwargs, aliases={"prices_or_spread": "values", "spread": "values"}
        )
    )


def _handle_gbm_process(**kwargs: Any) -> dict[str, Any]:
    return _stamp_engine(
        _call_core(gbm_calibrate, "gbm_process", kwargs, aliases={"prices": "values"}, ignored=("dt",))
    )


def _handle_jump_process(**kwargs: Any) -> dict[str, Any]:
    return _stamp_engine(
        _call_core(jump_calibrate, "jump_process", kwargs, aliases={"returns": "values"}, ignored=("dt",))
    )


# ── Account tool handlers (delegating to workspace/firm backend) ──────────────

#: Keys that would carry raw series or code across the boundary in a delivered summary.
_DELIVER_FORBIDDEN_KEYS = frozenset(
    {
        "prices",
        "returns",
        "closes",
        "raw_rows",
        "dataset_rows",
        "matrix",
        "equity_curve",
        "weights_path",
        "api_key",
        "apikey",
        "secret",
        "token",
        "password",
        "authorization",
        "source_code",
        "code",
    }
)
_DELIVER_MAX_LIST = 512
_DELIVER_MAX_DEPTH = 12


def _check_summary(value: Any, path: str = "summary", depth: int = 0) -> None:
    if depth > _DELIVER_MAX_DEPTH:
        raise ValueError(f"{path} is nested too deeply to deliver.")
    if isinstance(value, dict):
        for key, inner in value.items():
            if str(key).lower() in _DELIVER_FORBIDDEN_KEYS:
                raise ValueError(
                    f"{path}.{key} is not deliverable: raw series, code, and secrets stay local."
                )
            _check_summary(inner, f"{path}.{key}", depth + 1)
    elif isinstance(value, (list, tuple)):
        if len(value) > _DELIVER_MAX_LIST:
            raise ValueError(
                f"{path} has {len(value)} entries; summaries carry at most {_DELIVER_MAX_LIST} per list."
            )
        for idx, inner in enumerate(value):
            _check_summary(inner, f"{path}[{idx}]", depth + 1)


def _workspace_dir(*parts: str) -> Any:
    from pathlib import Path

    from ..project import find_project

    proj = find_project()
    root = (proj.root if proj is not None else Path.cwd()).resolve()
    target = root.joinpath(".alphaengine", *parts).resolve()
    if not target.is_relative_to(root):
        raise ValueError("refusing to write outside the workspace")
    target.mkdir(parents=True, exist_ok=True)
    return proj, root, target


def _child(directory: Any, filename: str) -> Any:
    path = (directory / filename).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError(f"refusing path outside {directory}")
    return path


def _handle_my_universe(
    tickers: list[str] | None = None, name: str = "default", **kwargs: Any
) -> dict[str, Any]:
    import json

    from ..project import validate_safe_id

    safe_name = validate_safe_id(str(name), "universe name")
    if tickers is not None and not isinstance(tickers, (list, tuple)):
        raise ValueError("my_universe tickers must be a list of ticker strings.")
    _proj, _root, uni_dir = _workspace_dir("universes")
    uni_file = _child(uni_dir, f"{safe_name}.json")

    updated_at: str | None = None
    if tickers is not None:
        clean_tickers = sorted({str(t).upper().strip() for t in tickers if str(t).strip()})
        if len(clean_tickers) > 10_000:
            raise ValueError("my_universe holds at most 10,000 tickers.")
        updated_at = datetime.now(timezone.utc).isoformat()
        data = {"name": safe_name, "tickers": clean_tickers, "updated_at": updated_at}
        uni_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        out_tickers = clean_tickers
    else:
        out_tickers = []
        if uni_file.is_file():
            try:
                loaded = json.loads(uni_file.read_text(encoding="utf-8"))
                out_tickers = [str(t) for t in (loaded.get("tickers") or [])]
                updated_at = loaded.get("updated_at")
            except (OSError, ValueError, AttributeError):
                out_tickers = []

    return {
        "engine_version": current_engine_version(),
        "status": "ok",
        "universe": safe_name,
        "tickers": out_tickers,
        "count": len(out_tickers),
        "updated_at": updated_at,
        "stored": f"persisted in researcher's workspace ({uni_file})",
    }


def _handle_deliver_run(run_id: str, summary: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    import hashlib
    import json

    from ..project import validate_safe_id

    safe_id = validate_safe_id(str(run_id), "run_id")
    if not isinstance(summary, dict):
        raise ValueError("deliver_run summary must be a dict (a RunManifest-style summary).")
    _check_summary(summary)
    clean_summary = _jsonable(summary)
    proj, _root, runs_dir = _workspace_dir("ledger", "runs")
    run_file = _child(runs_dir, f"{safe_id}.json")

    delivered_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "engine_version": current_engine_version(),
        "run_id": safe_id,
        "summary": clean_summary,
        "delivered_at": delivered_at,
    }
    content_bytes = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
    run_sha = hashlib.sha256(content_bytes).hexdigest()
    payload["content_sha256"] = run_sha
    run_file.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    cas_sha = clean_summary.get("manifest_sha256") or clean_summary.get("object_sha256")
    if cas_sha and proj is not None:
        with contextlib.suppress(Exception):
            proj.put_object({"run_id": safe_id, "summary": clean_summary})

    return {
        "engine_version": current_engine_version(),
        "status": "delivered",
        "run_id": safe_id,
        "content_sha256": run_sha,
        "delivered_at": delivered_at,
        "stored": f"persisted in firm research ledger and audit trail ({run_file})",
    }


def _handle_run_workflow(
    workflow: str | None = None, params: dict[str, Any] | None = None, **kwargs: Any
) -> dict[str, Any]:
    import json
    import uuid

    from ..project import validate_safe_id

    if workflow is None:
        workflow = kwargs.pop("workflow_name", None)
    if not workflow:
        raise ValueError("run_workflow needs workflow (or workflow_name).")
    safe_wf = validate_safe_id(str(workflow), "workflow")
    if params is not None and not isinstance(params, dict):
        raise ValueError("run_workflow params must be a dict.")
    clean_params = _jsonable(dict(params or {}))
    _check_summary(clean_params, "params")
    _proj, root, queue_dir = _workspace_dir("queue")
    job_id = f"job_{safe_wf}_{uuid.uuid4().hex[:12]}"
    job_file = _child(queue_dir, f"{job_id}.json")

    submitted_at = datetime.now(timezone.utc).isoformat()
    job_record = {
        "engine_version": current_engine_version(),
        "job_id": job_id,
        "workflow": safe_wf,
        "params": clean_params,
        "status": "submitted",
        "submitted_at": submitted_at,
    }
    job_file.write_text(json.dumps(job_record, indent=2, allow_nan=False), encoding="utf-8")

    hist_file = root / ".alphaengine" / "run_history.jsonl"
    with open(hist_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(job_record, allow_nan=False) + "\n")

    return {
        "engine_version": current_engine_version(),
        "status": "submitted",
        "workflow": safe_wf,
        "job_id": job_id,
        "submitted_at": submitted_at,
        "stored": f"persisted in firm execution queue and run history ({job_file})",
    }


_MCP_TOOLS: tuple[MCPToolSpec, ...] = (
    # ── Deterministic Core Quantitative Tools (1-22) ─────────────────────────
    MCPToolSpec(
        name="backtest",
        kind="deterministic",
        inputs_summary="signals, prices, slippage_bps=5.0, commission_bps=1.0, borrow_bps=50.0, fill_timing='close'",
        outputs_summary="dates, equity_curve, returns, trades, final_equity, total_return_pct, cost_breakdown",
        storage="stateless (in-memory compute)",
        description="Deterministic weight-target portfolio simulator net of slippage, commission, borrow, and liquidation.",
        handler=_handle_backtest,
    ),
    MCPToolSpec(
        name="score_backtest",
        kind="deterministic",
        inputs_summary="bt (backtest dict), n_trials=None, pnl_matrix=None, risk_free_rate=None",
        outputs_summary="performance, validation (dsr_probability, verdict, harvey_liu, min_track_record_length)",
        storage="stateless (in-memory compute)",
        description="Composes risk-adjusted performance with trial-adjusted deflation and returns canonical 2.0 Verdict.",
        handler=_handle_score_backtest,
    ),
    MCPToolSpec(
        name="deflated_sharpe",
        kind="deterministic",
        inputs_summary="returns, n_trials, trials_sharpe_std=None, risk_free_rate=None, matrix=None, n_eff=None",
        outputs_summary="dsr_probability, verdict, sr0_expected_max, n_trials, n_eff, n_eff_method",
        storage="stateless (in-memory compute)",
        description="Deflated Sharpe Ratio probability for multiple testing and non-normal skew/kurtosis (Bailey & LdP).",
        handler=_handle_deflated_sharpe,
    ),
    MCPToolSpec(
        name="pbo_cscv",
        kind="deterministic",
        inputs_summary="pnl_matrix (T x N), n_splits=10, max_combos=2000",
        outputs_summary="pbo, n_partitions, n_configs, n_eff, n_eff_method, logit_mean, verdict",
        storage="stateless (in-memory compute)",
        description="Probability of Backtest Overfitting via Combinatorially Symmetric Cross-Validation (Bailey et al.).",
        handler=_handle_pbo_cscv,
    ),
    MCPToolSpec(
        name="combinatorial_oos_stability",
        kind="deterministic",
        inputs_summary="returns, n_groups=8, n_test_groups=2, purge=1, embargo=1, n_trials=1",
        outputs_summary="sharpe_annualized (distribution), dsr_probability, verdict, pct_paths_positive",
        storage="stateless (in-memory compute)",
        description="Evaluates combinatorial out-of-sample stability with causal purge and embargo over held-out block paths.",
        handler=_handle_combinatorial_oos,
    ),
    MCPToolSpec(
        name="min_track_record_length",
        kind="deterministic",
        inputs_summary="returns, sr_benchmark=0.0, confidence=0.95, risk_free_rate=None",
        outputs_summary="min_track_record_length (bars), min_track_record_years, sufficient, shortfall_obs",
        storage="stateless (in-memory compute)",
        description="Calculates minimum track record length needed to distinguish skill from luck under skew and kurtosis.",
        handler=_handle_min_track_record,
    ),
    MCPToolSpec(
        name="harvey_liu",
        kind="deterministic",
        inputs_summary="sharpe_annualized, n_obs, n_trials, n_obs_effective=None, n_eff_participation=None",
        outputs_summary="t_stat, p_value_single, bonferroni_p, holm_p, bhy_p, haircut_sharpe_holm, at_n_eff",
        storage="stateless (in-memory compute)",
        description="Harvey & Liu (2015) multiple-testing Sharpe ratio haircut under Bonferroni, Holm, and BHY adjustments.",
        handler=_handle_harvey_liu,
    ),
    MCPToolSpec(
        name="performance_report",
        kind="deterministic",
        inputs_summary="returns, equity_curve=None, risk_free_rate=None, periods_per_year=252",
        outputs_summary="sharpe_ratio, sharpe_annualized, sortino_ratio, calmar_ratio, max_drawdown_pct, CAGR",
        storage="stateless (in-memory compute)",
        description="Calculates standard risk-adjusted performance figures excess of the pinned 4.21% Treasury bill rate.",
        handler=_handle_performance,
    ),
    MCPToolSpec(
        name="var_cvar",
        kind="deterministic",
        inputs_summary="returns, confidence=0.95 (or alpha=0.05), horizon_days=1, portfolio_value=100000",
        outputs_summary="historical, parametric, cornish_fisher, cvar (all as loss pct), confidence, horizon_days",
        storage="stateless (in-memory compute)",
        description="Computes Value at Risk and Conditional Value at Risk (Expected Shortfall) as positive loss fractions.",
        handler=_handle_var_cvar,
    ),
    MCPToolSpec(
        name="screen_universe",
        kind="deterministic",
        inputs_summary="prices, universe=None, rank_by='return_pct' (or rule='momentum'), top_n=20, filters=None",
        outputs_summary="rows (ranked shortlist), n_evaluated, n_passing, n_returned, n_insufficient",
        storage="stateless (in-memory compute)",
        description="Screens a universe of assets cross-sectionally without touching external networks.",
        handler=_handle_screen_universe,
    ),
    MCPToolSpec(
        name="fama_macbeth",
        kind="deterministic",
        inputs_summary="signal ({ticker: scores}), prices ({ticker: closes}), horizon=21",
        outputs_summary="lambda_mean, lambda_by_date, t_stat, n_dates, n_names",
        storage="stateless (in-memory compute)",
        description="Two-pass Fama-MacBeth cross-sectional regression estimating factor risk premia and HAC t-statistics.",
        handler=_handle_fama_macbeth,
    ),
    MCPToolSpec(
        name="quantile_book",
        kind="deterministic",
        inputs_summary="signal ({ticker: scores}), prices ({ticker: closes}), horizon=21, quantiles=5",
        outputs_summary="spread_pct, quantile returns, n_periods, n_names, turnover",
        storage="stateless (in-memory compute)",
        description="Sorts assets into quantile portfolios and evaluates bucket return spread and rank monotonicity.",
        handler=_handle_quantile_book,
    ),
    MCPToolSpec(
        name="hrp",
        kind="deterministic",
        inputs_summary="cov_matrix, names=None",
        outputs_summary="weights, n_assets, max_weight, min_weight, condition_number",
        storage="stateless (in-memory compute)",
        description="Hierarchical Risk Parity portfolio allocation (Lopez de Prado 2016) via quasi-diagonalization.",
        handler=_handle_hrp,
    ),
    MCPToolSpec(
        name="risk_parity",
        kind="deterministic",
        inputs_summary="cov_matrix, names=None",
        outputs_summary="weights, converged, n_assets, max_weight, min_weight, condition_number",
        storage="stateless (in-memory compute)",
        description="Equal risk contribution portfolio allocation balancing marginal volatility risk.",
        handler=_handle_risk_parity,
    ),
    MCPToolSpec(
        name="vol_target",
        kind="deterministic",
        inputs_summary="returns, target_vol=0.10, method='ewma', ewma_lambda=0.94, max_leverage=None",
        outputs_summary="leverages, mean_leverage, n_obs, n_warmup, method",
        storage="stateless (in-memory compute)",
        description="Computes volatility-targeted dynamic leverage allocation scalar.",
        handler=_handle_vol_target,
    ),
    MCPToolSpec(
        name="denoise_cov",
        kind="deterministic",
        inputs_summary="cov_matrix, n_obs=252 (or q=T/N)",
        outputs_summary="cov (denoised), lambda_plus, n_signal_eigenvalues, n_noise_eigenvalues, q",
        storage="stateless (in-memory compute)",
        description="Marcenko-Pastur spectral truncation denoising of empirical sample covariance matrices.",
        handler=_handle_denoise_cov,
    ),
    MCPToolSpec(
        name="detone_cov",
        kind="deterministic",
        inputs_summary="cov_matrix",
        outputs_summary="detoned_cov",
        storage="stateless (in-memory compute)",
        description="Removes dominant market eigenvector mode from correlation matrix to isolate idiosyncratic risk.",
        handler=_handle_detone_cov,
    ),
    MCPToolSpec(
        name="ewma_cov",
        kind="deterministic",
        inputs_summary="returns_matrix (T x N), lambda_param=0.94",
        outputs_summary="ewma_cov_matrix",
        storage="stateless (in-memory compute)",
        description="Exponentially weighted moving average covariance matrix estimation (RiskMetrics convention).",
        handler=_handle_ewma_cov,
    ),
    MCPToolSpec(
        name="dgp_stress",
        kind="deterministic",
        inputs_summary="returns, dgp='gbm', n_paths=200 (or n_scenarios), seed=7",
        outputs_summary="dgp, n_paths, sharpe_q10, sharpe_q50, sharpe_q90, frac_positive, sharpe_hist",
        storage="stateless (in-memory compute)",
        description="Data Generating Process parametric bootstrap and extreme scenario stress testing.",
        handler=_handle_dgp_stress,
    ),
    MCPToolSpec(
        name="ou_process",
        kind="deterministic",
        inputs_summary="prices_or_spread, dt=1.0",
        outputs_summary="kappa, theta, sigma, half_life, phi, mean_reverting",
        storage="stateless (in-memory compute)",
        description="Calibrates Ornstein-Uhlenbeck continuous-time mean-reverting process parameters via exact MLE.",
        handler=_handle_ou_process,
    ),
    MCPToolSpec(
        name="gbm_process",
        kind="deterministic",
        inputs_summary="prices, kind=None",
        outputs_summary="mu (drift), sigma (diffusion vol), n_obs",
        storage="stateless (in-memory compute)",
        description="Geometric Brownian Motion parameter calibration via log-return normal maximum likelihood.",
        handler=_handle_gbm_process,
    ),
    MCPToolSpec(
        name="jump_process",
        kind="deterministic",
        inputs_summary="returns, z=3.0, kind=None",
        outputs_summary="mu, sigma, jump_lambda, jump_mu, jump_sigma, n_jumps_named",
        storage="stateless (in-memory compute)",
        description="Merton jump-diffusion process calibration identifying discontinuous market shock parameters.",
        handler=_handle_jump_process,
    ),
    # ── Account & Workflow Tools (23-25) ─────────────────────────────────────
    MCPToolSpec(
        name="my_universe",
        kind="account",
        inputs_summary="tickers=None (omitted to read, list of tickers to update), name='default'",
        outputs_summary="tickers, count, updated_at",
        storage="persisted in researcher's workspace (.alphaengine/universes/)",
        description="Retrieves or updates the researcher's active asset universe.",
        handler=_handle_my_universe,
    ),
    MCPToolSpec(
        name="deliver_run",
        kind="account",
        inputs_summary="run_id, summary (RunManifest dict)",
        outputs_summary="run_id, content_sha256, delivered_at, status",
        storage="persisted in firm research ledger and audit trail",
        description="Uploads an allowlisted research summary and CAS manifest to the firm ledger.",
        handler=_handle_deliver_run,
    ),
    MCPToolSpec(
        name="run_workflow",
        kind="account",
        inputs_summary="workflow (or workflow_name), params=None",
        outputs_summary="job_id, status, submitted_at",
        storage="persisted in firm execution queue and run history",
        description="Submits a validated quantOS research workflow to the hosted execution cluster.",
        handler=_handle_run_workflow,
    ),
)

MCP_TOOLS_REGISTRY: dict[str, MCPToolSpec] = {tool.name: tool for tool in _MCP_TOOLS}


def get_mcp_tool(name: str) -> MCPToolSpec:
    """Return tool specification for the named MCP tool."""
    tool = MCP_TOOLS_REGISTRY.get(name.strip())
    if tool is None:
        raise KeyError(f"Unknown MCP tool {name!r}. Available tools: {sorted(MCP_TOOLS_REGISTRY)}")
    return tool


def list_mcp_tools() -> list[dict[str, Any]]:
    """Return specifications for all 25 audited MCP tools."""
    return [t.to_dict() for t in _MCP_TOOLS]


def call_mcp_tool(name: str, /, **kwargs: Any) -> dict[str, Any]:
    """Execute the named MCP tool with unified conventions and engine_version stamp."""
    tool = get_mcp_tool(name)
    if tool.handler is None:
        raise NotImplementedError(f"Tool {name!r} handler is not defined.")
    return _stamp_engine(tool.handler(**kwargs))


__all__ = [
    "MCPToolSpec",
    "MCP_TOOLS_REGISTRY",
    "get_mcp_tool",
    "list_mcp_tools",
    "call_mcp_tool",
]
