"""Interop loaders: build an AlphaEngine Study from QuantConnect or VectorBT outputs.

Both loaders compute the study's performance metrics, data hash, minimum track
record length, and canonical content seal from the imported return stream (or
equity curve), while preserving the trial-count honesty rules:

- When a trial count is explicitly provided or present in the artifact metadata,
  ``n_trials_source`` is set to ``"asserted"`` and ``deflated_sharpe`` is
  computed at that count.
- When no trial count is present, ``n_trials`` is ``0``, ``n_trials_source`` is
  ``"not_recorded"``, ``deflated_sharpe`` is ``None``, and ``verdict`` is
  ``None`` (refusing an ``edge`` or ``robust`` verdict on an unrecorded
  denominator).
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .core import deflated_sharpe, min_track_record_length, performance_report
from .study import Study
from .sweep.runner import _hash_data

__all__ = ["from_quantconnect", "from_vectorbt", "StudyWithReturns"]


class StudyWithReturns(Study):
    """A :class:`Study` carrying the local ``returns`` series in memory without serializing it to disk."""

    returns: list[float]

    def __init__(self, *args: Any, returns: list[float] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "returns", list(returns or []))


def _prices_to_returns(prices: Sequence[float]) -> list[float]:
    arr = np.asarray([float(x) for x in prices if x is not None and math.isfinite(float(x))], dtype=float)
    if arr.size < 2:
        return []
    prev = arr[:-1]
    valid = np.abs(prev) > 1e-12
    if not np.all(valid):
        return []
    return [float(x) for x in (arr[1:] - prev) / prev]


def _clean_returns(values: Sequence[Any]) -> list[float]:
    out: list[float] = []
    for v in values:
        if v is None or isinstance(v, bool):
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def _build_interop_study(
    returns: list[float],
    *,
    label: str,
    data_description: str,
    n_trials: int | None,
    notes: str = "",
    risk_free_rate: float | None = None,
) -> StudyWithReturns:
    if len(returns) < 2:
        raise ValueError(f"{data_description}: could not extract a return series (need >= 2 observations).")

    perf = performance_report(returns, risk_free_rate=risk_free_rate)
    data_hash = _hash_data(returns)
    try:
        mintrl: dict[str, Any] | None = min_track_record_length(returns, risk_free_rate=risk_free_rate)
    except Exception:  # noqa: BLE001
        mintrl = None

    if n_trials is not None and int(n_trials) >= 1:
        trials_val = int(n_trials)
        source = "asserted"
        dsr = deflated_sharpe(returns, n_trials=trials_val, risk_free_rate=risk_free_rate)
        dsr_val = dsr.get("deflated_sharpe")
        psr_val = dsr.get("psr_vs_zero")
        sr0_val = dsr.get("sr0_expected_max")
        verdict_val = dsr.get("verdict")
    else:
        trials_val = 0
        source = "not_recorded"
        dsr_unrec = deflated_sharpe(returns, n_trials=1, risk_free_rate=risk_free_rate)
        dsr_val = None
        psr_val = dsr_unrec.get("psr_vs_zero")
        sr0_val = None
        verdict_val = None
        refusal = (
            "Trial count was not recorded in the imported artifact; "
            "verdict is refused (pass n_trials=... to assert the number of trials searched)."
        )
        notes = f"{notes}\n{refusal}".strip() if notes else refusal

    study = StudyWithReturns(
        label=label,
        n_trials=trials_val,
        n_trials_source=source,
        data_hash=data_hash,
        data_description=data_description,
        verdict=verdict_val,
        deflated_sharpe=dsr_val,
        psr_vs_zero=psr_val,
        sr0_expected_max=sr0_val,
        min_track_record_length=mintrl,
        performance=perf,
        notes=notes,
        returns=returns,
    )
    study.seal()
    return study


def _extract_qc_series(raw: Mapping[str, Any]) -> list[float]:
    if isinstance(raw.get("returns"), list):
        cleaned = _clean_returns(raw["returns"])
        if len(cleaned) >= 2:
            return cleaned

    charts = raw.get("charts") or raw.get("Charts") or {}
    if isinstance(charts, Mapping):
        for chart_name in ("Strategy Equity", "Benchmark", "Equity"):
            chart = charts.get(chart_name) or {}
            if not isinstance(chart, Mapping):
                continue
            series_map = chart.get("series") or chart.get("Series") or {}
            if not isinstance(series_map, Mapping):
                continue
            # Prefer a direct Return series if present and non-empty
            for s_name in ("Return", "Daily Performance", "Returns"):
                s_obj = series_map.get(s_name)
                if isinstance(s_obj, Mapping):
                    vals = _extract_qc_points(s_obj.get("values") or s_obj.get("Values") or [], is_pct=True)
                    if len(vals) >= 2:
                        return vals
            # Otherwise convert Equity curve points to returns
            for s_name in ("Equity", "Strategy Equity", "Value"):
                s_obj = series_map.get(s_name)
                if isinstance(s_obj, Mapping):
                    eq = _extract_qc_points(s_obj.get("values") or s_obj.get("Values") or [], is_pct=False)
                    rets = _prices_to_returns(eq)
                    if len(rets) >= 2:
                        return rets

    # Fallback: rollingWindow or profitLoss map
    pl = raw.get("profitLoss") or raw.get("ProfitLoss")
    if isinstance(pl, Mapping) and len(pl) >= 2:
        ordered = [pl[k] for k in sorted(pl.keys())]
        cleaned = _clean_returns(ordered)
        if len(cleaned) >= 2:
            return cleaned

    return []


def _extract_qc_points(values: Any, *, is_pct: bool) -> list[float]:
    if not isinstance(values, list):
        return []
    pts: list[tuple[Any, float]] = []
    for idx, item in enumerate(values):
        t_key: Any = idx
        val: Any = None
        if isinstance(item, Mapping):
            t_key = item.get("x", idx)
            # QuantConnect candlestick equity points have [x, open, high, low, close] or {x, y}
            val = item.get("y", item.get("close", item.get("value")))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            t_key = item[0]
            val = item[4] if len(item) >= 5 else item[1]
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            val = item
        if val is None or isinstance(val, bool):
            continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            pts.append((t_key, f / 100.0 if is_pct and abs(f) > 0.5 else f))
    pts.sort(key=lambda pair: pair[0])
    return [p[1] for p in pts]


def _extract_qc_trials(raw: Mapping[str, Any]) -> int | None:
    for key in ("n_trials", "totalBacktests", "optimizationTrials", "parameterSetCount"):
        val = raw.get(key)
        if isinstance(val, int) and not isinstance(val, bool) and val >= 1:
            return val
    meta = raw.get(" algorithmConfiguration") or raw.get("algorithmConfiguration") or {}
    if isinstance(meta, Mapping):
        for key in ("n_trials", "optimizationTrials"):
            val = meta.get(key)
            if isinstance(val, int) and not isinstance(val, bool) and val >= 1:
                return val
    return None


def from_quantconnect(
    path: str | Path | Mapping[str, Any],
    *,
    n_trials: int | None = None,
    label: str = "",
    notes: str = "",
    risk_free_rate: float | None = None,
) -> StudyWithReturns:
    """Build a :class:`Study` from a QuantConnect backtest result JSON file (or parsed dict)."""
    if isinstance(path, Mapping):
        raw: Mapping[str, Any] = path
        desc = "quantconnect:dict"
        default_label = str(raw.get("name") or raw.get("projectName") or "quantconnect study")
    else:
        p = Path(path)
        raw_loaded = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw_loaded, Mapping):
            raise ValueError(f"Expected a JSON object in {p}, got {type(raw_loaded).__name__}")
        raw = raw_loaded
        desc = f"quantconnect:{p.name}"
        default_label = str(raw.get("name") or raw.get("projectName") or p.stem)

    returns = _extract_qc_series(raw)
    trials = n_trials if n_trials is not None else _extract_qc_trials(raw)
    return _build_interop_study(
        returns,
        label=label or default_label,
        data_description=desc,
        n_trials=trials,
        notes=notes,
        risk_free_rate=risk_free_rate,
    )


def _extract_vbt_from_csv(path: Path) -> tuple[list[float], int | None]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{path} is empty.")
    rows = list(csv.reader(text.splitlines()))
    if not rows:
        raise ValueError(f"{path} has no rows.")

    header = [h.strip() for h in rows[0]]
    body = rows[1:] if any(not _is_float(c) for c in header if c) else rows

    # Check which columns are numeric across body rows
    if not body:
        return [], None

    n_cols = max(len(r) for r in body)
    numeric_cols: list[tuple[str, list[float]]] = []
    for col_idx in range(n_cols):
        col_name = header[col_idx].lower() if col_idx < len(header) else f"col_{col_idx}"
        if col_name in ("date", "datetime", "timestamp", "time", "index", ""):
            continue
        vals: list[float] = []
        for r in body:
            if col_idx < len(r) and _is_float(r[col_idx]):
                vals.append(float(r[col_idx]))
        if len(vals) >= 2:
            numeric_cols.append((col_name, vals))

    if not numeric_cols:
        raise ValueError(f"Could not find numeric return or equity columns in {path}.")

    # Prefer explicit return column
    for name, vals in numeric_cols:
        if "ret" in name:
            return vals, (len(numeric_cols) if len(numeric_cols) > 1 else None)

    # Check if single or multi-column equity/value vs returns
    for name, vals in numeric_cols:
        if any(tok in name for tok in ("value", "equity", "close", "nav")):
            return _prices_to_returns(vals), None

    # If multiple strategy columns (e.g. a VectorBT parameter grid CSV export), pick best Sharpe
    # and record the column count as asserted trials
    if len(numeric_cols) > 1:
        best_vals: list[float] = []
        best_sr = -float("inf")
        for _name, vals in numeric_cols:
            series = _prices_to_returns(vals) if min(vals) > 10.0 else vals
            if len(series) >= 2:
                arr = np.asarray(series, dtype=float)
                sd = float(arr.std(ddof=1))
                sr = float(arr.mean() / sd) if sd > 0 else 0.0
                if sr > best_sr:
                    best_sr = sr
                    best_vals = series
        return best_vals, len(numeric_cols)

    only_vals = numeric_cols[0][1]
    if min(only_vals) > 10.0:
        return _prices_to_returns(only_vals), None
    return only_vals, None


def _is_float(val: str) -> bool:
    try:
        f = float(val.strip())
        return math.isfinite(f)
    except (TypeError, ValueError):
        return False


def _extract_vbt_object(obj: Any) -> tuple[list[float], int | None]:
    # Support VectorBT Portfolio-like objects (having .returns() or .returns)
    rets_attr = getattr(obj, "returns", None)
    raw = rets_attr() if callable(rets_attr) else rets_attr
    if raw is None and isinstance(obj, Mapping):
        raw = obj.get("returns")
        n_t = obj.get("n_trials")
        if isinstance(raw, Sequence):
            return _clean_returns(raw), int(n_t) if isinstance(n_t, int) else None

    if raw is None:
        raw = obj

    # Handle 2-D array / DataFrame of multiple columns (multi-trial VectorBT portfolio)
    values = getattr(raw, "values", raw)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 2:
        n_cols = int(arr.shape[1])
        best_col = 0
        best_sr = -float("inf")
        for j in range(n_cols):
            col = arr[:, j]
            col = col[np.isfinite(col)]
            if col.size >= 2:
                sd = float(col.std(ddof=1))
                sr = float(col.mean() / sd) if sd > 0 else 0.0
                if sr > best_sr:
                    best_sr = sr
                    best_col = j
        chosen = arr[:, best_col]
        return _clean_returns(chosen.tolist()), (n_cols if n_cols >= 2 else None)
    if arr.ndim == 1:
        return _clean_returns(arr.tolist()), None
    raise ValueError(f"Unsupported VectorBT input shape: {arr.shape}")


def from_vectorbt(
    portfolio_or_csv: Any,
    *,
    n_trials: int | None = None,
    label: str = "",
    notes: str = "",
    risk_free_rate: float | None = None,
) -> StudyWithReturns:
    """Build a :class:`Study` from a VectorBT ``Portfolio`` object, return array, or exported CSV."""
    if isinstance(portfolio_or_csv, (str, Path)):
        p = Path(portfolio_or_csv)
        returns, detected_trials = _extract_vbt_from_csv(p)
        desc = f"vectorbt:{p.name}"
        default_label = p.stem
    else:
        returns, detected_trials = _extract_vbt_object(portfolio_or_csv)
        desc = "vectorbt:portfolio"
        default_label = "vectorbt study"

    trials = n_trials if n_trials is not None else detected_trials
    return _build_interop_study(
        returns,
        label=label or default_label,
        data_description=desc,
        n_trials=trials,
        notes=notes,
        risk_free_rate=risk_free_rate,
    )
