"""
Performance, risk-adjusted metrics over a supplied return stream.

Pure math, no data layer, no FRED. The risk-free rate is an explicit parameter
(default 0.0) so results are reproducible and the no-fetch invariant holds, the
upstream backend performance.py fetched the 3-month T-bill from FRED, which is
exactly the landmine this copy removes.

Sharpe, Sortino, Calmar, max drawdown, total/annualized return, VaR/CVaR(95),
and alpha/beta/information-ratio when a benchmark return series is supplied.
Loss metrics (max_drawdown, var, cvar) are reported as POSITIVE magnitudes in
both decimal and percent; the sign convention is documented per field.
"""

from __future__ import annotations

import math

import numpy as np

_PPY = 252  # trading periods per year


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def performance_report(
    returns: list[float],
    *,
    equity_curve: list[float] | None = None,
    benchmark_returns: list[float] | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = _PPY,
) -> dict:
    """Risk-adjusted performance metrics for a daily return series (decimals).

    `risk_free_rate` is ANNUAL (e.g. 0.04); supply it explicitly, the engine
    never fetches it. Loss metrics are positive magnitudes.
    """
    arr = np.asarray([float(r) for r in (returns or []) if r is not None], dtype=float)
    n = arr.size
    if n < 30:
        return {"error": "need >= 30 observations", "n_obs": int(n)}

    ppy = int(periods_per_year)
    rf_per = float(risk_free_rate) / ppy
    excess = arr - rf_per
    sd = float(arr.std(ddof=1))

    sharpe = float(excess.mean() / sd) if sd > 0 else 0.0
    sharpe_ann = sharpe * math.sqrt(ppy)

    # Downside deviation is the TARGET SEMIDEVIATION: RMS of below-target
    # excess around the target (0), averaged over ALL N, not the std of the
    # negative subset (which measures spread around the negatives' own mean).
    # Math-identical to backend/quant/performance.sortino_ratio (M1 fix).
    downside = np.minimum(excess, 0.0)
    dsd = float(np.sqrt(np.mean(downside ** 2))) if np.count_nonzero(downside) else 0.0
    sortino_ann = (float(excess.mean()) / dsd) * math.sqrt(ppy) if dsd > 0 else 0.0

    eq = np.asarray(equity_curve, dtype=float) if equity_curve else np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(eq)
    drawdown = eq / peak - 1.0
    max_dd = float(drawdown.min()) if drawdown.size else 0.0

    total_return = float(eq[-1] - 1.0) if eq.size else 0.0
    ann_return = float((1.0 + total_return) ** (ppy / n) - 1.0) if total_return > -1 else -1.0
    calmar = float(ann_return / abs(max_dd)) if max_dd < 0 else 0.0

    cut = float(np.percentile(arr, 5))
    var95 = abs(cut)
    tail = arr[arr <= cut]
    cvar95 = abs(float(tail.mean())) if tail.size else var95

    out = {
        "n_obs": int(n),
        "sharpe_ratio": _clean(round(sharpe, 4)),            # per-period
        "sharpe_annualized": _clean(round(sharpe_ann, 4)),
        "sortino_ratio": _clean(round(sortino_ann, 4)),      # annualized
        "calmar_ratio": _clean(round(calmar, 4)),
        "max_drawdown_pct": _clean(round(abs(max_dd) * 100, 2)),   # positive magnitude
        "total_return_pct": _clean(round(total_return * 100, 2)),
        "annualized_return_pct": _clean(round(ann_return * 100, 2)),
        "var_95": _clean(round(var95, 4)),                   # positive daily loss fraction
        "cvar_95": _clean(round(cvar95, 4)),
        "volatility_annualized_pct": _clean(round(sd * math.sqrt(ppy) * 100, 2)),
        "risk_free_rate": round(float(risk_free_rate), 4),
    }

    if benchmark_returns:
        b = np.asarray([float(x) for x in benchmark_returns], dtype=float)
        m = min(len(b), n)
        if m >= 30:
            a2, b2 = arr[-m:] - rf_per, b[-m:] - rf_per
            var_b = float(np.var(b2, ddof=1))
            beta = float(np.cov(a2, b2, ddof=1)[0, 1] / var_b) if var_b > 0 else 0.0
            alpha_per = float(a2.mean() - beta * b2.mean())
            active = arr[-m:] - b[-m:]
            te = float(active.std(ddof=1))
            ir = (float(active.mean()) / te) * math.sqrt(ppy) if te > 0 else 0.0
            out["beta"] = _clean(round(beta, 4))
            out["alpha_annualized_pct"] = _clean(round(alpha_per * ppy * 100, 2))
            out["information_ratio"] = _clean(round(ir, 4))

    return out
