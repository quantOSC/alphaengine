"""
Risk, VaR / CVaR over a supplied portfolio return stream.

Lifted (math-identical) from backend/quant/risk.py's returns-based paths, with
the data/limits coupling removed: the caller supplies a daily portfolio return
series; nothing is fetched. Three layers of VaR rigor plus Expected Shortfall:

  1. Parametric Gaussian VaR, z·σ·√horizon.
  2. Cornish-Fisher VaR, expands z by skew/kurtosis (observed non-normality).
  3. Historical-percentile VaR + bootstrap CI (deterministic seed).
  4. CVaR (Expected Shortfall), mean of the tail beyond the percentile VaR.

z comes from scipy's inverse-normal so non-{0.95,0.99} confidences are exact.
Pure numpy/scipy. Deterministic given inputs on the pinned stack.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def _clean(val):
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val


def compute_var_cvar(
    portfolio_returns: list[float],
    *,
    confidence: float = 0.95,
    horizon_days: int = 1,
    portfolio_value: float = 100_000.0,
    bootstrap_samples: int = 1000,
) -> dict:
    """Portfolio VaR (parametric + Cornish-Fisher + historical) and CVaR.

    `portfolio_returns`: historical daily portfolio returns as decimals. VaR is
    reported as a positive loss fraction (and dollars at `portfolio_value`).
    """
    arr = np.array([r for r in (portfolio_returns or []) if r is not None and not (isinstance(r, float) and np.isnan(r))], dtype=float)
    n_obs = arr.size
    if n_obs < 20:
        return {"error": "need >= 20 observations", "n_obs": int(n_obs)}

    z = float(stats.norm.ppf(confidence))
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n_obs > 1 else 0.0

    # 1. Parametric Gaussian VaR (per-period vol scaled to horizon).
    parametric_daily = z * std * math.sqrt(horizon_days)
    result = {
        "n_obs": int(n_obs),
        "confidence": confidence,
        "horizon_days": int(horizon_days),
        "low_sample": bool(n_obs < 60),
        "parametric": {
            "var_pct": _clean(round(parametric_daily * 100, 2)),
            "var_dollars": _clean(round(parametric_daily * portfolio_value, 2)),
            "daily_vol_pct": _clean(round(std * 100, 2)),
        },
        "method": "parametric_gaussian+cornish_fisher+historical",
    }

    # 2. Cornish-Fisher adjusted VaR.
    if std > 0:
        skew = float(np.mean(((arr - mean) / std) ** 3))
        kurt = float(np.mean(((arr - mean) / std) ** 4) - 3.0)  # excess
        z_cf = (
            z
            + (z ** 2 - 1) * skew / 6.0
            + (z ** 3 - 3 * z) * kurt / 24.0
            - (2 * z ** 3 - 5 * z) * (skew ** 2) / 36.0
        )
        # The CF expansion is only valid near-normal; extreme skew/kurtosis
        # can drive z_cf <= 0, which would report a NEGATIVE loss (violating
        # the positive-loss convention). Outside the validity domain, fall
        # back to the Gaussian z and say so, never fabricate from a broken
        # expansion.
        cf_invalid = bool(z_cf <= 0)
        z_eff = z if cf_invalid else z_cf
        cf_daily = z_eff * std * math.sqrt(horizon_days)
        result["cornish_fisher"] = {
            "var_pct": _clean(round(cf_daily * 100, 2)),
            "var_dollars": _clean(round(cf_daily * portfolio_value, 2)),
            "skewness": _clean(round(skew, 3)),
            "excess_kurtosis": _clean(round(kurt, 3)),
            "z_adjusted": _clean(round(float(z_eff), 3)),
            "cf_fallback_gaussian": cf_invalid,
        }

    # 3. Historical-percentile VaR + bootstrap CI (deterministic seed).
    rng = np.random.default_rng(42)
    tail_pct = (1 - confidence) * 100
    samples = rng.choice(arr, size=(bootstrap_samples, n_obs), replace=True)
    boot_vars = np.percentile(samples, tail_pct, axis=1) * math.sqrt(horizon_days)
    lo, hi = np.percentile(boot_vars, [2.5, 97.5])
    point = float(np.percentile(arr, tail_pct)) * math.sqrt(horizon_days)
    # Report VaR as a positive loss magnitude. The tail percentiles are
    # negative returns, so abs() flips their order, sort so low <= high.
    ci_low, ci_high = sorted((abs(float(lo)) * 100, abs(float(hi)) * 100))
    result["historical"] = {
        "var_pct": _clean(round(abs(point) * 100, 2)),
        "ci_95_low_pct": _clean(round(ci_low, 2)),
        "ci_95_high_pct": _clean(round(ci_high, 2)),
        "bootstrap_samples": bootstrap_samples,
    }

    # 4. CVaR (Expected Shortfall), mean of the tail beyond the percentile VaR.
    var_cutoff = np.percentile(arr, tail_pct)
    tail = arr[arr <= var_cutoff]
    cvar = float(np.mean(tail)) if tail.size > 0 else float(var_cutoff)
    result["cvar"] = {
        "cvar_pct": _clean(round(abs(cvar) * 100, 2)),
        "cvar_dollars": _clean(round(abs(cvar) * portfolio_value, 2)),
        "var_pct": _clean(round(abs(float(var_cutoff)) * 100, 2)),
        "tail_observations": int(tail.size),
    }

    return result
