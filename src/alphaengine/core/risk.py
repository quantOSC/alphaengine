"""
Risk, VaR / CVaR over a supplied portfolio return stream.

Lifted (math-identical) from backend/quant/risk.py's returns-based paths, with
the data/limits coupling removed: the caller supplies a daily portfolio return
series; nothing is fetched. Three layers of VaR rigor plus Expected Shortfall:

  1. Parametric Gaussian VaR, -mean·h + z·σ·√h, floored at 0.
  2. Cornish-Fisher VaR, expands z by skew/kurtosis (observed non-normality).
  3. Historical-percentile VaR + bootstrap CI (deterministic seed).
  4. CVaR (Expected Shortfall), the tail mean scaled the same way as the percentile.

z comes from scipy's inverse-normal so non-{0.95,0.99} confidences are exact.
Pure numpy/scipy. Deterministic given inputs on the pinned stack.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from .trace import record


def _clean(val):
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val


def _parametric_loss(mean: float, z: float, std: float, horizon: int) -> float:
    """Positive loss fraction. The mean scales with the horizon; the shock with its square root."""
    h = float(horizon)
    return max(0.0, -mean * h + z * std * math.sqrt(h))


def _quantile_loss(point: float, mean: float, horizon: int) -> float:
    """Scale a one-day return quantile to a horizon without dropping the drift.

    At one day this is ``max(0, -point)``, the old absolute percentile, whenever
    that percentile is a loss. Over h days the mean moves with h and the
    distance from the mean moves with sqrt(h).
    """
    h = float(horizon)
    return max(0.0, -(mean * h + (point - mean) * math.sqrt(h)))


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
    arr = np.array(
        [
            r
            for r in (portfolio_returns or [])
            if r is not None and not (isinstance(r, float) and np.isnan(r))
        ],
        dtype=float,
    )
    n_obs = arr.size
    if n_obs < 20:
        return {"error": "need >= 20 observations", "n_obs": int(n_obs)}

    z = float(stats.norm.ppf(confidence))
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n_obs > 1 else 0.0

    # 1. Parametric Gaussian VaR. Drift over the horizon, shock over its square root.
    parametric_daily = _parametric_loss(mean, z, std, horizon_days)
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
            z + (z**2 - 1) * skew / 6.0 + (z**3 - 3 * z) * kurt / 24.0 - (2 * z**3 - 5 * z) * (skew**2) / 36.0
        )
        # The CF expansion is only valid near-normal; extreme skew/kurtosis
        # can drive z_cf <= 0, which would report a NEGATIVE loss (violating
        # the positive-loss convention). Outside the validity domain, fall
        # back to the Gaussian z and say so, never fabricate from a broken
        # expansion.
        cf_invalid = bool(z_cf <= 0)
        z_eff = z if cf_invalid else z_cf
        cf_daily = _parametric_loss(mean, z_eff, std, horizon_days)
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
    h = int(horizon_days)
    samples = rng.choice(arr, size=(bootstrap_samples, n_obs), replace=True)
    boot_points = np.percentile(samples, tail_pct, axis=1)
    boot_losses = np.maximum(0.0, -(mean * h + (boot_points - mean) * math.sqrt(h)))
    lo, hi = np.percentile(boot_losses, [2.5, 97.5])
    point = _quantile_loss(float(np.percentile(arr, tail_pct)), mean, h)
    result["historical"] = {
        "var_pct": _clean(round(point * 100, 2)),
        "ci_95_low_pct": _clean(round(float(lo) * 100, 2)),
        "ci_95_high_pct": _clean(round(float(hi) * 100, 2)),
        "bootstrap_samples": bootstrap_samples,
    }

    # 4. CVaR (Expected Shortfall). The tail mean is scaled like the percentile,
    # so a longer horizon does not leave expected shortfall as a one-day number.
    var_cutoff = float(np.percentile(arr, tail_pct))
    tail = arr[arr <= var_cutoff]
    tail_mean = float(np.mean(tail)) if tail.size > 0 else var_cutoff
    cvar = _quantile_loss(tail_mean, mean, h)
    var_loss = _quantile_loss(var_cutoff, mean, h)
    result["cvar"] = {
        "cvar_pct": _clean(round(cvar * 100, 2)),
        "cvar_dollars": _clean(round(cvar * portfolio_value, 2)),
        "var_pct": _clean(round(var_loss * 100, 2)),
        "tail_observations": int(tail.size),
    }

    record(
        "parametric_var",
        "max(0, -mean*h + z*std(r, ddof=1)*sqrt(h)), z = normsinv(confidence), as a positive percent",
        inputs={
            "n_obs": int(n_obs),
            "confidence": confidence,
            "horizon_days": int(horizon_days),
            "z": round(z, 6),
        },
        series={"returns": arr},
        result=result["parametric"]["var_pct"],
    )
    if "historical" in result:
        record(
            "historical_var",
            "max(0, -(mean*h + (percentile - mean)*sqrt(h))), as a positive percent",
            inputs={"n_obs": int(n_obs), "confidence": confidence, "horizon_days": int(horizon_days)},
            series={"returns": arr},
            result=result["historical"]["var_pct"],
        )
    record(
        "cvar",
        "the tail mean, scaled like the historical percentile, as a positive percent",
        inputs={"n_obs": int(n_obs), "confidence": confidence, "tail_observations": int(tail.size)},
        series={"returns": arr, "tail": tail},
        result=result["cvar"]["cvar_pct"],
    )
    return result
