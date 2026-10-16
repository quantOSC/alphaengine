"""Two-state Hamilton filter. The caller supplies the series. Nothing is fetched.

Hamilton, Econometrica 1989: an unobserved Markov state drives the mean and
the variance. The fitted probability path stays with the caller. The figures
are the two states, the transitions, and how sure the filter is now.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

MIN_OBS = 40


def hamilton_regime(values: list[float] | np.ndarray) -> dict:
    """Fit a two-state Gaussian Markov chain and filter it.

    State 1 is the higher-volatility state after the fit, so a bear-market
    reading is the high-vol state even if the optimizer numbered them backwards.
    """
    y = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=float)
    n = int(y.size)
    if n < MIN_OBS:
        return {"error": f"need >= {MIN_OBS} observations", "n_obs": n}

    dev = np.abs(y - float(np.median(y)))
    cut = float(np.median(dev))
    calm = y[dev <= cut]
    wild = y[dev > cut]
    if calm.size < 5 or wild.size < 5:
        calm, wild = y[: n // 2], y[n // 2 :]
    mu_c = float(calm.mean())
    mu_w = float(wild.mean())
    sd_c = max(float(calm.std(ddof=1)), 1e-6)
    sd_w = max(float(wild.std(ddof=1)), 1e-6)

    def nll(theta: np.ndarray) -> float:
        mu0, mu1, log_s0, log_s1, z00, z11 = (float(v) for v in theta)
        s0, s1 = math.exp(log_s0), math.exp(log_s1)
        if s0 < 1e-8 or s1 < 1e-8:
            return 1e12
        stay0 = 1.0 / (1.0 + math.exp(-z00))
        stay1 = 1.0 / (1.0 + math.exp(-z11))
        # trans[i, j] = P(next = j | now = i)
        trans = np.array([[stay0, 1.0 - stay0], [1.0 - stay1, stay1]])
        ll, _filt, _path = _filter(y, (mu0, mu1), (s0, s1), trans)
        if not math.isfinite(ll):
            return 1e12
        return -ll

    start = np.array([mu_c, mu_w, math.log(sd_c), math.log(sd_w), 2.0, 2.0])
    try:
        fit = minimize(nll, start, method="L-BFGS-B")
        fitted = fit.x
        converged = bool(fit.success)
    except (ValueError, FloatingPointError):
        fitted = start
        converged = False
    chosen = _best_state(y, (start, fitted))
    if chosen is None:
        return {"error": "the filter did not fit", "n_obs": n}
    mu0, mu1, s0, s1, stay0, stay1, filt, path = chosen
    high = path[:, 1] >= 0.5
    return {
        "n_obs": n,
        "method": "hamilton_1989",
        "mu_low": round(mu0, 6),
        "mu_high": round(mu1, 6),
        "sigma_low": round(s0, 6),
        "sigma_high": round(s1, 6),
        "p_stay_low": round(stay0, 4),
        "p_stay_high": round(stay1, 4),
        "prob_high_now": round(float(filt[1]), 4),
        "n_high": int(high.sum()),
        "n_low": int((~high).sum()),
        "converged": converged,
        "_path": path,
    }


def _best_state(
    y: np.ndarray, thetas: tuple[np.ndarray, ...]
) -> tuple[float, float, float, float, float, float, np.ndarray, np.ndarray] | None:
    """Keep the parameter set with the higher likelihood. A bad optimum loses to the split-sample start."""
    best_ll = float("-inf")
    best: tuple[float, float, float, float, float, float, np.ndarray, np.ndarray] | None = None
    for theta in thetas:
        unpacked = _unpack(y, theta)
        if unpacked is None:
            continue
        ll, mu0, mu1, s0, s1, stay0, stay1, filt, path = unpacked
        if ll > best_ll:
            best_ll = ll
            best = (mu0, mu1, s0, s1, stay0, stay1, filt, path)
    return best


def _unpack(
    y: np.ndarray, theta: np.ndarray
) -> tuple[float, float, float, float, float, float, float, float, np.ndarray, np.ndarray] | None:
    try:
        mu0, mu1, log_s0, log_s1, z00, z11 = (float(v) for v in theta)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (mu0, mu1, log_s0, log_s1, z00, z11)):
        return None
    s0, s1 = math.exp(log_s0), math.exp(log_s1)
    if s0 < 1e-8 or s1 < 1e-8:
        return None
    stay0 = 1.0 / (1.0 + math.exp(-z00))
    stay1 = 1.0 / (1.0 + math.exp(-z11))
    trans = np.array([[stay0, 1.0 - stay0], [1.0 - stay1, stay1]])
    if s1 < s0:
        mu0, mu1 = mu1, mu0
        s0, s1 = s1, s0
        stay0, stay1 = stay1, stay0
        trans = np.array([[stay0, 1.0 - stay0], [1.0 - stay1, stay1]])
    ll, filt, path = _filter(y, (mu0, mu1), (s0, s1), trans)
    if not math.isfinite(ll):
        return None
    return ll, mu0, mu1, s0, s1, stay0, stay1, filt, path


def _filter(
    y: np.ndarray,
    means: tuple[float, float],
    sigmas: tuple[float, float],
    trans: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Hamilton filter. Returns log likelihood, filtered probs now, and the path."""
    dens = np.column_stack(
        [
            norm.pdf(y, loc=means[0], scale=sigmas[0]),
            norm.pdf(y, loc=means[1], scale=sigmas[1]),
        ]
    )
    # Stationary start from the stay probabilities, falling back to even.
    stay0, stay1 = float(trans[0, 0]), float(trans[1, 1])
    denom = (1.0 - stay0) + (1.0 - stay1)
    if denom <= 1e-9:
        pred = np.array([0.5, 0.5])
    else:
        pred = np.array([(1.0 - stay1) / denom, (1.0 - stay0) / denom])
    ll = 0.0
    path = np.empty((y.size, 2), dtype=float)
    for t in range(y.size):
        joint = pred * dens[t]
        lik = float(joint.sum())
        if lik <= 0.0 or not math.isfinite(lik):
            return float("-inf"), pred, path
        filt = joint / lik
        path[t] = filt
        ll += math.log(lik)
        pred = filt @ trans
    return ll, path[-1], path
