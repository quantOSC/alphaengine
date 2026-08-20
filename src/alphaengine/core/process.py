"""Named data-generating processes for equity research.

THE MODELING SUITE THE REST OF THE LIBRARY WAS MISSING. IC, Fama-MacBeth and
HRP measure a path the caller already has. These estimate a LAW for that path
(OU, GBM, jumps, GARCH) and, when asked, stress a book under Monte Carlo draws
from that law. Paths stay on this machine. Figures that travel are scalars,
bucketed sketches, histograms and quantile bands.

DISCRETE TIME, DAILY DT=1. Continuous SDEs are written as their exact or Euler
maps so a study reproduces. Seeds are explicit: the same inputs and seed return
the same figures on every machine.

TRIAL COUNT HONESTY. A Monte Carlo stress records `n_trials = n_paths` with
`n_trials_source = monte_carlo`. An unrecorded path count cannot reach `edge`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .series_shapes import series_values

__all__ = [
    "ou_calibrate",
    "ou_simulate",
    "gbm_calibrate",
    "jump_calibrate",
    "garch_calibrate",
    "dgp_stress",
    "as_series",
    "sketch_curve",
    "sketch_band",
    "histogram",
]

_RND = 6
SKETCH = 256
HIST_BINS = 24
MIN_OBS = 30
MAX_PATHS = 512


def as_series(values: Any) -> np.ndarray:
    """One numeric series. Dated maps are sorted; anything else is empty."""
    arr, _ = series_values(values)
    if arr.ndim == 1 and arr.size:
        return arr[np.isfinite(arr)]
    try:
        out = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)
    return out[np.isfinite(out)]


def sketch_curve(values: np.ndarray | list[float], *, max_points: int = SKETCH) -> list[dict[str, Any]]:
    """Position-bucketed `{i, v}` sketch. Never longer than `max_points`."""
    arr = np.asarray(list(values), dtype=float)
    n = int(arr.size)
    if n == 0:
        return []
    k = min(int(max_points), n)
    bounds = [round(i * n / k) for i in range(k + 1)]
    out: list[dict[str, Any]] = []
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        if hi <= lo:
            continue
        out.append({"i": hi - 1, "v": round(float(arr[lo:hi][-1]), _RND)})
    return out


def sketch_band(
    lo: np.ndarray, mid: np.ndarray, hi: np.ndarray, *, max_points: int = SKETCH
) -> list[dict[str, Any]]:
    """Fan chart `{i, lo, mid, hi}` from three aligned paths."""
    n = min(len(lo), len(mid), len(hi))
    if n == 0:
        return []
    k = min(int(max_points), n)
    bounds = [round(i * n / k) for i in range(k + 1)]
    out: list[dict[str, Any]] = []
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        if b <= a:
            continue
        i = b - 1
        out.append(
            {
                "i": i,
                "lo": round(float(lo[i]), _RND),
                "mid": round(float(mid[i]), _RND),
                "hi": round(float(hi[i]), _RND),
            }
        )
    return out


def histogram(values: np.ndarray | list[float], *, bins: int = HIST_BINS) -> dict[str, Any]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    n_bins = max(4, min(int(bins), 48))
    if arr.size == 0:
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(arr, bins=n_bins)
    return {
        "edges": [round(float(e), _RND) for e in edges],
        "counts": [int(c) for c in counts],
    }


def _sharpe(returns: np.ndarray) -> float | None:
    r = returns[np.isfinite(returns)]
    if r.size < 2:
        return None
    sd = float(r.std(ddof=1))
    if sd == 0 or not math.isfinite(sd):
        return None
    return float(r.mean() / sd * math.sqrt(252.0))


def ou_calibrate(values: Any, *, dt: float = 1.0) -> dict[str, Any]:
    """Exact AR(1) map of a discrete OU: kappa, theta, sigma, half_life.

    X_t = theta*(1-phi) + phi*X_{t-1} + eps, phi = exp(-kappa*dt).
    None figures when the series is too short or not mean-reverting (phi not
    in (0, 1)).
    """
    x = as_series(values)
    n = int(x.size)
    empty = {
        "kappa": None,
        "theta": None,
        "sigma": None,
        "half_life": None,
        "phi": None,
        "n_obs": n,
        "method": "ou",
        "mean_reverting": False,
        "path_sketch": sketch_curve(x),
    }
    if n < MIN_OBS:
        return empty
    y, lag = x[1:], x[:-1]
    A = np.column_stack([np.ones(len(lag)), lag])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, phi = float(coef[0]), float(coef[1])
    if not (0.0 < phi < 1.0) or not math.isfinite(phi):
        empty["phi"] = round(phi, _RND) if math.isfinite(phi) else None
        return empty
    dt = max(float(dt), 1e-9)
    kappa = -math.log(phi) / dt
    theta = a / (1.0 - phi)
    resid = y - (a + phi * lag)
    var_e = float(np.mean(resid**2))
    denom = 1.0 - phi * phi
    sigma = math.sqrt(max(var_e * 2.0 * kappa / denom, 0.0)) if denom > 0 else math.sqrt(max(var_e, 0.0))
    half = math.log(2.0) / kappa if kappa > 0 else None
    return {
        "kappa": round(kappa, _RND),
        "theta": round(theta, _RND),
        "sigma": round(sigma, _RND),
        "half_life": None if half is None else round(half, _RND),
        "phi": round(phi, _RND),
        "n_obs": n,
        "method": "ou",
        "mean_reverting": True,
        "path_sketch": sketch_curve(x),
    }


def ou_simulate(
    *,
    kappa: float,
    theta: float,
    sigma: float,
    n_obs: int,
    n_paths: int = 200,
    x0: float | None = None,
    dt: float = 1.0,
    seed: int = 7,
) -> dict[str, Any]:
    """Euler paths of an OU. Raw paths stay with the caller via `paths`."""
    n = max(int(n_obs), 2)
    p = max(1, min(int(n_paths), MAX_PATHS))
    rng = np.random.default_rng(int(seed))
    dt = max(float(dt), 1e-9)
    start = float(theta if x0 is None else x0)
    shock = sigma * math.sqrt(dt)
    paths = np.empty((p, n), dtype=float)
    paths[:, 0] = start
    z = rng.normal(0.0, 1.0, size=(p, n - 1))
    for t in range(1, n):
        prev = paths[:, t - 1]
        paths[:, t] = prev + kappa * (theta - prev) * dt + shock * z[:, t - 1]
    lo = np.quantile(paths, 0.10, axis=0)
    mid = np.quantile(paths, 0.50, axis=0)
    hi = np.quantile(paths, 0.90, axis=0)
    terminal = paths[:, -1]
    hit = []
    band = 0.1 * abs(theta) + 1e-9
    for row in paths:
        ix = np.where(np.abs(row - theta) <= band)[0]
        hit.append(float(ix[0]) if len(ix) else float(n))
    return {
        "n_obs": n,
        "n_paths": p,
        "n_trials": p,
        "n_trials_source": "monte_carlo",
        "method": "ou",
        "kappa": round(float(kappa), _RND),
        "theta": round(float(theta), _RND),
        "sigma": round(float(sigma), _RND),
        "terminal_mean": round(float(terminal.mean()), _RND),
        "terminal_vol": round(float(terminal.std(ddof=1)), _RND) if p > 1 else None,
        "hit_time_median": round(float(np.median(hit)), _RND),
        "mean_path": sketch_curve(mid),
        "band": sketch_band(lo, mid, hi),
        "paths": paths,
    }


def gbm_calibrate(values: Any) -> dict[str, Any]:
    """Log-return moments of a price (or return) series. Discrete GBM."""
    x = as_series(values)
    n = int(x.size)
    if n < 3:
        return {"mu": None, "sigma": None, "n_obs": n, "method": "gbm"}
    if np.all(x > 0) and np.median(np.abs(x)) > 1.0:
        r = np.diff(np.log(x))
    else:
        r = x
    r = r[np.isfinite(r)]
    if r.size < 2:
        return {"mu": None, "sigma": None, "n_obs": n, "method": "gbm"}
    return {
        "mu": round(float(r.mean()), _RND),
        "sigma": round(float(r.std(ddof=1)), _RND),
        "n_obs": int(r.size),
        "method": "gbm",
    }


def jump_calibrate(values: Any, *, z: float = 3.0) -> dict[str, Any]:
    """Merton-style split: diffusion vs rare jumps at |r| > z * robust scale."""
    x = as_series(values)
    n = int(x.size)
    empty = {
        "mu": None,
        "sigma": None,
        "jump_lambda": None,
        "jump_mu": None,
        "jump_sigma": None,
        "n_jumps_named": 0,
        "n_obs": n,
        "method": "jump",
    }
    if n < MIN_OBS:
        return empty
    if np.all(x > 0) and np.median(np.abs(x)) > 1.0:
        r = np.diff(np.log(x))
    else:
        r = x
    r = r[np.isfinite(r)]
    if r.size < MIN_OBS:
        empty["n_obs"] = int(r.size)
        return empty
    mad = float(np.median(np.abs(r - np.median(r)))) * 1.4826
    scale = mad if mad > 1e-12 else float(r.std(ddof=1))
    thresh = float(z) * scale
    jumps = np.abs(r) > thresh
    n_j = int(jumps.sum())
    diff = r[~jumps]
    jmp = r[jumps]
    return {
        "mu": round(float(diff.mean()), _RND) if diff.size else None,
        "sigma": round(float(diff.std(ddof=1)), _RND) if diff.size > 1 else None,
        "jump_lambda": round(n_j / float(r.size), _RND),
        "jump_mu": round(float(jmp.mean()), _RND) if jmp.size else 0.0,
        "jump_sigma": round(float(jmp.std(ddof=1)), _RND) if jmp.size > 1 else 0.0,
        "n_jumps_named": n_j,
        "n_obs": int(r.size),
        "method": "jump",
        "threshold": round(thresh, _RND),
    }


def garch_calibrate(values: Any) -> dict[str, Any]:
    """GARCH(1,1) QMLE. Persistence is alpha+beta, clipped below 1."""
    x = as_series(values)
    if np.all(x > 0) and x.size > 2 and np.median(np.abs(x)) > 1.0:
        r = np.diff(np.log(x))
    else:
        r = x
    r = r[np.isfinite(r)]
    n = int(r.size)
    empty = {
        "omega": None,
        "alpha": None,
        "beta": None,
        "persistence": None,
        "n_obs": n,
        "method": "garch",
        "vol_path": [],
        "var_path": None,
    }
    if n < MIN_OBS:
        return empty
    var0 = float(np.mean(r**2))

    def nll(p: np.ndarray) -> float:
        omega, alpha, beta = float(p[0]), float(p[1]), float(p[2])
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e12
        s2 = var0
        acc = 0.0
        for x_t in r:
            s2 = max(s2, 1e-18)
            acc += math.log(s2) + (x_t * x_t) / s2
            s2 = omega + alpha * x_t * x_t + beta * s2
        return 0.5 * acc

    x0 = np.array([var0 * 0.05, 0.05, 0.90], dtype=float)
    bounds = [(1e-18, None), (0.0, 0.999), (0.0, 0.999)]
    cons = {"type": "ineq", "fun": lambda p: 0.998 - p[1] - p[2]}
    result = minimize(nll, x0, method="SLSQP", bounds=bounds, constraints=cons, tol=1e-12)
    omega, alpha, beta = result.x if result.success else x0
    omega, alpha, beta = float(omega), float(alpha), float(beta)
    persist = alpha + beta
    s2 = var0
    var_path = np.empty(n)
    for i, x_t in enumerate(r):
        s2 = max(s2, 1e-18)
        var_path[i] = s2
        s2 = omega + alpha * x_t * x_t + beta * s2
    vol = np.sqrt(var_path)
    return {
        "omega": round(omega, _RND),
        "alpha": round(alpha, _RND),
        "beta": round(beta, _RND),
        "persistence": round(persist, _RND),
        "n_obs": n,
        "method": "garch",
        "converged": bool(result.success),
        "vol_path": sketch_curve(vol),
        "var_path": var_path,
    }


def _simulate_returns(dgp: str, n: int, cal: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    if dgp == "ou":
        kappa = float(cal.get("kappa") or 0.1)
        theta = float(cal.get("theta") or 0.0)
        sigma = float(cal.get("sigma") or 0.01)
        x = np.empty(n)
        x[0] = theta
        for t in range(1, n):
            x[t] = x[t - 1] + kappa * (theta - x[t - 1]) + sigma * rng.normal()
        return np.diff(x, prepend=x[0])
    if dgp == "jump":
        mu = float(cal.get("mu") or 0.0)
        sig = float(cal.get("sigma") or 0.01)
        lam = float(cal.get("jump_lambda") or 0.0)
        jmu = float(cal.get("jump_mu") or 0.0)
        jsig = float(cal.get("jump_sigma") or 0.0)
        r = rng.normal(mu, max(sig, 1e-12), n)
        jumps = rng.random(n) < lam
        r = r + jumps * rng.normal(jmu, max(jsig, 1e-12), n)
        return r
    if dgp == "garch":
        omega = float(cal.get("omega") or 1e-6)
        alpha = float(cal.get("alpha") or 0.05)
        beta = float(cal.get("beta") or 0.9)
        s2 = omega / max(1.0 - alpha - beta, 1e-6)
        r = np.empty(n)
        for t in range(n):
            s2 = max(omega + alpha * (r[t - 1] ** 2 if t else s2) + beta * s2, 1e-18)
            r[t] = math.sqrt(s2) * rng.normal()
        return r
    mu = float(cal.get("mu") or 0.0)
    sig = float(cal.get("sigma") or 0.01)
    return rng.normal(mu, max(sig, 1e-12), n)


def dgp_stress(
    values: Any,
    *,
    dgp: str = "gbm",
    n_paths: int = 200,
    seed: int = 7,
    backtest_fn: Any = None,
) -> dict[str, Any]:
    """Replay a series (or backtest) under Monte Carlo draws from a named DGP.

    Default: Sharpe of each simulated return path. If `backtest_fn` is supplied
    it is called as `backtest_fn(data={"close": prices})` on a GBM-style close
    rebuilt from the simulated returns; a failing call falls back to the
    return-path Sharpe so a stress still reports.
    """
    x = as_series(values)
    name = str(dgp or "gbm").lower()
    if name not in ("ou", "gbm", "jump", "garch"):
        name = "gbm"
    if name == "ou":
        cal = ou_calibrate(x)
    elif name == "jump":
        cal = jump_calibrate(x)
    elif name == "garch":
        cal = garch_calibrate(x)
    else:
        cal = gbm_calibrate(x)
    n = max(int(cal.get("n_obs") or x.size), 2)
    p = max(1, min(int(n_paths), MAX_PATHS))
    rng = np.random.default_rng(int(seed))
    sharpes: list[float] = []
    for _ in range(p):
        r = _simulate_returns(name, n, cal, rng)
        if backtest_fn is not None:
            px = 100.0 * np.cumprod(1.0 + r)
            try:
                raw = backtest_fn(data={"close": px.tolist()})
                series = as_series(raw)
                s = _sharpe(series) if series.size else _sharpe(r)
            except Exception:
                s = _sharpe(r)
        else:
            s = _sharpe(r)
        if s is not None and math.isfinite(s):
            sharpes.append(float(s))
    arr = np.asarray(sharpes, dtype=float)
    q = np.quantile(arr, [0.10, 0.50, 0.90]) if arr.size else [None, None, None]
    pos = float(np.mean(arr > 0)) if arr.size else None
    figures = {k: cal[k] for k in cal if k not in ("path_sketch", "vol_path", "var_path", "paths")}
    figures.update(
        {
            "dgp": name,
            "n_paths": p,
            "n_trials": p,
            "n_trials_source": "monte_carlo",
            "n_ok": int(arr.size),
            "sharpe_q10": None if q[0] is None else round(float(q[0]), _RND),
            "sharpe_q50": None if q[1] is None else round(float(q[1]), _RND),
            "sharpe_q90": None if q[2] is None else round(float(q[2]), _RND),
            "frac_positive": None if pos is None else round(pos, _RND),
            "sharpe_hist": histogram(arr),
            "method": name,
        }
    )
    return figures
