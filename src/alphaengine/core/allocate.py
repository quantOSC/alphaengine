"""Book construction from a covariance: HRP, risk parity, vol targeting.

Hierarchical Risk Parity (López de Prado) allocates without inverting the
covariance — the sample matrix on a wide book is often singular, and a
quadratic optimiser that needs the inverse is then a machine for amplifying
noise. Risk parity equalises risk contribution, starting from inverse-vol.
Vol targeting is a scalar overlay on one return stream.

WEIGHTS SUM TO ONE AND STAY NON-NEGATIVE for HRP and risk parity. That is a
test, not a hope.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.optimize import minimize

from .covariance import _corr_from_cov, cov_diagnostics, ewma_cov, ledoit_wolf_cov

__all__ = ["hrp_weights", "risk_parity_weights", "vol_target", "cov_from_returns"]

_RND = 6


def _rounded_weights(raw: np.ndarray, labels: list[str]) -> dict[str, float]:
    """Round to `_RND` decimals and put the residue on the largest name.

    Independent rounding of a simplex can land 1e-6 short of 1. The book a
    desk actually books has to sum to one, so the last unit of rounding
    error is absorbed by the largest weight rather than reported as 0.999999.
    """
    values = [round(float(x), _RND) for x in raw]
    residue = round(1.0 - sum(values), _RND)
    if residue:
        i = int(np.argmax(raw))
        values[i] = round(values[i] + residue, _RND)
    return {labels[i]: values[i] for i in range(len(labels))}


def _cluster_var(cov: np.ndarray, ix: list[int]) -> float:
    sub = cov[np.ix_(ix, ix)]
    diag = np.clip(np.diag(sub), 1e-18, None)
    w = 1.0 / diag
    w = w / w.sum()
    return float(w @ sub @ w)


def _leaf_order(link: np.ndarray, n: int) -> list[int]:
    """Left-to-right leaf order of a scipy linkage dendrogram."""

    def leaves(node: int) -> list[int]:
        if node < n:
            return [node]
        row = int(node - n)
        left = int(link[row, 0])
        right = int(link[row, 1])
        return leaves(left) + leaves(right)

    root = n + int(link.shape[0]) - 1
    return leaves(root)


def _recursive_bisection(cov: np.ndarray, order: list[int]) -> np.ndarray:
    w = np.ones(len(order))
    clusters: list[list[int]] = [list(order)]
    while clusters:
        nxt: list[list[int]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left, right = cluster[:mid], cluster[mid:]
            v_l = _cluster_var(cov, left)
            v_r = _cluster_var(cov, right)
            denom = v_l + v_r
            alpha = 0.5 if denom <= 0 else 1.0 - v_l / denom
            w[left] *= alpha
            w[right] *= 1.0 - alpha
            if len(left) > 1:
                nxt.append(left)
            if len(right) > 1:
                nxt.append(right)
        clusters = nxt
    total = float(w.sum())
    return w / total if total else w


def hrp_weights(cov: np.ndarray, *, names: list[str] | None = None) -> dict[str, Any]:
    """Hierarchical Risk Parity weights from a covariance. No matrix inverse."""
    C = np.asarray(cov, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1] or C.shape[0] < 2:
        raise ValueError("hrp_weights needs a square covariance of at least two names.")
    n = C.shape[0]
    labels = names if names is not None and len(names) == n else [str(i) for i in range(n)]
    corr, _ = _corr_from_cov(C)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    condensed = dist[np.triu_indices(n, k=1)]
    link = linkage(condensed, method="single")
    order = _leaf_order(link, n)
    raw = _recursive_bisection(C, order)
    weights = _rounded_weights(raw, labels)
    out = {
        "weights": weights,
        "n_assets": n,
        "method": "hrp",
        "weight_sum": round(sum(weights.values()), _RND),
        "max_weight": round(float(raw.max()), _RND),
        "min_weight": round(float(raw.min()), _RND),
        "n_negative": int(np.sum(raw < -1e-12)),
    }
    out.update(cov_diagnostics(C))
    return out


def risk_parity_weights(cov: np.ndarray, *, names: list[str] | None = None) -> dict[str, Any]:
    """Equal-risk-contribution weights. Inverse-vol is the start, SLSQP the finish."""
    C = np.asarray(cov, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1] or C.shape[0] < 2:
        raise ValueError("risk_parity_weights needs a square covariance of at least two names.")
    n = C.shape[0]
    labels = names if names is not None and len(names) == n else [str(i) for i in range(n)]
    vol = np.sqrt(np.clip(np.diag(C), 1e-18, None))
    w0 = 1.0 / vol
    w0 = w0 / w0.sum()

    def objective(w: np.ndarray) -> float:
        w = np.clip(w, 1e-12, None)
        rc = w * (C @ w)
        return float(np.sum((rc - rc.mean()) ** 2))

    bounds = [(0.0, 1.0)] * n
    cons = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
    result = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons, tol=1e-12)
    raw = np.clip(result.x if result.success else w0, 0.0, None)
    raw = raw / raw.sum()
    weights = _rounded_weights(raw, labels)
    out = {
        "weights": weights,
        "n_assets": n,
        "method": "risk_parity",
        "weight_sum": round(sum(weights.values()), _RND),
        "max_weight": round(float(raw.max()), _RND),
        "min_weight": round(float(raw.min()), _RND),
        "converged": bool(result.success),
        "n_negative": int(np.sum(raw < -1e-12)),
    }
    out.update(cov_diagnostics(C))
    return out


def vol_target(
    returns: list[float] | np.ndarray,
    *,
    target: float = 0.10,
    ewma_lambda: float = 0.94,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Scalar overlay: leverage so EWMA vol matches `target` (annualised).

    The scaled path stays on this machine. Figures: realised vol of the overlay,
    mean leverage, how many observations. A zero or tiny EWMA vol takes
    leverage 0 rather than exploding.
    """
    r = np.asarray(returns, dtype=float).reshape(-1)
    r = r[np.isfinite(r)]
    n = int(r.size)
    if n < 2:
        return {
            "target": float(target),
            "n_obs": n,
            "mean_leverage": None,
            "realized_vol": None,
            "ewma_lambda": float(ewma_lambda),
        }
    lam = min(max(float(ewma_lambda), 1e-6), 0.999999)
    warm = min(20, n)
    var = float(np.mean(r[:warm] ** 2))
    scaled = np.empty(n)
    leverages = np.empty(n)
    ppy = max(int(periods_per_year), 1)
    tgt = float(target)
    for i, x in enumerate(r):
        var = lam * var + (1.0 - lam) * x * x
        vol_ann = math.sqrt(max(var, 0.0) * ppy)
        lev = 0.0 if vol_ann <= 1e-12 else tgt / vol_ann
        leverages[i] = lev
        scaled[i] = x * lev
    realized = float(scaled.std(ddof=1) * math.sqrt(ppy)) if n > 1 else None
    return {
        "target": tgt,
        "n_obs": n,
        "mean_leverage": round(float(leverages.mean()), _RND),
        "realized_vol": None if realized is None else round(realized, _RND),
        "ewma_lambda": lam,
        "periods_per_year": ppy,
        "max_leverage": round(float(leverages.max()), _RND),
    }


def cov_from_returns(
    returns: np.ndarray,
    *,
    method: str = "sample",
    lam: float = 0.94,
) -> np.ndarray:
    """Build a covariance the allocators can consume. `method`: sample|ewma|lw."""
    R = np.asarray(returns, dtype=float)
    method = str(method or "sample").lower()
    if method == "ewma":
        return ewma_cov(R, lam=lam)
    if method in ("lw", "ledoit_wolf", "ledoit-wolf"):
        return ledoit_wolf_cov(R)[0]
    if R.shape[0] < 2:
        raise ValueError("a covariance needs T >= 2.")
    X = R - R.mean(axis=0)
    return (X.T @ X) / X.shape[0]
