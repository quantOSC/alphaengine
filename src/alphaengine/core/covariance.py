"""Covariance estimators a desk re-runs overnight.

Sample covariance on a wide book is mostly noise. These are the published
fixes that stay on numpy/scipy: RiskMetrics EWMA, Ledoit-Wolf shrinkage toward
a scaled identity, and Marchenko-Pastur eigenvalue clipping (Laloux;
López de Prado). Detoning strips the market mode before you cluster.

THE MATRIX STAYS HERE. Callers who need weights ask `allocate`. Figures that
travel are diagnostics: condition number, shrinkage intensity, how many
eigenvalues survived the clip — never a 500-name covariance.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .panel import align_panel

__all__ = [
    "returns_matrix",
    "ewma_cov",
    "ledoit_wolf_cov",
    "denoise_cov",
    "detone_cov",
    "cov_diagnostics",
    "variance_explained",
]

_RND = 6


def returns_matrix(panel: dict) -> tuple[np.ndarray, list[str], list[str]]:
    """T x N return matrix, names, skipped. Same alignment as a signal panel."""
    return align_panel(panel)


def ewma_cov(returns: np.ndarray, *, lam: float = 0.94) -> np.ndarray:
    """RiskMetrics covariance. `returns` is T x N, treated as mean-zero.

    Weights are `(1-λ) λ^{T-1-t}`, renormalised to sum to 1 so a short sample
    does not leak a zero-initialized bias into the matrix.
    """
    R = np.asarray(returns, dtype=float)
    if R.ndim != 2 or R.shape[0] < 2 or R.shape[1] < 1:
        raise ValueError("ewma_cov needs a T x N return matrix with T >= 2.")
    lam = min(max(float(lam), 1e-6), 0.999999)
    T = R.shape[0]
    idx = np.arange(T, dtype=float)
    weights = (1.0 - lam) * lam ** (T - 1.0 - idx)
    weights = weights / weights.sum()
    return (R * weights[:, None]).T @ R


def ledoit_wolf_cov(returns: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Analytical Ledoit-Wolf shrinkage toward `μ I`. Returns (cov, shrinkage, μ).

    Centered sample covariance, intensity clipped to [0, 1]. No sklearn.
    """
    X = np.asarray(returns, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("ledoit_wolf_cov needs a T x N return matrix with T >= 2.")
    n_samples, n_features = X.shape
    X = X - X.mean(axis=0)
    emp = (X.T @ X) / n_samples
    mu = float(np.trace(emp) / n_features)
    delta_m = emp.copy()
    delta_m.flat[:: n_features + 1] -= mu
    delta = float((delta_m**2).sum() / n_features)
    x2 = X**2
    beta_ = float((1.0 / n_features) * np.sum((x2.T @ x2) / n_samples - emp**2))
    beta = min(beta_, delta)
    shrinkage = 0.0 if delta == 0 else min(1.0, max(0.0, beta / delta))
    shrunk = shrinkage * mu * np.eye(n_features) + (1.0 - shrinkage) * emp
    return shrunk, shrinkage, mu


def _corr_from_cov(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vol = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    corr = cov / np.outer(vol, vol)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr, vol


def denoise_cov(cov: np.ndarray, n_obs: int) -> dict[str, Any]:
    """Marchenko-Pastur clip on the correlation spectrum, mapped back to cov.

    `q = n_assets / n_obs`. Eigenvalues at or below `λ+ = (1 + sqrt(q))²` are
    replaced by their average. The diagonal of the reconstructed correlation
    is renormalised to 1 so the result stays a covariance.
    """
    C = np.asarray(cov, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("denoise_cov needs a square covariance.")
    n = C.shape[0]
    n_obs = max(int(n_obs), 1)
    q = n / float(n_obs)
    lambda_plus = (1.0 + math.sqrt(q)) ** 2
    corr, vol = _corr_from_cov(C)
    evals, evecs = np.linalg.eigh(corr)
    evals = np.clip(evals, 0.0, None)
    noise = evals <= lambda_plus + 1e-12
    n_signal = int((~noise).sum())
    cleaned = evals.copy()
    if noise.any():
        cleaned[noise] = float(cleaned[noise].mean())
    corr_d = evecs @ np.diag(cleaned) @ evecs.T
    d = np.sqrt(np.clip(np.diag(corr_d), 1e-18, None))
    corr_d = corr_d / np.outer(d, d)
    np.fill_diagonal(corr_d, 1.0)
    cov_d = corr_d * np.outer(vol, vol)
    return {
        "cov": cov_d,
        "n_signal_eigenvalues": n_signal,
        "n_noise_eigenvalues": int(noise.sum()),
        "lambda_plus": round(float(lambda_plus), _RND),
        "q": round(float(q), _RND),
        "n_assets": n,
        "n_obs": n_obs,
    }


def detone_cov(cov: np.ndarray) -> np.ndarray:
    """Strip the largest correlation eigenmode (the market), keep residual cov."""
    C = np.asarray(cov, dtype=float)
    corr, vol = _corr_from_cov(C)
    evals, evecs = np.linalg.eigh(corr)
    i = int(np.argmax(evals))
    market = evals[i] * np.outer(evecs[:, i], evecs[:, i])
    residual = corr - market
    d = np.sqrt(np.clip(np.diag(residual), 1e-18, None))
    residual = residual / np.outer(d, d)
    np.fill_diagonal(residual, 1.0)
    return residual * np.outer(vol, vol)


def variance_explained(cov: np.ndarray, *, cap: int = 32) -> list[dict[str, Any]]:
    """Leading eigenvalue shares. A PCA sketch, not the matrix."""
    C = np.asarray(cov, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1] or C.shape[0] < 1:
        return []
    evals = np.clip(np.linalg.eigvalsh(C), 0.0, None)[::-1]
    total = float(evals.sum())
    if total <= 0:
        return []
    n = min(int(cap), int(evals.size))
    return [{"k": i + 1, "share": round(float(evals[i] / total), _RND)} for i in range(n)]


def cov_diagnostics(cov: np.ndarray) -> dict[str, Any]:
    """Figures a covariance is allowed to emit. The matrix itself stays local."""
    C = np.asarray(cov, dtype=float)
    n = int(C.shape[0])
    evals = np.clip(np.linalg.eigvalsh(C), 0.0, None)
    cond = None
    if evals[-1] > 0:
        smallest = float(evals[evals > 1e-18].min()) if np.any(evals > 1e-18) else 0.0
        cond = round(float(evals[-1] / smallest), _RND) if smallest > 0 else None
    return {
        "n_assets": n,
        "trace": round(float(np.trace(C)), _RND),
        "condition_number": cond,
        "min_eigenvalue": round(float(evals[0]), _RND),
        "max_eigenvalue": round(float(evals[-1]), _RND),
    }


def triangle(cov: np.ndarray, names: list[str], *, cap: int = 512) -> list[dict[str, Any]] | None:
    """Upper-triangle entries if they fit the figure cap, else None."""
    n = cov.shape[0]
    n_pairs = n * (n + 1) // 2
    if n_pairs > cap:
        return None
    out: list[dict[str, Any]] = []
    for i in range(n):
        for j in range(i, n):
            out.append({"a": names[i], "b": names[j], "v": round(float(cov[i, j]), _RND)})
    return out
