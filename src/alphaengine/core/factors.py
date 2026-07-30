"""
Risk, factor decomposition over supplied portfolio + factor return streams.

Lifted (math-identical) from backend/quant/factors.compute_multi_factor_loadings,
with the FRED rfr fetch and quant.limits import removed: the caller supplies the
risk-free rate (defaults to 0.04 annual) and the VIF threshold is inlined. OLS
with HAC standard errors (statsmodels) gives factor betas, alpha, t-stats, R²,
and a multicollinearity diagnostic (VIF).

Pure numpy/statsmodels. Deterministic given inputs on the pinned stack.
"""

from __future__ import annotations

try:  # noqa: SIM105
    import statsmodels  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "factors needs statsmodels, which is not a core dependency of "
        "alphaengine. Install it with:  pip install 'alphaengine[factors]'  "
        "The core (deflated Sharpe, PBO, CPCV, performance, risk) needs "
        "only numpy and scipy and is unaffected."
    ) from exc

import math

import numpy as np
import statsmodels.api as sm

VIF_MAX_THRESHOLD = 10.0
_DEFAULT_RFR = 0.04


def _clean(val):
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val


def _compute_vif(X: np.ndarray, factor_names: list[str]) -> dict[str, float]:
    """VIF_j = 1/(1 - R_j²), R_j² from regressing factor j on the others.

    Several proxies are constructed as `X - SPY`, so they can share variance
    with "market"; VIF surfaces whether loadings are individually identified.
    """
    n_features = X.shape[1]
    vifs: dict[str, float] = {}
    for j in range(n_features):
        y_j = X[:, j]
        X_others = np.delete(X, j, axis=1)
        if X_others.shape[1] == 0:
            vifs[factor_names[j]] = 1.0
            continue
        try:
            X_const = np.column_stack([np.ones(len(y_j)), X_others])
            betas, *_ = np.linalg.lstsq(X_const, y_j, rcond=None)
            y_hat = X_const @ betas
            ss_res = float(np.sum((y_j - y_hat) ** 2))
            ss_tot = float(np.sum((y_j - y_j.mean()) ** 2))
            r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            vif = 1.0 / max(1e-9, 1.0 - r_sq) if r_sq < 0.9999 else float("inf")
            vifs[factor_names[j]] = vif if math.isfinite(vif) else float("inf")
        except Exception:
            vifs[factor_names[j]] = float("nan")
    return vifs


def _model_label(factor_names: list[str]) -> str:
    core_set = set(factor_names)
    if {"market", "size", "value", "profitability", "low_vol", "momentum"}.issubset(core_set):
        return "FF5-style + Low-Vol + Momentum"
    if {"market", "size", "value", "momentum"}.issubset(core_set):
        return "Carhart 4-factor"
    return f"{len(factor_names)}-factor"


def decompose_factors(
    portfolio_returns: list[float],
    factor_returns: dict[str, list[float]],
    *,
    risk_free_rate: float | None = None,
) -> dict:
    """Multi-factor regression (FF5 + Momentum style) over supplied returns.

    `factor_returns` = {"market": [...], "size": [...]...}. Returns alpha
    (annualized %), factor betas + t-stats, R²/adj-R², residual vol, and a VIF
    multicollinearity diagnostic. Excess returns use the supplied rfr (annual),
    defaulting to 4%.
    """
    factor_names = list(factor_returns.keys())
    if not factor_names:
        return {"error": "No factor data"}

    min_len = min(len(portfolio_returns), *[len(v) for v in factor_returns.values()])
    if min_len < 30:
        return {"error": "Need 30+ observations"}

    rfr = float(risk_free_rate) if risk_free_rate is not None else _DEFAULT_RFR
    y = np.array(portfolio_returns[-min_len:], dtype=float)
    rf_daily = rfr / 252
    y_excess = y - rf_daily

    X = np.column_stack([np.array(factor_returns[f][-min_len:], dtype=float) for f in factor_names])

    vifs: dict[str, float] = {}
    high_vif: list[str] = []
    if X.shape[1] >= 2:
        vifs = _compute_vif(X, factor_names)
        high_vif = [f for f, v in vifs.items() if math.isfinite(v) and v > VIF_MAX_THRESHOLD]

    X_const = sm.add_constant(X)
    model = sm.OLS(y_excess, X_const).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    betas = {}
    tstats = {}
    for i, name in enumerate(factor_names):
        betas[name] = _clean(round(float(model.params[i + 1]), 4))
        tstats[name] = _clean(round(float(model.tvalues[i + 1]), 2))

    alpha_pvalue = float(model.pvalues[0])
    return {
        "alpha": _clean(round(float(model.params[0] * 252 * 100), 2)),
        "alpha_tstat": _clean(round(float(model.tvalues[0]), 2)),
        "alpha_pvalue": _clean(round(alpha_pvalue, 4)),
        "alpha_significant_at_5pct": bool(alpha_pvalue < 0.05),
        "factor_betas": betas,
        "factor_tstats": tstats,
        "r_squared": _clean(round(float(model.rsquared), 3)),
        "adj_r_squared": _clean(round(float(model.rsquared_adj), 3)),
        "residual_vol": _clean(round(float(np.std(model.resid) * np.sqrt(252) * 100), 2)),
        "n_observations": int(min_len),
        "model": _model_label(factor_names),
        "vifs": {k: _clean(round(v, 2)) if math.isfinite(v) else None for k, v in vifs.items()},
        "high_vif_factors": high_vif,
        "multicollinearity_flag": bool(high_vif),
    }
