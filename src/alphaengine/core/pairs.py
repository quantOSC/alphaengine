"""
Signals, pair-trade analytics over supplied price series.

Lifted (math-identical) from backend/quant/pairs.py, with the data-fetch removed:
the caller supplies aligned price arrays; nothing is fetched. The four
primitives a PM checks before sizing a pair, TLS hedge ratio, Engle-Granger
cointegration (ADF), Ornstein-Uhlenbeck half-life, rolling-correlation
stability, and a discrete trade signal.

Public (beta cut):
  compute_spread_signal(a_closes, b_closes...) -> dict   # one pair, end to end
  find_cointegrated_pairs(prices...) -> dict             # screen many series

Pure numpy/scipy/statsmodels. Deterministic given inputs on the pinned stack.
"""

from __future__ import annotations

try:  # noqa: SIM105
    import statsmodels  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "pairs needs statsmodels, which is not a core dependency of "
        "alphaengine. Install it with:  pip install 'alphaengine[factors]'  "
        "The core (deflated Sharpe, PBO, CPCV, performance, risk) needs "
        "only numpy and scipy and is unaffected."
    ) from exc

import itertools
import math
from typing import Any

import numpy as np
from statsmodels.tsa.stattools import adfuller

# Pair-trading thresholds (documented so a caller can audit / override).
MIN_OBSERVATIONS = 126
COINTEGRATION_P_THRESHOLD = 0.05
MIN_HALF_LIFE_DAYS = 1.0
MAX_HALF_LIFE_DAYS = 60.0
MIN_STABILITY = 0.5
# Below this |rolling correlation| the legs barely co-move: a stationary spread
# here is reverting on idiosyncratic noise, not a structural relationship, the
# classic multiple-testing mirage. Cointegration still passes (ADF/half-life/
# stability), but the pair is flagged low structural quality.
MIN_COMOVEMENT_CORR = 0.5
ZSCORE_WINDOW = 60
STABILITY_WINDOW = 60
ENTRY_ZSCORE = 2.0

# ── Redundant-leg guard (share-class twins / fungible dual-listings) ─────────
# Two tickers for the SAME economic claim (GOOGL/GOOG, BRK.A/BRK.B) are ~1.0
# correlated, so they pass ADF + half-life + stability perfectly and top every
# cointegration screen, but their spread has no tradeable edge (it is the same
# bet minus a tiny structural basis). Exclude them: a pair needs two DISTINCT
# underlyings. Detected two ways, a curated same-issuer set, and a hard ceiling
# on mean rolling co-movement (distinct names rarely sustain |corr| >= 0.99).
MAX_COMOVEMENT_CORR = 0.99

_SAME_ISSUER_TWINS = frozenset(
    {
        frozenset({"GOOGL", "GOOG"}),
        frozenset({"BRK.A", "BRK.B"}),
        frozenset({"BRK-A", "BRK-B"}),
        frozenset({"BRKA", "BRKB"}),
        frozenset({"FOX", "FOXA"}),
        frozenset({"NWS", "NWSA"}),
        frozenset({"UA", "UAA"}),
        frozenset({"PARA", "PARAA"}),
        frozenset({"LEN", "LEN.B"}),
        frozenset({"LEN", "LEN-B"}),
        frozenset({"HEI", "HEI.A"}),
        frozenset({"HEI", "HEI-A"}),
        frozenset({"LBRDA", "LBRDK"}),
        frozenset({"CWEN", "CWEN.A"}),
        frozenset({"CWEN", "CWEN-A"}),
        frozenset({"MOG.A", "MOG.B"}),
        frozenset({"MOG-A", "MOG-B"}),
        frozenset({"GEF", "GEF.B"}),
        frozenset({"GEF", "GEF-B"}),
        frozenset({"CRD.A", "CRD.B"}),
        frozenset({"CRD-A", "CRD-B"}),
    }
)


def _redundant_pair_reason(symbol_a, symbol_b, mean_corr) -> str | None:
    """Return WHY a pair is redundant (same economic claim), else None.
    A redundant pair is not a tradeable spread and must be excluded from the
    screen even though its cointegration stats look ~perfect."""
    a = str(symbol_a or "").upper().strip()
    b = str(symbol_b or "").upper().strip()
    if a and b and frozenset({a, b}) in _SAME_ISSUER_TWINS:
        return "Same issuer / share-class twin, not a tradeable spread (same economic claim)"
    if mean_corr is not None and abs(mean_corr) >= MAX_COMOVEMENT_CORR:
        return (
            f"Near-identical co-movement (|corr|={abs(mean_corr):.3f} >= {MAX_COMOVEMENT_CORR}), "
            f"legs are effectively the same instrument, no tradeable spread"
        )
    return None


def _clean(val: Any) -> Any:
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val


def _series_to_list(s) -> list[float]:
    """A series is either a list of closes or a {date: close} dict (sorted by date)."""
    if isinstance(s, dict):
        return [float(s[k]) for k in sorted(s.keys())]
    return [float(x) for x in (s or []) if x is not None]


def _align_pair(a, b) -> tuple[list[float], list[float]]:
    """Align two price series. If both are dated dicts, align on the intersection
    of dates (the correct alignment for unaligned calendars). Otherwise treat as
    index-aligned lists and truncate to the common length from the most recent end."""
    if isinstance(a, dict) and isinstance(b, dict):
        common = sorted(set(a.keys()) & set(b.keys()))
        return [float(a[d]) for d in common], [float(b[d]) for d in common]
    al, bl = _series_to_list(a), _series_to_list(b)
    n = min(len(al), len(bl))
    return al[-n:], bl[-n:]


def _tls_hedge_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """Total Least Squares slope of `a ~ β·b` via SVD of the centered design.

    OLS biases β toward zero when b has noise (attenuation); TLS is the
    unbiased estimator when both legs have measurement noise (always true for
    two market prices).
    """
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    a_c = a - float(np.mean(a))
    b_c = b - float(np.mean(b))
    if np.std(b_c) < 1e-12:
        return float("nan")
    M = np.column_stack([b_c, a_c])
    try:
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return float("nan")
    v = Vt[-1]
    if abs(v[1]) < 1e-12:
        return float("nan")
    return float(-v[0] / v[1])


def _ou_half_life(spread: np.ndarray) -> float | None:
    """Mean-reversion half-life via AR(1) on the differenced spread.

    Δs_t = a + b·s_{t-1} + ε_t ; half_life = -ln(2)/b. None when b≥0.
    """
    spread = np.asarray(spread, dtype=float)
    spread = spread[np.isfinite(spread)]
    n = len(spread)
    if n < 30:
        return None
    s_lag = spread[:-1]
    s_diff = np.diff(spread)
    X = np.column_stack([np.ones(len(s_lag)), s_lag])
    try:
        coefs, *_ = np.linalg.lstsq(X, s_diff, rcond=None)
    except np.linalg.LinAlgError:
        return None
    b = float(coefs[1])
    if b >= 0 or not math.isfinite(b):
        return None
    half_life = -math.log(2.0) / b
    if not math.isfinite(half_life) or half_life <= 0:
        return None
    return half_life


def _rolling_correlation_stability(
    a_returns: np.ndarray, b_returns: np.ndarray, window: int = STABILITY_WINDOW
) -> dict:
    """Stability = 1 - std(rolling correlation). High = structurally consistent."""
    n = min(len(a_returns), len(b_returns))
    if n < window + 5:
        return {"stability": None, "mean_corr": None, "std_corr": None, "n_windows": 0}

    rolling: list[float] = []
    for i in range(window, n):
        chunk_a = a_returns[i - window : i]
        chunk_b = b_returns[i - window : i]
        if np.std(chunk_a) > 0 and np.std(chunk_b) > 0:
            c = float(np.corrcoef(chunk_a, chunk_b)[0, 1])
            if math.isfinite(c):
                rolling.append(c)

    if len(rolling) < 5:
        return {"stability": None, "mean_corr": None, "std_corr": None, "n_windows": 0}

    arr = np.array(rolling)
    mean_c = float(np.mean(arr))
    std_c = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    stability = max(0.0, min(1.0, 1.0 - std_c))
    return {"stability": stability, "mean_corr": mean_c, "std_corr": std_c, "n_windows": len(rolling)}


def compute_spread(a_closes: np.ndarray, b_closes: np.ndarray, hedge_ratio: float) -> np.ndarray:
    """Log-price spread: s_t = log(P_a) - β·log(P_b)."""
    a = np.asarray(a_closes, dtype=float)
    b = np.asarray(b_closes, dtype=float)
    return np.log(a) - hedge_ratio * np.log(b)


def engle_granger_test(spread: np.ndarray) -> dict:
    """ADF on the spread. Null: unit root (NOT cointegrated). Reject at p<0.05."""
    spread_clean = np.asarray(spread, dtype=float)
    spread_clean = spread_clean[np.isfinite(spread_clean)]
    if len(spread_clean) < 30:
        return {
            "p_value": None,
            "test_statistic": None,
            "method": "engle_granger",
            "error": "insufficient_data",
        }
    try:
        result = adfuller(spread_clean, regression="c", autolag="AIC")
        return {
            "p_value": float(result[1]),
            "test_statistic": float(result[0]),
            "critical_values": {k: float(v) for k, v in result[4].items()},
            "n_lags": int(result[2]),
            "method": "engle_granger",
        }
    except Exception as e:  # noqa: BLE001
        return {"p_value": None, "test_statistic": None, "method": "engle_granger", "error": str(e)}


def compute_spread_signal(
    a_closes,
    b_closes,
    *,
    symbol_a: str = "A",
    symbol_b: str = "B",
    zscore_window: int = ZSCORE_WINDOW,
    stability_window: int = STABILITY_WINDOW,
    significance: float = COINTEGRATION_P_THRESHOLD,
    max_half_life: float = MAX_HALF_LIFE_DAYS,
) -> dict:
    """End-to-end pair analysis over two supplied, index-aligned close series.

    Identical math to backend analyze_pair, minus the fetch: hedge ratio (TLS),
    cointegration p-value (Engle-Granger ADF), spread z-score, OU half-life,
    rolling-correlation stability, and a discrete trade signal. A pair is
    `cointegrated=True` only when ADF, half-life, and stability all pass.
    """
    # Accept lists or dated {date: close} dicts; align on common dates if dated.
    a_list, b_list = _align_pair(a_closes, b_closes)
    n = len(a_list)

    if symbol_a == symbol_b:
        return {
            "ticker_a": symbol_a,
            "ticker_b": symbol_b,
            "error": "Same ticker for both legs",
            "cointegrated": False,
        }
    if n < MIN_OBSERVATIONS:
        return {
            "ticker_a": symbol_a,
            "ticker_b": symbol_b,
            "n_observations": n,
            "error": f"Insufficient overlap: {n} obs (need {MIN_OBSERVATIONS}+)",
            "cointegrated": False,
        }

    a = np.array(a_list, dtype=float)
    b = np.array(b_list, dtype=float)
    if (a <= 0).any() or (b <= 0).any():
        return {
            "ticker_a": symbol_a,
            "ticker_b": symbol_b,
            "n_observations": n,
            "error": "Non-positive prices found (cannot take log)",
            "cointegrated": False,
        }
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return {
            "ticker_a": symbol_a,
            "ticker_b": symbol_b,
            "n_observations": n,
            "error": "Degenerate: constant price series",
            "cointegrated": False,
        }

    log_a = np.log(a)
    log_b = np.log(b)
    hedge_ratio = _tls_hedge_ratio(log_a, log_b)
    if not math.isfinite(hedge_ratio) or abs(hedge_ratio) < 1e-9:
        return {
            "ticker_a": symbol_a,
            "ticker_b": symbol_b,
            "n_observations": n,
            "error": "Hedge ratio degenerate",
            "cointegrated": False,
        }

    spread = compute_spread(a, b, hedge_ratio)
    if len(spread) < max(zscore_window, 30):
        return {
            "ticker_a": symbol_a,
            "ticker_b": symbol_b,
            "n_observations": n,
            "error": "Insufficient spread observations after alignment",
            "cointegrated": False,
        }

    eg = engle_granger_test(spread)
    p_value = eg.get("p_value")

    recent_spread = spread[-zscore_window:]
    mu = float(np.mean(recent_spread))
    sigma = float(np.std(recent_spread, ddof=1)) if len(recent_spread) > 1 else 0.0
    current_z: float | None
    if sigma > 1e-9 and math.isfinite(spread[-1]):
        current_z = float((spread[-1] - mu) / sigma)
    else:
        current_z = None

    half_life = _ou_half_life(spread)
    a_returns = np.diff(log_a)
    b_returns = np.diff(log_b)
    stability_info = _rolling_correlation_stability(a_returns, b_returns, window=stability_window)

    reasons: list[str] = []
    p_ok = p_value is not None and p_value < significance
    hl_ok = half_life is not None and MIN_HALF_LIFE_DAYS < half_life < max_half_life
    stab = stability_info.get("stability")
    stab_ok = stab is not None and stab > MIN_STABILITY

    if not p_ok:
        reasons.append(
            "ADF test unavailable"
            if p_value is None
            else f"Failed cointegration (ADF p={p_value:.3f} ≥ {significance})"
        )
    if not hl_ok:
        if half_life is None:
            reasons.append("Spread not mean-reverting (AR(1) coefficient ≥ 0)")
        elif half_life <= MIN_HALF_LIFE_DAYS:
            reasons.append(f"Half-life too short ({half_life:.1f}d)")
        else:
            reasons.append(f"Half-life too long ({half_life:.1f}d > {max_half_life:.0f}d)")
    if not stab_ok:
        reasons.append(
            "Stability test insufficient data"
            if stab is None
            else f"Unstable correlation (stability={stab:.2f} ≤ {MIN_STABILITY:.2f})"
        )

    cointegrated = bool(p_ok and hl_ok and stab_ok)
    if cointegrated and not reasons:
        reasons.append("Cointegrated, mean-reverting in tradable window, stable correlation")

    # Co-movement quality. Cointegration measures a STATIONARY spread; it can pass
    # when the legs barely move together (a stable-but-near-zero rolling
    # correlation), in which case the reversion is idiosyncratic noise, the
    # multiple-testing mirage. Flag it so a cointegrated hit isn't mistaken for a
    # structural pair. `cointegrated` is unchanged (ADF/half-life/stability); this
    # is an honesty signal layered on top.
    mean_corr = stability_info.get("mean_corr")
    low_comovement = mean_corr is not None and abs(mean_corr) < MIN_COMOVEMENT_CORR
    structural_quality = "unknown" if mean_corr is None else ("low" if low_comovement else "high")
    if cointegrated and low_comovement:
        reasons.append(
            f"Low leg co-movement (|corr|={abs(mean_corr):.2f} < {MIN_COMOVEMENT_CORR}), "
            f"likely spurious / reversion on idiosyncratic noise"
        )

    # Redundant legs (share-class twins / near-1.0 co-movement): the same economic
    # claim, not a spread. Overrides the cointegration verdict, which for a twin is
    # ~perfect and would otherwise top the screen. Exclude it from tradeable pairs.
    redundant_reason = _redundant_pair_reason(symbol_a, symbol_b, mean_corr)
    redundant = redundant_reason is not None
    if redundant:
        structural_quality = "redundant"
        reasons.append(redundant_reason)
        cointegrated = False

    trade_signal = "hold"
    if cointegrated and current_z is not None:
        if current_z > ENTRY_ZSCORE:
            trade_signal = "short_spread"
        elif current_z < -ENTRY_ZSCORE:
            trade_signal = "long_spread"

    share_ratio_at_close: float | None = None
    if math.isfinite(hedge_ratio) and b[-1] > 0:
        share_ratio_at_close = float(hedge_ratio * a[-1] / b[-1])

    return {
        "ticker_a": symbol_a,
        "ticker_b": symbol_b,
        "n_observations": n,
        "hedge_ratio": round(float(hedge_ratio), 4),
        "hedge_ratio_method": "total_least_squares_log_prices",
        "share_ratio_at_close": _clean(round(share_ratio_at_close, 4))
        if share_ratio_at_close is not None
        else None,
        "cointegration": {
            "p_value": _clean(round(p_value, 4)) if p_value is not None else None,
            "test_statistic": (
                _clean(round(eg.get("test_statistic", float("nan")), 3))
                if eg.get("test_statistic") is not None
                else None
            ),
            "critical_values": eg.get("critical_values"),
            "n_lags": eg.get("n_lags"),
            "method": eg.get("method"),
            "significant_at_5pct": bool(p_ok),
        },
        "half_life_days": _clean(round(half_life, 2)) if half_life is not None else None,
        "spread": {
            "current_value": _clean(round(float(spread[-1]), 6)),
            "rolling_mean": round(mu, 6),
            "rolling_std": round(sigma, 6),
            "current_zscore": _clean(round(current_z, 3)) if current_z is not None else None,
            "window": zscore_window,
        },
        "stability": {
            "rolling_correlation_mean": (
                _clean(round(stability_info["mean_corr"], 3))
                if stability_info["mean_corr"] is not None
                else None
            ),
            "rolling_correlation_std": (
                _clean(round(stability_info["std_corr"], 3))
                if stability_info["std_corr"] is not None
                else None
            ),
            "stability_score": (_clean(round(stab, 3)) if stab is not None else None),
            "n_windows": stability_info["n_windows"],
        },
        "cointegrated": cointegrated,
        "structural_quality": structural_quality,  # high | low | unknown | redundant
        "low_comovement": low_comovement,  # True => likely spurious
        "redundant": redundant,  # True => share-class twin / same instrument
        "trade_signal": trade_signal,
        "reasons": reasons,
    }


def find_cointegrated_pairs(
    prices: dict,
    *,
    candidates: list[tuple[str, str]] | None = None,
    zscore_window: int = ZSCORE_WINDOW,
    stability_window: int = STABILITY_WINDOW,
    cointegrated_only: bool = True,
    significance: float = COINTEGRATION_P_THRESHOLD,
    max_half_life: float = MAX_HALF_LIFE_DAYS,
) -> dict:
    """Screen a universe of supplied price series for cointegrated pairs.

    `prices`: {symbol: [close...]}, series are assumed to share a trading
    calendar; each pair is aligned to its common length from the most recent
    end. `candidates`: optional explicit pair list; default is all unique
    unordered pairs. Returns pairs sorted by ADF p-value ascending.
    """
    symbols = list(prices.keys())
    pairs = candidates if candidates is not None else list(itertools.combinations(symbols, 2))

    results: list[dict] = []
    for sa, sb in pairs:
        if sa not in prices or sb not in prices:
            continue
        res = compute_spread_signal(
            prices[sa],
            prices[sb],
            symbol_a=sa,
            symbol_b=sb,
            zscore_window=zscore_window,
            stability_window=stability_window,
            significance=significance,
            max_half_life=max_half_life,
        )
        results.append(res)

    def _p(r: dict):
        p = (r.get("cointegration") or {}).get("p_value")
        return p if p is not None else 1.0

    results.sort(key=_p)
    coint = [r for r in results if r.get("cointegrated")]
    selected = coint if cointegrated_only else results
    n_low = sum(1 for r in coint if r.get("low_comovement"))
    n_eval = len(results)
    exp_fp = round(significance * n_eval, 1)

    return {
        "n_evaluated": n_eval,
        "n_cointegrated": len(coint),
        # Honesty layer (the multiple-testing tax made explicit). Screening N pairs
        # at p<sig yields ~sig*N cointegrated hits by chance alone; low-comovement
        # hits are the likely-spurious ones. The host MUST deflate survivors by the
        # number of pairs searched, not 1.
        "n_low_comovement": n_low,
        "expected_false_positives": exp_fp,
        "n_trials_recommended": n_eval,
        "multiple_testing_note": (
            f"Screened {n_eval} pairs at p<{significance}; ~{exp_fp} cointegrated hits are expected "
            f"by chance alone. {n_low} of {len(coint)} cointegrated pairs have near-zero rolling "
            f"correlation (low_comovement=true, likely spurious). Validate survivors with "
            f"deflated_sharpe at n_trials={n_eval} and pbo_cscv before trading; do not deflate at n_trials=1."
        ),
        "pairs": selected,
    }
