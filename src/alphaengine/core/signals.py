"""
signals.py, does a SUPPLIED signal carry information about supplied prices.

Pure deterministic math (no fetch, no LLM, no randomness). Both panels are
caller-supplied and never leave the process: this module learns what a signal
says and what a price did from its arguments and nowhere else.

WHY THIS EXISTS SEPARATELY FROM `validation.py`
    A deflated Sharpe judges a STRATEGY — something with a simulator behind it.
    Most of a modeling week happens long before a simulator exists: a quant has
    a factor and asks whether it predicts anything at all. That question is
    answered with an information coefficient, a quantile spread and a decay
    profile, and making somebody build a backtest to ask it is why the question
    goes unasked.

THE ALIGNMENT IS BY COMMON TAIL, AND THAT IS STATED RATHER THAN HIDDEN
    Series arrive as {symbol: values} with no dates required. Each symbol's
    signal and price series are aligned on their shared tail, and names whose
    series are too short for the horizon are COUNTED AND NAMED, never silently
    dropped — a panel of 40 names is a different measurement from a panel of
    400 with 360 discarded.

NON-OVERLAPPING PERIODS, DELIBERATELY
    Evaluation dates step by the horizon. Overlapping forward windows make
    neighbouring ICs share most of their returns, which inflates the t-stat in
    exactly the direction that flatters — the same failure `n_trials = 1` was.
    Fewer honest periods beat many correlated ones.

THE FIGURES DO NOT JUDGE
    An implausibly high IC is the signature of lookahead — a signal that
    already contains the close it is scored against — but the threshold at
    which a run refuses is the workflow's judgment, held server-side. This
    module measures and reports; it never decides.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .panel import newey_west_tstat
from .series_shapes import series_values

#: The longest per-period sequence a reading reports, and the longest bounded
#: name list beside it.
#:
#: RAISED 64 -> 512 on 2026-08-08 with the wire cap. This one is easy to miss and
#: silently defeats the others: the core truncates `ic_by_period` HERE, before
#: `StepExecutor` ever buckets it, so leaving this at 64 would have held IC
#: through time at 64 points no matter what `IC_POINTS` said.
MAX_PERIODS = 512
_RND = 6

DEFAULT_HORIZON = 21
DEFAULT_QUANTILES = 5
#: The fewest names a cross-section can rank. Below this a rank correlation is
#: mostly noise about ties.
MIN_NAMES = 8


class PanelShapeError(ValueError):
    """The signal or price panel is not a mapping of symbol -> numeric series."""


def _series(values: Any) -> np.ndarray | None:
    """A numeric 1-D array, or None for anything else. Narrow on purpose —
    guessing at a frame-like object risks reading the wrong column silently.

    A `{date: value}` mapping and `[{date, close}]` rows go through the shared
    reader — the shapes a dated universe arrives in, and the prices side of a
    signal evaluation is exactly one of those. They used to read as None here,
    so a dated universe scored zero usable names while profiling clean."""
    if isinstance(values, dict) or (
        isinstance(values, (list, tuple)) and len(values) and isinstance(values[0], dict)
    ):
        arr = series_values(values)[0]
        return arr if arr.ndim == 1 and arr.size else None
    if isinstance(values, (list, tuple, np.ndarray)):
        try:
            arr = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            return None
        return arr if arr.ndim == 1 and arr.size else None
    return None


def _tail_panel(
    signal: dict, prices: dict, *, need: int
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Symbols present in BOTH panels, tail-aligned to a common length.

    Returns (S, P, names, skipped): S and P are [T x N] with T the shortest
    usable shared tail, and `skipped` names every symbol that could not be
    used — absent from one panel, non-numeric, or shorter than `need`.
    """
    if not isinstance(signal, dict) or not isinstance(prices, dict):
        raise PanelShapeError(
            "signal and prices are each {symbol: values} — a mapping per panel, the same symbols in both."
        )
    usable: list[tuple[str, np.ndarray, np.ndarray]] = []
    skipped: list[str] = []
    for name in sorted(set(map(str, signal)) | set(map(str, prices))):
        s = _series(signal.get(name))
        p = _series(prices.get(name))
        if s is None or p is None or len(s) < need or len(p) < need:
            skipped.append(name)
            continue
        usable.append((name, s, p))
    if not usable:
        return np.empty((0, 0)), np.empty((0, 0)), [], skipped

    depth = min(min(len(s), len(p)) for _, s, p in usable)
    names = [name for name, _, _ in usable]
    S = np.column_stack([s[-depth:] for _, s, _ in usable])
    P = np.column_stack([p[-depth:] for _, _, p in usable])
    return S, P, names, skipped


def _rank(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _cross_ic(signal_row: np.ndarray, fwd_row: np.ndarray) -> float | None:
    """Spearman rank correlation of one cross-section, or None when degenerate."""
    ok = np.isfinite(signal_row) & np.isfinite(fwd_row)
    if int(ok.sum()) < MIN_NAMES:
        return None
    s, f = signal_row[ok], fwd_row[ok]
    if np.ptp(s) == 0 or np.ptp(f) == 0:
        # A constant column has no ranking to correlate. None, not zero: zero
        # is a measurement and this is the absence of one.
        return None
    rs, rf = _rank(s), _rank(f)
    denom = float(np.std(rs) * np.std(rf))
    if denom == 0:
        return None
    return float(np.mean((rs - rs.mean()) * (rf - rf.mean())) / denom)


def _forward(P: np.ndarray, t: int, horizon: int) -> np.ndarray:
    start, end = P[t], P[t + horizon]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(start != 0, end / start - 1.0, np.nan)
    return out


def _cross_pearson(signal_row: np.ndarray, fwd_row: np.ndarray) -> float | None:
    """Pearson correlation of one cross-section, or None when degenerate."""
    ok = np.isfinite(signal_row) & np.isfinite(fwd_row)
    if int(ok.sum()) < MIN_NAMES:
        return None
    s, f = signal_row[ok], fwd_row[ok]
    if np.ptp(s) == 0 or np.ptp(f) == 0:
        return None
    denom = float(np.std(s) * np.std(f))
    if denom == 0:
        return None
    return float(np.mean((s - s.mean()) * (f - f.mean())) / denom)


def _period_ics(S: np.ndarray, P: np.ndarray, horizon: int) -> list[float]:
    """One IC per NON-OVERLAPPING horizon window, oldest first."""
    depth = S.shape[0]
    ics: list[float] = []
    for t in range(0, depth - horizon, horizon):
        ic = _cross_ic(S[t], _forward(P, t, horizon))
        if ic is not None:
            ics.append(ic)
    return ics


def _period_ics_pearson(S: np.ndarray, P: np.ndarray, horizon: int) -> list[float]:
    depth = S.shape[0]
    ics: list[float] = []
    for t in range(0, depth - horizon, horizon):
        ic = _cross_pearson(S[t], _forward(P, t, horizon))
        if ic is not None:
            ics.append(ic)
    return ics


def information_coefficient(signal: dict, prices: dict, *, horizon: int = DEFAULT_HORIZON) -> dict:
    """Does the signal's cross-sectional ranking predict forward returns?

    Args:
        signal: {symbol: values}, one signal reading per period per name.
        prices: {symbol: closes}, the same symbols.
        horizon: forward-return window in periods. Evaluation steps by this, so
            the ICs are non-overlapping.

    Returns:
        mean_ic, ic_t_stat, n_periods, ic_by_period (capped at MAX_PERIODS,
        most recent kept), n_names, n_skipped, skipped (named, bounded), and
        the horizon — everything a reader needs to know what the number is.
        `mean_ic` is None when nothing could be measured, never 0.
    """
    horizon = max(1, int(horizon))
    S, P, names, skipped = _tail_panel(signal, prices, need=horizon + 1)
    ics = _period_ics(S, P, horizon) if names else []

    mean_ic: float | None = None
    t_stat: float | None = None
    if ics:
        arr = np.asarray(ics)
        mean_ic = round(float(arr.mean()), _RND)
        if len(ics) > 1 and float(arr.std(ddof=1)) > 0:
            t_stat = round(float(arr.mean() / (arr.std(ddof=1) / np.sqrt(len(ics)))), _RND)

    return {
        "mean_ic": mean_ic,
        "ic_t_stat": t_stat,
        "n_periods": len(ics),
        "ic_by_period": [round(v, _RND) for v in ics[-MAX_PERIODS:]],
        "n_names": len(names),
        "n_skipped": len(skipped),
        "skipped": skipped[:MAX_PERIODS],
        "horizon": horizon,
    }


def signal_icir(
    signal: dict,
    prices: dict,
    *,
    horizon: int = DEFAULT_HORIZON,
    method: str = "spearman",
) -> dict:
    """Information coefficient, its ratio, and a Newey-West t-stat.

    ICIR = mean(IC) / std(IC) * sqrt(n_periods), on the SAME non-overlapping
    dates `information_coefficient` uses. Overlapping windows would inflate the
    t-stat in the direction that flatters, which is the failure this module
    already refuses.

    Spearman is the default so a ranking question stays a ranking question.
    Pearson is available as `method="pearson"` and is a different measurement,
    reported separately rather than mixed into the Spearman figures.

    The original `information_coefficient` return shape is untouched.
    """
    horizon = max(1, int(horizon))
    method = str(method or "spearman").lower()
    if method not in ("spearman", "pearson"):
        raise PanelShapeError("method is 'spearman' or 'pearson'.")
    S, P, names, skipped = _tail_panel(signal, prices, need=horizon + 1)
    if names:
        ics = _period_ics_pearson(S, P, horizon) if method == "pearson" else _period_ics(S, P, horizon)
    else:
        ics = []

    mean_ic: float | None = None
    t_stat: float | None = None
    t_stat_nw: float | None = None
    icir: float | None = None
    if ics:
        arr = np.asarray(ics, dtype=float)
        mean_ic = round(float(arr.mean()), _RND)
        if len(ics) > 1 and float(arr.std(ddof=1)) > 0:
            sd = float(arr.std(ddof=1))
            t_stat = round(float(arr.mean() / (sd / np.sqrt(len(ics)))), _RND)
            icir = round(float(arr.mean() / sd * np.sqrt(len(ics))), _RND)
        nw = newey_west_tstat(arr, lags=1)
        t_stat_nw = None if nw is None else round(float(nw), _RND)

    return {
        "mean_ic": mean_ic,
        "icir": icir,
        "ic_t_stat": t_stat,
        "ic_t_stat_nw": t_stat_nw,
        "n_periods": len(ics),
        "ic_by_period": [round(v, _RND) for v in ics[-MAX_PERIODS:]],
        "n_names": len(names),
        "n_skipped": len(skipped),
        "skipped": skipped[:MAX_PERIODS],
        "horizon": horizon,
        "method": method,
    }


def quantile_returns(
    signal: dict,
    prices: dict,
    *,
    horizon: int = DEFAULT_HORIZON,
    quantiles: int = DEFAULT_QUANTILES,
) -> dict:
    """Mean forward return per signal quantile, and the top-minus-bottom spread.

    The spread is the tradeable statement an IC only implies: what buying the
    top bucket against the bottom one earned per period, in percent. Reported
    per-period rather than annualised — annualising a 21-day spread invites
    compounding assumptions this function has no business making.
    """
    horizon = max(1, int(horizon))
    quantiles = max(2, min(int(quantiles), 10))
    S, P, names, skipped = _tail_panel(signal, prices, need=horizon + 1)

    sums = np.zeros(quantiles)
    counts = np.zeros(quantiles, dtype=int)
    n_periods = 0
    if names:
        depth = S.shape[0]
        for t in range(0, depth - horizon, horizon):
            fwd = _forward(P, t, horizon)
            ok = np.isfinite(S[t]) & np.isfinite(fwd)
            if int(ok.sum()) < max(MIN_NAMES, quantiles):
                continue
            s, f = S[t][ok], fwd[ok]
            if np.ptp(s) == 0:
                continue
            bucket = np.minimum((_rank(s) / len(s) * quantiles).astype(int), quantiles - 1)
            for q in range(quantiles):
                members = f[bucket == q]
                if members.size:
                    sums[q] += float(members.mean())
                    counts[q] += 1
            n_periods += 1

    means = [round(float(sums[q] / counts[q]) * 100.0, _RND) if counts[q] else None for q in range(quantiles)]
    spread: float | None = None
    if means[0] is not None and means[-1] is not None:
        spread = round(means[-1] - means[0], _RND)

    return {
        "quantile_mean_pct": means,
        "spread_pct": spread,
        "n_periods": n_periods,
        "quantiles": quantiles,
        "horizon": horizon,
        "n_names": len(names),
        "n_skipped": len(skipped),
    }


def signal_decay(signal: dict, prices: dict, *, horizons: tuple[int, ...] = (1, 5, 21, 63)) -> dict:
    """The IC at several horizons: how long the information lasts.

    A signal that only works at one day is a different instrument from one
    that holds for a quarter, and sizing or turnover decisions read straight
    off this shape. Horizons the data cannot support report None, never 0.
    """
    cleaned = sorted({max(1, int(h)) for h in horizons})
    out: list[float | None] = []
    for h in cleaned:
        reading = information_coefficient(signal, prices, horizon=h)
        out.append(reading["mean_ic"])

    half_life: float | None = None
    first = out[0] if out else None
    if first is not None and first != 0:
        for h, ic in zip(cleaned[1:], out[1:], strict=True):
            if ic is not None and abs(ic) <= abs(first) / 2.0:
                half_life = float(h)
                break

    return {
        "horizons": list(cleaned),
        "ic_at_horizon": out,
        "half_life_periods": half_life,
    }


__all__ = [
    "information_coefficient",
    "signal_icir",
    "quantile_returns",
    "signal_decay",
    "PanelShapeError",
    "MIN_NAMES",
    "MAX_PERIODS",
]
