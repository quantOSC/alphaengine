"""
stress.py, where a validated result breaks, and whether an idea is the book again.

Pure deterministic math over CALLER-SUPPLIED return series. Everything here
answers a question a study leaves open once it clears its gates:

    subperiod_stability   did one stretch pay for everything?
    cost_ladder           at how many basis points does it die?
    drawdown_anatomy      what does holding it actually feel like?
    overlap_stats         is this new, or my book wearing a new name?

NULLS ARE ABSENCES, NEVER ZEROS
    A cost ladder without a turnover figure reports None for every rung — a
    made-up turnover would produce a confident curve about a strategy nobody
    runs. The caller knows their turnover; this module does not guess it.

NO DATES REQUIRED, AND THE CONSEQUENCE IS STATED
    Subperiods are equal slices of the series, not calendar years. With no
    dates supplied that is the only honest cut; a reader comparing "segment 3"
    to a year on their own calendar is told by the field names that they
    cannot.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .series_shapes import series_values

_RND = 6
_TRADING_DAYS = 252

DEFAULT_SEGMENTS = 4
DEFAULT_BPS = (0.0, 5.0, 10.0, 25.0, 50.0)
#: Fewest observations a segment Sharpe can rest on.
MIN_SEGMENT_OBS = 20
#: Fewest shared observations an overlap reading can rest on.
MIN_OVERLAP_OBS = 20


def _returns(values: Any) -> np.ndarray | None:
    # A {date: value} mapping is a dated series, not a refusal: the shared
    # reader sorts it by key, exactly as every other consumer of the shape.
    if isinstance(values, dict):
        values = series_values(values)[0]
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr


def _sharpe(rets: np.ndarray) -> float | None:
    if rets.size < 2:
        return None
    sd = float(rets.std(ddof=1))
    if sd == 0:
        return None
    return float(rets.mean() / sd * np.sqrt(_TRADING_DAYS))


def subperiod_stability(returns: Any, *, segments: int = DEFAULT_SEGMENTS) -> dict:
    """The Sharpe of each equal slice of the record, and who paid for the total.

    `share_of_pnl_in_best_segment` is the figure that unmasks a one-regime
    result: 0.9 says a single stretch earned nearly everything and the rest
    was noise wearing a track record.
    """
    arr = _returns(returns)
    segments = max(2, min(int(segments), 12))
    if arr is None or arr.size < segments * MIN_SEGMENT_OBS:
        return {
            "segment_sharpes": None,
            "worst_segment_sharpe": None,
            "share_of_pnl_in_best_segment": None,
            "n_segments": segments,
            "n_obs": 0 if arr is None else int(arr.size),
        }

    slices = np.array_split(arr, segments)
    sharpes = [_sharpe(part) for part in slices]
    pnl = np.array([float(part.sum()) for part in slices])
    total = float(np.abs(pnl).sum())
    share: float | None = None
    if total > 0:
        share = round(float(np.max(pnl) / total), _RND)

    return {
        "segment_sharpes": [None if s is None else round(s, _RND) for s in sharpes],
        "worst_segment_sharpe": (
            None if any(s is None for s in sharpes) else round(min(s for s in sharpes if s is not None), _RND)
        ),
        "share_of_pnl_in_best_segment": share,
        "n_segments": segments,
        "n_obs": int(arr.size),
    }


def cost_ladder(
    returns: Any,
    *,
    turnover: float | None,
    bps_levels: tuple[float, ...] = DEFAULT_BPS,
) -> dict:
    """The Sharpe at each cost level, and the level where it stops being one.

    `turnover` is ONE-WAY turnover per period as a fraction of capital, and it
    is the caller's number: nothing here can know how often their strategy
    trades. Absent, every rung is None and `dies_at_bps` is None — an absence,
    not a survival.
    """
    levels = sorted({round(float(b), 4) for b in bps_levels})
    arr = _returns(returns)
    if arr is None or turnover is None:
        return {
            "bps_levels": levels,
            "sharpe_at_bps": [None] * len(levels),
            "dies_at_bps": None,
            "turnover": None if turnover is None else round(float(turnover), _RND),
            "n_obs": 0 if arr is None else int(arr.size),
        }

    per_period_cost = [float(turnover) * (b / 10_000.0) for b in levels]
    sharpes = [_sharpe(arr - c) for c in per_period_cost]
    dies_at: float | None = None
    for level, s in zip(levels, sharpes, strict=True):
        if s is not None and s <= 0:
            dies_at = level
            break

    return {
        "bps_levels": levels,
        "sharpe_at_bps": [None if s is None else round(s, _RND) for s in sharpes],
        "dies_at_bps": dies_at,
        "turnover": round(float(turnover), _RND),
        "n_obs": int(arr.size),
    }


def drawdown_anatomy(returns: Any) -> dict:
    """What holding the record felt like, beyond its worst single number.

    `time_underwater_share` is the fraction of periods spent below a prior
    peak — the figure that separates a strategy that dips from one that is
    underwater for a year and ends at the same max drawdown.
    """
    arr = _returns(returns)
    if arr is None or arr.size < 2:
        return {
            "max_drawdown_pct": None,
            "longest_underwater_periods": None,
            "n_drawdowns_over_10pct": None,
            "time_underwater_share": None,
            "n_obs": 0 if arr is None else int(arr.size),
        }

    equity = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, equity / peak - 1.0, 0.0)

    underwater = dd < 0
    longest = current = 0
    episodes_over_10 = 0
    trough = 0.0
    for below, depth in zip(underwater, dd, strict=True):
        if below:
            current += 1
            trough = min(trough, float(depth))
        else:
            if trough <= -0.10:
                episodes_over_10 += 1
            current, trough = 0, 0.0
        longest = max(longest, current)
    if trough <= -0.10:
        episodes_over_10 += 1

    return {
        # POSITIVE loss magnitude, the convention everywhere in this codebase.
        "max_drawdown_pct": round(float(-dd.min()) * 100.0, _RND),
        "longest_underwater_periods": int(longest),
        "n_drawdowns_over_10pct": int(episodes_over_10),
        "time_underwater_share": round(float(underwater.mean()), _RND),
        "n_obs": int(arr.size),
    }


def overlap_stats(candidate: Any, book: Any) -> dict:
    """How much of the candidate the book already owns.

    Correlation and beta of the candidate's returns to the book's, on their
    shared tail. The judgment — at what correlation an idea stops being new —
    belongs to whoever is orchestrating, not here.
    """
    cand, held = _returns(candidate), _returns(book)
    if cand is None or held is None:
        return {"correlation": None, "beta_to_book": None, "n_obs": 0}

    depth = min(cand.size, held.size)
    if depth < MIN_OVERLAP_OBS:
        return {"correlation": None, "beta_to_book": None, "n_obs": int(depth)}
    c, b = cand[-depth:], held[-depth:]
    sc, sb = float(c.std(ddof=1)), float(b.std(ddof=1))
    if sc == 0 or sb == 0:
        return {"correlation": None, "beta_to_book": None, "n_obs": int(depth)}

    cov = float(np.mean((c - c.mean()) * (b - b.mean())))
    return {
        "correlation": round(cov / (sc * sb) * depth / (depth - 1), _RND),
        "beta_to_book": round(cov / (sb * sb) * depth / (depth - 1), _RND),
        "n_obs": int(depth),
    }


__all__ = [
    "subperiod_stability",
    "cost_ladder",
    "drawdown_anatomy",
    "overlap_stats",
    "MIN_SEGMENT_OBS",
    "MIN_OVERLAP_OBS",
]
