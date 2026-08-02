"""
screen.py, rank a SUPPLIED universe of price series and return the shortlist.

Pure deterministic math (no fetch, no LLM, no randomness). Prices are
caller-supplied, exactly as in `technical.py`: this module never learns what a
symbol is worth from anywhere but its arguments.

WHY THIS EXISTS SEPARATELY FROM `technical_features`
    `technical_features` answers "what are the readings on every name", and its
    return grows linearly with the universe. That is the right shape for a
    handful of names in a notebook and the wrong shape for a workflow step: a
    500-name universe becomes a 500-entry payload, and the run's own record ends
    up holding a last close for every symbol somebody screened.

    `screen_universe` answers a narrower question — "which names clear these
    conditions, best first" — and returns a BOUNDED shortlist. The universe stays
    on the caller's machine and a fixed number of rows leave it. That is a better
    answer to the actual question and a smaller footprint, in that order.

THE METRICS SHARE ONE IMPLEMENTATION WITH `technical.py`
    RSI and SMA are imported from that module rather than re-derived here. Two
    copies of Wilder's smoothing would agree on the day they were written and
    drift afterwards, and the screen disagreeing with the feature table about the
    same symbol's RSI is precisely the kind of contradiction that makes a
    reviewer stop believing both.

    Math conventions are therefore `technical.py`'s, documented there.

INSUFFICIENT IS NOT FAILING
    A name whose history is too short to compute the metric being ranked is
    neither passed nor rejected: it is counted, named, and reported. Folding it
    into "did not pass" would silently convert missing data into a negative
    finding, and the difference between "this name does not clear the bar" and
    "we could not tell" is the whole product.

FILTERS CARRY VALUES, NOT EXPRESSIONS
    A filter is `{metric: {"min": x, "max": y}}`. There is deliberately no
    expression language, no lambda, and no string to eval. The same argument as
    the harness vocabulary's: the moment a filter can carry a program, this stops
    being a screen and becomes a remote-execution channel wearing a screen's
    clothes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .technical import _extract, _sma, _wilder_rsi

# A shortlist, not a data export. The harness refuses any list over 64 elements
# on both sides of the wire, and a screen that wanted to return more than this
# is not a screen. Requests above the cap are honoured up to it and the result
# says it was truncated rather than pretending the cap was the answer.
MAX_ROWS = 50

DEFAULT_TOP_N = 20
DEFAULT_RETURN_LOOKBACK = 63  # ~ one quarter of trading days
DEFAULT_VOL_LOOKBACK = 63
_TRADING_DAYS = 252
_RND = 6

#: What may be ranked or filtered on. An allowlist rather than getattr on a
#: string, so an unknown metric is a clean error at the boundary instead of a
#: mystery `None` that quietly ranks every name equal.
METRICS = (
    "rsi",
    "return_pct",
    "distance_from_sma_pct",
    "volatility_pct",
)


class UnknownMetric(ValueError):
    """Ranked or filtered on a metric this module does not compute.

    Raised rather than ignored. A silently dropped filter returns MORE names than
    the caller asked for, and a screen that quietly widens is worse than one that
    refuses: the extra names look like findings.
    """


def _return_pct(closes: np.ndarray, lookback: int) -> float | None:
    if len(closes) < lookback + 1:
        return None
    start = float(closes[-(lookback + 1)])
    if start == 0:
        return None
    return (float(closes[-1]) - start) / start * 100.0


def _volatility_pct(closes: np.ndarray, lookback: int) -> float | None:
    """Annualised standard deviation of simple daily returns, in percent."""
    if len(closes) < lookback + 1:
        return None
    window = closes[-(lookback + 1) :]
    prev = window[:-1]
    if np.any(prev == 0):
        return None
    rets = np.diff(window) / prev
    if rets.size < 2:
        return None
    return float(np.std(rets, ddof=1)) * float(np.sqrt(_TRADING_DAYS)) * 100.0


def _distance_from_sma_pct(closes: np.ndarray, window: int) -> float | None:
    value = _sma(closes, window)
    if value is None or value == 0:
        return None
    return (float(closes[-1]) - value) / value * 100.0


def _metrics_for(
    closes: np.ndarray,
    *,
    rsi_window: int,
    sma_window: int,
    return_lookback: int,
    vol_lookback: int,
) -> dict[str, float | None]:
    rsi = _wilder_rsi(closes, rsi_window)
    return {
        "rsi": round(rsi, 4) if rsi is not None else None,
        "return_pct": _round(_return_pct(closes, return_lookback)),
        "distance_from_sma_pct": _round(_distance_from_sma_pct(closes, sma_window)),
        "volatility_pct": _round(_volatility_pct(closes, vol_lookback)),
    }


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, _RND)


def _check_metric(name: str, where: str) -> None:
    if name not in METRICS:
        raise UnknownMetric(f"{where} names {name!r}; this build computes {list(METRICS)}")


def _passes(values: dict[str, float | None], filters: dict[str, dict[str, float]]) -> bool | None:
    """True, False, or None when a filtered metric could not be computed.

    None is the third answer and it is load-bearing. A name with no RSI has not
    failed an RSI filter, and reporting it as a rejection would be a claim the
    data does not support.
    """
    for metric, bound in filters.items():
        value = values.get(metric)
        if value is None:
            return None
        low, high = bound.get("min"), bound.get("max")
        if low is not None and value < float(low):
            return False
        if high is not None and value > float(high):
            return False
    return True


def screen_universe(
    prices: dict,
    *,
    rank_by: str = "return_pct",
    descending: bool = True,
    top_n: int = DEFAULT_TOP_N,
    filters: dict[str, dict[str, float]] | None = None,
    rsi_window: int = 14,
    sma_window: int = 200,
    return_lookback: int = DEFAULT_RETURN_LOOKBACK,
    vol_lookback: int = DEFAULT_VOL_LOOKBACK,
) -> dict:
    """Rank a universe of price series and return the shortlist.

    Args:
        prices: {symbol: series}, in any shape `technical.py` accepts — bare
            closes, {date: close}, or OHLC row dicts.
        rank_by: which metric orders the result. One of `METRICS`.
        descending: True ranks the largest first, which is what "outperforming"
            means for a return and what "overbought" means for RSI. Set False for
            a metric where small is the interesting end (volatility, say).
        top_n: how many rows come back, capped at `MAX_ROWS`.
        filters: {metric: {"min": x, "max": y}}. Absent bounds are not applied.
        rsi_window / sma_window / return_lookback / vol_lookback: metric windows.

    Returns:
        A dict with `rows` (the shortlist, ranked), the three counts that make it
        readable — how many names were in the universe, how many could be
        evaluated, how many cleared the filters — and the parameters used, so the
        result can be reproduced without the caller's notes.

    Raises:
        UnknownMetric: `rank_by` or a filter key is not in `METRICS`.
    """
    filters = dict(filters or {})
    _check_metric(rank_by, "rank_by")
    for key in filters:
        _check_metric(key, "filters")

    limit = max(1, min(int(top_n), MAX_ROWS))

    evaluated: list[dict[str, Any]] = []
    insufficient: list[str] = []
    undetermined: list[str] = []

    for symbol, series in (prices or {}).items():
        closes, _highs, _lows = _extract(series)
        if len(closes) == 0:
            insufficient.append(str(symbol))
            continue

        values = _metrics_for(
            closes,
            rsi_window=rsi_window,
            sma_window=sma_window,
            return_lookback=return_lookback,
            vol_lookback=vol_lookback,
        )

        score = values.get(rank_by)
        if score is None:
            # Cannot be ranked, so it cannot appear in a ranking. Named, not
            # dropped: a shortlist of 20 from a universe of 500 means something
            # different when 400 of them could not be measured.
            insufficient.append(str(symbol))
            continue

        verdict = _passes(values, filters)
        if verdict is None:
            undetermined.append(str(symbol))
            continue
        if verdict is False:
            evaluated.append({"symbol": str(symbol), "_passed": False})
            continue

        evaluated.append(
            {
                "symbol": str(symbol),
                "_passed": True,
                "score": score,
                "n_observations": int(len(closes)),
                "last_close": round(float(closes[-1]), _RND),
                **values,
            }
        )

    passing = [row for row in evaluated if row["_passed"]]
    passing.sort(key=lambda r: (float(r["score"]), str(r["symbol"])), reverse=bool(descending))

    rows: list[dict[str, Any]] = []
    for position, row in enumerate(passing[:limit], start=1):
        out = {k: v for k, v in row.items() if k != "_passed"}
        out["rank"] = position
        rows.append(out)

    return {
        "rows": rows,
        "rank_by": rank_by,
        "descending": bool(descending),
        "universe_size": len(prices or {}),
        # Could be measured on the ranked metric — including the names whose
        # filter verdict then came out unknown. The denominator that makes the
        # shortlist mean something, and it closes:
        #     universe_size == n_evaluated + n_insufficient
        #     n_evaluated   == n_passing + n_failing + n_undetermined
        "n_evaluated": len(evaluated) + len(undetermined),
        "n_failing": sum(1 for row in evaluated if not row["_passed"]),
        "n_passing": len(passing),
        "n_returned": len(rows),
        "truncated": len(passing) > len(rows),
        # Three distinct ways a name did not reach the list, kept apart because
        # they mean different things to whoever reads the screen.
        "n_insufficient": len(insufficient),
        "n_undetermined": len(undetermined),
        "insufficient": sorted(insufficient)[:MAX_ROWS],
        "undetermined": sorted(undetermined)[:MAX_ROWS],
        "params": {
            "top_n": limit,
            "filters": filters,
            "rsi_window": rsi_window,
            "sma_window": sma_window,
            "return_lookback": return_lookback,
            "vol_lookback": vol_lookback,
        },
    }


__all__ = ["screen_universe", "METRICS", "MAX_ROWS", "UnknownMetric"]
