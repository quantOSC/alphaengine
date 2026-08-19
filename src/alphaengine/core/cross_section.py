"""Fama-MacBeth and a quantile book with turnover.

THE TRADEABLE STATEMENT AN IC ONLY IMPLIES. A Spearman IC says the ranking
predicts; Fama-MacBeth says how many units of forward return attach to a unit
of signal, with a t-stat that does not pretend overlapping dates are independent.
A quantile book says what buying the top against the bottom earned, AND how
much of the book turned over to do it — without turnover the cost ladder is
fiction.

NON-OVERLAPPING PERIODS, SAME REASON AS `signals.py`
    Evaluation dates step by the horizon. Overlapping forward windows share
    most of their returns and inflate the t-stat in the direction that flatters.

SKIPS ARE NAMED
    Names too short or absent from one panel are counted, never dropped
    silently. Dates with too few finite names to identify the regression are
    counted too.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .panel import newey_west_tstat
from .signals import (
    DEFAULT_HORIZON,
    DEFAULT_QUANTILES,
    MAX_PERIODS,
    MIN_NAMES,
    _forward,
    _rank,
    _tail_panel,
)

__all__ = ["fama_macbeth", "quantile_book"]

_RND = 6


def fama_macbeth(
    signal: dict,
    prices: dict,
    *,
    horizon: int = DEFAULT_HORIZON,
) -> dict[str, Any]:
    """Per-date cross-sectional OLS of forward returns on the signal, then
    the time-series mean and Newey-West t-stat of the slope.

    An intercept is included. Dates with fewer finite names than `MIN_NAMES`
    are skipped and counted. `lambda_mean` is None when nothing could be
    measured, never 0.
    """
    horizon = max(1, int(horizon))
    S, P, names, skipped = _tail_panel(signal, prices, need=horizon + 1)

    lambdas: list[float] = []
    r2s: list[float] = []
    n_names_used: list[int] = []
    n_dates_skipped = 0
    if names:
        depth = S.shape[0]
        for t in range(0, depth - horizon, horizon):
            y = _forward(P, t, horizon)
            x = S[t]
            ok = np.isfinite(y) & np.isfinite(x)
            n = int(ok.sum())
            if n < MIN_NAMES or np.ptp(x[ok]) == 0:
                n_dates_skipped += 1
                continue
            yy = y[ok]
            xx = x[ok]
            X = np.column_stack([np.ones(n), xx])
            beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
            fitted = X @ beta
            ss_res = float(np.sum((yy - fitted) ** 2))
            ss_tot = float(np.sum((yy - yy.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            lambdas.append(float(beta[1]))
            r2s.append(r2)
            n_names_used.append(n)

    lambda_mean: float | None = None
    t_stat: float | None = None
    t_stat_nw: float | None = None
    if lambdas:
        arr = np.asarray(lambdas, dtype=float)
        lambda_mean = round(float(arr.mean()), _RND)
        nw = newey_west_tstat(arr, lags=1)
        t_stat_nw = None if nw is None else round(float(nw), _RND)
        if len(lambdas) > 1 and float(arr.std(ddof=1)) > 0:
            t_stat = round(float(arr.mean() / (arr.std(ddof=1) / np.sqrt(len(lambdas)))), _RND)

    return {
        "lambda_mean": lambda_mean,
        "t_stat": t_stat,
        "t_stat_nw": t_stat_nw,
        "r2_mean": None if not r2s else round(float(np.mean(r2s)), _RND),
        "n_dates": len(lambdas),
        "n_dates_skipped": n_dates_skipped,
        "n_names": len(names),
        "n_names_mean": None if not n_names_used else round(float(np.mean(n_names_used)), _RND),
        "n_skipped": len(skipped),
        "skipped": skipped[:MAX_PERIODS],
        "horizon": horizon,
        "lambda_by_date": [round(v, _RND) for v in lambdas[-MAX_PERIODS:]],
    }


def quantile_book(
    signal: dict,
    prices: dict,
    *,
    horizon: int = DEFAULT_HORIZON,
    quantiles: int = DEFAULT_QUANTILES,
) -> dict[str, Any]:
    """Top-minus-bottom forward return AND one-way turnover of membership.

    Weights are equal inside the top bucket (+1/n_long) and the bottom bucket
    (-1/n_short). One-way turnover is `0.5 * sum |w_t - w_{t-1}|`, which lives
    in [0, 2] for a long-short book. The first period has no predecessor and
    is not a turnover observation.

    The spread is per-period percent, not annualised — same reason as
    `quantile_returns`.
    """
    horizon = max(1, int(horizon))
    quantiles = max(2, min(int(quantiles), 10))
    S, P, names, skipped = _tail_panel(signal, prices, need=horizon + 1)

    spreads: list[float] = []
    turnovers: list[float] = []
    prev_w: np.ndarray | None = None
    n_periods = 0
    if names:
        depth = S.shape[0]
        n = len(names)
        for t in range(0, depth - horizon, horizon):
            fwd = _forward(P, t, horizon)
            ok = np.isfinite(S[t]) & np.isfinite(fwd)
            if int(ok.sum()) < max(MIN_NAMES, quantiles):
                continue
            s, f = S[t][ok], fwd[ok]
            if np.ptp(s) == 0:
                continue
            bucket = np.minimum((_rank(s) / len(s) * quantiles).astype(int), quantiles - 1)
            top, bot = bucket == (quantiles - 1), bucket == 0
            n_top, n_bot = int(top.sum()), int(bot.sum())
            if n_top == 0 or n_bot == 0:
                continue
            spread = float(f[top].mean() - f[bot].mean())
            spreads.append(spread)
            n_periods += 1

            w = np.zeros(n)
            idx = np.flatnonzero(ok)
            w[idx[top]] = 1.0 / n_top
            w[idx[bot]] = -1.0 / n_bot
            if prev_w is not None:
                turnovers.append(0.5 * float(np.abs(w - prev_w).sum()))
            prev_w = w

    spread_pct = None if not spreads else round(float(np.mean(spreads)) * 100.0, _RND)
    turnover = None if not turnovers else round(float(np.mean(turnovers)), _RND)
    return {
        "spread_pct": spread_pct,
        "turnover_one_way": turnover,
        "n_periods": n_periods,
        "n_turnover_obs": len(turnovers),
        "quantiles": quantiles,
        "horizon": horizon,
        "n_names": len(names),
        "n_skipped": len(skipped),
        "turnover_by_period": [round(v, _RND) for v in turnovers[-MAX_PERIODS:]],
    }
