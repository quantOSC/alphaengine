"""
profile.py, what is wrong with this data, said before anything runs on it.

Pure deterministic checks over a CALLER-SUPPLIED universe. Nothing here judges
a strategy; it judges the file — which is where 40 to 50 percent of a research
seat's week actually goes, and where the silent failures live: the unflagged
split that becomes a 90 percent "return", the stale series that becomes a
volatility of zero, the three names whose history quietly ends in March.

REPORTS, NEVER REPAIRS
    A gap stays a gap and a spike stays a spike. Filling or winsorising here
    would invent observations nobody supplied, and every coverage figure
    downstream would be a lie about data that never existed. The output names
    what a person should look at; the person decides.

WHAT THIS CANNOT SEE IS SAID, NOT SKIPPED
    Series may arrive as bare closes with no dates. Calendar gaps are then
    unknowable, and `dates_supplied: false` says so — a health report that
    silently skipped the checks it could not run would read as a clean bill.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_RND = 6
MAX_NAMED = 24

#: A one-period move at or beyond this fraction is flagged as a possible
#: unadjusted split or bad print. Deliberately far outside normal equity moves.
SPIKE = 0.40
#: This many consecutive identical closes reads as a stale series, not a
#: quiet one.
STALE_RUN = 5


def _closes(series: Any) -> tuple[np.ndarray, bool]:
    """(closes, dates_present) for one symbol's series, tolerant of both the
    shapes a universe arrives in: bare numerics, or {date, close} rows."""
    if isinstance(series, (list, tuple)):
        if series and isinstance(series[0], dict):
            vals = [
                float(r["close"])
                for r in series
                if isinstance(r, dict) and isinstance(r.get("close"), (int, float))
            ]
            return np.asarray(vals, dtype=float), True
        try:
            return np.asarray(series, dtype=float), False
        except (TypeError, ValueError):
            return np.empty(0), False
    return np.empty(0), False


def _stale_runs(closes: np.ndarray) -> int:
    if closes.size < STALE_RUN:
        return 0
    runs, current = 0, 1
    for prev, cur in zip(closes[:-1], closes[1:], strict=True):
        current = current + 1 if cur == prev else 1
        if current == STALE_RUN:
            runs += 1
    return runs


def profile_data(prices: dict) -> dict:
    """The health report for one universe.

    Returns counts and BOUNDED name lists for: length dispersion, short
    series, zero or negative closes, one-period spikes at or over `SPIKE`
    (split candidates), stale runs, and non-numeric entries. `dates_supplied`
    records whether calendar checks were even possible.
    """
    if not isinstance(prices, dict):
        return {
            "n_names": 0,
            "usable": 0,
            "dates_supplied": False,
            "unreadable": [],
            "n_unreadable": 0,
        }

    lengths: list[int] = []
    dates_any = False
    unreadable: list[str] = []
    nonpositive: list[str] = []
    spiky: list[str] = []
    stale: list[str] = []
    n_spikes = 0

    for symbol, series in prices.items():
        name = str(symbol)
        closes, dated = _closes(series)
        dates_any = dates_any or dated
        if closes.size == 0 or not np.all(np.isfinite(closes)):
            unreadable.append(name)
            continue
        lengths.append(int(closes.size))
        if np.any(closes <= 0):
            nonpositive.append(name)
        prev = closes[:-1]
        safe = prev != 0
        if closes.size >= 2 and np.any(safe):
            moves = np.abs(closes[1:][safe] / prev[safe] - 1.0)
            spikes_here = int((moves >= SPIKE).sum())
            if spikes_here:
                spiky.append(name)
                n_spikes += spikes_here
        if _stale_runs(closes):
            stale.append(name)

    arr = np.asarray(lengths, dtype=float) if lengths else np.empty(0)
    longest = int(arr.max()) if arr.size else 0
    short = (
        sorted(
            str(s)
            for s, series in prices.items()
            if str(s) not in unreadable and _closes(series)[0].size < longest
        )
        if longest
        else []
    )

    return {
        "n_names": len(prices),
        "usable": len(lengths),
        "n_unreadable": len(unreadable),
        "unreadable": sorted(unreadable)[:MAX_NAMED],
        "n_obs_max": longest or None,
        "n_obs_min": int(arr.min()) if arr.size else None,
        "n_obs_median": float(np.median(arr)) if arr.size else None,
        # Names that end early or start late relative to the longest series —
        # the shape survivorship arrives in.
        "n_ragged": len(short),
        "ragged": short[:MAX_NAMED],
        "n_nonpositive": len(nonpositive),
        "nonpositive": sorted(nonpositive)[:MAX_NAMED],
        # Possible unadjusted splits or bad prints. Candidates, not verdicts.
        "n_spike_names": len(spiky),
        "n_spikes": n_spikes,
        "spike_names": sorted(spiky)[:MAX_NAMED],
        "n_stale_names": len(stale),
        "stale_names": sorted(stale)[:MAX_NAMED],
        "dates_supplied": dates_any,
    }


__all__ = ["profile_data", "SPIKE", "STALE_RUN", "MAX_NAMED"]
