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

    The flag has to be earned in both directions, and for a while it was not:
    it was reported as `true` for row-shaped input while the dates themselves
    were dropped on the floor and no calendar check existed on either branch.
    A capability flag for a capability nobody wrote is worse than no flag,
    because a reader takes `true` to mean the checks ran. Where it now says
    `true`, `panel_first`, `panel_last`, `n_ends_early` and `live_by_period`
    are what it bought.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .series_shapes import series_values

_RND = 6
MAX_NAMED = 24

#: A one-period move at or beyond this fraction is flagged as a possible
#: unadjusted split or bad print. Deliberately far outside normal equity moves.
SPIKE = 0.40
#: This many consecutive identical closes reads as a stale series, not a
#: quiet one.
STALE_RUN = 5


# THREE SHAPES, AND THE THIRD IS WHY THE READER MOVED. A universe arrives as
# bare closes, as `[{date, close}]` rows, or as a `{date: close}` MAPPING — the
# last being exactly what the portal's own cache stores and serves. The mapping
# branch once fell through to `np.empty(0)` here, so a well-formed dated
# universe profiled as one hundred percent UNREADABLE. The reading now lives in
# `series_shapes.series_values`, shared with every other consumer of a
# per-symbol series, so this module can never again tolerate a shape the rest
# of the pipeline dies on.
_closes = series_values


def _stale_runs(closes: np.ndarray) -> int:
    if closes.size < STALE_RUN:
        return 0
    runs, current = 0, 1
    for prev, cur in zip(closes[:-1], closes[1:], strict=True):
        current = current + 1 if cur == prev else 1
        if current == STALE_RUN:
            runs += 1
    return runs


# ── the shapes, bounded ────────────────────────────────────────────────────
# A count answers "how many are ragged"; a DISTRIBUTION answers "how ragged,
# and are the short ones a handful or a third of the panel" — which is the
# question a person actually has before deciding whether to run on the file.
#
# Every figure below is bounded by construction and none of them is anybody's
# price history: deciles are ten numbers, a histogram is its bins, and a
# coverage count is a count of NAMES. The protocol refuses a list over 64
# elements, and that limit is the reason these are summaries rather than
# series — correctly, because the raw panel belongs on the caller's machine.
MAX_BUCKETS = 48
MAX_BINS = 20


def _deciles(values: np.ndarray) -> list[float] | None:
    """The ten-point shape of a distribution. FLAT MEANS ALIGNED — a panel
    where every name carries the same history draws a level line, which is a
    result and reads as one. A cliff at the left is a subset with short
    history, which is where survivorship enters a study."""
    if values.size == 0:
        return None
    qs = np.percentile(values, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    return [round(float(q), _RND) for q in qs]


def _histogram(values: np.ndarray) -> dict[str, list[float]] | None:
    """Names by history length. Returns None when every name is the same
    length: a one-bar histogram is not a chart, and the deciles already say
    "aligned" more clearly than a single column would."""
    if values.size == 0:
        return None
    lo, hi = float(values.min()), float(values.max())
    if lo == hi:
        return None
    bins = int(min(MAX_BINS, max(4, np.sqrt(values.size))))
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
    return {
        "edges": [round(float(e), _RND) for e in edges],
        "counts": [int(c) for c in counts],
    }


def _live_by_period(spans: list[tuple[str, str]], calendar: list[str]) -> list[dict[str, Any]] | None:
    """How many names carry data through each slice of the panel's calendar.

    THE SURVIVORSHIP CHART, and the one figure here that could not exist
    before dates were read. A count that climbs from the left is backfill —
    names added as history became available. A count that falls at the right
    is names whose history ENDS EARLY, which arrives downstream as a study
    quietly run on the survivors.

    Bucketed to at most `MAX_BUCKETS` so the result is a shape rather than a
    calendar. A name counts as live in a bucket when its own span covers any
    part of it; that is deliberately generous, because the failure worth
    catching is a name absent for a whole segment, not one missing a Tuesday.
    """
    if not calendar or not spans:
        return None
    n_dates = len(calendar)
    n_buckets = int(min(MAX_BUCKETS, n_dates))
    if n_buckets < 2:
        return None
    # Contiguous, near-equal slices of the calendar by POSITION, not by
    # elapsed time: the panel's own trading days are the axis a quant reads.
    bounds = [round(i * n_dates / n_buckets) for i in range(n_buckets + 1)]
    index = {d: i for i, d in enumerate(calendar)}
    counts = [0] * n_buckets
    for first, last in spans:
        a, b = index.get(first), index.get(last)
        if a is None or b is None:
            continue
        for k in range(n_buckets):
            if a < bounds[k + 1] and b >= bounds[k]:
                counts[k] += 1
    return [{"period": calendar[bounds[k]], "n_live": counts[k]} for k in range(n_buckets)]


def profile_data(prices: dict) -> dict:
    """The health report for one universe.

    Returns counts and BOUNDED name lists for: length dispersion, short
    series, zero or negative closes, one-period spikes at or over `SPIKE`
    (split candidates), stale runs, and non-numeric entries. `dates_supplied`
    records whether calendar checks were even possible.

    It also returns the SHAPES those counts summarise — history-length
    deciles and histogram, spike severity ranked, and, when dates are real,
    the panel's span, the names that end before it does, and how many names
    are live through each slice of its calendar. A count tells a reader
    whether to worry; the shape is what they would have drawn by hand before
    deciding, and it is bounded so that drawing it never moves the panel.
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
    #: (name, worst one-period move) for every flagged name. A 45 percent move
    #: may be a real day; a 400 percent move is a split nobody adjusted. The
    #: count alone cannot tell those apart and the severity can.
    worst_move: list[tuple[str, float]] = []
    #: (first, last) per readable name, and the union calendar they sit on.
    #: Only populated when the caller supplied dates.
    spans: list[tuple[str, str]] = []
    all_dates: set[str] = set()
    last_seen: dict[str, str] = {}

    for symbol, series in prices.items():
        name = str(symbol)
        closes, dates = _closes(series)
        dates_any = dates_any or dates is not None
        if closes.size == 0 or not np.all(np.isfinite(closes)):
            unreadable.append(name)
            continue
        lengths.append(int(closes.size))
        if dates:
            spans.append((dates[0], dates[-1]))
            all_dates.update(dates)
            last_seen[name] = dates[-1]
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
                worst_move.append((name, round(float(moves.max()) * 100.0, _RND)))
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

    # ── what dates make answerable, and only when they are really there ────
    #
    # `n_ragged` is a LENGTH test, so "starts late" and "ends early" arrive
    # under one number — and they are not the same finding. A name that starts
    # late is history you do not have; a name that ENDS early is a name that
    # left the panel, which is how survivorship gets into a study without
    # anybody choosing it. Dates separate them; nothing else can.
    calendar = sorted(all_dates)
    panel_last = calendar[-1] if calendar else None
    ends_early = sorted(n for n, last in last_seen.items() if panel_last and last < panel_last)
    dated: dict[str, Any] = {}
    if calendar:
        dated = {
            "panel_first": calendar[0],
            "panel_last": panel_last,
            "n_dates": len(calendar),
            "n_ends_early": len(ends_early),
            "ends_early": ends_early[:MAX_NAMED],
            "live_by_period": _live_by_period(spans, calendar),
        }

    worst_move.sort(key=lambda nv: nv[1], reverse=True)

    return {
        # ── the shapes ────────────────────────────────────────────────────
        # Counts say whether to worry; these say how much, and they are what
        # the run page draws. Every one is a summary of the panel and none of
        # them is the panel.
        "obs_deciles": _deciles(arr),
        "obs_hist": _histogram(arr),
        "spike_worst": [{"name": n, "move_pct": v} for n, v in worst_move[:MAX_NAMED]],
        **dated,
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


__all__ = ["profile_data", "SPIKE", "STALE_RUN", "MAX_NAMED", "MAX_BINS", "MAX_BUCKETS"]
