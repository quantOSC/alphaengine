"""
series_shapes.py, one reading of the three shapes a per-symbol series arrives in.

A universe's series reaches this package as bare numbers, as `[{date, close}]`
rows, or as a `{date: close}` MAPPING — the last being what the portal's own
cache stores and what the CSV loader emits for any dated file. For a while only
`profile_data` tolerated all three, so a dated universe profiled clean and then
DIED in the very next step: the screen, the sizing derivation and the signal
panels each assumed lists, and a well-formed mapping read as an empty series.
One reader, imported everywhere a per-symbol series is consumed, is the fix
that cannot drift.

THE SEMANTICS, PINNED
    - A mapping is sorted by its DATE KEY, because a mapping's order is the
      producer's business and every metric downstream reads left to right.
      Non-numeric values (including bools) are skipped, not guessed at.
    - Rows keep only the entries whose `close` is numeric. Dates are reported
      only when at least one row carries a non-empty `date` — claiming dates
      we do not have would turn every calendar figure downstream into a
      statement about the empty string.
    - A bare sequence stays exactly what it was: values with NO dates, which
      is what makes `dates_supplied: false` an earned flag rather than a
      default.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["series_values"]


def series_values(series: Any) -> tuple[np.ndarray, list[str] | None]:
    """(values, dates) for one symbol's series. `dates` is None when the caller
    supplied bare numbers, which is what makes every calendar check downstream
    unrunnable rather than clean.

    Accepts the three shapes named in the module docstring; anything else
    reads as empty rather than raising, because the callers each own their own
    refusal message and a shared helper guessing at one would name the wrong
    fix.
    """
    if isinstance(series, dict):
        # {date: close}. Sorted by key, because a mapping's order is the
        # producer's business and every check downstream reads left to right.
        pairs = [
            (str(k), float(v))
            for k, v in series.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        if not pairs:
            return np.empty(0), None
        pairs.sort(key=lambda kv: kv[0])
        return np.asarray([v for _, v in pairs], dtype=float), [k for k, _ in pairs]

    if isinstance(series, (list, tuple)):
        if series and isinstance(series[0], dict):
            rows = [r for r in series if isinstance(r, dict) and isinstance(r.get("close"), (int, float))]
            vals = np.asarray([float(r["close"]) for r in rows], dtype=float)
            dates = [str(r.get("date") or "") for r in rows]
            # A row shape with no date in it is still a row shape, and claiming
            # dates we do not have would turn every calendar figure into a
            # statement about the empty string.
            return vals, dates if any(dates) else None
        try:
            return np.asarray(series, dtype=float), None
        except (TypeError, ValueError):
            return np.empty(0), None
    return np.empty(0), None
