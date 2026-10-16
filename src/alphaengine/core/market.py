"""Read a loaded book as stocks and macro series that share a calendar.

Gaps stay gaps. A missing print is counted, not carried forward, because a
filled macro series is a series the file did not contain.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .relations import _looks_like_prices
from .series_shapes import series_values

# FRED-style names a macro file actually uses. A ticker is not in this set.
_MACRO = frozenset(
    {
        "dff",
        "fedfunds",
        "sofr",
        "dgs1mo",
        "dgs3mo",
        "dgs6mo",
        "dgs1",
        "dgs2",
        "dgs5",
        "dgs7",
        "dgs10",
        "dgs20",
        "dgs30",
        "t10y2y",
        "unrate",
        "payems",
        "cpi",
        "cpiaucsl",
        "pce",
        "pcepilfe",
        "gdp",
        "gdpc1",
        "indpro",
        "vix",
        "vixcls",
        "mortgage30us",
        "umcsent",
        "walcl",
    }
)


def series_kind(name: str, values: list[float]) -> str:
    """`macro`, `price`, or `series`. The name wins when it is a known macro print."""
    key = "".join(ch for ch in name.lower() if ch.isalnum())
    if key in _MACRO or key.startswith("dgs"):
        return "macro"
    if values and _looks_like_prices(values):
        return "price"
    return "series"


def _dated(series: Any) -> tuple[dict[str, float] | None, list[float]]:
    """A `{date: value}` mapping keeps its dates. A list does not grow any."""
    if isinstance(series, dict):
        out: dict[str, float] = {}
        for key, value in series.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number == number:  # not NaN
                out[str(key)] = number
        if out:
            return out, [out[k] for k in sorted(out)]
    values, _dates = series_values(series)
    if values.ndim != 1:
        return None, []
    clean = [float(v) for v in values.tolist() if np.isfinite(v)]
    return None, clean


def describe_market(data: Any) -> dict:
    """What is loaded: how many names, which are macro, which are prices, whether dates exist."""
    if not isinstance(data, dict) or not data:
        n = len(data) if isinstance(data, (list, tuple)) else 0
        return {"n_names": 1 if n else 0, "n_obs": n, "dates_supplied": False, "kind": "series"}
    macro: list[str] = []
    prices: list[str] = []
    other: list[str] = []
    dated = 0
    shortest = None
    for name, series in data.items():
        if str(name) in {"returns", "pnl", "book_returns", "signal", "prices"}:
            continue
        mapping, values = _dated(series)
        if not values:
            continue
        if mapping is not None:
            dated += 1
        kind = series_kind(str(name), values)
        bucket = {"macro": macro, "price": prices}.get(kind, other)
        bucket.append(str(name))
        shortest = len(values) if shortest is None else min(shortest, len(values))
    names = macro + prices + other
    return {
        "n_names": len(names),
        "n_macro": len(macro),
        "n_price": len(prices),
        "n_other": len(other),
        "dates_supplied": dated == len(names) and dated > 0,
        "n_dated": dated,
        "min_obs": shortest or 0,
        "macro": macro[:12],
        "prices": prices[:12],
    }


def align_market(data: Any) -> dict:
    """The dates every supplied series actually shares. Missing prints are counted, not filled."""
    if not isinstance(data, dict) or not data:
        return {"error": "load a file of named series first", "n_common": 0}
    dated: dict[str, dict[str, float]] = {}
    undated: list[str] = []
    for name, series in data.items():
        if str(name) in {"returns", "pnl", "book_returns", "signal", "prices"}:
            continue
        mapping, values = _dated(series)
        if mapping is None:
            if values:
                undated.append(str(name))
            continue
        dated[str(name)] = mapping
    if len(dated) < 1:
        return {
            "error": "these series have no dates, so a calendar cannot be shared",
            "n_undated": len(undated),
            "dates_supplied": False,
        }
    common = set.intersection(*(set(series) for series in dated.values()))
    union = set.union(*(set(series) for series in dated.values()))
    ordered = sorted(common)
    gaps = {name: len(union - set(series)) for name, series in dated.items()}
    kinds = {name: series_kind(name, list(series.values())) for name, series in dated.items()}
    return {
        "n_names": len(dated),
        "n_common": len(ordered),
        "start": ordered[0] if ordered else None,
        "end": ordered[-1] if ordered else None,
        "n_macro": sum(1 for kind in kinds.values() if kind == "macro"),
        "n_price": sum(1 for kind in kinds.values() if kind == "price"),
        "gaps": {name: count for name, count in gaps.items() if count},
        "n_undated": len(undated),
        "dates_supplied": True,
        "names": list(dated)[:24],
    }
