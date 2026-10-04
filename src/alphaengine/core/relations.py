"""Correlation, covariance, and cointegration across a loaded universe.

A universe is already {name: closes}. These read that panel on this machine.
Nothing here asks for a separate book series, and the matrix stays local: what
is said out loud is the counts and the strongest pairs, not every cell.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..charts import chart
from .panel import align_panel
from .series_shapes import series_values

_FIGURE_CAP = 512
_STRONG = 32

__all__ = [
    "KINDS",
    "as_closes",
    "as_returns",
    "correlation",
    "covariance",
    "cointegration",
    "overlap_against_the_rest",
    "portal_figures",
    "summary_lines",
]

KINDS = {
    "correlation": "correlation",
    "corr": "correlation",
    "covariance": "covariance",
    "cov": "covariance",
    "cointegration": "cointegration",
    "cointegrate": "cointegration",
    "coint": "cointegration",
}

_SKIP = frozenset({"returns", "pnl", "book_returns", "signal", "prices"})


def _values(series: Any) -> list[float] | None:
    arr, _dates = series_values(series)
    if arr.ndim != 1 or arr.size < 2:
        return None
    out = [float(v) for v in arr.tolist() if np.isfinite(v)]
    return out if len(out) >= 2 else None


def _looks_like_prices(values: list[float]) -> bool:
    """Closes sit well above 1. A return series does not."""
    return any(abs(v) > 1.5 for v in values)


def _to_returns(values: list[float]) -> list[float] | None:
    if not _looks_like_prices(values):
        return [round(v, 8) for v in values]
    if any(v == 0 for v in values[:-1]):
        return None
    return [round(values[i] / values[i - 1] - 1.0, 8) for i in range(1, len(values))]


def as_closes(data: Any) -> dict[str, list[float]]:
    """{name: levels}. A return series is left as-is; cointegration wants levels."""
    if not isinstance(data, dict):
        return {}
    out: dict[str, list[float]] = {}
    for name, series in data.items():
        if str(name) in _SKIP:
            continue
        values = _values(series)
        if values:
            out[str(name)] = values
    return out


def as_returns(data: Any) -> dict[str, list[float]]:
    """{name: per-period returns}, derived from closes when that is what was loaded."""
    out: dict[str, list[float]] = {}
    for name, values in as_closes(data).items():
        returns = _to_returns(values)
        if returns and len(returns) >= 2:
            out[name] = returns
    return out


def _matrix(panel: dict[str, list[float]], *, kind: str) -> dict[str, Any]:
    aligned, names, skipped = align_panel(panel, need=2)
    if len(names) < 2:
        raise ValueError("need at least two names with a shared history.")
    if kind == "correlation":
        grid = np.corrcoef(aligned, rowvar=False)
    else:
        grid = np.cov(aligned, rowvar=False, ddof=1)
    grid = np.atleast_2d(np.asarray(grid, dtype=float))
    rows: list[list[float | None]] = []
    for i in range(grid.shape[0]):
        row: list[float | None] = []
        for j in range(grid.shape[1]):
            value = float(grid[i, j])
            row.append(round(value, 4) if np.isfinite(value) else None)
        rows.append(row)
    pairs: list[dict[str, Any]] = []
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            value = rows[i][j]
            if value is None:
                continue
            pairs.append({"a": a, "b": names[j], "value": value})
    pairs.sort(key=lambda item: abs(float(item["value"])), reverse=True)
    return {
        "kind": kind,
        "names": names,
        "values": rows,
        "n_obs": int(aligned.shape[0]),
        "skipped": skipped,
        "pairs": pairs,
    }


def correlation(data: Any) -> dict[str, Any]:
    """Pearson correlation of every name with every other name."""
    return _matrix(as_returns(data), kind="correlation")


def covariance(data: Any) -> dict[str, Any]:
    """Sample covariance of every name with every other name."""
    return _matrix(as_returns(data), kind="covariance")


def cointegration(data: Any) -> dict[str, Any]:
    """Engle-Granger screen across the loaded closes. Levels, not returns."""
    try:
        from .pairs import find_cointegrated_pairs
    except ModuleNotFoundError as exc:
        raise ValueError(str(exc)) from exc
    closes = as_closes(data)
    prices = {name: values for name, values in closes.items() if _looks_like_prices(values)}
    if len(prices) < 2:
        raise ValueError("cointegration needs at least two names of closes, not a return series.")
    found = find_cointegrated_pairs(prices, cointegrated_only=False)
    rows = []
    for pair in found.get("pairs") or []:
        if not isinstance(pair, dict):
            continue
        coint = pair.get("cointegration") or {}
        rows.append(
            {
                "a": pair.get("ticker_a"),
                "b": pair.get("ticker_b"),
                "p": coint.get("p_value"),
                "half_life": pair.get("half_life_days"),
                "cointegrated": bool(pair.get("cointegrated")),
            }
        )
    rows.sort(key=lambda row: (not row["cointegrated"], row["p"] if row["p"] is not None else 1.0))
    return {
        "kind": "cointegration",
        "names": list(prices),
        "rows": rows,
        "n_pairs": int(found.get("n_evaluated") or len(rows)),
        "n_cointegrated": int(found.get("n_cointegrated") or sum(1 for row in rows if row["cointegrated"])),
        "expected_false_positives": found.get("expected_false_positives"),
        "multiple_testing_note": found.get("multiple_testing_note"),
        "skipped": [name for name in closes if name not in prices],
    }


def overlap_against_the_rest(data: Any, symbol: str) -> dict[str, Any]:
    """One name's returns, and the equal-weight of every other loaded name.

    That is the book `check_overlap` is asking for. It does not have to be
    loaded as a second series when the universe already holds the names.
    """
    panel = as_returns(data)
    key = next((name for name in panel if name.upper() == symbol.upper()), None)
    if key is None:
        known = ", ".join(list(panel)[:8]) or "none"
        raise ValueError(f"{symbol} is not in the loaded names. Loaded: {known}")
    candidate = panel.pop(key)
    if len(panel) < 1:
        raise ValueError(
            f"{key} is the only loaded name, so there is no rest of the book to compare it with."
        )
    aligned, names, _skipped = align_panel(panel, need=2)
    if len(names) < 1:
        raise ValueError("the other names do not share enough history to form a book.")
    depth = min(len(candidate), int(aligned.shape[0]))
    book = aligned[-depth:].mean(axis=1)
    series = np.asarray(candidate, dtype=float)[-depth:]
    return {
        "returns": [round(float(v), 8) for v in series.tolist()],
        "book_returns": [round(float(v), 8) for v in book.tolist()],
        "symbol": key,
        "book_names": names,
    }


def summary_lines(result: dict[str, Any]) -> list[str]:
    """What to say. The full matrix stays in the result, for the inspector."""
    kind = str(result.get("kind") or "")
    if kind == "cointegration":
        lines = [
            f"  cointegration   {result.get('n_cointegrated')} of {result.get('n_pairs')} pairs"
            "   across the loaded closes",
            "  check_overlap is a different question: one name against the rest of the book.",
        ]
        shown = 0
        for row in (result.get("rows") or [])[:8]:
            if row.get("p") is None and not row.get("cointegrated"):
                continue
            flag = "yes" if row.get("cointegrated") else "no"
            p = row.get("p")
            p_text = f"{float(p):.3f}" if isinstance(p, (int, float)) else "-"
            lines.append(f"  {row.get('a')}  {row.get('b')}   p {p_text}   cointegrated {flag}")
            shown += 1
        if not result.get("n_cointegrated") and shown:
            lines.append("  None cleared the cointegration bar. The pairs above are the closest.")
        elif not shown:
            lines.append("  The loaded closes are too short to test cointegration.")
        return lines
    names = result.get("names") or []
    lines = [f"  {kind}   {len(names)} names   {result.get('n_obs')} observations   full matrix on the right"]
    skipped = result.get("skipped") or []
    if skipped:
        lines.append(f"  {len(skipped)} names skipped (short history)")
    lines.append("  strongest pairs")
    for pair in (result.get("pairs") or [])[:8]:
        lines.append(f"  {pair['a']}  {pair['b']}   {pair['value']}")
    return lines


def portal_figures(result: dict[str, Any]) -> dict[str, Any]:
    """The payload the portal draws. Derived numbers only, each list at most 512 long.

    A 100-name correlation is a square of rows, not a triangle of pairs: the
    triangle does not fit the figure cap, and the rows do. Closes and returns
    never appear here.
    """
    kind = str(result.get("kind") or "")
    if kind == "cointegration":
        rows = []
        for row in result.get("rows") or []:
            if row.get("p") is None and not row.get("cointegrated"):
                continue
            rows.append(
                {
                    "a": row.get("a"),
                    "b": row.get("b"),
                    "p": row.get("p"),
                    "half_life": row.get("half_life"),
                    "cointegrated": bool(row.get("cointegrated")),
                }
            )
        truncated = len(rows) > _FIGURE_CAP
        names = [str(name) for name in (result.get("names") or [])]
        if len(names) > _FIGURE_CAP:
            names = names[:_FIGURE_CAP]
            truncated = True
        figures: dict[str, Any] = {
            "kind": "cointegration",
            "n_names": len(result.get("names") or []),
            "n_pairs": result.get("n_pairs"),
            "n_cointegrated": result.get("n_cointegrated"),
            "names": names,
            "pairs": rows[:_FIGURE_CAP],
            "truncated": truncated,
            "charts": [chart("rows", "pairs", "cointegrated pairs")],
        }
        if result.get("expected_false_positives") is not None:
            figures["expected_false_positives"] = result["expected_false_positives"]
        note = result.get("multiple_testing_note")
        if isinstance(note, str) and note:
            figures["multiple_testing_note"] = note
        return figures

    names = [str(name) for name in (result.get("names") or [])]
    values = [list(row) for row in (result.get("values") or [])]
    truncated = len(names) > _FIGURE_CAP
    if truncated:
        names = names[:_FIGURE_CAP]
        values = [list(row[:_FIGURE_CAP]) for row in values[:_FIGURE_CAP]]
    strongest: list[dict[str, Any]] = []
    kept = set(names)
    for pair in result.get("pairs") or []:
        if pair.get("a") not in kept or pair.get("b") not in kept:
            continue
        strongest.append({"a": pair["a"], "b": pair["b"], "v": pair["value"]})
        if len(strongest) >= _STRONG:
            break
    title = "correlation" if kind == "correlation" else "covariance"
    return {
        "kind": kind,
        "n_names": len(result.get("names") or []),
        "n_obs": result.get("n_obs"),
        "n_skipped": len(result.get("skipped") or []),
        "names": names,
        "matrix": values,
        "strongest": strongest,
        "truncated": truncated,
        "charts": [
            chart("matrix", "matrix", title),
            chart("rows", "strongest", "strongest pairs"),
        ],
    }
