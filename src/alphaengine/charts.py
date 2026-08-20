"""Frozen chart vocabulary the portal draws from figures.

A FIGURE IS A NAMED KIND, not a series. The executor attaches
`charts: [{kind, key, title}]` hints. The portal maps kind+key to a component.
Old portals ignore `charts` and still have the raw keys (`ic_by_period`,
`best_curve`, ...).

KINDS
    curve     {i, v}              line
    period    {p, ic|lambda}      CS time series
    band      {i, lo, mid, hi}    fan / Monte Carlo quantile
    hist      {edges, counts}     histogram
    cost      {bps, sharpe}       cost ladder
    scatter   {x, y}              overlap
    triangle  {a, b, v}           covariance heatmap
    rows      bounded dict rows   table
    quantiles [float]             bar

No hint may carry a nested series. Titles are one line, no em dash.
"""

from __future__ import annotations

__all__ = ["CHART_KINDS", "chart", "charts"]

CHART_KINDS = frozenset(
    {
        "curve",
        "period",
        "band",
        "hist",
        "cost",
        "scatter",
        "triangle",
        "rows",
        "quantiles",
    }
)


def chart(kind: str, key: str, title: str) -> dict[str, str]:
    """One portal hint. `kind` must be in CHART_KINDS."""
    k = str(kind)
    if k not in CHART_KINDS:
        raise ValueError(f"unknown chart kind {k!r}; expected one of {sorted(CHART_KINDS)}")
    return {"kind": k, "key": str(key), "title": str(title)}


def charts(*items: dict[str, str]) -> list[dict[str, str]]:
    """Hint list. Empty is allowed: the portal then uses raw keys only."""
    out: list[dict[str, str]] = []
    for item in items:
        if not item:
            continue
        out.append(chart(item["kind"], item["key"], item["title"]) if "kind" in item else dict(item))
    return out
