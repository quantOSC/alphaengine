"""A portfolio from the returns already loaded, in front of what the session recorded.

The weights are hierarchical risk parity, or risk parity when that is what
was asked. Both read a Ledoit-Wolf covariance of the loaded returns. Session
notes do not get turned into a second, quieter optimiser: they travel with
the result so a later question can see what was already known when the book
was built.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..charts import chart
from .allocate import hrp_weights, risk_parity_weights
from .covariance import ledoit_wolf_cov
from .relations import as_returns

__all__ = ["build_portfolio", "portfolio_figures"]

_CAP = 512


def build_portfolio(
    data: Any,
    *,
    method: str = "hrp",
    notes: list[str] | None = None,
    names: list[str] | None = None,
) -> dict[str, Any]:
    """Weights over the loaded names. `names` limits the book to a set already measured."""
    panel = as_returns(data)
    if names:
        wanted = {str(name) for name in names}
        panel = {name: series for name, series in panel.items() if name in wanted}
    if len(panel) < 2:
        raise ValueError("a portfolio needs at least two loaded names with a shared history.")
    from .panel import align_panel

    aligned, kept, skipped = align_panel(panel, need=2)
    if len(kept) < 2:
        raise ValueError("the loaded names do not share enough history to form a portfolio.")
    cov, shrinkage, _mu = ledoit_wolf_cov(np.asarray(aligned, dtype=float))
    chosen = "risk_parity" if method == "risk_parity" else "hrp"
    weights = (
        risk_parity_weights(cov, names=kept) if chosen == "risk_parity" else hrp_weights(cov, names=kept)
    )
    session = _session_heads(notes or [])
    rows = sorted(weights["weights"].items(), key=lambda item: float(item[1]), reverse=True)
    return {
        "kind": "portfolio",
        "method": chosen,
        "covariance": "ledoit_wolf",
        "shrinkage": round(float(shrinkage), 4),
        "names": kept,
        "weights": weights["weights"],
        "rows": [{"name": name, "weight": weight} for name, weight in rows],
        "n_assets": int(weights["n_assets"]),
        "n_obs": int(aligned.shape[0]),
        "n_skipped": len(skipped),
        "weight_sum": weights["weight_sum"],
        "max_weight": weights["max_weight"],
        "min_weight": weights["min_weight"],
        "session": session,
    }


def portfolio_figures(result: dict[str, Any]) -> dict[str, Any]:
    """Portal rows. Names and weights, never the return series they came from."""
    rows = list(result.get("rows") or [])
    truncated = len(rows) > _CAP
    return {
        "kind": "portfolio",
        "method": result.get("method"),
        "covariance": result.get("covariance"),
        "shrinkage": result.get("shrinkage"),
        "n_names": result.get("n_assets"),
        "n_obs": result.get("n_obs"),
        "n_skipped": result.get("n_skipped"),
        "weight_sum": result.get("weight_sum"),
        "max_weight": result.get("max_weight"),
        "weights": rows[:_CAP],
        "session": list(result.get("session") or [])[:_CAP],
        "truncated": truncated,
        "charts": [chart("rows", "weights", f"{result.get('method') or 'portfolio'} weights")],
    }


def _session_heads(notes: list[str]) -> list[str]:
    heads: list[str] = []
    for note in notes:
        head = str(note).split(":", 1)[0].strip()
        if head and head not in heads:
            heads.append(head)
    return heads[:_CAP]
