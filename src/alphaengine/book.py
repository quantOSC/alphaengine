"""A multi-strategy book: named sleeves, figures, never an OMS.

WHAT THIS IS
    The overlap/size/monitor questions, asked of a STACK of return series rather
    than one candidate vs one book series. A desk that runs six sleeves needs to
    know how a seventh sits on the whole stack, and whether any sleeve is
    already outside its lines.

WHAT THIS IS NOT
    Orders, shares, brokers, or target weights that become fills. `save_signals`
    still files ticker/rank/score/weight. A book here is a research object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .core.risk import compute_var_cvar
from .core.stress import overlap_stats
from .core.validation import min_track_record_length

__all__ = ["Book", "MAX_SLEEVES"]

#: A correlation matrix over 2000 names is a data export. Cap the book so the
#: figures that leave it stay figures.
MAX_SLEEVES = 16


def _as_returns(values: Any) -> list[float]:
    if isinstance(values, dict) and "returns" in values:
        values = values["returns"]
    return [float(x) for x in list(values)]


@dataclass
class Book:
    """Named sleeves of return series, plus optional current weights."""

    sleeves: dict[str, list[float]] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, returns: Any, *, weight: float | None = None) -> None:
        if len(self.sleeves) >= MAX_SLEEVES and name not in self.sleeves:
            raise ValueError(f"a book holds at most {MAX_SLEEVES} sleeves; remove one first")
        self.sleeves[str(name)] = _as_returns(returns)
        if weight is not None:
            self.weights[str(name)] = float(weight)

    def drop(self, name: str) -> None:
        self.sleeves.pop(name, None)
        self.weights.pop(name, None)

    @property
    def names(self) -> list[str]:
        return list(self.sleeves)

    def combined_returns(self) -> list[float]:
        """Equal-weight (or stated-weight) stack, aligned from the most recent end."""
        if not self.sleeves:
            return []
        series = {k: np.asarray(v, dtype=float) for k, v in self.sleeves.items()}
        n = min(int(a.size) for a in series.values())
        if n <= 0:
            return []
        stacked = np.column_stack([a[-n:] for a in series.values()])
        w = np.array([self.weights.get(k, 1.0) for k in series], dtype=float)
        if float(w.sum()) == 0:
            w = np.ones_like(w)
        w = w / w.sum()
        return [float(x) for x in (stacked @ w).tolist()]

    def overlap_matrix(
        self, candidate: Any | None = None, *, candidate_name: str = "candidate"
    ) -> dict[str, Any]:
        """Pairwise correlation/beta, bounded. Candidate optional."""
        names = list(self.sleeves)
        series = dict(self.sleeves)
        if candidate is not None:
            series[candidate_name] = _as_returns(candidate)
            names = [candidate_name, *names]
        names = names[:MAX_SLEEVES]
        pairs: list[dict[str, Any]] = []
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                stats = overlap_stats(series[a], series[b])
                pairs.append(
                    {
                        "a": a,
                        "b": b,
                        "correlation": stats.get("correlation"),
                        "beta_to_book": stats.get("beta_to_book"),
                        "n_obs": stats.get("n_obs"),
                    }
                )
        return {
            "n_sleeves": len(names),
            "names": names,
            "pairs": pairs,
        }

    def residual_weight(self, name: str, *, target: float = 1.0) -> dict[str, Any]:
        """How much of `target` is left after the sleeves already held.

        Refuses rather than sizing small when the candidate record is shorter
        than MinTRL — a small weight is still a claim.
        """
        if name not in self.sleeves:
            raise KeyError(name)
        returns = self.sleeves[name]
        mintrl = min_track_record_length(returns)
        if mintrl.get("error") or not mintrl.get("sufficient"):
            return {
                "name": name,
                "refused": True,
                "reason": "record_too_short",
                "min_track_record_length": mintrl,
            }
        held = sum(self.weights.get(k, 0.0) for k in self.sleeves if k != name)
        residual = max(0.0, float(target) - held)
        risk = compute_var_cvar(returns)
        return {
            "name": name,
            "refused": False,
            "held_weight": round(held, 6),
            "residual_weight": round(residual, 6),
            "cvar": risk.get("cvar"),
            "min_track_record_length": mintrl,
        }

    def monitor(self) -> dict[str, Any]:
        """Per-sleeve status. Nothing checked never reads as all-clear."""
        rows: list[dict[str, Any]] = []
        for name, returns in self.sleeves.items():
            if len(returns) < 2:
                rows.append({"name": name, "status": "unchecked", "n_obs": len(returns)})
                continue
            risk = compute_var_cvar(returns)
            mintrl = min_track_record_length(returns)
            status = "ok" if mintrl.get("sufficient") else "undetermined"
            rows.append(
                {
                    "name": name,
                    "status": status,
                    "n_obs": len(returns),
                    "cvar": risk.get("cvar"),
                    "sufficient": mintrl.get("sufficient"),
                }
            )
        if not rows or any(r["status"] == "unchecked" for r in rows):
            overall = "unchecked"
        elif any(r["status"] != "ok" for r in rows):
            overall = "undetermined"
        else:
            overall = "ok"
        return {"overall": overall, "sleeves": rows, "n_sleeves": len(rows)}
