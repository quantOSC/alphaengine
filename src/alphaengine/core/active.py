"""Grinold-Kahn information analysis: alpha, breadth, transfer coefficient.

THE FUNDAMENTAL LAW, AS FIGURES. Alpha = volatility * IC * score on a
cross-section that has already been made comparable. Implied IR =
TC * IC * sqrt(breadth). The alpha VECTOR stays on this machine; the wire
gets the scalars a portal table can show.

Scores that are absent or non-finite are counted in `n_skipped`, never
silently dropped.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .panel import cs_zscore

__all__ = ["grinold_alpha", "breadth_ir"]

_RND = 6


def grinold_alpha(
    panel: dict,
    *,
    ic: float,
    vols: dict | None = None,
) -> dict[str, Any]:
    """Last-date alpha = vol * IC * z-score. Alpha vector is in `alpha`."""
    z = cs_zscore(panel)
    scores = z.get("panel") or {}
    skipped = list(z.get("skipped") or [])
    names = [n for n in scores if scores[n] and scores[n][-1] is not None]
    alpha: dict[str, float] = {}
    ic_f = float(ic)
    vol_map: dict[str, float] = {}
    if isinstance(vols, dict):
        for k, v in vols.items():
            try:
                vol_map[str(k)] = abs(float(v if not isinstance(v, (list, tuple)) else v[-1]))
            except (TypeError, ValueError):
                continue
    for name in names:
        score = float(scores[name][-1])
        vol = vol_map.get(name, 1.0)
        alpha[name] = round(vol * ic_f * score, _RND)
    abs_vals = [abs(v) for v in alpha.values()]
    return {
        "alpha": alpha,
        "mean_abs_alpha": round(float(np.mean(abs_vals)), _RND) if abs_vals else None,
        "ic_used": round(ic_f, _RND),
        "n_names": len(alpha),
        "n_skipped": len(skipped) + (len(scores) - len(names)),
        "skipped": skipped,
        "method": "grinold",
    }


def breadth_ir(
    *,
    ic: float,
    n_names: int,
    holdings: dict | None = None,
    ideal: dict | None = None,
) -> dict[str, Any]:
    """Implied IR = TC * IC * sqrt(breadth). TC is 1 when holdings are absent."""
    ic_f = float(ic)
    br = max(int(n_names), 0)
    tc = 1.0
    if isinstance(holdings, dict) and isinstance(ideal, dict) and holdings and ideal:
        names = sorted(set(map(str, holdings)) & set(map(str, ideal)))
        if len(names) >= 2:
            h = np.array([float(holdings[n]) for n in names], dtype=float)
            w = np.array([float(ideal[n]) for n in names], dtype=float)
            if h.std() > 0 and w.std() > 0:
                tc = float(np.corrcoef(h, w)[0, 1])
                if not math.isfinite(tc):
                    tc = 0.0
            else:
                tc = 0.0
        elif names:
            tc = 0.0
        else:
            tc = 0.0
    ir = tc * ic_f * math.sqrt(br) if br else 0.0
    return {
        "ic": round(ic_f, _RND),
        "breadth": br,
        "transfer_coefficient": round(tc, _RND),
        "implied_ir": round(ir, _RND),
        "method": "fundamental_law",
    }
