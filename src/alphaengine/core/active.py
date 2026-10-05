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
    missing_vol = [name for name in names if name not in vol_map]
    for name in names:
        if name not in vol_map:
            continue
        score = float(scores[name][-1])
        alpha[name] = round(vol_map[name] * ic_f * score, _RND)
    abs_vals = [abs(v) for v in alpha.values()]
    note = None
    if not vol_map:
        note = "no volatility was supplied, so alpha is not reported as a return"
    elif missing_vol:
        note = f"{len(missing_vol)} names had no volatility and were left out of alpha"
    return {
        "alpha": alpha,
        "mean_abs_alpha": round(float(np.mean(abs_vals)), _RND) if abs_vals else None,
        "ic_used": round(ic_f, _RND),
        "n_names": len(alpha),
        "n_skipped": len(skipped) + (len(scores) - len(names)) + len(missing_vol),
        "skipped": skipped,
        "n_missing_vol": len(missing_vol),
        "note": note,
        "method": "grinold",
    }


def breadth_ir(
    *,
    ic: float,
    n_names: int,
    holdings: dict | None = None,
    ideal: dict | None = None,
    independent: bool = False,
    mean_correlation: float | None = None,
) -> dict[str, Any]:
    """Implied IR = TC * IC * sqrt(breadth). TC is 1 when holdings are absent.

    Breadth is the name count only when ``independent`` is set, or
    ``N / (1 + (N - 1) * rho)`` when ``mean_correlation`` is supplied.
    Omitting both leaves ``implied_ir`` empty rather than treating the names
    as independent bets.
    """
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
    breadth_used: float | None = None
    assumption = "not_assumed"
    ir: float | None = None
    if mean_correlation is not None and br:
        rho = float(mean_correlation)
        denom = 1.0 + (br - 1) * rho
        breadth_used = br / denom if denom > 0 else 0.0
        assumption = "mean_correlation"
        ir = tc * ic_f * math.sqrt(breadth_used) if breadth_used > 0 else 0.0
    elif independent:
        breadth_used = float(br)
        assumption = "independent"
        ir = tc * ic_f * math.sqrt(br) if br else 0.0
    return {
        "ic": round(ic_f, _RND),
        "breadth": br,
        "effective_breadth": None if breadth_used is None else round(breadth_used, _RND),
        "breadth_assumption": assumption,
        "mean_correlation": None if mean_correlation is None else round(float(mean_correlation), _RND),
        "transfer_coefficient": round(tc, _RND),
        "implied_ir": None if ir is None else round(ir, _RND),
        "method": "fundamental_law",
    }
