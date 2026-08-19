"""Cross-sectional panel transforms: rank, z-score, winsorize, neutralize.

THE MODELING MORNING, BEFORE A SIMULATOR EXISTS. A raw factor is not comparable
across names until it has been ranked or standardized, and it is not a bet
until the part explained by size or the market has been taken out. These are
that step: one date at a time, across the names that are actually there.

ALIGNMENT IS BY COMMON TAIL, AND SKIPS ARE NAMED
    Series arrive as `{symbol: values}`. Names that are absent, non-numeric, or
    shorter than the rest of the panel are COUNTED AND NAMED, never silently
    dropped — a z-score over 40 names is a different measurement from one over
    400 with 360 discarded.

NaNs INSIDE a usable series stay NaNs. A name that makes the panel is not
dropped because one date is missing; that date is skipped for that name.

NEUTRALIZE
    Residualize each date on supplied controls plus an intercept, via numpy
    `lstsq`. No controls means demean (the intercept only). Dates with too few
    finite names to identify the regression are counted, not imputed.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .series_shapes import series_values

__all__ = [
    "cs_rank",
    "cs_zscore",
    "cs_winsorize",
    "neutralize",
    "newey_west_tstat",
    "align_panel",
    "PanelShapeError",
]

_RND = 6
MAX_NAMED = 512
MIN_CS = 3


class PanelShapeError(ValueError):
    """The panel is not a mapping of symbol -> numeric series."""


def _series(values: Any) -> np.ndarray | None:
    if isinstance(values, dict) or (
        isinstance(values, (list, tuple)) and len(values) and isinstance(values[0], dict)
    ):
        arr = series_values(values)[0]
        return arr if arr.ndim == 1 and arr.size else None
    if isinstance(values, (list, tuple, np.ndarray)):
        try:
            arr = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            return None
        return arr if arr.ndim == 1 and arr.size else None
    return None


def align_panel(panel: dict, *, need: int = 1) -> tuple[np.ndarray, list[str], list[str]]:
    """Symbols with a usable series, tail-aligned. Returns (T x N, names, skipped)."""
    if not isinstance(panel, dict):
        raise PanelShapeError("a panel is {symbol: values} — a mapping of names to series.")
    usable: list[tuple[str, np.ndarray]] = []
    skipped: list[str] = []
    for name in sorted(map(str, panel)):
        s = _series(panel.get(name))
        if s is None or len(s) < need:
            skipped.append(name)
            continue
        usable.append((name, s))
    if not usable:
        return np.empty((0, 0)), [], skipped
    # Depth is the longest usable series. Shorter names are skipped and named,
    # not allowed to shrink the rest of the panel.
    depth = max(len(s) for _, s in usable)
    names: list[str] = []
    cols: list[np.ndarray] = []
    for name, s in usable:
        if len(s) < depth:
            skipped.append(name)
            continue
        names.append(name)
        cols.append(s[-depth:])
    if not names:
        return np.empty((0, 0)), [], skipped
    M = np.column_stack(cols)
    return M, names, skipped


def _rank_1d(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _row_rank(row: np.ndarray) -> np.ndarray:
    out = np.full(row.shape, np.nan)
    ok = np.isfinite(row)
    if int(ok.sum()) < 2:
        return out
    out[ok] = _rank_1d(row[ok])
    return out


def _row_zscore(row: np.ndarray) -> np.ndarray:
    out = np.full(row.shape, np.nan)
    ok = np.isfinite(row)
    n = int(ok.sum())
    if n < MIN_CS:
        return out
    x = row[ok]
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    if sd == 0 or not math.isfinite(sd):
        return out
    out[ok] = (x - x.mean()) / sd
    return out


def _row_winsor(row: np.ndarray, lower: float, upper: float) -> np.ndarray:
    out = np.full(row.shape, np.nan)
    ok = np.isfinite(row)
    n = int(ok.sum())
    if n < MIN_CS:
        return out
    x = row[ok]
    lo, hi = np.quantile(x, [lower, upper])
    out[ok] = np.clip(x, lo, hi)
    return out


def _pack(M: np.ndarray, names: list[str], skipped: list[str], method: str) -> dict[str, Any]:
    panel = {
        name: [None if not math.isfinite(v) else round(float(v), _RND) for v in M[:, j]]
        for j, name in enumerate(names)
    }
    return {
        "panel": panel,
        "n_names": len(names),
        "n_skipped": len(skipped),
        "skipped": skipped[:MAX_NAMED],
        "n_dates": int(M.shape[0]),
        "method": method,
    }


def cs_rank(panel: dict) -> dict[str, Any]:
    """Cross-sectional rank of each date. Ties take argsort order, matching IC."""
    M, names, skipped = align_panel(panel)
    if not names:
        return _pack(M, names, skipped, "rank")
    ranked = np.vstack([_row_rank(M[t]) for t in range(M.shape[0])])
    return _pack(ranked, names, skipped, "rank")


def cs_zscore(panel: dict) -> dict[str, Any]:
    """Cross-sectional z-score of each date. Constant rows stay all-NaN, not zero."""
    M, names, skipped = align_panel(panel)
    if not names:
        return _pack(M, names, skipped, "zscore")
    z = np.vstack([_row_zscore(M[t]) for t in range(M.shape[0])])
    return _pack(z, names, skipped, "zscore")


def cs_winsorize(panel: dict, *, lower: float = 0.01, upper: float = 0.99) -> dict[str, Any]:
    """Clip each date at cross-sectional quantiles. Bounds are in [0, 1]."""
    lo = min(max(float(lower), 0.0), 1.0)
    hi = min(max(float(upper), 0.0), 1.0)
    if hi < lo:
        lo, hi = hi, lo
    M, names, skipped = align_panel(panel)
    if not names:
        return _pack(M, names, skipped, "winsorize")
    w = np.vstack([_row_winsor(M[t], lo, hi) for t in range(M.shape[0])])
    out = _pack(w, names, skipped, "winsorize")
    out["lower"] = lo
    out["upper"] = hi
    return out


def _as_control_panels(controls: Any) -> dict[str, dict]:
    """Normalize controls to {control_name: {symbol: series}}."""
    if not controls:
        return {}
    if not isinstance(controls, dict):
        raise PanelShapeError("controls are {name: panel} or one {symbol: series} panel.")
    values = list(controls.values())
    if values and isinstance(values[0], dict) and not _looks_like_series(values[0]):
        return {str(k): v for k, v in controls.items() if isinstance(v, dict)}
    return {"control": controls}


def _looks_like_series(obj: dict) -> bool:
    """True when `obj` is {symbol: numeric series}, not {factor: panel}."""
    for v in obj.values():
        nested = (
            isinstance(v, dict) and v and isinstance(next(iter(v.values())), (list, tuple, np.ndarray, dict))
        )
        return not nested
    return True


def neutralize(panel: dict, controls: Any = None) -> dict[str, Any]:
    """Residualize each date on controls plus an intercept.

    `controls=None` is cross-sectional demeaning. A date with fewer finite
    names than regressors is skipped and counted; names too short to join the
    panel are in `skipped`.
    """
    M, names, skipped = align_panel(panel)
    n_dates_skipped = 0
    if not names:
        out = _pack(M, names, skipped, "neutralize")
        out["n_dates_skipped"] = 0
        out["n_controls"] = 0
        return out

    ctrl_panels = _as_control_panels(controls)
    ctrl_mats: list[np.ndarray] = []
    for spec in ctrl_panels.values():
        C, cnames, _ = align_panel(spec, need=1)
        if not cnames:
            continue
        idx = {n: i for i, n in enumerate(cnames)}
        col = np.full((M.shape[0], M.shape[1]), np.nan)
        depth = min(M.shape[0], C.shape[0])
        for j, name in enumerate(names):
            if name in idx:
                col[-depth:, j] = C[-depth:, idx[name]]
        ctrl_mats.append(col)

    k = 1 + len(ctrl_mats)
    resid = np.full_like(M, np.nan, dtype=float)
    for t in range(M.shape[0]):
        y = M[t]
        ok = np.isfinite(y)
        Xcols = [np.ones(M.shape[1])]
        for C in ctrl_mats:
            ok = ok & np.isfinite(C[t])
            Xcols.append(C[t])
        if int(ok.sum()) <= k:
            n_dates_skipped += 1
            continue
        y_ok = y[ok]
        X = np.column_stack([col[ok] for col in Xcols])
        beta, *_ = np.linalg.lstsq(X, y_ok, rcond=None)
        fitted = X @ beta
        row = np.full(M.shape[1], np.nan)
        row[ok] = y_ok - fitted
        resid[t] = row

    out = _pack(resid, names, skipped, "neutralize")
    out["n_dates_skipped"] = n_dates_skipped
    out["n_controls"] = len(ctrl_mats)
    return out


def newey_west_tstat(values: Any, *, lags: int = 1) -> float | None:
    """t-stat of the mean with Newey-West long-run variance. None when undefined.

    `lags` is the Bartlett kernel bandwidth. Non-overlapping ICs should pass 1.
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n < 2:
        return None
    mu = float(arr.mean())
    e = arr - mu
    gamma0 = float(np.dot(e, e) / n)
    lrv = gamma0
    L = max(0, min(int(lags), n - 1))
    for k in range(1, L + 1):
        gamma_k = float(np.dot(e[k:], e[:-k]) / n)
        weight = 1.0 - k / (L + 1)
        lrv += 2.0 * weight * gamma_k
    if lrv <= 0 or not math.isfinite(lrv):
        return None
    se = math.sqrt(lrv / n)
    if se == 0 or not math.isfinite(se):
        return None
    return float(mu / se)
