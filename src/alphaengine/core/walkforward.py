"""Walk-forward: rolling IS/OOS refit of the caller's own `backtest_fn`.

NOT a silent parameter search the library invents. Each window runs the grid
the caller declared; the trial count is `n_windows * len(grid)`, derived, never
asserted. Failed windows occupy a slot so the denominator does not quietly
shrink.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np

from ..sweep.runner import _expand, _hash_data, _sharpe, sweep

__all__ = ["walk_forward"]

BacktestFn = Callable[..., Any]


def _len_data(data: Any) -> int:
    if data is None:
        return 0
    if isinstance(data, dict) and "close" in data:
        return len(data["close"])
    if isinstance(data, dict):
        lengths = []
        for v in data.values():
            try:
                lengths.append(len(v))
            except TypeError:
                continue
        return min(lengths) if lengths else 0
    try:
        return len(data)
    except TypeError:
        return 0


def _slice_data(data: Any, start: int, end: int) -> Any:
    if isinstance(data, dict) and "close" in data:
        out = dict(data)
        out["close"] = list(data["close"])[start:end]
        if "returns" in data:
            out["returns"] = list(data["returns"])[start:end]
        return out
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            try:
                out[k] = list(v)[start:end]
            except TypeError:
                out[k] = v
        return out
    return list(data)[start:end]


def walk_forward(
    backtest_fn: BacktestFn,
    grid: Mapping[str, Iterable[Any]],
    *,
    data: Any,
    n_windows: int = 4,
    oos_fraction: float = 0.25,
    on_error: str = "record",
) -> dict[str, Any]:
    """Expanding IS, trailing OOS, one sweep per window.

    Args:
        backtest_fn: YOUR simulator, same contract as `sweep`.
        grid: the search. Its length is the per-window trial count.
        data: sliced by position. Dicts with `close` keep that shape.
        n_windows: how many IS/OOS folds. Derived into the trial count.
        oos_fraction: fraction of each window held out, most-recent end.
        on_error: "record" keeps a failed window in the denominator.

    Returns:
        Figures: per-window OOS Sharpe, derived n_trials, mean OOS Sharpe.
        The IS trial matrices stay local (not returned).
    """
    combos = _expand(grid)
    if not combos:
        raise ValueError("grid is empty: nothing to walk forward")
    n_windows = max(2, int(n_windows))
    n = _len_data(data)
    if n < n_windows * 8:
        raise ValueError(f"walk-forward needs a longer record: {n} observations for {n_windows} windows")

    # Expanding windows: window i uses data[:cut_i], OOS is the last oos_fraction.
    windows: list[dict[str, Any]] = []
    for i in range(n_windows):
        cut = int(round((i + 1) * n / n_windows))
        if cut < 16:
            windows.append({"index": i, "failed": "window_too_short", "n_obs": cut})
            continue
        oos_n = max(4, int(round(cut * oos_fraction)))
        is_end = cut - oos_n
        if is_end < 8:
            windows.append({"index": i, "failed": "is_too_short", "n_obs": cut})
            continue
        try:
            is_data = _slice_data(data, 0, is_end)
            oos_data = _slice_data(data, is_end, cut)
            result = sweep(backtest_fn, grid, data=is_data, on_error=on_error)
            best = result.best.params
            raw = backtest_fn(data=oos_data, **best) if oos_data is not None else backtest_fn(**best)
            oos = np.asarray(list(raw), dtype=float)
            sr = _sharpe(oos) if oos.size else 0.0
            windows.append(
                {
                    "index": i,
                    "n_is": is_end,
                    "n_oos": int(oos.size),
                    "n_trials_is": result.n_trials,
                    "oos_sharpe": round(float(sr), 6),
                    "failed": None,
                }
            )
        except Exception as exc:  # noqa: BLE001 - occupy the slot
            if on_error == "raise":
                raise
            windows.append({"index": i, "failed": f"{type(exc).__name__}: {exc}"})

    n_grid = len(combos)
    n_trials = n_windows * n_grid
    oos_ok = [w["oos_sharpe"] for w in windows if w.get("failed") is None and "oos_sharpe" in w]
    return {
        "n_windows": n_windows,
        "n_grid": n_grid,
        "n_trials": n_trials,
        "n_trials_source": "derived_from_walk_forward",
        "n_windows_ok": len(oos_ok),
        "n_windows_failed": n_windows - len(oos_ok),
        "mean_oos_sharpe": round(float(np.mean(oos_ok)), 6) if oos_ok else None,
        "windows": windows,
        "data_hash": _hash_data(data) if data is not None else "",
    }
