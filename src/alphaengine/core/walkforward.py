"""Walk-forward: rolling IS/OOS refit of the caller's own `backtest_fn`.

NOT a silent parameter search the library invents. Each window runs the grid
the caller declared; the trial count is `n_windows * len(grid)`, derived, never
asserted. Failed windows occupy a slot so the denominator does not quietly
shrink. `purge_bars` leaves an explicit causal gap `[is_end, oos_start)` between
the in-sample fit slice and the out-of-sample evaluation slice (`QOS-151`).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np

from .._version import current_engine_version
from ..sweep.runner import _expand, _hash_data, _sharpe, sweep
from .conventions import (
    build_annualization_spec,
    build_costs_model_spec,
    build_risk_free_spec,
)

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


def _notify_window(
    cb: Callable[..., Any] | None,
    idx: int,
    total: int,
    win: dict[str, Any],
) -> None:
    if cb is None:
        return
    try:
        sig = inspect.signature(cb)
        pos_params = [
            p
            for p in sig.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
        if len(pos_params) >= 3:
            cb(idx, total, win)
            return
        if len(pos_params) == 2:
            cb(idx, win)
            return
        if len(pos_params) == 1 or has_varargs:
            cb(win)
            return
    except (TypeError, ValueError):
        pass
    cb(win)


def walk_forward(
    backtest_fn: BacktestFn,
    grid: Mapping[str, Iterable[Any]],
    *,
    data: Any,
    n_windows: int = 4,
    oos_fraction: float = 0.25,
    purge_bars: int = 0,
    on_window: Callable[..., Any] | None = None,
    on_error: str = "record",
    risk_free_rate: float | None = None,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Expanding IS, purged gap, trailing OOS, one sweep per window.

    Args:
        backtest_fn: YOUR simulator, same contract as `sweep`.
        grid: the search. Its length is the per-window trial count.
        data: sliced by position. Dicts with `close` keep that shape.
        n_windows: how many IS/OOS folds. Derived into the trial count.
        oos_fraction: fraction of each window held out, most-recent end.
        purge_bars: bars excluded between IS end and OOS start (`>= 0`).
        on_window: optional progress callback invoked as each window finishes.
        on_error: "record" keeps a failed window in the denominator.
        risk_free_rate: annual cash rate (defaults to pinned DGS3MO).
        periods_per_year: trading periods per year (default 252).

    Returns:
        Figures: per-window OOS Sharpe, derived n_trials, mean OOS Sharpe,
        purge_bars, engine_version, and stated conventions.
    """
    combos = _expand(grid)
    if not combos:
        raise ValueError("grid is empty: nothing to walk forward")
    if int(purge_bars) < 0:
        raise ValueError(f"purge_bars must be >= 0 (got {purge_bars!r})")
    purge = int(purge_bars)
    ppy = max(int(periods_per_year), 1)
    n_windows = max(2, int(n_windows))
    n = _len_data(data)
    if n < n_windows * 8:
        raise ValueError(f"walk-forward needs a longer record: {n} observations for {n_windows} windows")

    # Expanding windows: window i uses data[:is_end] for IS, skips [is_end:oos_start]
    # (`purge_bars`), and evaluates OOS on data[oos_start:cut].
    windows: list[dict[str, Any]] = []
    for i in range(n_windows):
        cut = int(round((i + 1) * n / n_windows))
        if cut < 16:
            win_short = {"index": i, "failed": "window_too_short", "n_obs": cut, "purge_bars": purge}
            windows.append(win_short)
            _notify_window(on_window, i, n_windows, win_short)
            continue
        oos_n = max(4, int(round(cut * oos_fraction)))
        oos_start = cut - oos_n
        is_end = oos_start - purge
        if is_end < 8:
            win_is_short = {
                "index": i,
                "failed": "is_too_short_after_purge" if purge > 0 else "is_too_short",
                "n_obs": cut,
                "purge_bars": purge,
            }
            windows.append(win_is_short)
            _notify_window(on_window, i, n_windows, win_is_short)
            continue
        try:
            is_data = _slice_data(data, 0, is_end)
            oos_data = _slice_data(data, oos_start, cut)
            # The enclosing walk-forward reports n_windows * n_grid trials once.
            # Letting each window's sweep auto-record into a project ledger would
            # count the same attempts twice (once per window run, once here).
            from ..ledger import suppress_auto_sweep_record

            with suppress_auto_sweep_record():
                result = sweep(
                    backtest_fn,
                    grid,
                    data=is_data,
                    on_error=on_error,
                    risk_free_rate=risk_free_rate,
                    periods_per_year=ppy,
                )
            best = result.best.params
            raw = backtest_fn(data=oos_data, **best) if oos_data is not None else backtest_fn(**best)
            oos = np.asarray(list(raw), dtype=float)
            sr = _sharpe(oos, risk_free_rate=risk_free_rate, periods_per_year=ppy) if oos.size else 0.0
            win_ok = {
                "index": i,
                "is_end": is_end,
                "oos_start": oos_start,
                "purge_bars": purge,
                "n_is": is_end,
                "n_oos": int(oos.size),
                "n_trials_is": result.n_trials,
                "oos_sharpe": round(float(sr), 6),
                "failed": None,
            }
            windows.append(win_ok)
            _notify_window(on_window, i, n_windows, win_ok)
        except Exception as exc:  # noqa: BLE001 - occupy the slot
            if on_error == "raise":
                raise
            win_err = {
                "index": i,
                "purge_bars": purge,
                "failed": f"{type(exc).__name__}: {exc}",
            }
            windows.append(win_err)
            _notify_window(on_window, i, n_windows, win_err)

    n_grid = len(combos)
    n_trials = n_windows * n_grid
    oos_ok = [w["oos_sharpe"] for w in windows if w.get("failed") is None and "oos_sharpe" in w]
    return {
        "engine_version": current_engine_version(),
        "n_windows": n_windows,
        "n_grid": n_grid,
        "purge_bars": purge,
        "n_trials": n_trials,
        "n_trials_source": "derived_from_walk_forward",
        "n_windows_ok": len(oos_ok),
        "n_windows_failed": n_windows - len(oos_ok),
        "mean_oos_sharpe": round(float(np.mean(oos_ok)), 6) if oos_ok else None,
        "windows": windows,
        "data_hash": _hash_data(data) if data is not None else "",
        "annualization": build_annualization_spec(ppy),
        "risk_free": build_risk_free_spec(risk_free_rate, ppy),
        "costs_model": build_costs_model_spec(mode="user_backtest_fn"),
    }
