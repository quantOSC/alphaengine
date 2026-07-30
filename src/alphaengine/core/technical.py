"""
technical.py, price-derived technical features on a SUPPLIED price series.

Pure deterministic math (no fetch, no LLM, no randomness): Wilder RSI, SMA/EMA,
distance-from-MA, and ATR, with interpretive flags. Prices are caller-supplied
(model "b": the engine never fetches prices). A feature that needs more data than
supplied (RSI/EMA below their window, ATR with no high/low) is reported as
absent-with-a-reason, never guessed. Coverage is a first-class output.

Math conventions (so a caller can audit):
  - SMA(w)  = mean of the trailing w closes.
  - EMA(w)  = seeded with SMA of the first w closes, then alpha=2/(w+1) recursion.
  - RSI(w)  = Wilder: seed avg gain/loss = simple mean of the first w deltas, then
              avg = (prev*(w-1) + current)/w. RS = avg_gain/avg_loss;
              RSI = 100 - 100/(1+RS). All-up -> 100, all-down -> 0, flat -> 50.
  - ATR(w)  = Wilder smoothing of True Range; needs high/low (OHLC rows).
  - distance_pct = (last_close - MA) / MA * 100.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_SMA_WINDOWS = [50, 150, 200]
RSI_OVERBOUGHT = 70.0
RSI_OVERSOLD = 30.0
_RND = 6


def _unavailable(reason: str, **details) -> dict:
    """Local capability-error shape (kept identical in the backend copy for parity)."""
    return {"available": False, "reason": reason, **details}


def _extract(series: Any) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Return (closes, highs, lows). Accepts bare closes, {date: close}, or a list
    of OHLC row dicts. highs/lows are None unless OHLC rows carry them."""
    if isinstance(series, dict):
        dates = sorted(series.keys())
        return np.asarray([float(series[d]) for d in dates], dtype=float), None, None
    if isinstance(series, list) and series and isinstance(series[0], dict):
        rows = series
        if all("date" in r for r in rows):
            rows = sorted(rows, key=lambda r: r["date"])
        closes = np.asarray([float(r["close"]) for r in rows], dtype=float)
        if all(("high" in r and "low" in r) for r in rows):
            highs = np.asarray([float(r["high"]) for r in rows], dtype=float)
            lows = np.asarray([float(r["low"]) for r in rows], dtype=float)
            return closes, highs, lows
        return closes, None, None
    closes = np.asarray([float(x) for x in (series or []) if x is not None], dtype=float)
    return closes, None, None


def _sma(closes: np.ndarray, window: int) -> float | None:
    if len(closes) < window:
        return None
    return float(closes[-window:].mean())


def _ema(closes: np.ndarray, window: int) -> float | None:
    if len(closes) < window:
        return None
    alpha = 2.0 / (window + 1)
    ema = float(closes[:window].mean())
    for c in closes[window:]:
        ema = alpha * float(c) + (1 - alpha) * ema
    return ema


def _wilder_rsi(closes: np.ndarray, window: int) -> float | None:
    if len(closes) < window + 1:
        return None
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(gains[:window].mean())
    avg_loss = float(losses[:window].mean())
    for i in range(window, len(deltas)):
        avg_gain = (avg_gain * (window - 1) + float(gains[i])) / window
        avg_loss = (avg_loss * (window - 1) + float(losses[i])) / window
    if avg_loss == 0 and avg_gain == 0:
        return 50.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(highs: np.ndarray | None, lows: np.ndarray | None, closes: np.ndarray, window: int) -> float | None:
    if highs is None or lows is None or len(closes) < window + 1:
        return None
    n = len(closes)
    tr = np.empty(n - 1)
    for i in range(1, n):
        tr[i - 1] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
    atr = float(tr[:window].mean())
    for i in range(window, len(tr)):
        atr = (atr * (window - 1) + float(tr[i])) / window
    return atr


def _ma_block(last: float, value: float | None) -> dict | None:
    if value is None:
        return None
    return {
        "value": round(value, _RND),
        "above": bool(last > value),
        "distance_pct": round((last - value) / value * 100.0, 4) if value != 0 else None,
    }


def technical_features(
    prices: dict,
    *,
    sma_windows: list[int] | None = None,
    ema_windows: list[int] | None = None,
    rsi_window: int = 14,
    atr_window: int = 14,
) -> dict:
    """Per-symbol technical features over supplied prices. See module docstring for
    the math. Returns per-symbol values + interpretive flags, plus coverage."""
    sma_windows = list(sma_windows) if sma_windows else list(DEFAULT_SMA_WINDOWS)
    ema_windows = list(ema_windows) if ema_windows else []

    features: dict[str, dict] = {}
    coverage_n = 0
    for sym, series in prices.items():
        closes, highs, lows = _extract(series)
        n = len(closes)
        if n == 0:
            features[sym] = {
                "n_observations": 0,
                "last_close": None,
                "insufficient": ["no_data"],
                "flags": [],
            }
            continue
        coverage_n += 1
        last = float(closes[-1])
        res: dict[str, Any] = {
            "n_observations": n,
            "last_close": round(last, _RND),
            "sma": {},
            "ema": {},
            "flags": [],
            "insufficient": [],
        }

        for w in sma_windows:
            block = _ma_block(last, _sma(closes, w))
            res["sma"][str(w)] = block
            if block is None:
                res["insufficient"].append(f"sma_{w}")
            else:
                res["flags"].append(f"{'above' if block['above'] else 'below'}_sma_{w}")
        for w in ema_windows:
            block = _ma_block(last, _ema(closes, w))
            res["ema"][str(w)] = block
            if block is None:
                res["insufficient"].append(f"ema_{w}")
            else:
                res["flags"].append(f"{'above' if block['above'] else 'below'}_ema_{w}")

        rsi = _wilder_rsi(closes, rsi_window)
        if rsi is None:
            res["rsi"] = None
            res["insufficient"].append(f"rsi_{rsi_window}")
        else:
            ob, osd = rsi > RSI_OVERBOUGHT, rsi < RSI_OVERSOLD
            res["rsi"] = {
                "window": rsi_window,
                "value": round(rsi, 4),
                "overbought": bool(ob),
                "oversold": bool(osd),
            }
            if ob:
                res["flags"].append("rsi_overbought")
            if osd:
                res["flags"].append("rsi_oversold")

        atr = _atr(highs, lows, closes, atr_window)
        if atr is None:
            res["atr"] = (
                _unavailable("ATR needs high/low; supply OHLC rows", atr_window=atr_window)
                if highs is None
                else _unavailable(f"need >= {atr_window + 1} bars", atr_window=atr_window)
            )
        else:
            res["atr"] = {
                "window": atr_window,
                "value": round(atr, _RND),
                "pct_of_close": round(atr / last * 100.0, 4) if last != 0 else None,
            }

        features[sym] = res

    return {
        "features": features,
        "params": {
            "sma_windows": sma_windows,
            "ema_windows": ema_windows,
            "rsi_window": rsi_window,
            "atr_window": atr_window,
        },
        "universe_size": len(prices),
        "coverage_n": coverage_n,
    }


__all__ = ["technical_features"]
