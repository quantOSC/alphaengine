"""
backtest.py, deterministic signal+price backtest.
PURE: signals + prices in, fills simulated with slippage, commission, short
borrow, causal ADV market impact/participation caps, and final residual
liquidation costs in equity -> equity curve + per-trade log out.
No fetch, no LLM, no clock.

Input:
  signals : {ticker: [{date, target_weight | action, conviction?}...]}
            target_weight is a signed portfolio weight (negative = short),
            forward-filled until changed; `action` maps to a weight
            (enter_long/long/buy=+1, enter_short/short=-1, exit/flat/close=0,
            hold=carry prior). Weights are clamped to ±max_position_pct.
  prices  : {ticker: [{date, close, open?}...]}  (or {date: close} / bare closes)
  config  : slippage_bps, commission_bps, borrow_bps, fill_timing ('close'|'next_open'),
            initial_capital, max_position_pct, adv, adv_series, liquidate_final

No-look-ahead: a signal dated bar i is actioned at bar i's close ('close') or
bar i+1's open ('next_open'), never at a price or ADV reading that printed
after the signal reference bar.
"""

from __future__ import annotations

import bisect
import contextlib
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .._version import current_engine_version
from .conventions import (
    build_annualization_spec,
    build_costs_model_spec,
    build_risk_free_spec,
    verdict_from_dsr,
)
from .trace import record

MIN_BARS = 2

_NOTEBOOK_HOOK_CALLBACK: Callable[[dict[str, Any]], None] | None = None


def register_notebook_hook(callback: Callable[[dict[str, Any]], None] | None) -> None:
    """Register an active notebook session hook callback (QOS-181)."""
    global _NOTEBOOK_HOOK_CALLBACK
    _NOTEBOOK_HOOK_CALLBACK = callback


# action keyword -> target weight; None means "hold" (carry the prior weight).
_ACTION_WEIGHTS = {
    "enter_long": 1.0,
    "long": 1.0,
    "buy": 1.0,
    "enter_short": -1.0,
    "short": -1.0,
    "sell_short": -1.0,
    "exit": 0.0,
    "flat": 0.0,
    "close": 0.0,
    "sell": 0.0,
    "hold": None,
}


def _f(v) -> float | None:
    try:
        val = float(v)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def _norm_price_series(series) -> dict:
    """Coerce one ticker's prices to {date: {'close', 'open'}}. Accepts a list of
    {date, close, open?}, a {date: close} map, or bare closes (synthetic dates)."""
    out: dict = {}
    if isinstance(series, dict):
        for k, v in series.items():
            if isinstance(v, dict):
                c = _f(v.get("close", v.get("Close", v.get("price"))))
                o = _f(v.get("open", v.get("Open")))
                if c is not None:
                    out[str(k)] = {"close": c, "open": o}
            else:
                c = _f(v)
                if c is not None:
                    out[str(k)] = {"close": c, "open": None}
    elif isinstance(series, list):
        for i, row in enumerate(series):
            if isinstance(row, dict):
                d = str(row.get("date") or row.get("Date") or i)
                c = _f(row.get("close", row.get("Close", row.get("price"))))
                o = _f(row.get("open", row.get("Open")))
                if c is not None:
                    out[d] = {"close": c, "open": o}
            else:
                c = _f(row)
                if c is not None:
                    out[str(i)] = {"close": c, "open": None}
    return out


def _norm_adv_series(
    raw_adv_series: Mapping[str, Any] | None, prices_n: dict[str, dict]
) -> dict[str, tuple[list[str], list[float]]]:
    """Normalize per-ticker dated ADV into sorted ``(dates, values)`` tuples for O(log T) causal lookup."""
    out: dict[str, tuple[list[str], list[float]]] = {}
    if not isinstance(raw_adv_series, Mapping):
        return out
    for raw_tk, s in raw_adv_series.items():
        tk = str(raw_tk).upper()
        pairs: list[tuple[str, float]] = []
        if isinstance(s, Mapping):
            for k, v in s.items():
                fv = _f(v)
                if fv is not None and fv > 0:
                    pairs.append((str(k), fv))
        elif isinstance(s, Sequence) and not isinstance(s, (str, bytes)):
            tk_dates = sorted(prices_n.get(tk, {}))
            for i, row in enumerate(s):
                if isinstance(row, Mapping):
                    d = str(row.get("date") or row.get("Date") or (tk_dates[i] if i < len(tk_dates) else i))
                    fv = _f(row.get("adv", row.get("ADV", row.get("dollar_volume", row.get("value")))))
                else:
                    d = tk_dates[i] if i < len(tk_dates) else str(i)
                    fv = _f(row)
                if fv is not None and fv > 0:
                    pairs.append((d, fv))
        if pairs:
            pairs.sort(key=lambda x: x[0])
            dedup_d: list[str] = []
            dedup_v: list[float] = []
            for d_str, val in pairs:
                if dedup_d and dedup_d[-1] == d_str:
                    dedup_v[-1] = val
                else:
                    dedup_d.append(d_str)
                    dedup_v.append(val)
            out[tk] = (dedup_d, dedup_v)
    return out


def _norm_signal_series(series) -> list[tuple[str, float | None]]:
    """-> sorted [(date, target_weight or None-for-hold)]; weight from an explicit
    target_weight, else mapped from `action`."""
    out: list[tuple[str, float | None]] = []
    for row in series or []:
        if not isinstance(row, dict):
            continue
        d = str(row.get("date") or row.get("Date") or "")
        if not d:
            continue
        if row.get("target_weight") is not None:
            w = _f(row["target_weight"])
            if w is not None:
                out.append((d, w))
        elif row.get("action") is not None:
            out.append((d, _ACTION_WEIGHTS.get(str(row["action"]).lower().strip())))
    out.sort(key=lambda x: x[0])
    return out


def _apply_fill(p: dict, ticker: str, delta: float, exec_price: float, date: str, trades: list) -> None:
    """Update a position with a signed share delta at exec_price, emitting a
    closed-trade record on each reduction / close / flip (avg-cost accounting)."""
    cur = p["shares"]
    if cur == 0.0 or (cur > 0) == (delta > 0):  # opening or adding (same side)
        new = cur + delta
        if cur == 0.0:
            p["avg"] = exec_price
            p["entry_date"] = date
        else:
            p["avg"] = (p["avg"] * cur + exec_price * delta) / new
        p["shares"] = new
        return
    # Opposite side: reduce / close / flip, realize P&L on the closed portion.
    closing = min(abs(delta), abs(cur))
    sign = 1.0 if cur > 0 else -1.0
    entry = p["avg"]
    pnl = sign * (exec_price - entry) * closing
    pnl_pct = (sign * (exec_price - entry) / entry * 100.0) if entry else 0.0
    trades.append(
        {
            "ticker": ticker,
            "side": ("long" if cur > 0 else "short"),
            "entry_date": p["entry_date"],
            "entry_price": round(entry, 6),
            "exit_date": date,
            "exit_price": round(exec_price, 6),
            "shares": round(closing, 6),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 4),
        }
    )
    remaining = abs(cur) - closing
    if remaining <= 1e-12:
        flip = abs(delta) - abs(cur)
        if flip > 1e-12:  # overshoot -> open the flipped side
            p["shares"] = (1.0 if delta > 0 else -1.0) * flip
            p["avg"] = exec_price
            p["entry_date"] = date
        else:
            p["shares"], p["avg"], p["entry_date"] = 0.0, 0.0, None
    else:
        p["shares"] = sign * remaining  # same side, reduced; avg/entry unchanged


def run_backtest(
    signals: dict,
    prices: dict,
    *,
    slippage_bps: float = 5.0,
    commission_bps: float = 1.0,
    borrow_bps: float = 0.0,
    fill_timing: str = "close",
    initial_capital: float = 100000.0,
    max_position_pct: float = 1.0,
    adv: dict | None = None,
    adv_series: Mapping[str, Any] | None = None,
    impact_coef: float = 0.1,
    max_participation: float | None = None,
    liquidate_final: bool = True,
    periods_per_year: int = 252,
) -> dict:
    """Simulate a dated weight-target portfolio with causal fills and full costs.

    Features (`QOS-151`):
      - ``fill_timing``: ``"close"`` or ``"next_open"`` (waits for prior close;
        missing open falls back to that bar's close).
      - ``borrow_bps``: annual borrow rate in bps accrued per bar on carried short
        positions (`abs(shares) * close * (borrow_bps / 10000) / periods_per_year`).
      - ``adv`` / ``adv_series``: scalar `{ticker: dollar_adv}` or dated
        `{ticker: {date: dollar_adv}}` resolved strictly as-of the signal reference
        date (no full-sample ADV look-ahead) for Almgren square-root market impact
        and ``max_participation`` fill clipping.
      - ``liquidate_final``: when ``True`` (default), residual open positions at the
        final bar are liquidated in cash/equity with slippage, commission, and impact
        deducted rather than escaping exit costs.
    """
    prices_n = {str(tk).upper(): _norm_price_series(s) for tk, s in (prices or {}).items()}
    signals_n = {str(tk).upper(): _norm_signal_series(s) for tk, s in (signals or {}).items()}

    all_dates = sorted({d for s in prices_n.values() for d in s})
    if len(all_dates) < MIN_BARS:
        return {
            "error": f"need >= {MIN_BARS} price bars across the universe",
            "n_bars": len(all_dates),
            "engine_version": current_engine_version(),
        }

    ppy = max(int(periods_per_year), 1)
    slip = float(slippage_bps) / 10000.0
    comm = float(commission_bps) / 10000.0
    borrow_per_bar = (max(0.0, float(borrow_bps)) / 10000.0) / ppy
    cap = abs(float(max_position_pct))
    fill_open = str(fill_timing) == "next_open"

    _adv_scalar = {str(tk).upper(): float(v) for tk, v in (adv or {}).items() if _f(v) and float(v) > 0}
    _adv_dated = _norm_adv_series(adv_series, prices_n)
    has_adv = bool(_adv_scalar or _adv_dated)
    adv_source: str | None = "adv_series" if _adv_dated else ("scalar_adv" if _adv_scalar else None)
    _icoef = abs(float(impact_coef)) if impact_coef else 0.0
    _maxpart = abs(float(max_participation)) if max_participation else None

    def _resolve_adv(tk: str, asof_date: str) -> float | None:
        dated = _adv_dated.get(tk)
        if dated is not None:
            ds, vs = dated
            pos_idx = bisect.bisect_right(ds, asof_date) - 1
            if pos_idx >= 0:
                return vs[pos_idx]
            return None
        return _adv_scalar.get(tk)

    def _known_weights(tk: str) -> dict:
        sig = signals_n.get(tk, [])
        known, si, w = {}, 0, 0.0
        for d in all_dates:
            while si < len(sig) and sig[si][0] <= d:
                if sig[si][1] is not None:
                    w = max(-cap, min(cap, sig[si][1]))
                si += 1
            known[d] = w
        return known

    targets = {tk: _known_weights(tk) for tk in prices_n}
    tickers = sorted(prices_n)

    cash = float(initial_capital)
    pos = {tk: {"shares": 0.0, "avg": 0.0, "entry_date": None} for tk in prices_n}
    applied = {tk: 0.0 for tk in prices_n}
    trades: list[dict] = []
    equity_curve: list[float] = []
    dates_out: list[str] = []
    costs_per_bar: list[float] = []
    slip_cost_total = comm_cost_total = impact_cost_total = borrow_cost_total = 0.0
    gross_notional = 0.0
    n_capacity_limited = 0

    def _mark_equity(d: str) -> float:
        eq = cash
        for tk in tickers:
            bar = prices_n[tk].get(d)
            if bar and pos[tk]["shares"] != 0.0:
                eq += pos[tk]["shares"] * bar["close"]
        return eq

    for idx, d in enumerate(all_dates):
        bar_cost = 0.0
        # Short borrow accrues on short positions carried into bar `d` (`idx >= 1`).
        if borrow_per_bar > 0.0 and idx > 0:
            for tk in tickers:
                sh = pos[tk]["shares"]
                if sh < 0.0:
                    bar_b = prices_n[tk].get(d)
                    if bar_b and bar_b["close"] > 0:
                        b_cost = abs(sh) * bar_b["close"] * borrow_per_bar
                        cash -= b_cost
                        borrow_cost_total += b_cost
                        bar_cost += b_cost

        for tk in tickers:
            bar = prices_n[tk].get(d)
            if bar is None:
                continue
            if fill_open:
                if idx == 0:
                    continue
                fillp = bar["open"] if bar["open"] is not None else bar["close"]
                ref_date = all_dates[idx - 1]
            else:
                fillp = bar["close"]
                ref_date = d
            if not fillp or fillp <= 0:
                continue
            tw = targets[tk].get(ref_date, 0.0)
            if abs(tw - applied[tk]) < 1e-12:
                continue
            equity = _mark_equity(d)
            target_shares = tw * equity / fillp
            delta = target_shares - pos[tk]["shares"]
            applied[tk] = tw
            exec_price = fillp * (1 + slip) if delta > 0 else fillp * (1 - slip)

            adv_tk = _resolve_adv(tk, ref_date)
            if adv_tk and _maxpart:
                cur_notional = abs(delta) * exec_price
                if cur_notional > _maxpart * adv_tk:
                    delta *= (_maxpart * adv_tk) / cur_notional
                    n_capacity_limited += 1
                    achieved = pos[tk]["shares"] + delta
                    applied[tk] = (achieved * fillp / equity) if equity else tw
            if abs(delta * fillp) < 1e-9:
                continue
            slip_cost = abs(delta) * fillp * slip
            comm_cost = abs(delta) * exec_price * comm
            impact_cost = 0.0
            if adv_tk and _icoef:
                notional = abs(delta) * exec_price
                impact_cost = notional * _icoef * math.sqrt(notional / adv_tk)
            cash -= delta * exec_price
            cash -= comm_cost
            cash -= impact_cost
            slip_cost_total += slip_cost
            comm_cost_total += comm_cost
            impact_cost_total += impact_cost
            gross_notional += abs(delta) * exec_price
            bar_cost += slip_cost + comm_cost + impact_cost
            _apply_fill(pos[tk], tk, delta, exec_price, d, trades)
        equity_curve.append(round(_mark_equity(d), 2))
        dates_out.append(d)
        costs_per_bar.append(round(bar_cost, 6))

    # Close residual positions at the final close. When `liquidate_final=True`
    # (default in 2.0), exit slippage, commission, and market impact are deducted
    # from cash and reflected in `equity_curve[-1]` and `cost_breakdown`.
    last = all_dates[-1]
    liq_bar_cost = 0.0
    liquidated_any = False
    for tk in tickers:
        if pos[tk]["shares"] != 0.0 and prices_n[tk].get(last):
            fillp = prices_n[tk][last]["close"]
            delta = -pos[tk]["shares"]
            exec_price = fillp * (1 + slip) if delta > 0 else fillp * (1 - slip)
            notional = abs(delta) * exec_price
            gross_notional += notional
            if liquidate_final:
                slip_cost = abs(delta) * fillp * slip
                comm_cost = notional * comm
                adv_tk = _resolve_adv(tk, last)
                impact_cost = notional * _icoef * math.sqrt(notional / adv_tk) if (adv_tk and _icoef) else 0.0
                cash -= delta * exec_price
                cash -= comm_cost
                cash -= impact_cost
                slip_cost_total += slip_cost
                comm_cost_total += comm_cost
                impact_cost_total += impact_cost
                liq_bar_cost += slip_cost + comm_cost + impact_cost
                liquidated_any = True
            _apply_fill(pos[tk], tk, delta, exec_price, last, trades)

    if liquidate_final and liquidated_any and equity_curve:
        equity_curve[-1] = round(_mark_equity(last), 2)
        costs_per_bar[-1] = round(costs_per_bar[-1] + liq_bar_cost, 6)

    # Initial-capital-aware per-bar returns: bar 0 is measured against
    # `initial_capital` so bar-0 execution costs are never dropped from returns.
    returns: list[float] = []
    init_cap_f = float(initial_capital)
    for i, eq_val in enumerate(equity_curve):
        prev = init_cap_f if i == 0 else equity_curve[i - 1]
        returns.append(round((eq_val - prev) / prev, 8) if prev else 0.0)

    final_equity = equity_curve[-1] if equity_curve else init_cap_f
    total_return_pct = round((final_equity / init_cap_f - 1.0) * 100.0, 4) if init_cap_f else 0.0
    fill_mode = "next_open" if fill_open else "close"
    costs_model = build_costs_model_spec(
        mode="simulated",
        slippage_bps=float(slippage_bps),
        commission_bps=float(commission_bps),
        borrow_bps=float(borrow_bps),
        fill_timing=fill_mode,
        adv_source=adv_source,
        liquidate_final=bool(liquidate_final),
    )

    record(
        "equity",
        "marked equity after each bar net of slippage, commission, borrow, impact, and final liquidation",
        inputs={
            "n_bars": len(dates_out),
            "fill_timing": fill_mode,
            "initial_capital": init_cap_f,
            "liquidate_final": bool(liquidate_final),
        },
        series={"equity": equity_curve, "returns": returns},
        result=total_return_pct,
    )
    total_costs = slip_cost_total + comm_cost_total + impact_cost_total + borrow_cost_total
    out = {
        "engine_version": current_engine_version(),
        "n_bars": len(dates_out),
        "dates": dates_out,
        "equity_curve": equity_curve,
        "returns": returns,
        "trades": trades,
        "n_trades": len(trades),
        "initial_capital": init_cap_f,
        "final_equity": round(final_equity, 2),
        "total_return_pct": total_return_pct,
        "gross_traded_notional": round(gross_notional, 2),
        "cost_breakdown": {
            "slippage": round(slip_cost_total, 2),
            "commission": round(comm_cost_total, 2),
            "impact": round(impact_cost_total, 2),
            "borrow": round(borrow_cost_total, 2),
            "total": round(total_costs, 2),
        },
        "costs_per_bar": costs_per_bar,
        "capacity": (
            {
                "max_participation": _maxpart,
                "n_capacity_limited_fills": n_capacity_limited,
                "adv_source": adv_source,
            }
            if (_maxpart or has_adv)
            else None
        ),
        "costs_model": costs_model,
        "annualization": build_annualization_spec(ppy),
        "config": {
            "slippage_bps": float(slippage_bps),
            "commission_bps": float(commission_bps),
            "borrow_bps": float(borrow_bps),
            "fill_timing": fill_mode,
            "max_position_pct": cap,
            "impact_model": ("sqrt" if has_adv else None),
            "impact_coef": (_icoef if has_adv else None),
            "max_participation": _maxpart,
            "adv_source": adv_source,
            "liquidate_final": bool(liquidate_final),
            "periods_per_year": ppy,
        },
    }
    if _NOTEBOOK_HOOK_CALLBACK is not None:
        with contextlib.suppress(Exception):
            _NOTEBOOK_HOOK_CALLBACK(out)
    return out


# ── Scoring: compose performance + overfitting into a canonical 2.0 verdict ──


def _verdict(
    dsr: float | None,
    pbo: float | None,
    n_trials_source: str = "asserted",
    *,
    n_trials: int | float | None = None,
) -> str:
    """Canonical 2.0 verdict helper (`robust`, `marginal`, `likely_noise`, `unverified`)."""
    verdict, _ = verdict_from_dsr(
        dsr,
        pbo=pbo,
        n_trials=n_trials if n_trials is not None else (None if n_trials_source == "not_recorded" else 2),
        n_trials_source=n_trials_source,
    )
    return verdict


def _ar1_effective_n(returns: list) -> tuple:
    """AR(1) autocorrelation-honest effective sample size and lag-1 autocorr."""
    from .validation import _as_1d_float

    arr = _as_1d_float(returns)
    n = arr.size
    if n < 3:
        return (float(n), 0.0)
    a = arr[:-1] - arr[:-1].mean()
    b = arr[1:] - arr[1:].mean()
    denom = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    r1 = float((a * b).sum() / denom) if denom > 0 else 0.0
    factor = (1.0 - r1) / (1.0 + r1) if r1 > -0.999 else float(n)
    n_eff = max(1.0, min(float(n), n * factor))
    return (n_eff, r1)


def _trade_summary(trades: list) -> dict:
    if not trades:
        return {"n_trades": 0}
    pnls = [t.get("pnl", 0.0) for t in trades]
    pcts = [t.get("pnl_pct", 0.0) for t in trades]
    wins = [p for p in pnls if p > 0]
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    return {
        "n_trades": len(trades),
        "win_rate_pct": round(len(wins) / len(trades) * 100.0, 2),
        "total_pnl": round(sum(pnls), 2),
        "avg_pnl_pct": round(sum(pcts) / len(pcts), 4),
        "profit_factor": round(gross_win / gross_loss, 4) if gross_loss > 0 else None,
        "best_pct": round(max(pcts), 4),
        "worst_pct": round(min(pcts), 4),
    }


def _excursions(trades: list, prices: dict) -> dict:
    """MAE/MFE per trade from the close path between entry and exit."""
    pn = {str(tk).upper(): _norm_price_series(s) for tk, s in (prices or {}).items()}
    per, maes, mfes, n_cov = [], [], [], 0
    for t in trades:
        tk = str(t.get("ticker") or "").upper()
        series = pn.get(tk) or {}
        ed, xd, entry = t.get("entry_date"), t.get("exit_date"), t.get("entry_price")
        mae = mfe = None
        if series and entry and ed and xd:
            lo, hi = (ed, xd) if ed <= xd else (xd, ed)
            path = [b["close"] for d, b in series.items() if lo <= d <= hi]
            if path:
                sign = -1.0 if str(t.get("side")) == "short" else 1.0
                favs = [sign * (p - entry) / entry * 100.0 for p in path]
                mae, mfe = round(min(favs), 4), round(max(favs), 4)
                maes.append(mae)
                mfes.append(mfe)
                n_cov += 1
        per.append({"ticker": tk, "entry_date": ed, "exit_date": xd, "mae_pct": mae, "mfe_pct": mfe})
    return {
        "per_trade": per,
        "n_covered": n_cov,
        "avg_mae_pct": round(sum(maes) / len(maes), 4) if maes else None,
        "avg_mfe_pct": round(sum(mfes) / len(mfes), 4) if mfes else None,
    }


def _regime_attribution(trades: list, regime_series: dict) -> dict:
    rs = {str(k): str(v) for k, v in regime_series.items()} if isinstance(regime_series, dict) else {}
    buckets: dict = {}
    for t in trades:
        reg = rs.get(str(t.get("entry_date")))
        if reg is None:
            continue
        b = buckets.setdefault(reg, {"n": 0, "pnl": 0.0, "wins": 0})
        b["n"] += 1
        b["pnl"] += t.get("pnl", 0.0)
        if (t.get("pnl_pct") or 0) > 0:
            b["wins"] += 1
    return {
        reg: {
            "n_trades": b["n"],
            "total_pnl": round(b["pnl"], 2),
            "win_rate_pct": round(b["wins"] / b["n"] * 100.0, 2) if b["n"] else None,
        }
        for reg, b in buckets.items()
    }


def _attribution(
    bt: dict,
    returns: list,
    prices,
    factor_returns,
    regime_series,
    risk_free_rate: float | None,
    not_computed: dict,
) -> dict:
    trades = bt.get("trades") or []
    attribution: dict = {}
    if prices:
        attribution["excursions"] = _excursions(trades, prices)
    else:
        not_computed["attribution.excursions"] = "supply prices to compute MAE/MFE (intra-trade excursions)"
    if factor_returns:
        from .factors import decompose_factors

        fa = decompose_factors(returns, factor_returns, risk_free_rate=risk_free_rate)
        if "error" in fa:
            attribution["factors"] = None
            not_computed["attribution.factors"] = fa["error"]
        else:
            attribution["factors"] = fa
    else:
        not_computed["attribution.factors"] = (
            "supply factor_returns {factor: [...]} to attribute returns (no factor vendor)"
        )
    if regime_series:
        attribution["regime"] = _regime_attribution(trades, regime_series)
    else:
        not_computed["attribution.regime"] = (
            "supply regime_series {date: regime} to bucket trade P&L by regime at entry"
        )
    return attribution


def _cost_report(
    bt: dict,
    perf: dict | None,
    risk_free_rate: float | None,
    *,
    initial_capital: float | None = None,
    periods_per_year: int = 252,
) -> dict | None:
    """Turnover + initial-capital-aware net-of-cost vs gross Sharpe."""
    cb = bt.get("cost_breakdown")
    if not isinstance(cb, dict):
        return None
    init_cap = (
        float(initial_capital)
        if (initial_capital is not None and float(initial_capital) > 0)
        else (float(bt.get("initial_capital") or 0.0) or None)
    )
    n_bars = int(bt.get("n_bars") or 0)
    ppy = max(int(periods_per_year), 1)
    gross_notional = float(bt.get("gross_traded_notional") or 0.0)
    total_cost = float(cb.get("total") or 0.0)
    turnover_ratio = (gross_notional / init_cap) if init_cap else None
    out = {
        "cost_breakdown": cb,
        "cost_drag_pct": round(total_cost / init_cap * 100.0, 4) if init_cap else None,
        "turnover_ratio": round(turnover_ratio, 4) if turnover_ratio is not None else None,
        "annualized_turnover": (
            round(turnover_ratio * float(ppy) / n_bars, 4)
            if (turnover_ratio is not None and n_bars)
            else None
        ),
        "net_of_cost_sharpe": (perf.get("sharpe_annualized") if isinstance(perf, dict) else None),
        "gross_sharpe": None,
        "sharpe_cost_drag": None,
        "capacity": bt.get("capacity"),
    }
    cpb = bt.get("costs_per_bar")
    eq = bt.get("equity_curve")
    if isinstance(cpb, list) and isinstance(eq, list) and len(cpb) == len(eq) and len(eq) > 1:
        from .performance import performance_report

        cum = 0.0
        gross_marks: list[float] = []
        for i, e in enumerate(eq):
            cum += float(cpb[i])
            gross_marks.append(float(e) + cum)
        gross_eq = [init_cap, *gross_marks] if init_cap else gross_marks
        gross_rets = [
            round((gross_eq[i] - gross_eq[i - 1]) / gross_eq[i - 1], 8)
            for i in range(1, len(gross_eq))
            if gross_eq[i - 1]
        ]
        gp = performance_report(
            gross_rets,
            equity_curve=gross_eq,
            risk_free_rate=risk_free_rate,
            periods_per_year=ppy,
        )
        if isinstance(gp, dict) and "error" not in gp:
            out["gross_sharpe"] = gp.get("sharpe_annualized")
            if out["net_of_cost_sharpe"] is not None and out["gross_sharpe"] is not None:
                out["sharpe_cost_drag"] = round(out["gross_sharpe"] - out["net_of_cost_sharpe"], 4)
    return out


def score_backtest(
    bt: dict,
    *,
    n_trials: int | None = None,
    n_trials_source: str | None = None,
    ledger_runs: Sequence[Mapping[str, Any]] | int | None = None,
    initial_capital: float | None = None,
    risk_free_rate: float | None = None,
    periods_per_year: int = 252,
    benchmark_returns: list | None = None,
    pnl_matrix=None,
    trials_sharpe_std: float | None = None,
    prices: dict | None = None,
    factor_returns: dict | None = None,
    regime_series: dict | None = None,
    cpcv: bool = False,
    cpcv_n_groups: int = 8,
    cpcv_n_test_groups: int = 2,
    cpcv_purge: int = 1,
    cpcv_embargo: int = 1,
) -> dict:
    """Score a ``run_backtest`` result into canonical 2.0 performance + validation figures.

    Applies ``trials_for_deflated_sharpe`` across ``n_trials`` and ``ledger_runs``,
    enforces ``deflation_applied=False`` (capping verdict at ``marginal``) when
    ``n_trials <= 1`` or ``not_recorded``, anchors equity/return evaluation to
    ``initial_capital`` when present, and stamps ``engine_version``,
    ``annualization``, ``risk_free``, and ``costs_model``.
    """
    from .performance import performance_report
    from .validation import (
        _per_period_sharpe,
        combinatorial_oos_stability,
        deflated_sharpe,
        harvey_liu_haircut,
        min_track_record_length,
        n_eff_participation,
        pbo_cscv,
        trials_for_deflated_sharpe,
    )

    ppy = max(int(periods_per_year), 1)
    resolved_trials, source = trials_for_deflated_sharpe(
        n_trials,
        n_trials_source=n_trials_source,
        ledger_runs=ledger_runs,
    )
    n_trials_eff = max(1, int(resolved_trials)) if resolved_trials is not None else 1

    init_cap = (
        float(initial_capital)
        if (initial_capital is not None and float(initial_capital) > 0)
        else (_f(bt.get("initial_capital")) if bt.get("initial_capital") else None)
    )
    raw_eq = bt.get("equity_curve")
    raw_returns = bt.get("returns")

    if isinstance(raw_eq, list) and len(raw_eq) >= 1 and init_cap and init_cap > 0:
        if not raw_returns:
            returns = [
                round(
                    (
                        (float(raw_eq[i]) - (init_cap if i == 0 else float(raw_eq[i - 1])))
                        / (init_cap if i == 0 else float(raw_eq[i - 1]))
                    ),
                    8,
                )
                for i in range(len(raw_eq))
            ]
            eq_for_perf: list[float] | None = [init_cap, *[float(x) for x in raw_eq]]
        elif len(raw_returns) == len(raw_eq) - 1:
            r0 = round((float(raw_eq[0]) - init_cap) / init_cap, 8)
            returns = [r0, *[float(r) for r in raw_returns if r is not None]]
            eq_for_perf = [init_cap, *[float(x) for x in raw_eq]]
        elif len(raw_returns) == len(raw_eq):
            returns = [float(r) for r in raw_returns if r is not None]
            eq_for_perf = [init_cap, *[float(x) for x in raw_eq]]
        else:
            returns = list(raw_returns)
            eq_for_perf = raw_eq
    else:
        returns = list(raw_returns or [])
        eq_for_perf = raw_eq

    perf = performance_report(
        returns,
        equity_curve=eq_for_perf,
        benchmark_returns=benchmark_returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=ppy,
    )
    dsr = deflated_sharpe(
        returns,
        n_trials=n_trials_eff,
        trials_sharpe_std=trials_sharpe_std,
        risk_free_rate=risk_free_rate,
        periods_per_year=ppy,
    )
    pbo = pbo_cscv(pnl_matrix) if pnl_matrix is not None else None
    mintrl = min_track_record_length(returns, risk_free_rate=risk_free_rate, periods_per_year=ppy)

    dsr_ok = isinstance(dsr, dict) and "error" not in dsr
    pbo_ok = isinstance(pbo, dict) and "error" not in pbo
    mintrl_ok = isinstance(mintrl, dict) and "error" not in mintrl
    dsr_val = dsr.get("dsr_probability") if dsr_ok else None
    psr_val = dsr.get("psr_vs_zero") if dsr_ok else None
    pbo_val = pbo.get("pbo") if pbo_ok else None

    n_eff, r1 = _ar1_effective_n(returns)
    sr_pp = dsr.get("sharpe_per_period") if dsr_ok else None
    if sr_pp is None:
        arr_len = len([r for r in returns if r is not None])
        import numpy as _np

        sr_pp = (
            _per_period_sharpe(_np.asarray([float(r) for r in returns if r is not None], dtype=float))
            if arr_len >= 2
            else 0.0
        )
    t_stat = float(sr_pp) * math.sqrt(n_eff)

    n_eff_part: float | None = None
    trial_pvals: list[float] | None = None
    if pnl_matrix is not None:
        import numpy as _np
        from scipy import stats as _stats

        mat_arr = _np.asarray(pnl_matrix, dtype=float)
        if mat_arr.ndim == 2 and mat_arr.shape[1] >= 2 and mat_arr.shape[0] >= 2:
            n_eff_part = round(n_eff_participation(mat_arr), 4)
            df_mat = max(mat_arr.shape[0] - 1, 1)
            trial_pvals = []
            for j in range(mat_arr.shape[1]):
                col_j = mat_arr[:, j]
                sd_j = float(col_j.std(ddof=1))
                t_j = float(col_j.mean() / sd_j * math.sqrt(mat_arr.shape[0])) if sd_j > 0 else 0.0
                trial_pvals.append(float(2.0 * (1.0 - _stats.t.cdf(abs(t_j), df=df_mat))))

    n_obs_val = int(dsr.get("n_obs") if dsr_ok else len([r for r in returns if r is not None]))
    hl_haircut = (
        harvey_liu_haircut(
            float(sr_pp) * math.sqrt(ppy),
            n_obs_val,
            n_trials_eff,
            periods_per_year=ppy,
            n_obs_effective=n_eff,
            n_eff_participation=n_eff_part if n_eff_part is not None else float(n_trials_eff),
            trial_pvalues=trial_pvals,
        )
        if resolved_trials is not None and n_obs_val >= 2
        else None
    )

    verdict_val, deflation_applied = verdict_from_dsr(
        dsr_val,
        pbo=pbo_val,
        n_trials=resolved_trials,
        n_trials_source=source,
    )
    bt_cfg = bt.get("config") if isinstance(bt.get("config"), dict) else {}
    costs_model = (
        dict(bt["costs_model"])
        if isinstance(bt.get("costs_model"), dict)
        else build_costs_model_spec(
            mode="simulated" if bt_cfg else "pre_netted_returns",
            slippage_bps=_f(bt_cfg.get("slippage_bps")),
            commission_bps=_f(bt_cfg.get("commission_bps")),
            borrow_bps=_f(bt_cfg.get("borrow_bps")),
            fill_timing=bt_cfg.get("fill_timing"),
            adv_source=bt_cfg.get("adv_source"),
            liquidate_final=bt_cfg.get("liquidate_final"),
        )
    )

    validation = {
        "engine_version": current_engine_version(),
        "dsr_probability": dsr_val,
        "deflated_sharpe": dsr_val,
        "deflation_applied": deflation_applied,
        "pbo": pbo_val,
        "psr": psr_val,
        "n_trials": (int(resolved_trials) if resolved_trials is not None else None),
        "n_trials_source": source,
        "n_eff_participation": n_eff_part,
        "verdict": verdict_val,
        "n_obs": n_obs_val,
        "n_obs_effective": round(n_eff, 1),
        "autocorr_lag1": round(r1, 4),
        "sharpe_tstat": round(t_stat, 4),
        "harvey_liu_hurdle": 3.0,
        "passes_harvey_liu": bool(t_stat > 3.0),
        "harvey_liu": hl_haircut,
        "min_track_record_length": (mintrl.get("min_track_record_length") if mintrl_ok else None),
        "min_track_record_years": (mintrl.get("min_track_record_years") if mintrl_ok else None),
        "track_length_sufficient": (mintrl.get("sufficient") if mintrl_ok else None),
        "track_record_shortfall_obs": (mintrl.get("shortfall_obs") if mintrl_ok else None),
    }

    not_computed: dict = {}
    if not (isinstance(perf, dict) and "error" not in perf):
        not_computed["performance"] = (perf or {}).get("error", "insufficient observations (need >= 30 bars)")
        perf = None
    if not dsr_ok:
        not_computed["validation.deflated_sharpe"] = dsr.get(
            "error", "insufficient observations (need >= 8 bars)"
        )
    if pbo is None:
        not_computed["validation.pbo"] = (
            "supply a parameter-sweep pnl_matrix (one column per config) to run PBO"
        )
    elif not pbo_ok:
        not_computed["validation.pbo"] = pbo.get("error", "pbo unavailable")

    if cpcv:
        cp = combinatorial_oos_stability(
            returns,
            n_groups=cpcv_n_groups,
            n_test_groups=cpcv_n_test_groups,
            purge=cpcv_purge,
            embargo=cpcv_embargo,
            n_trials=n_trials_eff,
            risk_free_rate=risk_free_rate,
            periods_per_year=ppy,
        )
        if isinstance(cp, dict) and "error" not in cp:
            validation["cpcv"] = cp
        else:
            not_computed["validation.cpcv"] = (cp or {}).get("error", "cpcv unavailable")
    else:
        not_computed["validation.cpcv"] = (
            "set cpcv=true to run Combinatorial Purged Cross-Validation (OOS path distribution)"
        )

    attribution = _attribution(
        bt, returns, prices, factor_returns, regime_series, risk_free_rate, not_computed
    )
    costs = _cost_report(bt, perf, risk_free_rate, initial_capital=init_cap, periods_per_year=ppy)

    record(
        "verdict",
        "robust requires deflation_applied (recorded n_trials > 1), dsr_probability >= 0.9, and pbo <= 0.2 when pbo is present",
        inputs={
            "n_trials": validation["n_trials"],
            "n_trials_source": source,
            "deflation_applied": deflation_applied,
            "dsr_probability": dsr_val,
            "pbo": pbo_val,
        },
        series={"returns": [float(r) for r in returns if r is not None]},
        result=validation["verdict"],
    )
    return {
        "engine_version": current_engine_version(),
        "annualization": build_annualization_spec(ppy),
        "risk_free": build_risk_free_spec(risk_free_rate, ppy),
        "costs_model": costs_model,
        "performance": perf,
        "validation": validation,
        "dsr_detail": dsr if dsr_ok else None,
        "trade_summary": _trade_summary(bt.get("trades") or []),
        "attribution": attribution,
        "costs": costs,
        "not_computed": not_computed,
    }


__all__ = ["run_backtest", "score_backtest"]
