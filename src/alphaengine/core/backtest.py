"""
backtest.py, deterministic signal+price backtest (BACKTESTING_TRACKRECORD_PLAN.md
Pillar A). PURE: signals + prices in, fills simulated with slippage/commission,
equity curve + per-trade log out. No fetch, no LLM, no clock. Math-identical to
backend/quant/backtest.py (parity-guarded).

Input:
  signals : {ticker: [{date, target_weight | action, conviction?}...]}
            target_weight is a signed portfolio weight (negative = short),
            forward-filled until changed; `action` maps to a weight
            (enter_long/long/buy=+1, enter_short/short=-1, exit/flat/close=0,
            hold=carry prior). Weights are clamped to ±max_position_pct.
  prices  : {ticker: [{date, close, open?}...]}  (or {date: close} / bare closes)
  config  : slippage_bps, commission_bps, fill_timing ('close'|'next_open'),
            initial_capital, max_position_pct

Output: dates, equity_curve, returns, trades[], summary scalars. Scoring
(performance_report + deflated-Sharpe verdict) is layered on by the gateway tool;
this module only simulates and reconstructs trades.

No-look-ahead: a signal dated bar i is actioned at bar i's close ('close') or
bar i+1's open ('next_open'), never at a price it could not have transacted at.
"""

from __future__ import annotations

import math
from typing import Optional

MIN_BARS = 2

# action keyword -> target weight; None means "hold" (carry the prior weight).
_ACTION_WEIGHTS = {
    "enter_long": 1.0, "long": 1.0, "buy": 1.0,
    "enter_short": -1.0, "short": -1.0, "sell_short": -1.0,
    "exit": 0.0, "flat": 0.0, "close": 0.0, "sell": 0.0,
    "hold": None,
}


def _f(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


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


def _norm_signal_series(series) -> list[tuple[str, Optional[float]]]:
    """-> sorted [(date, target_weight or None-for-hold)]; weight from an explicit
    target_weight, else mapped from `action`."""
    out: list[tuple[str, Optional[float]]] = []
    for row in (series or []):
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
            out.append((d, _ACTION_WEIGHTS.get(str(row["action"]).lower().strip(), None)))
    out.sort(key=lambda x: x[0])
    return out


def _apply_fill(p: dict, ticker: str, delta: float, exec_price: float,
                date: str, trades: list) -> None:
    """Update a position with a signed share delta at exec_price, emitting a
    closed-trade record on each reduction / close / flip (avg-cost accounting)."""
    cur = p["shares"]
    if cur == 0.0 or (cur > 0) == (delta > 0):           # opening or adding (same side)
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
    trades.append({
        "ticker": ticker, "side": ("long" if cur > 0 else "short"),
        "entry_date": p["entry_date"], "entry_price": round(entry, 6),
        "exit_date": date, "exit_price": round(exec_price, 6),
        "shares": round(closing, 6), "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 4),
    })
    remaining = abs(cur) - closing
    if remaining <= 1e-12:
        flip = abs(delta) - abs(cur)
        if flip > 1e-12:                                  # overshoot -> open the flipped side
            p["shares"] = (1.0 if delta > 0 else -1.0) * flip
            p["avg"] = exec_price
            p["entry_date"] = date
        else:
            p["shares"], p["avg"], p["entry_date"] = 0.0, 0.0, None
    else:
        p["shares"] = sign * remaining                    # same side, reduced; avg/entry unchanged


def run_backtest(signals: dict, prices: dict, *, slippage_bps: float = 5.0,
                 commission_bps: float = 1.0, fill_timing: str = "close",
                 initial_capital: float = 100000.0, max_position_pct: float = 1.0,
                 adv: Optional[dict] = None, impact_coef: float = 0.1,
                 max_participation: Optional[float] = None) -> dict:
    """See the module docstring for the fill model. Capacity / market-impact are
    OPT-IN and OFF unless `adv` (a {ticker: average daily DOLLAR volume} map) is
    supplied, with adv=None the fill math is byte-identical to the cost-free path
    (the golden + parity guard depend on this):
      - square-root market impact: a per-fill cost = notional·impact_coef·√(notional/ADV)
        (Almgren-style) deducted from cash on top of slippage + commission.
      - capacity cap: when `max_participation` is set, a fill is clipped so its
        notional never exceeds max_participation·ADV; the un-filled remainder keeps
        working toward target on later bars (tagged n_capacity_limited_fills).
    Always reports a `cost_breakdown` (slippage/commission/impact $), the per-bar
    cost drag (`costs_per_bar`, for a gross-vs-net decomposition) and gross traded
    notional (for turnover), these are derived, never stored."""
    prices_n = {str(tk).upper(): _norm_price_series(s) for tk, s in (prices or {}).items()}
    signals_n = {str(tk).upper(): _norm_signal_series(s) for tk, s in (signals or {}).items()}

    all_dates = sorted({d for s in prices_n.values() for d in s.keys()})
    if len(all_dates) < MIN_BARS:
        return {"error": f"need >= {MIN_BARS} price bars across the universe",
                "n_bars": len(all_dates)}

    slip = float(slippage_bps) / 10000.0
    comm = float(commission_bps) / 10000.0
    cap = abs(float(max_position_pct))
    fill_open = str(fill_timing) == "next_open"
    # Capacity / market-impact (opt-in; inert when adv is falsy).
    _adv = {str(tk).upper(): float(v) for tk, v in (adv or {}).items() if _f(v) and float(v) > 0}
    _icoef = abs(float(impact_coef)) if impact_coef else 0.0
    _maxpart = abs(float(max_participation)) if max_participation else None

    # Per-ticker target weight known as of each bar's close (forward-filled,
    # clamped to ±cap; 'hold' carries the prior weight).
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
    applied = {tk: 0.0 for tk in prices_n}   # weight currently expressed in the position
    trades: list[dict] = []
    equity_curve: list[float] = []
    dates_out: list[str] = []
    costs_per_bar: list[float] = []          # total explicit cost charged to equity each bar
    slip_cost_total = comm_cost_total = impact_cost_total = 0.0
    gross_notional = 0.0                      # |delta|·exec traded, for turnover
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
        for tk in tickers:
            bar = prices_n[tk].get(d)
            if bar is None:
                continue
            if fill_open:
                fillp = bar["open"] if bar["open"] is not None else bar["close"]
                ref_date = all_dates[idx - 1] if idx > 0 else d   # act on prior close's signal
            else:
                fillp = bar["close"]
                ref_date = d
            if not fillp or fillp <= 0:
                continue
            tw = targets[tk].get(ref_date, 0.0)
            # Trade ONLY when the target weight CHANGES, enter on signal, hold,
            # exit on signal. No rebalance-to-weight drift between signals (which
            # would manufacture phantom trades every bar).
            if abs(tw - applied[tk]) < 1e-12:
                continue
            equity = _mark_equity(d)
            target_shares = tw * equity / fillp
            delta = target_shares - pos[tk]["shares"]
            applied[tk] = tw
            exec_price = fillp * (1 + slip) if delta > 0 else fillp * (1 - slip)
            # Capacity cap (opt-in): clip the fill to max_participation·ADV; the
            # remainder keeps working toward target on later bars.
            adv_tk = _adv.get(tk)
            if adv_tk and _maxpart:
                cur_notional = abs(delta) * exec_price
                if cur_notional > _maxpart * adv_tk:
                    delta *= (_maxpart * adv_tk) / cur_notional
                    n_capacity_limited += 1
                    achieved = pos[tk]["shares"] + delta
                    applied[tk] = (achieved * fillp / equity) if equity else tw  # not fully filled
            if abs(delta * fillp) < 1e-9:
                continue
            slip_cost = abs(delta) * fillp * slip           # slippage (already inside exec_price)
            comm_cost = abs(delta) * exec_price * comm
            impact_cost = 0.0
            if adv_tk and _icoef:                            # square-root market impact
                notional = abs(delta) * exec_price
                impact_cost = notional * _icoef * math.sqrt(notional / adv_tk)
            cash -= delta * exec_price                      # buy lowers cash; short/sell raises it
            cash -= comm_cost                                # commission
            cash -= impact_cost                              # market impact (0 unless adv given)
            slip_cost_total += slip_cost
            comm_cost_total += comm_cost
            impact_cost_total += impact_cost
            gross_notional += abs(delta) * exec_price
            bar_cost += slip_cost + comm_cost + impact_cost
            _apply_fill(pos[tk], tk, delta, exec_price, d, trades)
        equity_curve.append(round(_mark_equity(d), 2))
        dates_out.append(d)
        costs_per_bar.append(round(bar_cost, 6))

    # Close residual positions at the final close so every trade completes. This
    # completes the trade LOG only (it does not touch cash/equity, see the module
    # docstring); we still count its notional toward turnover.
    last = all_dates[-1]
    for tk in tickers:
        if pos[tk]["shares"] != 0.0 and prices_n[tk].get(last):
            fillp = prices_n[tk][last]["close"]
            delta = -pos[tk]["shares"]
            exec_price = fillp * (1 + slip) if delta > 0 else fillp * (1 - slip)
            gross_notional += abs(delta) * exec_price
            _apply_fill(pos[tk], tk, delta, exec_price, last, trades)

    returns = []
    for i in range(1, len(equity_curve)):
        prev = equity_curve[i - 1]
        returns.append(round((equity_curve[i] - prev) / prev, 8) if prev else 0.0)

    final_equity = equity_curve[-1] if equity_curve else float(initial_capital)
    total_return_pct = round((final_equity / float(initial_capital) - 1.0) * 100.0, 4)

    return {
        "n_bars": len(dates_out),
        "dates": dates_out,
        "equity_curve": equity_curve,
        "returns": returns,
        "trades": trades,
        "n_trades": len(trades),
        "initial_capital": float(initial_capital),
        "final_equity": round(final_equity, 2),
        "total_return_pct": total_return_pct,
        "gross_traded_notional": round(gross_notional, 2),
        "cost_breakdown": {
            "slippage": round(slip_cost_total, 2),
            "commission": round(comm_cost_total, 2),
            "impact": round(impact_cost_total, 2),
            "total": round(slip_cost_total + comm_cost_total + impact_cost_total, 2),
        },
        "costs_per_bar": costs_per_bar,
        "capacity": ({"max_participation": _maxpart, "n_capacity_limited_fills": n_capacity_limited}
                     if _maxpart else None),
        "config": {"slippage_bps": float(slippage_bps), "commission_bps": float(commission_bps),
                   "fill_timing": "next_open" if fill_open else "close",
                   "max_position_pct": cap,
                   "impact_model": ("sqrt" if _adv else None),
                   "impact_coef": (_icoef if _adv else None),
                   "max_participation": _maxpart},
    }


# ── Scoring (gateway-only): compose performance + overfitting into a verdict ──
# NOT parity-copied to the backend, it composes quant_core.performance +
# quant_core.validation, whose backend siblings differ in shape. Only run_backtest
# is the shared pure core. The verdict rule below IS the moat in code: 'edge' is
# unreachable without a populated deflated_sharpe, mirroring the envelope Signal's
# _edge_requires_validation rule, which a test cross-checks.


def _verdict(dsr: Optional[float], pbo: Optional[float]) -> str:
    if dsr is None:
        return "inconclusive"                  # no rigor figure -> never 'edge'
    if pbo is not None and pbo > 0.5:
        return "likely_noise"                  # overfit by PBO
    if dsr >= 0.9 and (pbo is None or pbo <= 0.2):
        return "edge"
    if dsr < 0.5:
        return "likely_noise"
    return "inconclusive"


def _ar1_effective_n(returns: list) -> tuple:
    """AR(1) autocorrelation-honest effective sample size and lag-1 autocorr.

    n_eff = n·(1-r1)/(1+r1). Positive serial correlation inflates a naive Sharpe
    t-stat; this discounts it. Conservative convention: capped at n, so NEGATIVE
    autocorrelation is not credited (never claims more significance than the raw
    sample). Returns (n_eff, r1)."""
    import numpy as np
    arr = np.asarray([float(r) for r in (returns or []) if r is not None], dtype=float)
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
    """MAE/MFE per trade from the close path between entry and exit, the
    intra-trade 'how far did it go against / for me' view. Excursion is signed by
    side (favorable positive). A trade with no covered prices reports None."""
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
                mae, mfe = round(min(favs), 4), round(max(favs), 4)   # worst adverse, best favorable
                maes.append(mae)
                mfes.append(mfe)
                n_cov += 1
        per.append({"ticker": tk, "entry_date": ed, "exit_date": xd, "mae_pct": mae, "mfe_pct": mfe})
    return {"per_trade": per, "n_covered": n_cov,
            "avg_mae_pct": round(sum(maes) / len(maes), 4) if maes else None,
            "avg_mfe_pct": round(sum(mfes) / len(mfes), 4) if mfes else None}


def _regime_attribution(trades: list, regime_series: dict) -> dict:
    """Bucket each trade's P&L by the regime label at its entry date (caller-
    supplied {date: regime}, so the tool stays no-fetch)."""
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
    return {reg: {"n_trades": b["n"], "total_pnl": round(b["pnl"], 2),
                  "win_rate_pct": round(b["wins"] / b["n"] * 100.0, 2) if b["n"] else None}
            for reg, b in buckets.items()}


def _attribution(bt: dict, returns: list, prices, factor_returns, regime_series,
                 risk_free_rate: float, not_computed: dict) -> dict:
    """MAE/MFE + factor + regime attribution. Each leg is OPTIONAL and pure: it
    runs only when its caller-supplied input is present, else names itself in
    not_computed (never fetched, never fabricated)."""
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
        not_computed["attribution.factors"] = "supply factor_returns {factor: [...]} to attribute returns (no factor vendor)"
    if regime_series:
        attribution["regime"] = _regime_attribution(trades, regime_series)
    else:
        not_computed["attribution.regime"] = "supply regime_series {date: regime} to bucket trade P&L by regime at entry"
    return attribution


def _cost_report(bt: dict, perf, risk_free_rate: float) -> Optional[dict]:
    """Turnover + net-of-cost vs gross Sharpe from the sim's cost bookkeeping.

    The equity curve is ALREADY net of slippage + commission + impact (they are
    deducted from cash inside run_backtest), so the served Sharpe IS the
    net-of-cost Sharpe. We add the per-bar cost drag back to reconstruct the
    GROSS (cost-free) curve and report both, plus turnover, the cost demolition
    a capacity-blind backtest hides."""
    cb = bt.get("cost_breakdown")
    if not isinstance(cb, dict):
        return None
    init_cap = float(bt.get("initial_capital") or 0.0) or None
    n_bars = int(bt.get("n_bars") or 0)
    gross_notional = float(bt.get("gross_traded_notional") or 0.0)
    total_cost = float(cb.get("total") or 0.0)
    turnover_ratio = (gross_notional / init_cap) if init_cap else None
    out = {
        "cost_breakdown": cb,
        "cost_drag_pct": round(total_cost / init_cap * 100.0, 4) if init_cap else None,
        "turnover_ratio": round(turnover_ratio, 4) if turnover_ratio is not None else None,
        "annualized_turnover": (round(turnover_ratio * 252.0 / n_bars, 4)
                                if (turnover_ratio is not None and n_bars) else None),
        "net_of_cost_sharpe": (perf.get("sharpe_annualized") if isinstance(perf, dict) else None),
        "gross_sharpe": None,
        "sharpe_cost_drag": None,
        "capacity": bt.get("capacity"),
    }
    # Reconstruct the gross (cost-free) equity curve: gross_i = net_i + Σ cost_≤i.
    cpb = bt.get("costs_per_bar")
    eq = bt.get("equity_curve")
    if isinstance(cpb, list) and isinstance(eq, list) and len(cpb) == len(eq) and len(eq) > 1:
        from .performance import performance_report
        cum = 0.0
        gross_eq = []
        for i, e in enumerate(eq):
            cum += float(cpb[i])
            gross_eq.append(e + cum)
        gross_rets = [round((gross_eq[i] - gross_eq[i - 1]) / gross_eq[i - 1], 8)
                      for i in range(1, len(gross_eq)) if gross_eq[i - 1]]
        gp = performance_report(gross_rets, equity_curve=gross_eq, risk_free_rate=risk_free_rate)
        if isinstance(gp, dict) and "error" not in gp:
            out["gross_sharpe"] = gp.get("sharpe_annualized")
            if out["net_of_cost_sharpe"] is not None and out["gross_sharpe"] is not None:
                out["sharpe_cost_drag"] = round(out["gross_sharpe"] - out["net_of_cost_sharpe"], 4)
    return out


def score_backtest(bt: dict, *, n_trials: int = 1, risk_free_rate: float = 0.0,
                   benchmark_returns: Optional[list] = None, pnl_matrix=None,
                   trials_sharpe_std: Optional[float] = None, prices: Optional[dict] = None,
                   factor_returns: Optional[dict] = None, regime_series: Optional[dict] = None,
                   cpcv: bool = False, cpcv_n_groups: int = 8, cpcv_n_test_groups: int = 2,
                   cpcv_purge: int = 1, cpcv_embargo: int = 1) -> dict:
    """Score a run_backtest result: performance_report + deflated-Sharpe (+ PBO
    when a parameter-sweep `pnl_matrix` is supplied) -> a MOAT-GATED verdict.

    `n_trials` is the number of strategy configs the caller searched; it deflates
    the Sharpe for multiple testing. The 'edge' verdict is structurally
    unreachable unless a deflated_sharpe figure is populated (the moat).

    The validation block always surfaces the AR(1)-honest effective sample size
    (`n_obs_effective`) and the Harvey-Liu multiple-testing hurdle (`sharpe_tstat`
    vs t > 3), and MinTRL (is the sample long enough to trust the Sharpe?). Set
    `cpcv=True` to add Combinatorial Purged Cross-Validation, the OOS Sharpe/DSR
    distribution across many purged held-out partitions. When the sim carried
    costs, a `costs` block reports turnover + net-of-cost vs gross Sharpe."""
    from .performance import performance_report
    from .validation import deflated_sharpe, pbo_cscv, min_track_record_length, cpcv_score

    returns = bt.get("returns") or []
    perf = performance_report(returns, equity_curve=bt.get("equity_curve"),
                              benchmark_returns=benchmark_returns, risk_free_rate=risk_free_rate)
    dsr = deflated_sharpe(returns, n_trials=max(1, int(n_trials)), trials_sharpe_std=trials_sharpe_std)
    pbo = pbo_cscv(pnl_matrix) if pnl_matrix is not None else None
    mintrl = min_track_record_length(returns)

    dsr_ok = isinstance(dsr, dict) and "error" not in dsr
    pbo_ok = isinstance(pbo, dict) and "error" not in pbo
    mintrl_ok = isinstance(mintrl, dict) and "error" not in mintrl
    dsr_val = dsr.get("deflated_sharpe") if dsr_ok else None
    psr_val = dsr.get("psr_vs_zero") if dsr_ok else None
    pbo_val = pbo.get("pbo") if pbo_ok else None

    # A2, DSR honesty: AR(1)-effective N + the Harvey-Liu t > 3 hurdle.
    n_eff, r1 = _ar1_effective_n(returns)
    sr_pp = dsr.get("sharpe_per_period") if dsr_ok else None
    if sr_pp is None:
        arr_len = len([r for r in returns if r is not None])
        from .validation import _per_period_sharpe
        import numpy as _np
        sr_pp = _per_period_sharpe(_np.asarray([float(r) for r in returns if r is not None], dtype=float)) if arr_len >= 2 else 0.0
    t_stat = float(sr_pp) * math.sqrt(n_eff)

    validation = {"deflated_sharpe": dsr_val, "pbo": pbo_val, "psr": psr_val,
                  "n_trials": int(n_trials), "verdict": _verdict(dsr_val, pbo_val),
                  # A2, autocorrelation-honest significance.
                  "n_obs": (dsr.get("n_obs") if dsr_ok else len([r for r in returns if r is not None])),
                  "n_obs_effective": round(n_eff, 1),
                  "autocorr_lag1": round(r1, 4),
                  "sharpe_tstat": round(t_stat, 4),
                  "harvey_liu_hurdle": 3.0,
                  "passes_harvey_liu": bool(t_stat > 3.0),
                  # A3, MinTRL: is the sample long enough to trust the Sharpe? (Bailey/LdP)
                  "min_track_record_length": (mintrl.get("min_track_record_length") if mintrl_ok else None),
                  "min_track_record_years": (mintrl.get("min_track_record_years") if mintrl_ok else None),
                  "track_length_sufficient": (mintrl.get("sufficient") if mintrl_ok else None),
                  "track_record_shortfall_obs": (mintrl.get("shortfall_obs") if mintrl_ok else None)}

    not_computed: dict = {}
    if not (isinstance(perf, dict) and "error" not in perf):
        not_computed["performance"] = (perf or {}).get("error", "insufficient observations (need >= 30 bars)")
        perf = None
    if not dsr_ok:
        not_computed["validation.deflated_sharpe"] = dsr.get("error", "insufficient observations (need >= 8 bars)")
    if pbo is None:
        not_computed["validation.pbo"] = "supply a parameter-sweep pnl_matrix (one column per config) to run PBO"
    elif not pbo_ok:
        not_computed["validation.pbo"] = pbo.get("error", "pbo unavailable")

    # A1, CPCV (opt-in): OOS Sharpe/DSR distribution across purged partitions.
    if cpcv:
        cp = cpcv_score(returns, n_groups=cpcv_n_groups, n_test_groups=cpcv_n_test_groups,
                        purge=cpcv_purge, embargo=cpcv_embargo, n_trials=n_trials)
        if isinstance(cp, dict) and "error" not in cp:
            validation["cpcv"] = cp
        else:
            not_computed["validation.cpcv"] = (cp or {}).get("error", "cpcv unavailable")
    else:
        not_computed["validation.cpcv"] = "set cpcv=true to run Combinatorial Purged Cross-Validation (OOS path distribution)"

    attribution = _attribution(bt, returns, prices, factor_returns, regime_series,
                               risk_free_rate, not_computed)

    # C2, turnover + net-of-cost vs gross Sharpe (from the sim's cost bookkeeping).
    costs = _cost_report(bt, perf, risk_free_rate)

    return {
        "performance": perf,
        "validation": validation,
        "dsr_detail": dsr if dsr_ok else None,
        "trade_summary": _trade_summary(bt.get("trades") or []),
        "attribution": attribution,
        "costs": costs,
        "not_computed": not_computed,
    }


__all__ = ["run_backtest", "score_backtest"]
