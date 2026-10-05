"""Audit tests for the math the goldens do not pin.

These lock the documented formulas (purge windows, fill timing, sample
correlation, target semideviation's neighbours). They do not replace
tests/test_goldens.py. A failure here is a formula that no longer matches
the comment next to it, which is the thing to fix — not the assertion.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from alphaengine.core import (
    compute_var_cvar,
    cpcv_score,
    drawdown_anatomy,
    overlap_stats,
    performance_report,
    run_backtest,
    score_backtest,
    subperiod_stability,
    technical_features,
)
from alphaengine.core.validation import _per_period_sharpe


def test_cpcv_purge_and_embargo_drop_the_edges_of_each_held_out_group() -> None:
    """Each test group loses `purge` observations at the front and `embargo` at the back.

    The path Sharpe is then the annualised per-period Sharpe of what remains,
    which is the definition in cpcv_score. Computed here from the same slices
    so a shift of the window fails even when the headline still looks plausible.
    """
    arr = np.linspace(-0.01, 0.02, 32)
    groups = list(np.array_split(np.arange(arr.size), 4))
    paths = []
    for group in groups:
        block = group[1:-1]
        path = arr[block]
        paths.append(_per_period_sharpe(path) * math.sqrt(252))

    got = cpcv_score(arr.tolist(), n_groups=4, n_test_groups=1, purge=1, embargo=1)
    assert got["n_paths"] == 4
    assert got["purge"] == 1
    assert got["embargo"] == 1
    assert got["sharpe_annualized"]["mean"] == round(float(np.mean(paths)), 4)

    untouched = cpcv_score(arr.tolist(), n_groups=4, n_test_groups=1, purge=0, embargo=0)
    assert untouched["sharpe_annualized"]["mean"] != got["sharpe_annualized"]["mean"]


def test_cpcv_with_nothing_left_after_purge_is_an_absence() -> None:
    got = cpcv_score(list(np.linspace(-0.01, 0.01, 16)), n_groups=4, n_test_groups=1, purge=2, embargo=2)
    assert "error" in got
    assert got["n_groups"] == 4


def test_close_fill_uses_that_bars_close_and_next_open_waits() -> None:
    dates = [f"2020-01-{day:02d}" for day in range(2, 12)]
    prices = {"AAA": [{"date": date, "open": 50.0 + i, "close": 100.0 + i} for i, date in enumerate(dates)]}
    signals = {
        "AAA": [
            {"date": dates[0], "target_weight": 1.0},
            {"date": dates[1], "target_weight": 0.0},
        ]
    }
    on_close = run_backtest(signals, prices, slippage_bps=0, commission_bps=0, fill_timing="close")
    on_open = run_backtest(signals, prices, slippage_bps=0, commission_bps=0, fill_timing="next_open")

    close_entry = next(t for t in on_close["trades"] if t["side"] == "long")
    open_entry = next(t for t in on_open["trades"] if t["side"] == "long")
    assert close_entry["entry_date"] == dates[0]
    assert close_entry["entry_price"] == 100.0
    assert open_entry["entry_date"] == dates[1]
    assert open_entry["entry_price"] == 51.0
    # The signal on day 0 is not filled at day 0's open, which printed before it.
    assert open_entry["entry_price"] != prices["AAA"][0]["open"]


def test_residual_close_completes_the_trade_log_and_not_the_equity() -> None:
    dates = [f"2020-02-{day:02d}" for day in range(1, 8)]
    prices = {"AAA": [{"date": date, "close": 100.0 + i} for i, date in enumerate(dates)]}
    signals = {"AAA": [{"date": dates[0], "target_weight": 1.0}]}
    bt = run_backtest(signals, prices, slippage_bps=0, commission_bps=0, fill_timing="close")

    assert bt["final_equity"] == bt["equity_curve"][-1]
    assert len(bt["equity_curve"]) == len(dates)
    assert bt["trades"][-1]["exit_date"] == dates[-1]
    assert bt["trades"][-1]["side"] == "long"


def test_unrecorded_trial_count_cannot_reach_edge() -> None:
    returns = np.random.default_rng(1).normal(0.002, 0.005, 80)
    bt = {
        "returns": returns.tolist(),
        "equity_curve": list(np.cumprod(1.0 + returns)),
        "trades": [],
        "costs_per_bar": [0.0] * 81,
    }
    omitted = score_backtest(bt, n_trials=None)
    recorded = score_backtest(bt, n_trials=1)
    assert omitted["validation"]["n_trials"] is None
    assert omitted["validation"]["n_trials_source"] == "not_recorded"
    assert omitted["validation"]["verdict"] != "edge"
    assert recorded["validation"]["n_trials_source"] == "asserted"


def test_score_backtest_cpcv_is_opt_in_and_reports_paths() -> None:
    bt = {"returns": list(np.linspace(-0.002, 0.004, 64)), "trades": [], "equity_curve": None}
    off = score_backtest(bt, n_trials=1, cpcv=False)
    on = score_backtest(
        bt, n_trials=1, cpcv=True, cpcv_n_groups=4, cpcv_n_test_groups=1, cpcv_purge=0, cpcv_embargo=0
    )
    assert "cpcv" not in off["validation"]
    assert on["validation"]["cpcv"]["n_paths"] > 0
    assert on["validation"]["cpcv"]["purge"] == 0


def test_sma_ema_rsi_and_atr_match_their_definitions() -> None:
    closes = [float(v) for v in range(1, 17)]
    features = technical_features(
        {"AAA": closes},
        sma_windows=[3],
        ema_windows=[3],
        rsi_window=14,
        atr_window=3,
    )["features"]["AAA"]
    assert features["sma"]["3"]["value"] == 15.0  # mean of 14, 15, 16
    # EMA(3): seed = mean(1,2,3) = 2, alpha = 0.5, then fold 4..16.
    ema = 2.0
    for price in closes[3:]:
        ema = 0.5 * price + 0.5 * ema
    assert features["ema"]["3"]["value"] == round(ema, 6)
    assert features["rsi"]["value"] == 100.0  # every delta is a gain
    assert features["atr"]["available"] is False

    rows = [
        {"date": f"d{i}", "open": 10.0, "high": 12.0, "low": 9.0, "close": 10.0 + i * 0.0} for i in range(6)
    ]
    # Constant close, high-low range is 3. True range is max(3, |12-10|, |9-10|) = 3
    # until the previous close moves. close is constant 10, so TR = 3 every bar.
    atr = technical_features({"AAA": rows}, sma_windows=[], atr_window=3)["features"]["AAA"]["atr"]
    assert atr["value"] == 3.0


def test_share_of_pnl_is_the_best_segment_over_absolute_segment_pnl() -> None:
    """max(segment pnl) / sum(|segment pnl|), the figure the docstring names.

    A negative segment is in the denominator as a magnitude, so it cannot
    shrink the share by cancelling the stretches that lost money.
    """
    segment = [0.01] * 20
    losing = [-0.01] * 20
    best = [0.05] * 20
    returns = segment + segment + losing + best
    got = subperiod_stability(returns, segments=4)
    share = 1.0 / (0.2 + 0.2 + 0.2 + 1.0)
    assert got["share_of_pnl_in_best_segment"] == round(share, 6)


def test_overlap_correlation_matches_the_sample_coefficient() -> None:
    rng = np.random.default_rng(4)
    candidate = rng.normal(0.0004, 0.01, 40)
    book = 0.3 * candidate + rng.normal(0, 0.004, 40)
    got = overlap_stats(candidate.tolist(), book.tolist())
    assert got["correlation"] == round(float(np.corrcoef(candidate, book)[0, 1]), 6)
    sample_cov = float(np.cov(candidate, book, ddof=1)[0, 1])
    sample_var = float(np.var(book, ddof=1))
    assert got["beta_to_book"] == round(sample_cov / sample_var, 6)


def test_drawdown_is_a_positive_magnitude() -> None:
    # Two up days, then a 10% drop that stays down, then a recovery.
    returns = [0.01, 0.01, -0.10, 0.0, 0.12]
    got = drawdown_anatomy(returns)
    equity = np.concatenate(([1.0], np.cumprod(1.0 + np.asarray(returns))))
    peak = np.maximum.accumulate(equity)
    depth = float(-(equity / peak - 1.0).min()) * 100.0
    assert got["max_drawdown_pct"] == round(depth, 6)
    assert got["max_drawdown_pct"] > 0


def test_parametric_var_includes_the_mean_over_the_horizon() -> None:
    rng = np.random.default_rng(5)
    returns = rng.normal(0.0, 0.01, 80).tolist()
    got = compute_var_cvar(returns, confidence=0.95, horizon_days=1)
    z = float(__import__("scipy").stats.norm.ppf(0.95))
    sigma = float(np.std(returns, ddof=1))
    mean = float(np.mean(returns))
    loss = max(0.0, -mean + z * sigma)
    assert got["parametric"]["var_pct"] == round(loss * 100, 2)


def test_performance_sortino_uses_target_semideviation_over_all_periods() -> None:
    report = performance_report([0.002] * 20 + [-0.001] * 20)
    excess = np.asarray([0.002] * 20 + [-0.001] * 20)  # rf default 0
    downside = np.minimum(excess, 0.0)
    dsd = float(np.sqrt(np.mean(downside**2)))
    sortino = (float(excess.mean()) / dsd) * math.sqrt(252)
    assert report["sortino_ratio"] == round(sortino, 4)


def test_the_default_cash_rate_is_zero_until_a_bill_is_supplied() -> None:
    """Neither formula invents a Treasury bill.

    A missing rate is excess of zero, in the performance report and in the
    factor regression. Passing 4% is how a study uses a bill, and that changes
    the intercept. Ken French factors are already excess, so the default does
    not subtract the rate from them a second time.
    """
    statsmodels = pytest.importorskip("statsmodels")
    assert statsmodels is not None
    from alphaengine.core.factors import _DEFAULT_RFR, decompose_factors

    rng = np.random.default_rng(6)
    market = rng.normal(0.0003, 0.01, 80)
    portfolio = (0.0002 + market).tolist()
    factors = {"market": market.tolist()}
    at_default = decompose_factors(portfolio, factors)
    at_bill = decompose_factors(portfolio, factors, risk_free_rate=0.04)
    assert _DEFAULT_RFR == 0.0
    assert performance_report(portfolio)["risk_free_rate"] == 0.0
    assert at_default["risk_free_rate"] == 0.0
    assert at_default["factors_are_excess"] is True
    assert at_default["alpha_annualization"] == "arithmetic"
    assert at_default["factor_betas"]["market"] == pytest.approx(1.0, abs=0.05)
    assert at_default["alpha"] == decompose_factors(portfolio, factors, risk_free_rate=0.0)["alpha"]
    assert at_bill["alpha"] != at_default["alpha"]


def test_spread_is_log_price_minus_hedge_times_log_price() -> None:
    pytest.importorskip("statsmodels")
    from alphaengine.core.pairs import compute_spread, compute_spread_signal

    a = np.array([10.0, 11.0, 12.0, 13.0])
    b = np.array([5.0, 5.5, 6.0, 6.5])
    got = compute_spread(a, b, 1.0)
    assert np.allclose(got, np.log(a) - np.log(b))

    dated_a = {"2020-01-02": 10.0, "2020-01-03": 11.0, "2020-01-04": 99.0}
    dated_b = {"2020-01-03": 5.0, "2020-01-04": 6.0}
    # Intersection only: the 2nd is not in both, so it cannot enter the spread.
    from alphaengine.core.pairs import _align_pair

    left, right = _align_pair(dated_a, dated_b)
    assert left == [11.0, 99.0]
    assert right == [5.0, 6.0]

    rng = np.random.default_rng(7)
    walk = np.cumsum(rng.normal(0, 0.01, 400))
    noise = rng.normal(0, 0.002, 400)
    leg_b = np.exp(walk + 4.0)
    leg_a = np.exp(1.5 * walk + noise + 4.0)
    signal = compute_spread_signal(leg_a.tolist(), leg_b.tolist(), symbol_a="AAA", symbol_b="BBB")
    assert signal["hedge_ratio"] == pytest.approx(1.5, abs=0.15)
    assert signal["n_observations"] == 400
