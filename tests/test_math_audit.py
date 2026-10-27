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
    combinatorial_oos_stability,
    compute_var_cvar,
    cpcv_score,
    deflated_sharpe,
    drawdown_anatomy,
    overlap_stats,
    pbo_cscv,
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
    which is the definition in combinatorial_oos_stability. Computed here from the same slices
    so a shift of the window fails even when the headline still looks plausible.
    """
    from alphaengine.core.rates import TREASURY_BILL_RATE

    arr = np.linspace(-0.01, 0.02, 32)
    groups = list(np.array_split(np.arange(arr.size), 4))
    rf = TREASURY_BILL_RATE / 252
    paths = []
    for group in groups:
        block = group[1:-1]
        path = arr[block]
        paths.append(_per_period_sharpe(path, risk_free_per_period=rf) * math.sqrt(252))

    got = combinatorial_oos_stability(arr.tolist(), n_groups=4, n_test_groups=1, purge=1, embargo=1)
    assert got["n_paths"] == 4
    assert got["purge"] == 1
    assert got["embargo"] == 1
    assert got["sharpe_annualized"]["mean"] == round(float(np.mean(paths)), 4)

    untouched = combinatorial_oos_stability(arr.tolist(), n_groups=4, n_test_groups=1, purge=0, embargo=0)
    assert untouched["sharpe_annualized"]["mean"] != got["sharpe_annualized"]["mean"]

    with pytest.deprecated_call(match="combinatorial_oos_stability"):
        alias_got = cpcv_score(arr.tolist(), n_groups=4, n_test_groups=1, purge=1, embargo=1)
    assert alias_got == got


def test_cpcv_with_nothing_left_after_purge_is_an_absence() -> None:
    got = combinatorial_oos_stability(
        list(np.linspace(-0.01, 0.01, 16)), n_groups=4, n_test_groups=1, purge=2, embargo=2
    )
    assert "error" in got
    assert got["n_groups"] == 4


def test_deflated_sharpe_n_trials_one_returns_psr_with_note() -> None:
    r = np.linspace(-0.005, 0.015, 60).tolist()
    got = deflated_sharpe(r, n_trials=1)
    assert got["sr0_expected_max"] == 0.0
    assert got["deflated_sharpe"] == got["psr_vs_zero"]
    assert "no multiple-testing deflation applies" in got["note"]


def test_pbo_cscv_validates_n_splits_even_and_at_least_two() -> None:
    m = np.random.default_rng(1).normal(0.0, 0.01, size=(40, 4)).tolist()
    for bad in (0, 1, 3, 5, -2, True):
        with pytest.raises(ValueError, match="n_splits must be an even integer >= 2"):
            pbo_cscv(m, n_splits=bad)  # type: ignore[arg-type]


def test_sweep_verdict_uses_cross_trial_sharpe_dispersion() -> None:
    from alphaengine.sweep import sweep

    base = np.random.default_rng(42).normal(0.0, 0.01, 300)

    def bt(data=None, drift: float = 0.0):
        arr = np.asarray(data, dtype=float)
        return (arr + float(drift)).tolist()

    res = sweep(bt, {"drift": [-0.004, -0.002, 0.0, 0.001, 0.002]}, data=base)
    v = res.verdict()
    ok = [t for t in res.trials if t.failed is None]
    expected_std = float(np.std([t.sharpe for t in ok], ddof=1))
    best_col = res.matrix[:, res.best.index].tolist()
    direct = deflated_sharpe(best_col, n_trials=res.n_trials, trials_sharpe_std=expected_std)
    no_disp = deflated_sharpe(best_col, n_trials=res.n_trials)
    assert v["deflated_sharpe"] == direct["deflated_sharpe"]
    assert v["sr0_expected_max"] == direct["sr0_expected_max"]
    assert v["sr0_expected_max"] > no_disp["sr0_expected_max"]


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
    from alphaengine.core.rates import TREASURY_BILL_RATE

    series = [0.002] * 20 + [-0.001] * 20
    report = performance_report(series)
    excess = np.asarray(series) - TREASURY_BILL_RATE / 252
    downside = np.minimum(excess, 0.0)
    dsd = float(np.sqrt(np.mean(downside**2)))
    sortino = (float(excess.mean()) / dsd) * math.sqrt(252)
    assert report["sortino_ratio"] == round(sortino, 4)


def test_an_omitted_cash_rate_is_the_pinned_three_month_bill() -> None:
    """Zero is a claim the caller has to make. Omitting the rate is the bill.

    The pin is the H.15 3-month Treasury constant maturity, investment basis,
    not a download. Passing 0 is excess of zero and does not match the pin.
    Passing the pin explicitly matches the omitted rate, and the source says
    the caller supplied it.
    """
    statsmodels = pytest.importorskip("statsmodels")
    assert statsmodels is not None
    from alphaengine.core.factors import decompose_factors
    from alphaengine.core.rates import TREASURY_BILL_AS_OF, TREASURY_BILL_RATE, TREASURY_BILL_SOURCE

    rng = np.random.default_rng(6)
    market = rng.normal(0.0003, 0.01, 80)
    portfolio = (0.0002 + market).tolist()
    factors = {"market": market.tolist()}
    at_default = decompose_factors(portfolio, factors)
    at_zero = decompose_factors(portfolio, factors, risk_free_rate=0.0)
    at_pin = decompose_factors(portfolio, factors, risk_free_rate=TREASURY_BILL_RATE)
    report = performance_report(portfolio)
    assert report["risk_free_rate"] == TREASURY_BILL_RATE
    assert report["risk_free_rate_as_of"] == TREASURY_BILL_AS_OF
    assert report["risk_free_rate_source"] == TREASURY_BILL_SOURCE
    assert at_default["risk_free_rate"] == TREASURY_BILL_RATE
    assert at_default["risk_free_rate_as_of"] == TREASURY_BILL_AS_OF
    assert at_default["factors_are_excess"] is True
    assert at_default["alpha_annualization"] == "arithmetic"
    assert at_default["hac_rule"] == "newey_west_1994"
    assert at_default["hac_maxlags"] == 3  # floor(4 * (80/100)^(2/9))
    assert at_default["factor_betas"]["market"] == pytest.approx(1.0, abs=0.05)
    assert at_default["alpha"] == at_pin["alpha"]
    assert at_pin["risk_free_rate_source"] == "caller"
    assert at_pin["risk_free_rate_as_of"] is None
    assert at_zero["alpha"] != at_default["alpha"]
    assert performance_report(portfolio, risk_free_rate=0.0)["risk_free_rate"] == 0.0


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


def test_a_penny_name_is_a_price_and_a_flat_return_is_not() -> None:
    from alphaengine.core.relations import _looks_like_prices, _to_returns

    penny = [0.40 + i * 0.002 for i in range(30)]
    assert _looks_like_prices(penny) is True
    assert _to_returns(penny)[0] == pytest.approx(penny[1] / penny[0] - 1.0, abs=1e-8)

    flat_return = [0.02] * 30
    assert _looks_like_prices(flat_return) is False
    assert _to_returns(flat_return) == [0.02] * 30

    # A two-bar gap from 100 to 50 is still a price. The median clears 1.5,
    # so one large move does not turn the closes into a return of 100.
    assert _looks_like_prices([100.0, 50.0]) is True
    assert _looks_like_prices([0.01, 0.02, -0.01]) is False


def test_the_pair_screen_keeps_only_the_fdr_survivors() -> None:
    pytest.importorskip("statsmodels")
    from alphaengine.core.pairs import _benjamini_hochberg

    # The smallest clears alpha/m. The rest sit above alpha, so they do not
    # pull the step-up cutoff forward.
    flags = _benjamini_hochberg([0.20] * 9 + [0.001], 0.05)
    assert flags[-1] is True
    assert flags[:-1] == [False] * 9
    # One pair is the raw test.
    assert _benjamini_hochberg([0.04], 0.05) == [True]
    assert _benjamini_hochberg([0.06], 0.05) == [False]


def test_array_like_list_ndarray_and_series_across_validation_and_metrics() -> None:
    """deflated_sharpe, min_track_record_length, combinatorial_oos_stability,
    cpcv_score, pbo_cscv, performance_report, and compute_var_cvar accept
    Python lists, numpy ndarrays, and pandas Series (whose __bool__ raises
    ValueError) and return identical figures.
    """
    from alphaengine.core import min_track_record_length

    class _FakeSeries:
        """Mimics pandas.Series: array-like via __array__, raises ValueError on bool()."""

        def __init__(self, data: list[float]) -> None:
            self._arr = np.asarray(data, dtype=float)

        def __bool__(self) -> bool:
            raise ValueError("The truth value of a Series is ambiguous.")

        def __len__(self) -> int:
            return int(self._arr.size)

        def __iter__(self):
            return iter(self._arr)

        def __array__(self, dtype=None):
            return np.asarray(self._arr, dtype=dtype)

    try:
        import pandas as pd

        series_cls = pd.Series
        df_cls = pd.DataFrame
    except ImportError:
        series_cls = _FakeSeries
        df_cls = None

    rng = np.random.default_rng(17)
    as_list = rng.normal(0.0008, 0.012, 120).tolist()
    as_ndarray = np.asarray(as_list, dtype=float)
    as_series = series_cls(as_list)
    as_fake_series = _FakeSeries(as_list)

    bench_list = rng.normal(0.0004, 0.01, 120).tolist()
    bench_ndarray = np.asarray(bench_list, dtype=float)
    bench_series = series_cls(bench_list)

    # 1. deflated_sharpe & min_track_record_length
    dsr_list = deflated_sharpe(as_list, n_trials=9)
    mart_list = min_track_record_length(as_list)
    for candidate in (as_ndarray, as_series, as_fake_series):
        assert deflated_sharpe(candidate, n_trials=9) == dsr_list
        assert min_track_record_length(candidate) == mart_list

    # 2. combinatorial_oos_stability & cpcv_score
    stab_list = combinatorial_oos_stability(as_list, n_groups=6, n_test_groups=2, n_trials=9)
    for candidate in (as_ndarray, as_series, as_fake_series):
        assert combinatorial_oos_stability(candidate, n_groups=6, n_test_groups=2, n_trials=9) == stab_list
        with pytest.deprecated_call(match="combinatorial_oos_stability"):
            assert cpcv_score(candidate, n_groups=6, n_test_groups=2, n_trials=9) == stab_list

    # 3. pbo_cscv (2-D list, 2-D ndarray, list of Series, and DataFrame if pandas present)
    mat_list = rng.normal(0.0005, 0.01, size=(80, 6)).tolist()
    mat_ndarray = np.asarray(mat_list, dtype=float)
    mat_series_rows = [series_cls(row) for row in mat_list]
    pbo_expected = pbo_cscv(mat_list, n_splits=8)
    assert pbo_cscv(mat_ndarray, n_splits=8) == pbo_expected
    assert pbo_cscv(mat_series_rows, n_splits=8) == pbo_expected
    if df_cls is not None:
        assert pbo_cscv(df_cls(mat_list), n_splits=8) == pbo_expected

    # 4. performance_report & compute_var_cvar
    perf_list = performance_report(as_list, benchmark_returns=bench_list)
    risk_list = compute_var_cvar(as_list)
    for candidate, bench in (
        (as_ndarray, bench_ndarray),
        (as_series, bench_series),
        (as_fake_series, _FakeSeries(bench_list)),
    ):
        assert performance_report(candidate, benchmark_returns=bench) == perf_list
        assert compute_var_cvar(candidate) == risk_list


def test_cornish_fisher_loss_z_known_answers_and_skew_direction() -> None:
    """Milestone 1 (H1): Cornish-Fisher loss z-score sign convention.

    At confidence=0.95 (z = 1.6448536):
      - S=-1, K=0 -> 1.910328 (old bug returned 1.341814)
      - S=+1, K=0 -> 1.341814
      - S=0,  K=3 -> 1.584311
      - S=0,  K=0 -> 1.644854
    """
    from alphaengine.core import cornish_fisher_loss_z

    assert cornish_fisher_loss_z(0.95, -1.0, 0.0) == pytest.approx(1.910328, abs=1e-5)
    assert cornish_fisher_loss_z(0.95, 1.0, 0.0) == pytest.approx(1.341814, abs=1e-5)
    assert cornish_fisher_loss_z(0.95, 0.0, 3.0) == pytest.approx(1.584311, abs=1e-5)
    assert cornish_fisher_loss_z(0.95, 0.0, 0.0) == pytest.approx(1.644854, abs=1e-5)

    # Seeded fixture with negative skew (~ -1.4) and positive excess kurtosis (~ 3.7)
    rng = np.random.default_rng(7)
    r = rng.normal(0.0, 0.0052, 4000) - 0.010 * (rng.exponential(1.0, 4000) - 1.0)
    r = r - r.mean()

    res = compute_var_cvar(r, confidence=0.95)
    cf = res["cornish_fisher"]
    assert cf["skewness"] == pytest.approx(-1.4, abs=0.1)
    assert cf["excess_kurtosis"] == pytest.approx(3.7, abs=0.4)

    gauss_var = res["parametric"]["var_pct"]
    cf_var = cf["var_pct"]
    hist_var = res["historical"]["var_pct"]
    assert cf_var > gauss_var
    assert abs(cf_var - hist_var) / hist_var <= 0.25

    # On the mirrored series (-r), positive skewness lowers CF VaR below Gaussian VaR
    res_mirrored = compute_var_cvar(-r, confidence=0.95)
    assert res_mirrored["cornish_fisher"]["var_pct"] < res_mirrored["parametric"]["var_pct"]


def test_correlated_trials_dsr_luck_bar_audit_case_and_n_eff_participation() -> None:
    """Milestone 2a (H5): correlated-trial DSR without Lo SE floor + n_eff_participation.

    1. Luck bar: seed=7, N=100 noise trials, T=1260 daily obs.
       - rho=0.8: sr0_expected_max * sqrt(252) within +/-0.06 of 0.51 and < 0.60
       - rho=0.0: sr0_expected_max * sqrt(252) within +/-0.08 of 1.13
    2. Audit case: N=50, rho=0.95, T=1000. Per-period sr0_expected_max drops ~4x
       from the floored ~0.072 to ~0.017 (+/-20%).
    3. Exact equicorrelated N=100, rho=0.8 matrix: n_eff_participation equals
       100^2 / ((1 + 99*0.8)^2 + 99*0.2^2) to 1e-9.
    4. Failed trials do not count toward N in SweepResult.verdict().
    """
    from alphaengine.core import n_eff_participation
    from alphaengine.sweep import sweep

    # 1. Luck bar (seed 7, N=100, T=1260)
    n_trials_luck, t_luck = 100, 1260
    z_raw = np.random.default_rng(7).standard_normal((n_trials_luck, t_luck)).T

    c_corr = np.full((n_trials_luck, n_trials_luck), 0.8)
    np.fill_diagonal(c_corr, 1.0)
    mat_corr = (z_raw @ np.linalg.cholesky(c_corr).T) * 0.01
    sharpes_corr = [float(mat_corr[:, j].mean() / mat_corr[:, j].std(ddof=1)) for j in range(n_trials_luck)]
    std_corr = float(np.std(sharpes_corr, ddof=1))
    dsr_corr = deflated_sharpe(mat_corr[:, 0], n_trials=n_trials_luck, trials_sharpe_std=std_corr)
    sr0_ann_corr = dsr_corr["sr0_expected_max"] * math.sqrt(252)
    assert sr0_ann_corr == pytest.approx(0.51, abs=0.06)
    assert sr0_ann_corr < 0.60

    mat_indep = z_raw * 0.01
    sharpes_indep = [
        float(mat_indep[:, j].mean() / mat_indep[:, j].std(ddof=1)) for j in range(n_trials_luck)
    ]
    std_indep = float(np.std(sharpes_indep, ddof=1))
    dsr_indep = deflated_sharpe(mat_indep[:, 0], n_trials=n_trials_luck, trials_sharpe_std=std_indep)
    sr0_ann_indep = dsr_indep["sr0_expected_max"] * math.sqrt(252)
    assert sr0_ann_indep == pytest.approx(1.13, abs=0.08)

    # 2. Audit case: N=50, rho=0.95, T=1000
    rng_audit = np.random.default_rng(7)
    n_audit, t_audit, rho_audit = 50, 1000, 0.95
    common_a = rng_audit.normal(0.0, 0.01, size=(t_audit, 1))
    idio_a = rng_audit.normal(0.0, 0.01, size=(t_audit, n_audit))
    mat_audit = math.sqrt(rho_audit) * common_a + math.sqrt(1.0 - rho_audit) * idio_a
    sharpes_audit = [float(mat_audit[:, j].mean() / mat_audit[:, j].std(ddof=1)) for j in range(n_audit)]
    std_audit = float(np.std(sharpes_audit, ddof=1))
    dsr_audit = deflated_sharpe(mat_audit[:, 0], n_trials=n_audit, trials_sharpe_std=std_audit)
    floored_audit = deflated_sharpe(mat_audit[:, 0], n_trials=n_audit, trials_sharpe_std=None)
    assert floored_audit["sr0_expected_max"] == pytest.approx(0.072, rel=0.10)
    assert dsr_audit["sr0_expected_max"] == pytest.approx(0.017, rel=0.20)

    # 3. Exact equicorrelated N=100, rho=0.8 correlation matrix -> n_eff_participation
    n_eq, rho_eq = 100, 0.8
    corr_eq = np.full((n_eq, n_eq), rho_eq, dtype=float)
    np.fill_diagonal(corr_eq, 1.0)
    expected_neff = (100.0**2) / ((1.0 + 99.0 * 0.8) ** 2 + 99.0 * (0.2**2))
    assert n_eff_participation(corr_eq) == pytest.approx(expected_neff, abs=1e-9)

    # 4. Sweep verdict excludes failed trials from N and reports n_eff_participation
    base_series = np.random.default_rng(7).normal(0.0005, 0.01, 200)

    def bt_with_failure(data=None, k: int = 0):
        if k == 0:
            raise ValueError("failed trial 0")
        return (np.asarray(data, dtype=float) + k * 0.0001).tolist()

    sr = sweep(bt_with_failure, {"k": [0, 1, 2, 3]}, data=base_series)
    assert sr.n_trials == 4
    v = sr.verdict()
    assert v["n_trials"] == 3
    assert v["run_n_trials"] == 3
    assert "n_eff_participation" in v and v["n_eff_participation"] >= 1.0
    assert "deflated_sharpe_at_n_eff" in v and isinstance(v["deflated_sharpe_at_n_eff"], float)


def _sim_garch_path(seed: int, n: int = 3000, omega: float = 2e-6, alpha: float = 0.08, beta: float = 0.90):
    rng = np.random.default_rng(seed)
    r = np.empty(n, dtype=float)
    s2 = omega / (1.0 - alpha - beta)
    for t in range(n):
        s2 = omega + alpha * (r[t - 1] ** 2 if t else s2) + beta * s2
        r[t] = math.sqrt(s2) * rng.normal()
    return r


def test_garch_calibrate_six_seeded_paths_and_hardcoded_arch_parity() -> None:
    """Milestone 3 (H2): 6 seeded GARCH(1,1) paths (n=3000, alpha=0.08, beta=0.90),
    hardcoded arch reference comparison on seed=1, and maxiter=0 non-convergence.
    """
    from alphaengine.core import dgp_stress, garch_calibrate, vol_target

    for seed in range(1, 7):
        r = _sim_garch_path(seed)
        cal = garch_calibrate(r)
        assert cal["converged"] is True
        assert 0.05 <= cal["alpha"] <= 0.11
        assert 0.86 <= cal["beta"] <= 0.94
        assert not (abs(cal["alpha"] - 0.05) < 1e-6 and abs(cal["beta"] - 0.90) < 1e-6)

    # Hardcoded arch reference on seed=1: arch_model(r*100, vol="GARCH", p=1, q=1, mean="Zero")
    # gives alpha[1] = 0.086216, beta[1] = 0.893527
    r1 = _sim_garch_path(1)
    cal1 = garch_calibrate(r1)
    assert cal1["alpha"] == pytest.approx(0.086216, abs=0.01)
    assert cal1["beta"] == pytest.approx(0.893527, abs=0.01)

    # Propagates converged=True to vol_target(method="garch") and dgp_stress(dgp="garch")
    vt_ok = vol_target(r1, method="garch")
    assert vt_ok["converged"] is True
    assert vt_ok["method"] == "garch"
    st_ok = dgp_stress(r1, dgp="garch", n_paths=20, seed=7)
    assert st_ok["converged"] is True

    # Forced optimizer failure (maxiter=0) reports converged=False and never returns x0
    cal_fail = garch_calibrate(r1, maxiter=0)
    assert cal_fail["converged"] is False
    assert cal_fail["alpha"] is None and cal_fail["beta"] is None
    vt_fail = vol_target(r1, method="garch", garch_var=cal_fail)
    assert vt_fail["converged"] is False


def test_garch_calibrate_matches_live_arch_package() -> None:
    """Milestone 3 (H2): live comparison against arch.arch_model on seed=1."""
    arch_mod = pytest.importorskip("arch")
    from alphaengine.core import garch_calibrate

    r1 = _sim_garch_path(1)
    am = arch_mod.arch_model(r1 * 100.0, vol="GARCH", p=1, q=1, mean="Zero").fit(disp="off")
    cal = garch_calibrate(r1)
    assert cal["converged"] is True
    assert cal["alpha"] == pytest.approx(float(am.params["alpha[1]"]), abs=0.01)
    assert cal["beta"] == pytest.approx(float(am.params["beta[1]"]), abs=0.01)


def test_engle_granger_mackinnon_coint_pvalues_and_false_positive_rate() -> None:
    """Milestone 4 (H3): Engle-Granger uses statsmodels coint MacKinnon p-values.

    1. 400 seeded pairs of independent random walks (n=500): share with p<0.05
       is in [0.02, 0.08] (univariate ADF on estimated residual gave ~0.22).
    2. Fixed pair p-value matches coint(log_a, log_b, trend='c')[1] to 1e-10.
    3. Constructed cointegrated pair (b = RW, a = 1.5*b + AR(0.5) noise) gives p < 0.01.
    """
    pytest.importorskip("statsmodels")
    from statsmodels.tsa.stattools import coint

    from alphaengine.core.pairs import compute_spread_signal, engle_granger_test

    rng = np.random.default_rng(2026)
    n_pairs = 400
    n_obs = 500
    false_pos = 0
    for _ in range(n_pairs):
        rw_a = np.cumsum(rng.normal(0.0, 0.01, n_obs))
        rw_b = np.cumsum(rng.normal(0.0, 0.01, n_obs))
        eg = engle_granger_test(rw_a, rw_b)
        assert eg["pvalue_method"] == "mackinnon_coint"
        if eg["p_value"] is not None and eg["p_value"] < 0.05:
            false_pos += 1
    fp_share = false_pos / n_pairs
    assert 0.02 <= fp_share <= 0.08

    # Fixed pair matches coint(log_a, log_b, trend="c")[1] to 1e-10
    rng_fixed = np.random.default_rng(99)
    fixed_a = np.cumsum(rng_fixed.normal(0.0, 0.01, n_obs))
    fixed_b = np.cumsum(rng_fixed.normal(0.0, 0.01, n_obs))
    expected_p = float(coint(fixed_a, fixed_b, trend="c")[1])
    got_eg = engle_granger_test(fixed_a, fixed_b)
    assert got_eg["p_value"] == pytest.approx(expected_p, abs=1e-10)

    # Constructed cointegrated pair: b = random walk, a = 1.5*b + AR(0.5) noise
    rng_coint = np.random.default_rng(7)
    b_log = 4.0 + np.cumsum(rng_coint.normal(0.0, 0.01, n_obs))
    ar_noise = np.empty(n_obs, dtype=float)
    ar_noise[0] = 0.0
    for t in range(1, n_obs):
        ar_noise[t] = 0.5 * ar_noise[t - 1] + rng_coint.normal(0.0, 0.005)
    a_log = 1.5 * b_log + ar_noise
    eg_coint = engle_granger_test(a_log, b_log)
    assert eg_coint["p_value"] is not None and eg_coint["p_value"] < 0.01

    sig = compute_spread_signal(
        np.exp(a_log).tolist(), np.exp(b_log).tolist(), symbol_a="AAA", symbol_b="BBB"
    )
    assert sig["cointegration"]["pvalue_method"] == "mackinnon_coint"
    assert sig["cointegration"]["p_value"] < 0.01
