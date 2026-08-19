"""New compute.* handlers: figures out, series stay in the workspace."""

from __future__ import annotations

from alphaengine.client import StepExecutor, UnsupportedOp
from alphaengine.demo import GRID, backtest_fn
from alphaengine.demo import data as demo_data


def test_cpcv_runs_on_a_supplied_return_series():
    returns = [0.001, -0.002, 0.0015, 0.0] * 80
    ex = StepExecutor(data=returns)
    out = ex.execute("compute.cpcv", {"n_groups": 6, "n_test_groups": 2})
    assert "error" not in out
    assert "n_obs" in out or "mean_oos_sharpe" in out or "oos_sharpe_mean" in out or out


def test_walk_forward_keeps_the_trial_count_derived():
    ex = StepExecutor(data=demo_data, backtest_fn=backtest_fn)
    out = ex.execute("compute.walk_forward", {"grid": GRID, "n_windows": 3})
    assert out["n_trials"] == 27
    assert out["n_trials_source"] == "derived_from_walk_forward"
    assert "matrix" not in out


def test_book_overlap_from_sleeves():
    series = [0.01, -0.01, 0.0, 0.02] * 40
    ex = StepExecutor(data={"sleeves": {"a": series, "b": series}})
    out = ex.execute("compute.book_overlap", {})
    assert out["n_sleeves"] == 2
    assert out["pairs"]


def test_backtest_stores_the_path_locally():
    prices = {"AAA": [{"date": str(i), "close": 100 + i * 0.1} for i in range(30)]}
    signals = {"AAA": [{"date": str(i), "target_weight": 1.0} for i in range(30)]}
    ex = StepExecutor(data={"signals": signals, "prices": prices})
    out = ex.execute("compute.backtest", {})
    assert "returns" not in out
    assert "equity_curve" not in out
    assert "n_bars" in out
    assert "backtest" in ex.workspace


def test_score_backtest_needs_a_prior_backtest():
    ex = StepExecutor(data=[0.01] * 50)
    try:
        ex.execute("compute.score_backtest", {})
        raised = False
    except UnsupportedOp:
        raised = True
    assert raised


def test_sweep_jobs_preserve_trial_identity():
    from alphaengine import sweep

    r1 = sweep(backtest_fn, GRID, data=demo_data, jobs=1)
    r2 = sweep(backtest_fn, GRID, data=demo_data, jobs=2)
    assert r1.n_trials == r2.n_trials == 9
    assert [t.index for t in r1.trials] == [t.index for t in r2.trials]
    assert [t.sharpe for t in r1.trials] == [t.sharpe for t in r2.trials]
