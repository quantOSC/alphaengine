"""Tests for AlphaEngine 2.0 conventions, version contract, methods registry,
and core math (`QOS-151`, `QOS-156`, `QOS-160`, `QOS-91`).
"""

from __future__ import annotations

import math
import pathlib
import subprocess
import sys

import numpy as np
import pytest

import alphaengine
from alphaengine import (
    MANIFEST_VERSION,
    METHODS_CATALOGUE,
    SYNC_CONTRACT_VERSION,
    VOLATILITY_FLOOR,
    Verdict,
    build_annualization_spec,
    build_costs_model_spec,
    build_risk_free_spec,
    current_engine_version,
    deflated_sharpe,
    engine_info,
    get_method,
    list_methods,
    normalize_verdict,
    sweep,
)
from alphaengine.core.backtest import run_backtest
from alphaengine.core.performance import performance_report
from alphaengine.core.validation import n_eff_participation
from alphaengine.core.walkforward import walk_forward


def test_version_single_source_and_offline_import() -> None:
    assert alphaengine.__version__ == "2.0.4"
    assert current_engine_version() == "2.0.4"
    assert SYNC_CONTRACT_VERSION == 2
    assert MANIFEST_VERSION == 1

    info = engine_info()
    assert info["engine_version"] == "2.0.4"
    assert info["sync_contract"] == 2
    assert info["manifest_version"] == 1
    assert isinstance(info["python_version"], str)

    # Fresh interpreter check: `import alphaengine` and `import alphaengine.validation`
    # must never import `httpx` or open any network socket (`QOS-151`).
    code = (
        "import socket, sys\n"
        "def _forbid(*a, **kw):\n"
        "    raise RuntimeError('socket opened on import')\n"
        "socket.socket = _forbid\n"
        "import alphaengine\n"
        "import alphaengine.validation\n"
        "assert 'httpx' not in sys.modules\n"
        "assert hasattr(alphaengine.validation, 'deflated_sharpe')\n"
        "assert hasattr(alphaengine.validation, 'walk_forward')\n"
        "assert hasattr(alphaengine.validation, 'Verdict')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    assert (repo_root / "ARCHITECTURE.md").is_file()


def test_verdict_vocabulary_and_deprecated_aliases() -> None:
    assert Verdict.ROBUST.value == "robust"
    assert Verdict.MARGINAL.value == "marginal"
    assert Verdict.LIKELY_NOISE.value == "likely_noise"
    assert Verdict.UNVERIFIED.value == "unverified"

    assert normalize_verdict("robust") == "robust"
    assert normalize_verdict("edge") == "robust"
    assert normalize_verdict("marginal") == "marginal"
    assert normalize_verdict("inconclusive") == "marginal"
    assert normalize_verdict("acceptable") == "marginal"
    assert normalize_verdict("likely_noise") == "likely_noise"
    assert normalize_verdict("overfit") == "likely_noise"
    assert normalize_verdict("unverified") == "unverified"
    assert normalize_verdict("not_recorded") == "unverified"
    assert normalize_verdict(None) is None

    # Non-statistical workflow status words pass through normalized to lowercase
    assert normalize_verdict("clean") == "clean"

    with pytest.warns(DeprecationWarning, match="deprecated"):
        assert normalize_verdict("overfit", warn=True) == "likely_noise"


def test_annualization_risk_free_and_costs_conventions() -> None:
    daily = build_annualization_spec(252)
    weekly = build_annualization_spec(52)
    monthly = build_annualization_spec(12)
    assert daily["periods_per_year"] == 252
    assert weekly["periods_per_year"] == 52
    assert monthly["periods_per_year"] == 12
    assert daily["sharpe_scaling"] == "sqrt(periods_per_year)"

    rf_const = build_risk_free_spec(0.05, periods_per_year=252)
    assert rf_const["annual_rate"] == 0.05
    assert math.isclose(rf_const["per_period_rate"], 0.05 / 252.0, abs_tol=1e-9)

    costs = build_costs_model_spec(
        mode="engine_simulated_weights",
        slippage_bps=5.0,
        commission_bps=1.0,
        liquidate_final=True,
    )
    assert costs["slippage_bps"] == 5.0
    assert costs["commission_bps"] == 1.0
    assert costs["liquidate_final"] is True


def test_flat_and_near_flat_returns_volatility_floor() -> None:
    assert VOLATILITY_FLOOR == 1e-6
    zeros = [0.0] * 60
    tiny = [1e-12 * ((i % 2) * 2 - 1) for i in range(60)]

    for series in (zeros, tiny):
        perf = performance_report(series)
        dsr = deflated_sharpe(series, n_trials=5)
        assert perf["sharpe_annualized"] == 0.0
        assert math.isfinite(perf["sharpe_annualized"])
        assert dsr["sharpe_annualized"] == 0.0
        assert math.isfinite(dsr["dsr_probability"])
        assert dsr["verdict"] == "likely_noise"


def test_risk_free_consistency_across_performance_dsr_and_sweep() -> None:
    rng = np.random.default_rng(42)
    r1 = rng.normal(0.0008, 0.012, size=120)
    r2 = rng.normal(0.0004, 0.011, size=120)

    rf_annual = 0.045
    perf = performance_report(r1, risk_free_rate=rf_annual, periods_per_year=252)
    dsr = deflated_sharpe(r1, n_trials=2, risk_free_rate=rf_annual, periods_per_year=252)
    assert math.isclose(perf["sharpe_annualized"], dsr["sharpe_annualized"], abs_tol=1e-4)

    res = sweep(
        lambda data, k=0: r1 if k == 0 else r2,
        {"k": [0, 1]},
        data=[1.0, 2.0],
        risk_free_rate=rf_annual,
        periods_per_year=252,
    )
    verd = res.verdict()
    assert math.isclose(verd["performance"]["sharpe_annualized"], perf["sharpe_annualized"], abs_tol=1e-4)


def test_dsr_probability_alias_single_trial_ceiling_and_n_eff_diagnostic() -> None:
    rng = np.random.default_rng(7)
    strong_rets = rng.normal(0.003, 0.008, size=200)

    one = deflated_sharpe(strong_rets, n_trials=1)
    assert one["deflation_applied"] is False
    assert one["sr0_expected_max"] == 0.0
    assert one["dsr_probability"] == one["deflated_sharpe"]
    # Single-trial runs cannot claim `robust` because selection bias is unmeasured (`n_trials == 1`)
    assert one["verdict"] == "marginal"

    multi = deflated_sharpe(strong_rets, n_trials=8)
    assert multi["deflation_applied"] is True
    assert multi["dsr_probability"] == multi["deflated_sharpe"]
    assert 0.0 <= multi["dsr_probability"] <= 1.0

    # Two identical return columns -> n_eff ~ 1.0 while raw n_trials remains 2
    col = rng.normal(0.001, 0.01, size=100)
    mat = np.column_stack([col, col + 1e-9])
    neff = n_eff_participation(mat)
    assert 1.0 <= neff <= 1.05

    sw = sweep(
        lambda data, a=1: col if a == 1 else col + 1e-9,
        {"a": [1, 2]},
        data=[1.0, 2.0],
    )
    sv = sw.verdict()
    assert sv["n_trials"] == 2
    assert 1.0 <= sv["n_eff_participation"] <= 1.05
    assert sv["deflated_sharpe_at_n_eff"] is not None


def test_walk_forward_purge_bars_strict_gap() -> None:
    prices = np.linspace(100.0, 140.0, 120)
    out = walk_forward(
        lambda data, lb=5: np.diff(data) / data[:-1],
        {"lb": [5, 10]},
        data=prices,
        n_windows=3,
        oos_fraction=0.25,
        purge_bars=5,
    )
    assert out["purge_bars"] == 5
    assert out["n_windows_ok"] >= 2
    for win in out["windows"]:
        if win.get("failed") is None:
            assert win["oos_start"] - win["is_end"] == 5

    with pytest.raises(ValueError, match="purge_bars"):
        walk_forward(lambda data, lb=5: np.diff(data), {"lb": [5]}, data=prices, purge_bars=-1)


def test_backtest_liquidate_final_borrow_bps_and_dated_adv_series() -> None:
    dates = [f"2025-01-{i + 1:02d}" for i in range(5)]
    prices = {
        "A": {d: 100.0 + 2.0 * i for i, d in enumerate(dates)},
        "B": {d: 50.0 - 1.0 * i for i, d in enumerate(dates)},
    }
    signals = {
        "A": [{"date": dates[0], "target_weight": 0.5}],
        "B": [{"date": dates[0], "target_weight": -0.5}],
    }

    bt_liq = run_backtest(
        signals,
        prices,
        slippage_bps=10.0,
        commission_bps=5.0,
        liquidate_final=True,
    )
    bt_no_liq = run_backtest(
        signals,
        prices,
        slippage_bps=10.0,
        commission_bps=5.0,
        liquidate_final=False,
    )
    assert bt_liq["cost_breakdown"]["total"] > bt_no_liq["cost_breakdown"]["total"]
    assert bt_liq["final_equity"] < bt_no_liq["final_equity"]

    bt_borrow = run_backtest(
        signals,
        prices,
        slippage_bps=5.0,
        commission_bps=1.0,
        borrow_bps=250.0,
    )
    assert bt_borrow["cost_breakdown"]["borrow"] > 0.0
    assert (
        bt_borrow["final_equity"]
        < run_backtest(signals, prices, slippage_bps=5.0, commission_bps=1.0)["final_equity"]
    )

    # Causal adv_series resolves ADV strictly as-of bar date (never future ADV spike)
    signals_rebal = {
        "A": [{"date": dates[0], "target_weight": 0.5}, {"date": dates[2], "target_weight": -0.5}],
        "B": [{"date": dates[0], "target_weight": -0.5}, {"date": dates[2], "target_weight": 0.5}],
    }
    adv_s_spike_later = {
        "A": {dates[0]: 1_000.0, dates[1]: 1_000.0, dates[2]: 1_000.0, dates[4]: 10_000_000.0},
        "B": {d: 1_000.0 for d in dates},
    }
    adv_s_drop_later = {
        "A": {dates[0]: 1_000.0, dates[1]: 1_000.0, dates[2]: 1_000.0, dates[4]: 200.0},
        "B": {d: 1_000.0 for d in dates},
    }
    bt_adv_early = run_backtest(
        signals_rebal,
        prices,
        adv_series=adv_s_spike_later,
        max_participation=0.05,
        initial_capital=100_000.0,
        liquidate_final=False,
    )
    bt_adv_changed = run_backtest(
        signals_rebal,
        prices,
        adv_series=adv_s_drop_later,
        max_participation=0.05,
        initial_capital=100_000.0,
        liquidate_final=False,
    )
    # Bars 0..3 (`[:4]`) resolve ADV strictly as-of each bar date, so mutating `dates[4]` has zero effect!
    assert bt_adv_early["equity_curve"][:4] == bt_adv_changed["equity_curve"][:4]
    assert bt_adv_early["costs_per_bar"][:4] == bt_adv_changed["costs_per_bar"][:4]


def test_methods_registry_catalogue() -> None:
    methods = list_methods()
    assert len(methods) >= 8
    ids = {m["method_id"] for m in methods}
    assert {
        "deflated_sharpe",
        "probabilistic_sharpe_ratio",
        "min_track_record_length",
        "pbo_cscv",
        "combinatorial_oos_stability",
        "performance_report",
        "run_backtest",
        "coded_signal_adapter",
    }.issubset(ids)
    assert set(METHODS_CATALOGUE) == ids
    dsr_spec = get_method("deflated_sharpe")
    assert dsr_spec["engine_version"] == "2.0.4"
