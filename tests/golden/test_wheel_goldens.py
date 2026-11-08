"""14-case Wheel Golden Suite for AlphaEngine 2.0 (`QOS-157`).

Every case here exercises the public `alphaengine` contract using paths
resolved strictly relative to `Path(__file__).resolve().parent` so the suite
runs identically from inside a temporary directory against an installed `.whl`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alphaengine import coded, sweep
from alphaengine.core import (
    VOL_FLOOR,
    deflated_sharpe,
    pbo_cscv,
    performance_report,
    run_backtest,
)
from alphaengine.core.market import align_market
from alphaengine.project import init_project
from alphaengine.validation import dsr_probability, walk_forward

pytestmark = pytest.mark.golden

GOLDEN_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = GOLDEN_DIR / "fixtures"
EXPECTED_PATH = GOLDEN_DIR / "expected" / "wheel_goldens_v2.json"


def _load_ohlc_fixture() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    with (FIXTURES_DIR / "ohlc_panel.csv").open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            tk = row["ticker"].strip().upper()
            open_raw = row["open"].strip()
            out.setdefault(tk, []).append(
                {
                    "date": row["date"].strip(),
                    "open": float(open_raw) if open_raw else None,
                    "close": float(row["close"].strip()),
                }
            )
    return out


def _load_adv_fixture() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {"AAPL": {}, "MSFT": {}}
    with (FIXTURES_DIR / "adv_series.csv").open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            d = row["date"].strip()
            out["AAPL"][d] = float(row["AAPL"])
            out["MSFT"][d] = float(row["MSFT"])
    return out


def _load_misaligned_fixture() -> tuple[list[str], dict[str, dict[str, float]]]:
    dates: list[str] = []
    dated_prices: dict[str, dict[str, float]] = {"AAPL": {}, "MSFT": {}, "NVDA": {}}
    with (FIXTURES_DIR / "misaligned_prices.csv").open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            d = row["date"].strip()
            dates.append(d)
            for tk in ("AAPL", "MSFT", "NVDA"):
                raw = row[tk].strip()
                if raw:
                    dated_prices[tk][d] = float(raw)
    return dates, dated_prices


def compute_all_golden_cases(tmp_root: Path) -> dict[str, dict[str, Any]]:
    """Compute deterministic outputs for all 14 `QOS-157` golden cases."""
    ohlc = _load_ohlc_fixture()
    adv_dated = _load_adv_fixture()
    _, misaligned_prices = _load_misaligned_fixture()

    signals_pair = {
        "AAPL": [
            {"date": "2025-01-02", "target_weight": 0.5},
            {"date": "2025-01-06", "target_weight": 0.0},
        ],
        "MSFT": [
            {"date": "2025-01-01", "target_weight": 0.4},
            {"date": "2025-01-05", "target_weight": 0.0},
        ],
    }

    # 1. next_open fills & missing open fallback
    # (2025-01-03 AAPL open is missing -> falls back to close 102.0)
    bt_next_open = run_backtest(
        signals=signals_pair,
        prices=ohlc,
        fill_timing="next_open",
        slippage_bps=5.0,
        commission_bps=2.0,
        initial_capital=100_000.0,
    )
    aapl_entry = [t for t in bt_next_open["trades"] if t["ticker"] == "AAPL"][0]
    msft_entry = [t for t in bt_next_open["trades"] if t["ticker"] == "MSFT"][0]
    case_01 = {
        "fill_timing": bt_next_open["costs_model"]["fill_timing"],
        "aapl_fill_date": aapl_entry["entry_date"],
        "aapl_ref_price": round(float(aapl_entry["entry_price"]), 6),
        "msft_fill_date": msft_entry["entry_date"],
        "msft_ref_price": round(float(msft_entry["entry_price"]), 6),
        "n_trades": len(bt_next_open["trades"]),
        "final_equity": round(float(bt_next_open["final_equity"]), 4),
        "total_return_pct": round(float(bt_next_open["total_return_pct"]), 4),
    }

    # 2. same_close (`fill_timing='close'`) rules
    bt_same_close = run_backtest(
        signals=signals_pair,
        prices=ohlc,
        fill_timing="close",
        slippage_bps=5.0,
        commission_bps=2.0,
        initial_capital=100_000.0,
    )
    aapl_close_entry = [t for t in bt_same_close["trades"] if t["ticker"] == "AAPL"][0]
    msft_close_entry = [t for t in bt_same_close["trades"] if t["ticker"] == "MSFT"][0]
    case_02 = {
        "fill_timing": bt_same_close["costs_model"]["fill_timing"],
        "aapl_fill_date": aapl_close_entry["entry_date"],
        "aapl_ref_price": round(float(aapl_close_entry["entry_price"]), 6),
        "msft_fill_date": msft_close_entry["entry_date"],
        "msft_ref_price": round(float(msft_close_entry["entry_price"]), 6),
        "final_equity": round(float(bt_same_close["final_equity"]), 4),
        "total_return_pct": round(float(bt_same_close["total_return_pct"]), 4),
        "differs_from_next_open": round(float(bt_same_close["final_equity"]), 4)
        != round(float(bt_next_open["final_equity"]), 4),
    }

    # 3. Final liquidation cost in equity (`liquidate_final=True` vs `False`)
    hold_to_end = {"AAPL": [{"date": "2025-01-01", "target_weight": 0.8}]}
    bt_liq_true = run_backtest(
        signals=hold_to_end,
        prices=ohlc,
        fill_timing="next_open",
        slippage_bps=20.0,
        commission_bps=10.0,
        liquidate_final=True,
        initial_capital=100_000.0,
    )
    bt_liq_false = run_backtest(
        signals=hold_to_end,
        prices=ohlc,
        fill_timing="next_open",
        slippage_bps=20.0,
        commission_bps=10.0,
        liquidate_final=False,
        initial_capital=100_000.0,
    )
    case_03 = {
        "final_equity_liquidated": round(float(bt_liq_true["final_equity"]), 4),
        "final_equity_unliquidated": round(float(bt_liq_false["final_equity"]), 4),
        "liquidation_drag_dollars": round(
            float(bt_liq_false["final_equity"]) - float(bt_liq_true["final_equity"]), 4
        ),
        "liquidate_final_lower_equity": (
            float(bt_liq_true["final_equity"]) < float(bt_liq_false["final_equity"])
        ),
    }

    # 4. `borrow_bps` short borrow accrual
    short_msft = {"MSFT": [{"date": "2025-01-01", "target_weight": -0.75}]}
    bt_no_borrow = run_backtest(
        signals=short_msft,
        prices=ohlc,
        fill_timing="next_open",
        borrow_bps=0.0,
        slippage_bps=0.0,
        commission_bps=0.0,
    )
    bt_with_borrow = run_backtest(
        signals=short_msft,
        prices=ohlc,
        fill_timing="next_open",
        borrow_bps=300.0,
        slippage_bps=0.0,
        commission_bps=0.0,
    )
    case_04 = {
        "final_equity_no_borrow": round(float(bt_no_borrow["final_equity"]), 4),
        "final_equity_300bps_borrow": round(float(bt_with_borrow["final_equity"]), 4),
        "borrow_drag_dollars": round(
            float(bt_no_borrow["final_equity"]) - float(bt_with_borrow["final_equity"]), 4
        ),
        "borrow_reduces_short_equity": (
            float(bt_with_borrow["final_equity"]) < float(bt_no_borrow["final_equity"])
        ),
    }

    # 5. Dated as-of `adv_series` participation caps
    large_aapl = {"AAPL": [{"date": "2025-01-02", "target_weight": 0.6}]}
    bt_uncapped = run_backtest(
        signals=large_aapl,
        prices=ohlc,
        fill_timing="next_open",
        initial_capital=2_000_000.0,
        slippage_bps=5.0,
    )
    bt_adv_capped = run_backtest(
        signals=large_aapl,
        prices=ohlc,
        fill_timing="next_open",
        initial_capital=2_000_000.0,
        slippage_bps=5.0,
        adv_series=adv_dated,
    )
    case_05 = {
        "uncapped_trades": len(bt_uncapped["trades"]),
        "capped_trades": len(bt_adv_capped["trades"]),
        "uncapped_final_equity": round(float(bt_uncapped["final_equity"]), 2),
        "capped_final_equity": round(float(bt_adv_capped["final_equity"]), 2),
        "adv_cap_splits_order": len(bt_adv_capped["trades"]) > len(bt_uncapped["trades"]),
    }

    # 6. Flat & near-flat returns (`VOLATILITY_FLOOR = 1e-6`)
    flat_rets = [0.0] * 120
    near_flat_rets = [1e-9 if i % 2 == 0 else -1e-9 for i in range(120)]
    perf_flat = performance_report(flat_rets, risk_free_rate=0.0)
    perf_near = performance_report(near_flat_rets, risk_free_rate=0.0)
    dsr_flat = deflated_sharpe(flat_rets, n_trials=5, risk_free_rate=0.0)
    dsr_near = deflated_sharpe(near_flat_rets, n_trials=5, risk_free_rate=0.0)
    case_06 = {
        "min_vol_floor": VOL_FLOOR,
        "flat_sharpe_annualized": perf_flat["sharpe_annualized"],
        "flat_vol_pct": perf_flat["volatility_annualized_pct"],
        "near_flat_sharpe_annualized": perf_near["sharpe_annualized"],
        "near_flat_vol_pct": perf_near["volatility_annualized_pct"],
        "flat_dsr": dsr_flat["deflated_sharpe"],
        "near_flat_dsr": dsr_near["deflated_sharpe"],
    }

    # 7. Misaligned datasets (`align_market`)
    aligned = align_market(misaligned_prices)
    case_07 = {
        "n_common": aligned["n_common"],
        "start": aligned["start"],
        "end": aligned["end"],
        "n_names": aligned["n_names"],
        "gaps": aligned["gaps"],
    }

    # 8. Sweep with failed grid combinations (`n_trials` includes failed combinations)
    rng8 = np.random.default_rng(108)
    base8 = rng8.normal(0.0005, 0.01, 160).tolist()

    def _bt_fail_some(data: Any = None, k: int = 1) -> list[float]:
        if k in (2, 4):
            raise ValueError(f"synthetic failure for k={k}")
        arr = np.asarray(data, dtype=float)
        return (arr + 0.00015 * k).tolist()

    sr8 = sweep(_bt_fail_some, {"k": [1, 2, 3, 4, 5]}, data=base8)
    v8 = sr8.verdict()
    failed_indices = [t.index for t in sr8.trials if t.failed is not None]
    case_08 = {
        "grid_size": 5,
        "sr_n_trials": sr8.n_trials,
        "verdict_n_trials": v8["n_trials"],
        "failed_indices": failed_indices,
        "best_params": sr8.best.params,
        "best_sharpe": round(float(sr8.best.sharpe), 4),
        "deflated_sharpe": v8["deflated_sharpe"],
    }

    # 9. Identical config repeated 3 times inside a project (`cumulative_trials == 3 * grid_size`)
    proj_dir = tmp_root / "case09_proj"
    proj_dir.mkdir(parents=True, exist_ok=True)
    init_project(proj_dir, name="golden-case09")
    import os

    prev_cwd = Path.cwd()
    os.chdir(proj_dir)
    try:
        series9 = np.random.default_rng(109).normal(0.0007, 0.01, 180).tolist()
        cum_counts: list[int] = []
        dsrs_over_repeats: list[float] = []
        for _ in range(3):
            r9 = sweep(
                lambda data=None, mult=1.0: (np.asarray(data, dtype=float) * float(mult)).tolist(),
                {"mult": [0.8, 1.0, 1.2, 1.4]},
                data=series9,
            )
            assert r9.cumulative_trials is not None
            cum_counts.append(int(r9.cumulative_trials))
            dsrs_over_repeats.append(float(r9.verdict()["deflated_sharpe"]))
    finally:
        os.chdir(prev_cwd)
    case_09 = {
        "cumulative_counts": cum_counts,
        "dsrs_over_repeats": dsrs_over_repeats,
        "monotone_non_increasing_dsr": dsrs_over_repeats == sorted(dsrs_over_repeats, reverse=True),
    }

    # 10. Risk-free rate consistency across `sweep`, `performance_report`, and `deflated_sharpe`
    series10 = np.random.default_rng(110).normal(0.0008, 0.011, 200).tolist()
    rf_custom = 0.05
    p10 = performance_report(series10, risk_free_rate=rf_custom, periods_per_year=252)
    d10 = deflated_sharpe(series10, n_trials=1, risk_free_rate=rf_custom, periods_per_year=252)
    s10 = sweep(
        lambda data=None, scale=1.0: (np.asarray(data, dtype=float) * float(scale)).tolist(),
        {"scale": [1.0]},
        data=series10,
        risk_free_rate=rf_custom,
        periods_per_year=252,
    )
    sv10 = s10.verdict()
    case_10 = {
        "perf_sharpe_ratio": p10["sharpe_ratio"],
        "dsr_sharpe_per_period": d10["sharpe_per_period"],
        "sweep_best_sharpe": round(float(s10.best.sharpe), 4),
        "perf_rf_source": p10["risk_free"]["source"],
        "dsr_rf_source": d10["risk_free"]["source"],
        "sweep_rf_source": sv10["risk_free"]["source"],
        "sharpe_matches_across_all_three": (
            p10["sharpe_ratio"] == d10["sharpe_per_period"] == round(float(s10.best.sharpe), 4)
        ),
    }

    # 11. `purge_bars` gap in `walk_forward`
    prices11 = (100.0 * np.cumprod(1.0 + np.random.default_rng(111).normal(0.0005, 0.01, 120))).tolist()

    def _wf_bt(data: Any = None, fast: int = 3) -> list[float]:
        arr = np.asarray(data, dtype=float)
        if arr.size < fast + 2:
            return [0.0, 0.0]
        rets = np.diff(arr) / arr[:-1]
        sig = np.sign(arr[fast - 1 : -1] - arr[:-fast])
        aligned_r = rets[fast - 1 :]
        return (sig * aligned_r).tolist()

    wf_no_purge = walk_forward(
        _wf_bt, {"fast": [3, 5]}, data=prices11, n_windows=4, oos_fraction=0.25, purge_bars=0
    )
    wf_purged = walk_forward(
        _wf_bt, {"fast": [3, 5]}, data=prices11, n_windows=4, oos_fraction=0.25, purge_bars=5
    )
    gaps_purged = [int(w["oos_start"]) - int(w["is_end"]) for w in wf_purged["windows"]]
    case_11 = {
        "no_purge_n_windows": wf_no_purge["n_windows"],
        "purged_n_windows": wf_purged["n_windows"],
        "purged_n_trials": wf_purged["n_trials"],
        "purged_gaps": gaps_purged,
        "no_purge_mean_oos_sharpe": wf_no_purge["mean_oos_sharpe"],
        "purged_mean_oos_sharpe": wf_purged["mean_oos_sharpe"],
    }

    # 12. One-trial `marginal` cap with `deflation_applied=False`
    strong12 = np.random.default_rng(112).normal(0.004, 0.008, 252).tolist()
    d12 = deflated_sharpe(strong12, n_trials=1)
    case_12 = {
        "n_trials": d12["n_trials"],
        "deflation_applied": d12["deflation_applied"],
        "sr0_expected_max": d12["sr0_expected_max"],
        "psr_vs_zero": d12["psr_vs_zero"],
        "deflated_sharpe": d12["deflated_sharpe"],
        "verdict": d12["verdict"],
    }

    # 13. DSR (`dsr_probability`), PSR, and PBO on pinned synthetic matrices
    mat13 = np.random.default_rng(2026).normal(0.0003, 0.011, size=(180, 6))
    best_col = mat13[:, 0].tolist()
    col_sharpes = [float(mat13[:, j].mean() / mat13[:, j].std(ddof=1)) for j in range(6)]
    trials_std = float(np.std(col_sharpes, ddof=1))
    dsr13 = deflated_sharpe(best_col, n_trials=6, trials_sharpe_std=trials_std)
    alias13 = dsr_probability(best_col, n_trials=6, trials_sharpe_std=trials_std)
    pbo13 = pbo_cscv(mat13.tolist(), n_splits=6)
    case_13 = {
        "dsr_matches_alias": dsr13["deflated_sharpe"] == alias13["deflated_sharpe"],
        "deflated_sharpe": dsr13["deflated_sharpe"],
        "psr_vs_zero": dsr13["psr_vs_zero"],
        "sr0_expected_max": dsr13["sr0_expected_max"],
        "dsr_verdict": dsr13["verdict"],
        "pbo": pbo13["pbo"],
        "pbo_logit_mean": pbo13["logit_mean"],
        "pbo_verdict": pbo13["verdict"],
    }

    # 14. `coded.run_signal()` weights -> expected equity curve & trial count
    seen_max_bars: list[int] = []

    def _causal_mom_signal(data_asof: dict[str, Any], params: dict[str, Any]) -> dict[str, float]:
        lookback = int(params.get("lookback", 2))
        closes_map = data_asof["prices"]
        max_len = max(len(v) for v in closes_map.values())
        seen_max_bars.append(max_len)
        out_w: dict[str, float] = {}
        for tk, closes in sorted(closes_map.items()):
            if len(closes) <= lookback:
                out_w[tk] = 0.0
            else:
                ret = (closes[-1] / closes[-1 - lookback]) - 1.0
                out_w[tk] = 0.4 if ret > 0 else -0.2
        return out_w

    coded_prices = {
        tk: [{"date": row["date"], "open": row["open"], "close": row["close"]} for row in rows]
        for tk, rows in ohlc.items()
    }
    sr14 = coded.run_signal(
        _causal_mom_signal,
        data=coded_prices,
        grid={"lookback": [1, 2, 3]},
        slippage_bps=5.0,
        commission_bps=2.0,
        borrow_bps=0.0,
    )
    v14 = sr14["verdict"]
    figs14 = sr14["figures"]
    case_14 = {
        "n_trials": sr14["n_trials"],
        "engine_verified": sr14.get("engine_verified", True),
        "causal_bar_growth": seen_max_bars[:8] == list(range(1, 9)),
        "best_trial_index": figs14["best_trial_index"],
        "deflated_sharpe": figs14["deflated_sharpe"],
        "verdict": v14,
    }

    return {
        "01_next_open_and_missing_open_fallback": case_01,
        "02_same_close_fills": case_02,
        "03_final_liquidation_cost": case_03,
        "04_borrow_bps_accrual": case_04,
        "05_dated_adv_participation_caps": case_05,
        "06_flat_and_near_flat_vol_floor": case_06,
        "07_misaligned_datasets_align_market": case_07,
        "08_sweep_failed_grid_combinations": case_08,
        "09_repeated_identical_project_runs": case_09,
        "10_risk_free_rate_consistency": case_10,
        "11_walk_forward_purge_bars": case_11,
        "12_single_trial_marginal_cap": case_12,
        "13_dsr_psr_pbo_pinned_matrices": case_13,
        "14_coded_run_signal_equity_and_trials": case_14,
    }


@pytest.fixture(scope="module")
def computed_goldens(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    tmp_root = tmp_path_factory.mktemp("wheel_goldens")
    return compute_all_golden_cases(tmp_root)


@pytest.fixture(scope="module")
def expected_goldens() -> dict[str, dict[str, Any]]:
    assert EXPECTED_PATH.is_file(), f"Missing pinned golden file: {EXPECTED_PATH}"
    return json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))


CASE_IDS = [
    "01_next_open_and_missing_open_fallback",
    "02_same_close_fills",
    "03_final_liquidation_cost",
    "04_borrow_bps_accrual",
    "05_dated_adv_participation_caps",
    "06_flat_and_near_flat_vol_floor",
    "07_misaligned_datasets_align_market",
    "08_sweep_failed_grid_combinations",
    "09_repeated_identical_project_runs",
    "10_risk_free_rate_consistency",
    "11_walk_forward_purge_bars",
    "12_single_trial_marginal_cap",
    "13_dsr_psr_pbo_pinned_matrices",
    "14_coded_run_signal_equity_and_trials",
]


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_wheel_golden_case(
    case_id: str,
    computed_goldens: dict[str, dict[str, Any]],
    expected_goldens: dict[str, dict[str, Any]],
) -> None:
    assert case_id in expected_goldens, f"Missing expected golden entry for {case_id}"
    assert computed_goldens[case_id] == expected_goldens[case_id], (
        f"Golden drift on {case_id}:\n"
        f"  got:  {computed_goldens[case_id]}\n"
        f"  want: {expected_goldens[case_id]}"
    )
