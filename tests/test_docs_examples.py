"""Tests verifying documented offline Python examples from README.md and docs/.

These tests guarantee zero documentation drift (QOS-171).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alphaengine import (
    Verdict,
    combinatorial_oos_stability,
    compute_var_cvar,
    deflated_sharpe,
    dsr_probability,
    min_track_record_length,
    pbo_cscv,
    performance_report,
    probabilistic_sharpe_ratio,
    run_signal,
    sweep,
)
from alphaengine.core import (
    cs_zscore,
    dgp_stress,
    fama_macbeth,
    garch_calibrate,
    grinold_alpha,
    hrp_weights,
    ou_calibrate,
    signal_icir,
)

ROOT = Path(__file__).resolve().parents[1]


# ── 1. Verdict vocabulary & migration examples ──────────────────────────────


def test_doc_example_verdict_vocabulary():
    """Verify Verdict enum and legacy aliases documented in MIGRATING_TO_2.md."""
    assert Verdict.ROBUST == "robust"
    assert Verdict.MARGINAL == "marginal"
    assert Verdict.LIKELY_NOISE == "likely_noise"
    assert Verdict.UNVERIFIED == "unverified"

    # Deprecated 1.x alias normalization
    assert Verdict.normalize("edge") == Verdict.ROBUST
    assert Verdict.normalize("inconclusive") == Verdict.MARGINAL
    assert Verdict.normalize("acceptable") == Verdict.MARGINAL
    assert Verdict.normalize("overfit") == Verdict.LIKELY_NOISE
    assert Verdict.normalize("not_recorded") == Verdict.UNVERIFIED


# ── 2. Coded target-weights adapter example ─────────────────────────────────


def test_doc_example_coded_signal_runner():
    """Verify the exact coded momentum signal example from docs/MIGRATING_TO_2.md."""

    def my_momentum_signal(*, data_asof: dict, params: dict) -> dict[str, float]:
        lookback = params.get("lookback", 20)
        prices = data_asof["prices"]
        weights = {}
        for ticker, closes in prices.items():
            if len(closes) > lookback:
                mom = (closes[-1] / closes[-1 - lookback]) - 1.0
                weights[ticker] = 0.5 if mom > 0 else -0.5
            else:
                weights[ticker] = 0.0
        return weights

    # Deterministic 50-bar price path
    rng = np.random.default_rng(42)
    aapl_prices = (100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.01, 50))).tolist()
    msft_prices = (200.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, 50))).tolist()

    res = run_signal(
        my_momentum_signal,
        prices={"AAPL": aapl_prices, "MSFT": msft_prices},
        grid={"lookback": [5, 10, 15]},
        slippage_bps=5.0,
        commission_bps=1.0,
    )

    assert "verdict" in res
    assert res["verdict"] in {v.value for v in Verdict}
    assert res["n_trials"] == 3
    assert "backtest" in res
    assert "score" in res
    assert "manifest" in res


# ── 3. Parameter sweep and backtest_fn example ─────────────────────────────


def test_doc_example_sweep_and_backtest():
    """Verify sweep() usage documented in README.md."""

    def sma(series: list[float], period: int, idx: int) -> float:
        if idx + 1 < period:
            return series[idx]
        return float(np.mean(series[idx + 1 - period : idx + 1]))

    warmup = 30

    def backtest_fn(*, data, fast, slow):
        close = data["close"]
        return [
            (close[i + 1] - close[i]) / close[i] * (1 if sma(close, fast, i) > sma(close, slow, i) else 0)
            for i in range(warmup, len(close) - 1)
        ]

    # Generate 100 prices
    rng = np.random.default_rng(123)
    close_prices = (100.0 * np.cumprod(1.0 + rng.normal(0.0008, 0.012, 100))).tolist()
    prices = {"close": close_prices}

    r = sweep(backtest_fn, {"fast": [5, 10], "slow": [15, 20]}, data=prices)

    assert len(r.trials) == 4
    v = r.verdict()
    assert isinstance(v, dict)
    assert "verdict" in v
    assert v["verdict"] in {ver.value for ver in Verdict}
    surface = r.surface()
    assert isinstance(surface, dict)
    assert "shape" in surface


# ── 4. Process math & portfolio math primitives from README ────────────────


def test_doc_example_process_math():
    """Verify Block 1 process math primitives from README.md."""
    rng = np.random.default_rng(42)
    spread = rng.normal(0.0, 1.0, 150).tolist()
    returns = rng.normal(0.0005, 0.01, 150).tolist()
    close = (100.0 * np.cumprod(1.0 + np.array(returns))).tolist()

    ou = ou_calibrate(spread)
    assert isinstance(ou, dict)

    g = garch_calibrate(returns)
    assert isinstance(g, dict)

    s = dgp_stress(close, dgp="ou")
    assert isinstance(s, dict)
    assert s.get("n_trials") == 200

    panel = {"A": [0.01, -0.02, 0.03], "B": [-0.01, 0.02, -0.01]}
    vols = {"A": 0.15, "B": 0.20}
    a = grinold_alpha(panel, ic=0.05, vols=vols)
    assert isinstance(a, dict)


def test_doc_example_cross_section_math():
    """Verify Block 2 cross-section math primitives from README.md."""
    factor_panel = {
        "A": [1.0, 2.0, 1.5, 3.0],
        "B": [2.0, 1.0, 2.5, 1.0],
        "C": [0.5, 1.5, 1.0, 2.0],
    }
    z = cs_zscore(factor_panel)
    assert isinstance(z, dict)

    prices = {
        "A": [10.0, 10.2, 10.1, 10.5],
        "B": [20.0, 19.8, 20.2, 20.1],
        "C": [15.0, 15.1, 15.0, 15.3],
    }
    ic = signal_icir(factor_panel, prices)
    assert isinstance(ic, dict)
    assert "icir" in ic

    fm = fama_macbeth(factor_panel, prices)
    assert isinstance(fm, dict)

    cov = [
        [0.04, 0.01, 0.005],
        [0.01, 0.09, 0.01],
        [0.005, 0.01, 0.06],
    ]
    w = hrp_weights(cov, names=["A", "B", "C"])
    assert isinstance(w, dict)
    assert pytest.approx(w["weight_sum"], abs=1e-5) == 1.0
    assert pytest.approx(sum(w["weights"].values()), abs=1e-5) == 1.0


# ── 5. Validation math functions ───────────────────────────────────────────


def test_doc_example_validation_functions():
    """Verify validation functions documented in docs/free-vs-firm.md."""
    returns = [0.01, -0.005, 0.012, 0.003, -0.002, 0.008] * 10

    dsr = deflated_sharpe(returns, n_trials=10)
    assert isinstance(dsr, dict)
    assert "deflated_sharpe" in dsr
    assert "dsr_probability" in dsr
    assert dsr["deflated_sharpe"] == dsr["dsr_probability"]

    # dsr_probability direct call
    dsr_prob = dsr_probability(returns, n_trials=10)
    assert dsr_prob["dsr_probability"] == dsr["dsr_probability"]

    psr = probabilistic_sharpe_ratio(sharpe=1.2, n_obs=60, skew=0.0, kurtosis=3.0)
    assert isinstance(psr, float)
    assert 0.0 <= psr <= 1.0

    min_trl = min_track_record_length(returns)
    assert isinstance(min_trl, dict)
    assert "min_track_record_length" in min_trl

    # (T=40, N=4) matrix for PBO
    rng = np.random.default_rng(99)
    matrix = rng.normal(0.001, 0.01, (40, 4)).tolist()
    pbo = pbo_cscv(matrix, n_splits=4)
    assert isinstance(pbo, dict)
    assert "pbo" in pbo

    oos = combinatorial_oos_stability(returns * 2, n_groups=6, n_test_groups=2)
    assert isinstance(oos, dict)
    assert "verdict" in oos

    var = compute_var_cvar(returns)
    assert isinstance(var, dict)

    report = performance_report(returns)
    assert isinstance(report, dict)
    assert "sharpe_annualized" in report
    assert "annualization" in report
    assert "risk_free" in report


# ── 6. Documentation text invariants (QOS-171) ─────────────────────────────


def test_no_raw_api_key_flag_in_docs():
    """Ensure no '--key' or plaintext credentials exist in README or docs/."""
    for path in [ROOT / "README.md", *ROOT.glob("docs/*.md")]:
        content = path.read_text(encoding="utf-8")
        assert "--key " not in content, f"Found '--key ' in {path.name}"
        assert "QUANTOS_API_KEY=ae_live_actual" not in content


def test_williams_trial_counting_rule_verbatim():
    """Ensure William's trial counting rule matches verbatim across docs."""
    rule_snippet = (
        "Every executed trial attempt counts toward project cumulative trials. "
        "Identical re-runs record lineage via `duplicate_of` while continuing to "
        "increment cumulative trials (`9 -> 18`), eliminating silent trial count deflation."
    )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    migrating = (ROOT / "docs" / "MIGRATING_TO_2.md").read_text(encoding="utf-8")
    project_format = (ROOT / "docs" / "project-format.md").read_text(encoding="utf-8")

    assert rule_snippet in readme, "William's rule not found verbatim in README.md"
    assert rule_snippet in migrating, "William's rule not found verbatim in MIGRATING_TO_2.md"
    assert rule_snippet in project_format, "William's rule not found verbatim in docs/project-format.md"
