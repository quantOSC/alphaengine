"""Daily CS maths and book construction: panel, ICIR, Fama-MacBeth, HRP.

Existing goldens in test_goldens.py stay frozen. These tests pin the NEW
public figures and the directional properties a desk actually relies on.
"""

from __future__ import annotations

import numpy as np
import pytest

from alphaengine.client import StepExecutor, UnsupportedOp
from alphaengine.core import (
    cs_rank,
    cs_winsorize,
    cs_zscore,
    denoise_cov,
    ewma_cov,
    fama_macbeth,
    hrp_weights,
    information_coefficient,
    ledoit_wolf_cov,
    neutralize,
    quantile_book,
    risk_parity_weights,
    signal_icir,
    vol_target,
)
from alphaengine.core.allocate import cov_from_returns
from alphaengine.core.panel import newey_west_tstat


def _cs_panel(n_names: int = 20, n_obs: int = 80, seed: int = 7) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    return {f"N{i:02d}": rng.normal(i * 0.1, 1.0, n_obs).tolist() for i in range(n_names)}


def _signal_prices(n_names: int = 30, n_obs: int = 260, strength: float = 1.0, seed: int = 3):
    rng = np.random.default_rng(seed)
    signal: dict[str, list[float]] = {}
    prices: dict[str, list[float]] = {}
    for i in range(n_names):
        s = (i - n_names / 2) / n_names
        drift = 0.0004 + strength * s * 0.004
        steps = rng.normal(drift, 0.01, n_obs)
        px = 100.0 * np.cumprod(1.0 + steps)
        signal[f"N{i:02d}"] = [float(s)] * n_obs
        prices[f"N{i:02d}"] = [round(float(v), 6) for v in px]
    return signal, prices


# ── panel transforms ───────────────────────────────────────────────────────


def test_cs_zscore_of_a_known_row_is_hand_computable():
    panel = {"A": [1.0, 1.0], "B": [2.0, 2.0], "C": [3.0, 3.0]}
    out = cs_zscore(panel)
    last = [out["panel"][k][-1] for k in ("A", "B", "C")]
    assert last[1] == pytest.approx(0.0, abs=1e-6)
    assert last[0] == pytest.approx(-1.0, abs=1e-6)
    assert last[2] == pytest.approx(1.0, abs=1e-6)
    assert out["n_names"] == 3
    assert out["n_skipped"] == 0


def test_short_names_are_counted_not_silently_dropped():
    panel = _cs_panel(n_names=8)
    panel["STUB"] = [1.0]
    out = cs_rank(panel)
    assert out["n_skipped"] == 1
    assert "STUB" in out["skipped"]
    assert "STUB" not in out["panel"]


def test_winsorize_clips_the_tails_and_keeps_the_middle():
    panel = {f"N{i}": [float(i)] * 5 for i in range(20)}
    out = cs_winsorize(panel, lower=0.1, upper=0.9)
    vals = sorted(out["panel"][k][-1] for k in out["panel"])
    assert vals[0] > 0.0
    assert vals[-1] < 19.0
    assert out["method"] == "winsorize"


def test_neutralize_demean_has_zero_cross_section_mean():
    panel = _cs_panel(n_names=12, n_obs=10)
    out = neutralize(panel)
    row = [out["panel"][k][-1] for k in sorted(out["panel"])]
    finite = [v for v in row if v is not None]
    # Packed figures are rounded to 6 decimals; the mean of a demeaned
    # cross-section is then ~1e-7, not machine zero.
    assert abs(sum(finite) / len(finite)) < 1e-5
    assert out["n_controls"] == 0


def test_neutralize_against_itself_leaves_a_residual_near_zero():
    panel = _cs_panel(n_names=15, n_obs=12)
    out = neutralize(panel, controls=panel)
    last = [v for v in (out["panel"][k][-1] for k in out["panel"]) if v is not None]
    assert max(abs(v) for v in last) < 1e-6
    assert out["n_controls"] == 1


# ── ICIR ───────────────────────────────────────────────────────────────────


def test_information_coefficient_shape_is_unchanged():
    signal, prices = _signal_prices()
    out = information_coefficient(signal, prices, horizon=10)
    assert set(out) == {
        "mean_ic",
        "ic_t_stat",
        "n_periods",
        "ic_by_period",
        "n_names",
        "n_skipped",
        "skipped",
        "horizon",
    }


def test_icir_matches_the_ols_t_stat_and_adds_newey_west():
    signal, prices = _signal_prices()
    out = signal_icir(signal, prices, horizon=10)
    assert out["mean_ic"] is not None and out["mean_ic"] > 0.15
    assert out["icir"] == out["ic_t_stat"]
    assert out["ic_t_stat_nw"] is not None
    assert out["method"] == "spearman"


def test_icir_pearson_is_a_different_measurement():
    signal, prices = _signal_prices()
    sp = signal_icir(signal, prices, horizon=10, method="spearman")
    pe = signal_icir(signal, prices, horizon=10, method="pearson")
    assert pe["method"] == "pearson"
    assert pe["mean_ic"] is not None and pe["mean_ic"] > 0.1
    assert sp["method"] != pe["method"]


def test_newey_west_on_white_noise_is_close_to_ordinary_t():
    x = np.random.default_rng(1).normal(0.5, 1.0, 200)
    nw = newey_west_tstat(x, lags=1)
    ordinary = float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))
    assert nw is not None
    assert abs(nw - ordinary) < 0.5


# ── Fama-MacBeth + quantile book ───────────────────────────────────────────


def test_fama_macbeth_slope_is_positive_on_a_planted_signal():
    signal, prices = _signal_prices()
    out = fama_macbeth(signal, prices, horizon=10)
    assert out["lambda_mean"] is not None and out["lambda_mean"] > 0
    assert out["t_stat"] is not None and out["t_stat"] > 0
    assert out["n_dates"] >= 6
    assert out["n_names"] == 30


def test_quantile_book_turnover_stays_in_unit_interval_times_two():
    signal, prices = _signal_prices()
    out = quantile_book(signal, prices, horizon=10, quantiles=5)
    assert out["spread_pct"] is not None and out["spread_pct"] > 0
    assert out["turnover_one_way"] is not None
    assert 0.0 <= out["turnover_one_way"] <= 2.0
    assert out["n_turnover_obs"] == out["n_periods"] - 1


def test_a_static_signal_has_zero_quantile_turnover():
    signal, prices = _signal_prices()
    out = quantile_book(signal, prices, horizon=10)
    assert out["turnover_one_way"] == pytest.approx(0.0, abs=1e-9)


# ── covariance + HRP ───────────────────────────────────────────────────────


def test_ewma_cov_is_psd():
    rng = np.random.default_rng(2)
    R = rng.normal(0, 0.01, size=(80, 6))
    cov = ewma_cov(R, lam=0.94)
    evals = np.linalg.eigvalsh(cov)
    assert evals.min() >= -1e-10
    assert cov.shape == (6, 6)


def test_ledoit_wolf_shrinkage_is_between_zero_and_one():
    rng = np.random.default_rng(4)
    R = rng.normal(0, 0.01, size=(50, 8))
    cov, shrinkage, mu = ledoit_wolf_cov(R)
    assert 0.0 <= shrinkage <= 1.0
    assert mu > 0
    assert np.linalg.eigvalsh(cov).min() >= -1e-10


def test_denoise_of_identity_clips_every_eigenvalue_and_stays_psd():
    cov = np.eye(6)
    out = denoise_cov(cov, n_obs=100)
    assert out["n_signal_eigenvalues"] == 0
    assert out["lambda_plus"] > 1.0
    evals = np.linalg.eigvalsh(out["cov"])
    assert evals.min() >= -1e-10


def test_hrp_weights_sum_to_one_and_stay_non_negative():
    rng = np.random.default_rng(5)
    R = rng.normal(0, 0.01, size=(120, 6))
    cov = cov_from_returns(R)
    out = hrp_weights(cov, names=[f"S{i}" for i in range(6)])
    w = list(out["weights"].values())
    assert out["n_negative"] == 0
    assert min(w) >= 0
    assert abs(sum(w) - 1.0) < 1e-6


def test_two_asset_equal_vol_hrp_is_half_and_half():
    cov = np.array([[0.04, 0.01], [0.01, 0.04]])
    out = hrp_weights(cov, names=["A", "B"])
    assert out["weights"]["A"] == pytest.approx(0.5, abs=1e-4)
    assert out["weights"]["B"] == pytest.approx(0.5, abs=1e-4)


def test_risk_parity_on_uncorrelated_inverse_vol():
    cov = np.diag([0.01, 0.04, 0.09])
    out = risk_parity_weights(cov, names=["a", "b", "c"])
    w = out["weights"]
    # uncorrelated ERC is inverse-vol: 1/0.1, 1/0.2, 1/0.3 normalised
    inv = np.array([10.0, 5.0, 10.0 / 3.0])
    inv = inv / inv.sum()
    assert w["a"] == pytest.approx(inv[0], abs=0.02)
    assert w["c"] == pytest.approx(inv[2], abs=0.02)
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_vol_target_leverage_is_positive_on_noisy_returns():
    r = np.random.default_rng(8).normal(0.0004, 0.01, 400).tolist()
    out = vol_target(r, target=0.10)
    assert out["n_obs"] == 400
    assert out["mean_leverage"] is not None and out["mean_leverage"] > 0
    assert out["realized_vol"] is not None and out["realized_vol"] > 0


# ── executor: figures out, series stay ─────────────────────────────────────


def test_panel_transform_does_not_emit_the_panel():
    ex = StepExecutor(data=_cs_panel())
    out = ex.execute("compute.panel_transform", {"method": "zscore"})
    assert "panel" not in out
    assert "panel" in ex.workspace
    assert out["method"] == "zscore"
    assert out["n_names"] == 20


def test_signal_icir_op_returns_icir_not_the_ic_series_raw():
    signal, prices = _signal_prices()
    ex = StepExecutor(data={"signal": signal, "prices": prices})
    out = ex.execute("compute.signal_icir", {"horizon": 10})
    assert out["icir"] is not None
    assert isinstance(out["ic_by_period"][0], dict)


def test_hrp_op_emits_weights_not_the_covariance():
    rng = np.random.default_rng(9)
    data = {f"S{i}": rng.normal(0, 0.01, 80).tolist() for i in range(5)}
    ex = StepExecutor(data=data)
    out = ex.execute("compute.hrp", {})
    assert "cov" not in out
    assert "cov" in ex.workspace
    assert abs(sum(out["weights"].values()) - 1.0) < 1e-6
    assert out["n_trials"] == 5
    assert out["n_trials_source"] == "derived_from_names"


def test_vol_target_op_refuses_a_universe():
    ex = StepExecutor(data={"A": [1.0, 2.0], "B": [3.0, 4.0]})
    with pytest.raises(UnsupportedOp):
        ex.execute("compute.vol_target", {})


def test_fama_macbeth_and_quantile_book_ops():
    signal, prices = _signal_prices()
    ex = StepExecutor(data={"signal": signal, "prices": prices})
    fm = ex.execute("compute.fama_macbeth", {"horizon": 10})
    qb = ex.execute("compute.quantile_book", {"horizon": 10})
    assert fm["lambda_mean"] is not None
    assert qb["turnover_one_way"] is not None
