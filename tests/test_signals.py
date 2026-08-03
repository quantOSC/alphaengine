"""Signal evaluation, stress and hygiene: the modeling week's own math.

THE TESTS THAT MATTER ARE THE DIRECTIONAL ONES. A planted signal must read
positive, its reverse must read negative, and a constant must read as NOTHING
— None, never zero, because zero is a measurement and a constant column is the
absence of one. Exact values are pinned where they are computable by hand;
everything else is pinned by construction with fixed seeds.
"""

from __future__ import annotations

import random

import pytest

from alphaengine.core import (
    cost_ladder,
    drawdown_anatomy,
    information_coefficient,
    overlap_stats,
    profile_data,
    quantile_returns,
    signal_decay,
    subperiod_stability,
)
from alphaengine.core.signals import PanelShapeError


def _panels(n_names: int = 30, n_obs: int = 260, strength: float = 1.0, seed: int = 3):
    """Prices whose forward returns the signal partially KNOWS, by construction.

    Each name's per-period drift is proportional to its (fixed) signal value,
    so ranking by signal ranks by expected forward return. `strength=0` breaks
    the link and the IC must collapse toward zero.
    """
    rng = random.Random(seed)
    signal: dict[str, list[float]] = {}
    prices: dict[str, list[float]] = {}
    for i in range(n_names):
        s = (i - n_names / 2) / n_names
        drift = 0.0004 + strength * s * 0.004
        px, series = 100.0, []
        for _ in range(n_obs):
            px *= 1.0 + rng.gauss(drift, 0.01)
            series.append(round(px, 6))
        signal[f"N{i:02d}"] = [s] * n_obs
        prices[f"N{i:02d}"] = series
    return signal, prices


# ── information coefficient ────────────────────────────────────────────────
def test_a_planted_signal_reads_positive_and_its_reverse_reads_negative():
    signal, prices = _panels()
    out = information_coefficient(signal, prices, horizon=10)
    assert out["mean_ic"] is not None and out["mean_ic"] > 0.15
    assert out["n_periods"] >= 6

    reversed_signal = {k: [-x for x in v] for k, v in signal.items()}
    back = information_coefficient(reversed_signal, prices, horizon=10)
    assert back["mean_ic"] is not None and back["mean_ic"] < -0.15


def test_a_constant_signal_measures_nothing_and_never_zero():
    """None is the absence of a measurement; 0 is a measurement. A constant
    column has no ranking to correlate and must not read as 'no edge'."""
    signal, prices = _panels()
    flat = {k: [1.0] * len(v) for k, v in signal.items()}
    out = information_coefficient(flat, prices, horizon=10)
    assert out["mean_ic"] is None
    assert out["n_periods"] == 0


def test_short_names_are_counted_and_named_not_silently_dropped():
    signal, prices = _panels(n_names=12)
    signal["STUB"] = [0.1]
    prices["STUB"] = [100.0]
    out = information_coefficient(signal, prices, horizon=10)
    assert out["n_skipped"] == 1
    assert "STUB" in out["skipped"]
    assert out["n_names"] == 12


def test_panels_must_be_mappings():
    with pytest.raises(PanelShapeError):
        information_coefficient([1, 2, 3], {"A": [1.0, 2.0]}, horizon=1)


def test_periods_are_non_overlapping():
    """260 observations at horizon 10 is at most 25 windows — stepping BY the
    horizon, not by one. Overlapping windows share returns and inflate the
    t-stat in the flattering direction."""
    signal, prices = _panels(n_obs=260)
    out = information_coefficient(signal, prices, horizon=10)
    assert out["n_periods"] <= 25


# ── quantiles and decay ────────────────────────────────────────────────────
def test_the_top_bucket_beats_the_bottom_on_a_planted_signal():
    signal, prices = _panels()
    out = quantile_returns(signal, prices, horizon=10, quantiles=5)
    assert out["spread_pct"] is not None and out["spread_pct"] > 0
    assert len(out["quantile_mean_pct"]) == 5


def test_decay_reports_none_for_horizons_the_data_cannot_support():
    signal, prices = _panels(n_obs=80)
    out = signal_decay(signal, prices, horizons=(5, 200))
    assert out["ic_at_horizon"][0] is not None
    assert out["ic_at_horizon"][1] is None


# ── stress ─────────────────────────────────────────────────────────────────
def test_one_hot_stretch_is_unmasked_by_the_best_segment_share():
    quiet = [0.0001] * 90
    hot = [0.01] * 90
    out = subperiod_stability(quiet + hot + quiet + quiet, segments=4)
    assert out["share_of_pnl_in_best_segment"] is not None
    assert out["share_of_pnl_in_best_segment"] > 0.9


def test_the_cost_ladder_finds_where_it_dies_and_never_guesses_turnover():
    rng = random.Random(11)
    rets = [rng.gauss(0.0004, 0.008) for _ in range(500)]
    out = cost_ladder(rets, turnover=1.0)
    assert out["bps_levels"] == [0.0, 5.0, 10.0, 25.0, 50.0]
    assert out["dies_at_bps"] is not None

    absent = cost_ladder(rets, turnover=None)
    assert absent["dies_at_bps"] is None
    assert all(s is None for s in absent["sharpe_at_bps"])
    assert absent["turnover"] is None


def test_drawdown_anatomy_reads_a_constructed_drawdown():
    rets = [0.01] * 50 + [-0.02] * 20 + [0.01] * 60
    out = drawdown_anatomy(rets)
    assert out["max_drawdown_pct"] == pytest.approx(100 * (1 - 0.98**20), rel=1e-6)
    assert out["longest_underwater_periods"] >= 20
    assert out["n_drawdowns_over_10pct"] == 1


# ── overlap ────────────────────────────────────────────────────────────────
def test_the_same_series_is_the_book_again_and_noise_is_not():
    rng = random.Random(7)
    book = [rng.gauss(0.0, 0.01) for _ in range(300)]
    same = overlap_stats(book, book)
    assert same["correlation"] == pytest.approx(1.0, abs=1e-9)
    assert same["beta_to_book"] == pytest.approx(1.0, abs=1e-9)

    other = [rng.gauss(0.0, 0.01) for _ in range(300)]
    fresh = overlap_stats(other, book)
    assert fresh["correlation"] is not None and abs(fresh["correlation"]) < 0.2


def test_too_little_shared_history_is_an_absence():
    out = overlap_stats([0.01] * 5, [0.02] * 5)
    assert out["correlation"] is None
    assert out["n_obs"] == 5


# ── hygiene ────────────────────────────────────────────────────────────────
def test_the_profile_names_what_is_wrong_and_repairs_nothing():
    prices = {
        "OK": [100.0 + i * 0.1 for i in range(100)],
        "SPLIT": [100.0] * 50 + [50.0] + [50.0 + i * 0.1 for i in range(49)],
        "STALE": [100.0] * 100,
        "SHORT": [100.0, 101.0],
        "BAD": ["x", "y"],
    }
    out = profile_data(prices)
    assert out["n_names"] == 5
    assert out["usable"] == 4
    assert "BAD" in out["unreadable"]
    assert "SPLIT" in out["spike_names"]
    assert "STALE" in out["stale_names"]
    assert "SHORT" in out["ragged"]
    assert out["dates_supplied"] is False


def test_row_shaped_series_report_dates_supplied():
    prices = {"A": [{"date": "2026-01-01", "close": 100.0}, {"date": "2026-01-02", "close": 101.0}]}
    out = profile_data(prices)
    assert out["dates_supplied"] is True
    assert out["usable"] == 1
