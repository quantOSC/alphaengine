"""Screening a supplied universe: ranking, bounds, and the three ways to miss.

The interesting tests here are not the ranking ones. They are the ones that pin
what happens to a name the screen could NOT measure, because that is where a
screen quietly becomes a survivorship artifact: drop the unmeasurable names and
every screen reports full coverage over whatever was left.
"""

from __future__ import annotations

import numpy as np
import pytest

from alphaengine.core import screen_universe
from alphaengine.core.screen import MAX_ROWS, METRICS, UnknownMetric


def walk(seed: int, n: int, drift: float = 0.0005) -> list[float]:
    """A price path. Seeded, so it is the same series on every machine."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, 0.01, n)
    return (100.0 * np.cumprod(1.0 + steps)).tolist()


def universe(n_names: int = 30, n_obs: int = 300) -> dict[str, list[float]]:
    return {f"SYM{i:02d}": walk(i, n_obs, drift=0.0002 * i) for i in range(n_names)}


# ── ranking ────────────────────────────────────────────────────────────────
def test_ranks_by_the_named_metric_best_first() -> None:
    out = screen_universe(universe(), rank_by="return_pct", top_n=10)
    scores = [row["score"] for row in out["rows"]]
    assert scores == sorted(scores, reverse=True)
    assert [row["rank"] for row in out["rows"]] == list(range(1, len(out["rows"]) + 1))


def test_descending_false_ranks_the_small_end_first() -> None:
    """Volatility is a metric where small is the interesting end, so the
    direction has to be a caller's choice rather than a property of the metric."""
    out = screen_universe(universe(), rank_by="volatility_pct", descending=False, top_n=5)
    scores = [row["score"] for row in out["rows"]]
    assert scores == sorted(scores)


def test_the_score_is_the_metric_that_was_ranked_on() -> None:
    out = screen_universe(universe(), rank_by="rsi", top_n=5)
    for row in out["rows"]:
        assert row["score"] == row["rsi"]


def test_every_advertised_metric_can_actually_be_ranked_on() -> None:
    """`METRICS` is what the error message offers the caller. A name in that
    tuple that does not work would send somebody in a circle."""
    for metric in METRICS:
        out = screen_universe(universe(), rank_by=metric, top_n=3)
        assert out["rank_by"] == metric
        assert out["rows"], f"{metric} ranked nothing"


# ── filters carry values, never expressions ────────────────────────────────
def test_a_filter_bounds_the_result() -> None:
    out = screen_universe(universe(), rank_by="rsi", filters={"rsi": {"min": 50.0}}, top_n=50)
    assert out["rows"], "the fixture should leave something above 50"
    for row in out["rows"]:
        assert row["rsi"] >= 50.0


def test_a_filter_on_a_metric_other_than_the_ranked_one_still_applies() -> None:
    out = screen_universe(
        universe(),
        rank_by="return_pct",
        filters={"volatility_pct": {"max": 100.0}},
        top_n=50,
    )
    for row in out["rows"]:
        assert row["volatility_pct"] <= 100.0


def test_an_unknown_metric_is_refused_rather_than_ignored() -> None:
    """A silently dropped filter returns MORE names than were asked for, and the
    extra ones look like findings."""
    with pytest.raises(UnknownMetric):
        screen_universe(universe(), rank_by="vibes")
    with pytest.raises(UnknownMetric):
        screen_universe(universe(), filters={"momentum_score": {"min": 1}})


# ── the three ways to miss, kept apart ─────────────────────────────────────
def test_a_name_too_short_to_measure_is_counted_not_dropped() -> None:
    """The whole point. A shortlist of 20 from 500 means something different
    when 400 could not be measured than when 400 were measured and rejected."""
    names = universe(20, 300)
    names.update({f"NEW{i}": walk(90 + i, 5) for i in range(5)})

    out = screen_universe(names, rank_by="return_pct", top_n=50)
    assert out["universe_size"] == 25
    assert out["n_insufficient"] == 5
    assert set(out["insufficient"]) == {f"NEW{i}" for i in range(5)}
    assert all(not row["symbol"].startswith("NEW") for row in out["rows"])


def test_an_empty_series_is_insufficient_not_an_error() -> None:
    names = universe(20, 300)
    names["EMPTY"] = []
    out = screen_universe(names, rank_by="return_pct")
    assert "EMPTY" in out["insufficient"]


def test_a_name_missing_a_FILTERED_metric_is_undetermined_not_failing() -> None:
    """Third state, and it is load-bearing. A name with no RSI has not failed an
    RSI filter — reporting it as rejected is a claim the data cannot support."""
    names = universe(20, 300)
    # Long enough for a 63-day return, too short for a 200-day SMA.
    names["MID"] = walk(77, 80)

    out = screen_universe(
        names,
        rank_by="return_pct",
        filters={"distance_from_sma_pct": {"min": 0.0}},
        top_n=50,
    )
    assert "MID" in out["undetermined"]
    assert "MID" not in out["insufficient"]
    assert all(row["symbol"] != "MID" for row in out["rows"])


def test_the_counts_close() -> None:
    """universe = evaluated + insufficient, and evaluated = passing + failing +
    undetermined. Arithmetic that does not close is a count somebody will read
    as a coverage figure and be wrong about."""
    names = universe(24, 300)
    names.update({"SHORT": walk(80, 4), "MID": walk(81, 80)})

    out = screen_universe(
        names,
        rank_by="return_pct",
        filters={"distance_from_sma_pct": {"min": -100.0}, "rsi": {"min": 0.0}},
        top_n=50,
    )
    assert out["universe_size"] == out["n_evaluated"] + out["n_insufficient"]
    assert out["n_evaluated"] == out["n_passing"] + out["n_failing"] + out["n_undetermined"]


# ── the result is a shortlist, not a data export ───────────────────────────
def test_top_n_is_honoured_and_truncation_is_declared() -> None:
    out = screen_universe(universe(30), rank_by="return_pct", top_n=5)
    assert out["n_returned"] == 5
    assert out["truncated"] is True
    assert out["n_passing"] > 5


def test_the_row_cap_holds_against_a_greedy_caller() -> None:
    """The harness refuses any list over 64 elements on both sides of the wire,
    so a screen that could return 500 rows would fail at the boundary instead of
    here — with an error about series length, three layers from the cause."""
    out = screen_universe(universe(200, 300), rank_by="return_pct", top_n=10_000)
    assert len(out["rows"]) <= MAX_ROWS
    assert len(out["insufficient"]) <= MAX_ROWS
    assert len(out["undetermined"]) <= MAX_ROWS


def test_an_empty_screen_is_a_result_not_a_failure() -> None:
    """'No name met these conditions' is an answer. It must not raise, and it
    must not come back looking like a coverage problem."""
    out = screen_universe(universe(), rank_by="rsi", filters={"rsi": {"min": 99.9}}, top_n=10)
    assert out["rows"] == []
    assert out["n_passing"] == 0
    assert out["n_evaluated"] > 0
    assert out["n_insufficient"] == 0


def test_an_empty_universe_says_so_rather_than_dividing_by_it() -> None:
    out = screen_universe({}, rank_by="return_pct")
    assert out["universe_size"] == 0
    assert out["rows"] == []


# ── the metrics agree with the feature table ───────────────────────────────
def test_rsi_matches_technical_features_on_the_same_series() -> None:
    """One implementation, imported rather than re-derived. A screen disagreeing
    with the feature table about the same symbol's RSI is the kind of
    contradiction that makes a reviewer stop believing both."""
    from alphaengine.core import technical_features

    names = universe(21, 300)
    screened = {row["symbol"]: row["rsi"] for row in screen_universe(names, rank_by="rsi", top_n=50)["rows"]}
    features = technical_features(names)["features"]

    assert screened, "nothing to compare"
    for symbol, rsi in screened.items():
        assert rsi == features[symbol]["rsi"]["value"]


# ── determinism ────────────────────────────────────────────────────────────
def test_the_same_universe_screens_the_same_way_twice() -> None:
    names = universe()
    assert screen_universe(names, rank_by="return_pct") == screen_universe(names, rank_by="return_pct")


def test_a_dated_mapping_universe_screens_identically_to_bare_lists() -> None:
    """`{date: close}` is what a dated CSV loads as and what the portal's cache
    serves. The dates change WHAT CAN BE CHECKED, never what the numbers are:
    the same closes must produce the numerically identical shortlist."""
    lists = universe(12, 300)
    dated = {sym: {f"2026-{i:04d}": v for i, v in enumerate(series)} for sym, series in lists.items()}
    for metric in ("return_pct", "rsi"):
        assert screen_universe(dated, rank_by=metric, top_n=10) == screen_universe(
            lists, rank_by=metric, top_n=10
        )


def test_ties_break_on_the_symbol_so_the_order_is_total() -> None:
    """Two names with an identical score must not swap places between runs. A
    ranking that is stable only up to ties is not reproducible, and reproducible
    is the claim this package makes on its front page."""
    flat = {name: [100.0] * 300 for name in ("CCC", "AAA", "BBB")}
    out = screen_universe(flat, rank_by="return_pct", top_n=3)
    assert [row["symbol"] for row in out["rows"]] == ["CCC", "BBB", "AAA"]
