"""Bakeoff: agents propose and attack. The referee names the survivor. The clock stops the room."""

from __future__ import annotations

import time

import numpy as np

from alphaengine.agent.bakeoff import HARD_CEILING, run_bakeoff
from alphaengine.core.bakeoff import choose_survivor, referee
from alphaengine.tui.dispatch import Desk, _begin_bakeoff


def _book() -> dict[str, list[float]]:
    rng = np.random.default_rng(4)
    return {
        "AAA": (100 * np.cumprod(1 + rng.normal(0.001, 0.01, 80))).tolist(),
        "BBB": (50 * np.cumprod(1 + rng.normal(0.0002, 0.008, 80))).tolist(),
    }


def test_the_attacked_strategy_beats_the_in_sample_spike() -> None:
    rows = [
        {"index": 0, "failed": None, "sharpe": 2.0, "standing": -1.0},
        {"index": 1, "failed": None, "sharpe": 0.3, "standing": 0.2},
        {"index": 2, "failed": "the rule did not run", "sharpe": None, "standing": None},
    ]
    assert choose_survivor(rows) == 1


def test_a_claimed_winner_is_ignored_and_a_failure_counts() -> None:
    scored = referee(
        _book(),
        [{"rule": "equal", "winner": "reversion"}, {"rule": "invented"}],
        ["last_half"],
    )
    assert scored["n_trials"] == 2
    assert scored["n_trials_source"] == "bakeoff"
    assert scored["n_failed"] == 1
    assert str(scored["survivor"]).startswith("equal")
    assert "reversion" not in str(scored["survivor"])


def test_a_bakeoff_cannot_start_another_and_cannot_run_past_a_minute() -> None:
    calls: list[str] = []

    def think(prompt: str) -> str:
        calls.append(prompt)
        if "ADVERSARY" in prompt:
            return '{"attack": "last_half", "winner": "momentum"}'
        return '{"tool": "bakeoff", "winner": "equal"}'

    done = run_bakeoff("bake these off", think, _book(), rounds=9, seats=9, seconds=10_000)
    assert len(calls) == 12
    assert done.figures["n_trials"] == 8
    assert done.figures["n_failed"] == 8
    assert done.figures["survivor"] is None
    assert done.figures["budget_seconds"] == HARD_CEILING
    assert done.figures["stopped"] == "complete"
    assert done.trace.answer.startswith("Referee.")


def test_multi_pass_gan_bakeoff_adapts_across_rounds_and_scores_attacks() -> None:
    seen_boards: list[str] = []
    gen_Script = iter(
        [
            '{"rule": "equal", "why": "baseline equal weight"}',
            '{"rule": "momentum", "lookback": 15, "why": "trend sleeve"}',
            '{"rule": "hrp", "lookback": 25, "why": "cluster diversification"}',
            '{"rule": "risk_parity", "why": "equal risk contribution"}',
            '{"rule": "var_sorted", "why": "inverse tail loss"}',
            '{"rule": "low_vol", "why": "defensive low variance"}',
            '{"rule": "reversion", "lookback": 10, "why": "short horizon mean reversion"}',
            '{"rule": "inverse_vol", "why": "volatility scaled"}',
        ]
    )
    adv_script = iter(
        [
            '{"attack": "last_half", "why": "out-of-sample regime"}',
            '{"attack": "high_vol", "why": "volatility spike days"}',
            '{"attack": "tail_stress", "why": "1.5x downside amplification"}',
            '{"attack": "rate_shock", "why": "funding and dispersion shock"}',
        ]
    )

    def think(prompt: str) -> str:
        seen_boards.append(prompt)
        if "ADVERSARY" in prompt:
            return next(adv_script)
        return next(gen_Script)

    done = run_bakeoff("construct a resilient book", think, _book())
    assert len(seen_boards) == 12
    assert "Pass 4 of 4" in seen_boards[-1]
    assert "standing" in seen_boards[-2]
    assert done.figures["n_trials"] == 8
    assert done.figures["n_failed"] == 0
    assert done.figures["n_passes"] == 4
    assert done.figures["attacks"] == ["last_half", "high_vol", "tail_stress", "rate_shock"]
    assert done.trace.bakeoff is not None
    assert len(done.trace.bakeoff["passes"]) == 4
    assert len(done.trace.bakeoff["curves"]) == 8
    assert done.trace.bakeoff["top_ranked_config"] == done.trace.bakeoff["survivor"]
    assert done.trace.bakeoff["model_weights"] == done.trace.bakeoff["survivor_weights"]
    assert done.trace.bakeoff["model_weights"]
    assert "Top-ranked config" in done.trace.answer


def test_the_clock_ends_the_bakeoff() -> None:
    def think(prompt: str) -> str:
        time.sleep(0.4)
        return '{"rule": "equal"}'

    started = time.monotonic()
    done = run_bakeoff("bake these off", think, _book(), seconds=0.2)
    assert time.monotonic() - started < 1.5
    assert done.figures["stopped"] == "clock"
    assert done.figures["n_trials"] == 0
    assert done.figures["budget_seconds"] == 0.2


def test_a_sentence_gets_one_bakeoff() -> None:
    def think(prompt: str) -> str:
        if "ADVERSARY" in prompt:
            return '{"attack": "drop_best"}'
        return '{"rule": "equal"}'

    desk = Desk(session=object(), url="fake://", args=None, data=_book(), think=think)
    desk.question = "bakeoff the book"
    first = _begin_bakeoff(desk, think, seconds=5)
    second = _begin_bakeoff(desk, think, seconds=5)
    assert str(first.figures["survivor"]).startswith("equal")
    assert first.figures["n_trials_source"] == "bakeoff"
    assert second.figures["stopped"] == "already_ran"
    assert "Referee." in first.trace.answer
