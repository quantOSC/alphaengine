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
    assert len(calls) == 6
    assert done.figures["n_trials"] == 4
    assert done.figures["n_failed"] == 4
    assert done.figures["survivor"] is None
    assert done.figures["budget_seconds"] == HARD_CEILING
    assert done.figures["stopped"] == "complete"
    assert done.trace.answer.startswith("Referee.")


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
