"""Bakeoff mode. Two roles, one clock, and a referee that does not take votes.

Generators may submit a rule. An adversary may submit an attack. Neither may
start another bakeoff, and neither may name the survivor. The loop is two
rounds of two proposals plus one attack after each round: six model calls,
and then it stops even if a model asks to continue.

The clock is the other stop. The default budget is 45 seconds. Nothing can
stretch it past 60. When the budget is gone, the call in flight is abandoned
and the referee scores whatever already came back.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..core.bakeoff import ATTACKS, RULES, referee, trial_paths
from .sim import Call, Think, Trace

DEFAULT_SECONDS = 45.0
HARD_CEILING = 60.0
MAX_ROUNDS = 2
MAX_SEATS = 2

_DROPPED = {"winner", "survivor", "score", "sharpe", "verdict", "deflated_sharpe"}

_GENERATOR = """BAKEOFF GENERATOR
Propose one strategy. You do not score it, you do not name a winner, and you
do not start another bakeoff.
Rules: @@RULES@@
lookback, for momentum and reversion, is 5 to 60.
Question:
@@QUESTION@@
Board:
@@BOARD@@
Reply with one JSON object: {"rule": "momentum", "lookback": 20}
"""

_ADVERSARY = """BAKEOFF ADVERSARY
Name one attack. You do not propose a rule, you do not name a winner, and you
do not start another bakeoff.
Attacks: @@ATTACKS@@
Question:
@@QUESTION@@
Board:
@@BOARD@@
Reply with one JSON object: {"attack": "last_half"}
"""


@dataclass
class Bakeoff:
    """The trace the session shows, and the figures a tool result is allowed to carry."""

    trace: Trace
    figures: dict[str, Any]


def run_bakeoff(
    question: str,
    think: Think,
    data: Any,
    *,
    rounds: int = MAX_ROUNDS,
    seats: int = MAX_SEATS,
    seconds: float = DEFAULT_SECONDS,
    on_call: Callable[[Call], None] | None = None,
) -> Bakeoff:
    """Run one bakeoff. Returns when the rounds end or the clock does, whichever is first."""
    budget = _budget(seconds)
    deadline = time.monotonic() + budget
    n_rounds = min(max(int(rounds), 1), MAX_ROUNDS)
    n_seats = min(max(int(seats), 1), MAX_SEATS)
    submissions: list[dict[str, Any]] = []
    attacks: list[str] = []
    trace = Trace(question=question)
    stopped = "complete"

    def board() -> str:
        if not submissions and not attacks:
            return "nothing proposed yet"
        scored = referee(data, submissions, attacks)
        lines = list(scored.get("trials") or [])
        if attacks:
            lines.append("attacks " + ", ".join(attacks))
        return "\n".join(str(line) for line in lines) or "nothing proposed yet"

    def emit(call: Call) -> None:
        trace.calls.append(call)
        if on_call is not None:
            on_call(call)

    for _round in range(n_rounds):
        for _seat in range(n_seats):
            text, in_time = _ask(think, _fill(_GENERATOR, question, board()), deadline)
            if not in_time:
                stopped = "clock"
                break
            submission = _proposal(text)
            submissions.append(submission)
            emit(Call(tool="submit", args=submission, note=str(submission.get("rule") or "refused")))
        if stopped == "clock":
            break
        text, in_time = _ask(think, _fill(_ADVERSARY, question, board()), deadline)
        if not in_time:
            stopped = "clock"
            break
        attack = _attack(text)
        if attack is None:
            emit(Call(tool="attack", args={}, error="that attack is not in the menu"))
            continue
        attacks.append(attack)
        emit(Call(tool="attack", args={"attack": attack}, note=attack))

    figures = referee(data, submissions, attacks)
    elapsed = min(budget, time.monotonic() - (deadline - budget))
    figures["seconds"] = round(elapsed, 1)
    figures["stopped"] = stopped
    figures["budget_seconds"] = budget
    trace.answer = _verdict(figures)
    trace.bakeoff = {
        "survivor": figures.get("survivor"),
        "survivor_sharpe": figures.get("survivor_sharpe"),
        "attacked_sharpe": figures.get("attacked_sharpe"),
        "deflated_sharpe": figures.get("deflated_sharpe"),
        "verdict": figures.get("verdict"),
        "n_trials": figures.get("n_trials"),
        "n_failed": figures.get("n_failed"),
        "attacks": list(figures.get("attacks") or []),
        "trials": list(figures.get("trials") or []),
        "curves": trial_paths(data, submissions),
    }
    return Bakeoff(trace=trace, figures=figures)


def _budget(seconds: float) -> float:
    """A positive request, never longer than a minute. A bad request uses the default."""
    try:
        number = float(seconds)
    except (TypeError, ValueError):
        return DEFAULT_SECONDS
    if not math.isfinite(number) or number <= 0:
        return DEFAULT_SECONDS
    return min(HARD_CEILING, number)


def _ask(think: Think, prompt: str, deadline: float) -> tuple[str, bool]:
    """One model call. The second value is false when the clock won and the loop must stop.

    The worker is a daemon so a model that keeps talking after the deadline
    cannot hold the process open. That reply is never read.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return "", False
    box: list[str] = []

    def run() -> None:
        try:
            text = think(prompt)
        except Exception:  # noqa: BLE001 — a broken call is a refused turn, not a crash
            return
        if isinstance(text, str):
            box.append(text)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(remaining)
    if worker.is_alive():
        return "", False
    return (box[0] if box else ""), True


def _fill(template: str, question: str, board: str) -> str:
    return (
        template.replace("@@QUESTION@@", question)
        .replace("@@BOARD@@", board)
        .replace("@@RULES@@", ", ".join(RULES))
        .replace("@@ATTACKS@@", ", ".join(ATTACKS))
    )


def _proposal(text: str) -> dict[str, Any]:
    obj = _object(text)
    if obj is None:
        return {"rule": ""}
    if "bakeoff" in {str(obj.get("tool") or ""), str(obj.get("rule") or "")}:
        return {"rule": ""}
    rule = obj.get("rule")
    if not isinstance(rule, str):
        return {"rule": ""}
    out: dict[str, Any] = {"rule": rule}
    if "lookback" in obj:
        out["lookback"] = obj["lookback"]
    if isinstance(obj.get("names"), list):
        out["names"] = [str(name) for name in obj["names"][:12]]
    return out


def _attack(text: str) -> str | None:
    obj = _object(text)
    if obj is None:
        return None
    attack = str(obj.get("attack") or "").strip().lower().replace("-", "_").replace(" ", "_")
    if attack not in ATTACKS:
        return None
    return attack


def _object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    nested = obj.get("args")
    if isinstance(nested, dict) and "rule" not in obj and "attack" not in obj:
        obj = nested
    return {str(key): value for key, value in obj.items() if str(key) not in _DROPPED}


def _verdict(figures: dict[str, Any]) -> str:
    survivor = figures.get("survivor")
    if not survivor:
        return "Referee. No proposal survived."
    return (
        f"Referee. {figures.get('n_trials')} candidates, source bakeoff. "
        f"Survivor {survivor}. Verdict {figures.get('verdict')}."
    )
