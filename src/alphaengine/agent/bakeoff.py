"""Bakeoff mode. Two roles, one clock, and a referee that does not take votes.

Generators may submit a rule. An adversary may submit an attack. Neither may
start another bakeoff, and neither may name the top-ranked config. The loop is
four passes of two proposals plus one attack after each pass: twelve model calls,
and then it stops even if a model asks to continue.

The clock is the other stop. The default budget is 45 seconds. Nothing can
stretch it past 60. When the budget is gone, the call in flight is abandoned
and the referee scores whatever already came back.

DEPRECATION NOTE
    User-facing terminology now uses ``top_ranked_config`` ("top-ranked config"),
    ``top_ranked_sharpe``, and ``model_weights`` ("model weights"). The keys
    ``survivor``, ``survivor_sharpe``, and ``survivor_weights`` remain in returned
    dicts as backward-compatible API aliases and are deprecated for user-facing
    display.
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
MAX_ROUNDS = 4
MAX_SEATS = 2

_DROPPED = {"winner", "survivor", "score", "sharpe", "verdict", "deflated_sharpe"}

_GENERATOR = """BAKEOFF GENERATOR
Pass @@ROUND@@ of @@ROUNDS@@ (Seat @@SEAT@@ of @@SEATS@@).
Propose one strategy that survives the adversary's active attacks and improves on
the current leader's post-attack standing. Do not repeat a rule+lookback+names
already on the board. You do not score it, you do not name a winner, and you
do not start another bakeoff.
Rules: @@RULES@@
lookback (5 to 60) sets the signal window for momentum/reversion or the calibration
window for hrp, risk_parity, inverse_vol, var_sorted, and low_vol.
Optional names: a subset of 2 to 12 symbols from [@@NAMES@@].
Active attacks: @@ACTIVE_ATTACKS@@
Question:
@@QUESTION@@
Board:
@@BOARD@@
Reply with one JSON object: {"rule": "hrp", "lookback": 30, "why": "short reason"}
"""

_ADVERSARY = """BAKEOFF ADVERSARY
Pass @@ROUND@@ of @@ROUNDS@@.
Name one attack to stress the current leader's regime or tail vulnerability.
Prefer an attack from Unused attacks so each pass adds a new stress dimension.
You do not propose a rule, you do not name a winner, and you do not start another bakeoff.
Attacks: @@ATTACKS@@
Active attacks: @@ACTIVE_ATTACKS@@
Unused attacks: @@UNUSED_ATTACKS@@
Question:
@@QUESTION@@
Board:
@@BOARD@@
Reply with one JSON object: {"attack": "tail_stress", "why": "short reason"}
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
    passes: list[dict[str, Any]] = []
    trace = Trace(question=question)
    stopped = "complete"
    names = _universe_names(data)

    def board() -> str:
        if not submissions and not attacks:
            return "nothing proposed yet"
        scored = referee(data, submissions, attacks)
        lines = list(scored.get("trials") or [])
        if scored.get("survivor"):
            lines.append(
                f"leader {scored.get('survivor')}  "
                f"sharpe {scored.get('survivor_sharpe')}  "
                f"standing {scored.get('attacked_sharpe')}"
            )
        if attacks:
            lines.append("attacks " + ", ".join(dict.fromkeys(attacks)))
        return "\n".join(str(line) for line in lines) or "nothing proposed yet"

    def emit(call: Call) -> None:
        trace.calls.append(call)
        if on_call is not None:
            on_call(call)

    for round_idx in range(1, n_rounds + 1):
        for seat_idx in range(1, n_seats + 1):
            prompt = _fill(
                _GENERATOR,
                question,
                board(),
                round_idx=round_idx,
                n_rounds=n_rounds,
                seat_idx=seat_idx,
                n_seats=n_seats,
                attacks=attacks,
                names=names,
            )
            text, in_time = _ask(think, prompt, deadline)
            if not in_time:
                stopped = "clock"
                break
            submission, why = _proposal(text)
            submission["_round"] = round_idx
            submission["_seat"] = seat_idx
            submissions.append(submission)
            emit(_submit_call(data, submissions, attacks, submission, why, round_idx, seat_idx))
        if stopped == "clock":
            break
        prompt = _fill(
            _ADVERSARY,
            question,
            board(),
            round_idx=round_idx,
            n_rounds=n_rounds,
            seat_idx=0,
            n_seats=n_seats,
            attacks=attacks,
            names=names,
        )
        text, in_time = _ask(think, prompt, deadline)
        if not in_time:
            stopped = "clock"
            break
        attack, why = _attack(text)
        if attack is None:
            emit(
                Call(
                    tool="attack",
                    args={"pass": round_idx},
                    error="that attack is not in the menu",
                )
            )
            continue
        attacks.append(attack)
        scored_pass = referee(data, submissions, attacks)
        passes.append(
            {
                "pass": round_idx,
                "attack": attack,
                "why": why,
                "leader": scored_pass.get("survivor"),
                "leader_sharpe": scored_pass.get("survivor_sharpe"),
                "leader_standing": scored_pass.get("attacked_sharpe"),
                "n_trials": len(submissions),
            }
        )
        emit(_attack_call(scored_pass, attack, why, round_idx))

    figures = referee(data, submissions, attacks)
    elapsed = min(budget, time.monotonic() - (deadline - budget))
    figures["seconds"] = round(elapsed, 1)
    figures["stopped"] = stopped
    figures["budget_seconds"] = budget
    figures["n_passes"] = len(passes) or n_rounds
    top_cfg = figures.get("top_ranked_config") or figures.get("survivor")
    top_sh = (
        figures.get("top_ranked_sharpe")
        if figures.get("top_ranked_sharpe") is not None
        else figures.get("survivor_sharpe")
    )
    figures["top_ranked_config"] = top_cfg
    figures["top_ranked_sharpe"] = top_sh
    trace.answer = _verdict(figures)
    curves = trial_paths(data, submissions, attacks)
    top_label = str(top_cfg or "")
    top_curve = next((c for c in curves if c.get("label") == top_label), None)
    weights = (top_curve or {}).get("weights") or {}
    trace.bakeoff = {
        "top_ranked_config": top_cfg,
        "top_ranked_sharpe": top_sh,
        "model_weights": weights,
        # Deprecated aliases kept for API backward compatibility:
        "survivor": top_cfg,
        "survivor_sharpe": top_sh,
        "survivor_weights": weights,
        "attacked_sharpe": figures.get("attacked_sharpe"),
        "deflated_sharpe": figures.get("deflated_sharpe"),
        "verdict": figures.get("verdict"),
        "n_trials": figures.get("n_trials"),
        "n_failed": figures.get("n_failed"),
        "n_passes": len(passes) or n_rounds,
        "attacks": list(figures.get("attacks") or []),
        "trials": list(figures.get("trials") or []),
        "passes": passes,
        "curves": curves,
    }
    return Bakeoff(trace=trace, figures=figures)


def _submit_call(
    data: Any,
    submissions: list[dict[str, Any]],
    attacks: list[str],
    submission: dict[str, Any],
    why: str,
    round_idx: int,
    seat_idx: int,
) -> Call:
    clean_args = {k: v for k, v in submission.items() if not str(k).startswith("_")}
    scored = referee(data, submissions, attacks)
    trials = list(scored.get("trials") or [])
    last_line = str(trials[-1]) if trials else ""
    rule_name = str(submission.get("rule") or "")
    if not rule_name or "that rule is not in the bakeoff" in last_line:
        return Call(
            tool="submit",
            args=clean_args,
            figures={"pass": round_idx, "seat": seat_idx, "rule": rule_name or "refused"},
            note=f"pass {round_idx}.{seat_idx}  refused",
        )
    parts = last_line.split()
    label = parts[1] if len(parts) > 1 else rule_name
    call_figs: dict[str, Any] = {
        "pass": round_idx,
        "seat": seat_idx,
        "proposal": label,
        "leader": scored.get("survivor"),
        "leader_standing": scored.get("attacked_sharpe"),
    }
    if "lookback" in clean_args:
        call_figs["lookback"] = clean_args["lookback"]
    if why:
        call_figs["why"] = why
    note_tail = " ".join(parts[1:]) if len(parts) > 1 else rule_name
    note = f"pass {round_idx}.{seat_idx}  {note_tail}"
    return Call(tool="submit", args=clean_args, figures=call_figs, note=note)


def _attack_call(
    scored: dict[str, Any],
    attack: str,
    why: str,
    round_idx: int,
) -> Call:
    leader = scored.get("survivor") or "none"
    standing = scored.get("attacked_sharpe")
    call_figs: dict[str, Any] = {
        "pass": round_idx,
        "attack": attack,
        "leader": leader,
        "leader_sharpe": scored.get("survivor_sharpe"),
        "leader_standing": standing,
        "n_trials": scored.get("n_trials"),
    }
    if why:
        call_figs["why"] = why
    note = f"pass {round_idx}  {attack}  leader {leader}  standing {standing}"
    return Call(tool="attack", args={"attack": attack}, figures=call_figs, note=note)


def _universe_names(data: Any) -> list[str]:
    if not isinstance(data, dict):
        return []
    return [str(k) for k in list(data.keys())[:24]]


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


def _fill(
    template: str,
    question: str,
    board: str,
    *,
    round_idx: int = 1,
    n_rounds: int = MAX_ROUNDS,
    seat_idx: int = 1,
    n_seats: int = MAX_SEATS,
    attacks: list[str] | None = None,
    names: list[str] | None = None,
) -> str:
    active = list(dict.fromkeys(attacks or []))
    unused = [a for a in ATTACKS if a not in active]
    return (
        template.replace("@@QUESTION@@", question)
        .replace("@@BOARD@@", board)
        .replace("@@ROUND@@", str(round_idx))
        .replace("@@ROUNDS@@", str(n_rounds))
        .replace("@@SEAT@@", str(seat_idx))
        .replace("@@SEATS@@", str(n_seats))
        .replace("@@RULES@@", ", ".join(RULES))
        .replace("@@ATTACKS@@", ", ".join(ATTACKS))
        .replace("@@ACTIVE_ATTACKS@@", ", ".join(active) or "none yet")
        .replace("@@UNUSED_ATTACKS@@", ", ".join(unused) or "all attacks active")
        .replace("@@NAMES@@", ", ".join(names or []) or "loaded book")
    )


def _proposal(text: str) -> tuple[dict[str, Any], str]:
    obj = _object(text)
    if obj is None:
        return {"rule": ""}, ""
    if "bakeoff" in {str(obj.get("tool") or ""), str(obj.get("rule") or "")}:
        return {"rule": ""}, ""
    rule = obj.get("rule")
    if not isinstance(rule, str):
        return {"rule": ""}, ""
    out: dict[str, Any] = {"rule": rule}
    if "lookback" in obj and obj["lookback"] is not None:
        out["lookback"] = obj["lookback"]
    if isinstance(obj.get("names"), list):
        out["names"] = [str(name) for name in obj["names"][:12]]
    why = str(obj.get("why") or "").strip()[:80]
    return out, why


def _attack(text: str) -> tuple[str | None, str]:
    obj = _object(text)
    if obj is None:
        return None, ""
    attack = str(obj.get("attack") or "").strip().lower().replace("-", "_").replace(" ", "_")
    if attack not in ATTACKS:
        return None, ""
    why = str(obj.get("why") or "").strip()[:80]
    return attack, why


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
    top_cfg = figures.get("top_ranked_config") or figures.get("survivor")
    if not top_cfg:
        return "Referee. No proposal cleared the checks."
    top_sh = (
        figures.get("top_ranked_sharpe")
        if figures.get("top_ranked_sharpe") is not None
        else figures.get("survivor_sharpe")
    )
    return (
        f"Referee. {figures.get('n_trials')} candidates across {figures.get('n_passes', MAX_ROUNDS)} passes, "
        f"source bakeoff. Top-ranked config {top_cfg} (in-sample Sharpe {top_sh}, "
        f"post-attack standing {figures.get('attacked_sharpe')}, "
        f"deflated Sharpe {figures.get('deflated_sharpe')}). Verdict {figures.get('verdict')}."
    )
