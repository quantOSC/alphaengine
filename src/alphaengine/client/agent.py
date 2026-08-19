"""Bring your own model.

`AgentDriver` was named in three documents for weeks and existed in zero lines
of code. This is it, and the shape it was always described as having is the
shape that matters:

    IT TAKES A CALLABLE. There is no `api_key` parameter on anything here, no
    provider enum, no `import anthropic`. That is not minimalism — it is how
    "runs under the customer's own model key" is satisfied STRUCTURALLY rather
    than promised. A field we could put a key in is a field somebody eventually
    puts a key in, and then we are storing customer model credentials, which is
    more convenient and strictly worse. There is nowhere to put one.

    THE PACKAGE DEPENDS ON NO LLM SDK. `think` is `(prompt: str) -> str`. Wrap
    Anthropic, OpenAI, a local llama.cpp, a company gateway, or a function that
    returns a canned answer for a test. The package cannot tell and must not
    care.

── WHAT THE MODEL IS AND IS NOT ALLOWED TO DO ─────────────────────────────────

The model picks the NEXT STEP FROM `directive.permitted`. It does not author
steps, invent ops, set thresholds or decide when a run is finished. Those live
on the server, in the workflow, and the client is forbidden from knowing them —
"the loop must contain no workflow knowledge" is the standing rule this file has
to be careful not to break.

So `pick()` returns AN INDEX into a list the server produced. That return type
is the enforcement: an index cannot express an op the server did not offer. A
model that hallucinates `compute.make_me_money` produces an out-of-range integer
and gets refused, rather than producing a plausible dictionary that flows on.

── WHY A RUN DRIVEN THIS WAY IS MARKED NOT REPRODUCIBLE ───────────────────────

Two runs of the same exploratory workflow over the same data can take different
paths, because a model chose. That is the point of exploratory mode and it is
also a fact that must travel with the artifact — a study whose PATH was chosen
by a model is a different epistemic object from one whose path was fixed in
advance, and collapsing them would be exactly the kind of quiet overstatement
the trial-count work exists to prevent.

The server already carries the vocabulary (`Workflow.agency`, SCRIPTED vs
EXPLORATORY, with a `reproducible` property). This half just has to be honest
about which one ran.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, Protocol

__all__ = ["AgentDriver", "Think", "AgentRefusal"]

Figures = dict[str, Any]


class Think(Protocol):
    """Any callable that takes a prompt and returns text.

    Deliberately the weakest possible interface. Everything else in this module
    is built so that this is all a caller has to supply.

    THE `/` IS LOad-BEARING. Without it the protocol requires the parameter to
    be *named* `prompt`, so an ordinary `lambda p: ...` or a
    `Callable[[str], str]` fails to satisfy it — mypy caught exactly that. We do
    not care what the caller names their argument; positional-only says so.
    """

    def __call__(self, prompt: str, /) -> str: ...


class AgentRefusal(RuntimeError):
    """The model produced something that is not a choice among what was offered."""


# The prompt is a TEMPLATE OVER THE SERVER'S OWN WORDS. It contains no workflow
# knowledge — no op names, no ordering, no thresholds — because everything it
# describes is read out of the directive at call time. If this string ever
# starts saying "usually you should run the sweep first", the rule has been
# broken and the client has become a second, worse copy of the workflow.
_PROMPT = """\
You are choosing the next step in a quantitative research run.

The goal, in the researcher's words:
{goal}

What has happened so far:
{history}

You may choose exactly ONE of these permitted steps:
{options}

Reply with ONLY a JSON object:
  {{"choice": <integer index>, "why": "<one short sentence>"}}

Choose the index of the step that best advances the goal. You may not invent a
step, and you may not choose anything not listed above.
"""


class AgentDriver:
    """Drives a run by asking a model which permitted step to take next.

    Usage is deliberately three lines:

        driver = AgentDriver(think=my_model, goal="validate this momentum idea")
        run = session.open("validate_study", data=data, backtest_fn=fn)
        driver.drive(run)
    """

    def __init__(
        self,
        think: Think,
        *,
        goal: str,
        on_thought: Callable[[str], None] | None = None,
        max_steps: int = 60,
        sink: Any | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        self.think = think
        self.goal = goal
        # A narrator, so the terminal can show WHY the model chose what it chose.
        # A loop you cannot watch is a loop you cannot trust, and that applies
        # doubly when something non-deterministic is making the choices.
        self.on_thought = on_thought or (lambda _s: None)
        self.max_steps = max_steps
        self.history: list[str] = []
        self.sink = sink
        self.provider = provider
        self.model = model

    def as_choice(self) -> Any:
        """Index-bound `choose` for `alphaengine.agent.AgentDriver`.

        The library driver is the primitive: it takes `(permitted, figures) -> int`
        and cannot name an op. This wrapper is the prompt+goal layer that USES it.
        """

        def choose(permitted: list[Figures], _figures: Figures) -> int:
            return self.pick(permitted)

        return choose

    # ── the one decision ───────────────────────────────────────────────────

    def pick(self, permitted: list[Figures]) -> int:
        """Return an INDEX into `permitted`. Never a step, never an op name.

        The index is the safety property. A model cannot name an operation the
        server did not offer, because the only thing it can return is a position
        in the server's own list.
        """
        if not permitted:
            raise AgentRefusal("nothing is permitted; there is no choice to make")
        if len(permitted) == 1:
            # No sense spending a model call on a list of one, and no sense
            # letting one be got wrong.
            return 0

        options = "\n".join(
            f"  {i}. {p.get('op', '?')}" + (f"  {_brief(p.get('params'))}" if p.get("params") else "")
            for i, p in enumerate(permitted)
        )
        prompt = _PROMPT.format(
            goal=self.goal,
            history="\n".join(f"  - {h}" for h in self.history[-12:]) or "  (nothing yet)",
            options=options,
        )
        raw = self.think(prompt)
        idx, why = _parse_choice(raw, len(permitted))
        if why:
            self.on_thought(why)
            self.history.append(f"chose {permitted[idx].get('op')}: {why}")
        else:
            self.history.append(f"chose {permitted[idx].get('op')}")
        if self.sink is not None:
            import hashlib

            from ..events import make_event

            self.sink.emit(
                make_event(
                    "pick",
                    provider=self.provider,
                    model=self.model,
                    choice=idx,
                    why=why,
                    workflow=str(permitted[idx].get("op") or ""),
                    agency="exploratory",
                    prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:16],
                    prompt_chars=len(prompt),
                )
            )
        return idx

    # ── the loop ───────────────────────────────────────────────────────────

    def drive(self, run: Any) -> Any:
        """Run to completion, asking the model at each fork.

        The structure mirrors `Run.drive` deliberately, including the
        two-identical-failures stop: a step this build cannot execute fails the
        same way every time, and the server correctly re-offers it, so without a
        counter an agentic run spins exactly as the scripted one used to.
        """
        n = 0
        failures: dict[str, int] = {}
        while run.status == "open" and n < self.max_steps:
            if not run.permitted:
                run.resume()
                if run.status != "open" or not run.permitted:
                    break

            step = run.permitted[self.pick(run.permitted)]
            before = run.status
            run.step(step)
            n += 1

            still = any(p.get("step_id") == step.get("step_id") for p in run.permitted)
            if still and before == "open" and run.status == "open":
                key = str(step.get("step_id"))
                failures[key] = failures.get(key, 0) + 1
                self.history.append(f"{step.get('op')} could not be executed here")
                if failures[key] >= 2:
                    run.status = "abandoned"
                    run.stopped = {"reason": "step_failed", "op": step.get("op")}
                    return run
            else:
                self.history.append(f"{step.get('op')} done")

        if run.status == "open" and n >= self.max_steps:
            raise AgentRefusal(
                f"the run was still open after {self.max_steps} steps; "
                "stopping rather than continuing to spend model calls"
            )
        return run


# ── parsing, which has to assume the model is sloppy ───────────────────────


def _brief(params: Any) -> str:
    """A short, SAFE rendering of a step's params for the prompt.

    Truncated hard. Params can carry a grid, and a grid is frequently bigger
    intellectual property than the returns — there is no reason to send all of
    it to a third-party model just to label a menu item.
    """
    try:
        s = json.dumps(params, default=str)
    except (TypeError, ValueError):
        s = str(params)
    return s[:120] + ("…" if len(s) > 120 else "")


def _parse_choice(raw: str, n: int) -> tuple[int, str]:
    """Pull `{"choice": i, "why": "..."}` out of whatever the model actually said.

    Models wrap JSON in prose, in code fences, or answer with a bare number.
    Each of those is a real thing they do, so each is handled — but an answer
    that is not a valid index is REFUSED rather than defaulted to 0. Silently
    taking the first option when the model said something unparseable would
    produce a run that looks agent-driven and is not.
    """
    text = (raw or "").strip()

    obj: Any = None
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
        except ValueError:
            obj = None

    if isinstance(obj, dict) and "choice" in obj:
        try:
            idx = int(obj["choice"])
        except (TypeError, ValueError):
            raise AgentRefusal(f"choice was not an integer: {obj['choice']!r}") from None
        why = str(obj.get("why") or "").strip()
    else:
        bare = re.search(r"-?\d+", text)
        if not bare:
            raise AgentRefusal(f"no choice found in the model's reply: {raw[:200]!r}")
        idx, why = int(bare.group(0)), ""

    if not 0 <= idx < n:
        raise AgentRefusal(
            f"the model chose step {idx}, which was not offered (0..{n - 1}). It cannot invent a step."
        )
    return idx, why
