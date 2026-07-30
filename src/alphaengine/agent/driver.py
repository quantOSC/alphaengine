"""The agent driver: a model chooses among the steps it was just offered.

    from alphaengine.agent import AgentDriver

    driver = AgentDriver(choose=my_model_fn)   # your key, your model, your call
    driver.drive(run)

WHAT BOUNDS THE AGENT, CONCRETELY
    Its action space is the permitted set the server sent this turn. Not the
    library's function list, not a tool catalogue, not anything it can name.

    So it cannot call something that was not offered, cannot skip a gate (gates
    are evaluated server side on the result), cannot proceed past a stop (no
    next step is issued), and cannot act without a record (the only thing it can
    do is execute steps, and every step writes).

    That is true by construction rather than by policy, which is the sentence
    worth putting in front of a compliance reviewer. This module is deliberately
    small: the guarantee comes from what it CANNOT reach, so the code that could
    weaken it is code that is not here.

YOUR MODEL, YOUR KEY, YOUR MACHINE
    `choose` is a callable you supply. This package has no model dependency, no
    API key handling, and makes no inference call. It never sees your
    credentials because it never has a reason to.

THE ONE THING THE AGENT ACTUALLY DECIDES
    On a selection="all" turn, nothing: the steps do not branch and all of them
    run. On selection="any", it picks one. That is the real decision point, and
    keeping it that narrow is what makes an agent-driven run as auditable as a
    hand-driven one.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["AgentDriver", "Choice", "RefusedChoice"]

# Your model, given the permitted steps and what the run has produced so far,
# returns the index of the step to take. Nothing else: not a step it invented,
# not a parameter override, not an instruction.
Choice = Callable[[list[dict[str, Any]], dict[str, Any]], int]


class RefusedChoice(RuntimeError):
    """The chooser returned something outside the permitted set.

    Raised rather than clamped. Silently coercing an out-of-range choice to a
    valid one would mean the run continues while the record says the agent chose
    something it did not, and a trace that misreports the decision is worse than
    a run that stops.
    """


@dataclass
class AgentDriver:
    """Drives a run by asking `choose` which permitted step to take."""

    choose: Choice
    max_steps: int = 200

    def drive(self, run: Any) -> Any:
        """Run until it stops or closes.

        A stop returns normally. The agent is not given the chance to react to
        one, because there is nothing to react to: no next step is issued, and
        an agent that could argue with a gate would not be bounded by it.
        """
        n = 0
        while run.status == "open":
            if n >= self.max_steps:
                raise RuntimeError(f"run {run.run_id} exceeded {self.max_steps} steps")

            if not run.permitted:
                run.resume()
                if run.status != "open" or not run.permitted:
                    break

            if run.selection == "all":
                # No decision to make: these do not branch, so all of them run.
                # The agent is not consulted, which is correct rather than a
                # shortcut. Asking it to "choose" among steps that all execute
                # would invent a decision and then record it as one.
                for step in list(run.permitted):
                    run.step(step)
                    n += 1
                    if run.status != "open":
                        return run
                continue

            index = self._ask(run)
            run.step(run.permitted[index])
            n += 1
        return run

    def _ask(self, run: Any) -> int:
        permitted = list(run.permitted)
        # The chooser sees the steps and the run's own figures. It does not see
        # the workflow, because neither does this process.
        try:
            index = int(self.choose(permitted, dict(getattr(run, "figures", {}) or {})))
        except (TypeError, ValueError) as exc:
            raise RefusedChoice(f"chooser did not return a step index: {exc}") from exc

        if not 0 <= index < len(permitted):
            raise RefusedChoice(
                f"chooser returned index {index}, outside the {len(permitted)} "
                "steps that were permitted. It may only pick from what was offered."
            )
        logger.info("agent chose %s", permitted[index]["op"])
        return index
