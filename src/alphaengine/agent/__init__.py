"""Drive a workflow run with a model you supply, then answer from what it found.

    from alphaengine.agent import AgentDriver, synthesize

    run = AgentDriver(choose=my_model).drive(run)
    answer = synthesize("which names held up?", run, write=my_model)

TWO CALLS, TWO CONTRACTS, AND THE SPLIT IS THE DESIGN.

`drive` bounds the agent to the permitted set the server sent this turn, so it
cannot call an unoffered operation, skip a gate, proceed past a stop, or act
without a record. `choose` returns an INDEX and nothing else — see `driver.py`
for why each guarantee holds by construction.

That narrowness is why the agent could not answer: no call had saying anything as
its job. `synthesize` is that call, and it is separate precisely so the interface
whose narrowness is the guarantee stays narrow. It may write freely and may cite
only numbers the run recorded — checked after the fact, refused rather than
repaired, exactly as an out-of-range index is refused rather than clamped.

No model dependency, no key handling, no inference call. `choose` and `write` are
both yours.
"""

from __future__ import annotations

from .answer import Answer, UncitedFigure, Writer, caveats_of, figures_of, synthesize
from .driver import AgentDriver, Choice, RefusedChoice

__all__ = [
    "AgentDriver",
    "Choice",
    "RefusedChoice",
    "synthesize",
    "Answer",
    "UncitedFigure",
    "Writer",
    "figures_of",
    "caveats_of",
]
