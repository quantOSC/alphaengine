"""Drive a workflow run with a model you supply.

    from alphaengine.agent import AgentDriver

The agent's action space is the permitted set the server sent this turn, so it
cannot call an unoffered operation, skip a gate, proceed past a stop, or act
without a record. See `driver.py` for why each of those holds by construction.

No model dependency, no key handling, no inference call. `choose` is yours.
"""

from __future__ import annotations

from .driver import AgentDriver, Choice, RefusedChoice

__all__ = ["AgentDriver", "Choice", "RefusedChoice"]
