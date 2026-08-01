"""Client for a workflow server: execute steps locally, report figures.

    from alphaengine.client import connect

    session = connect("https://...", api_key="...")
    run = session.open("validate_study", data=prices, backtest_fn=mine, grid={...})
    run.drive()

THE SPLIT
    The server holds the workflow: which steps, in what order, under what
    conditions, and what stops a run. This side holds your data and executes
    what it is asked to, one operation at a time.

    Neither half moves. Your prices stay on your machine and the sequencing
    stays on the server, which is what makes the arrangement stable rather than
    a standoff between two parties who each want the other's part.

EVERYTHING ELSE IN THIS PACKAGE WORKS WITHOUT ANY OF THIS
    `sweep`, the core maths and the study artifact need no server, no account
    and no network. This subpackage is additive. If it is not for you, importing
    the top level never touches it.
"""

from __future__ import annotations

from .executor import MAX_FIGURE_LIST, StepExecutor, UnsupportedOp
from .session import Offline, Run, ServerError, Session, connect, connect_or_offline

__all__ = [
    "connect",
    "connect_or_offline",
    "Session",
    "Run",
    "StepExecutor",
    "UnsupportedOp",
    # The line between a figure and a series, exported so a custom handler can
    # check its own output against the same number the guard enforces.
    "MAX_FIGURE_LIST",
    "Offline",
    "ServerError",
]
