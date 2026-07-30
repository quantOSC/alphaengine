"""AlphaEngine, validated research tooling for investment strategies.

Answers one question honestly: **is this result an edge, or did you try enough
things that something was bound to look good?**

    from alphaengine import sweep

    r = sweep(backtest_fn, grid, data=prices)   # runs every combination
    r.surface()                                 # stable plateau, or a knife edge
    r.verdict()                                 # deflated for the trials you ran
    r.save("study.json")                        # the artifact, on your disk

WHY A SWEEP AND NOT A CALCULATOR
    The correction that makes a Sharpe ratio honest needs to know how many
    variants you tested. Ask a person for that number and you get the number
    that flatters them, not from dishonesty, but because nobody counts the
    thing they threw away. Run the grid and the count is `len(grid)`, so the
    question never has to be asked.

WHAT THIS LIBRARY IS NOT
    Not a backtester. `sweep()` takes YOUR function. We orchestrate and measure;
    you simulate. The engine you already trust stays the engine you trust.

OFFLINE BY CONSTRUCTION
    Importing this module makes no network call and needs no account. numpy and
    scipy, nothing else. Everything above runs on a laptop with the wifi off,
    and your data never leaves the machine.
"""

from ._version import __version__
from .study import SCHEMA_VERSION, Study, load, save
from .sweep import SweepResult, sweep

# `sweep` is bound here to the FUNCTION, deliberately shadowing the subpackage
# of the same name. The documented entry point is `from alphaengine import
# sweep`, and a user who writes that and gets a module back has hit a bug on
# their first line. `from alphaengine.sweep import ...` still resolves through
# sys.modules for anyone who wants the module explicitly.
__all__ = [
    "__version__",
    "sweep",
    "SweepResult",
    "Study",
    "save",
    "load",
    "SCHEMA_VERSION",
]
