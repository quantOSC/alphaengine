"""Run the parameter grid, so nobody has to be asked how many things they tried.

The correction that makes a Sharpe ratio honest needs one input: the number of
variants tested. Ask a person for it and you get the number that flatters
them. Not through dishonesty: nobody counts what they threw away, and the
count is genuinely hard to reconstruct after the fact.

Run the grid and the count is `len(grid)`. The question never gets asked.

The deflation is the by-product. What you get back is the NEIGHBOURHOOD: whether
your result sits on a broad plateau or a knife edge, and where the plateau's
centre is. That is the output worth having, because it makes the strategy better
rather than just making the number smaller.
"""

from .runner import SweepResult, sweep

__all__ = ["sweep", "SweepResult"]
