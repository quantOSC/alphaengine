"""A project module you can actually run, shipped INSIDE the package.

    alphaengine demo                                       # just run it
    alphaengine run validate_study --project alphaengine.demo

IT LIVES IN THE PACKAGE ON PURPOSE. It started life in `examples/`, which meant
it reached nobody: the wheel does not carry the repo, so `pip install
alphaengine` gave you a CLI with nothing to point `--project` at, and the
README's first instruction was `git clone`. Asking somebody to clone a repo to
try a pip-installable tool is a first run most people do not complete.

Two dependencies and no data, so shipping it costs nothing and every install can
demonstrate itself offline.

A PROJECT MODULE IS NOT A CONFIG FILE. It is an ordinary Python module of yours
that exposes two names:

    data         whatever your backtest function takes — a DataFrame, a dict of
                 series, an array. The harness never inspects it and it never
                 leaves this machine.
    backtest_fn  your simulator, called once per parameter combination as
                 `backtest_fn(data=data, **params)`.

That is the whole contract. It is deliberately the shape of the notebook cell
you were going to write anyway, because the alternative — a YAML schema
describing your strategy — means reimplementing your research in our vocabulary
before you can find out whether it survives.

TWO THINGS `backtest_fn` MUST DO, both easy to get wrong the first time:

  1. RETURN A BARE 1-D RETURN SERIES. Not a dict, not a stats object — the
     per-period returns themselves. `sweep` does `np.asarray(list(raw))`, so a
     dict of results iterates its KEYS and fails on the first string.
  2. RETURN THE SAME LENGTH FOR EVERY COMBINATION. PBO splits the trial matrix
     into time blocks and compares configurations within each block, which is
     only meaningful if they line up in time. Ragged output is refused rather
     than truncated, because silently trimming would produce a confident number
     over series that do not correspond. That is why `WARMUP` below is a
     constant covering the slowest window in the grid, instead of each
     configuration starting wherever its own window happens to fill.

This one is a moving-average crossover on synthetic prices. The prices come from
a fixed seed rather than a download, so the example runs offline, gives the same
figures on every machine, and needs no data licence. It is a DEMONSTRATION OF
THE WIRING, not a strategy: a crossover on a random walk has no edge, and the
verdict at the end says so. That is the example working, not failing.
"""

from __future__ import annotations

import random

# Long enough to cover the slowest window in the documented grid (slow=200).
# Every configuration starts here, so every return series is the same length.
WARMUP = 200


def _prices(n: int = 1200, seed: int = 7) -> list[float]:
    """A synthetic close series. Fixed seed: the example must reproduce."""
    rng = random.Random(seed)
    px, out = 100.0, []
    for _ in range(n):
        px *= 1.0 + rng.gauss(0.0004, 0.011)
        out.append(round(px, 4))
    return out


# What `--project alphaengine.demo` hands to the harness.
data = {"close": _prices()}

# The grid the README and the CLI examples use. Nine combinations, so a
# derived trial count of exactly 9 — a number you can check by hand.
GRID = {"fast": [5, 10, 20], "slow": [50, 100, 200]}


def backtest_fn(*, data: dict[str, list[float]], fast: int = 10, slow: int = 50) -> list[float]:
    """One moving-average crossover, long/flat, as a daily return series.

    `fast >= slow` is a degenerate corner of the grid. It returns a flat series
    of the right LENGTH rather than an empty one: a combination that fails is
    still a combination that was tried, and dropping it would quietly shrink the
    denominator every deflated figure downstream divides by.

    NOTE the annotation on `data`: it is `dict[str, list[float]]` only because
    THIS demo happens to hand over a dict of lists. `sweep` never inspects it —
    yours can be a DataFrame, an array, or an object of your own, and the
    harness passes it through untouched.
    """
    close: list[float] = data["close"]
    n = len(close)

    def sma(k: int, i: int) -> float:
        return float(sum(close[i - k + 1 : i + 1]) / k)

    if fast >= slow or slow > WARMUP:
        return [0.0] * (n - WARMUP - 1)

    out: list[float] = []
    position = 0
    for i in range(WARMUP, n - 1):
        position = 1 if sma(fast, i) > sma(slow, i) else 0
        step = (close[i + 1] - close[i]) / close[i]
        out.append(round(step * position, 8))
    return out


def run() -> int:
    """The offline half, end to end. No server, no account, no network.

    Printed rather than returned because the point is to SEE it — this is the
    first thing a new install runs, and `alphaengine demo` should answer "does
    any of this work" in one command.
    """
    from alphaengine import sweep

    result = sweep(backtest_fn, GRID, data=data)
    verdict = result.verdict()
    surface = result.surface()

    dsr = verdict.get("deflated_sharpe")
    print(f"trials     {result.n_trials}  ({verdict.get('n_trials_source')})")
    print(f"verdict    {verdict.get('verdict')}")
    print(f"surface    {surface.get('shape')}")
    print(f"dsr        {dsr if dsr is not None else 'not recorded'}")
    print()
    print("A crossover on a random walk has no edge, and the verdict says so")
    print("rather than flattering it. That is this working, not failing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
