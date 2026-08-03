"""A built-in project module: a signal panel and its prices, for `evaluate_signal`.

    alphaengine run evaluate_signal --project alphaengine.demo_signal

The same discipline as every demo module: a fixed seed, offline, reproducible,
no data licence. And the same honesty: the planted link between signal and
forward return is WEAK on purpose — the run should close with a small IC and
its caveats, because a demonstration whose example looks like a miracle is
teaching the wrong lesson about what a real signal reads like.

`--project` reads `data` off a module; here that is the `{'signal': ...,
'prices': ...}` pair the workflow scores.
"""

from __future__ import annotations

import random

_N_NAMES = 40
_N_OBS = 320


def _panels() -> dict[str, dict[str, list[float]]]:
    signal: dict[str, list[float]] = {}
    prices: dict[str, list[float]] = {}
    for i in range(_N_NAMES):
        rng = random.Random(500 + i)
        s = (i - _N_NAMES / 2) / _N_NAMES
        # A drift the signal partially knows. Weak by design: real ICs are
        # small, and the demo should read like the real thing.
        drift = 0.0003 + s * 0.0012
        px, series = 100.0, []
        for _ in range(_N_OBS):
            px *= 1.0 + rng.gauss(drift, 0.011)
            series.append(round(px, 4))
        signal[f"SYM{i:02d}"] = [s] * _N_OBS
        prices[f"SYM{i:02d}"] = series
    return {"signal": signal, "prices": prices}


#: What `--project alphaengine.demo_signal` hands to the harness.
data = _panels()

__all__ = ["data"]
