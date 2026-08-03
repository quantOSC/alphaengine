"""A built-in project module: a candidate and a book, for `check_overlap`.

    alphaengine run check_overlap --project alphaengine.demo_book

Fixed seeds, offline, reproducible. The candidate is built as HALF the book
plus its own noise, so the overlap check has something honest to find: a
correlation that is real, visible, and short of "the book again" — which is
what most genuinely-related ideas look like.
"""

from __future__ import annotations

import random

_N_OBS = 300


def _series() -> dict[str, list[float]]:
    rng = random.Random(42)
    book = [rng.gauss(0.0003, 0.009) for _ in range(_N_OBS)]
    candidate = [0.5 * b + rng.gauss(0.0002, 0.007) for b in book]
    return {
        "returns": [round(v, 8) for v in candidate],
        "book_returns": [round(v, 8) for v in book],
    }


#: What `--project alphaengine.demo_book` hands to the harness.
data = _series()

__all__ = ["data"]
