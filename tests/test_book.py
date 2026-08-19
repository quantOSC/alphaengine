"""Book overlay and walk-forward: figures, derived trial counts, no OMS."""

from __future__ import annotations

import pytest

from alphaengine.book import MAX_SLEEVES, Book
from alphaengine.core.walkforward import walk_forward
from alphaengine.demo import GRID, backtest_fn, data


def test_book_overlap_is_bounded_and_named():
    book = Book()
    book.add("a", [0.01, -0.02, 0.00, 0.03] * 40)
    book.add("b", [0.00, 0.01, -0.01, 0.02] * 40)
    out = book.overlap_matrix()
    assert out["n_sleeves"] == 2
    assert len(out["pairs"]) == 1
    assert {out["pairs"][0]["a"], out["pairs"][0]["b"]} == {"a", "b"}


def test_book_refuses_a_seventeenth_sleeve():
    book = Book()
    series = [0.001] * 50
    for i in range(MAX_SLEEVES):
        book.add(f"s{i}", series)
    with pytest.raises(ValueError, match="at most"):
        book.add("extra", series)


def test_monitor_never_reads_empty_as_all_clear():
    assert Book().monitor()["overall"] == "unchecked"


def test_walk_forward_derives_n_trials_from_windows_times_grid():
    out = walk_forward(backtest_fn, GRID, data=data, n_windows=3, oos_fraction=0.3)
    assert out["n_trials_source"] == "derived_from_walk_forward"
    assert out["n_trials"] == 3 * 9
    assert out["n_windows"] == 3
    assert out["windows"]
