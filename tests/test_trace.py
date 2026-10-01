"""The recorder is a side channel. The published numbers do not move."""

from __future__ import annotations

import numpy as np

from alphaengine.core import deflated_sharpe, performance_report
from alphaengine.core.trace import collecting


def rets(seed: int, n: int) -> list[float]:
    return np.random.default_rng(seed).normal(0.0006, 0.011, n).tolist()


def test_a_bound_collector_does_not_change_a_golden() -> None:
    returns = rets(11, 500)
    off = performance_report(returns)
    with collecting() as collector:
        on = performance_report(returns)
    assert on == off
    sharpe = next(line for line in collector.lines if line.id == "sharpe_annualized")
    assert sharpe.result == off["sharpe_annualized"]
    assert len(sharpe.series["returns"]) == 500
    assert "sqrt" in sharpe.formula

    raw = rets(9, 200)
    plain = deflated_sharpe(raw, n_trials=5)
    with collecting() as collector:
        traced = deflated_sharpe(raw, n_trials=5)
    assert traced == plain
    assert any(
        line.id == "deflated_sharpe" and line.result == plain["deflated_sharpe"] for line in collector.lines
    )


def test_an_unbound_call_records_nothing() -> None:
    with collecting() as outer:
        performance_report(rets(11, 500))
    assert outer.lines
    # Leaving the block unbinds. A later call must not append to it.
    performance_report(rets(11, 500))
    assert all(line.id != "sentinel" for line in outer.lines)
    n = len(outer.lines)
    performance_report(rets(11, 500))
    assert len(outer.lines) == n
