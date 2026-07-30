"""Frozen values. These are the public contract.

WHY THIS FILE EXISTS
    Once this package is on PyPI, a study written today has to reproduce in two
    years or the whole reproducibility claim collapses. So a changed return
    value is a BREAKING CHANGE requiring a major version bump, even when the
    signature is untouched and the new number is arguably better.

    These values were carried over from the platform's parity guards, which
    asserted that two copies of the math agreed with each other. There is only
    one copy now, so agreement is no longer the thing to test, what matters is
    that today's answer is still tomorrow's answer. Same protection, absolute
    instead of relative.

IF A GOLDEN MOVES
    Do not update the number to make the test pass. Either the change is a bug,
    or it is a deliberate correction that needs a major bump and a changelog
    entry explaining what will stop reproducing and why the new answer is right.

Comparisons are exact: these functions round their own outputs, so there is no
float noise to tolerate, and a tolerance would quietly absorb a real drift.
"""

import numpy as np
import pytest

from alphaengine.core import (
    compute_var_cvar,
    deflated_sharpe,
    pbo_cscv,
    performance_report,
)

pytestmark = pytest.mark.golden


def rets(seed: int, n: int) -> list[float]:
    """A fixed return stream. Seeded, so it is the same series on every machine."""
    return np.random.default_rng(seed).normal(0.0006, 0.011, n).tolist()


# ── deflated Sharpe ────────────────────────────────────────────────────────
# The statistic the verdict gates on. Note how hard the trial count bites:
# the same series is "marginal" at one trial and "likely_noise" at forty. That
# is the whole argument for deriving the count rather than asking for it.
DSR_GOLDEN = {
    1:   {"deflated_sharpe": 0.7566, "sr0_expected_max": 0.0371, "verdict": "marginal"},
    5:   {"deflated_sharpe": 0.5090, "sr0_expected_max": 0.0851, "verdict": "marginal"},
    40:  {"deflated_sharpe": 0.1650, "sr0_expected_max": 0.1561, "verdict": "likely_noise"},
    250: {"deflated_sharpe": 0.0524, "sr0_expected_max": 0.2024, "verdict": "likely_noise"},
}


@pytest.mark.parametrize("n_trials", sorted(DSR_GOLDEN))
def test_deflated_sharpe_golden(n_trials: int) -> None:
    got = deflated_sharpe(rets(9, 200), n_trials=n_trials)
    for key, want in DSR_GOLDEN[n_trials].items():
        assert got[key] == want, f"DSR drift on {key} at n_trials={n_trials}"


def test_deflated_sharpe_is_monotone_in_trials() -> None:
    """More trials tried can never make a result look better.

    A property rather than a value, because it is the invariant a user is
    trusting when they report a deflated figure. If this inverts, the number is
    worse than useless, it rewards searching harder.
    """
    r = rets(9, 200)
    seq = [deflated_sharpe(r, n_trials=n)["deflated_sharpe"] for n in (1, 5, 40, 250)]
    assert seq == sorted(seq, reverse=True), f"DSR not monotone decreasing in n_trials: {seq}"


def test_psr_is_independent_of_trials() -> None:
    """PSR asks a different question and must not move with the trial count."""
    r = rets(9, 200)
    vals = {deflated_sharpe(r, n_trials=n)["psr_vs_zero"] for n in (1, 5, 40, 250)}
    assert vals == {0.8879}


# ── probability of backtest overfitting ────────────────────────────────────
def test_pbo_cscv_golden() -> None:
    m = np.random.default_rng(3).normal(0.0002, 0.01, size=(240, 6))
    got = pbo_cscv(m.tolist(), n_splits=8)
    assert got["pbo"] == 0.7
    assert got["n_partitions"] == 70
    assert got["n_configs"] == 6
    assert got["n_splits"] == 8
    assert got["logit_mean"] == -0.3963
    assert got["verdict"] == "overfit"


# ── performance ────────────────────────────────────────────────────────────
def test_performance_report_golden() -> None:
    got = performance_report(rets(11, 500))
    want = {
        "n_obs": 500,
        "sharpe_ratio": 0.0884,
        "sharpe_annualized": 1.4033,
        "sortino_ratio": 2.1031,
        "calmar_ratio": 2.7428,
        "max_drawdown_pct": 9.12,
        "total_return_pct": 55.75,
        "annualized_return_pct": 25.02,
        "volatility_annualized_pct": 16.94,
    }
    for key, value in want.items():
        assert got[key] == value, f"performance drift on {key}"


def test_annualized_return_is_geometric() -> None:
    """Compounded, not mean*252.

    Pinned because it has drifted before: an arithmetic annualisation does not
    reconcile with Calmar or with the gateway, and the disagreement only shows
    up as two slightly different numbers on two screens.
    """
    r = rets(11, 500)
    got = performance_report(r)["annualized_return_pct"]
    compounded = (float(np.prod(1 + np.asarray(r))) ** (252 / len(r)) - 1) * 100
    assert abs(got - compounded) < 0.02


# ── value at risk ──────────────────────────────────────────────────────────
def test_var_cvar_golden_and_sign_convention() -> None:
    got = compute_var_cvar(rets(13, 400))
    assert got["n_obs"] == 400
    assert got["confidence"] == 0.95
    assert got["parametric"]["var_pct"] == 1.91
    assert got["parametric"]["daily_vol_pct"] == 1.16

    # Losses are reported as POSITIVE magnitudes everywhere, and CVaR is at
    # least VaR by construction. Both have been broken before, in ways that
    # reached a dashboard rather than a test.
    assert got["parametric"]["var_pct"] > 0
    hist = got.get("historical") or {}
    if hist.get("var_pct") is not None and hist.get("cvar_pct") is not None:
        assert hist["var_pct"] > 0
        assert hist["cvar_pct"] >= hist["var_pct"], "CVaR must be at least VaR"
