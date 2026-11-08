"""Golden test suite for effective independent trials and notebook hook (QOS-181).

Verifies:
1. Correlated parameter grid: effective independent trials N_eff < raw N,
   with method labeled 'eigenvalue_participation_ratio'.
2. Independent parameter grid: effective independent trials N_eff ≈ raw N.
3. Interactive notebook hook: ad-hoc backtests automatically log to the local
   project ledger so n_trials cannot silently be asserted as 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alphaengine.core.backtest import run_backtest, score_backtest
from alphaengine.core.validation import (
    deflated_sharpe,
    n_eff_participation,
    pbo_cscv,
)
from alphaengine.notebook import (
    disable_notebook_hook,
    enable_notebook_hook,
    get_notebook_trials,
)
from alphaengine.sweep.runner import SweepResult, Trial

pytestmark = pytest.mark.golden


def test_correlated_grid_effective_n_less_than_raw_n() -> None:
    """A highly correlated grid (rho >= 0.90) compresses effective degrees of freedom: N_eff << N."""
    rng = np.random.default_rng(42)
    t_obs = 240
    n_configs = 10

    # Generate a strong common factor + small idiosyncratic noise -> high pairwise correlation
    factor = rng.normal(0.0005, 0.01, size=(t_obs, 1))
    noise = rng.normal(0.0, 0.002, size=(t_obs, n_configs))
    matrix = np.tile(factor, (1, n_configs)) + noise

    # Raw N is 10, but eigenvalue participation ratio should be strictly < 3.0
    neff = n_eff_participation(matrix)
    assert 1.0 <= neff < 3.0, f"Expected N_eff < 3.0 for correlated grid, got {neff}"

    # DSR reports both raw n_trials and n_eff with method labeled
    dsr = deflated_sharpe(matrix[:, 0], n_trials=n_configs, matrix=matrix)
    assert dsr["n_trials"] == n_configs
    assert dsr["n_eff"] == pytest.approx(neff, abs=1e-3)
    assert dsr["n_eff_method"] == "eigenvalue_participation_ratio"
    # Deflation is less severe under N_eff than under raw N
    assert dsr["deflated_sharpe_at_n_eff"] >= dsr["deflated_sharpe"]

    # PBO CSCV reports both n_configs and n_eff with method labeled
    pbo = pbo_cscv(matrix, n_splits=8)
    assert pbo["n_configs"] == n_configs
    assert pbo["n_eff"] == pytest.approx(neff, abs=1e-3)
    assert pbo["n_eff_method"] == "eigenvalue_participation_ratio"

    # SweepResult.verdict reports n_trials, n_eff, and method
    trials = [
        Trial(
            index=i,
            params={"param": i},
            params_hash=f"hash_{i}",
            n_obs=t_obs,
            sharpe=float(np.mean(matrix[:, i]) / np.std(matrix[:, i], ddof=1)),
            sharpe_annualized=float(np.mean(matrix[:, i]) / np.std(matrix[:, i], ddof=1) * np.sqrt(252)),
            total_return_pct=float(np.sum(matrix[:, i]) * 100.0),
        )
        for i in range(n_configs)
    ]
    sweep_res = SweepResult(
        trials=trials,
        matrix=matrix,
        data_hash="data_hash_123",
    )
    v = sweep_res.verdict()
    assert v["n_trials"] == n_configs
    assert v["n_eff"] == pytest.approx(neff, abs=1e-3)
    assert v["n_eff_method"] == "eigenvalue_participation_ratio"
    assert v["deflated_sharpe_at_n_eff"] is not None


def test_independent_grid_effective_n_approx_raw_n() -> None:
    """An independent grid of orthogonal returns preserves all degrees of freedom: N_eff ≈ N."""
    rng = np.random.default_rng(123)
    t_obs = 5000
    n_configs = 10

    # Independent normal returns -> correlation matrix close to identity
    matrix = rng.normal(0.0002, 0.01, size=(t_obs, n_configs))
    neff = n_eff_participation(matrix)

    # N_eff should be very close to raw N = 10
    assert abs(neff - n_configs) < 0.6, f"Expected N_eff ≈ 10, got {neff}"

    pbo = pbo_cscv(matrix, n_splits=8)
    assert pbo["n_configs"] == n_configs
    assert pbo["n_eff"] == pytest.approx(neff, abs=1e-3)
    assert pbo["n_eff_method"] == "eigenvalue_participation_ratio"


def test_notebook_session_records_ad_hoc_backtest_trials(tmp_path: Path) -> None:
    """Notebook hook logs ad-hoc single runs so n_trials cannot silently be asserted as 1."""
    enable_notebook_hook(project_dir=tmp_path, family="notebook_test")
    try:
        # Construct synthetic price and signal data
        prices = {
            "AAPL": [
                {"date": f"2026-01-{i:02d}", "open": 100.0 + i, "close": 101.0 + i} for i in range(1, 25)
            ]
        }

        # Run 3 ad-hoc backtests
        res1 = run_backtest(
            {"AAPL": [{"date": "2026-01-02", "target_weight": 0.5}]},
            prices,
            fill_timing="close",
        )
        res2 = run_backtest(
            {"AAPL": [{"date": "2026-01-02", "target_weight": 0.8}]},
            prices,
            fill_timing="close",
        )
        res3 = run_backtest(
            {"AAPL": [{"date": "2026-01-02", "target_weight": 1.0}]},
            prices,
            fill_timing="close",
        )

        assert res1.get("notebook_attempt_id") is not None
        assert res2.get("notebook_attempt_id") is not None
        assert res3.get("notebook_attempt_id") is not None

        # Cumulative trials in session must be >= 3
        session_trials = get_notebook_trials()
        assert session_trials >= 3, f"Expected notebook trials >= 3, got {session_trials}"

        # Check ledger files exist and contain at least 3 attempts
        ledger_trials_file = tmp_path / ".alphaengine" / "ledger" / "trials.jsonl"
        assert ledger_trials_file.is_file()
        trial_lines = [line for line in ledger_trials_file.read_text().splitlines() if line.strip()]
        assert len(trial_lines) >= 3

        # Now score res3 while trying to assert n_trials = 1
        scored: dict[str, Any] = score_backtest(res3, n_trials=1)
        val = scored.get("validation", {})

        # The ledger floor MUST override the asserted n_trials=1
        assert val.get("n_trials", 0) >= 3
        assert val.get("n_trials_source") == "derived_from_ledger"
    finally:
        disable_notebook_hook()
