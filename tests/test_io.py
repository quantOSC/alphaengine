"""Tests for QuantConnect and VectorBT interop loaders (`alphaengine.io`)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import alphaengine
from alphaengine.io import from_quantconnect, from_vectorbt
from alphaengine.study import load, save


def _qc_fixture(*, total_backtests: int | None = None) -> dict:
    rng = np.random.default_rng(19)
    eq = (100_000.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.01, 45))).tolist()
    values = [{"x": 1_700_000_000 + i * 86400, "y": round(v, 2)} for i, v in enumerate(eq)]
    payload: dict = {
        "projectName": "QC Momentum Sleeve",
        "charts": {
            "Strategy Equity": {
                "series": {
                    "Equity": {"values": values},
                }
            }
        },
    }
    if total_backtests is not None:
        payload["totalBacktests"] = total_backtests
    return payload


def test_from_quantconnect_without_trial_count_marks_not_recorded_and_refuses_verdict(
    tmp_path: Path,
) -> None:
    path = tmp_path / "qc_single.json"
    path.write_text(json.dumps(_qc_fixture()), encoding="utf-8")

    study = alphaengine.io.from_quantconnect(path)
    assert study.label == "QC Momentum Sleeve"
    assert len(study.returns) == 44
    assert study.n_trials == 0
    assert study.n_trials_source == "not_recorded"
    assert study.verdict is None
    assert study.deflated_sharpe is None
    assert "not recorded" in study.notes
    assert len(study.data_hash) == 64
    assert study.content_sha256 is not None and study.verify()

    # Saved study on disk does not leak raw returns
    saved_path = save(study, tmp_path / "study.json")
    raw_saved = json.loads(saved_path.read_text(encoding="utf-8"))
    assert "returns" not in raw_saved
    reloaded = load(saved_path)
    assert reloaded.n_trials_source == "not_recorded"
    assert reloaded.verdict is None


def test_from_quantconnect_with_trial_count_marks_asserted_and_computes_verdict(
    tmp_path: Path,
) -> None:
    path = tmp_path / "qc_opt.json"
    path.write_text(json.dumps(_qc_fixture(total_backtests=12)), encoding="utf-8")

    study = from_quantconnect(path)
    assert len(study.returns) == 44
    assert study.n_trials == 12
    assert study.n_trials_source == "asserted"
    assert study.deflated_sharpe is not None
    assert study.verdict in {"likely_noise", "marginal", "robust"}


def test_from_vectorbt_csv_and_portfolio_object(tmp_path: Path) -> None:
    rng = np.random.default_rng(23)
    rets = rng.normal(0.0006, 0.009, 50)

    # Single-series CSV -> trial count not recorded, verdict refused
    csv_path = tmp_path / "vbt_returns.csv"
    lines = ["date,returns"] + [f"2025-01-{i + 1:02d},{r:.6f}" for i, r in enumerate(rets[:28])]
    csv_path.write_text("\n".join(lines), encoding="utf-8")

    s_csv = from_vectorbt(csv_path)
    assert len(s_csv.returns) == 28
    assert s_csv.n_trials == 0
    assert s_csv.n_trials_source == "not_recorded"
    assert s_csv.verdict is None
    assert s_csv.deflated_sharpe is None

    # Multi-column portfolio object -> trial count asserted from column count
    class _FakePortfolio:
        def returns(self):
            return np.column_stack([rets, rets * 0.8, rets * 1.1])

    s_pf = from_vectorbt(_FakePortfolio(), label="VBT Grid")
    assert s_pf.label == "VBT Grid"
    assert len(s_pf.returns) == 50
    assert s_pf.n_trials == 3
    assert s_pf.n_trials_source == "asserted"
    assert s_pf.deflated_sharpe is not None
    assert s_pf.verdict in {"likely_noise", "marginal", "robust"}


def test_from_vectorbt_multi_column_csv_and_short_input_error(tmp_path: Path) -> None:
    csv_multi = tmp_path / "vbt_grid.csv"
    rows = ["date,cfg_a,cfg_b"] + [
        f"2025-01-{i + 1:02d},{0.001 * ((i % 5) - 2):.5f},{0.0015 * ((i % 4) - 1):.5f}" for i in range(25)
    ]
    csv_multi.write_text("\n".join(rows), encoding="utf-8")
    study = from_vectorbt(csv_multi)
    assert study.n_trials == 2
    assert study.n_trials_source == "asserted"

    with pytest.raises(ValueError, match="could not extract a return series"):
        from_quantconnect({"returns": [0.01]})
