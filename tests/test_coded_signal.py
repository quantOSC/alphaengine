"""Tests for AlphaEngine 2.0 coded-backtest signal adapter (`QOS-154`, `QOS-121`, `QOS-123`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from alphaengine import (
    MAX_CODED_GRID,
    CodedSignalError,
    check_signal_module,
    check_signal_source,
    cli,
    run_signal,
)
from alphaengine.ledger import cumulative_trials_for
from alphaengine.project import find_project, init_project


def _make_panel(n: int = 40) -> dict[str, list[float]]:
    return {
        "AAPL": [100.0 + 0.6 * i + (0.4 if i % 3 == 0 else -0.1) for i in range(n)],
        "MSFT": [200.0 - 0.3 * i + (0.2 if i % 2 == 0 else -0.2) for i in range(n)],
    }


def test_static_checker_blocks_forbidden_imports_builtins_and_records_zero_trials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    proj = init_project(tmp_path, name="coded-static-test")

    bad_sources = [
        ("import os\ndef signal(*, data_asof, params):\n    return {'AAPL': 0.5}\n", "forbidden_import"),
        ("import httpx\ndef signal(*, data_asof, params):\n    return {'AAPL': 0.5}\n", "forbidden_import"),
        (
            "from alphaengine.client import Session\ndef signal(*, data_asof, params):\n    return {}\n",
            "forbidden_import",
        ),
        (
            "def signal(*, data_asof, params):\n    open('x.txt')\n    return {'AAPL': 0.5}\n",
            "forbidden_builtin",
        ),
        ("async def signal(*, data_asof, params):\n    return {'AAPL': 0.5}\n", "async_signal"),
        ("def wrong_name(*, data_asof, params):\n    return {'AAPL': 0.5}\n", "missing_signal_fn"),
        ("def signal(only_pos):\n    return {'AAPL': 0.5}\n", "invalid_signature"),
        ("def signal(*, data_asof, params)\n    return {}\n", "syntax_error"),
    ]
    for idx, (src, expected_code) in enumerate(bad_sources):
        res = check_signal_source(src)
        assert not res.ok
        assert any(d.code == expected_code for d in res.diagnostics), res.diagnostics
        assert all(d.line is not None and d.remediation for d in res.diagnostics)

        sig_file = tmp_path / f"bad_sig_{idx}.py"
        sig_file.write_text(src, encoding="utf-8")
        with pytest.raises(CodedSignalError):
            run_signal(sig_file, prices=_make_panel(20), idea_id="idea_bad")

    # Static check failures MUST record 0 trials!
    assert cumulative_trials_for(proj) == (0, 0)
    assert proj.read_jsonl("ledger/trials.jsonl") == []


def test_causal_data_asof_prevents_lookahead_by_construction(tmp_path: Path) -> None:
    sig_code = """
import numpy as np

def signal(*, data_asof, params):
    lookback = int(params.get("lookback", 5))
    prices = data_asof["prices"]
    out = {}
    for tk, series in prices.items():
        arr = np.asarray(series, dtype=float)
        if arr.size < lookback + 1:
            out[tk] = 0.0
        else:
            ret = arr[-1] / arr[-lookback - 1] - 1.0
            out[tk] = 0.4 if ret > 0 else -0.4
    return out
"""
    sig_file = tmp_path / "mom_sig.py"
    sig_file.write_text(sig_code, encoding="utf-8")
    assert check_signal_module(sig_file).ok

    panel_a = _make_panel(30)
    panel_b = {k: list(v) for k, v in panel_a.items()}
    # Mutate ONLY the final bar (`t=29`) radically in panel_b
    panel_b["AAPL"][-1] = 99999.0
    panel_b["MSFT"][-1] = 1.0

    # Record lengths seen by `signal()` at each bar `t`
    seen_lengths: list[int] = []

    def spy_signal(*, data_asof: dict[str, Any], params: dict[str, Any]) -> dict[str, float]:
        _ = params
        seen_lengths.append(len(data_asof["prices"]["AAPL"]))
        arr = data_asof["prices"]["AAPL"]
        return {"AAPL": 0.5 if arr[-1] >= arr[0] else -0.5, "MSFT": 0.0}

    run_signal(spy_signal, prices=panel_a, record_in_project=False)
    assert seen_lengths == list(range(1, 31))

    res_mod_a = run_signal(sig_file, prices=panel_a, params={"lookback": 4}, record_in_project=False)
    res_mod_b = run_signal(sig_file, prices=panel_b, params={"lookback": 4}, record_in_project=False)
    # Equity curve through bar 28 (`[:-1]`) MUST be identical despite future spike at bar 29!
    assert res_mod_a["backtest"]["equity_curve"][:-1] == res_mod_b["backtest"]["equity_curve"][:-1]


def test_mandatory_costs_grid_cap_and_runtime_failure_trial_counting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    proj = init_project(tmp_path, name="coded-trials-test")
    panel = _make_panel(35)

    good_sig = tmp_path / "good_sig.py"
    good_sig.write_text(
        "def signal(*, data_asof, params):\n"
        "    w = float(params.get('w', 0.3))\n"
        "    if w > 0.9:\n"
        "        raise ValueError('bad param inside user signal')\n"
        "    return {'AAPL': w, 'MSFT': -w}\n",
        encoding="utf-8",
    )

    # 1. Zero costs refused unless `allow_zero_costs_unverified=True` (`0` trials on refusal)
    with pytest.raises(CodedSignalError, match="positive transaction costs"):
        run_signal(good_sig, prices=panel, slippage_bps=0.0, idea_id="idea_1")
    assert cumulative_trials_for(proj) == (0, 0)

    unv = run_signal(
        good_sig,
        prices=panel,
        slippage_bps=0.0,
        commission_bps=0.0,
        allow_zero_costs_unverified=True,
    )
    assert unv["verdict"] == "unverified"
    assert unv["engine_verified"] is False
    assert unv["can_seal"] is False
    assert cumulative_trials_for(proj) == (0, 1)

    # 2. Grid > 400 refused pre-execution (`0` additional trials)
    with pytest.raises(CodedSignalError, match=str(MAX_CODED_GRID)):
        run_signal(good_sig, prices=panel, grid={"w": [0.001 * i for i in range(MAX_CODED_GRID + 1)]})
    assert cumulative_trials_for(proj) == (0, 1)

    # 3. Sweep of 3 combos where 1 fails at runtime -> all 3 count in `coded:idea_1`!
    out = run_signal(good_sig, prices=panel, grid={"w": [0.2, 0.4, 0.95]}, idea_id="idea_1")
    assert out["n_trials"] == 4  # max(family_trials=3, project_trials=4)
    assert out["figures"]["n_ok_trials"] == 2
    assert out["figures"]["n_failed_trials"] == 1
    assert out["costs_model"]["slippage_bps"] == 5.0
    assert out["costs_model"]["commission_bps"] == 1.0
    assert cumulative_trials_for(proj, idea_id="idea_1") == (3, 4)

    # 4. Single runtime crash inside `signal()` still counts as 1 trial (`QOS-154`)!
    with pytest.raises(CodedSignalError, match="failed"):
        run_signal(good_sig, prices=panel, params={"w": 0.99}, idea_id="idea_1")
    assert cumulative_trials_for(proj, idea_id="idea_1") == (4, 5)


def test_cli_run_with_signal_and_rerun_reproducibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QUANTOS_API_KEY", "ae_live_coded_cli_test")

    assert cli.main(["init", "--name", "coded-cli-proj"]) == 0
    capsys.readouterr()

    csv_path = tmp_path / "prices.csv"
    lines = ["date,AAPL,MSFT"] + [
        f"2025-{(i // 28) + 1:02d}-{(i % 28) + 1:02d},{100.0 + 0.5 * i:.2f},{200.0 - 0.2 * i:.2f}"
        for i in range(40)
    ]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    sig_path = tmp_path / "my_signal.py"
    sig_path.write_text(
        "def signal(*, data_asof, params):\n"
        "    lb = int(params.get('lb', 3))\n"
        "    p = data_asof['prices']['AAPL']\n"
        "    w = 0.4 if len(p) > lb and p[-1] > p[-lb] else -0.2\n"
        "    return {'AAPL': w, 'MSFT': -0.5 * w}\n",
        encoding="utf-8",
    )

    assert (
        cli.main(
            [
                "validate",
                "--signal",
                str(sig_path),
                "--data",
                str(csv_path),
                "--input",
                'grid={"lb": [2, 4]}',
                "--json",
            ]
        )
        == 0
    )
    env = json.loads(capsys.readouterr().out)
    assert env["command"] == "coded_signal"
    assert env["n_trials"] == 2
    assert env["manifest"]["costs_model"]["slippage_bps"] == 5.0
    assert env["manifest"]["costs_model"]["commission_bps"] == 1.0

    # Rerun reproduces figures and verifies provenance!
    assert cli.main(["rerun", "HEAD", "--json"]) == 0
    rerun_out = json.loads(capsys.readouterr().out)
    assert rerun_out["hash_verified"] is True
    assert rerun_out["figures_reproduced"] is True
    assert rerun_out["provenance_verified"] is True

    proj = find_project(tmp_path)
    assert proj is not None
    assert cumulative_trials_for(proj) == (0, 2)
