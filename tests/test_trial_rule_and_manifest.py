"""Tests for William's 2.0 trial counting rule (`QOS-153`) and reproducibility manifests (`QOS-158`)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from alphaengine import LedgerError, RunManifest, build_run_manifest, cli, load, save, sweep
from alphaengine.ledger import (
    cumulative_trials_for,
    record_run,
    record_trials,
    rerun_record_report,
    resolve_run,
)
from alphaengine.memo import seal_from_project, verify_study_file
from alphaengine.project import init_project


def test_williams_trial_rule_every_attempt_counts_including_identical_reruns_and_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    proj = init_project(tmp_path, name="trial-rule-test")

    rng = np.random.default_rng(123)
    base_rets = rng.normal(0.001, 0.01, size=80)

    def bt_fn(data: list[float], lb: int = 5) -> np.ndarray:
        _ = data
        if lb == 99:
            raise RuntimeError("boom on invalid lookback")
        return base_rets * (1.0 + 0.01 * lb)

    # Run 1: 3 grid combinations (2 succeed, 1 fails) -> all 3 count!
    res1 = sweep(bt_fn, {"lb": [5, 10, 99]}, data=[1.0, 2.0, 3.0])
    assert res1.n_trials == 3
    assert len([t for t in res1.trials if t.failed is not None]) == 1
    assert all(len(t.attempt_id) >= 32 for t in res1.trials)
    assert cumulative_trials_for(proj, family="sweep:lb") == (3, 3)

    # Run 2: EXACT SAME grid and data repeated -> counts again (`3 -> 6`)!
    res2 = sweep(bt_fn, {"lb": [5, 10, 99]}, data=[1.0, 2.0, 3.0])
    assert res2.n_trials == 3
    assert cumulative_trials_for(proj, family="sweep:lb") == (6, 6)

    # Run 3: EXACT SAME grid repeated a third time -> counts again (`6 -> 9`)!
    sweep(bt_fn, {"lb": [5, 10, 99]}, data=[1.0, 2.0, 3.0])
    assert cumulative_trials_for(proj, family="sweep:lb") == (9, 9)

    # Replaying an already-recorded `attempt_id` (e.g. idempotent retry) does NOT double-count
    existing_ledger = res2.ledger()
    added = record_trials(proj, run_id="run_replay", trials=existing_ledger, family="sweep:lb")
    assert len(added) == 0
    assert cumulative_trials_for(proj, family="sweep:lb") == (9, 9)

    # All-failed sweep still records every attempted combination in the ledger before raising!
    with pytest.raises(RuntimeError, match="every one of the 2 combinations failed"):
        sweep(
            lambda data, lb=1: (_ for _ in ()).throw(RuntimeError("fail")),
            {"lb": [1, 2]},
            data=[1.0, 2.0],
        )
    assert cumulative_trials_for(proj, family="sweep:lb") == (11, 11)


def test_ledger_fail_closed_on_corrupted_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    proj = init_project(tmp_path, name="fail-closed-test")

    # Corrupt `runs.jsonl`
    runs_file = proj.ledger_dir / "runs.jsonl"
    runs_file.parent.mkdir(parents=True, exist_ok=True)
    runs_file.write_text("{not valid json\n", encoding="utf-8")

    with pytest.raises(LedgerError, match="Corrupted"):
        sweep(lambda data, k=1: [0.01 * k] * 30, {"k": [1, 2]}, data=[1.0, 2.0])

    with pytest.raises(LedgerError, match="Corrupted"):
        record_run(
            proj,
            command="validate",
            params={"k": 1},
            dataset_hashes=["abc"],
            n_trials=1,
            n_trials_source="single_signal",
            figures={"sharpe": 1.0},
            verdict="marginal",
            family="test_fam",
        )


def test_run_manifest_cas_and_three_status_verification_and_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QUANTOS_API_KEY", "ae_live_manifest_test_key")

    assert cli.main(["init", "--name", "manifest-proj"]) == 0
    assert cli.main(["demo"]) == 0
    capsys.readouterr()

    from alphaengine.project import find_project

    proj = find_project(tmp_path)
    assert proj is not None
    head = resolve_run(proj, "HEAD")

    # Manifest recorded and stored in CAS
    man_dict = head.get("manifest")
    assert isinstance(man_dict, dict)
    man = RunManifest.from_dict(man_dict)
    assert man.engine_version == "2.0.4"
    assert man.manifest_version == 1
    man_sha = head.get("manifest_sha256")
    assert isinstance(man_sha, str) and len(man_sha) == 64
    assert proj.get_object(man_sha)["engine_version"] == "2.0.4"

    # Rerun checks all 3 statuses: hash_verified, figures_reproduced, provenance_verified
    run_family = str(head.get("family") or "demo")
    prior_fam, prior_proj = cumulative_trials_for(proj, family=run_family)
    rep = rerun_record_report(proj, head)
    assert rep["hash_verified"] is True
    assert rep["figures_reproduced"] is True
    assert rep["provenance_verified"] is True
    assert rep["n_trials_added"] == 1
    after_api_fam, after_api_proj = cumulative_trials_for(proj, family=run_family)
    assert after_api_proj == prior_proj + 1
    assert after_api_fam == prior_fam + 1

    # ae rerun counts as exactly 1 new trial (William's 2.0.1 decision)
    assert cli.main(["rerun", "HEAD", "--json"]) == 0
    cli_rep = json.loads(capsys.readouterr().out)
    assert cli_rep["hash_verified"] is True
    assert cli_rep["figures_reproduced"] is True
    assert cli_rep["provenance_verified"] is True
    assert cli_rep["n_trials_added"] == 1
    after_rerun_fam, after_rerun_proj = cumulative_trials_for(proj, family=run_family)
    assert after_rerun_proj == after_api_proj + 1
    assert after_rerun_fam == after_api_fam + 1

    # Two more reruns add exactly 2
    before_two_fam, before_two_proj = cumulative_trials_for(proj, family=run_family)
    assert cli.main(["rerun", "HEAD", "--json"]) == 0
    capsys.readouterr()
    assert cli.main(["rerun", "HEAD", "--json"]) == 0
    capsys.readouterr()
    after_two_fam, after_two_proj = cumulative_trials_for(proj, family=run_family)
    assert after_two_proj == before_two_proj + 2
    assert after_two_fam == before_two_fam + 2

    # Seal study from run and verify 3 statuses via `verify_study_file` and `ae verify`
    study, study_path = seal_from_project(proj, "HEAD")
    assert isinstance(study.manifest, dict)
    ok, v_info = verify_study_file(study_path, proj=proj)
    assert ok is True
    assert v_info["hash_verified"] is True
    assert v_info["figures_reproduced"] is True
    assert v_info["provenance_verified"] is True

    assert cli.main(["verify", str(study_path)]) == 0
    ver_out = capsys.readouterr().out
    assert "Verified." in ver_out
    assert "hash_verified=true" in ver_out
    assert "figures_reproduced=true" in ver_out
    assert "provenance_verified=true" in ver_out

    # Simulate engine version drift on the recorded run manifest -> provenance_verified=False!
    drifted_run = dict(head)
    drifted_man = dict(man_dict)
    drifted_man["engine_version"] = "1.9.9"
    drifted_run["manifest"] = drifted_man
    drift_rep = rerun_record_report(proj, drifted_run)
    assert drift_rep["hash_verified"] is True
    assert drift_rep["figures_reproduced"] is True
    assert drift_rep["provenance_verified"] is False
    assert "engine_version" in drift_rep["provenance"]["drift"]

    # When provenance check fails, ae verify MUST report failure and NOT print "Verified."
    drifted_study = load(study_path)
    drifted_study.manifest["engine_version"] = "1.8.0"
    drifted_study.seal()
    drifted_study_path = proj.studies_dir / "drifted_study.json"
    save(drifted_study, drifted_study_path)
    assert cli.main(["verify", str(drifted_study_path)]) == 1
    drift_out = capsys.readouterr().out
    assert "Verified." not in drift_out
    assert "Verification failed" in drift_out
    assert "provenance drift" in drift_out

    # Standalone `build_run_manifest` helper works cleanly outside any project
    standalone = build_run_manifest(params={"lookback": 20}, dataset_hashes=["deadbeef"])
    assert standalone.engine_version == "2.0.4"
    assert standalone.annualization["periods_per_year"] == 252
