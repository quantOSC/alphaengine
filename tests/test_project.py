"""Tests for the `.alphaengine/` local research project workspace (Item 1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from alphaengine import cli
from alphaengine.project import (
    PROJECT_DIR_NAME,
    PROJECT_FORMAT_VERSION,
    SUBDIRS,
    find_project,
    init_project,
)


def test_ae_init_creates_project_layout_and_project_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    rc = cli.main(["init", "--name", "macro-desk"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Initialized." in out
    assert "macro-desk" in out

    dot = tmp_path / PROJECT_DIR_NAME
    assert dot.is_dir()
    assert (dot / "project.toml").is_file()
    for sub in SUBDIRS:
        assert (dot / sub).is_dir(), f"missing {sub}/ directory"

    proj = find_project(tmp_path)
    assert proj is not None
    assert proj.name == "macro-desk"
    assert proj.schema_version == PROJECT_FORMAT_VERSION
    assert proj.created
    assert proj.root == tmp_path.resolve()


def test_find_project_walks_up_from_nested_subdirectories_and_returns_none_outside(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    assert find_project(outside) is None

    proj_root = tmp_path / "workspace"
    init_project(proj_root, name="stat-arb")
    nested = proj_root / "notebooks" / "2026" / "q4"
    nested.mkdir(parents=True)

    found = find_project(nested)
    assert found is not None
    assert found.root == proj_root.resolve()
    assert found.name == "stat-arb"


def test_content_addressed_objects_and_append_only_jsonl(tmp_path: Path) -> None:
    proj = init_project(tmp_path, name="cas-test")
    payload = {"b": [1, 2, 3], "a": "hello", "score": 1.23456789}
    digest = proj.put_object(payload)
    assert len(digest) == 64

    # Deterministic: key order does not change digest
    digest2 = proj.put_object({"score": 1.23456789, "a": "hello", "b": [1, 2, 3]})
    assert digest == digest2

    loaded = proj.get_object(digest)
    assert loaded["a"] == "hello"
    assert loaded["b"] == [1, 2, 3]

    # Tampering with an object on disk is caught by get_object
    obj_file = proj.objects_dir / f"{digest}.json"
    raw = obj_file.read_text(encoding="utf-8")
    obj_file.write_text(raw.replace("hello", "tampered"), encoding="utf-8")
    with pytest.raises(ValueError, match="failed integrity check"):
        proj.get_object(digest)
    # Restore valid content
    obj_file.write_text(raw, encoding="utf-8")

    # Append-only JSONL writes both the line and its content-addressed object
    rec_sha = proj.append_jsonl("ledger/runs.jsonl", {"run_id": "run_001", "verdict": "robust"})
    rows = proj.read_jsonl("ledger/runs.jsonl")
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run_001"
    assert rows[0]["object_sha256"] == rec_sha
    assert proj.get_object(rec_sha)["run_id"] == "run_001"


def test_pyproject_declares_ae_console_script_and_docs_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject_text = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'ae = "alphaengine.cli:main"' in pyproject_text
    doc = root / "docs" / "project-format.md"
    assert doc.is_file()
    assert PROJECT_FORMAT_VERSION in doc.read_text(encoding="utf-8")


def test_project_ledger_accumulates_trials_and_supports_log_show_diff_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import numpy as np

    from alphaengine import sweep

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("QUANTOS_API_KEY", raising=False)
    assert cli.main(["init", "--name", "ledger-test"]) == 0
    capsys.readouterr()

    # Write a local wide CSV and a small project module
    csv_path = tmp_path / "prices.csv"
    rng = np.random.default_rng(7)
    lines = ["date,AAPL,MSFT,NVDA"]
    p_a, p_m, p_n = 100.0, 150.0, 200.0
    for i in range(60):
        p_a *= 1.0 + float(rng.normal(0.0005, 0.01))
        p_m *= 1.0 + float(rng.normal(0.0004, 0.01))
        p_n *= 1.0 + float(rng.normal(0.0006, 0.012))
        lines.append(f"2025-01-{i + 1:02d},{p_a:.4f},{p_m:.4f},{p_n:.4f}")
    csv_path.write_text("\n".join(lines), encoding="utf-8")

    # Run 1: `ae validate --project alphaengine.demo --family mom --idea idea_1`
    assert cli.main(["validate", "--project", "alphaengine.demo", "--family", "mom", "--idea", "idea_1"]) == 0
    out1 = capsys.readouterr().out
    assert "trials (this run)" in out1 and "9" in out1
    assert "trials (cumulative)" in out1 and "9" in out1

    # Run 2: second run on the same idea/family accumulates trials (9 + 9 = 18)
    assert cli.main(["validate", "--project", "alphaengine.demo", "--family", "mom", "--idea", "idea_1"]) == 0
    out2 = capsys.readouterr().out
    assert "trials (this run)" in out2 and "9" in out2
    assert "18" in out2

    # Run 3: `ae process ou` on a different family
    assert cli.main(["process", "ou", "--family", "ou_stress"]) == 0
    out3 = capsys.readouterr().out
    assert "cumulative trials 80" in out3

    # Direct Python `sweep()` inside a project also records to ledger and accumulates trials
    r_py = sweep(
        lambda data=None, fast=5: [0.001 * ((i % 5) - 2) for i in range(40)],
        {"fast": [5, 10, 15]},
        data=[100.0 + i for i in range(45)],
    )
    assert r_py.run_id is not None
    assert r_py.cumulative_trials == 3
    assert r_py.verdict()["cumulative_trials"] == 3

    # `ae log` with filters
    assert cli.main(["log", "--idea", "idea_1"]) == 0
    log_idea = capsys.readouterr().out
    assert "validate_study" in log_idea
    assert "process" not in log_idea

    assert cli.main(["log", "--family", "ou_stress"]) == 0
    log_fam = capsys.readouterr().out
    assert "process" in log_fam

    # `ae show` on the first and second runs
    proj = find_project(tmp_path)
    assert proj is not None
    runs = proj.read_jsonl("ledger/runs.jsonl")
    assert len(runs) == 4
    r1_id = str(runs[0]["run_id"])
    r3_id = str(runs[2]["run_id"])

    assert cli.main(["show", r1_id[:8]]) == 0
    show_out = capsys.readouterr().out
    assert r1_id in show_out
    assert "params_hash" in show_out
    assert "dataset_hashes" in show_out

    # `ae diff <run-a> <run-b>`
    assert cli.main(["diff", r1_id, r3_id]) == 0
    diff_out = capsys.readouterr().out
    assert "diff" in diff_out
    assert "figure changes" in diff_out

    # `ae rerun <run>` reproduces both workflow runs and Python sweep runs
    assert cli.main(["rerun", r1_id]) == 0
    rerun1 = capsys.readouterr().out
    assert "Reproduced." in rerun1

    assert cli.main(["rerun", str(r_py.run_id)]) == 0
    rerun_py = capsys.readouterr().out
    assert "Reproduced." in rerun_py
