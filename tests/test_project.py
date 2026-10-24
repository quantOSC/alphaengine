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
