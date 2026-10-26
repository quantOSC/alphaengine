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


@pytest.fixture(autouse=True)
def _default_firm_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "_cfg"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "_cfg"))
    monkeypatch.setenv("QUANTOS_API_KEY", "ae_live_firm_test_key")


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
    assert "trials (cumulative)" in out1 and "9" in out1

    # Identical re-run: same config hash must NOT increase cumulative N (stays 9, not 18)
    assert cli.main(["validate", "--project", "alphaengine.demo", "--family", "mom", "--idea", "idea_1"]) == 0
    out_dup = capsys.readouterr().out
    assert "trials (cumulative)" in out_dup and "9" in out_dup

    # Distinct config on the same dataset: adds 9 new trials (9 + 9 = 18) and updates headline DSR
    assert (
        cli.main(
            [
                "validate",
                "--project",
                "alphaengine.demo",
                "--family",
                "mom",
                "--idea",
                "idea_1",
                "--input",
                'grid={"fast":[6,12,24],"slow":[60,120,240]}',
            ]
        )
        == 0
    )
    out2 = capsys.readouterr().out
    assert "trials (cumulative)" in out2 and "18" in out2

    # Run 3: `ae process ou` on a different family accumulates into the project-wide ledger (18 + 80 = 98)
    assert cli.main(["process", "ou", "--family", "ou_stress"]) == 0
    out3 = capsys.readouterr().out
    assert "cumulative trials 98" in out3

    # Direct Python `sweep()` inside a project also records to ledger and accumulates trials (98 + 3 = 101)
    r_py = sweep(
        lambda data=None, fast=5: [0.001 * ((i % 5) - 2) for i in range(40)],
        {"fast": [5, 10, 15]},
        data=[100.0 + i for i in range(45)],
    )
    assert r_py.run_id is not None
    assert r_py.cumulative_trials == 101
    assert r_py.verdict()["n_trials"] == 101
    assert r_py.verdict()["cumulative_trials"] == 101

    # `ae log` with filters
    assert cli.main(["log", "--idea", "idea_1"]) == 0
    log_idea = capsys.readouterr().out
    assert "validate_study" in log_idea
    assert "process" not in log_idea

    assert cli.main(["log", "--family", "ou_stress"]) == 0
    log_fam = capsys.readouterr().out
    assert "process" in log_fam

    # `ae show` on the first and third runs
    proj = find_project(tmp_path)
    assert proj is not None
    runs = proj.read_jsonl("ledger/runs.jsonl")
    assert len(runs) == 5
    assert runs[1]["duplicate_of"] == runs[0]["run_id"]
    assert runs[1]["cumulative_trials"] == 9
    assert runs[2]["cumulative_trials"] == 18
    assert runs[2]["figures"]["deflated_sharpe"] == runs[2]["figures"]["cumulative_deflated_sharpe"]
    r1_id = str(runs[0]["run_id"])
    r3_id = str(runs[3]["run_id"])

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


def test_dataset_registry_hygiene_checks_and_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert cli.main(["init", "--name", "data-test"]) == 0
    capsys.readouterr()

    # Create a CSV with:
    # - duplicate timestamp on 2025-01-02
    # - missing cell (gap) on 2025-01-03 for MSFT
    # - survivorship hint (LEAVER ends early on 2025-01-02 while AAPL/MSFT continue to 2025-01-15)
    # - look-ahead hint (row dated 2025-01-15 with asof=2025-01-10)
    dirty_csv = tmp_path / "dirty_prices.csv"
    dirty_csv.write_text(
        "\n".join(
            [
                "date,asof,AAPL,MSFT,LEAVER",
                "2025-01-01,2025-01-01,100.0,200.0,50.0",
                "2025-01-02,2025-01-02,101.0,201.0,49.0",
                "2025-01-02,2025-01-02,101.5,201.5,48.5",
                "2025-01-03,2025-01-03,102.0,,",
                "2025-01-15,2025-01-10,105.0,205.0,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert cli.main(["data", "add", str(dirty_csv), "--name", "dirty_eq", "--asof-col", "asof"]) == 0
    add_out = capsys.readouterr().out
    assert "Registered." in add_out
    assert "dirty_eq" in add_out
    assert "duplicates=1" in add_out
    assert "lookahead_rows=1" in add_out
    assert "early_ending_series=1" in add_out

    assert cli.main(["data", "list"]) == 0
    list_out = capsys.readouterr().out
    assert "dirty_eq" in list_out

    assert cli.main(["data", "show", "dirty_eq"]) == 0
    show_out = capsys.readouterr().out
    assert "LEAVER" in show_out
    assert "lookahead_hints" in show_out

    # Now register a clean panel and run `diagnose --data clean_panel` (by registered name!)
    clean_csv = tmp_path / "clean.csv"
    rows = ["date,AAPL,MSFT"] + [
        f"2025-01-{i + 1:02d},{100.0 + i * 0.5:.2f},{200.0 + i * 0.8:.2f}" for i in range(25)
    ]
    clean_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert cli.main(["data", "add", str(clean_csv), "--name", "clean_panel"]) == 0
    capsys.readouterr()

    assert cli.main(["diagnose", "--data", "clean_panel"]) == 0
    capsys.readouterr()

    assert cli.main(["data", "lineage", "HEAD"]) == 0
    lin_out = capsys.readouterr().out
    assert "clean_panel" in lin_out
    assert "diagnose_data" in lin_out


def test_json_envelope_pipes_and_declarative_yaml_pipelines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import io
    import json

    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    assert cli.main(["init", "--name", "pipe-test"]) == 0
    capsys.readouterr()

    # 1. Test `ae screen --json | ae validate --stdin --json | ae size --stdin --json`
    assert cli.main(["screen", "--json"]) == 0
    screen_json = capsys.readouterr().out
    env1 = json.loads(screen_json)
    assert env1["schema_version"] == "1.0"
    assert env1["command"] == "screen_universe"
    assert env1["context"]["top_symbol"]
    assert len(env1["context"]["returns"]) >= 20

    monkeypatch.setattr("sys.stdin", io.StringIO(screen_json))
    assert cli.main(["validate", "--stdin", "--json"]) == 0
    val_json = capsys.readouterr().out
    env2 = json.loads(val_json)
    assert env2["command"] == "validate_study"
    assert env2["n_trials"] == 60
    assert env2["figures"]["deflated_sharpe"] is not None

    monkeypatch.setattr("sys.stdin", io.StringIO(val_json))
    assert cli.main(["size", "--stdin", "--json"]) == 0
    size_json = capsys.readouterr().out
    env3 = json.loads(size_json)
    assert env3["command"] == "size_position"
    assert "cvar_pct" in env3["figures"]

    # 2. Test declarative YAML pipelines shipped in `examples/`
    ex1 = root / "examples" / "screen_validate_size.yaml"
    ex2 = root / "examples" / "process_stress_monitor.yaml"
    assert ex1.is_file() and ex2.is_file()
    assert (root / "docs" / "result-format.md").is_file()

    assert cli.main(["run", str(ex1), "--json"]) == 0
    pipe1 = json.loads(capsys.readouterr().out)
    assert pipe1["command"] == "pipeline"
    assert len(pipe1["steps"]) == 4
    assert [s["command"] for s in pipe1["steps"]] == [
        "diagnose_data",
        "screen_universe",
        "validate_study",
        "size_position",
    ]

    assert cli.main(["run", str(ex2)]) == 0
    pipe2_out = capsys.readouterr().out
    assert "process" in pipe2_out
    assert "stress_study" in pipe2_out
    assert "monitor_sleeve" in pipe2_out

    proj = find_project(tmp_path)
    assert proj is not None
    # 3 piped runs + 4 steps in ex1 + 3 steps in ex2 = 10 runs in ledger
    assert len(proj.read_jsonl("ledger/runs.jsonl")) == 10


def test_hypotheses_as_objects_kill_rules_and_prior_work_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    monkeypatch.chdir(tmp_path)
    assert cli.main(["init", "--name", "idea-test"]) == 0
    capsys.readouterr()

    # 1. Create an idea with a kill rule
    assert (
        cli.main(
            [
                "idea",
                "new",
                "Moving average crossover on demo walk",
                "--kill-if",
                "DSR < 0.9",
                "--family",
                "ma_crossover",
                "--json",
            ]
        )
        == 0
    )
    idea_rec = json.loads(capsys.readouterr().out)
    idea_id = idea_rec["idea_id"]
    assert idea_rec["status"] == "open"
    assert idea_rec["family"] == "ma_crossover"

    # 2. Run 1 attached to `--idea <id>`: transitions status to `testing`
    assert cli.main(["validate", "--project", "alphaengine.demo", "--idea", idea_id]) == 0
    run1_out = capsys.readouterr().out
    assert "prior work" not in run1_out

    # 3. Run 2 with an identical config emits prior-work warning and does NOT double-count trials (stays 9)
    assert cli.main(["validate", "--project", "alphaengine.demo", "--idea", idea_id]) == 0
    run2_out = capsys.readouterr().out
    assert "prior work" in run2_out
    assert "similar to run_" in run2_out

    # 3b. Run 3 with a distinct grid emits prior-work warning and accumulates trials (9 -> 18)
    assert (
        cli.main(
            [
                "validate",
                "--project",
                "alphaengine.demo",
                "--idea",
                idea_id,
                "--input",
                'grid={"fast":[7,14,28],"slow":[60,120,240]}',
            ]
        )
        == 0
    )
    capsys.readouterr()

    # 4. `ae idea show <id>` shows linked runs, cumulative trials (18), and kill rule TRIGGERED
    assert cli.main(["idea", "show", idea_id]) == 0
    show_out = capsys.readouterr().out
    assert "testing" in show_out
    assert "18" in show_out
    assert "kill_triggered" in show_out
    assert "TRIGGERED" in show_out

    # 5. Update status to `killed` and verify `ae idea list`
    assert cli.main(["idea", "status", idea_id[:8], "killed"]) == 0
    capsys.readouterr()
    assert cli.main(["idea", "list"]) == 0
    list_out = capsys.readouterr().out
    assert idea_id in list_out
    assert "killed" in list_out


def test_seal_run_or_idea_verify_study_with_data_and_export_memo_md_and_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    monkeypatch.chdir(tmp_path)
    assert cli.main(["init", "--name", "seal-memo-test"]) == 0
    capsys.readouterr()

    # Create idea and run `validate`
    assert cli.main(["idea", "new", "Demo MA crossover check", "--kill-if", "DSR < 0.9", "--json"]) == 0
    idea_id = json.loads(capsys.readouterr().out)["idea_id"]

    assert cli.main(["validate", "--project", "alphaengine.demo", "--idea", idea_id, "--json"]) == 0
    run_env = json.loads(capsys.readouterr().out)
    run_id = run_env["run_id"]

    # 1. `ae seal <idea>` writes `.alphaengine/studies/<idea_id>.json` with schema 1.2 & trial ledger
    assert cli.main(["seal", idea_id, "--json"]) == 0
    sealed_idea = json.loads(capsys.readouterr().out)
    study_path = Path(sealed_idea["path"])
    assert study_path.is_file()
    assert len(sealed_idea["content_sha256"]) == 64
    assert sealed_idea["study"]["schema_version"] == "1.2"
    assert len(sealed_idea["study"]["trial_ledger"]) == 9
    assert sealed_idea["study"]["psr_vs_zero"] is not None
    assert sealed_idea["study"]["sr0_expected_max"] is not None
    assert sealed_idea["study"]["min_track_record_length"] is not None
    assert sealed_idea["study"]["selection"] is not None

    # 2. `ae verify <study.json>` verifies the hash and re-runs the project run to confirm figures
    assert cli.main(["verify", str(study_path)]) == 0
    ver_out = capsys.readouterr().out
    assert "Verified." in ver_out
    assert "figures reproduced" in ver_out

    # 3. Also test `ae verify <study.json> --data <file>` on a returns CSV study
    rets_csv = tmp_path / "returns.csv"
    from alphaengine.demo_returns import data as demo_rets

    rets_csv.write_text("returns\n" + "\n".join(str(x) for x in demo_rets) + "\n", encoding="utf-8")
    assert cli.main(["size", "--data", str(rets_csv), "--json"]) == 0
    size_env = json.loads(capsys.readouterr().out)
    assert cli.main(["seal", size_env["run_id"], "--json"]) == 0
    size_study_path = Path(json.loads(capsys.readouterr().out)["path"])

    assert cli.main(["verify", str(size_study_path), "--data", str(rets_csv)]) == 0
    ver_data_out = capsys.readouterr().out
    assert "Verified." in ver_data_out

    # Wrong data file fails verification
    wrong_csv = tmp_path / "wrong.csv"
    wrong_csv.write_text("returns\n" + "\n".join(str(x * -2.0) for x in demo_rets) + "\n", encoding="utf-8")
    assert cli.main(["verify", str(size_study_path), "--data", str(wrong_csv)]) == 1
    capsys.readouterr()

    # 4. `ae report memo <idea> --format md` cites `[run_id]` on every figure
    assert cli.main(["report", "memo", idea_id, "--format", "md"]) == 0
    memo_out = capsys.readouterr().out
    assert f"[{run_id}]" in memo_out
    assert "Research tool, not investment advice." in memo_out
    assert (tmp_path / ".alphaengine" / "studies" / f"memo_{idea_id}.md").is_file()

    # 5. `ae report memo <run> --format pdf` writes a PDF memo when matplotlib is installed
    pytest.importorskip("matplotlib")
    assert cli.main(["report", "memo", run_id, "--format", "pdf"]) == 0
    capsys.readouterr()
    pdf_memo = tmp_path / ".alphaengine" / "studies" / f"memo_{run_id}.pdf"
    assert pdf_memo.is_file()
    assert pdf_memo.read_bytes().startswith(b"%PDF")


def test_sync_remote_push_dry_run_404_fallback_pull_and_512_no_dataset_rows_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import io
    import json
    import urllib.error
    import urllib.request

    from alphaengine.client.session import MAX_FIGURE_LIST
    from alphaengine.sync import _guard_sync_payload

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ALPHAENGINE_SHARE_SERIES", raising=False)
    assert cli.main(["init", "--name", "sync-test"]) == 0
    capsys.readouterr()

    # Register a dataset with secret price values that must NEVER appear in sync payloads
    secret_csv = tmp_path / "secret_prices.csv"
    secret_csv.write_text(
        "date,AAPL,MSFT\n2025-01-01,9876.5432,5432.1098\n2025-01-02,9888.1111,5444.2222\n",
        encoding="utf-8",
    )
    assert cli.main(["data", "add", str(secret_csv), "--name", "secret_eq"]) == 0
    capsys.readouterr()

    # Create an idea, run a validation, and seal it
    assert cli.main(["idea", "new", "Momentum persistence", "--kill-if", "DSR < 0.9", "--json"]) == 0
    idea_id = json.loads(capsys.readouterr().out)["idea_id"]
    assert cli.main(["validate", "--project", "alphaengine.demo", "--idea", idea_id]) == 0
    assert cli.main(["seal", idea_id]) == 0
    capsys.readouterr()

    # 1. `ae remote add quantos <url>`
    assert cli.main(["remote", "add", "quantos", "https://sync.example.invalid"]) == 0
    rem_out = capsys.readouterr().out
    assert rem_out.split()[-1] == "https://sync.example.invalid"

    # 2. `ae push --dry-run` prints exact payload without touching network
    assert cli.main(["push", "--dry-run"]) == 0
    dry_text = capsys.readouterr().out
    dry_payload = json.loads(dry_text)
    assert dry_payload["project"]["name"] == "sync-test"
    assert len(dry_payload["studies"]) == 1
    assert len(dry_payload["runs"]) == 1
    assert len(dry_payload["ideas"]) == 1
    assert "9876.5432" not in dry_text
    assert "repro_spec" not in dry_text
    assert "datasets" not in dry_payload

    # Guard explicitly refuses any > 512 element list or forbidden raw-data key
    with pytest.raises(ValueError, match="elements is a series"):
        _guard_sync_payload({"runs": [{"figures": {"curve": list(range(MAX_FIGURE_LIST + 1))}}]})
    with pytest.raises(ValueError, match="forbidden field"):
        _guard_sync_payload({"runs": [{"raw_rows": [1, 2, 3]}]})

    # 3. `ae push` against `/api/workspace/sync` (200 OK)
    sent_requests: list[tuple[str, dict]] = []

    def fake_urlopen_ok(req, timeout=None):
        sent_requests.append((req.full_url, json.loads(req.data.decode("utf-8")) if req.data else {}))

        class _Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return _Resp(b'{"ok": true}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen_ok)
    assert cli.main(["push", "--key", "ae_live_test"]) == 0
    push_ok_out = capsys.readouterr().out
    assert "/api/workspace/sync" in push_ok_out
    assert len(sent_requests) == 1
    assert sent_requests[0][0] == "https://sync.example.invalid/api/workspace/sync"

    # 4. `ae push` when `/api/workspace/sync` returns 404 falls back to `/api/me/runs/ingest`
    fallback_requests: list[str] = []

    def fake_urlopen_404(req, timeout=None):
        fallback_requests.append(req.full_url)
        if req.full_url.endswith("/api/workspace/sync"):
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, io.BytesIO(b""))

        class _Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return _Resp(b'{"id": "ing_1"}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen_404)
    assert cli.main(["push", "--key", "ae_live_test"]) == 0
    fb_out = capsys.readouterr().out
    assert "fell back to /api/me/runs/ingest" in fb_out
    assert fallback_requests == [
        "https://sync.example.invalid/api/workspace/sync",
        "https://sync.example.invalid/api/me/runs/ingest",
    ]

    # 5. `ae pull` imports team runs and ideas marked `read_only: True`
    def fake_urlopen_pull(req, timeout=None):
        class _Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        body = {
            "runs": [
                {
                    "run_id": "run_team_99",
                    "timestamp": "2026-10-24T12:00:00Z",
                    "command": "validate_study",
                    "family": "team_alpha",
                    "n_trials": 12,
                    "n_trials_source": "derived_from_grid",
                    "verdict": "robust",
                    "figures": {"deflated_sharpe": 0.94},
                }
            ],
            "ideas": [
                {
                    "idea_id": "idea_team_99",
                    "thesis": "Team cross-asset carry",
                    "family": "team_alpha",
                    "status": "testing",
                }
            ],
        }
        return _Resp(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen_pull)
    assert cli.main(["pull", "--key", "ae_live_test"]) == 0
    pull_out = capsys.readouterr().out
    assert "1 runs, 1 ideas (marked read-only)" in pull_out

    proj = find_project(tmp_path)
    assert proj is not None
    pulled_run = [r for r in proj.read_jsonl("ledger/runs.jsonl") if r.get("run_id") == "run_team_99"][0]
    assert pulled_run["read_only"] is True
    pulled_idea = [i for i in proj.read_jsonl("ideas/ideas.jsonl") if i.get("idea_id") == "idea_team_99"][0]
    assert pulled_idea["read_only"] is True


def test_research_shell_tab_completion_status_line_inline_commands_and_agent_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys
    import types

    from alphaengine.commands import COMMANDS
    from alphaengine.complete import enable, make_completer, project_completions, verbs_of
    from alphaengine.repl import SLASH, SessionState, run_shell

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("alphaengine.events.data_dir", lambda: tmp_path / "events")
    assert cli.main(["init", "--name", "shell-os"]) == 0
    capsys.readouterr()

    # Register a dataset, an idea, and two runs
    csv_p = tmp_path / "panel.csv"
    csv_p.write_text(
        "\n".join(
            ["date,AAPL,MSFT"] + [f"2025-01-{i + 1:02d},{100.0 + i:.2f},{200.0 + i:.2f}" for i in range(25)]
        )
        + "\n",
        encoding="utf-8",
    )
    assert cli.main(["data", "add", str(csv_p), "--name", "eq_panel"]) == 0
    assert (
        cli.main(["idea", "new", "Shell hypothesis", "--kill-if", "DSR < 0.9", "--family", "shell_fam"]) == 0
    )
    assert cli.main(["diagnose", "--data", "eq_panel", "--family", "shell_fam"]) == 0
    assert cli.main(["process", "gbm", "--family", "shell_fam"]) == 0
    capsys.readouterr()

    proj = find_project(tmp_path)
    assert proj is not None
    runs = proj.read_jsonl("ledger/runs.jsonl")
    ideas = proj.read_jsonl("ideas/ideas.jsonl")
    r_a = str(runs[0]["run_id"])
    r_b = str(runs[1]["run_id"])
    idea_id = str(ideas[0]["idea_id"])

    # 1. Test `complete.py`: verbs_of, project_completions, make_completer, enable()
    verbs = verbs_of(COMMANDS, extra=SLASH)
    pool = project_completions(proj, verbs=verbs, symbols=["AAPL", "MSFT"])
    assert r_a in pool and r_b in pool
    assert "eq_panel" in pool
    assert idea_id in pool
    assert "log" in pool and "/log" in pool

    comp = make_completer(verbs=verbs, symbols=["AAPL"], proj=proj)
    assert comp(r_a[:6], 0) == r_a
    assert comp("eq_pa", 0) == "eq_panel"
    assert comp(idea_id[:7], 0) == idea_id
    assert comp("nonexistent_prefix_xyz", 0) is None

    # Fake readline module to exercise `enable()` and its atexit hook
    rl_calls: dict[str, object] = {}
    fake_rl = types.ModuleType("readline")
    fake_rl.read_history_file = lambda p: rl_calls.setdefault("read", p)  # type: ignore[attr-defined]
    fake_rl.set_history_length = lambda n: rl_calls.setdefault("len", n)  # type: ignore[attr-defined]
    fake_rl.set_completer = lambda fn: rl_calls.setdefault("comp", fn)  # type: ignore[attr-defined]
    fake_rl.set_completer_delims = lambda d: rl_calls.setdefault("delims", d)  # type: ignore[attr-defined]
    fake_rl.parse_and_bind = lambda b: rl_calls.setdefault("bind", b)  # type: ignore[attr-defined]
    fake_rl.write_history_file = lambda p: rl_calls.setdefault("write", p)  # type: ignore[attr-defined]
    (tmp_path / "events").mkdir(parents=True, exist_ok=True)
    (tmp_path / "events" / "history").write_text("help\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "readline", fake_rl)
    enable(verbs=verbs, symbols=["AAPL"], proj=proj)
    assert callable(rl_calls.get("comp"))
    assert rl_calls.get("len") == 1000

    # 2. Test `SessionState` in `repl.py`
    st = SessionState()

    class _FakeRunObj:
        run_id = "r_mem_1"
        artifact = {"figures": {"rows": [{"symbol": "AAPL"}, {"symbol": "MSFT"}]}}

    st.last_answer = types.SimpleNamespace(text="AAPL led.")
    st.remember_run(_FakeRunObj(), question="which led?")
    assert st.last_shortlist == ["AAPL", "MSFT"]
    assert st.resolve_followup_symbol("now size AAPL") == "AAPL"
    assert st.resolve_followup_symbol("now size TSLA") is None

    # 3. Test `ae shell` (`run_shell`) status line & inline commands
    script_lines = iter(
        [
            "help",
            "log --family shell_fam",
            f"show {r_a}",
            f"diff {r_a} {r_b}",
            f"idea show {idea_id}",
            f"seal {r_a}",
            f"verify {r_a}",
            "remote list",
            "push --dry-run",
            "quit",
        ]
    )
    assert run_shell(input_fn=lambda _prompt: next(script_lines)) == 0
    shell_out = capsys.readouterr().out
    # Status line includes project name and cumulative trial count (1 + 80 = 81 trials)
    assert "project:shell-os (81 trials)" in shell_out
    assert r_a in shell_out and r_b in shell_out
    assert "Shell hypothesis" in shell_out

    # 4. Agent sentence inside project records a run to `.alphaengine/ledger/runs.jsonl`
    from alphaengine.tui.dispatch import Desk, submit

    desk = Desk(
        session=object(),
        url="https://api.example.invalid",
        args=None,
        data={"AAPL": [100.0 + i for i in range(40)], "MSFT": [200.0 + i for i in range(40)]},
        think=lambda p: (
            '{"tool":"profile","args":{}}'
            if "none yet" in p.split("Calls already made this turn:")[-1]
            else '{"tool":"done","args":{"answer":"Measured 2 names across 40 observations."}}'
        ),
    )
    before_count = len(proj.read_jsonl("ledger/runs.jsonl"))
    assert submit(desk, "profile the loaded dataset") is False
    after_runs = proj.read_jsonl("ledger/runs.jsonl")
    assert len(after_runs) == before_count + 1
    assert after_runs[-1]["command"] == "agent_tools"
    assert after_runs[-1]["n_trials_source"] == "agent_proposal"

    # Exercise inline dispatch branches in `tui/dispatch.py`
    assert submit(desk, f"rerun {r_a}") is False
    assert submit(desk, "data list") is False
    assert submit(desk, "data show eq_panel") is False
    assert submit(desk, f"data lineage {r_a}") is False
    assert submit(desk, "idea list") is False
    assert submit(desk, f"idea status {idea_id} promoted") is False
    assert submit(desk, f"report memo {r_a} --format md") is False
    assert submit(desk, "plugin list") is False
    assert submit(desk, "diff only_one_arg") is False
    capsys.readouterr()


def test_plugin_system_loaders_methods_templates_and_entrypoint_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import importlib.metadata
    import json

    from alphaengine.plugins import (
        PluginSpec,
        discover_plugins,
        load_with_plugin,
        register_plugin,
        run_method_plugin,
    )

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs" / "plugins.md").is_file()

    # 1. `ae plugin list` shows built-in loader, method, and template plugins
    assert cli.main(["plugin", "list"]) == 0
    list_out = capsys.readouterr().out
    for expected in ("quantconnect", "vectorbt", "csv", "deflated_sharpe", "committee_memo"):
        assert expected in list_out

    assert cli.main(["plugin", "list", "--json"]) == 0
    pl_json = json.loads(capsys.readouterr().out)
    names = {p["name"] for p in pl_json["plugins"]}
    assert {"quantconnect", "vectorbt", "csv", "deflated_sharpe", "committee_memo"} <= names

    # 2. QuantConnect and VectorBT loaders work via `load_with_plugin`
    qc_study = load_with_plugin(
        "quantconnect",
        {"returns": [0.001 * ((i % 7) - 3) for i in range(40)], "totalBacktests": 5},
    )
    assert qc_study.n_trials == 5
    assert qc_study.n_trials_source == "asserted"

    vbt_study = load_with_plugin(
        "vectorbt",
        {"returns": [0.001 * ((i % 5) - 2) for i in range(40)]},
    )
    assert vbt_study.n_trials_source == "not_recorded"
    assert vbt_study.verdict is None

    # 3. Method plugin returns a `v1.0` result envelope
    env_dsr = run_method_plugin(
        "deflated_sharpe",
        [0.002 * ((i % 5) - 1) for i in range(50)],
        n_trials=4,
    )
    assert env_dsr["schema_version"] == "1.0"
    assert env_dsr["command"] == "deflated_sharpe"
    assert "deflated_sharpe" in env_dsr["figures"]

    # 4. Register a custom method plugin and mock an entry point
    register_plugin(
        PluginSpec(
            name="custom_tail_ratio",
            kind="method",
            description="Custom tail ratio diagnostic",
            fn=lambda rets: {"tail_ratio": 1.35, "n_trials": 2, "verdict": "robust"},
            source="test",
        )
    )
    custom_env = run_method_plugin("custom_tail_ratio", [0.01, -0.01, 0.02])
    assert custom_env["schema_version"] == "1.0"
    assert custom_env["figures"]["tail_ratio"] == 1.35

    class _FakeEP:
        name = "ep_diagnostic"
        dist = type("_D", (), {"name": "ext-pkg"})()

        @staticmethod
        def load():
            def fn(rets):
                """Entrypoint diagnostic."""
                return {"ep_ok": True, "len": len(rets)}

            fn.plugin_kind = "method"  # type: ignore[attr-defined]
            return fn

    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group=None: [_FakeEP()])
    all_found = {p.name: p for p in discover_plugins()}
    assert "ep_diagnostic" in all_found
    assert all_found["ep_diagnostic"].source == "ext-pkg"


def test_watch_pipeline_reruns_on_dataset_change_and_monitor_live_detects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from alphaengine.watch import watch_pipeline

    monkeypatch.chdir(tmp_path)
    assert cli.main(["init", "--name", "watch-monitor"]) == 0
    capsys.readouterr()

    # Create a registered dataset and a 1-step pipeline
    ds_csv = tmp_path / "prices.csv"
    ds_csv.write_text(
        "\n".join(
            ["date,AAPL,MSFT"]
            + [f"2025-01-{i + 1:02d},{100.0 + i * 0.5:.2f},{200.0 + i * 0.5:.2f}" for i in range(30)]
        )
        + "\n",
        encoding="utf-8",
    )
    assert cli.main(["data", "add", str(ds_csv), "--name", "watch_ds"]) == 0
    capsys.readouterr()

    pipe_yaml = tmp_path / "watch_pipe.yaml"
    pipe_yaml.write_text(
        "name: watch-test\nfamily: watch_fam\ndata: prices.csv\nsteps:\n  - command: diagnose\n",
        encoding="utf-8",
    )

    # 1. `watch_pipeline` runs once initially, then during poll cycle 1 we mutate `prices.csv`
    # and verify it automatically triggers a second pipeline run!
    proj = find_project(tmp_path)
    assert proj is not None
    poll_count = 0

    def mutate_on_first_sleep(_sec: float) -> None:
        nonlocal poll_count
        poll_count += 1
        if poll_count == 1:
            ds_csv.write_text(
                ds_csv.read_text(encoding="utf-8") + "2025-02-01,120.00,220.00\n",
                encoding="utf-8",
            )

    n_exec = watch_pipeline(
        pipe_yaml,
        proj=proj,
        interval=0.01,
        max_cycles=2,
        sleep_fn=mutate_on_first_sleep,
    )
    assert n_exec == 2

    # Also test CLI `ae watch <pipeline.yaml> --cycles 1 --interval 0.01`
    assert cli.main(["watch", str(pipe_yaml), "--cycles", "1", "--interval", "0.01"]) == 0
    watch_out = capsys.readouterr().out
    assert "watch" in watch_out

    # 2. Record a backtest run with calm positive returns, then run `ae monitor <run> --live <file>`
    bt_csv = tmp_path / "bt_returns.csv"
    bt_csv.write_text(
        "returns\n" + "\n".join(f"{0.002 + (i % 5) * 0.0005:.6f}" for i in range(80)) + "\n",
        encoding="utf-8",
    )
    assert cli.main(["validate", "--data", str(bt_csv), "--input", "n_trials=3", "--json"]) == 0
    bt_env = json.loads(capsys.readouterr().out)
    bt_run_id = bt_env["run_id"]
    bt_cum_n = bt_env["cumulative_trials"]
    bt_proj_n = bt_env["project_trials"]

    # Calm live returns within the expected range -> `within_expected_range`
    calm_live = tmp_path / "live_calm.csv"
    calm_live.write_text(
        "returns\n" + "\n".join(f"{0.002 + (i % 5) * 0.0005:.6f}" for i in range(40)) + "\n",
        encoding="utf-8",
    )
    assert cli.main(["monitor", bt_run_id, "--live", str(calm_live), "--json"]) == 0
    calm_env = json.loads(capsys.readouterr().out)
    assert calm_env["verdict"] == "within_expected_range"
    assert calm_env["figures"]["drift_flags"] == []
    assert calm_env["figures"]["run_n_trials"] == 0
    assert calm_env["cumulative_trials"] == bt_cum_n
    assert calm_env["project_trials"] == bt_proj_n

    # Re-running live monitor on same or updated live file never inflates cumulative N
    assert cli.main(["monitor", bt_run_id, "--live", str(calm_live), "--json"]) == 0
    calm_env_2 = json.loads(capsys.readouterr().out)
    assert calm_env_2["cumulative_trials"] == bt_cum_n
    assert calm_env_2["project_trials"] == bt_proj_n

    # Severe drawdown live returns outside the backtest's bootstrapped interval -> `drift_flagged`
    crash_live = tmp_path / "live_crash.csv"
    crash_live.write_text(
        "returns\n" + "\n".join("-0.045" for _ in range(30)) + "\n",
        encoding="utf-8",
    )
    assert cli.main(["monitor", bt_run_id, "--live", str(crash_live)]) == 0
    crash_out = capsys.readouterr().out
    assert "drift_flagged" in crash_out
    assert "Research tool, not investment advice." in crash_out

    from alphaengine.ledger import cumulative_trials_for

    _, final_proj_n = cumulative_trials_for(proj)
    assert final_proj_n == bt_proj_n


def test_require_firm_key_choke_point_keyring_and_no_key_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys
    import types

    from alphaengine import auth
    from alphaengine.auth import FIRM_KEY_ERROR_MESSAGE, FirmKeyError, redact_secrets, require_firm_key
    from alphaengine.envelope import build_envelope, emit_json
    from alphaengine.study import Study, save

    monkeypatch.chdir(tmp_path)

    # First create a sealed study file so we can verify `ae verify` works keyless
    sealed_path = tmp_path / "sealed.json"
    st = Study(label="keyless verify", n_trials=3, verdict="marginal")
    st.seal()
    save(st, sealed_path)

    # Remove all keys
    monkeypatch.delenv("QUANTOS_API_KEY", raising=False)
    auth.clear_stored(purge=True)

    with pytest.raises(FirmKeyError) as exc_info:
        require_firm_key()
    assert str(exc_info.value) == FIRM_KEY_ERROR_MESSAGE

    # 1. Every gated command refuses without an `ae_live_` key, including `validate` against a dead host
    gated_invocations = [
        ["validate", "--url", "http://127.0.0.1:9"],
        ["validate"],
        ["screen"],
        ["diagnose"],
        ["signal"],
        ["stress"],
        ["overlap"],
        ["size"],
        ["monitor"],
        ["workflows"],
        ["runs"],
        ["gaps"],
        ["tonight"],
        ["run", "validate_study", "--url", "http://127.0.0.1:9"],
        ["init"],
        ["log"],
        ["show", "HEAD"],
        ["diff", "a", "b"],
        ["rerun", "HEAD"],
        ["watch", "pipe.yaml"],
        ["data", "list"],
        ["idea", "list"],
        ["seal", str(sealed_path)],
        ["report", str(sealed_path)],
        ["remote", "list"],
        ["push", "--dry-run"],
        ["pull"],
        ["plugin", "list"],
        ["shell"],
    ]
    for argv in gated_invocations:
        rc = cli.main(argv)
        out = capsys.readouterr().out
        assert rc == 2, f"Expected exit 2 without key for {argv}, got {rc}"
        assert FIRM_KEY_ERROR_MESSAGE in out, f"Missing firm key error message for {argv}: {out}"

    # 2. Free keyless commands (`demo`, `process`, `verify`) succeed without any key
    assert cli.main(["verify", str(sealed_path)]) == 0
    assert "Verified." in capsys.readouterr().out
    assert cli.main(["process", "gbm"]) == 0
    capsys.readouterr()
    assert cli.main(["demo"]) == 0
    capsys.readouterr()

    # 3. OS keychain (`keyring`) storage with 0600 fallback
    kr_store: dict[tuple[str, str], str] = {}
    fake_kr = types.ModuleType("keyring")
    fake_kr.set_password = lambda svc, k, v: kr_store.__setitem__((svc, k), v)  # type: ignore[attr-defined]
    fake_kr.get_password = lambda svc, k: kr_store.get((svc, k))  # type: ignore[attr-defined]
    fake_kr.delete_password = lambda svc, k: kr_store.pop((svc, k), None)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "keyring", fake_kr)
    monkeypatch.setattr(auth, "_ALLOW_KEYRING_IN_TEST", True)

    auth.save_key("QUANTOS_API_KEY", "ae_live_keychain_secret_999")
    assert kr_store.get(("alphaengine", "QUANTOS_API_KEY")) == "ae_live_keychain_secret_999"
    assert auth.config_path().is_file()
    assert require_firm_key() == "ae_live_keychain_secret_999"
    auth.remove_key("QUANTOS_API_KEY")
    assert ("alphaengine", "QUANTOS_API_KEY") not in kr_store

    # 4. Key never appears in logs, errors, tracebacks, or `--json` output
    secret = "ae_live_super_secret_token_123456"
    assert secret not in redact_secrets(f"Authorization failed for Bearer {secret} in traceback")
    assert "ae_live_[REDACTED]" in redact_secrets(f"Key={secret}")

    cli.say(f"Error talking to server with key {secret}")
    say_out = capsys.readouterr().out
    assert secret not in say_out
    assert "ae_live_[REDACTED]" in say_out

    emit_json(build_envelope(command="test", figures={"leaked_note": f"used {secret}"}))
    json_out = capsys.readouterr().out
    assert secret not in json_out
    assert "ae_live_[REDACTED]" in json_out


def test_canonical_sha256_contract_and_legacy_verifier(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import hashlib
    import json

    from alphaengine.contracts import hash_golden_path
    from alphaengine.project import sha256_hex
    from alphaengine.study import Study, canonical_sha256, verify_study_mode
    from alphaengine.study.schema import _normalize_legacy_round10

    assert hash_golden_path().is_file()
    assert canonical_sha256({"x": 1.5}) == sha256_hex({"x": 1.5})
    assert canonical_sha256({"zero": -0.0}) == canonical_sha256({"zero": 0.0})

    # Build a study sealed with the legacy 10-decimal numeric float format and verify it is labeled 'legacy'
    st = Study(label="legacy study", n_trials=4, deflated_sharpe=0.5123456789)
    raw = st.to_dict()
    raw.pop("content_sha256", None)
    leg_blob = json.dumps(
        _normalize_legacy_round10(raw), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    leg_sha = hashlib.sha256(leg_blob.encode("utf-8")).hexdigest()
    st.content_sha256 = leg_sha

    assert verify_study_mode(st) == "legacy"
    assert st.verify() is True

    leg_path = tmp_path / "legacy_study.json"
    leg_path.write_text(json.dumps(st.to_dict(), indent=2), encoding="utf-8")
    assert cli.main(["verify", str(leg_path)]) == 0
    ver_out = capsys.readouterr().out
    assert "Verified (legacy)." in ver_out
