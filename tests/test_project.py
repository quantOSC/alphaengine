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


def test_dataset_registry_hygiene_checks_and_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("QUANTOS_API_KEY", raising=False)
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
    monkeypatch.delenv("QUANTOS_API_KEY", raising=False)
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
    monkeypatch.delenv("QUANTOS_API_KEY", raising=False)
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

    # 3. Run 2 on the same idea/family: emits prior-work warning!
    assert cli.main(["validate", "--project", "alphaengine.demo", "--idea", idea_id]) == 0
    run2_out = capsys.readouterr().out
    assert "prior work" in run2_out
    assert "similar to run_" in run2_out

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
    monkeypatch.delenv("QUANTOS_API_KEY", raising=False)
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
    monkeypatch.delenv("QUANTOS_API_KEY", raising=False)
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
    assert "https://sync.example.invalid" in rem_out

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
