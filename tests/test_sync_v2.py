"""Tests for Sync Contract v2 (`QOS-152`, `QOS-161`–`QOS-167`)."""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alphaengine import cli, sweep
from alphaengine.ideas import create_idea, list_ideas
from alphaengine.memo import seal_from_project
from alphaengine.project import init_project
from alphaengine.study import Study, compute_content_sha256, load, save, verify_study_mode
from alphaengine.sync import (
    EXIT_AUTH_ERROR,
    EXIT_CONTRACT_MISMATCH,
    EXIT_NETWORK_OR_SERVER_ERROR,
    EXIT_OK,
    EXIT_PARTIAL_REJECTED,
    SYNC_CONTRACT_VERSION,
    PushRequest,
    PushResponse,
    SealedStudyWire,
    SyncAuthError,
    SyncContractMismatchError,
    SyncServerError,
    SyncValidationError,
    build_sync_payload,
    compute_push_exit_code,
    export_json_schemas,
    pull_workspace,
    push_workspace,
    remote_add,
)
from alphaengine.sync.schema import EXIT_ALL_REJECTED, PushItemAck


class _FakeHTTPResponse:
    def __init__(self, body: dict[str, Any], status: int = 200) -> None:
        self._raw = json.dumps(body).encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _raise_http_error(url: str, code: int, body: dict[str, Any]) -> None:
    fp = io.BytesIO(json.dumps(body).encode("utf-8"))
    raise urllib.error.HTTPError(url, code, f"HTTP {code}", hdrs=None, fp=fp)  # type: ignore[arg-type]


def test_study_seal_v2_roundtrip_and_local_extras_invariance(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    sw = sweep(
        lambda data, lb=5: rng.normal(0.001, 0.01, size=60) * (1.0 + 0.01 * lb),
        {"lb": [5, 10]},
        data=[1.0, 2.0],
    )
    st = Study.from_sweep(sw, label="seal-v2-check", notes="local research note")
    assert st.seal_version == 2
    assert len(st.content_sha256) == 64

    study_file = save(st, tmp_path / "study_v2.json")
    raw_disk = json.loads(study_file.read_text(encoding="utf-8"))
    # Local extra non-wire keys (such as `best_params` or local UI annotations)
    # must NOT alter `seal_version=2` `content_sha256` (`QOS-161`)!
    raw_disk["best_params"] = {"lb": 5}
    raw_disk["local_ui_bookmark"] = "pinned"
    assert compute_content_sha256(raw_disk) == st.content_sha256
    assert verify_study_mode(raw_disk) == "canonical"

    # Wire serialization verifies the exact same SHA-256 on receipt
    wire = SealedStudyWire.from_dict(st.to_dict())
    assert wire.seal_version == 2
    assert wire.content_sha256 == st.content_sha256

    # Tampering any sealed figure breaks verification immediately
    tampered = dict(raw_disk)
    tampered["verdict"] = "robust" if tampered.get("verdict") != "robust" else "likely_noise"
    assert verify_study_mode(tampered) is None
    with pytest.raises(SyncValidationError, match="content_sha256"):
        SealedStudyWire.from_dict(tampered)

    # Legacy 1.2 study without `seal_version == 2` verifies cleanly as `legacy_v1`
    loaded = load(study_file)
    assert loaded.content_sha256 == st.content_sha256


def test_sync_v2_payload_schema_share_series_and_json_schema_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    proj = init_project(tmp_path, name="sync-schema-test")
    create_idea(proj, thesis="Carry roll yield", kill_rule="dsr_probability < 0.5")

    rng = np.random.default_rng(19)
    long_rets = rng.normal(0.001, 0.01, size=600)  # > 512 elements -> series!
    sweep(lambda data, k=1: long_rets * k, {"k": [1, 2]}, data=[1.0, 2.0])
    seal_from_project(proj, "HEAD")

    # Default `share_series=False`: series (>512) stripped, forbidden v1 keys absent!
    payload_no_series = build_sync_payload(proj, share_series=False)
    assert payload_no_series["sync_contract"] == SYNC_CONTRACT_VERSION == 2
    assert payload_no_series["engine_version"] == "2.0.2"
    assert len(payload_no_series["studies"]) == 1
    assert len(payload_no_series["runs"]) == 1
    assert len(payload_no_series["ideas"]) == 1
    assert len(payload_no_series["trial_events"]) == 2

    idea_w = payload_no_series["ideas"][0]
    assert "idea_id" in idea_w and "kill_rule" in idea_w
    assert "kill_if" not in idea_w and "id" not in idea_w

    run_w = payload_no_series["runs"][0]
    assert run_w["share_series"] is False
    assert run_w["series"] is None
    assert "cumulative_trials" not in run_w
    assert "project_trials" not in run_w
    assert "repro_spec" not in run_w
    # Missing metrics remain `None` (never `1.20` or `0.0` defaults, `QOS-165`)
    assert "turnover" in run_w["metrics"]

    # Round-trip validation through `PushRequest.from_dict`
    parsed_req = PushRequest.from_dict(payload_no_series)
    assert parsed_req.sync_contract == 2
    assert len(parsed_req.trial_events) == 2
    assert all(ev.status in ("ok", "failed") and ev.attempt_id for ev in parsed_req.trial_events)

    # `share_series=True` opts in to per-run `share_series=True` (`QOS-166`)
    payload_with_series = build_sync_payload(proj, share_series=True)
    assert payload_with_series["runs"][0]["share_series"] is True

    # JSON Schema export covers PushRequest, PushResponse, PullRequest, PullResponse
    schemas = export_json_schemas()
    assert {"PushRequest", "PushResponse", "PullRequest", "PullResponse"}.issubset(set(schemas))


def test_push_per_item_acks_exit_codes_and_no_silent_404_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    proj = init_project(tmp_path, name="push-acks-test")
    remote_add(proj, "quantos", "https://api.quantos.example")
    idea = create_idea(proj, thesis="Value spread", kill_rule="pbo > 0.5")

    called_urls: list[str] = []

    # 1. Partial rejection -> EXIT_PARTIAL_REJECTED (2)
    def fake_partial(req: Any, timeout: int = 30) -> _FakeHTTPResponse:
        _ = timeout
        called_urls.append(req.full_url)
        return _FakeHTTPResponse(
            {
                "sync_contract": 2,
                "engine_version": "2.0.1",
                "items": [
                    {"type": "idea", "id": idea["idea_id"], "status": "accepted"},
                    {
                        "type": "run",
                        "id": "run_bad",
                        "status": "rejected",
                        "code": "invalid_hash",
                        "detail": "bad sha",
                    },
                ],
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_partial)
    res_part = push_workspace(proj, api_key="ae_live_test_key_123456")
    assert called_urls == ["https://api.quantos.example/api/workspace/sync"]
    assert res_part["exit_code"] == EXIT_PARTIAL_REJECTED == 2
    assert len(res_part["rejected"]) == 1

    # 2. All accepted/duplicate -> EXIT_OK (0); all rejected -> EXIT_ALL_REJECTED (3)
    assert (
        compute_push_exit_code(
            [PushItemAck(type="idea", local_id="i1", status="duplicate")],
            submitted_count=1,
        )
        == EXIT_OK
    )
    assert (
        compute_push_exit_code(
            [PushItemAck(type="idea", local_id="i1", status="rejected", reason="err")],
            submitted_count=1,
        )
        == EXIT_ALL_REJECTED
    )

    # 3. HTTP 404 MUST raise SyncServerError (`exit_code == 5`) with NO silent fallback (`QOS-162`, `QOS-167`)
    called_404: list[str] = []

    def fake_404(req: Any, timeout: int = 30) -> _FakeHTTPResponse:
        _ = timeout
        called_404.append(req.full_url)
        _raise_http_error(req.full_url, 404, {"detail": "route not mounted"})
        raise AssertionError("unreachable")

    monkeypatch.setattr("urllib.request.urlopen", fake_404)
    with pytest.raises(SyncServerError) as exc_404:
        push_workspace(proj, api_key="ae_live_test_key_123456")
    assert exc_404.value.exit_code == EXIT_NETWORK_OR_SERVER_ERROR == 5
    assert len(called_404) == 1  # Never retried `/api/harness/runs`!

    # 4. HTTP 401 -> SyncAuthError (`exit_code == 4`)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=30: _raise_http_error(req.full_url, 401, {"detail": "bad token"}),
    )
    with pytest.raises(SyncAuthError) as exc_401:
        push_workspace(proj, api_key="ae_live_test_key_123456")
    assert exc_401.value.exit_code == EXIT_AUTH_ERROR == 4

    # 5. HTTP 409 / `sync_contract` drift -> SyncContractMismatchError (`exit_code == 6`)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=30: _raise_http_error(
            req.full_url, 409, {"sync_contract": 1, "detail": "server expects v1"}
        ),
    )
    with pytest.raises(SyncContractMismatchError) as exc_409:
        push_workspace(proj, api_key="ae_live_test_key_123456")
    assert exc_409.value.exit_code == EXIT_CONTRACT_MISMATCH == 6

    # Response with mismatched `sync_contract` also raises SyncContractMismatchError (`exit_code == 6`)
    with pytest.raises(SyncContractMismatchError):
        PushResponse.from_dict({"sync_contract": 99, "engine_version": "9.0.0", "items": []})


def test_pull_cursor_pagination_idea_upsert_and_optional_none_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    proj = init_project(tmp_path, name="pull-cursor-test")
    remote_add(proj, "quantos", "https://api.quantos.example")

    seen_urls: list[str] = []
    pages = [
        {
            "sync_contract": 2,
            "engine_version": "2.0.1",
            "runs": [
                {
                    "run_id": "run_remote_1",
                    "timestamp": "2026-11-01T10:00:00Z",
                    "command": "validate",
                    "idea_id": "idea_shared_1",
                    "family": "mom",
                    "n_trials": 5,
                    "n_trials_source": "derived_from_grid",
                    "verdict": "marginal",
                    "metrics": {
                        "sharpe_annualized": None,  # MUST stay None, never 1.20!
                        "dsr_probability": 0.62,
                        "cagr": None,
                    },
                    "figures": {"dsr_probability": 0.62},
                }
            ],
            "ideas": [
                {
                    "idea_id": "idea_shared_1",
                    "thesis": "Initial cross-sectional momentum thesis",
                    "kill_rule": "dsr_probability < 0.5",
                    "status": "open",
                    "updated_at": "2026-11-01T10:00:00Z",
                }
            ],
            "studies": [],
            "next_cursor": "cur_page_2",
        },
        {
            "sync_contract": 2,
            "engine_version": "2.0.1",
            "runs": [],
            "ideas": [
                {
                    "idea_id": "idea_shared_1",
                    "thesis": "Revised cross-sectional momentum thesis (v2)",
                    "kill_rule": "dsr_probability < 0.6",
                    "status": "open",
                    "updated_at": "2026-11-02T12:00:00Z",
                }
            ],
            "studies": [],
            "next_cursor": None,
        },
    ]

    def fake_pull(req: Any, timeout: int = 30) -> _FakeHTTPResponse:
        _ = timeout
        seen_urls.append(req.full_url)
        idx = min(len(seen_urls) - 1, len(pages) - 1)
        return _FakeHTTPResponse(pages[idx])

    monkeypatch.setattr("urllib.request.urlopen", fake_pull)
    out = pull_workspace(proj, api_key="ae_live_test_key_123456", limit=50)

    assert len(seen_urls) == 2
    assert "/api/workspace/pull?limit=50" in seen_urls[0]
    assert "since=cur_page_2" in seen_urls[1]
    assert out["cursor"] == "cur_page_2"
    assert proj.get_sync_cursor("quantos") == "cur_page_2"

    # Idea `idea_shared_1` was revised in page 2 -> MUST update in place without duplicate rows (`QOS-163`)!
    ideas = list_ideas(proj)
    assert len(ideas) == 1
    assert ideas[0]["idea_id"] == "idea_shared_1"
    assert ideas[0]["thesis"] == "Revised cross-sectional momentum thesis (v2)"
    assert ideas[0]["kill_rule"] == "dsr_probability < 0.6"
    assert ideas[0]["read_only"] is True

    # Pulled run preserves `sharpe_annualized: None` without inventing `1.20` (`QOS-165`) and is `read_only`
    runs = proj.read_jsonl("ledger/runs.jsonl")
    assert len(runs) == 1
    assert runs[0]["run_id"] == "run_remote_1"
    assert runs[0]["read_only"] is True
    assert runs[0]["metrics"]["sharpe_annualized"] is None
    assert runs[0]["metrics"]["dsr_probability"] == 0.62

    # CLI exit code check on 404 during `ae pull`: exits 5, never falls back to `/api/harness/runs`
    monkeypatch.setenv("QUANTOS_API_KEY", "ae_live_test_key_123456")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=30: _raise_http_error(req.full_url, 404, {"detail": "missing"}),
    )
    rc = cli.main(["pull"])
    assert rc == EXIT_NETWORK_OR_SERVER_ERROR == 5


def test_unreachable_host_is_not_a_contract_upgrade() -> None:
    from alphaengine.sync import PullResponse, describe_unreachable, portal_host_note

    msg = describe_unreachable(
        "https://api.qosai.app",
        OSError("[Errno 11001] getaddrinfo failed"),
    )
    assert "does not resolve" in msg
    assert "not a sync-contract mismatch" in msg
    assert "ae remote add quantos" in msg
    assert portal_host_note("https://api.qosai.app")
    assert portal_host_note("https://api.quantos.example") is None
    with pytest.raises(SyncContractMismatchError, match="not the quantOS sync API"):
        PullResponse.from_dict({"runs": [], "ideas": [], "studies": []})


def test_remote_add_without_url_replaces_a_portal_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from alphaengine._version import DEFAULT_API_URL
    from alphaengine.sync import resolve_remote_url

    monkeypatch.delenv("ALPHAENGINE_API_URL", raising=False)
    monkeypatch.delenv("QUANTOS_API_URL", raising=False)
    proj = init_project(tmp_path, name="remote-reset")
    proj.remotes["quantos"] = "https://api.qosai.app"
    proj.save_config()

    resolved = remote_add(proj, "quantos")
    assert resolved == DEFAULT_API_URL.rstrip("/")
    assert proj.remotes["quantos"] == resolved

    proj.remotes["quantos"] = "https://qosai.app"
    proj.save_config()
    assert resolve_remote_url(proj, "quantos") == DEFAULT_API_URL.rstrip("/")
    assert proj.remotes["quantos"] == DEFAULT_API_URL.rstrip("/")

    custom = remote_add(proj, "quantos", "https://api.quantos.example")
    assert custom == "https://api.quantos.example"
    assert resolve_remote_url(proj, "quantos") == custom
