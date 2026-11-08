"""Opt-in workspace sync v2 (`ae remote`, `ae push`, `ae pull`).

Synchronizes sealed studies (`seal_version = 2`), allowlisted ledger run records,
hypothesis objects (`idea_id`, `kill_rule`), and idempotent per-attempt
`trial_events` with a remote `quantOS` workspace (`POST /api/workspace/sync` and
`GET /api/workspace/pull`, `sync_contract = 2`).

HARD DATA BOUNDARY
------------------
- Source code and raw dataset files/rows NEVER leave the machine.
- Lists longer than `MAX_FIGURE_LIST` (`512`) are stripped/refused unless
  `share_series=True` (`--share-series` or `ALPHAENGINE_SHARE_SERIES=1`).
- No silent endpoint fallbacks: HTTP `404`/`405` surfaces directly (`exit_code = 5`),
  `401`/`403` exits `4`, and `409`/`426` or `sync_contract` drift exits `6`.
"""

from __future__ import annotations

import contextlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Mapping
from typing import Any

from .._version import API_URL_ENV, DEFAULT_API_URL, LEGACY_API_URL_ENV, current_engine_version
from ..client.session import share_series_enabled
from ..core.conventions import normalize_verdict
from ..ideas import list_ideas, upsert_idea_revision
from ..project import Project
from ..study import Study, load, save
from ..study.report import MAX_FIGURE_LIST
from .schema import (
    EXIT_AUTH_ERROR,
    EXIT_CONTRACT_MISMATCH,
    EXIT_NETWORK_OR_SERVER_ERROR,
    EXIT_OK,
    EXIT_PARTIAL_REJECTED,
    SYNC_CONTRACT_VERSION,
    PullRequest,
    PullResponse,
    PushItemAck,
    PushRequest,
    PushResponse,
    SealedStudyWire,
    SyncAuthError,
    SyncContractMismatchError,
    SyncError,
    SyncIdeaPayload,
    SyncNetworkError,
    SyncRunPayload,
    SyncServerError,
    SyncTrialEventPayload,
    SyncValidationError,
    compute_push_exit_code,
    export_json_schemas,
    extract_optional_metrics,
)

__all__ = [
    "SYNC_CONTRACT_VERSION",
    "EXIT_OK",
    "EXIT_PARTIAL_REJECTED",
    "EXIT_AUTH_ERROR",
    "EXIT_NETWORK_OR_SERVER_ERROR",
    "EXIT_CONTRACT_MISMATCH",
    "SyncError",
    "SyncValidationError",
    "SyncAuthError",
    "SyncNetworkError",
    "SyncServerError",
    "SyncContractMismatchError",
    "SealedStudyWire",
    "SyncRunPayload",
    "SyncIdeaPayload",
    "SyncTrialEventPayload",
    "PushRequest",
    "PushItemAck",
    "PushResponse",
    "PullRequest",
    "PullResponse",
    "compute_push_exit_code",
    "export_json_schemas",
    "remote_add",
    "resolve_remote_url",
    "build_sync_payload",
    "format_push_preview",
    "push_workspace",
    "pull_workspace",
    "_guard_sync_payload",
]

_SYNC_PUSH_PATH = "/api/workspace/sync"
_SYNC_PULL_PATH = "/api/workspace/pull"
_FORBIDDEN_KEYS = frozenset(
    {
        "raw_rows",
        "dataset_rows",
        "prices",
        "matrix",
        "repro_spec",
        "cumulative_trials",
        "project_trials",
        "kill_if",
    }
)


def resolve_remote_url(
    proj: Project | None,
    remote_name: str = "quantos",
    explicit_url: str | None = None,
) -> str:
    """Resolve and validate the remote URL from explicit flag, `project.toml`, env, or default."""
    from ..auth import validate_clean_url

    if explicit_url and explicit_url.strip():
        return validate_clean_url(explicit_url.strip()).rstrip("/")
    if proj is not None and remote_name in proj.remotes and proj.remotes[remote_name]:
        return validate_clean_url(proj.remotes[remote_name]).rstrip("/")
    env_url = os.environ.get(API_URL_ENV) or os.environ.get(LEGACY_API_URL_ENV)
    if env_url and env_url.strip():
        return validate_clean_url(env_url.strip()).rstrip("/")
    return validate_clean_url(DEFAULT_API_URL).rstrip("/")


def remote_add(proj: Project, name: str = "quantos", url: str | None = None) -> str:
    """Register or update a clean named remote in `.alphaengine/project.toml` (`QOS-155`)."""
    clean_name = (name or "quantos").strip()
    resolved = resolve_remote_url(proj, remote_name=clean_name, explicit_url=url)
    proj.remotes[clean_name] = resolved
    proj.save_config()
    return resolved


def _sanitize_figures(node: Any, *, share_series: bool) -> tuple[Any, dict[str, list[float]]]:
    """Sanitize figures for wire transmission and extract opt-in series when `share_series=True`."""
    extracted_series: dict[str, list[float]] = {}

    def _walk(item: Any, key_hint: str) -> Any:
        if isinstance(item, Mapping):
            out: dict[str, Any] = {}
            for k, v in item.items():
                ks = str(k)
                if ks in _FORBIDDEN_KEYS:
                    continue
                out[ks] = _walk(v, f"{key_hint}.{ks}" if key_hint else ks)
            return out
        if isinstance(item, (list, tuple)):
            if len(item) > MAX_FIGURE_LIST:
                if share_series:
                    if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in item):
                        extracted_series[key_hint or "series"] = [float(x) for x in item]
                    return [_walk(v, key_hint) for v in item]
                return []
            return [_walk(v, key_hint) for v in item]
        return item

    cleaned = _walk(node, "")
    return cleaned, extracted_series


def _guard_sync_payload(payload: dict[str, Any], *, share_series: bool = False) -> dict[str, Any]:
    """Enforce that no forbidden raw-data keys or `> 512` element lists leave without opt-in."""

    def walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for k, v in node.items():
                ks = str(k)
                if ks in _FORBIDDEN_KEYS:
                    raise ValueError(f"refusing to sync forbidden field {path}.{ks}")
                walk(v, f"{path}.{ks}")
        elif isinstance(node, (list, tuple)):
            if not share_series and len(node) > MAX_FIGURE_LIST:
                raise ValueError(
                    f"refusing to send {path}: {len(node)} elements is a series (> {MAX_FIGURE_LIST}). "
                    "Pass --share-series to opt in."
                )
            for idx, item in enumerate(node):
                walk(item, f"{path}[{idx}]")

    walk(payload, "sync")
    return payload


def build_sync_payload(proj: Project, *, share_series: bool = False) -> dict[str, Any]:
    """Assemble the validated `PushRequest` v2 dictionary (`QOS-152`, `QOS-161`–`QOS-166`)."""
    share = share_series_enabled(share_series)
    eng_ver = current_engine_version()

    # 1. Sealed studies in `.alphaengine/studies/*.json` (Fix 1/7: exact sealed projection)
    studies_out: list[dict[str, Any]] = []
    if proj.studies_dir.is_dir():
        for sp in sorted(proj.studies_dir.glob("*.json")):
            if sp.name == "HEAD.json":
                continue
            st = load(sp)
            wire_study = SealedStudyWire.from_study(st)
            studies_out.append(wire_study.to_dict())

    # 2. Local ledger runs (exclude `read_only` pulled runs; Fix 5/7 & 6/7)
    raw_runs = proj.read_jsonl("ledger/runs.jsonl", strict=True)
    local_run_ids: set[str] = set()
    runs_out: list[SyncRunPayload] = []
    for r in raw_runs:
        if r.get("read_only"):
            continue
        rid = str(r.get("run_id") or "").strip()
        if not rid:
            continue
        local_run_ids.add(rid)
        raw_figs = r.get("figures") if isinstance(r.get("figures"), Mapping) else {}
        cleaned_figs, series_map = _sanitize_figures(raw_figs, share_series=share)
        metrics_map = extract_optional_metrics(raw_figs)
        cum_n = r.get("cumulative_trials") if r.get("cumulative_trials") is not None else r.get("n_trials")
        run_payload = SyncRunPayload(
            run_id=rid,
            timestamp=str(r.get("timestamp") or ""),
            command=str(r.get("command") or "run"),
            idea_id=str(r["idea_id"]) if r.get("idea_id") else None,
            family=str(r.get("family") or "ad_hoc"),
            params_hash=str(r.get("params_hash") or ""),
            dataset_hashes=[str(h) for h in (r.get("dataset_hashes") or []) if h],
            n_trials=int(cum_n)
            if (cum_n is not None and r.get("n_trials_source") != "not_recorded")
            else None,
            run_n_trials=int(r["run_n_trials"]) if r.get("run_n_trials") is not None else None,
            n_trials_source=str(r["n_trials_source"]) if r.get("n_trials_source") else None,
            verdict=str(r["verdict"]) if r.get("verdict") else None,
            engine_version=str(r.get("engine_version") or eng_ver),
            object_sha256=str(r["object_sha256"]) if r.get("object_sha256") else None,
            manifest=dict(r["manifest"]) if isinstance(r.get("manifest"), Mapping) else None,
            share_series=share,
            metrics=metrics_map,
            figures=cleaned_figs if isinstance(cleaned_figs, dict) else {},
            series=series_map if (share and series_map) else None,
        )
        runs_out.append(run_payload)

    # 3. Local ideas (Fix 3/7: canonical `idea_id` + `kill_rule`)
    ideas_out: list[SyncIdeaPayload] = []
    for idea in list_ideas(proj):
        if idea.get("read_only"):
            continue
        iid = str(idea.get("idea_id") or "").strip()
        thesis = str(idea.get("thesis") or "").strip()
        if not iid or not thesis:
            continue
        kr = str(
            idea.get("kill_rule") if idea.get("kill_rule") is not None else (idea.get("kill_if") or "")
        ).strip()
        ideas_out.append(
            SyncIdeaPayload(
                idea_id=iid,
                thesis=thesis,
                kill_rule=kr,
                family=str(idea.get("family") or "general"),
                status=str(idea.get("status") or "open"),
                created_at=str(idea.get("created_at") or ""),
                updated_at=str(idea.get("updated_at") or ""),
                object_sha256=str(idea["object_sha256"]) if idea.get("object_sha256") else None,
            )
        )

    # 4. Idempotent per-attempt `trial_events` (Fix 4/7: replaces cumulative counters)
    raw_trials = proj.read_jsonl("ledger/trials.jsonl", strict=True)
    seen_attempt_ids: set[str] = set()
    runs_with_trial_rows: set[str] = set()
    trial_events_out: list[SyncTrialEventPayload] = []

    for tr in raw_trials:
        if tr.get("read_only"):
            continue
        tr_run_id = str(tr.get("run_id") or "").strip()
        if tr_run_id and local_run_ids and tr_run_id not in local_run_ids:
            continue
        aid = str(tr.get("attempt_id") or "").strip()
        if not aid or aid in seen_attempt_ids:
            continue
        seen_attempt_ids.add(aid)
        if tr_run_id:
            runs_with_trial_rows.add(tr_run_id)
        status_raw = "failed" if tr.get("failed") is not None else str(tr.get("status") or "ok")
        status_norm = "failed" if status_raw == "failed" else "ok"
        trial_events_out.append(
            SyncTrialEventPayload(
                attempt_id=aid,
                run_id=tr_run_id or "standalone",
                family=str(tr.get("family") or "default"),
                idea_id=str(tr["idea_id"]) if tr.get("idea_id") else None,
                params_hash=str(tr.get("params_hash") or ""),
                status=status_norm,
                executed_at=str(tr.get("executed_at") or tr.get("timestamp") or ""),
            )
        )

    # Single-attempt local runs without separate rows in `trials.jsonl` also emit their `attempt_id`
    for r in raw_runs:
        if r.get("read_only"):
            continue
        rid = str(r.get("run_id") or "").strip()
        if not rid or rid in runs_with_trial_rows:
            continue
        src = str(r.get("n_trials_source") or "")
        if src in ("monte_carlo", "not_recorded"):
            continue
        run_nt = r.get("run_n_trials") if r.get("run_n_trials") is not None else r.get("n_trials")
        if not isinstance(run_nt, int) or isinstance(run_nt, bool) or run_nt <= 0:
            continue
        base_aid = str(r.get("attempt_id") or uuid.uuid5(uuid.NAMESPACE_URL, f"alphaengine:{rid}")).strip()
        if run_nt == 1:
            if base_aid not in seen_attempt_ids:
                seen_attempt_ids.add(base_aid)
                trial_events_out.append(
                    SyncTrialEventPayload(
                        attempt_id=base_aid,
                        run_id=rid,
                        family=str(r.get("family") or "ad_hoc"),
                        idea_id=str(r["idea_id"]) if r.get("idea_id") else None,
                        params_hash=str(r.get("params_hash") or ""),
                        status="ok",
                        executed_at=str(r.get("timestamp") or ""),
                    )
                )
        else:
            for idx in range(run_nt):
                synth_aid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"alphaengine:{rid}:{idx}"))
                if synth_aid in seen_attempt_ids:
                    continue
                seen_attempt_ids.add(synth_aid)
                trial_events_out.append(
                    SyncTrialEventPayload(
                        attempt_id=synth_aid,
                        run_id=rid,
                        family=str(r.get("family") or "ad_hoc"),
                        idea_id=str(r["idea_id"]) if r.get("idea_id") else None,
                        params_hash=str(r.get("params_hash") or ""),
                        status="ok",
                        executed_at=str(r.get("timestamp") or ""),
                    )
                )

    push_req = PushRequest(
        project={
            "name": proj.name,
            "schema_version": proj.schema_version,
            "engine_version": eng_ver,
        },
        studies=studies_out,
        runs=runs_out,
        ideas=ideas_out,
        trial_events=trial_events_out,
        sync_contract=SYNC_CONTRACT_VERSION,
        engine_version=eng_ver,
    )
    payload_dict = push_req.to_dict()
    return _guard_sync_payload(payload_dict, share_series=share)


def format_push_preview(
    proj: Project,
    payload: Mapping[str, Any],
    *,
    endpoint: str,
    remote_name: str = "quantos",
    show_payload: bool = False,
) -> str:
    """Format a human-readable `ae push` boundary & payload preview (`QOS-168`)."""
    from ..auth import redact_secrets

    studies = payload.get("studies") or []
    runs = payload.get("runs") or []
    ideas = payload.get("ideas") or []
    trial_events = payload.get("trial_events") or []
    any_series = any(bool(isinstance(r, Mapping) and r.get("share_series")) for r in runs)
    lines = [
        f"  push preview  ->  {remote_name} ({endpoint})",
        f"    project          {proj.name}",
        f"    sync_contract    {payload.get('sync_contract', SYNC_CONTRACT_VERSION)}  "
        f"(engine {payload.get('engine_version', current_engine_version())})",
        f"    items            {len(studies)} studies, {len(runs)} runs, "
        f"{len(ideas)} ideas, {len(trial_events)} trial_events",
        "    leaving machine  summary figures & metrics, dataset SHA-256 digests, "
        f"hypothesis text & kill rules, trial attempt IDs, share_series={'yes' if any_series else 'no'}",
        "    staying local    Code and raw datasets stay local.",
    ]
    if show_payload:
        redacted_json = redact_secrets(json.dumps(dict(payload), indent=2, sort_keys=True, default=str))
        lines.append("")
        lines.append(redacted_json)
    return "\n".join(lines)


def _extract_http_detail(err: urllib.error.HTTPError) -> tuple[str, dict[str, Any]]:
    from ..auth import redact_secrets

    raw_body = ""
    with contextlib.suppress(Exception):
        raw_body = err.read().decode("utf-8", errors="replace")
    parsed: dict[str, Any] = {}
    if raw_body:
        with contextlib.suppress(Exception):
            obj = json.loads(raw_body)
            if isinstance(obj, dict):
                parsed = obj
    detail = parsed.get("detail") or parsed.get("message") or parsed.get("error") or raw_body[:240]
    return redact_secrets(str(detail or "").strip()), parsed


def push_workspace(
    proj: Project,
    *,
    remote_name: str = "quantos",
    api_key: str | None = None,
    base_url: str | None = None,
    share_series: bool = False,
    dry_run: bool = False,
    show_payload: bool = False,
    timeout: int = 30,
) -> dict[str, Any]:
    """Push studies, runs, ideas, and `trial_events` to `POST /api/workspace/sync` (`QOS-161`–`QOS-167`).

    Never falls back silently to `/api/me/runs/ingest`. Raises typed `SyncError`
    subclasses carrying structured `exit_code` (`4`, `5`, `6`), or returns a result
    dict carrying per-item `items`, `counts`, and `exit_code` (`0`, `2`, `3`).
    """
    payload = build_sync_payload(proj, share_series=share_series)
    root = resolve_remote_url(proj, remote_name=remote_name, explicit_url=base_url)
    endpoint = f"{root}{_SYNC_PUSH_PATH}"
    preview = format_push_preview(
        proj,
        payload,
        endpoint=endpoint,
        remote_name=remote_name,
        show_payload=show_payload,
    )

    if dry_run:
        return {
            "dry_run": True,
            "remote": remote_name,
            "endpoint": endpoint,
            "sync_contract": SYNC_CONTRACT_VERSION,
            "engine_version": payload["engine_version"],
            "preview": preview,
            "payload": payload,
            "exit_code": EXIT_OK,
        }

    from ..auth import FirmKeyError, redact_secrets, require_firm_key

    try:
        key = require_firm_key(api_key)
    except FirmKeyError as exc:
        raise SyncAuthError(f"No API key. {exc}", exit_code=EXIT_AUTH_ERROR, status_code=401) from exc

    submitted_total = (
        len(payload["studies"]) + len(payload["runs"]) + len(payload["ideas"]) + len(payload["trial_events"])
    )
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {key.strip()}",
        "User-Agent": f"alphaengine/{current_engine_version()}",
        "X-AlphaEngine-Sync-Contract": str(SYNC_CONTRACT_VERSION),
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_resp = json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as err:
        detail, parsed_err = _extract_http_detail(err)
        suffix = f" {detail}" if detail else ""
        if err.code in (401, 403):
            raise SyncAuthError(
                f"Authentication/entitlement rejected ({err.code}).{suffix}",
                status_code=err.code,
            ) from err
        if err.code in (409, 426) or (
            isinstance(parsed_err.get("sync_contract"), int)
            and parsed_err.get("sync_contract") != SYNC_CONTRACT_VERSION
        ):
            raise SyncContractMismatchError(
                f"Sync contract mismatch ({err.code}): "
                f"client uses sync_contract={SYNC_CONTRACT_VERSION}.{suffix}",
                status_code=err.code,
            ) from err
        raise SyncServerError(
            f"POST {_SYNC_PUSH_PATH} failed with HTTP {err.code}.{suffix}",
            status_code=err.code,
        ) from err
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        reason = getattr(err, "reason", err)
        raise SyncNetworkError(f"Could not reach {root}: {redact_secrets(str(reason))}.") from err

    if not isinstance(raw_resp, Mapping):
        raise SyncServerError("Invalid non-object JSON response from /api/workspace/sync.")

    parsed = PushResponse.from_dict(raw_resp, submitted_count=submitted_total)
    counts = parsed.counts
    exit_code = parsed.exit_code
    rejected_items = [it.to_dict() for it in parsed.items if it.status == "rejected"]

    return {
        "ok": exit_code == EXIT_OK,
        "remote": remote_name,
        "endpoint": endpoint,
        "path_used": _SYNC_PUSH_PATH,
        "sync_contract": parsed.sync_contract,
        "server_engine_version": parsed.engine_version,
        "preview": preview,
        "items": [it.to_dict() for it in parsed.items],
        "counts": counts,
        "rejected": rejected_items,
        "exit_code": exit_code,
    }


def pull_workspace(
    proj: Project,
    *,
    remote_name: str = "quantos",
    api_key: str | None = None,
    base_url: str | None = None,
    since: str | None = None,
    limit: int = 100,
    max_pages: int = 50,
    timeout: int = 30,
) -> dict[str, Any]:
    """Pull team runs, ideas, and studies via `GET /api/workspace/pull` with cursor pagination (`QOS-162`).

    - Never silently falls back to `/api/harness/runs`.
    - Updates per-remote cursor in `.alphaengine/project.toml` (`[sync_cursors]`).
    - Upserts revised ideas by `idea_id` without duplicating rows (`QOS-163`).
    - Preserves `Optional[float] = None` metrics without inventing defaults (`QOS-165`).
    - Marks pulled items `read_only: True` (`QOS-168`).
    """
    from ..auth import FirmKeyError, redact_secrets, require_firm_key

    try:
        key = require_firm_key(api_key)
    except FirmKeyError as exc:
        raise SyncAuthError(f"No API key. {exc}", exit_code=EXIT_AUTH_ERROR, status_code=401) from exc

    root = resolve_remote_url(proj, remote_name=remote_name, explicit_url=base_url)
    cursor: str | None = since if since is not None else proj.get_sync_cursor(remote_name)
    last_cursor: str | None = cursor

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {key.strip()}",
        "User-Agent": f"alphaengine/{current_engine_version()}",
        "X-AlphaEngine-Sync-Contract": str(SYNC_CONTRACT_VERSION),
    }

    fetched_runs: list[dict[str, Any]] = []
    fetched_ideas: list[dict[str, Any]] = []
    fetched_studies: list[dict[str, Any]] = []
    server_engine_version = "unknown"
    pages = 0

    while pages < max_pages:
        pages += 1
        pull_req = PullRequest(since=cursor, limit=limit)
        qs = urllib.parse.urlencode(pull_req.to_query_params())
        url = f"{root}{_SYNC_PULL_PATH}?{qs}"
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_page = json.loads(resp.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as err:
            detail, parsed_err = _extract_http_detail(err)
            suffix = f" {detail}" if detail else ""
            if err.code in (401, 403):
                raise SyncAuthError(
                    f"Authentication/entitlement rejected ({err.code}).{suffix}",
                    status_code=err.code,
                ) from err
            if err.code in (409, 426) or (
                isinstance(parsed_err.get("sync_contract"), int)
                and parsed_err.get("sync_contract") != SYNC_CONTRACT_VERSION
            ):
                raise SyncContractMismatchError(
                    f"Sync contract mismatch ({err.code}): "
                    f"client uses sync_contract={SYNC_CONTRACT_VERSION}.{suffix}",
                    status_code=err.code,
                ) from err
            raise SyncServerError(
                f"GET {_SYNC_PULL_PATH} failed with HTTP {err.code}.{suffix}",
                status_code=err.code,
            ) from err
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            reason = getattr(err, "reason", err)
            raise SyncNetworkError(f"Could not reach {root}: {redact_secrets(str(reason))}.") from err

        if not isinstance(raw_page, Mapping):
            raise SyncServerError("Invalid non-object JSON response from /api/workspace/pull.")

        page = PullResponse.from_dict(raw_page)
        server_engine_version = page.engine_version
        fetched_runs.extend(page.runs)
        fetched_ideas.extend(page.ideas)
        fetched_studies.extend(page.studies)

        if page.next_cursor and page.next_cursor != cursor:
            last_cursor = page.next_cursor
            cursor = page.next_cursor
        else:
            break

    # Apply runs (append new, update changed read-only runs, skip identical/local runs)
    existing_runs = proj.read_jsonl("ledger/runs.jsonl", strict=True)
    run_index_by_id: dict[str, int] = {}
    for idx, r_row in enumerate(existing_runs):
        rid_existing = str(r_row.get("run_id") or "")
        if rid_existing:
            run_index_by_id[rid_existing] = idx

    run_counts = {"new": 0, "updated": 0, "skipped": 0, "failed": 0}
    runs_mutated = False

    for r in fetched_runs:
        rid = str(r.get("run_id") or "").strip()
        if not rid:
            run_counts["failed"] += 1
            continue
        raw_figs = dict(r.get("figures") or {}) if isinstance(r.get("figures"), Mapping) else {}
        raw_metrics = r.get("metrics") if isinstance(r.get("metrics"), Mapping) else None
        metrics_clean = (
            {
                str(k): (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None)
                for k, v in raw_metrics.items()
            }
            if raw_metrics is not None
            else extract_optional_metrics(raw_figs)
        )
        raw_verdict = r.get("verdict")
        norm_v: str | None = None
        if raw_verdict:
            try:
                norm_v = normalize_verdict(str(raw_verdict))
            except ValueError:
                norm_v = str(raw_verdict)
        nt_raw = r.get("n_trials")
        nt_val = int(nt_raw) if isinstance(nt_raw, int) and not isinstance(nt_raw, bool) else None

        entry: dict[str, Any] = {
            "run_id": rid,
            "timestamp": str(r.get("timestamp") or ""),
            "command": str(r.get("command") or "remote_run"),
            "idea_id": str(r["idea_id"]) if r.get("idea_id") else None,
            "family": str(r.get("family") or "remote"),
            "params": {},
            "params_hash": str(r.get("params_hash") or ""),
            "dataset_hashes": [str(h) for h in (r.get("dataset_hashes") or []) if h],
            "n_trials": nt_val,
            "run_n_trials": int(r["run_n_trials"])
            if isinstance(r.get("run_n_trials"), int) and not isinstance(r.get("run_n_trials"), bool)
            else nt_val,
            "n_trials_source": str(r.get("n_trials_source") or "remote"),
            "cumulative_trials": nt_val or 0,
            "project_trials": 0,
            "metrics": metrics_clean,
            "figures": raw_figs,
            "verdict": norm_v,
            "engine_version": str(r.get("engine_version") or server_engine_version),
            "read_only": True,
            "source": "remote",
        }
        if rid in run_index_by_id:
            pos = run_index_by_id[rid]
            prev_run = existing_runs[pos]
            if not prev_run.get("read_only"):
                run_counts["skipped"] += 1
                continue
            if prev_run == entry:
                run_counts["skipped"] += 1
                continue
            existing_runs[pos] = entry
            runs_mutated = True
            run_counts["updated"] += 1
        else:
            run_index_by_id[rid] = len(existing_runs)
            existing_runs.append(entry)
            runs_mutated = True
            run_counts["new"] += 1

    if runs_mutated:
        proj.write_jsonl("ledger/runs.jsonl", existing_runs)

    # Apply ideas via `upsert_idea_revision` (Fix 3/7: updates same `idea_id`, no duplicate rows)
    idea_counts = {"new": 0, "updated": 0, "skipped": 0, "failed": 0}
    for raw_idea in fetched_ideas:
        try:
            parsed_idea = SyncIdeaPayload.from_dict(raw_idea, strict_wire=False)
        except SyncValidationError:
            idea_counts["failed"] += 1
            continue
        action, _ = upsert_idea_revision(
            proj,
            parsed_idea.to_dict(),
            read_only=True,
            source="remote",
        )
        if action in ("new", "inserted"):
            idea_counts["new"] += 1
        elif action == "updated":
            idea_counts["updated"] += 1
        else:
            idea_counts["skipped"] += 1

    # Apply sealed studies (verify seal before saving locally)
    study_counts = {"new": 0, "updated": 0, "skipped": 0, "failed": 0}
    for raw_st in fetched_studies:
        try:
            wire_st = SealedStudyWire.from_dict(raw_st)
            st_obj = Study.from_dict(wire_st.to_dict())
        except Exception:  # noqa: BLE001
            study_counts["failed"] += 1
            continue
        safe_id = wire_st.content_sha256[:16]
        target_path = proj.studies_dir / f"remote_{safe_id}.json"
        if target_path.is_file():
            study_counts["skipped"] += 1
            continue
        save(st_obj, target_path)
        study_counts["new"] += 1

    if last_cursor and last_cursor != proj.get_sync_cursor(remote_name):
        proj.set_sync_cursor(remote_name, last_cursor)

    return {
        "ok": True,
        "remote": remote_name,
        "path_used": _SYNC_PULL_PATH,
        "sync_contract": SYNC_CONTRACT_VERSION,
        "server_engine_version": server_engine_version,
        "pages": pages,
        "cursor": proj.get_sync_cursor(remote_name),
        "counts": {
            "runs": run_counts,
            "ideas": idea_counts,
            "studies": study_counts,
        },
        "pulled_runs": run_counts["new"] + run_counts["updated"],
        "pulled_ideas": idea_counts["new"] + idea_counts["updated"],
        "pulled_studies": study_counts["new"] + study_counts["updated"],
        "read_only_notice": (
            "Pulled records are marked read_only (summary figures & metadata only; "
            "code and raw datasets stay local and are not fetched)."
        ),
        "exit_code": EXIT_OK,
    }
