"""Opt-in workspace sync (`ae remote`, `ae push`, `ae pull`).

Synchronizes sealed studies, allowlisted ledger run records, and hypothesis
objects with a remote quantOS workspace (`POST /api/workspace/sync`), falling
back to `POST /api/me/runs/ingest` when `/api/workspace/sync` returns 404.

HARD DATA BOUNDARY
- Raw dataset files and dataset rows are NEVER included in sync payloads.
- Full series (> 512 elements) are refused by `_guard_sync_payload` unless the
  caller explicitly opts in with `--share-series` or `ALPHAENGINE_SHARE_SERIES=1`.
- `ae push --dry-run` prints the exact JSON payload without opening a socket.
"""

from __future__ import annotations

import contextlib
import json
import os
from typing import Any

from ._version import API_URL_ENV, DEFAULT_API_URL, LEGACY_API_URL_ENV, __version__
from .client.session import share_series_enabled
from .ideas import list_ideas
from .project import Project
from .study import load
from .study.report import MAX_FIGURE_LIST, ReportError, _payload

__all__ = [
    "remote_add",
    "resolve_remote_url",
    "build_sync_payload",
    "push_workspace",
    "pull_workspace",
    "_guard_sync_payload",
]

_ENV_KEY = "QUANTOS_API_KEY"
_SYNC_PATH = "/api/workspace/sync"
_INGEST_PATH = "/api/me/runs/ingest"
_FORBIDDEN_KEYS = frozenset({"raw_rows", "dataset_rows", "prices", "matrix", "repro_spec"})


def resolve_remote_url(proj: Project, remote_name: str = "quantos", explicit_url: str | None = None) -> str:
    """Resolve the remote URL from explicit flag, project.toml `[remotes]`, env vars, or default."""
    if explicit_url and explicit_url.strip():
        return explicit_url.strip().rstrip("/")
    if remote_name in proj.remotes and proj.remotes[remote_name]:
        return proj.remotes[remote_name].rstrip("/")
    env_url = os.environ.get(API_URL_ENV) or os.environ.get(LEGACY_API_URL_ENV)
    if env_url and env_url.strip():
        return env_url.strip().rstrip("/")
    return DEFAULT_API_URL.rstrip("/")


def remote_add(proj: Project, name: str = "quantos", url: str | None = None) -> str:
    """Register or update a named remote in `.alphaengine/project.toml`."""
    clean_name = (name or "quantos").strip()
    resolved = resolve_remote_url(proj, remote_name=clean_name, explicit_url=url)
    proj.remotes[clean_name] = resolved
    proj.save_config()
    return resolved


def _sanitize_figures(node: Any, *, share_series: bool) -> Any:
    if isinstance(node, dict):
        return {
            str(k): _sanitize_figures(v, share_series=share_series)
            for k, v in node.items()
            if str(k) not in _FORBIDDEN_KEYS
        }
    if isinstance(node, (list, tuple)):
        if not share_series and len(node) > MAX_FIGURE_LIST:
            return []
        return [_sanitize_figures(v, share_series=share_series) for v in node]
    return node


def _guard_sync_payload(payload: dict[str, Any], *, share_series: bool = False) -> dict[str, Any]:
    """Enforce that no forbidden raw-data keys or >512 element lists leave the machine without opt-in."""

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if str(k) in _FORBIDDEN_KEYS:
                    raise ValueError(f"refusing to sync forbidden field {path}.{k}")
                walk(v, f"{path}.{k}")
        elif isinstance(node, (list, tuple)):
            if not share_series and len(node) > MAX_FIGURE_LIST:
                raise ValueError(
                    f"refusing to send {path}: {len(node)} elements is a series (> {MAX_FIGURE_LIST}). "
                    "Pass --share-series to opt in."
                )
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]")

    walk(payload, "sync")
    return payload


def build_sync_payload(proj: Project, *, share_series: bool = False) -> dict[str, Any]:
    """Assemble the allowlisted sync payload: sealed studies, ledger run records, and ideas."""
    share = share_series_enabled(share_series)

    # 1. Sealed studies in `.alphaengine/studies/*.json`
    studies_out: list[dict[str, Any]] = []
    if proj.studies_dir.is_dir():
        for sp in sorted(proj.studies_dir.glob("*.json")):
            try:
                st = load(sp)
                studies_out.append(_payload(st))
            except Exception:  # noqa: BLE001
                continue

    # 2. Local ledger runs (exclude read_only pulled runs and strip repro_spec/raw data)
    runs_out: list[dict[str, Any]] = []
    for r in proj.read_jsonl("ledger/runs.jsonl"):
        if r.get("read_only"):
            continue
        cum_n = r.get("cumulative_trials") if r.get("cumulative_trials") is not None else r.get("n_trials")
        clean_run = {
            "run_id": r.get("run_id"),
            "timestamp": r.get("timestamp"),
            "command": r.get("command"),
            "idea_id": r.get("idea_id"),
            "family": r.get("family"),
            "params_hash": r.get("params_hash"),
            "dataset_hashes": list(r.get("dataset_hashes") or []),
            "n_trials": cum_n if r.get("n_trials_source") != "not_recorded" else None,
            "run_n_trials": r.get("run_n_trials"),
            "n_trials_source": r.get("n_trials_source"),
            "cumulative_trials": cum_n,
            "project_trials": r.get("project_trials"),
            "duplicate_of": r.get("duplicate_of"),
            "verdict": r.get("verdict"),
            "engine_version": r.get("engine_version") or __version__,
            "object_sha256": r.get("object_sha256"),
            "figures": _sanitize_figures(r.get("figures") or {}, share_series=share),
        }
        runs_out.append(clean_run)

    # 3. Local ideas (exclude read_only pulled ideas)
    ideas_out: list[dict[str, Any]] = []
    for idea in list_ideas(proj):
        if idea.get("read_only"):
            continue
        ideas_out.append(
            {
                "idea_id": idea.get("idea_id"),
                "thesis": idea.get("thesis"),
                "kill_if": idea.get("kill_if"),
                "family": idea.get("family"),
                "status": idea.get("status"),
                "created_at": idea.get("created_at"),
                "updated_at": idea.get("updated_at"),
                "object_sha256": idea.get("object_sha256"),
            }
        )

    payload: dict[str, Any] = {
        "project": {
            "name": proj.name,
            "schema_version": proj.schema_version,
            "engine_version": __version__,
        },
        "studies": studies_out,
        "runs": runs_out,
        "ideas": ideas_out,
    }
    return _guard_sync_payload(payload, share_series=share)


def push_workspace(
    proj: Project,
    *,
    remote_name: str = "quantos",
    api_key: str | None = None,
    base_url: str | None = None,
    share_series: bool = False,
    dry_run: bool = False,
    timeout: int = 30,
) -> dict[str, Any]:
    """Push studies, runs, and ideas to `/api/workspace/sync` (404 falls back to `/api/me/runs/ingest`)."""
    payload = build_sync_payload(proj, share_series=share_series)
    root = resolve_remote_url(proj, remote_name=remote_name, explicit_url=base_url)

    if dry_run:
        return {
            "dry_run": True,
            "endpoint": f"{root}{_SYNC_PATH}",
            "payload": payload,
        }

    import urllib.error
    import urllib.request

    from .auth import FirmKeyError, redact_secrets, require_firm_key

    try:
        key = require_firm_key(api_key)
    except FirmKeyError as exc:
        raise ReportError(f"No API key. {exc}") from exc

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key.strip()}",
        "User-Agent": f"alphaengine/{__version__}",
    }
    req = urllib.request.Request(
        f"{root}{_SYNC_PATH}",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            recorded = json.loads(resp.read().decode("utf-8") or "{}")
            return {
                "ok": True,
                "path_used": _SYNC_PATH,
                "fallback": False,
                "n_studies": len(payload["studies"]),
                "n_runs": len(payload["runs"]),
                "n_ideas": len(payload["ideas"]),
                "response": recorded,
            }
    except urllib.error.HTTPError as e:
        if e.code in (404, 405):
            # Fallback to `/api/me/runs/ingest` for studies
            studies_to_send = list(payload["studies"])
            if not studies_to_send and payload["runs"]:
                from .memo import seal_from_project

                with contextlib.suppress(Exception):
                    st, _ = seal_from_project(proj, "HEAD")
                    studies_to_send.append(_payload(st))
            ingested: list[Any] = []
            for st_body in studies_to_send:
                ing_req = urllib.request.Request(
                    f"{root}{_INGEST_PATH}",
                    data=json.dumps(st_body).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(ing_req, timeout=timeout) as resp2:
                        ingested.append(json.loads(resp2.read().decode("utf-8") or "{}"))
                except urllib.error.HTTPError as e2:
                    detail2 = ""
                    with contextlib.suppress(Exception):
                        detail2 = json.loads(e2.read().decode("utf-8")).get("detail", "")
                    raise ReportError(
                        f"Fallback {_INGEST_PATH} refused study ({e2.code}). {detail2}".strip()
                    ) from e2
            return {
                "ok": True,
                "path_used": _INGEST_PATH,
                "fallback": True,
                "n_studies": len(ingested),
                "n_runs": 0,
                "n_ideas": 0,
                "response": {"ingested": ingested},
            }
        detail = ""
        with contextlib.suppress(Exception):
            detail = redact_secrets(str(json.loads(e.read().decode("utf-8")).get("detail", "")))
        if e.code in (401, 403):
            raise ReportError(f"The key was rejected ({e.code}). {detail}".strip()) from e
        raise ReportError(f"The server refused sync ({e.code}). {detail}".strip()) from e
    except urllib.error.URLError as e:
        raise ReportError(f"Could not reach {root}: {redact_secrets(str(e.reason))}.") from e


def pull_workspace(
    proj: Project,
    *,
    remote_name: str = "quantos",
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Pull team runs and ideas into the local ledger, marked `read_only: True`."""
    import urllib.error
    import urllib.request

    from .auth import FirmKeyError, require_firm_key

    try:
        key = require_firm_key(api_key)
    except FirmKeyError as exc:
        raise ReportError(f"No API key. {exc}") from exc

    root = resolve_remote_url(proj, remote_name=remote_name, explicit_url=base_url)
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {key.strip()}",
        "User-Agent": f"alphaengine/{__version__}",
    }
    req = urllib.request.Request(f"{root}{_SYNC_PATH}", headers=headers, method="GET")

    remote_runs: list[Any] = []
    remote_ideas: list[Any] = []
    path_used = _SYNC_PATH

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8") or "{}")
            remote_runs = list(data.get("runs") or [])
            remote_ideas = list(data.get("ideas") or [])
    except urllib.error.HTTPError as e:
        if e.code in (404, 405):
            path_used = "/api/harness/runs"
            req_runs = urllib.request.Request(
                f"{root}/api/harness/runs?limit=25", headers=headers, method="GET"
            )
            with urllib.request.urlopen(req_runs, timeout=timeout) as resp_r:
                r_data = json.loads(resp_r.read().decode("utf-8") or "{}")
                remote_runs = list(r_data.get("runs") or [])
        else:
            raise ReportError(f"Could not pull workspace ({e.code}).") from e
    except urllib.error.URLError as e:
        raise ReportError(f"Could not reach {root}: {e.reason}.") from e

    existing_run_ids = {str(r.get("run_id")) for r in proj.read_jsonl("ledger/runs.jsonl")}
    existing_idea_ids = {str(i.get("idea_id")) for i in list_ideas(proj)}

    added_runs = 0
    for r in remote_runs:
        if not isinstance(r, dict):
            continue
        rid = str(r.get("run_id") or r.get("id") or "")
        if not rid or rid in existing_run_ids:
            continue
        entry = {
            "run_id": rid,
            "timestamp": str(r.get("timestamp") or r.get("created_at") or ""),
            "command": str(r.get("command") or r.get("kind") or "remote_run"),
            "idea_id": r.get("idea_id"),
            "family": str(r.get("family") or r.get("kind") or "remote"),
            "params": {},
            "params_hash": str(r.get("params_hash") or ""),
            "dataset_hashes": list(r.get("dataset_hashes") or []),
            "n_trials": r.get("n_trials") if r.get("n_trials") is not None else r.get("honest_n"),
            "n_trials_source": str(r.get("n_trials_source") or "remote"),
            "cumulative_trials": int(
                r.get("cumulative_trials") or r.get("n_trials") or r.get("honest_n") or 0
            ),
            "project_trials": int(r.get("project_trials") or 0),
            "figures": dict(r.get("figures") or {}),
            "verdict": r.get("verdict") or r.get("status"),
            "engine_version": str(r.get("engine_version") or __version__),
            "read_only": True,
            "source": "remote",
        }
        proj.append_jsonl("ledger/runs.jsonl", entry)
        existing_run_ids.add(rid)
        added_runs += 1

    added_ideas = 0
    for idea in remote_ideas:
        if not isinstance(idea, dict):
            continue
        iid = str(idea.get("idea_id") or idea.get("id") or "")
        if not iid or iid in existing_idea_ids:
            continue
        entry_i = {
            "idea_id": iid,
            "thesis": str(idea.get("thesis") or idea.get("statement") or idea.get("name") or ""),
            "kill_if": str(idea.get("kill_if") or ""),
            "family": str(idea.get("family") or "remote"),
            "status": str(idea.get("status") or "open"),
            "created_at": str(idea.get("created_at") or ""),
            "updated_at": str(idea.get("updated_at") or ""),
            "read_only": True,
            "source": "remote",
        }
        proj.append_jsonl("ideas/ideas.jsonl", entry_i)
        existing_idea_ids.add(iid)
        added_ideas += 1

    return {
        "ok": True,
        "path_used": path_used,
        "pulled_runs": added_runs,
        "pulled_ideas": added_ideas,
    }
