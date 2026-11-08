"""Send a study to a QuantOS workspace.

THE ONE THING THIS MODULE IS FOR. A study is complete on your own disk and
always will be — `save()` needs no account, no key and no network. What it
cannot do is reach the person who has to act on it. `report()` is the crossing:
work done in a notebook, offline, becomes visible to a PM without anyone
re-typing it.

Everything here is opt-in and explicit. Importing `alphaengine` still makes no
network call; nothing reports unless you call this function.

WHAT TRAVELS, AND WHAT CANNOT (`QOS-161`).
    `_payload(study)` emits the exact `sealed_study_projection(study)` plus
    `content_sha256`, so the payload received by the server hashes to the exact
    same `content_sha256` sealed on disk (`best_params` never travels).
"""

from __future__ import annotations

import contextlib
import json
import os
from typing import TYPE_CHECKING, Any

from .._version import API_URL_ENV, DEFAULT_API_URL, LEGACY_API_URL_ENV
from .schema import compute_content_sha256, sealed_study_projection

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .schema import Study

__all__ = ["report", "ReportError"]

DEFAULT_BASE_URL = DEFAULT_API_URL
_ENV_KEY = "QUANTOS_API_KEY"
_ENV_API_URL = API_URL_ENV
_ENV_URL = LEGACY_API_URL_ENV
_PATH = "/api/me/runs/ingest"
_TIMEOUT = 30


class ReportError(RuntimeError):
    """Reporting failed. The study on your disk is untouched."""


# Held equal with `client.executor.MAX_FIGURE_LIST` (`test_client.py` enforces parity).
MAX_FIGURE_LIST = 512


def _guard(payload: dict[str, Any]) -> dict[str, Any]:
    """Refuse to send anything series-shaped, whatever it is called."""

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, (list, tuple)):
            if len(node) > MAX_FIGURE_LIST:
                raise ValueError(
                    f"refusing to send {path}: {len(node)} elements is a series, "
                    "not a figure. Your data stays on your machine."
                )
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(payload, "study")
    return payload


def _payload(
    study: Study,
    *,
    assignment_id: str | None = None,
    thesis_id: str | None = None,
) -> dict[str, Any]:
    """Return the exact `sealed_study_projection(study)` + `content_sha256` (`QOS-161`).

    `best_params` is never included. If `assignment_id` or `thesis_id` are
    overridden at report time, the projection's `content_sha256` is recomputed
    over the updated projection so server-side verification still matches.
    """
    proj = sealed_study_projection(study)
    overridden = False
    if assignment_id is not None and assignment_id != proj.get("assignment_id"):
        proj["assignment_id"] = assignment_id
        overridden = True
    if thesis_id is not None and thesis_id != proj.get("thesis_id"):
        proj["thesis_id"] = thesis_id
        overridden = True
    sha = compute_content_sha256(proj) if (overridden or not study.content_sha256) else study.content_sha256
    return {**proj, "content_sha256": sha}


def report(
    study: Study,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = _TIMEOUT,
    assignment_id: str | None = None,
    thesis_id: str | None = None,
) -> dict[str, Any]:
    """Send a study to your workspace and return what the server recorded."""
    import urllib.error
    import urllib.request

    from ..auth import FirmKeyError, redact_secrets, require_firm_key, validate_clean_url

    try:
        key = require_firm_key(api_key)
    except FirmKeyError as exc:
        raise ReportError(f"No API key. {exc}") from exc

    root = (base_url or os.environ.get(_ENV_API_URL) or os.environ.get(_ENV_URL) or DEFAULT_BASE_URL).rstrip(
        "/"
    )
    try:
        validate_clean_url(root)
    except ValueError as exc:
        raise ReportError(str(exc)) from exc

    body = _guard(_payload(study, assignment_id=assignment_id, thesis_id=thesis_id))

    req = urllib.request.Request(
        root + _PATH,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key.strip()}",
            "User-Agent": f"alphaengine/{study.engine_version}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            recorded: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
            return recorded
    except urllib.error.HTTPError as e:
        detail = ""
        with contextlib.suppress(Exception):
            detail = redact_secrets(str(json.loads(e.read().decode("utf-8")).get("detail", "")))
        if e.code in (401, 403):
            raise ReportError(
                f"The key was rejected ({e.code}). It may be revoked, or from a different "
                f"workspace. {detail}".strip()
            ) from e
        raise ReportError(f"The server refused the study ({e.code}). {detail}".strip()) from e
    except urllib.error.URLError as e:
        reason = redact_secrets(str(e.reason))
        raise ReportError(f"Could not reach {root}: {reason}. Your study is still on your disk.") from e
