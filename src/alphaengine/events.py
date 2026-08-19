"""Allowlisted model and run events. Figures, never series; hashes, never prompts.

WHAT THIS MODULE IS FOR
    A logged-in QuantOS user should be able to see, on the portal, what the
    model chose and what it answered — the same sentences the terminal already
    printed. An air-gapped desk should have the same record on disk.

WHAT NEVER TRAVELS
    Prices, returns, grids, `best_params`, trial matrices, raw prompts, LLM
    keys, QuantOS keys, or a synthesis that failed the citation guard. The
    guard is the producer: an event is built FROM an allowlist, not stripped
    from a dump.

A 404 MUST NOT FAIL A RUN
    The portal may be an older build. Telemetry is additive. A missing route
    is logged once and then silent.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

__all__ = [
    "EVENT_KEYS",
    "EventSink",
    "allowlist",
    "data_dir",
    "make_event",
]

logger = logging.getLogger(__name__)

#: Explicit rather than `asdict()`. A field added later does not start leaving
#: the machine because somebody forgot to exclude it.
EVENT_KEYS: frozenset[str] = frozenset(
    {
        "event_id",
        "t",
        "run_id",
        "kind",
        "provider",
        "model",
        "choice",
        "why",
        "answer_text",
        "answer_caveats",
        "usage",
        "prompt_hash",
        "prompt_chars",
        "completion_chars",
        "workflow",
        "agency",
        "error",
    }
)

_KINDS = frozenset({"pick", "thought", "answer", "refusal", "error", "completion"})
_MAX_WHY = 240
_MAX_ANSWER = 4000
_logged_missing = False


def data_dir() -> Path:
    """`~/.local/share/alphaengine`, or `%LOCALAPPDATA%\\alphaengine`.

    NEVER THE PROJECT DIRECTORY. A run dump next to `study.json` is a dump that
    gets committed. `XDG_DATA_HOME` is honoured so this is testable.
    """
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "alphaengine"


def allowlist(event: dict[str, Any]) -> dict[str, Any]:
    """Keep only EVENT_KEYS. Truncate prose. Drop anything series-shaped."""
    out: dict[str, Any] = {}
    for key in EVENT_KEYS:
        if key not in event:
            continue
        value = event[key]
        if value is None:
            continue
        if key == "why" and isinstance(value, str):
            out[key] = value[:_MAX_WHY]
        elif key == "answer_text" and isinstance(value, str):
            out[key] = value[:_MAX_ANSWER]
        elif key == "answer_caveats" and isinstance(value, (list, tuple)):
            out[key] = [str(v)[:_MAX_WHY] for v in value[:16]]
        elif key == "usage" and isinstance(value, dict):
            out[key] = {
                k: int(v)
                for k, v in value.items()
                if k in ("prompt_tokens", "completion_tokens", "total_tokens") and _is_int(v)
            }
        elif isinstance(value, (list, tuple, dict)):
            # A list that is not caveats is a series in disguise.
            continue
        else:
            out[key] = value
    return out


def _is_int(value: Any) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def make_event(
    kind: str,
    *,
    run_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    if kind not in _KINDS:
        kind = "error"
        fields = {**fields, "error": f"unknown event kind was coerced: {kind}"}
    raw: dict[str, Any] = {
        "event_id": uuid.uuid4().hex,
        "t": time.time(),
        "kind": kind,
        "run_id": run_id,
        "provider": provider,
        "model": model,
        **fields,
    }
    return allowlist(raw)


class EventSink:
    """Write jsonl always; POST to the portal when a session and run exist.

    Offline maths and `demo` construct a sink with `session=None` and still get
    a local file. A 404/offline POST is swallowed after one log line.
    """

    def __init__(self, session: Any | None = None, *, run_id: str | None = None) -> None:
        self.session = session
        self.run_id = run_id
        self._portal_ok = True
        self.path = data_dir() / "runs" / f"{run_id or 'session'}.jsonl"

    def bind(self, run_id: str) -> None:
        self.run_id = run_id
        self.path = data_dir() / "runs" / f"{run_id}.jsonl"

    def emit(self, event: dict[str, Any]) -> dict[str, Any]:
        payload = allowlist({**event, "run_id": event.get("run_id") or self.run_id})
        self._write(payload)
        if self.session is not None and payload.get("run_id") and self._portal_ok:
            self._post(payload)
        return payload

    def _write(self, payload: dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, default=str) + "\n")
        except OSError as exc:  # pragma: no cover - disk full / perms
            logger.debug("could not write event log: %s", exc)

    def _post(self, payload: dict[str, Any]) -> None:
        global _logged_missing
        post = getattr(self.session, "post_event", None)
        if post is None:
            return
        try:
            post(str(payload.get("run_id")), payload)
        except Exception as exc:  # noqa: BLE001 - telemetry must never fail a run
            status = getattr(exc, "status", None)
            if status in (404, 405) or "404" in str(exc):
                self._portal_ok = False
                if not _logged_missing:
                    logger.info("portal has no agent-events route; events stay on disk")
                    _logged_missing = True
                return
            # Offline / 401 / anything else: stay local, do not abort the research.
            logger.debug("event post skipped: %s", exc)
