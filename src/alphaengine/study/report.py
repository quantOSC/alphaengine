"""Send a study to a QuantOS workspace.

THE ONE THING THIS MODULE IS FOR. A study is complete on your own disk and
always will be — `save()` needs no account, no key and no network. What it
cannot do is reach the person who has to act on it. `report()` is the crossing:
work done in a notebook, offline, becomes visible to a PM without anyone
re-typing it.

Everything here is opt-in and explicit. Importing `alphaengine` still makes no
network call; nothing reports unless you call this function.

WHAT TRAVELS, AND WHAT CANNOT.
    A study carries what was TRIED and what came BACK — the trial count and how
    it was obtained, a content hash of the data, the verdict, the shape of the
    neighbourhood, the performance figures. It does NOT carry your returns, your
    prices or your parameter grid.

    That is enforced rather than promised. `_guard` walks the whole payload and
    refuses anything series-shaped by LENGTH, not by field name, so a renamed
    key cannot smuggle one through. The same guard runs on the server; the
    duplication is the design, because a check that only runs on the client is
    not a check, and one that only runs on the server tells you too late and
    without naming the field.

NO NEW DEPENDENCIES. urllib from the standard library, not requests. Two runtime
dependencies is a feature of this package and a third is a cost paid by every
user of the offline half, who did not ask for this.
"""

from __future__ import annotations

import contextlib
import json
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .schema import Study

__all__ = ["report", "ReportError"]

DEFAULT_BASE_URL = "https://alpha-backend-production-51df.up.railway.app"
_ENV_KEY = "QUANTOS_API_KEY"
_ENV_URL = "QUANTOS_API_URL"
_PATH = "/api/me/runs/ingest"
_TIMEOUT = 30


class ReportError(RuntimeError):
    """Reporting failed. The study on your disk is untouched."""


# The longest list that is still a set of figures rather than a series. Held
# here rather than imported from `client` so that `alphaengine.study` never
# pulls the client in — a guarantee `test_client.py` enforces, and one worth
# more than not repeating fifteen lines.
#
# DUPLICATED ON PURPOSE, WHICH MEANS IT CAN DRIFT BY ACCIDENT. Raised with
# `client.executor.MAX_FIGURE_LIST` on 2026-08-08; `test_client.py` holds the
# two equal so the deliberate duplication cannot become an accidental
# disagreement.
MAX_FIGURE_LIST = 512


def _guard(payload: dict[str, Any]) -> dict[str, Any]:
    """Refuse to send anything series-shaped, whatever it is called.

    Keyed on LENGTH, not on field name, so renaming a key does not get a series
    past it. Mirrored on the server on purpose: a check that runs only on the
    client is not a check, and one that runs only on the server tells you too
    late and without naming the field.
    """

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


def _payload(study: Study) -> dict[str, Any]:
    """The subset that crosses. Explicit rather than `asdict()`.

    An allowlist, not a denylist: a field added to `Study` later does not start
    leaving the machine because somebody forgot to exclude it. `best_params` is
    absent on purpose — the grid is frequently bigger IP than the returns.
    """
    out: dict[str, Any] = {
        "label": study.label or "(untitled study)",
        "schema_version": study.schema_version,
        "engine_version": study.engine_version,
        "n_trials": study.n_trials,
        "n_trials_source": study.n_trials_source,
        "data_hash": study.data_hash,
        "data_description": study.data_description,
        "verdict": study.verdict,
        "deflated_sharpe": study.deflated_sharpe,
        "surface": study.surface or {},
        "performance": study.performance or {},
        "notes": study.notes,
    }
    if study.process:
        out["process"] = study.process
    if study.charts:
        out["charts"] = study.charts
    return {k: v for k, v in out.items() if v is not None}


def report(
    study: Study,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = _TIMEOUT,
) -> dict[str, Any]:
    """Send a study to your workspace and return what the server recorded.

    Args:
        study: the study to send. Unchanged by this call.
        api_key: a portal-issued `ae_live_*` key. Falls back to the
            ``QUANTOS_API_KEY`` environment variable, which is where a key
            belongs — a key pasted into a notebook is a key committed to git.
        base_url: the platform to report to. Falls back to ``QUANTOS_API_URL``,
            then to the public API. Set it for a self-hosted or VPC deployment.
        timeout: seconds before giving up.

    Raises:
        ReportError: no key, the key was rejected, or the server could not be
            reached. Never raises for a study that is merely unimpressive — a
            `likely_noise` verdict is a result and reports like any other.

    A failure here costs you nothing. The study is already on your disk; this is
    a copy travelling, not a handoff.
    """
    # Imported here, not at module scope: `import alphaengine` must touch no
    # networking code at all, which `test_smoke.py` enforces.
    import urllib.error
    import urllib.request

    key = api_key or os.environ.get(_ENV_KEY, "")
    if not key.strip():
        raise ReportError(
            f"No API key. Pass api_key=..., or set {_ENV_KEY}. Create one in the portal under Settings."
        )

    root = (base_url or os.environ.get(_ENV_URL) or DEFAULT_BASE_URL).rstrip("/")
    body = _guard(_payload(study))

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
        # The body is best-effort context for the message. A server that
        # answers with HTML, an empty body, or nothing at all still has to
        # produce a usable error rather than a second exception on top of the
        # first, which is why the failure to read it is suppressed entirely.
        detail = ""
        with contextlib.suppress(Exception):
            detail = json.loads(e.read().decode("utf-8")).get("detail", "")
        if e.code in (401, 403):
            raise ReportError(
                f"The key was rejected ({e.code}). It may be revoked, or from a different "
                f"workspace. {detail}".strip()
            ) from e
        raise ReportError(f"The server refused the study ({e.code}). {detail}".strip()) from e
    except urllib.error.URLError as e:
        raise ReportError(f"Could not reach {root}: {e.reason}. Your study is still on your disk.") from e
