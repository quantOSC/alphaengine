"""Staying signed in, on this machine, for this user.

── WHAT CHANGED AND WHY IT IS NOT A CONTRADICTION ─────────────────────────────

The first version of the key prompt held the credential in the process and
wrote nothing, on the reasoning that we are not in the business of storing
customer keys. That reasoning is still right about one thing and was wrong
about another, and the distinction is worth being precise about:

  STILL TRUE — QuantOS never holds your key. There is no key field on any
    object we ship, `AgentDriver` takes a callable, and nothing here transmits a
    credential anywhere except to the service it belongs to.

  WAS WRONG — "therefore nothing may touch the disk". Re-pasting a key into
    every new terminal is not security, it is friction, and friction is what
    makes people put `QUANTOS_API_KEY=ae_live_...` in a shell profile or a
    committed `.env` — which is strictly worse than a mode-0600 file that this
    tool manages and can revoke.

  `gh`, `docker`, `npm` and `aws` all do exactly this, and they do it for the
  same reason.

── THE RULES THIS FILE KEEPS ──────────────────────────────────────────────────

  NEVER THE PROJECT DIRECTORY. Credentials live under the USER's config dir, so
    they cannot be committed, cannot be copied with the repo, and cannot end up
    in a Docker build context. A key in a project folder is a key in somebody's
    git history eventually.

  MODE 0600, and the file is created with those bits rather than chmod'ed after
    — a credential that is briefly world-readable was briefly world-readable.
    Windows has no mode bits; the per-user profile directory is already ACL'd to
    that user, which is the platform's equivalent and is what every other tool
    relies on there.

  THE ENVIRONMENT ALWAYS WINS. If `QUANTOS_API_KEY` is set in the shell, the
    stored file is ignored entirely. CI, a VPC deploy and a colleague's machine
    all set the variable, and a stale file silently overriding it is the kind of
    bug that takes a day to find.

  LOGGING OUT REALLY DELETES IT. Not a flag in the file — the file goes.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

__all__ = ["config_path", "load_stored", "save_key", "clear_stored", "apply_stored"]

#: Which environment variables this file is allowed to hold.
#: `QUANTOS_API_KEY` is ours. The provider keys are here because a user who has
#: said "sign me in" means all of it, and leaving one of the three to be pasted
#: every session defeats the point.
MANAGED = (
    "QUANTOS_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "OPENROUTER_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "ALPHAENGINE_API_KEY",
    "ALPHAENGINE_BASE_URL",
    "ALPHAENGINE_MODEL",
    "ALPHAENGINE_PROVIDER",
    "OPENAI_BASE_URL",
)


def config_path() -> Path:
    """`%APPDATA%\\alphaengine\\credentials.json`, or `~/.config/alphaengine/…`.

    `XDG_CONFIG_HOME` is honoured because on Linux it is the correct answer and
    because it makes this testable without touching a real home directory.
    """
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "alphaengine" / "credentials.json"


def load_stored() -> dict[str, str]:
    """Whatever is on disk, or `{}`. Never raises — a corrupt file is not a
    reason to refuse to start, it is a reason to be signed out."""
    path = config_path()
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {k: str(v) for k, v in raw.items() if k in MANAGED and isinstance(v, str) and v}


def save_key(name: str, value: str) -> Path:
    """Store one credential at mode 0600, creating the file with those bits."""
    if name not in MANAGED:
        raise ValueError(f"{name} is not a credential this tool manages")

    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    data = load_stored()
    data[name] = value

    # Created with 0600 rather than chmod'ed afterwards: a file that is briefly
    # world-readable was briefly world-readable, and on a shared box that is the
    # whole window an attacker needs.
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    # Atomic: a half-written credentials file is a login that half works.
    os.replace(tmp, path)
    return path


def clear_stored() -> bool:
    """Delete the file. True if there was one."""
    path = config_path()
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return False


def apply_stored() -> list[str]:
    """Put stored credentials into this process, WITHOUT overriding the shell.

    Returns the names that were applied, so the caller can say what happened —
    "signed in" is a fact the user should see rather than infer.
    """
    applied = []
    for name, value in load_stored().items():
        if os.environ.get(name):
            continue  # the environment always wins
        os.environ[name] = value
        applied.append(name)
    return applied
