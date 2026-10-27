"""Staying signed in, on this machine, for this user, and gating firm workflows.

── PRODUCT BOUNDARY ───────────────────────────────────────────────────────────

The open-source Python library (`import alphaengine`), `alphaengine demo`,
`alphaengine process`, and `ae verify` on sealed studies are free and run
offline without an account or key.

CLI workflows (`run`, `diagnose`, `screen`, `signal`, `validate`, `stress`,
`overlap`, `size`, `monitor`, `watch`) and the CLI OS (`init`, `log`, `show`,
`diff`, `rerun`, `data`, `idea`, `seal`, `report`, `remote`, `push`, `pull`,
`plugin`, `shell`) require a firm-issued `ae_live_` key from Connect in the
quantOS portal, checked at a single client choke point (`require_firm_key`) and
verified server-side on every API request.

── CREDENTIAL STORAGE ─────────────────────────────────────────────────────────

1. Environment variables (`QUANTOS_API_KEY`, etc.) always win.
2. When persisting credentials via `login`, `save_key` stores the secret in the
   OS keychain (`keyring`) when available, with a mode-0600 file under the
   user's config directory (`~/.config/alphaengine/credentials.json` or
   `%APPDATA%\\alphaengine\\credentials.json`) as fallback.
3. Secrets are redacted from errors, logs, tracebacks, and `--json` output.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import stat
import threading
from pathlib import Path
from typing import Any

__all__ = [
    "FIRM_KEY_ERROR_MESSAGE",
    "FirmKeyError",
    "MANAGED",
    "config_path",
    "load_stored",
    "save_key",
    "remove_key",
    "clear_stored",
    "purge_stored",
    "apply_stored",
    "require_firm_key",
    "redact_secrets",
]

FIRM_KEY_ERROR_MESSAGE = (
    "This workflow needs an AlphaEngine key from your firm's quantOS portal (Connect). "
    "The library itself is free and offline: see docs/free-vs-firm.md."
)

_KEYRING_SERVICE = "alphaengine"
_ALLOW_KEYRING_IN_TEST = False

#: Pattern matching `ae_live_*` and common provider keys so they never leak in logs/errors/JSON.
_SECRET_RE = re.compile(r"\b(ae_live_[A-Za-z0-9_\-]{3,}|sk-ant-[A-Za-z0-9_\-]{6,}|sk-[A-Za-z0-9_\-]{12,})")


class FirmKeyError(RuntimeError):
    """Raised when a gated CLI workflow or CLI OS command is invoked without a firm `ae_live_` key."""


#: Which environment variables this module is allowed to hold.
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
    "ALPHAENGINE_API_URL",
    "ALPHAENGINE_BASE_URL",
    "ALPHAENGINE_MODEL",
    "ALPHAENGINE_PROVIDER",
    "OPENAI_BASE_URL",
)


def redact_secrets(text: str) -> str:
    """Redact any `ae_live_*` or provider API key from `text`."""
    if not text:
        return text

    def _mask(match: re.Match[str]) -> str:
        tok = match.group(1)
        if tok.startswith("ae_live_"):
            return "ae_live_[REDACTED]"
        return "[REDACTED_KEY]"

    return _SECRET_RE.sub(_mask, text)


_KEYRING: Any = None
_KEYRING_WAIT_S = 0.4


def _keyring_module() -> Any | None:
    """The OS vault is optional. A hung import must not delay the first line of output."""
    global _KEYRING
    if os.environ.get("PYTEST_CURRENT_TEST") and not _ALLOW_KEYRING_IN_TEST:
        return None
    if _KEYRING is False:
        return None
    if _KEYRING is not None:
        return _KEYRING
    box: dict[str, Any] = {}

    def _load() -> None:
        try:
            import importlib  # noqa: PLC0415

            box["mod"] = importlib.import_module("keyring")
        except Exception:  # noqa: BLE001
            box["mod"] = False

    worker = threading.Thread(target=_load, name="ae-keyring", daemon=True)
    worker.start()
    worker.join(_KEYRING_WAIT_S)
    if worker.is_alive() or "mod" not in box or box["mod"] is False:
        _KEYRING = False
        return None
    _KEYRING = box["mod"]
    return _KEYRING


def _keyring_get(name: str) -> str | None:
    kr = _keyring_module()
    if kr is None:
        return None
    box: dict[str, Any] = {}

    def _read() -> None:
        try:
            box["val"] = kr.get_password(_KEYRING_SERVICE, name)
        except Exception as exc:  # noqa: BLE001
            box["err"] = exc

    worker = threading.Thread(target=_read, name="ae-keyring-get", daemon=True)
    worker.start()
    worker.join(_KEYRING_WAIT_S)
    if worker.is_alive() or "err" in box:
        return None
    val = box.get("val")
    return str(val) if isinstance(val, str) and val.strip() else None


def _keyring_set(name: str, value: str) -> bool:
    kr = _keyring_module()
    if kr is None:
        return False
    try:
        kr.set_password(_KEYRING_SERVICE, name, value)
        return True
    except Exception:  # noqa: BLE001
        return False


def _keyring_delete(name: str) -> bool:
    kr = _keyring_module()
    if kr is None:
        return False
    try:
        kr.delete_password(_KEYRING_SERVICE, name)
        return True
    except Exception:  # noqa: BLE001
        return False


def config_path() -> Path:
    """`%APPDATA%\\alphaengine\\credentials.json`, or `~/.config/alphaengine/…`."""
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "alphaengine" / "credentials.json"


def _load_file_stored() -> dict[str, str]:
    path = config_path()
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {k: str(v) for k, v in raw.items() if k in MANAGED and isinstance(v, str) and v}


def load_stored() -> dict[str, str]:
    """Load stored credentials from OS keychain when available, with the 0600 file as fallback."""
    data = _load_file_stored()
    if _keyring_module() is not None:
        for name in MANAGED:
            kr_val = _keyring_get(name)
            if kr_val:
                data[name] = kr_val
    return data


def save_key(name: str, value: str) -> Path:
    """Store one credential in the OS keychain when available and mode-0600 fallback file."""
    if name not in MANAGED:
        raise ValueError(f"{name} is not a credential this tool manages")

    _keyring_set(name, value)

    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    data = _load_file_stored()
    data[name] = value

    # Created with 0600 rather than chmod'ed afterwards.
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)
    return path


def remove_key(name: str) -> bool:
    """Remove a single credential from OS keychain and ``credentials.json``."""
    kr_removed = _keyring_delete(name)
    path = config_path()
    data = _load_file_stored()
    if name not in data:
        return kr_removed
    data.pop(name, None)
    if not data:
        return clear_stored() or kr_removed
    tmp = path.with_suffix(".tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        return kr_removed
    return True


def clear_stored(*, purge: bool = False) -> bool:
    """Delete stored credentials from keyring and the 0600 fallback file."""
    for name in MANAGED:
        _keyring_delete(name)
    path = config_path()
    removed = False
    try:
        path.unlink()
        removed = True
    except FileNotFoundError:
        pass
    except OSError:
        pass
    if purge:
        with contextlib.suppress(OSError):
            path.with_suffix(".tmp").unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            path.parent.rmdir()
        for name in MANAGED:
            os.environ.pop(name, None)
    return removed


def purge_stored() -> bool:
    """Remove ``credentials.json``, clean up empty config dir, and clear in-process keys."""
    return clear_stored(purge=True)


def apply_stored() -> list[str]:
    """Put stored credentials into this process, WITHOUT overriding the shell."""
    applied = []
    for name, value in load_stored().items():
        if os.environ.get(name):
            continue  # the environment always wins
        os.environ[name] = value
        applied.append(name)
    return applied


def _clean_firm_key(raw: str | None) -> str:
    text = (raw or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        text = text[1:-1].strip()
    if text.lower().startswith("bearer "):
        text = text[7:].strip()
    return text


def require_firm_key(explicit_key: str | None = None) -> str:
    """Single choke point for every gated CLI workflow and CLI OS command.

    Returns the validated `ae_live_*` key string, or raises :class:`FirmKeyError`
    with the canonical message pointing to Connect and `docs/free-vs-firm.md`.
    Note: The server is the authority and verifies the key's firm tier on every
    network call; this client check provides a clear immediate error.
    """
    if explicit_key is not None:
        candidate = _clean_firm_key(explicit_key)
    else:
        env_val = os.environ.get("QUANTOS_API_KEY")
        if env_val:
            candidate = _clean_firm_key(env_val)
        else:
            stored = load_stored().get("QUANTOS_API_KEY", "")
            candidate = _clean_firm_key(stored)
            if candidate:
                os.environ["QUANTOS_API_KEY"] = candidate

    if not candidate or not candidate.startswith("ae_live_"):
        raise FirmKeyError(FIRM_KEY_ERROR_MESSAGE)
    return candidate
