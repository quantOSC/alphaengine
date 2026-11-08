"""Staying signed in, on this machine, for this user, and gating firm workflows.

── PRODUCT BOUNDARY ───────────────────────────────────────────────────────────

The open-source Python library (`import alphaengine`), `alphaengine demo`,
`alphaengine process`, `ae verify` on sealed studies, and diagnostic commands
(`ae version`, `ae whoami`, `ae doctor`) run without an account or key.

CLI workflows (`run`, `diagnose`, `screen`, `signal`, `validate`, `stress`,
`overlap`, `size`, `monitor`, `watch`) and workspace commands (`init`, `log`,
`show`, `diff`, `rerun`, `data`, `idea`, `seal`, `report`, `remote`, `push`,
`pull`, `plugin`, `shell`) require a firm-issued `ae_live_` key from Connect in
the quantOS portal, checked at `require_firm_key` and verified server-side.

── CREDENTIAL STORAGE (`QOS-155`) ─────────────────────────────────────────────

1. Environment variables (`QUANTOS_API_KEY`, etc.) always win.
2. When persisting credentials via `login`, `save_credential` / `save_key`
   writes **ONLY** to the OS keychain (`keyring`) when a usable backend is
   available — `credentials.json` is NOT created.
3. Only when no usable OS keyring is available (or `prefer_file=True` /
   `--no-keyring` is explicitly passed) does `save_credential` write a
   mode-`0600` file (`~/.config/alphaengine/credentials.json` or
   `%APPDATA%\\alphaengine\\credentials.json`), returning `storage="file"` so
   the CLI prints an explicit plaintext-file disclosure.
4. Remote/API URLs are checked by `validate_clean_url` so credentials can never
   be embedded in `user:pass@host` or query strings.
5. Secrets are redacted from errors, logs, tracebacks, and `--json` output.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import stat
import threading
import urllib.parse
from pathlib import Path
from typing import Any

__all__ = [
    "FIRM_KEY_ERROR_MESSAGE",
    "PLAINTEXT_FILE_DISCLOSURE",
    "KEY_FLAG_WARNING",
    "FirmKeyError",
    "MANAGED",
    "config_path",
    "keyring_available",
    "has_legacy_file_credentials",
    "migrate_file_to_keyring",
    "validate_clean_url",
    "mask_secret_prefix",
    "credential_status",
    "load_stored",
    "save_credential",
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

PLAINTEXT_FILE_DISCLOSURE = (
    "No OS keyring backend is available; storing credential in plaintext mode-0600 file"
)

KEY_FLAG_WARNING = (
    "Warning: passing `--key` on the command line exposes your credential in shell history "
    "and process lists. Prefer `alphaengine login` or `QUANTOS_API_KEY`."
)

_KEYRING_SERVICE = "alphaengine"
_ALLOW_KEYRING_IN_TEST = False

#: Pattern matching `ae_live_*` and common provider keys so they never leak in logs/errors/JSON.
_SECRET_RE = re.compile(r"\b(ae_live_[A-Za-z0-9_\-]{3,}|sk-ant-[A-Za-z0-9_\-]{6,}|sk-[A-Za-z0-9_\-]{12,})")

_SECRET_QUERY_PARAMS = frozenset(
    {
        "key",
        "api_key",
        "apikey",
        "token",
        "access_token",
        "auth",
        "authorization",
        "secret",
        "password",
        "quantos_api_key",
    }
)


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

#: Tracks `(source, value)` applied into `os.environ` by `apply_stored()` so `credential_status()`
#: accurately distinguishes parent-shell `env` variables from `keyring` or `file` storage.
_APPLIED_ENV_SOURCES: dict[str, tuple[str, str]] = {}


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


def mask_secret_prefix(value: str | None) -> str | None:
    """Return a non-sensitive masked display string (never matching or leaking raw secrets)."""
    cleaned = (value or "").strip()
    if not cleaned:
        return None
    if cleaned.startswith("ae_live_"):
        tail = cleaned[len("ae_live_") :]
        hint = tail[:2] if len(tail) >= 4 else ""
        return f"ae_live_{hint}…****" if hint else "ae_live_****"
    if cleaned.startswith("sk-ant-"):
        tail = cleaned[len("sk-ant-") :]
        hint = tail[:2] if len(tail) >= 4 else ""
        return f"sk-ant-{hint}…****" if hint else "sk-ant-****"
    if cleaned.startswith("sk-"):
        tail = cleaned[len("sk-") :]
        hint = tail[:2] if len(tail) >= 4 else ""
        return f"sk-{hint}…****" if hint else "sk-****"
    return "****"


def validate_clean_url(url: str) -> str:
    """Reject remote/API URLs carrying embedded userinfo or secret query/path tokens (`QOS-155`)."""
    raw = (url or "").strip()
    if not raw:
        raise ValueError("Remote URL must be non-empty.")
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc or not parsed.hostname:
        raise ValueError(f"Invalid URL {redact_secrets(raw)!r}: expected http:// or https:// host.")
    if parsed.username or parsed.password or "@" in parsed.netloc:
        raise ValueError(
            "Refusing URL with embedded credentials (user:pass@host). "
            "Store API keys via `alphaengine login` or `QUANTOS_API_KEY`."
        )
    if _SECRET_RE.search(raw):
        raise ValueError(
            "Refusing URL containing an inline API key token. "
            "Never embed `ae_live_*` or provider keys in URLs."
        )
    for k, v in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True):
        if k.strip().lower() in _SECRET_QUERY_PARAMS or _SECRET_RE.search(v):
            raise ValueError(
                f"Refusing URL with credential query parameter {k!r}. "
                "Pass credentials via Authorization header (`alphaengine login` or `QUANTOS_API_KEY`)."
            )
    return raw


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
        # Under tests, `sys.modules["keyring"]` may be swapped per test via monkeypatch
        import sys  # noqa: PLC0415

        mod_now = sys.modules.get("keyring")
        if mod_now is not None and mod_now is not _KEYRING:
            _KEYRING = mod_now
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


def keyring_available() -> bool:
    """Return True when an active, non-fail OS keyring backend is usable."""
    kr = _keyring_module()
    if kr is None:
        return False
    get_kr = getattr(kr, "get_keyring", None)
    if callable(get_kr):
        try:
            backend = get_kr()
            cls_name = type(backend).__name__.lower()
            mod_name = type(backend).__module__.lower()
            if "fail" in cls_name or "null" in cls_name or "fail" in mod_name or "null" in mod_name:
                return False
        except Exception:  # noqa: BLE001
            return False
    return callable(getattr(kr, "set_password", None)) and callable(getattr(kr, "get_password", None))


def _keyring_get(name: str) -> str | None:
    if not keyring_available():
        return None
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
    if not keyring_available():
        return False
    kr = _keyring_module()
    if kr is None:
        return False
    try:
        kr.set_password(_KEYRING_SERVICE, name, value)
        return True
    except Exception:  # noqa: BLE001
        return False


def _keyring_delete(name: str) -> bool:
    if not keyring_available():
        return False
    kr = _keyring_module()
    if kr is None:
        return False
    had_existing = _keyring_get(name) is not None
    try:
        kr.delete_password(_KEYRING_SERVICE, name)
        return had_existing
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


def _write_mode_0600_json(path: Path, data: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


def has_legacy_file_credentials() -> bool:
    """Return True when `credentials.json` exists on disk with at least one managed key."""
    return bool(_load_file_stored())


def migrate_file_to_keyring(*, delete_file: bool = True) -> dict[str, Any]:
    """Migrate `credentials.json` entries into OS keyring and optionally delete the file (`QOS-155`)."""
    file_data = _load_file_stored()
    if not keyring_available():
        return {
            "keyring_available": False,
            "migrated": [],
            "deleted_file": False,
        }
    migrated: list[str] = []
    for k, v in file_data.items():
        if _keyring_set(k, v):
            migrated.append(k)
            if k in _APPLIED_ENV_SOURCES:
                _APPLIED_ENV_SOURCES[k] = ("keyring", v)
    deleted = False
    path = config_path()
    if delete_file and len(migrated) == len(file_data) and path.exists():
        with contextlib.suppress(OSError):
            path.unlink()
            deleted = True
        with contextlib.suppress(OSError):
            path.parent.rmdir()
    return {
        "keyring_available": True,
        "migrated": migrated,
        "deleted_file": deleted,
    }


def _load_stored_with_sources() -> dict[str, tuple[str, str]]:
    """Return `{name: (source, value)}` where `keyring` takes precedence over `file`."""
    out: dict[str, tuple[str, str]] = {}
    for k, v in _load_file_stored().items():
        out[k] = ("file", v)
    if keyring_available():
        for name in MANAGED:
            kr_val = _keyring_get(name)
            if kr_val:
                out[name] = ("keyring", kr_val)
    return out


def load_stored() -> dict[str, str]:
    """Load stored credentials from OS keychain when available, with the 0600 file as fallback."""
    return {k: pair[1] for k, pair in _load_stored_with_sources().items()}


def save_credential(name: str, value: str, *, prefer_file: bool = False) -> dict[str, Any]:
    """Persist one managed credential keyring-first (`QOS-155`).

    - When OS keyring is usable and `prefer_file=False`, writes **ONLY** to OS keyring
      and never creates `credentials.json` (removing any stale copy of `name` from
      `credentials.json` if a legacy file existed).
    - Writes mode-`0600` `credentials.json` **ONLY** when no usable keyring exists
      or `prefer_file=True`.
    """
    if name not in MANAGED:
        raise ValueError(f"{name} is not a credential this tool manages")

    kr_ok = keyring_available()
    path = config_path()

    if not prefer_file and kr_ok and _keyring_set(name, value):
        # Clean up any legacy plaintext entry for `name` if `credentials.json` existed
        existing_file = _load_file_stored()
        if name in existing_file:
            existing_file.pop(name, None)
            if existing_file:
                with contextlib.suppress(OSError):
                    _write_mode_0600_json(path, existing_file)
            else:
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)
        return {
            "storage": "keyring",
            "path": path,
            "file_written": False,
            "keyring_available": True,
        }

    data = _load_file_stored()
    data[name] = value
    _write_mode_0600_json(path, data)
    return {
        "storage": "file",
        "path": path,
        "file_written": True,
        "keyring_available": kr_ok,
    }


def save_key(name: str, value: str, *, prefer_file: bool = False) -> Path:
    """Persist one credential keyring-first (`QOS-155`) and return `config_path()`."""
    res = save_credential(name, value, prefer_file=prefer_file)
    p = res["path"]
    assert isinstance(p, Path)
    return p


def remove_key(name: str) -> bool:
    """Remove a single credential from OS keychain and `credentials.json`."""
    _APPLIED_ENV_SOURCES.pop(name, None)
    kr_removed = _keyring_delete(name)
    path = config_path()
    data = _load_file_stored()
    if name not in data:
        return kr_removed
    data.pop(name, None)
    if not data:
        file_removed = False
        with contextlib.suppress(OSError):
            if path.exists():
                path.unlink()
                file_removed = True
        return file_removed or kr_removed
    try:
        _write_mode_0600_json(path, data)
    except OSError:
        path.with_suffix(".tmp").unlink(missing_ok=True)
        return kr_removed
    return True


def clear_stored(*, purge: bool = False) -> bool:
    """Delete stored credentials from OS keyring and the `0600` fallback file."""
    kr_any = False
    for name in MANAGED:
        if _keyring_delete(name):
            kr_any = True
    path = config_path()
    file_removed = False
    try:
        path.unlink()
        file_removed = True
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
    _APPLIED_ENV_SOURCES.clear()
    return file_removed or kr_any


def purge_stored() -> bool:
    """Purge OS keyring credentials, `credentials.json`, empty config dir, and process env keys."""
    return clear_stored(purge=True)


def apply_stored() -> list[str]:
    """Put stored credentials into this process WITHOUT overriding parent shell environment variables."""
    applied: list[str] = []
    for name, (src, value) in _load_stored_with_sources().items():
        if os.environ.get(name):
            continue  # the environment always wins
        os.environ[name] = value
        _APPLIED_ENV_SOURCES[name] = (src, value)
        applied.append(name)
    return applied


def credential_status() -> dict[str, Any]:
    """Inspect active credential sources and masked prefixes without exposing raw secrets (`QOS-155`)."""
    kr_ok = keyring_available()
    file_data = _load_file_stored()
    entries: dict[str, dict[str, Any]] = {}

    secret_vars = (
        "QUANTOS_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "OPENROUTER_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "ALPHAENGINE_API_KEY",
    )

    for name in secret_vars:
        env_val = os.environ.get(name)
        kr_val = _keyring_get(name) if kr_ok else None
        fl_val = file_data.get(name)

        source = "none"
        active_val: str | None = None
        if env_val:
            applied_pair = _APPLIED_ENV_SOURCES.get(name)
            source = applied_pair[0] if applied_pair is not None and applied_pair[1] == env_val else "env"
            active_val = env_val
        elif kr_val:
            source = "keyring"
            active_val = kr_val
        elif fl_val:
            source = "file"
            active_val = fl_val

        entries[name] = {
            "configured": active_val is not None,
            "source": source,
            "masked": mask_secret_prefix(active_val),
            "in_keyring": kr_val is not None,
            "in_file": fl_val is not None,
        }

    path = config_path()
    return {
        "keyring_available": kr_ok,
        "credentials_file": str(path),
        "credentials_file_exists": path.is_file(),
        "legacy_file_migration_recommended": bool(kr_ok and file_data),
        "quantos": entries["QUANTOS_API_KEY"],
        "providers": {k: v for k, v in entries.items() if k != "QUANTOS_API_KEY"},
    }


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
    """
    if explicit_key is not None:
        candidate = _clean_firm_key(explicit_key)
    else:
        env_val = os.environ.get("QUANTOS_API_KEY")
        if env_val:
            candidate = _clean_firm_key(env_val)
        else:
            stored_map = _load_stored_with_sources()
            pair = stored_map.get("QUANTOS_API_KEY")
            candidate = _clean_firm_key(pair[1] if pair else "")
            if candidate and pair is not None:
                os.environ["QUANTOS_API_KEY"] = candidate
                _APPLIED_ENV_SOURCES["QUANTOS_API_KEY"] = (pair[0], candidate)

    if not candidate or not candidate.startswith("ae_live_"):
        raise FirmKeyError(FIRM_KEY_ERROR_MESSAGE)
    return candidate
