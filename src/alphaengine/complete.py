"""Readline history and tab completion for the research session.

Stdlib only. A TUI framework would break the two-dependency promise for people
who never run the agent. History lives under the user data dir, never the
project directory.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

__all__ = ["enable", "verbs_of"]


def _history_path() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "alphaengine" / "history"


def verbs_of(commands: Iterable[Any], extra: Iterable[str] = ()) -> list[str]:
    out = [c.verb for c in commands if getattr(c, "verb", "").isalpha()]
    out.extend(extra)
    return sorted(set(out))


def enable(*, verbs: list[str], symbols: list[str] | None = None) -> None:
    """Attach history + completer to this process. No-ops when readline is absent."""
    try:
        import readline  # noqa: PLC0415
    except ImportError:  # pragma: no cover - Windows without pyreadline
        return

    path = _history_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            readline.read_history_file(str(path))
    except OSError:
        pass
    readline.set_history_length(1000)

    candidates = list(verbs)
    for slash in verbs:
        candidates.append("/" + slash)
    if symbols:
        candidates.extend(symbols)

    def complete(text: str, state: int) -> str | None:
        matches = [c for c in candidates if c.startswith(text)]
        return matches[state] if state < len(matches) else None

    readline.set_completer(complete)
    readline.set_completer_delims(" \t\n")
    try:
        readline.parse_and_bind("tab: complete")
    except Exception:  # noqa: BLE001 - libedit uses a different bind
        readline.parse_and_bind("bind ^I rl_complete")

    import atexit

    def _save() -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            readline.write_history_file(str(path))
        except OSError:
            pass

    atexit.register(_save)
