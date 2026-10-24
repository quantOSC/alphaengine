"""Readline history and tab completion for the research shell (`ae shell`).

Stdlib only. Completes command verbs as well as project object identifiers
(run IDs from `.alphaengine/ledger/runs.jsonl`, dataset names from
`.alphaengine/data/datasets.jsonl`, and hypothesis IDs from
`.alphaengine/ideas/ideas.jsonl`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

__all__ = [
    "enable",
    "verbs_of",
    "project_completions",
    "make_completer",
]


def _history_path() -> Path:
    from .events import data_dir

    return data_dir() / "history"


def verbs_of(commands: Iterable[Any], extra: Iterable[str] = ()) -> list[str]:
    out = [c.verb for c in commands if getattr(c, "verb", "").isalpha()]
    out.extend(extra)
    return sorted(set(out))


def project_completions(
    proj: Any = None,
    *,
    verbs: Iterable[str] = (),
    symbols: Iterable[str] | None = None,
) -> list[str]:
    """Return sorted completion candidates including verbs, run IDs, dataset names, and idea IDs."""
    from .project import find_project

    active_proj = proj if proj is not None else find_project()
    candidates: set[str] = set()
    for v in verbs:
        candidates.add(str(v))
        candidates.add("/" + str(v))
    if symbols:
        for s in symbols:
            candidates.add(str(s))

    if active_proj is not None:
        candidates.add("HEAD")
        for r in active_proj.read_jsonl("ledger/runs.jsonl"):
            rid = str(r.get("run_id") or "")
            if rid:
                candidates.add(rid)
            fam = str(r.get("family") or "")
            if fam:
                candidates.add(fam)
        for d in active_proj.read_jsonl("data/datasets.jsonl"):
            nm = str(d.get("name") or "")
            if nm:
                candidates.add(nm)
        for i in active_proj.read_jsonl("ideas/ideas.jsonl"):
            iid = str(i.get("idea_id") or "")
            if iid:
                candidates.add(iid)

    return sorted(candidates)


def make_completer(
    *,
    verbs: list[str],
    symbols: list[str] | None = None,
    proj: Any = None,
) -> Callable[[str, int], str | None]:
    """Build a readline completer function over verbs and live project objects."""

    def complete(text: str, state: int) -> str | None:
        pool = project_completions(proj, verbs=verbs, symbols=symbols)
        matches = [c for c in pool if c.startswith(text)]
        return matches[state] if state < len(matches) else None

    return complete


def enable(*, verbs: list[str], symbols: list[str] | None = None, proj: Any = None) -> None:
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

    completer = make_completer(verbs=verbs, symbols=symbols, proj=proj)
    readline.set_completer(completer)
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
