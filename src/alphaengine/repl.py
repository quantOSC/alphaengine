"""Research session state and interactive line shell (`ae shell`).

Holds data, book, last run, pinned model, conversation turns, and provides
`run_shell()` for interactive project sessions with tab completion over run IDs,
dataset names, and idea IDs, plus a status line displaying the project name and
cumulative trial count.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .book import Book
from .events import EventSink

__all__ = ["SessionState", "SLASH", "run_shell"]

#: Verbs that also work with a leading `/`, Claude-Code style.
SLASH = (
    "init",
    "login",
    "load",
    "demo",
    "process",
    "help",
    "model",
    "models",
    "key",
    "data",
    "idea",
    "universe",
    "project",
    "book",
    "run",
    "log",
    "show",
    "diff",
    "rerun",
    "seal",
    "verify",
    "remote",
    "push",
    "pull",
    "trace",
    "gaps",
    "tonight",
    "runs",
    "quiet",
    "verbose",
    "status",
    "logout",
    "quit",
    "commands",
)


@dataclass
class SessionState:
    """Everything a session remembers between prompts."""

    data: Any = None
    backtest_fn: Any = None
    book: Book = field(default_factory=Book)
    last: Any = None
    last_answer: Any = None
    last_shortlist: list[str] = field(default_factory=list)
    loaded_project: str | None = None
    pinned_model: str | None = None
    conversation: list[tuple[str, str]] = field(default_factory=list)
    quiet: bool = False
    sink: EventSink = field(default_factory=EventSink)

    def remember_run(self, run: Any, question: str | None = None) -> None:
        self.last = run
        if run is not None and getattr(run, "run_id", None):
            self.sink.bind(str(run.run_id))
        rows = ((getattr(run, "artifact", None) or {}).get("figures") or {}).get("rows")
        if isinstance(rows, list):
            names = [str(r.get("symbol")) for r in rows if isinstance(r, dict) and r.get("symbol")]
            if names:
                self.last_shortlist = names
        if question and self.last_answer is not None:
            self.conversation.append((question, getattr(self.last_answer, "text", "") or ""))

    def resolve_followup_symbol(self, text: str) -> str | None:
        """After a screen, `now size SYM42` resolves against the last shortlist."""
        if not self.last_shortlist:
            return None
        upper = text.upper()
        hits = [s for s in self.last_shortlist if s.upper() in upper]
        return hits[0] if len(hits) == 1 else None


def run_shell(
    args: argparse.Namespace | None = None,
    *,
    input_fn: Callable[[str], str] = input,
) -> int:
    """Run the interactive research shell with project status line and tab completion."""
    import os

    from .cli import PROMPT, ProjectError, _session, resolve_data, say, status_line
    from .commands import COMMANDS
    from .complete import enable, verbs_of
    from .project import find_project
    from .tui.dispatch import Desk, submit

    ns = args or argparse.Namespace(url=None, key=None, project=None, data=None, universe=None, symbol=None)
    session, url = _session(getattr(ns, "url", None), getattr(ns, "key", None))
    data: Any = None
    backtest_fn: Any = None
    loaded: str | None = None
    try:
        data, backtest_fn = resolve_data(ns, session)
    except (ProjectError, ValueError) as exc:
        say(str(exc))
    if getattr(ns, "project", None):
        loaded = str(ns.project)
    elif getattr(ns, "data", None):
        loaded = f"data:{ns.data}"

    keyed = bool(getattr(ns, "key", None) or os.environ.get("QUANTOS_API_KEY"))
    proj = find_project()
    symbols = [str(k) for k in data] if isinstance(data, dict) else None
    enable(verbs=verbs_of(COMMANDS, extra=SLASH), symbols=symbols, proj=proj)

    desk = Desk(
        session=session,
        url=url,
        args=ns,
        data=data,
        backtest_fn=backtest_fn,
        loaded=loaded,
        keyed=keyed,
    )

    say(status_line(url, data=desk.data, project=desk.loaded, keyed=desk.keyed))
    while True:
        try:
            line = input_fn(f"{PROMPT} ")
        except (EOFError, KeyboardInterrupt):
            say("")
            break
        if submit(desk, line):
            break
    return 0
