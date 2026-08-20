"""Research session state for the terminal OS.

Holds data, book, last run, pinned model, conversation turns. The REPL in
`cli.py` owns the prompt loop so `say` / `boot` stay in one place; this module
is the memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .book import Book
from .events import EventSink

__all__ = ["SessionState", "SLASH"]

#: Verbs that also work with a leading `/`, Claude-Code style.
SLASH = (
    "login",
    "load",
    "demo",
    "process",
    "help",
    "model",
    "models",
    "key",
    "data",
    "universe",
    "project",
    "book",
    "run",
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
