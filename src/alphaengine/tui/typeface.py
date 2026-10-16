"""Type scale for the session trace.

The terminal has one face. Hierarchy is weight, tracking, and the existing
palette: teal for a running step, amber for an index, light for the words.
"""

from __future__ import annotations

from rich.text import Text

SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def user_line(text: str) -> Text:
    line = Text()
    line.append("  ASK   ", style="bold #7ec8c4")
    line.append(text, style="#e6edf3")
    return line


def tool_line(index: str, tool: str, rest: str) -> Text:
    line = Text()
    line.append(f"  {index}   ", style="bold #e8b15a")
    line.append(tool.upper(), style="bold #9ecbff")
    detail = " ".join(rest.split())
    if detail:
        line.append("\n")
        line.append("        ", style="#6b7380")
        line.append(detail, style="#c5cdd6")
    return line


def body_line(text: str) -> Text:
    line = Text()
    line.append("  ", style="#e6edf3")
    line.append(text, style="#e6edf3")
    return line


def meta_line(text: str) -> Text:
    return Text(f"  {text}", style="italic #8b949e")


def spin_line(glyph: str, label: str) -> Text:
    line = Text()
    line.append(f"  {glyph}   ", style="bold #e8b15a")
    line.append(label, style="#9ecbff")
    return line
