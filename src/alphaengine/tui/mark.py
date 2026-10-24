"""The idle seal. A ring turns around a fixed wordmark."""

from __future__ import annotations

import math

from rich.text import Text

from .._version import __version__

_TAG = "the research loop, on your machine"
_NOTICE = "Research tool, not investment advice."
_HINT = "use /details for a look at important commands"
_WORD = " ".join("ALPHAENGINE")

# Housekeeping only. The tools stay off this card.
_ROWS = (
    ("login", "sign in to the QuantOS account"),
    ("logout", "remove the stored account key"),
    ("key", "paste a model key"),
    ("load <name>", "load a book stored on the account"),
    ("view", "the books on this account"),
    ("status", "what is signed in and loaded"),
    ("quit", "leave the session"),
)


def seal(t: float) -> Text:
    """One frame. `t` is seconds. The wordmark does not move."""
    cols, rows = 29, 11
    cx = cols // 2
    cy = rows // 2
    n = 32
    cells = [[" " for _ in range(cols)] for _ in range(rows)]
    ink = [[0 for _ in range(cols)] for _ in range(rows)]
    lead = int(t * 6.0) % n
    order = sorted(range(n), key=lambda i: (i - lead) % n == 0)
    for i in order:
        ang = (2.0 * math.pi * i / n) - math.pi / 2.0
        x = int(round(cx + math.cos(ang) * 12.0))
        y = int(round(cy + math.sin(ang) * 4.0))
        if not (0 <= x < cols and 0 <= y < rows):
            continue
        dist = (i - lead) % n
        if dist == 0:
            cells[y][x] = "*"
            ink[y][x] = 3
        elif dist < 3 and ink[y][x] < 2:
            cells[y][x] = "o"
            ink[y][x] = 2
        elif ink[y][x] < 1:
            cells[y][x] = "."
            ink[y][x] = 1
    spin = int(-t * 3.0) % 8
    for i in range(8):
        ang = (2.0 * math.pi * i / 8.0) + math.pi / 8.0
        x = int(round(cx + math.cos(ang) * 4.0))
        y = int(round(cy + math.sin(ang) * 1.5))
        if not (0 <= x < cols and 0 <= y < rows) or ink[y][x]:
            continue
        cells[y][x] = "+" if i == spin else "·"
        ink[y][x] = 3 if i == spin else 1
    for offset, ch in ((-1, "A"), (0, "E")):
        cells[cy][cx + offset] = ch
        ink[cy][cx + offset] = 4

    styles = {1: "#3a555c", 2: "#7ec8c4", 3: "#e8b15a", 4: "bold #e8eef2"}
    out = Text(justify="center")
    for y in range(rows):
        for x in range(cols):
            level = ink[y][x]
            if level:
                out.append(cells[y][x], style=styles[level])
            else:
                out.append(cells[y][x])
        out.append("\n")
    out.append("\n")
    out.append(_WORD + "\n", style="bold #e8eef2")
    out.append(__version__ + "\n", style="#9ecbff")
    out.append(_TAG + "\n", style="#6a9ea3")
    out.append(_NOTICE + "\n", style="#6b7380")
    out.append(_HINT, style="#8b949e")
    return out


def details_text() -> Text:
    """The card behind `/details`. Account, books, and leaving. Not the tools."""
    out = Text()
    for verb, note in _ROWS:
        out.append(f"{verb:<14}", style="bold #e8eef2")
        out.append(note + "\n", style="#8b949e")
    out.append("\n")
    out.append("yes writes a report to Downloads.", style="#6b7380")
    return out
