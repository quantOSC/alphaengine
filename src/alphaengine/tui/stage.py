"""The live instrument. A surface, a scan, and the tool stack as it is built.

The surface is the same parameter field the boot canvas draws. While a
scenario runs it moves faster, and the stack under it is the trace: each
tool lights a row when it returns. Nothing here is a progress bar. The scan
does not know how long the next call will take.
"""

from __future__ import annotations

from ..cli import _pulse, surface_lines

_MARK = "▸"


def stage_lines(
    t: float,
    *,
    activity: str = "",
    stack: list[str] | None = None,
    rows: int = 6,
    cols: int = 64,
    version: str = "1.0",
) -> list[str]:
    """One frame. `t` is seconds. A real terminal colours the surface."""
    moving = t * (1.8 if activity else 0.45)
    field = surface_lines(moving, rows=rows, cols=cols)
    beam = _pulse(t * (1.4 if activity else 0.6), width=22)
    label = activity or "standing by"
    head = f"AE {version}    {beam}    {label}"
    lines = [head, *field]
    rows_stack = list(stack or [])
    if not rows_stack:
        lines.append("  ·  the trace draws here")
        return lines
    shown = rows_stack[-4:]
    start = len(rows_stack) - len(shown) + 1
    last = len(shown) - 1
    for i, item in enumerate(shown):
        mark = _MARK if i == last and activity else "·"
        lines.append(f"  {mark}  {start + i:02d}  {item}")
    return lines
