"""`alphaengine` — the terminal entry point.

    $ alphaengine                      # a session
    $ alphaengine workflows            # what the server offers
    $ alphaengine run validate_study --project research.momentum
    $ alphaengine run screen_universe --data prices.csv --universe core
    $ alphaengine version

── THE GAP THIS CLOSES ────────────────────────────────────────────────────────

The harness was complete on both sides and NOTHING COULD START A RUN. The server
held the workflow, this package held the executor, and a quant who wanted to use
either had to write Python to wire them together. The consumer was built before
the producer, twice over.

── WHY THIS IS NOT A STANDALONE BINARY, AND CANNOT BE ─────────────────────────

`compute.*` ops run IN-PROCESS against your DataFrames. That forces the whole
shape of this tool:

    * it must be installed in the venv you do research in, NOT with pipx or
      `uv tool install` — those isolate the package from the libraries and data
      the ops need to touch, and isolation is the one thing we cannot have;
    * it is therefore on PATH only when that venv is active, which is ordinary
      Python friction and is the price of your data never moving;
    * two dependencies, numpy and scipy, both already in every environment we
      target, so installing this into a working research venv changes nothing
      about that venv.

You cannot have tool isolation AND in-process data access. Everything else here
follows from choosing the second.

── WHERE YOUR DATA AND YOUR BACKTEST COME FROM ────────────────────────────────

THREE DOORS, and the third one only says which names:

    --project research.momentum     a module exposing `data` and `backtest_fn`
    --data    prices.csv            a local file: wide, long, or one series
    --universe core                 a universe registered in the portal. Brings
                                    its stored closes if you saved them there;
                                    otherwise narrows --data to its names

`--project` is the ONLY door that can supply a simulator, because a simulator is
code, which is why `validate_study` needs it and the other three workflows do
not.

`--universe` is the one that needs a word about §9. The rule is that WE never
fetch market data on your behalf and never hold a series as our own. Pulling
back closes YOU uploaded, encrypted at rest, into the same account that put them
there is neither: nothing new crosses a boundary, a thing you already own moves
between two of your own surfaces. Refusing that would not have been a data
boundary, it would have been a missing function wearing one — upload an S&P
universe to the portal, then be told the OS cannot see it.

A local `--data` file still WINS over the stored copy, because it is the more
explicit statement.

A PROJECT MODULE that you write and we import: an ordinary Python module
exposing `data` and, if a workflow sweeps, `backtest_fn`.

    # research/momentum.py
    import pandas as pd
    data = pd.read_parquet("prices.parquet")

    def backtest_fn(*, data, fast, slow):
        ...
        return returns

    $ alphaengine run validate_study --project research.momentum

A module rather than a config file because the thing being supplied is CODE and
a path to it — and because a quant already has this module; it is the notebook
cell they were going to run anyway.

── WHAT THE LOOP LOOKS LIKE FROM HERE ─────────────────────────────────────────

The server decides, this machine executes, figures go back. Every step is
printed as it happens, because a loop you cannot watch is a loop you cannot
trust:

    server →  compute.sweep
    local  ·  240 configs .......... 12s
    server →  compute.deflated_sharpe
    local  ·  DSR 0.41 · n_trials 240 derived
    server →  Stop: the surface is a knife edge

NO LLM DEPENDENCY, AND NO KEY FIELD. `AgentDriver` takes a callable, so bring
your own model by passing a function. There is nowhere in this tool to put a
model key, ours or yours, and that is how "runs under your own account" is
satisfied structurally rather than promised.
"""

from __future__ import annotations

import argparse
import contextvars
import importlib
import json
import math
import os
import pathlib
import re
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from ._version import API_URL_ENV, DEFAULT_API_URL, LEGACY_API_URL_ENV

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from .commands import Command

__all__ = ["main"]

_ENV_KEY = "QUANTOS_API_KEY"
_ENV_URL = LEGACY_API_URL_ENV
_ENV_API_URL = API_URL_ENV
DEFAULT_BASE_URL = DEFAULT_API_URL


def resolve_api_url(explicit: str | None = None) -> str:
    """Resolve the API URL: explicit argument, ``ALPHAENGINE_API_URL``, ``QUANTOS_API_URL``, then default."""
    return explicit or os.environ.get(_ENV_API_URL) or os.environ.get(_ENV_URL) or DEFAULT_BASE_URL


#: WHERE A RUN IS READ. The CLI printed a bare run id and left the reader to
#: work out that `/portal/runs/<id>` exists — and named four screens that had
#: been redirects for months besides ("Work", "the sleeve page", "your Book",
#: "accept it in the portal", the last pointing at a surface that does not
#: exist at all). An id nobody can turn into a URL is a reference to nothing.
#:
#: Overridable, and derived from the API host when that has been pointed
#: somewhere else, so a self-hosted deployment does not send its users to ours.
DEFAULT_PORTAL_URL = "https://qosai.app"


def portal_url(api_url: str | None = None) -> str:
    """The portal this account's runs live on.

    `$QUANTOS_PORTAL_URL` wins. Otherwise the public portal — EXCEPT when the
    API has been pointed at localhost, where sending somebody to the hosted
    portal to read a run that exists only on their machine is the config-default
    failure this codebase has already paid for four times: present, wrong, and
    answering every check.
    """
    override = os.environ.get("QUANTOS_PORTAL_URL")
    if override:
        return override.rstrip("/")
    host = resolve_api_url(api_url).lower()
    if "localhost" in host or "127.0.0.1" in host:
        return "http://localhost:3000"
    return DEFAULT_PORTAL_URL


# ── output ─────────────────────────────────────────────────────────────────
#
# ANSI only when the stream is a TTY. Piping `alphaengine run` into a file or a
# CI log must produce plain text, not escape codes — a run record with colour
# bytes in it is a run record somebody has to clean up before reading.


def _tty() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _truecolor() -> bool:
    """24-bit colour when the terminal actually advertised it.

    Guessing truecolor from `TERM=xterm-256color` paints wrong on a 256-colour
    screen. `COLORTERM` is the signal terminals set on purpose.
    """
    return os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit")


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _tty() else text


def dim(t: str) -> str:
    return _c("2", t)


def bold(t: str) -> str:
    return _c("1", t)


def red(t: str) -> str:
    return _c("31", t)


def green(t: str) -> str:
    return _c("32", t)


def yellow(t: str) -> str:
    return _c("33", t)


def cyan(t: str) -> str:
    return _c("36", t)


def _rgb(r: int, g: int, b: int, text: str) -> str:
    """A single-character (or short) paint. Falls back to the 256-colour cube."""
    if not _tty() or not text:
        return text
    if _truecolor():
        return f"\033[38;2;{r};{g};{b}m{text}\033[0m"
    qr, qg, qb = min(5, r * 6 // 256), min(5, g * 6 // 256), min(5, b * 6 // 256)
    return f"\033[38;5;{16 + 36 * qr + 6 * qg + qb}m{text}\033[0m"


def _mix(start: tuple[int, int, int], end: tuple[int, int, int], t: float, ch: str) -> str:
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    r = int(start[0] + (end[0] - start[0]) * t)
    g = int(start[1] + (end[1] - start[1]) * t)
    b = int(start[2] + (end[2] - start[2]) * t)
    return _rgb(r, g, b, ch)


def _accent(text: str) -> str:
    """The one brand colour. Warm gold, the wait-star Claude Code taught people
    to read as 'something is happening'."""
    return _rgb(255, 176, 72, text)


_HUES: tuple[tuple[int, int, int], ...] = (
    (255, 176, 72),
    (255, 120, 80),
    (220, 110, 190),
    (130, 150, 255),
    (64, 200, 210),
    (90, 220, 140),
    (210, 220, 80),
)


def _spin_mark(frame: int) -> str:
    r, g, b = _HUES[frame % len(_HUES)]
    return _rgb(r, g, b, STAR)


# ── glyphs, which are not always available ─────────────────────────────────
#
# The same discipline as the colour above, applied to CHARACTERS. This CLI uses
# `·` and `→` as separators, and on Windows `sys.stdout` defaults to the
# console's legacy code page — cp437 cannot encode either one, and cp1252
# encodes `·` to a byte that anything reading the log as UTF-8 renders as `�`.
#
# Observed, not theorised: `alphaengine run validate_study` on a stock Windows
# Python printed `validate_study � https://...`.
#
# Forcing UTF-8 onto the stream is the tempting fix and the wrong one — it
# overrides a choice the terminal made, and on a console that genuinely cannot
# render the glyph it trades one mojibake for another. So we ask the stream what
# it can encode, once, and fall back to ASCII when the answer is no. A separator
# is decoration; the words either side carry the meaning.


def _stream_handles(text: str) -> bool:
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        text.encode(enc)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


_UNICODE_OK = _stream_handles("·→")

#: Mid-dot separator, or an ASCII hyphen where the stream cannot take one.
DOT = "·" if _UNICODE_OK else "-"
#: Rightwards arrow, or the ASCII spelling.
ARROW = "→" if _UNICODE_OK else "->"
#: A lit and an unlit marker. Both surfaces that rank things by "can you do this
#: yet" use the same pair, because they are answering the same question.
ON = "●" if _UNICODE_OK else "*"
OFF = "○" if _UNICODE_OK else "-"
#: A failed step's marker.
CROSS = "×" if _UNICODE_OK else "x"
#: Nested detail, Codex-style. A step that has already run, never what comes next.
NEST = "└" if _UNICODE_OK else "\\"
#: REPL composer. Claude Code uses `>`; we keep a heavier caret so a research
#: session does not look like a shell waiting for bash.
PROMPT = "❯" if _UNICODE_OK else ">"
#: The wait-mark Claude Code popularised: a star that changes colour while work
#: is blocked. ASCII `*` where the stream cannot take it.
STAR = "✻" if _UNICODE_OK else "*"
#: Wordmark. The product is the README banner in a terminal: tracked
#: ALPHAENGINE over a teal-to-amber mesh, not a mascot. `ae` is the ASCII
#: fallback for streams that cannot take the Greek letter.
PIN = "α" if _stream_handles("α") else "ae"

#: The questions, as commands: what a quant TYPES maps to what the server
#: RUNS. One table, read by the parser, the dispatcher and the shell alike.
QUESTION_VERBS: dict[str, str] = {
    "diagnose": "diagnose_data",
    "screen": "screen_universe",
    "signal": "evaluate_signal",
    "validate": "validate_study",
    "stress": "stress_study",
    "overlap": "check_overlap",
    "size": "size_position",
    "monitor": "monitor_sleeve",
}
#: The same table read backwards, so a gap naming a WORKFLOW can print the verb
#: a person actually types. Derived rather than written out twice, because two
#: hand-maintained copies of this map is precisely the drift `commands.py`
#: exists to prevent.
QUESTION_FOR_WORKFLOW: dict[str, str] = {w: v for v, w in QUESTION_VERBS.items()}
#: What each workflow answers, in a reader's words. The server publishes a name.
#: A name is not a choice. These sentences describe the choice and say nothing
#: about the order of steps inside it. The model sees the same sentences.
_WORKFLOW_ANSWERS: dict[str, str] = {
    "diagnose_data": "can I trust this data",
    "screen_universe": "what in my universe is worth a look",
    "evaluate_signal": "does this signal carry information",
    "validate_study": "does this result hold up once you count the tries",
    "stress_study": "where does it break",
    "check_overlap": "is it new, or my book again",
    "size_position": "what sizing does the risk model imply",
    "monitor_sleeve": "has anything crossed a line",
}
#: The held-back-rows marker. Probed like the rest: cp1252 renders a literal
#: ellipsis as mojibake in exactly the place a user reads their result.
ELLIPSIS = "…" if _UNICODE_OK else "..."


_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_SINK: contextvars.ContextVar[Callable[[str], None] | None] = contextvars.ContextVar(
    "alphaengine_narration", default=None
)


@contextmanager
def narration_to(sink: Callable[[str], None]) -> Iterator[None]:
    """Send `say` and the working line to `sink` instead of the console.

    The full-screen session reuses this module's run loop. A spinner written
    to stdout would paint over that session, so the sink replaces both.
    """
    token = _SINK.set(sink)
    try:
        yield
    finally:
        _SINK.reset(token)


def say(*parts: str) -> None:
    sink = _SINK.get()
    if sink is not None:
        sink(_ANSI.sub("", " ".join(parts)))
        return
    print(*parts, flush=True)


#: Spinner frames. Braille where the stream takes it, ASCII where it does not —
#: the same degradation `DOT` and `ARROW` get, for the same reason: a Windows
#: cp1252 console rendered the pretty version as mojibake.
_SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏" if _UNICODE_OK else "|/-\\"
_SPARK = "▁▂▃▄▅▆▇█" if _UNICODE_OK else ".:-=*#"
_FILL = "█" if _UNICODE_OK else "#"
_EMPTY = "░" if _UNICODE_OK else "."


def sparkline(values: Sequence[float], *, width: int | None = None) -> str:
    """A one-line plot of `values`, resampled to `width` when given.

    Height is relative to the sample itself. An absolute scale would pretend we
    know the ceiling; we do not. Used for the boot surface, the working pulse,
    and a recap of trial Sharpes — never for a progress bar.
    """
    if not values:
        return ""
    n = len(values)
    w = n if width is None else max(1, int(width))
    sampled: list[float] = []
    for i in range(w):
        idx = int(round(i * (n - 1) / max(w - 1, 1))) if w > 1 else 0
        sampled.append(float(values[idx]))
    lo, hi = min(sampled), max(sampled)
    span = hi - lo
    last = len(_SPARK) - 1
    out: list[str] = []
    for v in sampled:
        t = 0.5 if span == 0 else (v - lo) / span
        out.append(_SPARK[min(last, int(t * last + 1e-9))])
    return "".join(out)


def _spark_paint(values: Sequence[float], *, width: int, cool: bool = True) -> str:
    """Colour each bar by its height so a plateau reads as a ridge, not a fence."""
    body = sparkline(values, width=width)
    if not _tty():
        return body
    last = max(len(_SPARK) - 1, 1)
    start = (24, 92, 88) if cool else (92, 36, 16)
    end = (130, 255, 196) if cool else (255, 176, 64)
    parts: list[str] = []
    for ch in body:
        rank = _SPARK.find(ch)
        t = (rank if rank >= 0 else 0) / last
        parts.append(_mix(start, end, t, ch))
    return "".join(parts)


def _bar(value: float, peak: float, *, width: int = 14) -> str:
    """A relative meter. `peak` is the largest number on this screen, not 100%."""
    filled = 0 if peak <= 0 else max(0, min(width, int(round(width * abs(value) / peak))))
    body = _FILL * filled + _EMPTY * (width - filled)
    if not _tty():
        return body
    if value < 0:
        return red(body)
    return cyan(body) if filled < width else green(body)


def _pulse(elapsed: float, *, width: int = 14) -> str:
    """A travelling heartbeat. Deliberately NOT a progress bar: nothing here
    knows how long the step will take, and a bar that fills at a guessed rate
    is a lie told smoothly."""
    return sparkline(_scan_wave(elapsed, width), width=width)


def _scan_wave(elapsed: float, width: int) -> list[float]:
    """A plateau moving across the line. The peak wraps; duration is unknown."""
    mu = (elapsed * 0.42) % 1.0
    denom = max(width - 1, 1)
    return [
        math.exp(-0.5 * ((i / denom - mu) / 0.14) ** 2)
        + 0.28 * math.exp(-0.5 * ((i / denom - ((mu + 0.52) % 1.0)) / 0.10) ** 2)
        for i in range(width)
    ]


def _live_tty() -> bool:
    """Motion only on a real terminal.

    Tests monkeypatch `_tty` so colour and glyphs can be asserted. Sleeping
    through boot frames or a travelling mesh would stall pytest. A real
    `isatty()` is the signal; `PYTEST_CURRENT_TEST` is the belt.
    """
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    if os.environ.get("NO_COLOR") is not None:
        return False
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def _canvas_cols() -> int:
    try:
        n = os.get_terminal_size().columns
    except OSError:
        n = 80
    return max(48, min(72, n - 6))


CANVAS_ROWS = 12
_MESH_OK = _stream_handles("╱╲")
_SLOPE_UP = "╱" if _MESH_OK else "/"
_SLOPE_DN = "╲" if _MESH_OK else "\\"


def _height_at(x: float, z: float, t: float) -> float:
    """Banner topography: plateau left, valley, knife-edge ridge on the right."""
    plateau = math.exp(-((x - 0.30) ** 2) / 0.085 - ((z - 0.58) ** 2) / 0.16)
    valley = -0.22 * math.exp(-((x - 0.56) ** 2) / 0.028 - ((z - 0.50) ** 2) / 0.20)
    ridge = math.exp(-((x - 0.80) ** 2) / 0.0032 - ((z - 0.38) ** 2) / 0.13)
    ripple = math.sin(2 * math.pi * (1.25 * x + 0.85 * z - 0.16 * t))
    ripple *= 0.10 * math.exp(-((z - 0.48) ** 2) / 0.40)
    # Recede into a dark sky at the back, the way the README banner does.
    sky = 0.22 + 0.78 * z
    h = (0.08 + 0.60 * plateau + 0.86 * ridge + valley + ripple) * sky
    return 0.0 if h < 0.0 else 1.0 if h > 1.0 else h


def _teal(text: str) -> str:
    """The banner's secondary colour: subtitle and data-boundary line."""
    return _rgb(64, 200, 210, text)


def _paint_cell(h: float, dhx: float, z: float, x: float) -> str:
    """Teal on the plateau, amber on the ridge: the README banner, in glyphs."""
    last = len(_SPARK) - 1
    steep = abs(dhx) > 0.09
    on_x = abs((x * 18.0) % 1.0) < 0.055
    on_z = abs((z * 9.0) % 1.0) < 0.11
    if steep:
        ch = _SLOPE_UP if dhx > 0 else _SLOPE_DN
    elif on_x and on_z and _MESH_OK:
        ch = "·"
    else:
        ch = _SPARK[min(last, int(h * last + 1e-9))]
    if not _tty():
        return ch
    # Hue is LEFT-RIGHT (teal → amber), brightness is HEIGHT. No mid-green.
    lo = (8, int(36 + 28 * z), int(44 + 36 * z))
    hi_l = (56, 214, 214)
    hi_r = (255, 168, 48)
    hi = (
        int(hi_l[0] * (1.0 - x) + hi_r[0] * x),
        int(hi_l[1] * (1.0 - x) + hi_r[1] * x),
        int(hi_l[2] * (1.0 - x) + hi_r[2] * x),
    )
    return _mix(lo, hi, h, ch)


def surface_lines(t: float = 0.0, *, rows: int = CANVAS_ROWS, cols: int | None = None) -> list[str]:
    """A living parameter surface. The thing this tool actually judges.

    Perspective: back rows indent and narrow so the plateau reads as a volume,
    not a stack of sparklines. Every line is padded to `cols` visible cells
    so a CSI erase of the block stays rectangular.
    """
    cols = _canvas_cols() if cols is None else max(16, cols)
    out: list[str] = []
    for zi in range(rows):
        z = zi / max(rows - 1, 1)
        indent = int((1.0 - z) * 4)
        width = max(12, cols - indent * 2)
        cells: list[str] = []
        row_h: list[float] = []
        for xi in range(width):
            x = xi / max(width - 1, 1)
            h = _height_at(x, z, t)
            row_h.append(h)
            left = row_h[xi - 1] if xi else h
            cells.append(_paint_cell(h, h - left, z, x))
        body = " " * indent + "".join(cells)
        pad = cols - _visible_len(body)
        out.append(body + " " * max(0, pad))
    return out


def _pad_to(text: str, width: int) -> str:
    vis = _visible_len(text)
    extra = max(0, width - vis)
    left = extra // 2
    return " " * left + text + " " * (extra - left)


def _wordmark_lines(width: int) -> list[str]:
    """The README banner type, in the terminal: tracked wordmark, one subtitle."""
    title = " ".join("ALPHAENGINE")
    sub = "the research loop, on your machine"
    return [
        _pad_to(bold(title), width),
        _pad_to(_teal(sub), width),
    ]


def ridge_lines(values: Sequence[float], *, rows: int = 6, cols: int = 40) -> list[str]:
    """A filled mountain from real numbers, used to recap a grid of Sharpes."""
    if not values:
        values = [0.0]
    n = len(values)
    sampled: list[float] = []
    for i in range(cols):
        idx = int(round(i * (n - 1) / max(cols - 1, 1))) if cols > 1 else 0
        sampled.append(float(values[idx]))
    lo, hi = min(sampled), max(sampled)
    span = hi - lo or 1.0
    norm = [(v - lo) / span for v in sampled]
    out: list[str] = []
    last = len(_SPARK) - 1
    for zi in range(rows):
        floor = 1.0 - (zi + 1) / rows
        cells: list[str] = []
        for v in norm:
            h = 0.0 if v <= floor else min(1.0, (v - floor) * rows)
            ch = _SPARK[min(last, int(h * last + 1e-9))]
            cells.append(_mix((24, 92, 88), (130, 255, 196), h, ch) if _tty() else ch)
        out.append("".join(cells))
    return out


def _write_block(lines: Sequence[str], *, replace: int = 0) -> int:
    """Paint `lines` in place. `replace` is how many rows to rewind first."""
    if replace:
        sys.stdout.write(f"\r\033[{replace}A\033[J")
    for ln in lines:
        sys.stdout.write(ln + "\n")
    sys.stdout.flush()
    return len(lines)


def _play_canvas(*, frames: int = 12) -> None:
    """Boot sculpture. Sleeps only on a live TTY; tests get one still frame."""
    cols = _canvas_cols()
    rows = CANVAS_ROWS

    def block(t: float) -> list[str]:
        return ["  " + ln for ln in surface_lines(t, rows=rows, cols=cols)]

    if _live_tty() and frames > 1:
        painted = 0
        for i in range(frames):
            painted = _write_block(block(i * 0.09), replace=painted)
            time.sleep(0.05)
        return
    if _tty():
        _write_block(block(0.0))


class Working:
    """An in-place canvas for work that blocks, with the elapsed time.

    ── WHY THIS EXISTS ────────────────────────────────────────────────────────

    Every narration in this file was `print()`, which meant that each of the
    three things that actually take time — the model call, the network round
    trip, and the in-process compute of a step — ran in total silence. The
    screen showed the line BEFORE a step and then nothing until the step
    returned, so a thirty-second sweep and a hung process looked identical.

    For a product whose claim is that the research loop got shorter, being
    unable to show the loop running is not a cosmetic gap. The owner's words
    were "no animation going on showing status of work being done".

    ── WHAT IT DOES NOT DO ────────────────────────────────────────────────────

    It never invents progress. There is no percentage and no bar, because
    nothing here knows how long a step will take and a bar that fills at a
    guessed rate is a lie told smoothly. It reports two true things: that work
    is still running, and how long it has been running.

    NOT A TTY, NOT A SPINNER. A CI log or a piped run gets one plain line at the
    start and the ordinary completion line at the end — carriage returns in a
    transcript are noise, and the transcript is the artifact there.

    MOTION IS GATED ON `_live_tty()`. Tests that monkeypatch `_tty` still see
    colour and glyphs, but they must not wait on frames.
    """

    def __init__(self, label: str, *, plain: str = "") -> None:
        self.label = label
        #: The label with no escape codes, so the line can be erased by writing
        #: spaces over it. Deliberately not derived by stripping ANSI: the caller
        #: knows what it wrote, and a regex that misses one sequence leaves
        #: fragments on screen.
        self.plain = plain or label
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self._nlines = 0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started

    def _block(self, frame: int) -> list[str]:
        t = f"{self.elapsed:.1f}s"
        if _UNICODE_OK:
            cols = _canvas_cols()
            lines = [f"  {ln}" for ln in surface_lines(self.elapsed, rows=CANVAS_ROWS, cols=cols)]
            lines.append(f"  {_spin_mark(frame)}  {self.label}  {dim(t)}")
            return lines
        return [f"  {_SPIN[frame % len(_SPIN)]} {self.label}  {t}"]

    def __enter__(self) -> Working:
        self._started = time.monotonic()
        sink = _SINK.get()
        if sink is not None:
            # The session paints this on the status line. It is not a trace line.
            sink("\x1e" + self.plain)
            return self
        if not _tty():
            return self
        self._nlines = _write_block(self._block(0))
        if _live_tty():
            self._thread = threading.Thread(target=self._animate, daemon=True)
            self._thread.start()
        return self

    def _animate(self) -> None:
        i = 1
        while not self._stop.wait(0.07):
            self._nlines = _write_block(self._block(i), replace=self._nlines)
            i += 1

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if self._nlines and _tty():
            sys.stdout.write(f"\r\033[{self._nlines}A\033[J")
            sys.stdout.flush()


# ── the boot screen ────────────────────────────────────────────────────────
#
# WHAT A BOOT SCREEN IS FOR, and it is not decoration. Somebody typing
# `alphaengine` in a project has three questions and no way to answer them
# without one: am I pointed at the right workspace, can I actually reach it,
# and is my data loaded. A banner that answers those is a banner that saves a
# failed run; a banner that just says the name is a logo.
#
# So the box below is mostly STATE. The wordmark is three lines of it and the
# rest is: where it is talking to, whether it is authenticated, which project
# module is loaded, and what to type next. Anything that cannot be answered
# gets a visible warning rather than silence, because "no API key" discovered
# at boot costs nothing and discovered mid-run costs the run.
#
# It degrades the whole way down: box-drawing to ASCII when the stream cannot
# encode it, colour dropped when stdout is not a TTY, and the entire banner
# suppressed for a non-interactive invocation — piping `alphaengine run` into a
# CI log must produce a transcript, not an ANSI painting.

# THE MARK IS A PARAMETER SURFACE, which is the one thing this tool looks at.
#
# Figlet art was the first attempt and it was wrong for this product — a bank of
# ASCII capitals is a hobby-project signal, and it says nothing. A plateau says
# what a good result looks like: a broad region where neighbouring parameters
# all work. Beside it, the knife edge that does not survive. Somebody who boots
# this twice has already learned the distinction the verdict turns on.
#
# Two glyph sets. The blocks need Unicode; the ASCII fallback keeps the same
# silhouette so the meaning survives a cp437 console.
# The two are the SAME WIDTH on purpose — they sit on consecutive lines and the
# text after them has to start at the same column, or the second line reads as a
# misprint rather than as a comparison.
_PLATEAU_U = "▁▂▄▆█████▆▄▂▁"
_EDGE_U = "▁▁▁▁▁▁█▁▁▁▁▁▁"
_PLATEAU_A = ".:-=+#####+=-:"
_EDGE_A = "..........#..."

PLATEAU = _PLATEAU_U if _UNICODE_OK else _PLATEAU_A
EDGE = _EDGE_U if _UNICODE_OK else _EDGE_A


def _boxed(rows: list[tuple[str, str]], width: int = 62, *, title: str = "") -> list[str]:
    """A key/value box. Unicode rules when the stream takes them, ASCII when not."""
    if _UNICODE_OK:
        tl, tr, bl, br, h, v = "╭", "╮", "╰", "╯", "─", "│"
    else:
        tl, tr, bl, br, h, v = "+", "+", "+", "+", "-", "|"
    inner = width - 2
    if title:
        label = f"{h} {title} "
        rest = inner - _visible_len(label)
        out = [dim(tl) + dim(label) + dim(h * max(0, rest) + tr)]
    else:
        out = [dim(tl + h * inner + tr)]
    for k, val in rows:
        if k == "":  # a full-width line, already coloured by the caller
            pad = inner - 1 - _visible_len(val)
            out.append(dim(v) + " " + val + " " * max(0, pad) + dim(v))
            continue
        label = k.ljust(9)
        room = inner - 1 - 9 - 1
        val = _fit(val, room)
        pad = room - _visible_len(val)
        out.append(dim(v) + f" {dim(label)} {val}" + " " * max(0, pad) + dim(v))
    out.append(dim(bl + h * inner + br))
    return out


def _fit(s: str, room: int) -> str:
    """Truncate to `room` VISIBLE characters, keeping ANSI codes intact.

    The default server URL is 51 characters and overflowed the box on a fresh
    install — the first thing an unconfigured user would ever see, broken. The
    tail is what distinguishes one host from another, so the middle goes.
    """
    if _visible_len(s) <= room:
        return s
    ell = ELLIPSIS
    keep = room - len(ell)
    head, tail = keep // 2, keep - keep // 2
    plain, out, seen, i = "", [], 0, 0
    while i < len(s):
        if s[i] == "\033":  # copy the escape verbatim, it costs no width
            j = i
            while j < len(s) and s[j] != "m":
                j += 1
            out.append(s[i : j + 1])
            i = j + 1
            continue
        if seen < head or seen >= _visible_len(s) - tail:
            out.append(s[i])
        elif seen == head:
            out.append(ell)
        seen += 1
        i += 1
    del plain
    return "".join(out)


def _visible_len(s: str) -> int:
    """Length ignoring ANSI escapes, so padding survives colour."""
    n, i = 0, 0
    while i < len(s):
        if s[i] == "\033":
            while i < len(s) and s[i] != "m":
                i += 1
            i += 1
            continue
        n += 1
        i += 1
    return n


# ── the project module ─────────────────────────────────────────────────────


class ProjectError(RuntimeError):
    """The project module could not be loaded, or does not expose what a run needs."""


#: The parameter grid the loaded project declared, if any. Set by
#: `load_project`, read by `project_grid()` at the three places a run opens.
#: The grid is the CALLER'S input — their parameter space, the thing the trial
#: count is derived from — and until this existed it never left the module:
#: `validate_study` received `grid={}` and the sweep crashed on it.
_PROJECT_GRID: dict[str, Any] | None = None


def project_grid() -> dict[str, Any] | None:
    """The grid the last-loaded project module declared, or None."""
    return _PROJECT_GRID


def load_project(spec: str | None) -> tuple[Any, Any]:
    """Import the caller's module and take `data` / `backtest_fn` off it.

    The current directory goes on `sys.path` first, because the module being
    imported is the user's own project and not something installed. That is the
    normal shape of "run this against my repo".

    A `GRID` (or `grid`) attribute is recorded too — see `project_grid`. It is
    not part of the return shape because five call sites unpack a pair, and the
    grid is consumed in exactly one place: opening a run.
    """
    global _PROJECT_GRID
    if not spec:
        _PROJECT_GRID = None
        return None, None

    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())

    try:
        mod = importlib.import_module(spec)
    except ImportError as exc:
        raise ProjectError(
            f"could not import {spec!r} from {os.getcwd()}.\n"
            f"  Pass a MODULE PATH, not a file path: --project research.momentum"
        ) from exc

    data = getattr(mod, "data", None)
    backtest_fn = getattr(mod, "backtest_fn", None)
    grid = getattr(mod, "GRID", None) or getattr(mod, "grid", None)
    _PROJECT_GRID = dict(grid) if isinstance(grid, dict) and grid else None
    if data is None and backtest_fn is None:
        raise ProjectError(
            f"{spec} exposes neither `data` nor `backtest_fn`.\n"
            f"  A project module is an ordinary module with your prices in `data`\n"
            f"  and, if the workflow sweeps, your simulator in `backtest_fn`."
        )
    return data, backtest_fn


# ── session ────────────────────────────────────────────────────────────────


def _session(
    base_url: str | None,
    api_key: str | None,
    *,
    share_series: bool = False,
) -> tuple[Any, str]:
    """Imported lazily so `import alphaengine` still touches no networking code.

    `test_smoke.py` asserts that property, and it is worth more than the two
    lines this costs: the offline half of the package is what a researcher
    installs without asking anyone's permission.
    """
    from .client import connect

    url = resolve_api_url(base_url)
    key = api_key or os.environ.get(_ENV_KEY) or ""
    return connect(url, api_key=key or None, share_series=bool(share_series)), url


def _explain_offline(url: str) -> None:
    say(red("No workflow server at ") + bold(url) + red("."))
    say("")
    say("  The maths in this package does not need one. `sweep`, the whole")
    say("  validation path and `save` run offline, with no account:")
    say("")
    say(dim("      from alphaengine import sweep"))
    say(dim("      r = sweep(my_backtest, grid, data=prices)"))
    say(dim("      r.verdict()"))
    say("")
    say("  Workflows need the server because the SEQUENCE lives there.")


# ── commands ───────────────────────────────────────────────────────────────


def cmd_init(args: argparse.Namespace) -> int:
    """Create `.alphaengine/` in the target directory (or current directory)."""
    from .project import PROJECT_FORMAT_VERSION, init_project

    target = getattr(args, "directory", None) or "."
    name = getattr(args, "name", None)
    proj = init_project(target, name=name)
    say(
        green("Initialized.")
        + f"  {proj.dot_dir}  "
        + dim(f"({proj.name} {DOT} schema {PROJECT_FORMAT_VERSION})")
    )
    return 0


def cmd_version(_args: argparse.Namespace) -> int:
    from ._version import __version__

    say(f"alphaengine {__version__}")
    return 0


def cmd_workflows(args: argparse.Namespace) -> int:
    from .client import Offline, ServerError

    session, url = _session(args.url, args.key)
    try:
        rows = session.workflows()
    except Offline:
        _explain_offline(url)
        return 2
    except ServerError as exc:
        refused(exc)
        if is_auth_error(exc) and offer_key():
            say(dim("  Key set. Run that again."))
        return 2

    if not rows:
        say("The server offers no workflows.")
        return 0

    width = max(len(str(r.get("name"))) for r in rows)
    say("")
    for r in rows:
        name = str(r.get("name"))
        needs = ", ".join(str(n) for n in (r.get("requires") or [])) or "nothing"
        # REPRODUCIBILITY IS STATED UP FRONT, not discovered when two runs of
        # the same question disagree. Same reasoning as a trial count that says
        # whether it was derived or asserted.
        repro = green("reproducible") if r.get("reproducible") else yellow("may differ")
        say(f"  {bold(name.ljust(width))}   {_workflow_answer(r)}")
        say(f"  {' ' * width}   {dim('needs ' + needs)}  {DOT}  {repro}  {dim(str(r.get('version')))}")
    say("")
    say(dim("  alphaengine run <name> --data prices.csv   ") + dim(f"{DOT}  `commands run` for the flags"))
    return 0


def cmd_gaps(args: argparse.Namespace) -> int:
    """WHAT THE RECORD SAYS IS UNANSWERED, and what closes each one.

    `runs` shows what you already tried. This shows what you have NOT — the
    other half of the same question, and the one you want at 9am rather than at
    the end of a task.

    Every line names its remedy, because a gap that cannot name one is an
    observation rather than a next step. Both the gap and the remedy are
    DERIVED from the record: a model asked to find gaps can decline to find
    them, and can invent them.
    """
    from .client import Offline, ServerError

    session, url = _session(args.url, args.key)
    try:
        out = session.gaps()
    except Offline:
        _explain_offline(url)
        return 2
    except ServerError as exc:
        refused(exc)
        if is_auth_error(exc) and offer_key():
            say(dim("  Key set. Run that again."))
        return 2

    rows = list(out.get("queued") or []) + list(out.get("skipped") or [])
    if not rows:
        say("")
        reason = str(out.get("reason") or "")
        say(dim(f"  {reason}") if reason else dim("  Nothing open. The record answers what it was asked."))
        return 0

    say("")
    say(f"  {bold(str(len(rows)))} open on this account")
    for r in rows:
        workflow = str(r.get("workflow") or "")
        # THE SENTENCE IS THE FINDING. Printed whole rather than clipped to a
        # label, because "nobody diagnosed this data before the run read it" is
        # what tells somebody what to do and "data_never_diagnosed" is not.
        say(f"  {dim(DOT)} {str(r.get('because') or r.get('code') or '').strip()}")
        blocked = r.get("reason")
        if blocked:
            say(f"      {yellow('blocked')}  {dim(str(blocked))}")
        else:
            say(f"      {dim('->')} alphaengine {QUESTION_FOR_WORKFLOW.get(workflow, workflow)}")
    say("")
    return 0


def cmd_tonight(args: argparse.Namespace) -> int:
    """WHAT WOULD RUN TONIGHT, without running any of it.

    The same derivation the scheduler uses, so this is what WILL happen rather
    than a description of it.

    Everything the budget cut is printed too. A plan that shows only what it
    kept reads as "this is everything that mattered", which is the one thing a
    plan must never imply.
    """
    from .client import Offline, ServerError

    session, url = _session(args.url, args.key)
    try:
        out = session.plan(getattr(args, "budget", 0) or 0)
    except Offline:
        _explain_offline(url)
        return 2
    except ServerError as exc:
        refused(exc)
        if is_auth_error(exc) and offer_key():
            say(dim("  Key set. Run that again."))
        return 2

    queued = list(out.get("queued") or [])
    skipped = list(out.get("skipped") or [])
    if not queued and not skipped:
        say("")
        reason = str(out.get("reason") or "")
        say(dim(f"  {reason}") if reason else dim("  Nothing queued. The record has no open gap to close."))
        return 0

    say("")
    if queued:
        say(f"  {bold('queued')}  {dim(f'{len(queued)} of tonight')}")
        for q in queued:
            inputs = q.get("inputs") or {}
            on = str(inputs.get("universe") or inputs.get("symbol") or "")
            say(f"    {green(DOT)} {str(q.get('workflow') or '').ljust(18)}{dim(on)}")
            say(f"      {dim(str(q.get('because') or ''))}")
    if skipped:
        say("")
        say(f"  {dim('skipped')}")
        for s in skipped:
            say(f"    {dim(DOT)} {str(s.get('workflow') or '').ljust(18)}{dim(str(s.get('reason') or ''))}")
    say("")
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    """YOUR OWN WEEK, from where the work happens.

    The terminal could start a run and could not show you one. "What did I
    already try" was a question you had to open a browser to ask, from the one
    place you are least likely to want to leave — and a quant's week is fifteen
    runs of which most are stops.

    THE STOPS ARE THE POINT. They are the majority of a week's output and the
    part that evaporates everywhere else, so a stop prints its REASON rather
    than only its token: the sentence is the most useful thing the run produced.
    """
    from .client import Offline, ServerError

    session, url = _session(args.url, args.key)
    try:
        rows = session.recent_runs(limit=args.limit)
    except Offline:
        _explain_offline(url)
        return 2
    except ServerError as exc:
        refused(exc)
        if is_auth_error(exc) and offer_key():
            say(dim("  Key set. Run that again."))
        return 2

    if not rows:
        say("")
        say(dim("  Nothing recorded on this account yet."))
        say(dim("  A run files itself as it goes: ") + "alphaengine screen --universe <name>")
        return 0

    say("")
    for r in rows:
        when = str(r.get("created_at") or "")[:10]
        kind = str(r.get("kind") or "workflow")
        verdict = str(r.get("verdict") or r.get("status") or "")
        mark = (
            yellow("stopped")
            if verdict == "stopped"
            else green("closed")
            if verdict == "completed"
            else dim(verdict or "open")
        )
        line = f"  {dim(when)}  {bold(kind.ljust(16))}  {mark}"
        # THE DENOMINATOR TRAVELS WITH THE COUNT, here as everywhere. A number
        # that was counted and one that was asserted are different claims.
        if r.get("honest_n") is not None:
            src = str(r.get("n_trials_source") or "unknown").replace("_", " ")
            line += dim(f"  n={r['honest_n']} {DOT} {src}")
        filed = r.get("signal_file")
        if isinstance(filed, dict) and filed.get("version"):
            line += dim(f"  {DOT} filed v{filed['version']}")
        say(line)
        reason = str(r.get("reason") or "").strip()
        if reason:
            for text in _wrap(reason, 68):
                say("    " + dim(text))
    say("")
    say(dim(f"  {len(rows)} shown {DOT} open one in the portal to act on it"))
    return 0


def preflight(catalogue: list[dict[str, Any]], name: str, *, data: Any, backtest_fn: Any) -> str | None:
    """What is missing before a run starts. `None` when nothing is.

    ── THE FAILURE THIS REPLACES ──────────────────────────────────────────────

    `validate_study` sweeps, so it needs the caller's simulator. Without one,
    `compute.sweep` raised UnsupportedOp, the server correctly re-offered the
    step, the second identical failure abandoned the run — and the user watched
    a workflow start, grind, and die naming an op they had never heard of. The
    real problem was knowable before the first request.

    The server publishes `requires` for exactly this. It is an input contract,
    not workflow knowledge: it says what to bring, never what happens to it.
    """
    row = next((r for r in catalogue if r.get("name") == name), None)
    if row is None:
        known = ", ".join(str(r.get("name")) for r in catalogue) or "none"
        return f"No workflow called {name!r}. This workspace offers: {known}"

    have = {
        "data": data is not None,
        "backtest_fn": backtest_fn is not None,
        # SHAPE, not only presence. `universe` and `returns` are both "data", and
        # a caller who brings the wrong one used to get an UnsupportedOp from
        # inside a running workflow — the same failure this function exists to
        # prevent, arriving one level finer.
        "universe": _looks_like_universe(data),
        "returns": _looks_like_returns(data),
        "signal": (
            isinstance(data, dict)
            and isinstance(data.get("signal"), dict)
            and isinstance(data.get("prices"), dict)
        ),
        "book_returns": isinstance(data, dict) and data.get("book_returns") is not None,
    }
    missing = [r for r in (row.get("requires") or []) if not have.get(str(r), True)]
    if not missing:
        return None

    # check_overlap wants one candidate and a separate book. A loaded universe
    # already holds both: name one symbol and the other names are the book.
    # Telling somebody to pass --symbol and then discarding the rest of the
    # names was the loop. The session builds that book itself; this message is
    # the shell form of the same move.
    if name == "check_overlap" and _looks_like_universe(data) and isinstance(data, dict):
        example = next(iter(sorted(str(k) for k in data)), "MU")
        return (
            f"check_overlap compares ONE name with the equal-weight of the others, "
            f"and a universe of {len(data)} names is loaded.\n"
            f"      run check_overlap --symbol {example}\n"
            "  That name is the candidate. The other names are the book.\n"
            "  For every name against every other name:\n"
            "      run correlation"
        )

    # THE ANSWER THE LOADED DATA MAKES POSSIBLE, before the generic doors. A
    # user with a universe loaded who is told "load a universe" is in a loop
    # with no exit — the real move is naming which of their names to measure.
    if "returns" in missing and _looks_like_universe(data) and isinstance(data, dict):
        example = next(iter(sorted(str(k) for k in data)), "MU")
        return (
            f"{name} runs on ONE return series, and a universe of {len(data)} names "
            "is loaded.\n"
            "  Name the one to measure:\n"
            f"      run {name} --symbol {example}\n"
            "  Its closes become the return series the measurement runs on."
        )

    # THE SAME LOOP, ONE REQUIREMENT OVER. A signal gap answered with "load a
    # universe" told somebody whose universe was ALREADY loaded to load it
    # again — the status line said `universe:s&p` while the refusal prescribed
    # exactly that. The universe covers the PRICE half; the signal half is the
    # caller's own scores, and scores live in their code, not the portal.
    if "signal" in missing and _looks_like_universe(data):
        return (
            f"{name} measures a SIGNAL against prices, and what is loaded is prices alone.\n"
            f"  {_SHAPES['signal']}\n"
            "  Your universe covers the prices; bring the scores from a module:\n"
            f"      alphaengine run {name} --project your_pkg.your_signal\n"
            "  See the shape working first:\n"
            f"      alphaengine run {name} --project alphaengine.demo_signal"
        )

    what = " and ".join(f"`{m}`" for m in missing)
    shapes = "\n".join(f"  {_SHAPES[m]}" for m in missing if m in _SHAPES)
    # THE DOORS, EASIEST FIRST. This named `--project` and nothing else, so
    # somebody holding a CSV was told to go and write a Python module — and a
    # message that offers only the hardest route reads as "you cannot do this".
    # `--data` is one flag and it is the answer most of the time.
    if "backtest_fn" in missing:
        # Only a module can carry a simulator, so here the module is not the
        # hard route, it is the only one.
        doors = [f"      alphaengine run {name} --project research.momentum"]
    else:
        doors = [f"      alphaengine run {name} --data prices.csv"]
        if "universe" in missing:
            doors.append(
                f"      alphaengine run {name} --universe <name>        # one you registered in the portal"
            )
        doors.append(f"      alphaengine run {name} --project research.momentum")

    return (
        f"{name} needs {what}, and none was loaded.\n"
        + (shapes + "\n" if shapes else "")
        + "\n  Load some:\n"
        + "\n".join(doors)
        + "\n\n  Or see it work on the built-in example first:\n"
        "      alphaengine demo"
    )


#: What each named input has to look like, said once, in the message that fires.
#: Guidance the CALLER needs about their own module — not a description of what
#: the workflow does with it, which stays server side.
_SHAPES = {
    "universe": "`data` is a universe: {symbol: prices}, prices being closes, {date: close}, or OHLC rows.",
    "returns": (
        "`data` is a return series: per-period returns, a mapping with them under "
        "`returns`, or one name out of a loaded universe via --symbol."
    ),
    "backtest_fn": "`backtest_fn` is your own simulator; we orchestrate and measure, you simulate.",
    "signal": (
        "`data` is {'signal': {symbol: values}, 'prices': {symbol: closes}} -- the "
        "panel and what it is scored against. A --project module is the door."
    ),
    "book_returns": (
        "`data` carries both series: {'returns': [...], 'book_returns': [...]} -- "
        "the candidate and what the book already holds."
    ),
}


def _looks_like_universe(data: Any) -> bool:
    """A non-empty mapping of symbol -> something sequence-like.

    Deliberately shallow. This is a pre-run courtesy, not a validator: the
    executor still refuses properly, and a check that tried to be exhaustive here
    would reject a legitimate frame-like object it had never heard of.
    """
    if not isinstance(data, dict) or not data:
        return False
    if any(key in data for key in ("returns", "pnl")):
        return False
    first = next(iter(data.values()), None)
    return isinstance(first, (list, tuple, dict)) or hasattr(first, "__len__")


def _looks_like_returns(data: Any) -> bool:
    if data is None:
        return False
    if isinstance(data, dict):
        return any(data.get(key) is not None for key in ("returns", "pnl"))
    try:
        return len(data) > 0 and not isinstance(data, (str, bytes))
    except TypeError:
        return False


def _unquote(spec: str) -> str:
    spec = spec.strip()
    if len(spec) >= 2 and spec[0] == spec[-1] and spec[0] in "\"'":
        return spec[1:-1]
    return spec


_DATA_SUFFIXES = {".csv", ".parquet", ".pq", ".tsv", ".json", ".feather", ".pkl", ".pickle"}


def classify_load(spec: str) -> str:
    """Which door this token opens: `data`, `project`, or `universe`.

    A path that exists wins because a file on disk is the most specific thing
    the caller could have said. A data suffix is next, so `load prices.csv`
    still routes to the file loader when the file is missing (and then fails
    usefully there). A dotted name that is not a path is a module. Everything
    else is a universe.
    """
    spec = _unquote(spec)
    if not spec:
        raise ValueError("load what? a file, a project module, or a universe name.")
    path = pathlib.Path(spec).expanduser()
    try:
        if path.is_file():
            return "data"
    except OSError:
        pass
    if path.suffix.lower() in _DATA_SUFFIXES:
        return "data"
    if os.sep in spec or (os.altsep and os.altsep in spec):
        return "data"
    if "." in spec and not spec.startswith("."):
        return "project"
    return "universe"


def resolve_data(args: argparse.Namespace, session: Any = None) -> tuple[Any, Any]:
    """Everything a run can be fed, resolved in one place.

    THREE DOORS, AND THE ORDER IS THE ARGUMENT:

        --project   a module exposing `data` and `backtest_fn`. The only door
                    that can supply a SIMULATOR, so it is the only one
                    `validate_study` can use.
        --data      a local CSV. Numbers live in files, and "write a Python
                    module first" was a real barrier in front of the three
                    workflows that need numbers rather than code.
        --universe  a universe REGISTERED IN THE PORTAL, which supplies the
                    symbol list and never the prices -- those are parameters, by
                    design (§9: a universe definition is names plus how to source
                    them, never a series). Combines with `--data`: the portal
                    says which names, the file supplies their closes.

    `--project` wins where they overlap, because a module that defines both is
    the most specific thing the caller could have said.
    """
    data, backtest_fn = None, None

    if getattr(args, "project", None):
        data, backtest_fn = load_project(args.project)

    if getattr(args, "data", None):
        from .connectors import load_panel

        raw_data_arg = str(args.data)
        target_path = pathlib.Path(raw_data_arg).expanduser()
        if not target_path.exists():
            from .project import find_project

            proj = find_project()
            if proj is not None:
                with contextlib_suppress():
                    from .datasets import get_dataset

                    ds = get_dataset(proj, raw_data_arg)
                    cand = pathlib.Path(str(ds.get("abs_path") or (proj.root / str(ds.get("path") or ""))))
                    if cand.exists():
                        raw_data_arg = str(cand)
        loaded = load_panel(raw_data_arg)
        # A project module that also defined `data` keeps it: the explicit module
        # is the more specific statement, and silently replacing it would make
        # two flags fight where the user can see neither winning.
        data = data if data is not None else loaded

    universe = getattr(args, "universe", None)

    if universe:
        data = _narrow_to_universe(session, universe, data)

    # ── ONE NAME OUT OF MANY ───────────────────────────────────────────────
    #
    # `--symbol MU` turns a loaded universe into the single return series the
    # sizing and monitoring workflows measure. Without this rung the ladder had
    # a hole a user actually fell into: told "size_position needs returns",
    # they loaded their universe, passed it again, and got the same refusal —
    # a loop with no exit, because a universe of closes could not become one
    # series of returns.
    symbol = getattr(args, "symbol", None)
    if symbol:
        from .loaders import DataShapeError

        derived = _returns_from_universe(data, str(symbol))
        if derived is None:
            names = ", ".join(sorted(k for k in data)[:8]) if isinstance(data, dict) else "none loaded"
            raise DataShapeError(
                f"--symbol {symbol} needs that name in a loaded universe with at least "
                f"two closes. Loaded: {names}"
            )
        say(dim(f"  {str(symbol).upper()} {DOT} {len(derived)} returns {DOT} derived from its closes"))
        data = derived

    return data, backtest_fn


def _narrow_to_universe(session: Any, wanted: str, data: Any, rows: Any = None) -> Any:
    """Filter loaded prices to a universe the portal holds.

    THE SPLIT THAT MAKES THIS SAFE: the portal returns SYMBOLS, this machine
    holds the prices, and the two meet here rather than on the server. A
    universe definition is parameters -- names plus how to source them -- and
    that is why fetching one moves no market data.
    """
    from .loaders import DataShapeError, coverage

    if session is None:
        raise DataShapeError("--universe needs a workflow server; none is configured.")

    if rows is None:
        rows = session.universes()
    match = next(
        (u for u in rows if wanted in (str(u.get("id")), str(u.get("name")))),
        None,
    )
    if match is None:
        known = ", ".join(str(u.get("name")) for u in rows) or "none"
        raise DataShapeError(f"no universe called {wanted!r}. Yours: {known}")

    # SYMBOLS LIVE UNDER `definition`, NOT AT THE TOP LEVEL. The portal returns
    # {id, name, version, definition, created_at, updated_at} and the symbol
    # list is inside `definition` alongside `cache_id`. Reading
    # `match["symbols"]` therefore found nothing on EVERY universe, so a real
    # one reported "names no symbols" and could never load.
    #
    # The test that was supposed to catch this used a fixture invented from the
    # shape I assumed rather than the one the server sends, which is worth more
    # than the bug: a stub that agrees with the code instead of with the server
    # tests nothing and reads as coverage. See `test_the_client_reads_the_shape_
    # the_server_actually_sends` in the platform repo, which imports both.
    definition = match.get("definition") or {}
    symbols = [str(x) for x in (definition.get("symbols") or match.get("symbols") or [])]

    # ── NO LOCAL FILE? PULL THE CLOSES YOU STORED WITH IT ──────────────────
    #
    # ATTEMPTED BEFORE THE SYMBOL CHECK, and the order is the fix. A missing
    # symbol list has nothing to do with whether the closes are stored, so
    # gating the fetch on it refused a universe whose prices were sitting right
    # there. Symbols only matter for the narrowing path below, where there is a
    # local file to narrow.
    #
    # A universe registered WITH prices has them in the portal, encrypted, and
    # the portal will decrypt them back to the same account. Requiring the user
    # to find the same CSV on disk first was not a data boundary, it was a
    # missing function wearing one: you upload an S&P universe, then the OS
    # tells you it cannot see it.
    #
    # The local file still WINS when there is one. `--data` is the more explicit
    # statement, and silently preferring a stored copy over the file somebody
    # just pointed at is the kind of helpfulness nobody can debug.
    if data is None:
        from .client import ServerError

        universe_id = str(match.get("id"))
        try:
            fetched = session.universe_series(universe_id)
        except ServerError as exc:
            if exc.status == 404:
                raise DataShapeError(
                    f"{wanted!r} holds its names on your account and not its prices, "
                    "so there is nothing stored to run on. On the portal's Data page "
                    'it will say "this browser only" — press "Store on my account" '
                    "and it will load here. Or pass --data with the closes."
                ) from exc
            raise
        prices = fetched.get("prices") or {}
        if not prices:
            raise DataShapeError(f"{wanted!r} resolved to no stored prices.")
        # A list is observations and so is a {date: close} mapping — its length
        # IS its date count. Anything else contributes nothing to the line.
        n_obs = max((len(v) for v in prices.values() if isinstance(v, (list, dict))), default=0)
        say(
            dim(
                f"  universe {wanted} {DOT} {len(prices)} names {DOT} "
                f"{n_obs} observations {DOT} from the portal"
            )
        )
        return prices

    # From here there IS a local file, so the symbol list is what narrows it and
    # its absence is genuinely fatal to this path.
    if not symbols:
        raise DataShapeError(
            f"universe {wanted!r} lists no symbols, so there is nothing to narrow "
            "your file to. Drop --universe and run on the file as it stands."
        )

    kept, missing = coverage(data, symbols)
    if not kept:
        raise DataShapeError(f"none of the {len(symbols)} names in {wanted!r} are in your file.")
    if missing:
        # SAID, NOT SWALLOWED. A universe of 500 screened against a file holding
        # 40 of them is a different result from a universe of 40, and this is the
        # last place that difference is visible.
        shown = ", ".join(missing[:8]) + (ELLIPSIS if len(missing) > 8 else "")
        say(yellow(f"  {len(missing)} of {len(symbols)} names are not in your file: {shown}"))
    say(dim(f"  universe {wanted} {DOT} {len(kept)} of {len(symbols)} names loaded"))
    return kept


# ── `--sleeve` AND `--thesis` ARE GONE (0.5.0) ─────────────────────────────
#
# Four helpers lived here: `_sleeve_row`, `_resolve_sleeve`, `_offer_sleeve` and
# `_propose_sleeve`. They resolved a sleeve by name to open a run against, and
# turned a closed screen into a PROPOSAL — the one thing this package ever
# created.
#
# Their three routes (`/api/me/sleeves`, `/api/me/theses`,
# `/api/me/proposals/from-os`) belonged to the thesis object model, which came
# out with the retired agent desk it was built around. A CLI flag whose endpoint
# does not exist is worse than an absent one: it fails at the END of a run, after
# the work, naming a server error rather than a missing feature.
#
# THE SPLIT THEY EXISTED FOR IS UNCHANGED and now runs through a smaller object.
# The quant still runs the research where the data is and the PM still decides
# where the authority is; what travels between them is the run itself, handed
# over with an addressee (`delivered_to`) and decided on the run page where its
# figures, its caveats and its review pass already are.


def _publish_signals(session: Any, run: Any, *, workspace_id: str | None, label: str | None) -> None:
    """Turn a closed screen into a dated, versioned, diffable ranked file.

    ── WHY A RUN HAS TO LEAVE SOMETHING BEHIND ────────────────────────────────

    The platform had no machine-produced artifact. Every durable object began as
    a human typing into a form, so the permission layer, the freeze layer and
    the merge layer all governed a flow whose source was a text box — which is
    why the product read as disconnected pieces even though the connective
    tissue had been built. Nothing flowed through it.

    A screen produced a ranking and printed it. This writes it: ticker, rank,
    score, dated, versioned, and diffable against yesterday's. The diff is the
    part that matters to a desk — nobody reads a hundred ranked names every
    morning, they read what moved — and it was already implemented on the
    platform with nothing to read.

    THE CAVEATS TRAVEL WITH IT. `n_insufficient` and `truncated` are exactly the
    facts that make a shortlist honest, and a list handed on without them is the
    failure every control in this product exists to prevent. They are attached
    to the object, not printed beside it, because the terminal is not where this
    gets read.

    A FAILURE HERE NEVER FAILS THE RUN. The figures are already on screen and
    the run's own record is already sealed server-side; losing the file costs a
    convenience, and taking the run down with it would trade the result for the
    receipt.
    """
    figures = (run.artifact or {}).get("figures") or {}
    package = figures.get("package") or {}
    rows = package.get("rows") or []
    if not rows:
        return

    ranked = [
        {"ticker": str(r.get("symbol") or r.get("ticker") or ""), "rank": i + 1, "score": r.get("score")}
        for i, r in enumerate(rows)
    ]
    notes: list[str] = []
    n_insufficient = package.get("n_insufficient")
    if n_insufficient:
        notes.append(
            f"{n_insufficient} names in the universe had too little history to "
            "be measured on this metric and are absent from this ranking."
        )
    if package.get("truncated"):
        notes.append(f"More names passed than were returned; this is the top {len(ranked)} of them.")
    universe_size, n_evaluated = package.get("universe_size"), package.get("n_evaluated")
    if universe_size and n_evaluated is not None and n_evaluated < universe_size:
        notes.append(f"{n_evaluated} of {universe_size} names were evaluated.")

    import datetime

    try:
        saved = session.save_signals(
            asof=datetime.date.today().isoformat(),
            rows=ranked,
            label=label or "signals",
            workspace_id=workspace_id,
            engine_version=package.get("engine_version"),
            notes=notes or None,
            source_run_id=getattr(run, "run_id", None),
        )
    except Exception as exc:  # noqa: BLE001 - named, never fatal; see the docstring
        say(dim(f"  The ranking could not be filed: {exc}"))
        return

    version = saved.get("version")
    say("")
    say(
        f"  {green('Filed.')} {len(ranked)} names {DOT} "
        + dim(f"{saved.get('label')} {DOT} {saved.get('asof')}")
        + (dim(f" {DOT} v{version}") if version else "")
    )
    _say_diff(run)


def _found(before: set[str], run: Any) -> str:
    """What the step that just ran FOUND, in the reader's words.

    ── THE RULE THAT MAKES THIS SAFE: NARRATION IS RETROSPECTIVE ──────────────

    `protocol.py` is emphatic that the client must never learn the graph — a
    step names an operation and never a purpose, and "a progress bar is a graph
    disclosure in a friendly hat". So this may never say what comes NEXT.

    It does not have to. Every word here describes a step that has ALREADY run,
    and the past is not the graph's future. `executor.execute` returns the
    figures and `Run.step` records them by op before the round trip, so the
    client already knows what it just found out, in numbers, as it finds them.

    Only figures this step added, and only ones carrying a LABEL.
    `_FIGURE_LABELS` is the same vocabulary the closing recap uses — a figure
    called one thing while it streams and another when it lands is two figures
    to a reader.
    """
    fresh: dict[str, Any] = {}
    for op, blob in (getattr(run, "figures", {}) or {}).items():
        if op in before or not isinstance(blob, dict):
            continue
        fresh.update(blob)

    parts: list[str] = []
    # A STATUS OR A VERDICT OUTRANKS A NUMBER. "the book again" is the entire
    # result of an overlap run, and printing a correlation instead is how the
    # one conclusion becomes the one thing left unsaid.
    for key in ("status", "verdict"):
        word = fresh.get(key)
        if isinstance(word, str) and word:
            parts.append(bold(word.replace("_", " ")))
            break
    for key, (label, unit) in _FIGURE_LABELS.items():
        if len(parts) >= 3:  # three is a line; more is a table, and one lands below
            break
        v = fresh.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            parts.append(f"{label} {bold(_num(v))}{(' ' + unit) if unit else ''}")
    return f" {DOT} ".join(parts)


def _drive(run: Any, *, quiet: bool = False) -> None:
    """Execute the run, narrating each step BY WHAT IT FOUND.

    Wraps `Run.step` rather than calling `drive()` so there is something to
    watch. The loop itself — what is permitted, in what order, what stops it —
    is still entirely the server's.

    IT USED TO PRINT THE OP NAME AND A DURATION: `server -> compute.screen 1.2s`.
    That is a progress log. A quant watching it learned which functions fired
    and never what was being found out — and the process IS the product here, so
    a run that will not say what it is discovering is showing the wrong half.
    """
    n = 0
    failures: dict[str, int] = {}
    while run.status == "open" and n < 200:
        if not run.permitted:
            run.resume()
            if run.status != "open" or not run.permitted:
                break

        batch = run.permitted if run.selection == "all" else run.permitted[:1]
        for step in batch:
            op = step.get("op", "?")
            before = run.status
            seen = set(getattr(run, "figures", {}) or {})
            # THE STEP IS WATCHED WHILE IT RUNS, not announced and then silent.
            # `compute.*` executes in-process against the caller's own frames, so
            # this is the one place in the product where a long wait is entirely
            # local and entirely invisible.
            with Working(bold(op), plain=str(op)) as w:
                run.step(step)
            n += 1
            took = dim(f"{w.elapsed:.1f}s")

            still = any(p.get("step_id") == step.get("step_id") for p in run.permitted)
            if still and before == "open" and run.status == "open":
                key = str(step.get("step_id"))
                failures[key] = failures.get(key, 0) + 1
                if not quiet:
                    # THE REASON, NOT JUST THE OP. "could not be executed here"
                    # with no why sent people to the traceback that no longer
                    # prints. The message came from the executor and was written
                    # to be read.
                    say(f"  step {n:<2}  {red(CROSS)}  {bold(op)}")
                    why = getattr(run, "last_error", None)
                    if why:
                        for line in str(why).splitlines():
                            say(red(f"           {line}"))
                    say(f"           {took}")
                if failures[key] >= 2:
                    run.status = "abandoned"
                    run.stopped = {
                        "reason": "step_failed",
                        "op": op,
                        "error": getattr(run, "last_error", None),
                    }
                    return
            elif not quiet:
                # WHAT IT FOUND, or the op when it produced nothing nameable —
                # a step with no labelled figure is a real thing (`record.*`,
                # `emit.*` seal rather than measure) and printing a blank line
                # for it would read as a stall.
                found = _found(seen, run)
                say(f"  step {n:<2}  {bold(op)}")
                if found:
                    say(f"           {found}")
                say(f"           {took}")
                rejected = getattr(run, "trace_rejected", None)
                if rejected:
                    say(yellow(f"           portal did not accept the series trace: {rejected}"))

            if run.status != "open":
                return


def _publish_panel(session: Any, result: dict[str, Any], *, thesis_id: str | None = None) -> None:
    """Hand a matrix, a pair table, or portfolio weights to the portal.

    A missing route is not a failure. The result is already on this machine.
    """
    if session is None or not hasattr(session, "file_panel"):
        return
    from .client.executor import _guard

    if result.get("kind") == "portfolio":
        from .core.portfolio import portfolio_figures

        figures = portfolio_figures(result)
    else:
        from .core.relations import portal_figures

        figures = portal_figures(result)
    try:
        _guard(figures)
    except ValueError as exc:
        say(yellow(str(exc)))
        return
    filed = session.file_panel(figures, thesis_id=thesis_id)
    if not isinstance(filed, dict):
        return
    if filed.get("filed"):
        say(dim("  filed to the portal"))
        return
    if filed.get("reason") == "missing":
        say(dim("  held on this machine; the portal has no panels route yet"))
        return
    if filed.get("reason") and filed.get("reason") != "offline":
        say(dim(f"  held on this machine ({filed.get('reason')})"))


def _print_cross(
    data: Any,
    kind: str,
    *,
    session: Any = None,
    thesis_id: str | None = None,
) -> int:
    from .core.relations import cointegration, correlation, covariance, summary_lines

    try:
        if kind == "correlation":
            result = correlation(data)
        elif kind == "covariance":
            result = covariance(data)
        else:
            result = cointegration(data)
    except ValueError as exc:
        say(red(str(exc)))
        return 2
    for line in summary_lines(result):
        say(line)
    _publish_panel(session, result, thesis_id=thesis_id)
    return 0


def _print_portfolio(
    data: Any,
    *,
    session: Any = None,
    thesis_id: str | None = None,
    method: str = "hrp",
) -> int:
    """One-shot portfolio. The session path can also ask for risk parity."""
    from .core.portfolio import build_portfolio

    try:
        result = build_portfolio(data, method=method)
    except ValueError as exc:
        say(red(str(exc)))
        return 2
    say(f"  {result['method']}  {result['n_assets']} names  Ledoit-Wolf covariance")
    for row in (result.get("rows") or [])[:8]:
        say(f"  {row['name']}  {row['weight']}")
    _publish_panel(session, result, thesis_id=thesis_id)
    return 0


def _overlap_payload(data: Any, symbol: str) -> dict[str, Any] | None:
    """Candidate plus the equal-weight of the other loaded names, or None."""
    from .core.relations import overlap_against_the_rest

    if not _looks_like_universe(data):
        return None
    try:
        prepared = overlap_against_the_rest(data, symbol)
    except ValueError as exc:
        say(red(str(exc)))
        return None
    others = len(prepared["book_names"])
    say(dim(f"  {prepared['symbol']} against the equal-weight of {others} other names."))
    return {"returns": prepared["returns"], "book_returns": prepared["book_returns"]}


def cmd_run(args: argparse.Namespace) -> int:
    from .client import Offline, ServerError
    from .core.relations import KINDS
    from .project import find_project

    proj = find_project()
    wf_str = str(getattr(args, "workflow", "") or "")
    if wf_str.lower().endswith((".yaml", ".yml")):
        return _cmd_run_pipeline(args, wf_str, proj)

    session, url = _session(
        args.url,
        args.key,
        share_series=bool(getattr(args, "share_series", False)),
    )
    kind = KINDS.get(str(args.workflow).lower().replace("-", "_"))
    if kind:
        try:
            data, _fn = resolve_data(args, session)
        except (ProjectError, ValueError) as exc:
            say(red(str(exc)))
            return 2
        return _print_cross(data, kind, session=session)
    if str(args.workflow).lower().replace("-", "_") in ("portfolio", "allocate"):
        try:
            data, _fn = resolve_data(args, session)
        except (ProjectError, ValueError) as exc:
            say(red(str(exc)))
            return 2
        return _print_portfolio(data, session=session)
    symbol = getattr(args, "symbol", None)
    resolve_args = args
    if args.workflow == "check_overlap" and symbol:
        resolve_args = argparse.Namespace(**vars(args))
        resolve_args.symbol = None
    try:
        data, backtest_fn = resolve_data(resolve_args, session)
        workspace_id = None
    except (ProjectError, ValueError) as exc:
        say(red(str(exc)))
        return 2
    if args.workflow == "check_overlap" and _looks_like_universe(data):
        if symbol:
            payload = _overlap_payload(data, str(symbol))
            if payload is None:
                return 2
            data = payload
        else:
            say("  check_overlap measures one name against the equal-weight of the others.")
            say(dim("  No name was given, so this is every name against every other name."))
            return _print_cross(data, "correlation", session=session)

    inputs: dict[str, Any] = {}
    for pair in args.input or []:
        if "=" not in pair:
            say(red(f"--input expects key=value, got {pair!r}"))
            return 2
        k, v = pair.split("=", 1)
        # A value that reads as JSON travels as the STRUCTURE it spells, so
        # `tolerances={"max_drawdown_pct": 20}` reaches the workflow as a
        # mapping rather than a string it silently treats as no tolerances.
        if v[:1] in "{[":
            try:
                inputs[k] = json.loads(v)
            except ValueError:
                inputs[k] = v
        else:
            inputs[k] = v
    if args.label:
        inputs["label"] = args.label

    # Inside a local `.alphaengine/` project, or when `--json` / `--stdin` is passed
    # and no remote `--url` or `--key` was specified, execute locally.
    if (
        (proj is not None or getattr(args, "json_out", False) or getattr(args, "stdin_in", False))
        and getattr(args, "url", None) is None
        and getattr(args, "key", None) is None
        and not os.environ.get(_ENV_KEY)
    ):
        return _run_local_in_project(args, proj, data=data, backtest_fn=backtest_fn, inputs=inputs)

    try:
        say(f"{bold(args.workflow)} {dim(DOT + ' ' + url)}")
        catalogue = session.workflows()
        # THE ACCOUNT IS REACHED FOR BEFORE REFUSING — same courtesy as the
        # agent path. `alphaengine run screen_universe`, signed in, with one
        # stored universe on the account, should run on it and say so.
        ready: list[str] = []
        row = next((r for r in catalogue if r.get("name") == args.workflow), None)
        if data is None and row is not None and "universe" in (row.get("requires") or []):
            wanted, ready = _stored_universes(session)
            if wanted:
                try:
                    data = _narrow_to_universe(session, wanted, None)
                except Exception as exc:  # noqa: BLE001 - reachable and unloadable: refuse below
                    say(yellow(f"  {wanted} is on your account and could not be loaded: {exc}"))
        # ASK WHAT IT NEEDS BEFORE STARTING IT. One extra GET against a run that
        # would otherwise fail twice and abandon.
        gap = preflight(catalogue, args.workflow, data=data, backtest_fn=backtest_fn)
        if gap:
            say("")
            say(yellow(gap))
            if len(ready) > 1:
                say(yellow(f"  Your account holds {len(ready)} stored universes: {', '.join(ready)}."))
                say(dim("  Pass --universe <name> to say which."))
            return 2
        # The project's declared grid travels as a run input unless the caller
        # named one explicitly. Without this, a sweep received `grid={}`.
        if "grid" not in inputs and project_grid():
            inputs["grid"] = project_grid()
        thesis_id = None
        named_thesis = getattr(args, "thesis", None)
        if named_thesis:
            chosen, candidates = match_thesis(session.theses(), str(named_thesis))
            if chosen is None:
                if candidates:
                    names = ", ".join(thesis_label(r) for r in candidates)
                    say(yellow(f"Which thesis? {names}"))
                else:
                    say(yellow(f"No thesis named {named_thesis!r} on this account."))
                return 2
            thesis_id = thesis_id_of(chosen)
            if not thesis_id:
                say(yellow(f"{thesis_label(chosen)} has no id, so a run cannot be attached to it."))
                return 2
            say(dim(f"  thesis  {thesis_label(chosen)}"))
        run = session.open(
            args.workflow,
            data=data,
            backtest_fn=backtest_fn,
            workspace_id=workspace_id,
            thesis_id=thesis_id,
            **inputs,
        )
        _drive(run, quiet=args.quiet)
    except Offline:
        _explain_offline(url)
        return 2
    except ServerError as exc:
        refused(exc)
        if is_auth_error(exc) and offer_key():
            say(dim("  Key set. Run that again."))
        return 2

    code = _report(run)
    if run.status == "closed":
        _publish_signals(session, run, workspace_id=workspace_id, label=args.label)
    if proj is not None:
        _record_remote_run_in_project(args, proj, run, data=data, inputs=inputs)
    return code


_LOCAL_CATALOGUE: list[dict[str, Any]] = [
    {"name": "diagnose_data", "version": "1.0.0", "requires": ["universe"], "reproducible": True},
    {"name": "screen_universe", "version": "1.0.0", "requires": ["universe"], "reproducible": True},
    {"name": "evaluate_signal", "version": "1.0.0", "requires": ["signal"], "reproducible": True},
    {"name": "validate_study", "version": "1.0.0", "requires": ["data", "backtest_fn"], "reproducible": True},
    {"name": "stress_study", "version": "1.0.0", "requires": ["returns"], "reproducible": True},
    {
        "name": "check_overlap",
        "version": "1.0.0",
        "requires": ["returns", "book_returns"],
        "reproducible": True,
    },
    {"name": "size_position", "version": "1.0.0", "requires": ["returns"], "reproducible": True},
    {"name": "monitor_sleeve", "version": "1.0.0", "requires": ["returns"], "reproducible": True},
]


def _cmd_run_pipeline(args: argparse.Namespace, pipeline_file: str, proj: Any) -> int:
    from .envelope import ENVELOPE_SCHEMA_VERSION, emit_json
    from .pipeline import run_pipeline

    try:
        envelopes = run_pipeline(pipeline_file, proj=proj)
    except Exception as exc:  # noqa: BLE001
        say(red(str(exc)))
        return 2
    if getattr(args, "json_out", False):
        emit_json(
            {
                "schema_version": ENVELOPE_SCHEMA_VERSION,
                "command": "pipeline",
                "pipeline": str(pipeline_file),
                "steps": envelopes,
            }
        )
        return 0
    say("")
    say(f"  {bold('pipeline')}  {pipeline_file}  {dim(DOT)}  {len(envelopes)} steps")
    for idx, env in enumerate(envelopes, start=1):
        nt_val = env.get("n_trials")
        ct_val = env.get("cumulative_trials")
        say(
            f"    {idx:02d}  {bold(str(env.get('command')).ljust(18))}  "
            f"{str(env.get('verdict') or '').ljust(12)}  "
            f"{dim(str(env.get('run_id')))}  "
            f"{dim(f'trials={nt_val} cum={ct_val}')}"
        )
    say("")
    return 0


def _run_local_in_project(
    args: argparse.Namespace,
    proj: Any,
    *,
    data: Any,
    backtest_fn: Any,
    inputs: dict[str, Any],
) -> int:
    from .envelope import build_envelope, emit_json, read_stdin_envelope, returns_from_panel_or_series
    from .ledger import execute_local_workflow, hash_dataset_any, record_run, record_trials

    wf = str(args.workflow)
    json_out = bool(getattr(args, "json_out", False))
    stdin_in = bool(getattr(args, "stdin_in", False))
    upstream_env: dict[str, Any] | None = None

    if stdin_in:
        try:
            upstream_env = read_stdin_envelope()
        except Exception as exc:  # noqa: BLE001
            say(red(str(exc)))
            return 2
        up_ctx = upstream_env.get("context") or {}
        if data is None and up_ctx.get("returns"):
            data = [float(x) for x in up_ctx["returns"]]
        elif data is None and up_ctx.get("data_path"):
            from .connectors import load_panel

            dp = pathlib.Path(str(up_ctx["data_path"]))
            if dp.is_file():
                data = load_panel(dp)
        if (
            wf in ("validate_study", "size_position", "stress_study", "monitor_sleeve")
            and backtest_fn is None
            and "n_trials" not in inputs
            and upstream_env.get("n_trials")
        ):
            inputs["n_trials"] = int(upstream_env["n_trials"])

    # In `--json` or `--stdin` mode with no explicit data flag, provide the built-in demo data
    # so `ae screen --json | ae validate --stdin | ae size --stdin` works out of the box.
    if data is None and (json_out or stdin_in):
        if wf in ("screen_universe", "diagnose_data"):
            from .demo_universe import data as demo_u

            data = demo_u
        elif wf in ("size_position", "stress_study", "monitor_sleeve") or (
            wf == "validate_study" and backtest_fn is None
        ):
            from .demo_returns import data as demo_r

            data = demo_r

    if "grid" not in inputs and project_grid():
        inputs["grid"] = project_grid()

    local_cat = list(_LOCAL_CATALOGUE)
    if wf == "validate_study" and stdin_in and backtest_fn is None:
        local_cat = [
            {"name": "validate_study", "version": "1.0.0", "requires": ["returns"], "reproducible": True}
            if r.get("name") == "validate_study"
            else r
            for r in local_cat
        ]
    gap = preflight(local_cat, wf, data=data, backtest_fn=backtest_fn)
    if gap:
        say("")
        say(yellow(gap))
        return 2

    try:
        figures, verdict, n_trials, source, returns_col, trial_rows = execute_local_workflow(
            wf,
            data=data,
            backtest_fn=backtest_fn,
            grid=inputs.get("grid"),
            inputs=inputs,
        )
    except Exception as exc:  # noqa: BLE001
        say(red(str(exc)))
        return 2

    # Build pipe handoff context for downstream `--stdin` consumers
    pipe_ctx: dict[str, Any] = {
        "data_path": getattr(args, "data", None),
        "project_module": getattr(args, "project", None),
    }
    rows = figures.get("rows")
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        top_sym = str(rows[0].get("symbol") or rows[0].get("ticker") or "")
        pipe_ctx["top_symbol"] = top_sym
        pipe_ctx["shortlist"] = [str(r.get("symbol") or r.get("ticker") or "") for r in rows[:10]]
        rets_top, _ = returns_from_panel_or_series(data, top_sym)
        if rets_top:
            pipe_ctx["returns"] = rets_top
    elif returns_col:
        pipe_ctx["returns"] = returns_col
    elif isinstance(data, list) and data and isinstance(data[0], (int, float)):
        pipe_ctx["returns"] = [float(x) for x in data[:512]]

    idea_id = getattr(args, "idea", None) or (upstream_env.get("idea_id") if upstream_env else None)
    family = (
        getattr(args, "family", None)
        or (upstream_env.get("family") if upstream_env else None)
        or getattr(args, "project", None)
        or wf
    )

    ds_hashes = hash_dataset_any(data)
    raw_data_flag = getattr(args, "data", None)
    if raw_data_flag and proj is not None:
        with contextlib_suppress():
            from .datasets import get_dataset

            ds_rec = get_dataset(proj, str(raw_data_flag))
            if ds_rec.get("sha256") and ds_rec["sha256"] not in ds_hashes:
                ds_hashes.insert(0, str(ds_rec["sha256"]))
        p_flag = pathlib.Path(str(raw_data_flag)).expanduser()
        if p_flag.is_file():
            import hashlib as _hl

            f_sha = _hl.sha256(p_flag.read_bytes()).hexdigest()
            if f_sha not in ds_hashes:
                ds_hashes.insert(0, f_sha)

    if proj is not None:
        data_obj_sha: str | None = None
        if data is not None:
            try:
                data_obj_sha = proj.put_object(data)
            except Exception:  # noqa: BLE001
                data_obj_sha = None
        repro_spec: dict[str, Any] = {
            "kind": "workflow",
            "workflow": wf,
            "project_module": getattr(args, "project", None),
            "data_path": getattr(args, "data", None),
            "data_object_sha256": data_obj_sha,
            "symbol": getattr(args, "symbol", None),
            "grid": inputs.get("grid"),
            "inputs": inputs,
        }
        rec = record_run(
            proj,
            command=wf,
            params={"workflow": wf, **inputs},
            dataset_hashes=ds_hashes,
            n_trials=n_trials,
            n_trials_source=source,
            figures=figures,
            verdict=verdict,
            idea_id=idea_id,
            family=str(family),
            repro_spec=repro_spec,
            returns_for_deflation=returns_col,
        )
        if trial_rows:
            record_trials(
                proj,
                run_id=str(rec["run_id"]),
                trials=trial_rows,
                idea_id=idea_id,
                family=str(family),
                timestamp=str(rec["timestamp"]),
            )
        env = build_envelope(
            command=wf,
            figures=rec["figures"],
            verdict=rec["verdict"],
            run_id=str(rec["run_id"]),
            timestamp=str(rec["timestamp"]),
            idea_id=idea_id,
            family=str(rec["family"]),
            params_hash=str(rec["params_hash"]),
            dataset_hashes=rec["dataset_hashes"],
            n_trials=rec["n_trials"],
            n_trials_source=str(rec["n_trials_source"]),
            cumulative_trials=int(rec["cumulative_trials"]),
            project_trials=int(rec["project_trials"]),
            context=pipe_ctx,
        )
    else:
        env = build_envelope(
            command=wf,
            figures=figures,
            verdict=verdict,
            idea_id=idea_id,
            family=str(family),
            dataset_hashes=ds_hashes,
            n_trials=n_trials,
            n_trials_source=source,
            context=pipe_ctx,
        )

    if json_out:
        emit_json(env)
        return 0

    if not getattr(args, "quiet", False):
        say(f"  {green(ON)}  {bold(wf)}  {dim(DOT)}  {env['run_id']}")
        _render_artifact({"figures": {"local": env["figures"]}})
        say("")
        src_note = dim(f"({env['n_trials_source']})")
        say(f"  {dim('trials (this run)')}     {env['n_trials']}  {src_note}")
        say(
            f"  {dim('trials (cumulative)')}   {env['cumulative_trials']}  "
            f"{dim('(family ' + str(env['family']) + ', project total ' + str(env['project_trials']) + ')')}"
        )
        if env["figures"].get("cumulative_deflated_sharpe") is not None:
            say(
                f"  {dim('cumulative DSR')}        {env['figures']['cumulative_deflated_sharpe']}  "
                f"{dim('(' + str(env['figures'].get('cumulative_verdict') or '') + ')')}"
            )
    return 0


def _record_remote_run_in_project(
    args: argparse.Namespace,
    proj: Any,
    run: Any,
    *,
    data: Any,
    inputs: dict[str, Any],
) -> None:
    from .ledger import hash_dataset_any, record_run

    flat: dict[str, Any] = {}
    for blob in (getattr(run, "figures", {}) or {}).values():
        if isinstance(blob, dict):
            flat.update(blob)
    n_trials = flat.get("n_trials")
    source = str(flat.get("n_trials_source") or ("derived_from_grid" if n_trials else "not_recorded"))
    verdict = str(flat.get("verdict") or getattr(run, "status", "") or "") or None
    record_run(
        proj,
        command=str(args.workflow),
        params={"workflow": str(args.workflow), **inputs},
        dataset_hashes=hash_dataset_any(data),
        n_trials=int(n_trials) if isinstance(n_trials, int) else None,
        n_trials_source=source,
        figures=flat,
        verdict=verdict,
        idea_id=getattr(args, "idea", None),
        family=getattr(args, "family", None) or str(args.workflow),
        run_id=str(getattr(run, "run_id", None) or ""),
    )


def _report(run: Any) -> int:
    """What happened, and an exit code that means it.

    A STOP IS NOT A FAILURE and exits 0. "This did not clear the bar" is the
    system working exactly as intended, and a non-zero exit would make every CI
    pipeline treat an honest refusal as a broken build — which is precisely the
    pressure that gets honesty controls disabled.
    """
    say("")
    if run.status == "closed":
        say(f"  {_accent(ON)}  {green('Closed')}  " + dim(str((run.artifact or {}).get("workflow", ""))))
        _render_artifact(run.artifact or {})
        _say_gaps(run)
        _where_it_lives(run)
        _say_next(run)
        return 0
    if run.status == "stopped":
        stop = run.stopped or {}
        say(f"  {yellow(ON)}  {yellow('Stopped')}  " + str(stop.get("reason", "")))
        say(dim(f"     {NEST} A stop is a result. The run did what it was built to do."))
        # A STOP GETS ITS GAPS TOO, and needs them more than a close does: the
        # run refused to answer, so what it could not cover is most of what
        # there is to know about it.
        _say_gaps(run)
        _where_it_lives(run)
        return 0
    if run.status == "abandoned":
        stop = run.stopped or {}
        error = stop.get("error")
        say(f"  {red(CROSS)}  {red('Abandoned')}  {stop.get('op')} failed twice, identically.")
        if error:
            for line in str(error).splitlines():
                say(f"  {line}")
        else:
            say(dim("  Supply a handler for it, or upgrade alphaengine."))
        return 1
    say(dim(f"Run left {run.status}. `alphaengine` can resume it: {run.run_id}"))
    return 1


def _say_diff(run: Any) -> None:
    """WHAT MOVED since the last file of the same label.

    Nobody reads a hundred ranked names every morning; they read what changed. A
    weekly screen's real question is the delta, and until now it was answerable
    only in a browser — from the one place the person asking it is least likely
    to want to leave.

    NO PREDECESSOR IS ITS OWN SENTENCE. "First file of this label — nothing to
    compare against" and "nothing moved" are opposite answers on the one morning
    somebody is looking hardest, and collapsing them is the failure the
    `first_file` flag exists to prevent.
    """
    try:
        diff = run.filed_diff()
    except Exception:  # noqa: BLE001 — a delta is worth losing; a result is not
        return
    if not diff:
        return

    if diff.get("first_file"):
        say(dim("  First file under this label — nothing to compare against yet."))
        return

    n_in = int(diff.get("n_entered") or 0)
    n_out = int(diff.get("n_exited") or 0)
    n_moved = int(diff.get("n_moved") or 0)
    n_held = len(diff.get("held") or [])
    if not (n_in or n_out or n_moved):
        say(dim(f"  Nothing moved against {diff.get('compared_with')} {DOT} {n_held} names, same ranking."))
        return

    parts = []
    if n_in:
        parts.append(green(f"{n_in} in"))
    if n_out:
        parts.append(red(f"{n_out} out"))
    if n_moved:
        parts.append(f"{n_moved} re-ranked")
    parts.append(dim(f"{n_held} unchanged"))
    say(f"  {dim('vs ' + str(diff.get('compared_with')))} {DOT} " + f" {DOT} ".join(parts))

    # THE NAMES, not just the counts — a count tells you something happened and
    # a desk still has to open a browser to find out what.
    for key, colour, word in (("entered", green, "in"), ("exited", red, "out")):
        names = [str(r.get("ticker")) for r in (diff.get(key) or []) if r.get("ticker")]
        if names:
            shown = ", ".join(names[:12])
            more = f" +{len(names) - 12} more" if len(names) > 12 else ""
            say(f"    {colour(word)}  {dim(shown + more)}")


def _say_gaps(run: Any) -> None:
    """WHAT THIS RUN DOES NOT TELL YOU, printed where it can still be acted on.

    The quant sees the gap at the moment of the run — before delivering, while
    closing it is still cheap — and the PM sees the SAME derived object on the
    run's page at the moment of the decision. One derivation, two readings; that
    is the whole reason it lives on the server rather than in this file.

    THE VOICE IS THE ONE THIS TERMINAL ALREADY USES FOR MEASUREMENT FACTS:
    quiet, factual, no colour. A gap is not a warning — it is a statement about
    what was and was not asked, and amber on "nobody has measured these scores
    yet" would make a normal Tuesday read as an alarm.

    Never fatal. `Run.gaps()` swallows its own transport failures and returns
    `[]`, so a server that cannot answer costs the block and not the result.
    """
    try:
        gaps = run.gaps()
    except Exception:  # noqa: BLE001 — a recap must never take the run down
        return
    if not gaps:
        return

    say("")
    say(dim("  What this does not tell you"))
    for g in gaps:
        says = str(g.get("says") or "").strip()
        if not says:
            continue
        for i, line in enumerate(_wrap(says, 72)):
            say(("    " if i else "  " + dim("·") + " ") + dim(line))
        # THE REMEDY IS A COMMAND, not a suggestion to think about it. A gap
        # that cannot name one prints nothing here rather than a vague verb.
        closes = str(g.get("closes_with") or "")
        if closes:
            say("    " + dim("closes with ") + f"alphaengine run {closes}")


def _say_next(run: Any) -> None:
    """The next verb, derived from what the run actually produced.

    A result that names its successor is what makes a catalogue a WEEK instead
    of eight drawers. This is guidance to the HUMAN, printed after the record —
    the loop itself still contains no workflow knowledge, and an agent never
    reads these lines.
    """
    figures = (run.artifact or {}).get("figures") or {}
    flat: dict[str, Any] = {}
    for value in figures.values():
        if isinstance(value, dict):
            flat.update(value)

    line: str | None = None
    rows = flat.get("rows")
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        top = rows[0].get("symbol") or rows[0].get("ticker")
        if top:
            line = f"size the top name: run size_position --symbol {top}"
    elif flat.get("verdict") == "the_book_again":
        line = "nothing to size: the book already holds this bet."
    elif flat.get("verdict") in ("diversifier", "related"):
        line = "size it: run size_position with the candidate's returns"
    elif "dies_at_bps" in flat:
        line = (
            "size it: run size_position"
            if flat.get("dies_at_bps") is not None
            else "rerun with --input turnover=<one-way per period> to get the cost curve"
        )
    elif flat.get("deflated_sharpe") is not None:
        line = "find where it breaks: run stress_study --input turnover=<one-way per period>"
    elif flat.get("target_weight") is not None:
        line = "record the fill on the Portfolio, or take the whole list at once."
    elif flat.get("mean_ic") is not None:
        line = "it carries information; give it a simulator: run validate_study --project <module>"
    elif "usable" in flat:
        # EVERY problem counter, or the hint lies: this said "the data is
        # clean" over a run whose own tiles showed a name leaving the panel —
        # ragged and ends-early were missing from the very list whose job is
        # to notice them.
        dirty = any(
            int(flat.get(k) or 0)
            for k in (
                "n_spike_names",
                "n_stale_names",
                "n_unreadable",
                "n_ragged",
                "n_nonpositive",
                "n_ends_early",
            )
        )
        line = (
            "the named names need looking at before anything runs on them."
            if dirty
            else "the data is clean: run screen_universe"
        )
    elif flat.get("status") == "breached":
        line = "it crossed a line: the breach is waiting for the desk in the portal."

    if line:
        say(dim(f"  next {DOT} ") + line)


def _where_it_lives(run: Any) -> None:
    """The bridge to the portal, printed on every RECORDED outcome.

    A run that ends only in the terminal is a notebook cell with better
    logging. The record is on the account — every step, including the failed
    ones — and the portal's Work screen renders it with the honesty controls
    attached. Stops are recorded too, which is the point: the dead ends are
    the majority of a week's output and the part that evaporates everywhere
    else.
    """
    run_id = getattr(run, "run_id", None)
    if not run_id:
        return
    say("")
    # THE URL, NOT THE ID. A reader who has to know that `/portal/runs/<id>`
    # exists is a reader who does not open it. Most terminals make this
    # clickable; the ones that do not still let it be copied whole.
    say(dim("  Recorded ") + f"{portal_url()}/portal/runs/{run_id}")


#: How many shortlist rows the terminal prints before saying how many it kept
#: back. The whole list is on the account and in the signal file; a screen that
#: scrolls a hundred names past somebody is not showing them more, it is showing
#: them less of what matters.
_SHOWN_ROWS = 15

#: What a figure MEANS, and the unit it is in. A run reports `n_evaluated`; a
#: quant reads "measured". Anything not named here IS NOT PRINTED — a wrong
#: label on a right number is worse than no label, and this file already found
#: one shipped ("Observations" on a count of names). The cost of that rule is
#: the flip side: a figure a workflow emits and nobody labels here is a figure
#: the terminal silently withholds, which is how a run's own headline finding
#: can fail to reach the person who ran it.
_FIGURE_LABELS: dict[str, tuple[str, str]] = {
    "universe_size": ("universe", "names"),
    "n_evaluated": ("measured", "names"),
    "n_passing": ("passed", "names"),
    "n_insufficient": ("too little history", "names"),
    "n_returned": ("returned", "names"),
    "deflated_sharpe": ("deflated Sharpe", ""),
    "sharpe_annualized": ("Sharpe (ann.)", ""),
    "n_trials": ("trials", ""),
    "pbo": ("PBO", ""),
    "target_weight": ("target weight", "of capital"),
    "cvar_pct": ("CVaR", "%"),
    "var_pct": ("VaR", "%"),
    "max_drawdown_pct": ("max drawdown", "%"),
    "volatility_annualized_pct": ("volatility (ann.)", "%"),
    "min_track_record_length": ("min track record", "obs"),
    "n_obs": ("observations", ""),
    # The modeling-week artifacts (QUANTOS.md Phase 6.5).
    "mean_ic": ("mean IC", ""),
    "ic_t_stat": ("IC t-stat", ""),
    "spread_pct": ("top minus bottom", "% per period"),
    "half_life_periods": ("half-life", "periods"),
    "dies_at_bps": ("dies at", "bps"),
    "worst_segment_sharpe": ("worst segment Sharpe", ""),
    "share_of_pnl_in_best_segment": ("best segment's P&L share", ""),
    "longest_underwater_periods": ("longest underwater", "periods"),
    "time_underwater_share": ("time underwater", "share"),
    "correlation": ("correlation to book", ""),
    "beta_to_book": ("beta to book", ""),
    "n_names": ("names measured", ""),
    "usable": ("usable", "names"),
    "n_unreadable": ("unreadable", "names"),
    "n_ragged": ("ragged history", "names"),
    # SURVIVORSHIP, SEPARATED FROM RAGGEDNESS — same distinction the portal
    # draws. Ragged is a length test; this counts the names that LEFT, which
    # is the half that biases a study rather than merely weakening it.
    "n_ends_early": ("end before the panel", "names"),
    "n_nonpositive": ("zero or negative closes", "names"),
    "n_spike_names": ("split candidates", "names"),
    "n_stale_names": ("stale", "names"),
    "n_dates": ("panel length", "dates"),
}


def _num(value: Any, places: int = 2) -> str:
    """A number at the precision it actually carries.

    The portal shipped `173.719419` and `50.1046` into a table a PM reads. Six
    decimals on a percentage is not precision, it is noise wearing precision's
    clothes — the underlying series has nothing like that resolution, and the
    digits crowd out the two that matter.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, int) or float(value).is_integer():
        return f"{int(value):,}"
    return f"{float(value):,.{places}f}"


def _head(key: str) -> str:
    """A column head that fits without being amputated.

    `volatility_pct` truncated to a fixed width reads `volatility_p`, which is
    the same class of sloppiness as six-decimal percentages: the machine's field
    name leaking through where a person's word belongs. The unit suffix carries
    no information a quant needs in a header — every column here is a percent or
    a price and the values say which.
    """
    for suffix in ("_pct", "_percent", "_ratio"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
            break
    return key.replace("_", " ")[:14]


def _render_artifact(art: dict[str, Any]) -> None:
    """Show what the run FOUND, not that it finished.

    ── WHY THIS EXISTS ────────────────────────────────────────────────────────

    The previous version iterated the artifact and printed any value that was
    not a dict or a list. The artifact is `{workflow, emits, figures,
    citations}` — every value is a dict or a list — so it printed NOTHING. A
    closed screen said `Closed. screen_universe@1.0.0` and stopped, while the
    twenty names it had just ranked sat in the portal. The quant who ran it
    could not see their own result without opening a browser.

    ORDER IS THE RANKING, THEN WHAT QUALIFIES IT. The list is what was asked
    for; the counts are what make it readable. Printing the counts first buries
    the answer, and printing the list alone hands over a shortlist with no
    denominator — which is the exact shape of a survivorship claim.
    """
    figures = art.get("figures") or {}
    if not isinstance(figures, dict):
        return

    # Flatten one level: figures are keyed by stage (or by op), and a reader
    # wants the values, not the graph. Later stages win, which is right — a
    # stage that restates a figure is restating it more completely.
    flat: dict[str, Any] = {}
    for value in figures.values():
        if isinstance(value, dict):
            flat.update(value)

    # A MONITOR'S STATUS LEADS. It is four-way on purpose — `breaches: []`
    # reading as "all clear" when the truth is "nobody was watching" is this
    # product's own failure mode, and until this rendered, the terminal
    # committed it: the demo monitor closed `unchecked` and printed a Sharpe.
    if str(flat.get("status")) in ("ok", "breached", "undetermined", "unchecked"):
        _render_monitor_status(flat)

    # THE OVERLAP VERDICT IS A STRING and the figure block below prints only
    # numbers, so without this the one conclusion the run reached would be the
    # one thing its output omitted.
    verdict = flat.get("verdict")
    if verdict == "the_book_again":
        say("")
        say(
            red(f"  the book again {DOT} ")
            + "this candidate is substantially the bet the book already holds."
        )
    elif verdict == "diversifier":
        say("")
        say(green(f"  diversifier {DOT} ") + dim("low correlation to the book on their shared history."))
    elif verdict == "related":
        say("")
        say(
            yellow(f"  related {DOT} ")
            + "meaningfully correlated to the book; size it as an add, not a hedge."
        )

    rows = flat.get("rows")
    if isinstance(rows, list) and rows:
        _render_rows(rows, flat)

    shown = [
        (label, flat[key], unit)
        for key, (label, unit) in _FIGURE_LABELS.items()
        if isinstance(flat.get(key), (int, float)) and not isinstance(flat.get(key), bool)
    ]
    if shown:
        say("")
        # Two spaces after the pad, so a label that fills its field still
        # separates from its value — "worst segment Sharpe-3.20" welded a
        # 20-character label straight into a negative number.
        width = max(20, max(len(label) for label, _, _ in shown))
        for label, value, unit in shown:
            tail = f" {dim(unit)}" if unit else ""
            say(f"  {dim(label.ljust(width))}  {bold(_num(value))}{tail}")

    # SAID EVEN THOUGH THE TABLE IS RIGHT THERE. A shortlist of 20 means one
    # thing when 480 names were measured and rejected, and quite another when
    # 480 could not be measured at all.
    insufficient = flat.get("n_insufficient")
    if isinstance(insufficient, int) and insufficient > 0:
        say("")
        say(
            yellow(f"  {insufficient} names")
            + dim(" had too little history to measure and are not in this ranking.")
        )
    if flat.get("truncated"):
        say(dim("  More names passed than were returned; this is the top of them."))


def _render_monitor_status(flat: dict[str, Any]) -> None:
    """The four-way answer, in the words each state deserves.

    A breach LEADS and is itemised — observed against tolerance, because "which
    line, by how much" is the entire content of a breach. `unchecked` gets a
    sentence rather than a green blank: nothing was being watched, and that is
    a fact about the run, not an absence of news.
    """
    status = str(flat.get("status"))
    n_checked = flat.get("n_checked")
    say("")
    if status == "breached":
        breaches = [b for b in (flat.get("breaches") or []) if isinstance(b, dict)]
        say(red(f"  BREACHED {DOT} {len(breaches)} of {n_checked} limits checked"))
        for b in breaches:
            word = "above" if b.get("direction") == "above" else "below"
            say(
                red(f"    {b.get('limit')}")
                + f"  {_num(b.get('observed'))}, {word} its tolerance of {_num(b.get('tolerance'))}"
            )
        say(dim("  A monitor reports rather than stops: the artifact exists because something crossed."))
    elif status == "ok":
        say(
            green(f"  ok {DOT} ")
            + dim(f"{n_checked} limit{'s' if n_checked != 1 else ''} checked, none crossed")
        )
    elif status == "undetermined":
        names = ", ".join(
            str(u.get("limit")) for u in (flat.get("undetermined") or []) if isinstance(u, dict)
        )
        say(yellow(f"  undetermined {DOT} {names or 'some limits'} could not be measured on this data."))
    else:
        say(
            yellow(f"  unchecked {DOT} ")
            + "no tolerances were stated, so nothing was being watched. Not a green light."
        )
        say(dim('  State some:  --input tolerances={"max_drawdown_pct": 20, "var_pct": 3}'))
    unknown = flat.get("unknown_limits") or []
    if unknown:
        names = ", ".join(map(str, unknown))
        say(yellow(f"  named and not understood: {names}. Not watched, said rather than dropped."))


def _render_rows(rows: list[Any], flat: dict[str, Any]) -> None:
    """The shortlist, as a table a person can read down.

    The ranked metric is named in the header and NOT repeated as a second
    column. The portal shipped `SCORE` and `RETURN PCT` side by side holding the
    identical number, because `score` IS the ranked metric — two columns, one
    fact, and a reader left wondering which one to trust.
    """
    rank_by = str(flat.get("rank_by") or "score")
    metrics: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            for key, value in row.items():
                if (
                    key not in ("symbol", "ticker", "score", "_passed")
                    and key not in metrics
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                ):
                    metrics.append(key)
    metrics = metrics[:3]

    say("")
    say("  " + dim(f"ranked by {rank_by}") + dim(f"  {DOT}  {len(rows)} names"))
    # A two-space gutter between fields, on headers and values alike. Without
    # it, a header that exactly fills its field abuts its neighbour — the live
    # run printed `returnn observations`, one column's name welded to the next.
    head = f"  {'#':>3}  {'symbol':<8}{_head(rank_by):>14}"
    for m in metrics:
        head += f"  {_head(m):>14}"
    say(dim(head))

    scores = [abs(float(r.get("score") or 0)) for r in rows[:_SHOWN_ROWS] if isinstance(r, dict)]
    peak = max(scores) if scores else 1.0

    for i, row in enumerate(rows[:_SHOWN_ROWS], start=1):
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or row.get("ticker") or "?")
        score = row.get("score")
        meter = _bar(float(score or 0), peak, width=8)
        line = f"  {i:>3}  {bold(symbol.ljust(8))}{_num(score):>14}  {meter}"
        for m in metrics:
            line += f"  {_num(row.get(m)):>14}"
        say(line)

    if len(rows) > _SHOWN_ROWS:
        say(dim(f"       {ELLIPSIS} {len(rows) - _SHOWN_ROWS} more in the filed signal file"))


NEWLINE = chr(10)


def _extract_flags(line: str) -> tuple[str, list[str]]:
    """Pull `--data x` / `--universe y` / `--project z` out of any line.

    Returns (prose, flags). The prose is what the agent sees, so
    `screen my book --universe sp100` asks about the book and loads sp100,
    rather than asking a model to interpret a flag as part of a question.

    Only the four data flags. Anything else stays in the prose: a sentence
    containing "--" is more likely to be a sentence than a mistyped flag, and
    stripping tokens a user meant to say is worse than passing one through.
    """
    parts = line.split()
    prose: list[str] = []
    flags: list[str] = []
    i = 0
    while i < len(parts):
        tok = parts[i]
        if tok in ("--data", "--universe", "--project", "--symbol", "--thesis") and i + 1 < len(parts):
            flags += [tok, parts[i + 1]]
            i += 2
            continue
        prose.append(tok)
        i += 1
    return " ".join(prose), flags


def _universe_named_in(question: str, session: Any) -> str | None:
    """A universe of the caller's OWN that the question names.

    DETERMINISTIC AND NO MODEL CALL. "screen my sp100 universe" against a
    portal holding `sp100` is a substring match, and spending an inference on it
    would be slower, less reliable, and impossible to explain when it went wrong.

    Matched against the user's own registered names ONLY, so this can never
    invent a universe -- it either recognises something they own or returns
    nothing. Longest match wins, so `sp500-ex-financials` beats `sp500`.
    """
    try:
        rows = session.universes()
    except Exception:  # noqa: BLE001 - offline or unauthenticated, not an error here
        return None

    low = question.lower()
    hits = [
        str(u.get("name"))
        for u in rows
        if str(u.get("name") or "").lower() in low and str(u.get("name") or "")
    ]
    return max(hits, key=len) if hits else None


def _returns_from_universe(data: Any, symbol: str) -> list[float] | None:
    """One name's closes, as the per-period return series a sizing runs on.

    THE MISSING RUNG OF THE DATA LADDER. A user loaded their universe, was told
    `size_position needs returns`, passed the universe again, and got the same
    refusal — a loop with no exit, because nothing could turn 100 names' closes
    into the ONE series the workflow measures. Deriving returns from a named
    symbol's closes is data shaping, not workflow knowledge: the same job the
    CSV loader does when it reads a wide file.

    Handles all three series shapes a universe arrives in: bare floats (the
    portal's stored closes), `{date, close}` rows, and a `{date: close}`
    mapping (what a dated wide/long CSV loads as).
    """
    if not isinstance(data, dict):
        return None
    series = data.get(symbol.upper())
    closes: list[float]
    if isinstance(series, dict):
        # The shared reader sorts a mapping by its date key — the same order
        # every other consumer of the shape reads it in.
        from .core import series_values

        closes = [float(v) for v in series_values(series)[0]]
    elif isinstance(series, list):
        closes = []
        for row in series:
            if isinstance(row, dict):
                value = row.get("close")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    closes.append(float(value))
            elif isinstance(row, (int, float)) and not isinstance(row, bool):
                closes.append(float(row))
    else:
        return None
    if len(closes) < 2 or any(c == 0 for c in closes[:-1]):
        return None
    return [round(closes[i] / closes[i - 1] - 1.0, 8) for i in range(1, len(closes))]


def _symbol_in(question: str, data: Any) -> str | None:
    """The ONE universe name a question mentions, or None.

    Deterministic, matched against the loaded universe's own keys — "size MU"
    can only ever resolve to a name the caller supplied. Two matches is a real
    ambiguity and returns None rather than a guess.
    """
    if not isinstance(data, dict):
        return None
    tokens = {t.strip(".,!?:;()'\"").upper() for t in question.split()}
    hits = [k for k in data if isinstance(k, str) and k.upper() in tokens]
    return hits[0] if len(hits) == 1 else None


def _stored_universes(session: Any, question: str = "") -> tuple[str | None, list[str]]:
    """Which stored universe this run can reach for, and everything it could.

    "Run deep analysis on the given universe" was refused with "load some data"
    while the account held exactly one universe WITH stored closes — from where
    the user stands, the data IS given, and the refusal reads as the product
    being broken. Signed in, the portal's stored universes are the caller's
    own; reaching for one is the data-follows-you promise kept.

    Deterministic ladder, no model call: a universe the question NAMES wins
    (longest match); otherwise the account's ONLY stored universe; otherwise
    nothing — several candidates is a real ambiguity, and running on data
    nobody chose is worse than asking. The full list travels back so the
    refusal can name the choices.

    Only universes carrying `cache_id` qualify: names-without-closes cannot
    feed a run, and offering one would move the error two steps later.
    """
    try:
        rows = session.universes()
    except Exception:  # noqa: BLE001 - offline or unauthenticated; nothing to reach for
        return None, []
    ready = [
        str(u.get("name"))
        for u in rows
        if str(u.get("name") or "") and (u.get("definition") or {}).get("cache_id")
    ]
    low = question.lower()
    named = [n for n in ready if n.lower() in low]
    if named:
        return max(named, key=len), ready
    return (ready[0] if len(ready) == 1 else None), ready


def thesis_label(row: dict[str, Any]) -> str:
    """The name a person types. The id is what the run carries."""
    return str(row.get("name") or row.get("title") or row.get("id") or "").strip()


def thesis_text(row: dict[str, Any]) -> str:
    """The statement, under whichever key the portal sent it."""
    for key in ("statement", "summary", "body", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def thesis_id_of(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("thesis_id") or "").strip()


def match_thesis(rows: list[Any], query: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """One thesis, or the candidates when the name matches more than one.

    Exact name or id wins. A unique substring is enough. Several matches are
    returned so the caller can name them instead of guessing.
    """
    usable = [r for r in rows if isinstance(r, dict)]
    q = query.strip().casefold()
    exact = [r for r in usable if thesis_label(r).casefold() == q or thesis_id_of(r).casefold() == q]
    if len(exact) == 1:
        return exact[0], []
    if len(exact) > 1:
        return None, exact
    partial = [r for r in usable if q and q in thesis_label(r).casefold()]
    if len(partial) == 1:
        return partial[0], []
    return None, partial


def pop_flag(flags: list[str], name: str) -> tuple[str | None, list[str]]:
    """Take one ``--name value`` out of a flag list. The rest stay in order."""
    kept: list[str] = []
    value: str | None = None
    i = 0
    while i < len(flags):
        if flags[i] == name and i + 1 < len(flags):
            value = flags[i + 1]
            i += 2
            continue
        kept.append(flags[i])
        i += 1
    return value, kept


def _split_run(rest: str) -> tuple[str, list[str]]:
    """`screen_universe --universe sp100` -> ("screen_universe", [...]).

    Deliberately not argparse: the first token is a workflow name and the tail
    may be anything, and argparse would exit the PROCESS on a bad flag rather
    than the line. A session that dies on a typo is worse than one that says so.
    """
    parts = rest.split()
    if not parts:
        return "", []
    return parts[0], parts[1:]


def _open_spec(kind: str, spec: str, session: Any, data: Any, backtest_fn: Any) -> tuple[Any, Any, str]:
    """Load one door mid-session. Returns (data, backtest_fn, source label)."""
    spec = _unquote(spec)
    if kind == "project":
        data, backtest_fn = load_project(spec)
        say(dim(f"  loaded {spec}"))
        return data, backtest_fn, spec
    ns = argparse.Namespace(
        project=None,
        data=spec if kind == "data" else None,
        universe=spec if kind == "universe" else None,
        symbol=None,
    )
    loaded, fn = resolve_data(ns, session)
    if loaded is not None:
        data = loaded
    if fn is not None:
        backtest_fn = fn
    label = f"{kind}:{spec}"
    phrase = _shape_phrase(data)
    say(dim(f"  loaded {label}" + (f" {DOT} {phrase}" if phrase else "")))
    return data, backtest_fn, label


def _apply_flags(flags: list[str], session: Any, data: Any, backtest_fn: Any) -> tuple[Any, Any]:
    """Load whatever `--data` / `--universe` / `--project` name, mid-session.

    THE SAME FLAGS THE SHELL TAKES, because a message that prints
    `--universe sp100` must mean the same thing wherever it is read. Anything
    else is a tool teaching two dialects of itself.
    """
    ns = argparse.Namespace(project=None, data=None, universe=None, symbol=None)
    i = 0
    unknown: list[str] = []
    while i < len(flags):
        f = flags[i]
        value = flags[i + 1] if i + 1 < len(flags) else None
        if f == "--share-series":
            if session is not None:
                session.share_series = True
            i += 1
            continue
        if f in ("--data", "--universe", "--project", "--symbol") and value:
            setattr(ns, f[2:], value)
            i += 2
            continue
        unknown.append(f)
        i += 1
    if unknown:
        raise ValueError(
            f"I do not know {' '.join(unknown)} in a session. "
            "Here `run` takes --data, --universe, --project, --symbol and --share-series."
        )
    # `--symbol` narrows whatever the SESSION holds, so it is applied after the
    # merge below — inside `resolve_data` it saw only this line's own loads,
    # and `run size_position --symbol MU` against an already-loaded universe
    # answered "none loaded" about data sitting right there.
    symbol, ns.symbol = ns.symbol, None
    loaded, fn = resolve_data(ns, session)
    merged = loaded if loaded is not None else data
    if symbol:
        derived = _returns_from_universe(merged, str(symbol))
        if derived is None:
            names = (
                ", ".join(sorted(str(k) for k in merged)[:8]) if isinstance(merged, dict) else "none loaded"
            )
            raise ValueError(
                f"--symbol {symbol} needs that name in a loaded universe with at "
                f"least two closes. Loaded: {names}"
            )
        say(dim(f"  {str(symbol).upper()} {DOT} {len(derived)} returns {DOT} derived from its closes"))
        merged = derived
    return merged, (fn or backtest_fn)


def _describe_source(flags: list[str]) -> str | None:
    for key in ("--universe", "--data", "--project", "--symbol"):
        if key in flags:
            i = flags.index(key)
            if i + 1 < len(flags):
                return f"{key[2:]}:{flags[i + 1]}"
    return None


def _repl_gap(gap: str, name: str) -> str:
    """Preflight's message, rewritten for somebody who is already inside.

    THE LOOP THIS BREAKS. The shell-form message printed
    `alphaengine run screen_universe --universe <name>` -- correct advice at a
    shell prompt and nonsense at this one, where `alphaengine` is not a word the
    session knows. A user typed it, the session read it as a question, the agent
    chose the same workflow, and preflight printed the same message again.

    A tool that answers a refusal with instructions it cannot itself accept has
    stopped being a tool.
    """
    # A message that already names the exact next line — `run x --symbol MU`,
    # or a --project door for a gap loaders cannot fill — is right as it
    # stands; wrapping it in "load it first" told somebody with a universe
    # loaded to go and load a universe, which is the same loop this function
    # exists to break, reintroduced one level up. Only the words change for
    # the session: `alphaengine run` is not a line this prompt accepts.
    speaks_for_itself = "--symbol" in gap or (
        "--project" in gap and "--data" not in gap and "--universe" not in gap
    )
    if speaks_for_itself:
        text = gap.replace("alphaengine run ", "run ").replace("alphaengine demo", "demo")
        return NEWLINE.join(f"  {line}" if line.strip() else "" for line in text.splitlines())
    head = gap.splitlines()[0]
    return (
        f"  {head}"
        + NEWLINE
        + NEWLINE
        + "  From here, load it first:"
        + NEWLINE
        + "      load prices.csv            a local file"
        + NEWLINE
        + "      load <universe>            registered in the portal"
        + NEWLINE
        + f"      run {name} --universe <name>"
        + NEWLINE
        + NEWLINE
        + "  `workflows` lists what each one needs."
    )


def _near_verb(word: str) -> str | None:
    """The command `word` was probably meant to be, or None.

    Deliberately conservative: a shared prefix of three, or one character out.
    Anything looser starts hijacking real questions, and a research prompt whose
    input is second-guessed is worse than one that occasionally wastes a call.
    """
    from .commands import COMMANDS

    verbs = [c.verb for c in COMMANDS if c.verb.isalpha()]
    if word in verbs or len(word) < 3:
        return None

    best, best_d = None, 99
    for v in verbs:
        d = _edits(word.lower(), v)
        # EDIT DISTANCE, NOT A SHARED PREFIX. A prefix rule misses the most
        # common typo there is -- `wrokflows` transposes the second and third
        # letters, so it shares no prefix with `workflows` at all and sailed
        # through to the agent as a research question.
        if d < best_d:
            best, best_d = v, d
    # Two edits on a word of five or more, one on anything shorter. Looser than
    # that starts hijacking real questions, and a research prompt that
    # second-guesses its input is worse than one that occasionally wastes a call.
    limit = 2 if len(word) >= 5 else 1
    return best if best_d <= limit else None


def _edits(a: str, b: str) -> int:
    """Levenshtein distance. Small enough to write than to depend on."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _help_text() -> str:
    """The short list: how you actually run work.

    The full directory is `commands`. Dumping every verb at `help` made the
    session look like a program you had to learn rather than a prompt you
    type at.
    """
    rows = (
        ("demo", "see it work offline, no account"),
        ("process", "fit a named DGP and stress it, offline"),
        ("login", "sign in, or login anthropic for a model"),
        ("load <what>", "a CSV, a project module, or a universe"),
        ("screen", "what is worth a look"),
        ("validate", "is this real, once you count the tries"),
        ("run <workflow>", "scripted: same data, same place twice"),
        ("<a sentence>", "your model picks the path"),
    )
    width = max(len(cmd) for cmd, _ in rows)
    lines = ["", "  " + dim("Type a question, or:")]
    for cmd, why in rows:
        lines.append(f"    {bold(cmd.ljust(width))}  {dim(why)}")
    lines += [
        "",
        "  " + dim("Slash works too: /demo /login /load /help. `commands` for everything."),
    ]
    return "\n".join(lines)


def _signature(c: Command) -> str:
    return f"{c.verb} {c.args}".strip()


def _shape_phrase(data: Any) -> str | None:
    """`100 series` for a panel, `499 obs` for one series — never the values.

    A bare `len()` called a single symbol's returns "499 series" and a wide
    DataFrame's rows the same: the count was right and the noun was wrong,
    and a wrong noun on a right number teaches a reader to distrust the one
    line that exists to be re-read.
    """
    cols = getattr(data, "columns", None)
    if cols is not None:
        try:
            return f"{len(cols)} series"
        except TypeError:
            return None
    if isinstance(data, dict):
        return f"{len(data)} series"
    try:
        n = len(data)
    except TypeError:
        return None
    return f"{n} obs"


def status_line(url: str, *, data: Any, project: str | None, keyed: bool) -> str:
    """One quiet line above the prompt: what is loaded, and what can run.

    THE QUESTION THIS ANSWERS is the one asked most often inside a session and
    the one the boot screen answers once and then scrolls away: is my data in?
    Without it a user types `keys` or `status` purely to re-read state that was
    printed two minutes ago, and a run that silently had no data looked
    identical to one that did -- which is exactly how "the S&P 500 is a handful
    of names" happened.

    Deliberately ONE LINE and dim. Reverse-video chips became wallpaper; this
    is the Codex status strip: facts, not a second UI.
    """
    from .model import available_models

    bits: list[str] = []

    if data is None:
        bits.append("no data")
    else:
        phrase = _shape_phrase(data)
        label = project or "loaded"
        bits.append(label + (f" {DOT} {phrase}" if phrase else ""))

    bits.append("signed in" if keyed else "not signed in")

    models = available_models()
    bits.append(models[0][0] if models else "no model")

    return "  " + dim(f" {DOT} ".join(bits))


def cmd_commands(args: argparse.Namespace) -> int:
    """The directory. What you can type, grouped by the question it answers.

    ── WHY A DIRECTORY AND NOT A LONGER HELP ──────────────────────────────────

    The boot screen answers "where am I" — which workspace, signed in, what is
    loaded. NOTHING answered "what can I do". A reader who does not already know
    the verb cannot find it, and a flat list of thirteen makes them scan for a
    word they have not learned yet.

    Grouped by question, because that is the order somebody has them in: how do
    I start, how do I do the work, how do I get my data in, how do I manage the
    session.
    """
    from .commands import DATA_DOORS, GROUPS, by_verb, in_group

    wanted = (getattr(args, "verb", None) or "").strip()
    if wanted:
        return _one_command(by_verb(wanted), wanted)

    say("")
    for group, title in GROUPS:
        rows = in_group(group)
        if not rows:
            continue
        say("  " + dim(title))
        width = max(len(_signature(c)) for c in rows)
        for c in rows:
            where = "" if c.scope == "both" else dim(f"  ({c.scope})")
            say(f"    {_signature(c).ljust(width)}   {dim(c.purpose)}{where}")
        say("")

    # THE DOORS GET THEIR OWN BLOCK. They are flags rather than verbs, and they
    # are the single thing people are most often hunting for when they open a
    # directory at all — "how do I give it my data" is not answerable from a
    # list of verbs.
    say("  " + dim("Getting your data in"))
    width = max(len(d) for d, _ in DATA_DOORS)
    for door, what in DATA_DOORS:
        say(f"    {door.ljust(width)}   {dim(what)}")
    say("")
    say("  " + dim("`commands <verb>` for one in full."))
    return 0


def _one_command(c: Command | None, wanted: str) -> int:
    from .commands import COMMANDS, FLAGS

    if c is None:
        near = [x.verb for x in COMMANDS if x.verb.startswith(wanted[:3])]
        say(red(f"No command called {wanted!r}."))
        if near:
            say(dim("  Did you mean: " + ", ".join(near) + "?"))
        say(dim("  `commands` lists everything."))
        return 2

    say("")
    say("  " + bold(_signature(c)))
    say("  " + dim(c.purpose))
    if c.scope != "both":
        say("  " + dim(f"Works in: {c.scope}"))
    if c.body:
        say("")
        for para in c.body.split("\n\n"):
            for line in _wrap(para.replace("\n", " "), width=72):
                say("  " + line)
            say("")
    if c.flags:
        say("  " + bold("Flags"))
        width = max(len(f"{FLAGS[f].name} {FLAGS[f].takes}".strip()) for f in c.flags)
        for f in c.flags:
            flag = FLAGS[f]
            sig = f"{flag.name} {flag.takes}".strip()
            say(f"    {sig.ljust(width)}   {dim(flag.purpose)}")
        say("")
    if c.examples:
        say("  " + bold("Examples"))
        for ex in c.examples:
            say("    " + cyan(ex))
        say("")
    return 0


# ── the capability ladder ──────────────────────────────────────────────────
#
# THREE RUNGS, AND THE FIRST IS FREE FOREVER.
#
#   MATH      no key of any kind. `sweep`, deflation, the surface, `save`. The
#             whole open core, offline, with no account. This rung is not a
#             trial and never expires.
#   HARNESS   a QuantOS key. Workflows: the SEQUENCE lives on the server, so
#             this is the paid rung. Scripted and reproducible.
#   AGENT     a model key OF YOUR OWN, on top of the harness. Plain English at
#             the prompt; the model chooses each step from what the server
#             permits. Exploratory, and not reproducible.
#
# WHY THE LADDER IS DRAWN AT ALL. Somebody with no keys was previously told
# "No API key" and left to guess what that cost them — which reads as "this
# tool does not work" when the truth is "two thirds of it works and always
# will". Showing what is unlocked, what is not, and the single environment
# variable that moves each line is the difference between a paywall and an
# honest boundary.
#
# The rungs are ordered because they genuinely nest: the agent drives the
# harness, and the harness runs the math. You cannot buy the third without the
# second, and you never have to buy the first.


def _shell_set(name: str, stored: dict[str, str]) -> bool:
    """True when the live value did NOT come from the stored file.

    `apply_stored` never overrides the environment, so a value that differs from
    the stored one was exported in the shell.
    """
    return os.environ.get(name, "") != stored.get(name, "")


def ladder_lines(*, keyed: bool) -> list[str]:
    """The three rungs, with what is unlocked and how to unlock the rest."""
    from .model import available_models, keys_without_sdk

    models = available_models()
    # A KEY WITH NO SDK IS A DIFFERENT SITUATION FROM NO KEY, and it wants a
    # different sentence. "Add your own model key" told somebody who already had
    # one to go and get one, which reads as the product being broken.
    stranded = keys_without_sdk()
    on, off = ON, OFF

    rungs = [
        (
            True,  # always
            "the maths",
            "search a grid, deflate it, see the shape",
            dim("yours, offline, always"),
        ),
        (
            keyed,
            "workflows",
            "a sequence worth trusting, end to end",
            green("ready") if keyed else yellow("sign in to unlock"),
        ),
        (
            keyed and bool(models),
            "ask anything",
            "describe what you want to know",
            (
                green(f"ready  {DOT}  {models[0][0]}")
                if (keyed and models)
                else yellow(f"{stranded[0]} key found {DOT} SDK missing")
                if (keyed and stranded)
                else yellow("add your own model key")
                if keyed
                else dim("sign in first")
            ),
        ),
    ]

    out: list[str] = []
    for live, name, _does, note in rungs:
        mark = green(on) if live else dim(off)
        label = bold(name.ljust(14)) if live else dim(name.ljust(14))
        out.append(f"  {mark}  {label}{note}")

    # ONE ACTIONABLE LINE, and only when something is locked. The rungs
    # themselves stay in plain language — a menu that recites environment
    # variables reads like a manual — but a reader who cannot do the thing they
    # came for still needs to be told exactly how, without going to look it up.
    if not keyed:
        out += ["", "  " + dim("Type ") + bold("login") + dim(f" to sign in, or export {_ENV_KEY}.")]
    elif not models and stranded:
        # ONE STEP AWAY, so say which step. This is the state the owner hit: a
        # key set, the SDK absent, and a boot screen that suggested getting a key.
        out += [
            "",
            "  "
            + dim(f"Your {stranded[0]} key is set and its SDK is not installed. Run ")
            + bold(f"pip install {stranded[0]}")
            + dim("."),
        ]
    elif not models:
        out += [
            "",
            "  "
            + dim("Type ")
            + bold("login anthropic")
            + dim(" (or ")
            + bold("login openai")
            + dim(") to ask questions in your own words."),
        ]
    return out


def _quoted(name: str) -> str:
    """A universe name as it must be TYPED. Names contain spaces."""
    return f'"{name}"' if " " in name else name


def account_data_lines(session: Any) -> list[str]:
    """What this account has registered in the portal, and which of it can run.

    ── WHY THE BOOT SCREEN MAKES A NETWORK CALL AT ALL ────────────────────────

    It said `data  none  ·  pass --project to load yours` to a user signed in to
    an account holding a registered universe, and never mentioned `--universe`.
    Both halves of the product were working and the screen that exists to say
    where you are could not see one of them. "The OS cannot answer a question
    about data it cannot reach" was the diagnosis; being unable to NAME the data
    is the same failure one step earlier.

    ONE REQUEST, AND ONLY WHEN IT HELPS. The caller checks that we are signed in
    and that nothing was loaded from disk first, so this never fires on a run
    that already knows its data. A failure returns nothing: an offline boot
    screen is a normal boot screen, not an error.

    IT ALSO NAMES THE HALF-REGISTERED CASE, which is the one that cost the most
    time — a universe whose names reached the account and whose prices did not
    looks identical to a healthy one from every surface except the run that
    fails on it.
    """
    if session is None:
        return []
    try:
        rows = session.universes()
    except Exception:  # noqa: BLE001 - offline, refused, or slow: say nothing
        return []
    if not rows:
        return []

    ready: list[tuple[str, int, int]] = []
    unstored: list[tuple[str, int]] = []
    for u in rows:
        definition = u.get("definition") or {}
        name = str(u.get("name") or "")
        symbols = definition.get("symbols") or []
        n_obs = definition.get("n_obs")
        if definition.get("cache_id"):
            ready.append((name, len(symbols), int(n_obs or 0)))
        else:
            unstored.append((name, len(symbols)))

    out = ["  " + dim("Registered in the portal")]
    for name, n_names, n_obs in ready[:6]:
        shape = f"{n_names} names" + (f" {DOT} {n_obs} obs" if n_obs else "")
        out.append(
            f"  {green(ON)} {bold(name.ljust(22))}{dim(shape.ljust(26))}" + dim(f"--universe {_quoted(name)}")
        )
    for name, n_names in unstored[:6]:
        out.append(
            f"  {dim(OFF)} {dim(name.ljust(22))}{dim(f'{n_names} names'.ljust(26))}"
            + yellow("prices not stored")
        )
    hidden = (len(ready) - 6) + (len(unstored) - 6)
    if hidden > 0:
        out.append("  " + dim(f"  and {hidden} more"))
    if unstored:
        # SAID HERE BECAUSE HERE IS WHERE IT IS ACTIONABLE. The alternative is
        # discovering it from a run that refuses, which reads as the run's fault.
        out.append("")
        out.append("  " + dim("A universe with no stored prices holds names only. Open it on the "))
        out.append("  " + dim("portal's Data page and store them to run on it from here."))
    return out


def account_sleeve_lines(session: Any) -> list[str]:
    """The books this account can run against, if it is in a pod.

    Listed beside the universes because they answer the same question — what
    can I point this at — and because a sleeve carrying its own universe makes
    the two ONE choice rather than two. An account with no pod gets nothing
    here rather than an empty heading: working solo is the ordinary case, not a
    setup step somebody skipped.
    """
    if session is None:
        return []
    try:
        rows = session.sleeves()
    except Exception:  # noqa: BLE001 - offline, or no pod: say nothing
        return []
    if not rows:
        return []

    out = ["  " + dim("Sleeves in your pod")]
    for s in rows[:6]:
        name = str(s.get("name") or "")
        stage = str(s.get("stage") or "")
        lit = bool(s.get("universe_id"))
        shape = stage + (f" {DOT} brings its own data" if lit else "")
        label = bold(name.ljust(22)) if lit else dim(name.ljust(22))
        out.append(
            f"  {green(ON) if lit else dim(OFF)} {label}"
            + dim(shape.ljust(30))
            + dim(f"--sleeve {_quoted(name)}")
        )
    if len(rows) > 6:
        out.append("  " + dim(f"  and {len(rows) - 6} more"))
    return out


def _next_moves(*, keyed: bool, data: Any, project: str | None) -> list[str]:
    """The next three things to type. A boot screen that cannot name a first
    move is a logo."""
    from .model import available_models

    models = available_models()
    rows: list[tuple[str, str]] = []
    if not keyed:
        rows = [
            ("demo", "the offline half, no account"),
            ("login", f"unlock workflows, or export {_ENV_KEY}"),
            ("login anthropic", "then ask in your own words"),
        ]
    elif data is None:
        rows = [
            ("load prices.csv", "a local file"),
            ("load <universe>", "stored closes from the portal"),
            ("demo", "see a grid survive, or not"),
        ]
        if not models:
            rows[2] = ("login anthropic", "ask in your own words")
    else:
        rows = [
            ("screen", "what is worth a look"),
            ("validate", "count the tries"),
        ]
        if models:
            rows.append(('"which names rank highest?"', "the model picks the path"))
        else:
            rows.append(("login anthropic", "ask in your own words"))
        if project:
            rows[1] = ("validate", "count the tries on this project")
    out = ["", "  " + dim("Next")]
    width = max(len(cmd) for cmd, _ in rows)
    for cmd, why in rows:
        out.append(f"    {bold(cmd.ljust(width))}  {dim(why)}")
    return out


def boot(url: str, *, project: str | None, data: Any, keyed: bool, session: Any = None) -> None:
    """The opening screen. A transcript header, not a dashboard.

    Suppressed entirely when stdout is not a TTY: a CI log wants a transcript.
    """
    from ._version import __version__

    if not _tty():
        say(f"alphaengine {__version__}  {DOT}  {url}")
        return

    cols = _canvas_cols()
    say("")
    for line in _wordmark_lines(cols):
        say("  " + line)
    say("")
    _play_canvas()
    say(
        "  "
        + _teal(
            "computation runs locally; when you are signed in to quantOS, summary figures are reported "
            "to your account; full series are shared only if you opt in."
        )
    )
    say("  " + dim("Research tool, not investment advice."))
    say("")

    # Data is described by SHAPE, never printed. A boot screen that echoes the
    # first rows of somebody's price series into their scrollback — where it
    # will sit until the buffer rolls, and into any recording of the session —
    # is a boundary violation for the sake of a nicer banner.
    bits: list[str] = []
    bits.append(dim(__version__))
    if data is None:
        bits.append(dim("no data"))
    else:
        phrase = _shape_phrase(data)
        label = project or "loaded"
        bits.append(green(label + (f" {DOT} {phrase}" if phrase else "")))
    bits.append(green("signed in") if keyed else dim("not signed in"))
    if keyed:
        from .auth import load_stored

        stored = load_stored()
        if "QUANTOS_API_KEY" in stored and not _shell_set("QUANTOS_API_KEY", stored):
            bits.append(dim("stored"))
        else:
            bits.append(dim("from your shell"))
    say("     " + f" {dim(DOT)} ".join(bits))

    for line in _next_moves(keyed=keyed, data=data, project=project):
        say(line)

    # ONLY WHEN IT CHANGES THE ANSWER: signed in, and nothing already loaded
    # from disk. A run that knows its data does not need a directory of the
    # account's, and the call is not worth making to print something ignorable.
    if keyed and data is None:
        for lines in (account_data_lines(session), account_sleeve_lines(session)):
            if lines:
                say("")
                for line in lines:
                    say(line)

    say("")


def refused(exc: Any) -> None:
    """Report a server refusal, and make an AUTH refusal actionable.

    ── WHY THIS IS NOT JUST A RED LINE ────────────────────────────────────────

    "The server refused: Authentication required" is true, useless, and actively
    misleading to somebody who IS a paying user — they are signed into the
    portal in a browser two windows away, so the natural reading is that
    something is broken. It is not: a browser session is not a CLI credential
    and never will be, because this process has no cookie jar and should not
    acquire one.

    So a 401 says what is actually wrong, where the key comes from, and offers
    to take it right here. Every other status keeps the plain message, because
    the fix for a 404 or a 422 is not a credential.
    """
    status = getattr(exc, "status", None)
    detail = getattr(exc, "detail", str(exc))

    if status not in (401, 403):
        say(red(f"The server refused: {detail}"))
        return

    say("")
    say(
        yellow("  Not authenticated.")
        + dim(
            "  This command requires a portal-issued ae_live_ key; "
            "your browser login does not reach this process."
        )
    )
    say(dim("  A browser session is a cookie; this needs a key. They are different"))
    say(dim("  credentials on purpose."))
    say("")
    say(
        "  " + dim("Need one:") + " a portal-issued ae_live_ key (request access at https://qosai.app/access)"
    )
    say("  " + dim("Get one:") + f"  the portal {ARROW} Data {ARROW} API keys {ARROW} New key")
    say("  " + dim("Then:") + "     " + bold("login") + dim(f"   (or export {_ENV_KEY}=ae_live_...)"))
    say(
        "  "
        + dim("Offline:")
        + "  "
        + bold("alphaengine demo")
        + dim(", ")
        + bold("alphaengine process")
        + dim(", and the Python API (")
        + bold("import alphaengine")
        + dim(") work without an account.")
    )
    say("")


def offer_key(which: str = "quantos") -> bool:
    """Ask whether to enter a key now. True if one was entered.

    Only ever called on an interactive terminal — a prompt in a CI log is a
    hang, so callers check `_tty()` first.
    """
    if not _tty() or _SINK.get() is not None:
        return False
    try:
        answer = input(bold("  Enter a key now? ") + dim("[Y/n] ")).strip().lower()
    except (EOFError, KeyboardInterrupt):
        say("")
        return False
    if answer and not answer.startswith("y"):
        return False
    return enter_key(which)


def is_auth_error(exc: Any) -> bool:
    return getattr(exc, "status", None) in (401, 403)


_KEY_PROMPTS: dict[str, tuple[str, str]] = {
    "quantos": ("QUANTOS_API_KEY", "QuantOS key (ae_live_…) — created in the portal, on Data"),
    "anthropic": ("ANTHROPIC_API_KEY", "Anthropic key (sk-ant-…)"),
    "openai": ("OPENAI_API_KEY", "OpenAI key (sk-…) or any OpenAI-compatible key"),
    "gemini": ("GEMINI_API_KEY", "Gemini / Google AI key"),
    "google": ("GOOGLE_API_KEY", "Google AI key"),
    "groq": ("GROQ_API_KEY", "Groq key"),
    "openrouter": ("OPENROUTER_API_KEY", "OpenRouter key"),
    "azure": ("AZURE_OPENAI_API_KEY", "Azure OpenAI key (also set AZURE_OPENAI_ENDPOINT)"),
    "gateway": ("ALPHAENGINE_API_KEY", "Private gateway key (also set ALPHAENGINE_BASE_URL)"),
}


def enter_key(which: str) -> bool:
    """Prompt for a key and hold it FOR THIS SESSION ONLY.

    Persistence is optional and lives in `auth.save_key` (mode 0600, user config
    dir). QuantOS never holds the key.
    """
    import getpass

    if which not in _KEY_PROMPTS:
        say(red(f"unknown key {which!r} — try: {', '.join(_KEY_PROMPTS)}"))
        return False
    env, what = _KEY_PROMPTS[which]

    say(dim(f"  {what}"))
    try:
        value = getpass.getpass("  paste it (input hidden): ").strip()
    except (EOFError, KeyboardInterrupt):
        say("")
        return False
    if not value:
        say(dim("  nothing entered."))
        return False

    os.environ[env] = value
    if which != "quantos":
        from .model import provider_for_env

        prov = provider_for_env(env)
        if prov:
            os.environ["ALPHAENGINE_PROVIDER"] = prov
            os.environ.pop("ALPHAENGINE_MODEL", None)
    say(green(f"  {env} accepted."))

    # STAY SIGNED IN. Re-pasting a key into every new terminal is not security,
    # it is friction — and friction is what puts `QUANTOS_API_KEY=ae_live_...`
    # into a shell profile or a committed .env, which is strictly worse than a
    # mode-0600 file this tool manages and can delete.
    try:
        keep = (
            input(
                bold("  Stay signed in on this machine? ")
                + dim("[Y/n]  (yes writes a 0600 file to your user profile) ")
            )
            .strip()
            .lower()
        )
    except (EOFError, KeyboardInterrupt):
        say("")
        keep = "n"
    if keep and not keep.startswith("y"):
        say(dim("  Held for this session only."))
        return True

    from .auth import save_key

    try:
        path = save_key(env, value)
        if which != "quantos":
            from .model import provider_for_env

            prov = provider_for_env(env)
            if prov:
                save_key("ALPHAENGINE_PROVIDER", prov)
    except OSError as exc:
        say(yellow(f"  Could not write the credentials file ({exc}). Session only."))
        return True
    say(green("  Signed in.") + dim(f"  {path}  (mode 0600, `logout` to remove)"))
    return True


def _is_workflow(session: Any, name: str) -> bool:
    """Is this the name of a workflow, or the first word of a sentence?

    Asked of the SERVER rather than guessed, because the catalogue is the only
    authority on what a workflow is called. A network failure answers False,
    which routes the line to the agent — the friendlier wrong answer of the two.
    """
    try:
        return any(str(w.get("name")) == name for w in session.workflows())
    except Exception:  # noqa: BLE001 - offline/refused is handled downstream
        return False


def _workflow_answer(row: dict[str, Any]) -> str:
    """What this workflow answers, from the server when it says, else the local sentence."""
    for key in ("answers", "purpose", "description", "summary"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return _WORKFLOW_ANSWERS.get(str(row.get("name") or ""), "")


def _catalogue_option(row: dict[str, Any]) -> dict[str, Any]:
    """One catalogue row, as a choice the model can tell from the others.

    The name alone is why every question was landing on the same workflow.
    The sentence, what it needs, and whether it repeats are the distinction.
    """
    name = str(row.get("name") or "?")
    params: dict[str, Any] = {}
    answers = _workflow_answer(row)
    if answers:
        params["answers"] = answers[:180]
    needs = [str(item) for item in (row.get("requires") or []) if str(item)]
    if needs:
        params["needs"] = ", ".join(needs[:6])
    agency = row.get("agency")
    if agency:
        params["agency"] = str(agency)
    if row.get("reproducible") is True:
        params["reproducible"] = "yes"
    elif row.get("reproducible") is False:
        params["reproducible"] = "no"
    return {"op": name, "params": params, "answers": answers}


def _prior_line(run: Any) -> str:
    """One line of what the desk already ran, so the next question is not blind."""
    from .client.agent import summarize_figures

    workflow = str((getattr(run, "artifact", None) or {}).get("workflow") or "the last run")
    line = f"{workflow} is {getattr(run, 'status', '')}."
    measured = summarize_figures(getattr(run, "figures", None))
    if measured != "(nothing yet)":
        line = f"{line} {' '.join(measured.split())[:240]}"
    return line


def _ask(
    session: Any,
    url: str,
    request: str,
    *,
    data: Any,
    backtest_fn: Any,
    workspace_id: str | None = None,
    thesis_id: str | None = None,
    thesis_name: str | None = None,
    thesis_statement: str | None = None,
    prior: str | None = None,
    think: Callable[[str], str] | None = None,
) -> Any:
    """Natural language → a workflow chosen and driven by the user's own model.

    THE PART THAT IS EASY TO GET WRONG: the model picks the WORKFLOW from the
    server's catalogue and then picks each STEP from what the server permits. It
    never composes a sequence. Everything it is allowed to do is a choice among
    options somebody else produced, which is why this can be non-deterministic
    without being unsafe.
    """
    from .client import AgentDriver, AgentRefusal, Offline, ServerError
    from .client.agent import prefer_index
    from .events import EventSink
    from .model import NoModelConfigured, build_think

    # ── NOT STREAMED, AND THE REASON IS WHAT THE CALL RETURNS ──────────────
    #
    # `pick` asks the model for a STRUCTURED answer — `{"choice": n, "why":
    # "..."}` — and streaming it printed the raw JSON to the user's terminal,
    # code fence and all, wrapped at 74 columns, immediately followed by
    # `on_thought` printing the parsed `why` again. Two renderings of one
    # sentence, one of them machine format.
    #
    # The lesson generalises: stream a call whose output is PROSE, never one
    # whose output is a wire format. `on_thought` already narrates this one, in
    # the words a person wrote it in.
    label = "your model"
    if think is None:
        try:
            think, label = build_think()
        except NoModelConfigured as exc:
            say(yellow(str(exc)))
            return None

    provider, _, model_name = label.partition("/")
    sink = EventSink(session=session if os.environ.get(_ENV_KEY) else None)

    try:
        catalogue = session.workflows()
    except Offline:
        _explain_offline(url)
        return None
    except ServerError as exc:
        refused(exc)
        return None

    if not catalogue:
        say(dim("The server offers no workflows, so there is nothing to choose from."))
        return None

    say(f"  model    {dim(label)}")
    if thesis_name:
        say(dim(f"  thesis   {thesis_name}"))
    if prior:
        say(dim("  prior    " + prior.splitlines()[0][:88]))
    goal = request
    if thesis_statement:
        goal = f"{request}\n\nThesis:\n{thesis_statement}"
    if prior:
        goal = f"{goal}\n\nAlready on the desk:\n{prior}"

    # Choosing the workflow is the same shape as choosing a step: an index into
    # a list the server produced. Reusing `AgentDriver.pick` rather than writing
    # a second selector keeps one place where a model's answer is validated.
    driver = AgentDriver(
        think,
        goal=goal,
        on_thought=lambda why: say(f"  why      {why}"),
        on_step=lambda line: say(f"  {line}"),
        sink=sink,
        provider=provider or None,
        model=model_name or None,
    )
    options = [_catalogue_option(w) for w in catalogue]
    try:
        say(dim("  reading  the catalogue"))
        with Working(dim("reading the catalogue"), plain="reading the catalogue"):
            matched = prefer_index(request, options)
            if matched is None:
                index = driver.pick(options)
            else:
                index = matched
                say(dim("  matched  the question uses this workflow's own words"))
        chosen = catalogue[index]
    except AgentRefusal as exc:
        say(red(f"The model did not choose a workflow: {exc}"))
        return None

    name = str(chosen.get("name"))

    # THE ACCOUNT IS REACHED FOR BEFORE REFUSING. A signed-in user whose portal
    # holds a stored universe was told to "load some" data — a refusal that
    # reads as the OS being broken, because from where they stand the data IS
    # given. If the chosen workflow wants a universe and none is loaded, the
    # one the question names — or the account's only one — is fetched, with
    # the usual line saying which data ran. Several candidates stay a refusal
    # that names them: running on data nobody chose is the worse failure.
    requires = [str(r) for r in (chosen.get("requires") or [])]
    ready: list[str] = []
    if data is None and ("universe" in requires or "returns" in requires):
        wanted, ready = _stored_universes(session, request)
        if wanted:
            try:
                data = _narrow_to_universe(session, wanted, None)
            except Exception as exc:  # noqa: BLE001 - reachable and unloadable: say so, refuse below
                say(yellow(f"  {wanted} is on your account and could not be loaded: {exc}"))

    # check_overlap on a universe is one name against the equal-weight of the
    # rest. Collapsing --symbol here used to throw the book away, which is why
    # the next line was "needs book_returns, and none was loaded."
    if name == "check_overlap" and _looks_like_universe(data):
        sym = _symbol_in(request, data)
        if sym:
            payload = _overlap_payload(data, sym)
            if payload is None:
                return None
            data = payload
        else:
            say("  check_overlap measures one name against the equal-weight of the others.")
            say(dim("  No name was given, so this is every name against every other name."))
            _print_cross(data, "correlation", session=session, thesis_id=thesis_id)
            return None

    # "size MU" against a loaded universe is a complete instruction: the name
    # the question mentions picks the series out of the caller's own data.
    # One match only — two is a real ambiguity and the refusal below names the
    # move rather than this code guessing between them.
    if "returns" in requires and not _looks_like_returns(data):
        sym = _symbol_in(request, data)
        if sym:
            derived = _returns_from_universe(data, sym)
            if derived:
                say(dim(f"  {sym} {DOT} {len(derived)} returns {DOT} derived from its closes"))
                data = derived

    # PREFLIGHT ON THIS PATH TOO. It was called only from `cmd_run`, so the
    # guard built to say "this workflow needs X and none was loaded" BEFORE a
    # run churns did not run on the path people actually type into.
    #
    # The cost of that was not a missing message, it was a WRONG one: with no
    # data the run started, `data.resolve` reported zero names, and the screen's
    # gate answered with the sentence written for a small universe -- "that is a
    # handful of names rather than a universe" -- when nothing had been supplied
    # at all. A refusal that names the wrong cause is worse than no refusal,
    # because it sends somebody to fix a thing that was never broken.
    gap = preflight(catalogue, name, data=data, backtest_fn=backtest_fn)
    if gap:
        say(yellow(f"  {name} is the right workflow for that, and it cannot start yet."))
        for line in gap.splitlines()[1:]:
            say(dim(line) if line.strip() else "")
        if len(ready) > 1:
            say("")
            say(yellow(f"  Your account holds {len(ready)} stored universes: {', '.join(ready)}."))
            say(dim('  Name one in the question ("screen my s&p universe") or pass --universe <name>.'))
        return None

    repro = chosen.get("reproducible")
    say(f"  workflow {bold(name)}   " + (green("reproducible") if repro else yellow("exploratory")))

    try:
        run = session.open(
            name,
            data=data,
            backtest_fn=backtest_fn,
            workspace_id=workspace_id,
            thesis_id=thesis_id,
            **({"grid": project_grid()} if project_grid() else {}),
        )
        sink.bind(run.run_id)
        driver.drive(run)
    except Offline:
        _explain_offline(url)
        return None
    except ServerError as exc:
        refused(exc)
        return None
    except AgentRefusal as exc:
        say(red(f"The run stopped: {exc}"))
        return None

    say(dim("  result"))
    _report(run)
    if run.status == "closed":
        _publish_signals(session, run, workspace_id=workspace_id, label=None)
    _answer(request, run, think, sink=sink, provider=provider, model=model_name)
    # SAID AT THE END, WHERE IT TRAVELS WITH THE RESULT. A study whose path was
    # chosen by a model is a different object from one whose path was fixed in
    # advance, and the difference has to reach whoever reads the artifact.
    say(dim("  The model chose this path. Another run may take a different one."))
    return run


_ANSWER_PROMPT = """You are answering a question from a research run that has already finished.

The question: {question}

Everything the run recorded, as name -> number:
{figures}

Write two or three sentences answering the question from these figures.

RULES, and the first is enforced after you reply:
  * Quote ONLY numbers that appear above. An answer citing anything else is
    thrown away, not corrected.
  * If the figures do not answer the question, say that plainly. "The run did
    not measure that" is a good answer.
  * No recommendation the figures do not support. No hedging language added to
    a clear result, and no confidence added to an unclear one.
  * Do not restate caveats about missing or unrecorded values; those are added
    afterwards from the figures themselves and would be duplicated."""


def _answer(
    request: str,
    run: Any,
    think: Callable[[str], str],
    *,
    sink: Any | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> None:
    """The synthesis pass, which is the difference between a loop running and
    the system answering.

    A SEPARATE CALL FROM THE ONE THAT DROVE THE RUN, on purpose. `pick` returns
    an index into what the server offered, and that narrowness is the safety
    property; widening it to also produce prose would trade the guarantee for a
    convenience. This asks a different question with a different contract, and
    the contract is enforced by `synthesize` rather than by the prompt: an answer
    quoting a figure the run did not produce is REFUSED.

    A failure here loses the answer and keeps the run. The figures are already
    printed above by `_report`, so the worst case is the behaviour that shipped
    before this existed.
    """
    from .agent import UncitedFigure, synthesize

    def write(question: str, figures: dict[str, Any]) -> str:
        lines = "\n".join(f"  {k} = {v}" for k, v in sorted(figures.items()))
        return think(_ANSWER_PROMPT.format(question=question, figures=lines or "  (none)"))

    # WATCHED, NOT STREAMED, and the asymmetry with the reasoning above is the
    # point: an answer can be REFUSED after it is generated, when it cites a
    # figure the run did not produce. Streaming it would put text on the screen
    # that the guard then throws away, which is worse than a wait — a reader who
    # saw it will remember the number whether or not we retracted it.
    try:
        with Working(dim("composing the answer"), plain="composing the answer"):
            answer = synthesize(request, run, write=write)
    except UncitedFigure as exc:
        # NAMED, NOT SWALLOWED. This is the guard working, and a user who sees
        # nothing cannot tell it from a model that had nothing to say.
        say(yellow(f"  The answer was refused: {exc}"))
        if sink is not None:
            from .events import make_event

            sink.emit(
                make_event(
                    "refusal",
                    run_id=getattr(run, "run_id", None),
                    error=str(exc),
                    provider=provider,
                    model=model,
                )
            )
        return
    except Exception:  # noqa: BLE001
        say(dim("  No answer could be composed from this run."))
        return

    if sink is not None:
        from .events import make_event

        sink.emit(
            make_event(
                "answer",
                run_id=getattr(run, "run_id", None),
                provider=provider,
                model=model,
                answer_text=answer.text,
                answer_caveats=list(answer.caveats),
                agency="exploratory",
            )
        )

    say("")
    for line in _wrap(answer.text):
        say(f"  {dim(NEST)} {line}")
    # THE CAVEATS ARE NOT OPTIONAL AND ARE NOT STYLED AWAY. Each one is a claim
    # the figures cannot support without it, and every honesty control in this
    # product exists to survive exactly this last step to the screen.
    for caveat in answer.caveats:
        for i, line in enumerate(_wrap(caveat, width=74)):
            say(dim(f"    {'· ' if i == 0 else '  '}{line}"))
    if answer.verdict_ceiling:
        say(dim(f"    {DOT} No claim above {bold(answer.verdict_ceiling)} is supportable here."))


def _wrap(text: str, width: int = 76) -> list[str]:
    """Hard-wrap without importing textwrap for one call site."""
    out: list[str] = []
    line = ""
    for word in str(text).split():
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


def cmd_session(args: argparse.Namespace) -> int:
    """The full-screen session. One-shot commands stay on this parser."""
    try:
        from .tui.app import launch
    except ImportError:
        say("The session is a full-screen terminal.")
        say("Install it with:  pip install 'alphaengine[tui]'")
        return 2
    return launch(args)


def cmd_repl(args: argparse.Namespace) -> int:
    """One session. The full-screen terminal is it."""
    return cmd_session(args)


# ── entry point ────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """The global flags are attached to EVERY subparser as well as the root.

    argparse binds an option to the parser that declared it, so a flag declared
    only at the root has to be typed BEFORE the subcommand:

        alphaengine --url http://... workflows      # works
        alphaengine workflows --url http://...      # "unrecognized arguments"

    Nobody types the first one. Sharing a parent parser makes both positions
    valid, which costs three lines and removes an error message whose advice
    ("unrecognized arguments") points at the wrong thing entirely.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--url", help=f"workflow server (default ${_ENV_URL} or the public API)")
    common.add_argument("--key", help=f"portal-issued ae_live_ key (default ${_ENV_KEY})")
    common.add_argument("--project", help="module exposing `data` and `backtest_fn`, e.g. research.momentum")
    common.add_argument(
        "--data", help="a local CSV: wide (date,AAPL,MSFT), long (date,symbol,close), or one series"
    )
    common.add_argument(
        "--universe", help="name or id of a universe registered in the portal; narrows --data to it"
    )
    common.add_argument(
        "--symbol",
        help="one name out of a loaded universe; its closes become the return series sizing runs on",
    )
    common.add_argument(
        "--share-series",
        action="store_true",
        help="opt in to uploading full series on run traces (or set ALPHAENGINE_SHARE_SERIES=1)",
    )
    common.add_argument("--idea", help="attach the run or filter the log to a local hypothesis id")
    common.add_argument("--family", help="strategy or experiment family for cumulative trial counting")
    common.add_argument(
        "--json",
        dest="json_out",
        action="store_true",
        help="emit a versioned v1.0 JSON result envelope to stdout",
    )
    common.add_argument(
        "--stdin",
        dest="stdin_in",
        action="store_true",
        help="read an upstream JSON result envelope from standard input",
    )

    p = argparse.ArgumentParser(
        prog="alphaengine",
        parents=[common],
        description=(
            "Run validated research workflows. Computation runs locally; when you are signed in to "
            "quantOS, summary figures are reported to your account; full series are shared only if "
            "you opt in. Research tool, not investment advice."
        ),
    )
    sub = p.add_subparsers(dest="command")
    ini = sub.add_parser(
        "init",
        parents=[common],
        help="initialize a local .alphaengine/ research project in this directory",
    )
    ini.add_argument("directory", nargs="?", default=None, help="directory to initialize (default .)")
    ini.add_argument("--name", help="project name (defaults to directory name)")
    sub.add_parser("version", parents=[common], help="print the version")
    sub.add_parser("workflows", parents=[common], help="list what the server offers")
    sub.add_parser("demo", parents=[common], help="run the built-in example, offline")
    pr = sub.add_parser(
        "process", parents=[common], help="fit a named process to a series and stress it, offline"
    )
    pr.add_argument("dgp", nargs="?", default="gbm", help="ou, gbm, jump, or garch (default gbm)")
    rl = sub.add_parser("runs", parents=[common], help="your own week: what ran, and what it decided")
    rl.add_argument("--limit", type=int, default=25, help="how many to show (default 25)")
    lg_cmd = sub.add_parser(
        "log",
        parents=[common],
        help="list local project ledger runs, trial counts, and verdicts",
    )
    lg_cmd.add_argument("--since", help="filter to runs on or after YYYY-MM-DD")
    lg_cmd.add_argument("--verdict", help="filter by verdict")
    lg_cmd.add_argument("--limit", type=int, default=25, help="how many rows to show (default 25)")
    sh_cmd = sub.add_parser(
        "show",
        parents=[common],
        help="inspect one recorded project run, its hashes, trials, and figures",
    )
    sh_cmd.add_argument("run_ref", metavar="run", nargs="?", default="HEAD", help="run id, prefix, or HEAD")
    df_cmd = sub.add_parser(
        "diff",
        parents=[common],
        help="compare data, parameter, trial, and figure changes between two runs",
    )
    df_cmd.add_argument("run_a", metavar="run-a", help="first run id or prefix")
    df_cmd.add_argument("run_b", metavar="run-b", help="second run id or prefix")
    rr_cmd = sub.add_parser(
        "rerun",
        parents=[common],
        help="reproduce a recorded project run from its inputs and check figures",
    )
    rr_cmd.add_argument("run_ref", metavar="run", nargs="?", default="HEAD", help="run id, prefix, or HEAD")
    dt_cmd = sub.add_parser(
        "data",
        parents=[common],
        help="register, inspect, and trace local CSV/Parquet datasets",
    )
    dt_cmd.add_argument("action", nargs="?", default="list", help="add, list, show, or lineage")
    dt_cmd.add_argument("target", nargs="?", default=None, help="file path, dataset name, or run id")
    dt_cmd.add_argument("--name", help="name for the registered dataset")
    dt_cmd.add_argument(
        "--asof-col",
        dest="asof_col",
        help="as-of column name or cutoff date for look-ahead checks",
    )
    sub.add_parser("gaps", parents=[common], help="what your record says is unanswered")
    tn = sub.add_parser("tonight", parents=[common], help="what would run unattended, without running it")
    tn.add_argument("--budget", type=int, default=0, help="how many runs a night is worth")
    lo = sub.add_parser("logout", parents=[common], help="remove stored credentials")
    lo.add_argument(
        "--purge",
        action="store_true",
        help="remove ~/.config/alphaengine/credentials.json and clean up its config directory",
    )
    lg = sub.add_parser("login", parents=[common], help="sign in, or enter a model key")
    lg.add_argument("provider", nargs="?", default="quantos", help="quantos, anthropic, openai, ...")
    sub.add_parser("models", parents=[common], help="which model providers this machine can use")
    tr = sub.add_parser("trace", parents=[common], help="local model/run events for a run")
    tr.add_argument("run_id", nargs="?", help="run id; defaults to the latest local dump")
    d = sub.add_parser("commands", parents=[common], help="every command, grouped")
    d.add_argument("verb", nargs="?", help="expand one command in full")

    sl = sub.add_parser(
        "seal",
        parents=[common],
        help="write canonical content_sha256 into a study and verify on re-run",
    )
    sl.add_argument("study_path", metavar="study.json", help="path to study.json")

    rp = sub.add_parser(
        "report",
        parents=[common],
        help="send a saved study to quantOS (/api/me/runs/ingest or --retail) or export a PDF report",
    )
    rp.add_argument("study_path", nargs="?", metavar="study.json", help="path to study.json")
    rp.add_argument(
        "--retail",
        action="store_true",
        help="post to /api/retail/ledger/import instead of /api/me/runs/ingest",
    )

    r = sub.add_parser("run", parents=[common], help="run one workflow to completion")
    r.add_argument("workflow")
    r.add_argument("--label", help="what to call the artifact")
    r.add_argument("--thesis", help="name or id of a thesis on this account; attached to the run")
    r.add_argument("--input", action="append", help="workflow input as key=value (repeatable)")
    r.add_argument("--quiet", action="store_true", help="only the result")
    r.add_argument("--stream", action="store_true", help="print the answer after the citation guard")

    # THE QUESTIONS ARE THE COMMANDS. `run screen_universe` is the machine's
    # spelling; `alphaengine screen` is the person's. Each verb is the full
    # `run` command with the workflow pre-chosen — same flags, same behaviour,
    # one implementation — so the intuitive door and the explicit door cannot
    # drift.
    for verb, workflow in QUESTION_VERBS.items():
        q = sub.add_parser(verb, parents=[common], help=f"run {workflow}")
        q.add_argument("--label", help="what to call the artifact")
        q.add_argument("--thesis", help="name or id of a thesis on this account; attached to the run")
        q.add_argument("--input", action="append", help="workflow input as key=value (repeatable)")
        q.add_argument("--quiet", action="store_true", help="only the result")
        q.add_argument("--stream", action="store_true", help="print the answer after the citation guard")
        q.set_defaults(workflow=workflow)
    return p


def _process_series(data: Any) -> Any:
    """One numeric path from loaded data. Never prints the values."""
    from .core.process import as_series

    if isinstance(data, dict):
        for key in ("close", "spread", "series", "returns"):
            if key in data:
                arr = as_series(data[key])
                if arr.size:
                    return arr
        first = next(iter(data.values()), None)
        return as_series(first)
    return as_series(data)


def cmd_process(args: argparse.Namespace, data: Any = None) -> int:
    """Calibrate a named DGP and stress it. Offline, figures only."""
    from .core.process import dgp_stress

    dgp = str(getattr(args, "dgp", None) or "gbm").lower().strip()
    if dgp not in ("ou", "gbm", "jump", "garch"):
        say(red("process takes ou, gbm, jump, or garch."))
        return 2
    source = "loaded"
    if data is None and getattr(args, "stdin_in", False):
        from .envelope import read_stdin_envelope

        try:
            up_env = read_stdin_envelope()
            up_rets = (up_env.get("context") or {}).get("returns")
            if isinstance(up_rets, list) and up_rets:
                data = up_rets
        except Exception as exc:  # noqa: BLE001
            say(red(str(exc)))
            return 2
    if data is None:
        loaded, _bt = resolve_data(args)
        data = loaded
    if data is None:
        from . import demo

        data = demo.data
        source = "demo walk"
    series = _process_series(data)
    if getattr(series, "size", 0) < 3:
        say(red("process needs a numeric series (a close, a spread, or returns)."))
        return 2
    json_out = bool(getattr(args, "json_out", False))
    if json_out:
        figures = dgp_stress(series, dgp=dgp, n_paths=80, seed=7)
    else:
        with Working(f"fitting {dgp}", plain=f"fitting {dgp}"):
            figures = dgp_stress(series, dgp=dgp, n_paths=80, seed=7)
        say("")
        say(f"  {bold('process')}  {dim(DOT)}  {dgp}  {dim(DOT)}  {source}")
        say("")
    keys = (
        "kappa",
        "theta",
        "sigma",
        "half_life",
        "phi",
        "mean_reverting",
        "mu",
        "omega",
        "alpha",
        "beta",
        "persistence",
        "jump_lambda",
        "jump_mu",
        "jump_sigma",
        "n_jumps_named",
        "n_obs",
        "n_paths",
        "n_trials",
        "n_trials_source",
        "sharpe_q10",
        "sharpe_q50",
        "sharpe_q90",
        "frac_positive",
    )
    present = [k for k in keys if figures.get(k) is not None]
    if not json_out:
        width = max((len(k) for k in present), default=8)
        for k in present:
            say(f"    {dim(k.ljust(width))}  {figures[k]}")
        hist = figures.get("sharpe_hist") or {}
        counts = hist.get("counts") or []
        if counts:
            say("")
            say("    " + dim("sharpe hist") + "  " + sparkline([float(c) for c in counts], width=36))
    from .project import find_project

    proj = find_project()
    verdict_val = "positive" if float(figures.get("sharpe_q50") or 0.0) > 0 else "likely_noise"
    rec: dict[str, Any] | None = None
    if proj is not None:
        from .ledger import hash_dataset_any, record_run

        data_sha = None
        with contextlib_suppress():
            data_sha = proj.put_object(series.tolist())
        rec = record_run(
            proj,
            command="process",
            params={"dgp": dgp, "n_paths": 80, "seed": 7},
            dataset_hashes=hash_dataset_any(series),
            n_trials=int(figures.get("n_trials") or 80),
            n_trials_source=str(figures.get("n_trials_source") or "monte_carlo"),
            figures={k: figures[k] for k in present},
            verdict=verdict_val,
            idea_id=getattr(args, "idea", None),
            family=getattr(args, "family", None) or f"process:{dgp}",
            repro_spec={
                "kind": "process",
                "dgp": dgp,
                "n_paths": 80,
                "seed": 7,
                "data_object_sha256": data_sha,
            },
        )
        if not json_out:
            say("")
            say(
                f"    {dim('recorded')}  {rec['run_id']}  {dim(DOT)}  "
                f"run trials {rec['n_trials']}  {dim(DOT)}  cumulative trials {rec['cumulative_trials']}"
            )
    if json_out:
        from .envelope import build_envelope, emit_json, returns_from_panel_or_series
        from .ledger import hash_dataset_any

        nt_proc = int(figures.get("n_trials") or 80)
        rets_ctx, _ = returns_from_panel_or_series(series.tolist())
        emit_json(
            build_envelope(
                command="process",
                figures=rec["figures"] if rec else {k: figures[k] for k in present},
                verdict=verdict_val,
                run_id=str(rec["run_id"]) if rec else None,
                timestamp=str(rec["timestamp"]) if rec else None,
                idea_id=getattr(args, "idea", None),
                family=str(rec["family"]) if rec else f"process:{dgp}",
                dataset_hashes=hash_dataset_any(series),
                n_trials=nt_proc,
                n_trials_source=str(figures.get("n_trials_source") or "monte_carlo"),
                cumulative_trials=int(rec["cumulative_trials"]) if rec else nt_proc,
                project_trials=int(rec["project_trials"]) if rec else nt_proc,
                context={"returns": (rets_ctx or [])[:512]},
            )
        )
        return 0
    say("")
    say(
        "  "
        + _teal(
            "computation runs locally; when you are signed in to quantOS, summary figures are reported "
            "to your account; full series are shared only if you opt in."
        )
    )
    return 0


def contextlib_suppress() -> Any:
    import contextlib

    return contextlib.suppress(Exception)


def cmd_demo(args: argparse.Namespace) -> int:
    """The built-in example. No server, no account, no network, no repo.

    This exists because `pip install alphaengine` used to leave you with a CLI
    and nothing to point it at — the example lived in `examples/`, which the
    wheel does not carry, so the README's first instruction was `git clone`.
    Asking somebody to clone a repo to try a pip-installable tool is a first run
    most people do not finish.
    """
    from . import demo
    from .ledger import hash_dataset_any, record_run, record_trials, suppress_auto_sweep_record
    from .project import find_project
    from .sweep import sweep

    proj = find_project()
    if proj is None:
        return demo.run()

    with suppress_auto_sweep_record():
        rc = demo.run()
        res = sweep(demo.backtest_fn, demo.GRID, data=demo.data)
        v = res.verdict()
        rec = record_run(
            proj,
            command="demo",
            params={"grid": demo.GRID},
            dataset_hashes=hash_dataset_any(demo.data),
            n_trials=res.n_trials,
            n_trials_source="derived_from_grid",
            figures={
                "n_trials": v["n_trials"],
                "n_trials_source": v["n_trials_source"],
                "deflated_sharpe": v.get("deflated_sharpe"),
                "psr_vs_zero": v.get("psr_vs_zero"),
                "sr0_expected_max": v.get("sr0_expected_max"),
                "performance": v.get("performance", {}),
                "surface": res.surface(),
            },
            verdict=v.get("verdict"),
            idea_id=getattr(args, "idea", None),
            family=getattr(args, "family", None) or "demo",
            repro_spec={"kind": "demo"},
            returns_for_deflation=res.matrix[:, res.best.index].tolist(),
        )
        record_trials(
            proj,
            run_id=str(rec["run_id"]),
            trials=res.ledger(),
            idea_id=getattr(args, "idea", None),
            family=str(rec["family"]),
            timestamp=str(rec["timestamp"]),
        )
        say(
            f"  {dim('recorded')}  {rec['run_id']}  {dim(DOT)}  "
            f"run trials {rec['n_trials']}  {dim(DOT)}  cumulative trials {rec['cumulative_trials']}"
        )
    return rc


def cmd_models(_args: argparse.Namespace) -> int:
    """Which providers this machine can actually use, and which are one install away."""
    from .model import available_models, keys_without_sdk, provider_labels

    ready = available_models()
    stranded = keys_without_sdk()
    if ready:
        say("")
        say("  " + bold("usable"))
        for label, model in ready:
            say(f"    {green(ON)} {bold(label)}/{dim(model)}")
    else:
        say(dim("  no usable model on this machine"))
    if stranded:
        say(yellow("  key set, SDK missing"))
        for label in stranded:
            say(f"    {yellow(OFF)} {label}")
    say(dim("  providers: " + ", ".join(provider_labels())))
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    """Print the local jsonl dump. Portal POST is best-effort elsewhere."""
    from .events import data_dir

    run_id = getattr(args, "run_id", None)
    root = data_dir() / "runs"
    path = None
    if run_id:
        path = root / f"{run_id}.jsonl"
    elif root.exists():
        dumps = sorted(root.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        path = dumps[0] if dumps else None
    if path is None or not path.exists():
        say(dim("  no local run events yet"))
        return 0
    say(dim(f"  {path}"))
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        say(red(str(exc)))
        return 2
    for line in text.splitlines()[-50:]:
        say("  " + line)
    return 0


def cmd_logout(args: argparse.Namespace) -> int:
    """Delete the stored credentials. The file goes, not a flag inside it."""
    from .auth import clear_stored, config_path

    purge = bool(getattr(args, "purge", False))
    target = config_path()
    if clear_stored(purge=purge):
        action = "Purged." if purge else "Signed out."
        say(green(action) + dim(f"  removed {target}"))
    else:
        action = (
            "Purged — no credentials file was stored."
            if purge
            else "Not signed in — there was nothing stored."
        )
        say(dim(action))
    say(dim("Any key exported in your parent shell is untouched; unset it there."))
    return 0


def cmd_login(args: argparse.Namespace) -> int:
    """Sign in from a shell, without opening a session first."""
    which = getattr(args, "provider", None) or "quantos"
    return 0 if enter_key(which) else 2


def cmd_seal(args: argparse.Namespace) -> int:
    """Write canonical content_sha256 to a study, or verify if already sealed."""
    import json

    from .study import compute_content_sha256, load, save

    path = pathlib.Path(str(getattr(args, "study_path", "")))
    if not path.is_file():
        say(red(f"No study file at {path}."))
        return 2
    try:
        raw_disk = json.loads(path.read_text(encoding="utf-8"))
        study = load(path)
    except Exception as exc:  # noqa: BLE001
        say(red(f"Could not read {path}: {exc}"))
        return 2

    existing = raw_disk.get("content_sha256")
    computed = compute_content_sha256(study)
    if existing:
        if existing == computed:
            say(green("Verified.") + f"  {computed}  " + dim(str(path)))
            return 0
        say(
            red("Seal mismatch.")
            + f"  stored {existing[:16]}... != computed {computed[:16]}...  "
            + dim(str(path))
        )
        return 1

    study.content_sha256 = computed
    save(study, path)
    say(green("Sealed.") + f"  {computed}  " + dim(str(path)))
    return 0


def cmd_report_study(args: argparse.Namespace) -> int:
    """Report a saved study.json to quantOS, or generate a local PDF report to Downloads when omitted."""
    raw_study_path = getattr(args, "study_path", None)
    if not raw_study_path:
        from .agent.brief import study_for
        from .agent.sim import Trace

        session, _url = _session(getattr(args, "url", None), getattr(args, "key", None))
        data: Any = None
        try:
            data, _bt = resolve_data(args, session)
        except Exception:  # noqa: BLE001
            data = None
        if not isinstance(data, dict) or len(data) < 2:
            from .demo_universe import data as demo_data

            data = demo_data
        trace = Trace(
            question="portfolio construction, scenario stress, Monte Carlo simulation, and bakeoff report",
            panel=data,
        )
        study_dict = study_for(trace, deep=True)
        trace.study = study_dict
        trace.findings = str(study_dict.get("findings") or "")
        trace.answer = trace.findings
        out_path = trace.save()
        say(green("Reported.") + f"  {out_path}")
        return 0

    from .study import ReportError, load
    from .study.report import report as send_report

    path = pathlib.Path(str(raw_study_path))
    if not path.is_file():
        say(red(f"No study file at {path}."))
        return 2
    try:
        study = load(path)
    except Exception as exc:  # noqa: BLE001
        say(red(f"Could not read {path}: {exc}"))
        return 2

    if study.content_sha256 and not study.verify():
        say(red(f"Refusing to report {path}: content_sha256 seal does not match study content."))
        return 2

    retail = bool(getattr(args, "retail", False))
    source = str(getattr(args, "source", "AlphaEngine CLI") or "AlphaEngine CLI")
    family = str(getattr(args, "family", "alphaengine_study") or "alphaengine_study")
    try:
        res = send_report(
            study,
            api_key=getattr(args, "key", None),
            base_url=getattr(args, "url", None),
            retail=retail,
            source=source,
            family=family,
        )
    except ReportError as exc:
        msg = str(exc)
        say(red(msg))
        if "The key was rejected" in msg:
            from .auth import remove_key

            remove_key("QUANTOS_API_KEY")
            os.environ.pop("QUANTOS_API_KEY", None)
            if offer_key("quantos"):
                say(dim("  Key set. Run that again."))
        return 2

    target = "/api/retail/ledger/import" if retail else "/api/me/runs/ingest"
    rec_id = res.get("id") or res.get("study_id") or (res.get("imported") or {}).get("id") or ""
    tail = f"  ({rec_id})" if rec_id else ""
    say(green("Reported.") + f"  {study.label or path.name} {ARROW} {target}{tail}")
    return 0


def cmd_log(args: argparse.Namespace) -> int:
    """List runs recorded in the local `.alphaengine/` project ledger."""
    from .ledger import list_runs
    from .project import find_project

    proj = find_project()
    if proj is None:
        say(red("Not inside a local project. Run `alphaengine init` first."))
        return 2
    rows = list_runs(
        proj,
        idea_id=getattr(args, "idea", None),
        family=getattr(args, "family", None),
        since=getattr(args, "since", None),
        verdict=getattr(args, "verdict", None),
        limit=int(getattr(args, "limit", 25) or 25),
    )
    if not rows:
        say(dim("  no matching runs in the project ledger"))
        return 0
    say("")
    for r in rows:
        ts = str(r.get("timestamp") or "")[:19].replace("T", " ")
        rid = str(r.get("run_id") or "")
        cmd = str(r.get("command") or "")
        fam = str(r.get("family") or "")
        verd = str(r.get("verdict") or "unscored")
        nt = r.get("n_trials")
        ct = r.get("cumulative_trials")
        src = str(r.get("n_trials_source") or "")
        trials_txt = f"n={nt} ({src}) {DOT} cum={ct}" if nt is not None else f"n=not_recorded {DOT} cum={ct}"
        idea_tag = f" {DOT} idea:{r['idea_id']}" if r.get("idea_id") else ""
        say(
            f"  {bold(rid)}  {dim(ts)}  {cmd.ljust(16)}  {verd.ljust(12)}  "
            f"{dim(trials_txt)}  {dim('[' + fam + idea_tag + ']')}"
        )
    say("")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Inspect one recorded run from `.alphaengine/ledger/runs.jsonl`."""
    from .ledger import resolve_run
    from .project import find_project

    proj = find_project()
    if proj is None:
        say(red("Not inside a local project. Run `alphaengine init` first."))
        return 2
    ref = str(getattr(args, "run_ref", None) or "HEAD")
    try:
        r = resolve_run(proj, ref)
    except KeyError as exc:
        say(red(str(exc)))
        return 2

    say("")
    say(f"  {bold(str(r.get('run_id')))}  {dim(DOT)}  {r.get('command')}  {dim(DOT)}  {r.get('timestamp')}")
    say(f"    {dim('family')}             {r.get('family')}")
    if r.get("idea_id"):
        say(f"    {dim('idea_id')}            {r.get('idea_id')}")
    say(f"    {dim('params_hash')}        {r.get('params_hash')}")
    ds_hashes = ", ".join(str(h) for h in (r.get("dataset_hashes") or [])) or "none"
    say(f"    {dim('dataset_hashes')}     {ds_hashes}")
    say(f"    {dim('trials (run)')}       {r.get('n_trials')} ({r.get('n_trials_source')})")
    cum_t = r.get("cumulative_trials")
    proj_t = r.get("project_trials")
    say(f"    {dim('trials (cumulative)')} {cum_t} (project total: {proj_t})")
    say(f"    {dim('verdict')}            {r.get('verdict')}")
    say(f"    {dim('engine_version')}     {r.get('engine_version')}")
    figs = r.get("figures") or {}
    if isinstance(figs, dict) and figs:
        say("")
        say("  " + bold("figures"))
        for k, v in sorted(figs.items()):
            if isinstance(v, (int, float, str, bool)) or v is None:
                say(f"    {dim(str(k).ljust(28))}  {v}")
    say("")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    """Compare two runs from the project ledger."""
    from .ledger import diff_runs, resolve_run
    from .project import find_project

    proj = find_project()
    if proj is None:
        say(red("Not inside a local project. Run `alphaengine init` first."))
        return 2
    try:
        ra = resolve_run(proj, str(getattr(args, "run_a", "")))
        rb = resolve_run(proj, str(getattr(args, "run_b", "")))
    except KeyError as exc:
        say(red(str(exc)))
        return 2

    d = diff_runs(ra, rb)
    say("")
    say(f"  {bold('diff')}  {d['run_a']}  {ARROW}  {d['run_b']}")
    say(
        f"    {dim('data')}        "
        + (
            "changed " + f"{d['dataset_hashes_a']} {ARROW} {d['dataset_hashes_b']}"
            if d["data_changed"]
            else "unchanged"
        )
    )
    say(
        f"    {dim('parameters')}  "
        + ("changed" if d["params_changed"] else "unchanged")
        + f"  ({str(d['params_hash_a'])[:12]} {ARROW} {str(d['params_hash_b'])[:12]})"
    )
    for pk, pv in sorted((d.get("param_changes") or {}).items()):
        say(f"      {pk}: {pv['a']} {ARROW} {pv['b']}")
    say(
        f"    {dim('trials')}      run {d['n_trials_a']} {ARROW} {d['n_trials_b']}  {DOT}  "
        f"cumulative {d['cumulative_trials_a']} {ARROW} {d['cumulative_trials_b']}"
    )
    say(f"    {dim('verdict')}     {d['verdict_a']} {ARROW} {d['verdict_b']}")
    fchanges = d.get("figure_changes") or {}
    if fchanges:
        say("")
        say("  " + bold("figure changes"))
        for fk, fv in sorted(fchanges.items()):
            delta_str = f"  ({fv['delta']:+g})" if isinstance(fv.get("delta"), (int, float)) else ""
            say(f"    {dim(fk.ljust(28))}  {fv['a']} {ARROW} {fv['b']}{delta_str}")
    say("")
    return 0


def cmd_rerun(args: argparse.Namespace) -> int:
    """Re-execute a recorded project run from its stored inputs and check figures."""
    from .ledger import rerun_record, resolve_run
    from .project import find_project

    proj = find_project()
    if proj is None:
        say(red("Not inside a local project. Run `alphaengine init` first."))
        return 2
    ref = str(getattr(args, "run_ref", None) or "HEAD")
    try:
        r = resolve_run(proj, ref)
    except KeyError as exc:
        say(red(str(exc)))
        return 2
    try:
        matched, _repro_figs, mismatches = rerun_record(proj, r)
    except Exception as exc:  # noqa: BLE001
        say(red(f"Could not rerun {r.get('run_id')}: {exc}"))
        return 2

    if matched:
        say(green("Reproduced.") + f"  {r.get('run_id')} ({r.get('command')}) — all recorded figures match.")
        return 0

    say(red("Figure drift detected.") + f"  {r.get('run_id')} ({r.get('command')})")
    for k, m in sorted(mismatches.items()):
        say(f"    {k}: recorded {m['recorded']} != reproduced {m['reproduced']}")
    return 1


def cmd_data(args: argparse.Namespace) -> int:
    """Register, list, inspect, or trace local datasets in `.alphaengine/data/`."""
    from .datasets import add_dataset, dataset_lineage, get_dataset, list_datasets
    from .project import find_project

    proj = find_project()
    if proj is None:
        say(red("Not inside a local project. Run `alphaengine init` first."))
        return 2

    action = str(getattr(args, "action", None) or "list").lower().strip()
    target = getattr(args, "target", None)

    if action == "add":
        if not target:
            say(red("data add requires a file path: alphaengine data add <file> [--name NAME]"))
            return 2
        try:
            rec = add_dataset(
                proj,
                target,
                name=getattr(args, "name", None),
                asof_col=getattr(args, "asof_col", None),
            )
        except Exception as exc:  # noqa: BLE001
            say(red(str(exc)))
            return 2
        checks = rec.get("checks") or {}
        dups = (checks.get("duplicate_timestamps") or {}).get("count", 0)
        gaps = (checks.get("gaps") or {}).get("missing_cells", 0) + (checks.get("gaps") or {}).get(
            "calendar_gaps_gt_5d", 0
        )
        la = (checks.get("lookahead_hints") or {}).get("n_future_rows", 0)
        surv = (checks.get("survivorship_hints") or {}).get("n_ends_early", 0)
        say(
            green("Registered.")
            + f"  {bold(str(rec['name']))}  {dim(DOT)}  {rec['n_rows']} rows  {dim(DOT)}  "
            + f"{rec.get('start_date') or 'undated'}..{rec.get('end_date') or 'undated'}  "
            + dim(f"sha256:{str(rec['sha256'])[:16]}...")
        )
        say(
            f"  {dim('checks')}  duplicates={dups}  gaps={gaps}  "
            f"lookahead_rows={la}  early_ending_series={surv}"
        )
        return 0

    if action == "list":
        rows = list_datasets(proj)
        if not rows:
            say(dim("  no datasets registered yet"))
            return 0
        say("")
        for r in rows:
            span = f"{r.get('start_date') or '-'}..{r.get('end_date') or '-'}"
            say(
                f"  {bold(str(r.get('name')).ljust(18))}  {str(r.get('format')).ljust(8)}  "
                f"{str(r.get('n_rows')) + ' rows':<12}  {dim(span.ljust(24))}  "
                f"{dim(str(r.get('sha256'))[:16])}"
            )
        say("")
        return 0

    if action == "show":
        if not target:
            say(red("data show requires a dataset name: alphaengine data show <name>"))
            return 2
        try:
            r = get_dataset(proj, str(target))
        except KeyError as exc:
            say(red(str(exc)))
            return 2
        say("")
        say(f"  {bold(str(r.get('name')))}  {dim(DOT)}  {r.get('path')}  ({r.get('format')})")
        say(f"    {dim('sha256')}         {r.get('sha256')}")
        say(f"    {dim('n_rows')}         {r.get('n_rows')}")
        say(f"    {dim('date_range')}     {r.get('start_date')} .. {r.get('end_date')}")
        cols = r.get("columns") or []
        if cols:
            say("    " + dim("columns"))
            for c in cols:
                say(
                    f"      {str(c.get('name')).ljust(18)}  {c.get('dtype')}  (missing: {c.get('n_missing')})"
                )
        checks = r.get("checks") or {}
        say("    " + dim("hygiene checks"))
        for ck, cv in sorted(checks.items()):
            say(f"      {ck.ljust(22)}  {cv}")
        say("")
        return 0

    if action == "lineage":
        ref = str(target or "HEAD")
        try:
            lin = dataset_lineage(proj, ref)
        except KeyError as exc:
            say(red(str(exc)))
            return 2
        say("")
        say(f"  {bold('lineage')}  {lin['run_id']} ({lin['command']})")
        say(f"    {dim('dataset_hashes')}  {', '.join(lin['dataset_hashes']) or 'none'}")
        ds_list = lin.get("datasets") or []
        if ds_list:
            say("    " + dim("registered datasets"))
            for d in ds_list:
                say(f"      {bold(str(d.get('name')))}  {d.get('path')}  ({str(d.get('sha256'))[:16]}...)")
        rel = lin.get("related_runs") or []
        if rel:
            say("    " + dim("runs sharing dataset"))
            for rr in rel:
                say(f"      {rr.get('run_id')}  {rr.get('command')}  {rr.get('verdict')}")
        say("")
        return 0

    say(red(f"Unknown data action {action!r}. Use add, list, show, or lineage."))
    return 2


def main(argv: list[str] | None = None) -> int:
    # SIGN-IN IS RESTORED BEFORE ANYTHING READS A CREDENTIAL, and in exactly one
    # place. Doing it per-command is how `alphaengine run` ends up 401-ing for a
    # user who is signed in, because one dispatch path forgot.
    #
    # The environment always wins over the file — CI, a VPC deploy and a
    # colleague's machine all set the variable, and a stale file silently
    # overriding it is a bug that takes a day to find.
    from .auth import apply_stored
    from .model import NoModelConfigured

    apply_stored()

    args = build_parser().parse_args(argv)
    handlers: dict[str | None, Callable[[argparse.Namespace], int]] = {
        "init": cmd_init,
        "version": cmd_version,
        "workflows": cmd_workflows,
        "runs": cmd_runs,
        "log": cmd_log,
        "show": cmd_show,
        "diff": cmd_diff,
        "rerun": cmd_rerun,
        "data": cmd_data,
        "gaps": cmd_gaps,
        "tonight": cmd_tonight,
        "demo": cmd_demo,
        "process": cmd_process,
        "logout": cmd_logout,
        "login": cmd_login,
        "commands": cmd_commands,
        "models": cmd_models,
        "trace": cmd_trace,
        "seal": cmd_seal,
        "report": cmd_report_study,
        "run": cmd_run,
        None: cmd_session,  # bare `alphaengine` opens the full-screen session
    }
    for verb in QUESTION_VERBS:
        handlers[verb] = cmd_run
    handler = handlers[args.command]
    try:
        return handler(args)
    except KeyboardInterrupt:
        say("")
        return 130
    except NoModelConfigured as exc:
        # A CONFIGURATION PROBLEM IS NOT A CRASH and must never look like one.
        # This escaped as two full tracebacks with the one actionable line
        # buried underneath, because the guard sat around `build_think()` while
        # the failure came from a closure called three frames later. That is
        # fixed at the source in `model.build_think`; this is the backstop, so
        # no future path can put a traceback in front of a user for a missing
        # key or a missing install.
        say(yellow(str(exc)))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
