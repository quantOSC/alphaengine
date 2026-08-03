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
import importlib
import os
import pathlib
import sys
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from .commands import Command

__all__ = ["main"]

_ENV_KEY = "QUANTOS_API_KEY"
_ENV_URL = "QUANTOS_API_URL"
DEFAULT_BASE_URL = "https://alpha-backend-production-51df.up.railway.app"


# ── output ─────────────────────────────────────────────────────────────────
#
# ANSI only when the stream is a TTY. Piping `alphaengine run` into a file or a
# CI log must produce plain text, not escape codes — a run record with colour
# bytes in it is a run record somebody has to clean up before reading.


def _tty() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


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


def say(*parts: str) -> None:
    print(*parts, flush=True)


#: Spinner frames. Braille where the stream takes it, ASCII where it does not —
#: the same degradation `DOT` and `ARROW` get, for the same reason: a Windows
#: cp1252 console rendered the pretty version as mojibake.
_SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏" if _UNICODE_OK else "|/-\\"


class Working:
    """An in-place status line for work that blocks, with the elapsed time.

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
        self._painted = 0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started

    def __enter__(self) -> Working:
        self._started = time.monotonic()
        if not _tty():
            return self
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()
        return self

    def _animate(self) -> None:
        i = 0
        # A first frame immediately, then every 100ms. Waiting out the first
        # interval before drawing anything makes a fast step flash a blank line.
        while True:
            text = f"  {_SPIN[i % len(_SPIN)]} {self.label}  {self.elapsed:.1f}s"
            plain_len = len(f"  {_SPIN[0]} {self.plain}  {self.elapsed:.1f}s")
            sys.stdout.write("\r" + text)
            sys.stdout.flush()
            self._painted = max(self._painted, plain_len)
            if self._stop.wait(0.1):
                return
            i += 1

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            # Erased with spaces rather than an ANSI erase-line: this runs on
            # consoles old enough that colour is the only escape sequence worth
            # assuming, and a stray `[2K` printed literally is worse than a
            # slightly wider blank.
            sys.stdout.write("\r" + " " * (self._painted + 2) + "\r")
            sys.stdout.flush()


def cyan(t: str) -> str:
    return _c("36", t)


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


def _boxed(rows: list[tuple[str, str]], width: int = 62) -> list[str]:
    """A key/value box. Unicode rules when the stream takes them, ASCII when not."""
    if _UNICODE_OK:
        tl, tr, bl, br, h, v = "╭", "╮", "╰", "╯", "─", "│"
    else:
        tl, tr, bl, br, h, v = "+", "+", "+", "+", "-", "|"
    inner = width - 2
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
    ell = "…" if _UNICODE_OK else "..."
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


def load_project(spec: str | None) -> tuple[Any, Any]:
    """Import the caller's module and take `data` / `backtest_fn` off it.

    The current directory goes on `sys.path` first, because the module being
    imported is the user's own project and not something installed. That is the
    normal shape of "run this against my repo".
    """
    if not spec:
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
    if data is None and backtest_fn is None:
        raise ProjectError(
            f"{spec} exposes neither `data` nor `backtest_fn`.\n"
            f"  A project module is an ordinary module with your prices in `data`\n"
            f"  and, if the workflow sweeps, your simulator in `backtest_fn`."
        )
    return data, backtest_fn


# ── session ────────────────────────────────────────────────────────────────


def _session(base_url: str | None, api_key: str | None) -> tuple[Any, str]:
    """Imported lazily so `import alphaengine` still touches no networking code.

    `test_smoke.py` asserts that property, and it is worth more than the two
    lines this costs: the offline half of the package is what a researcher
    installs without asking anyone's permission.
    """
    from .client import connect

    url = base_url or os.environ.get(_ENV_URL) or DEFAULT_BASE_URL
    key = api_key or os.environ.get(_ENV_KEY) or ""
    return connect(url, api_key=key or None), url


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

    # WHAT EACH ONE ANSWERS, in the reader's words. The server publishes a name,
    # a version and an input contract -- correctly, since anything more would be
    # workflow knowledge crossing the wire -- but a name is not a choice. These
    # sentences live here, on the client, for exactly that reason: they describe
    # what a reader is picking between and disclose nothing about how it works.
    ANSWERS = {
        "screen_universe": "what in my universe is worth a look",
        "validate_study": "does this result hold up once you count the tries",
        "size_position": "how much of it should I hold",
        "monitor_sleeve": "has anything crossed a line",
    }

    width = max(len(str(r.get("name"))) for r in rows)
    say("")
    for r in rows:
        name = str(r.get("name"))
        needs = ", ".join(str(n) for n in (r.get("requires") or [])) or "nothing"
        # REPRODUCIBILITY IS STATED UP FRONT, not discovered when two runs of
        # the same question disagree. Same reasoning as a trial count that says
        # whether it was derived or asserted.
        repro = green("reproducible") if r.get("reproducible") else yellow("may differ")
        say(f"  {bold(name.ljust(width))}   {ANSWERS.get(name, '')}")
        say(f"  {' ' * width}   {dim('needs ' + needs)}  {DOT}  {repro}  {dim(str(r.get('version')))}")
    say("")
    say(dim("  alphaengine run <name> --data prices.csv   ") + dim(f"{DOT}  `commands run` for the flags"))
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
    }
    missing = [r for r in (row.get("requires") or []) if not have.get(str(r), True)]
    if not missing:
        return None

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
        "`data` is a return series: a sequence of per-period returns, or a mapping with them under `returns`."
    ),
    "backtest_fn": "`backtest_fn` is your own simulator; we orchestrate and measure, you simulate.",
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
        from .loaders import load_csv

        loaded = load_csv(args.data)
        # A project module that also defined `data` keeps it: the explicit module
        # is the more specific statement, and silently replacing it would make
        # two flags fight where the user can see neither winning.
        data = data if data is not None else loaded

    if getattr(args, "universe", None):
        data = _narrow_to_universe(session, args.universe, data)

    return data, backtest_fn


def _narrow_to_universe(session: Any, wanted: str, data: Any) -> Any:
    """Filter loaded prices to a universe the portal holds.

    THE SPLIT THAT MAKES THIS SAFE: the portal returns SYMBOLS, this machine
    holds the prices, and the two meet here rather than on the server. A
    universe definition is parameters -- names plus how to source them -- and
    that is why fetching one moves no market data.
    """
    from .loaders import DataShapeError, coverage

    if session is None:
        raise DataShapeError("--universe needs a workflow server; none is configured.")

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
        n_obs = max((len(v) for v in prices.values() if isinstance(v, list)), default=0)
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
        shown = ", ".join(missing[:8]) + ("…" if len(missing) > 8 else "")
        say(yellow(f"  {len(missing)} of {len(symbols)} names are not in your file: {shown}"))
    say(dim(f"  universe {wanted} {DOT} {len(kept)} of {len(symbols)} names loaded"))
    return kept


def _resolve_sleeve(session: Any, wanted: str | None) -> str | None:
    """A sleeve name or id -> the workspace id a run is opened against.

    Resolved by NAME because that is what a person types. A pod calls it
    "Systematic Macro", not a uuid, and requiring the uuid is how a correct
    mechanism ends up never receiving input — which is what happened here: no
    caller ever sent `workspace_id`, so the risk budget resolved to nothing and
    every sleeve's P&L reported zero.

    A miss NAMES what is available rather than failing bare, and an account with
    no pod is told that plainly instead of being shown an empty list.
    """
    if not wanted or session is None:
        return None
    from .loaders import DataShapeError

    try:
        rows = session.sleeves()
    except Exception as exc:  # noqa: BLE001 - offline or refused, said plainly
        raise DataShapeError(f"--sleeve needs the portal, and it could not be reached: {exc}") from exc

    if not rows:
        raise DataShapeError(
            "your account is not in a pod, so it has no sleeves. Drop --sleeve "
            "and the run works exactly the same; a sleeve only adds the budget "
            "it is bound by and the book it rolls into."
        )
    # THE KEY IS `id`, NOT `workspace_id`. The server's sleeve serializer returns
    # `{id, pod_id, name, stage, benchmark, initial_capital, pm_user_id, ...}` —
    # the workspace IS the sleeve, so its primary key is `id`, and the name
    # `workspace_id` only appears on the routes that TAKE one. Reading the
    # parameter name off the route and assuming it names the response field is
    # the same mistake that made `--universe` read a top-level `symbols` key the
    # server has never sent. Pinned by a contract test in the platform repo that
    # imports this function and the real serializer together.
    match = next(
        (s for s in rows if wanted in (str(s.get("id")), str(s.get("name")))),
        None,
    )
    if match is None:
        known = ", ".join(str(s.get("name")) for s in rows) or "none"
        raise DataShapeError(f"no sleeve called {wanted!r}. Yours: {known}")
    return str(match.get("id"))


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
    say(dim("  It is on your account, dated and versioned. Tomorrow's run diffs against it."))


def _drive(run: Any, *, quiet: bool = False) -> None:
    """Execute the run, narrating each step.

    Wraps `Run.step` rather than calling `drive()` so there is something to
    watch. The loop itself — what is permitted, in what order, what stops it —
    is still entirely the server's.
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
            # THE STEP IS WATCHED WHILE IT RUNS, not announced and then silent.
            # `compute.*` executes in-process against the caller's own frames, so
            # this is the one place in the product where a long wait is entirely
            # local and entirely invisible.
            with Working(f"{dim('running ' + ARROW)}  {bold(op)}", plain=f"running {ARROW}  {op}") as w:
                run.step(step)
            n += 1
            took = dim(f"{w.elapsed:.1f}s")

            still = any(p.get("step_id") == step.get("step_id") for p in run.permitted)
            if still and before == "open" and run.status == "open":
                key = str(step.get("step_id"))
                failures[key] = failures.get(key, 0) + 1
                if not quiet:
                    say(f"  {red('local  ' + DOT)}  {op} could not be executed here  {took}")
                if failures[key] >= 2:
                    run.status = "abandoned"
                    run.stopped = {"reason": "step_failed", "op": op}
                    return
            elif not quiet:
                say(f"  {dim('server ' + ARROW)}  {bold(op)}  {took}")

            if run.status != "open":
                return


def cmd_run(args: argparse.Namespace) -> int:
    from .client import Offline, ServerError

    session, url = _session(args.url, args.key)
    try:
        data, backtest_fn = resolve_data(args, session)
        workspace_id = _resolve_sleeve(session, getattr(args, "sleeve", None))
    except (ProjectError, ValueError) as exc:
        say(red(str(exc)))
        return 2

    inputs: dict[str, Any] = {}
    for pair in args.input or []:
        if "=" not in pair:
            say(red(f"--input expects key=value, got {pair!r}"))
            return 2
        k, v = pair.split("=", 1)
        inputs[k] = v
    if args.label:
        inputs["label"] = args.label

    try:
        say(f"{bold(args.workflow)} {dim(DOT + ' ' + url)}")
        # ASK WHAT IT NEEDS BEFORE STARTING IT. One extra GET against a run that
        # would otherwise fail twice and abandon.
        gap = preflight(session.workflows(), args.workflow, data=data, backtest_fn=backtest_fn)
        if gap:
            say("")
            say(yellow(gap))
            return 2
        run = session.open(
            args.workflow,
            data=data,
            backtest_fn=backtest_fn,
            workspace_id=workspace_id,
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
    return code


def _report(run: Any) -> int:
    """What happened, and an exit code that means it.

    A STOP IS NOT A FAILURE and exits 0. "This did not clear the bar" is the
    system working exactly as intended, and a non-zero exit would make every CI
    pipeline treat an honest refusal as a broken build — which is precisely the
    pressure that gets honesty controls disabled.
    """
    say("")
    if run.status == "closed":
        say(green("Closed.") + " " + dim(str((run.artifact or {}).get("workflow", ""))))
        art = run.artifact or {}
        for k, v in art.items():
            if k != "workflow" and not isinstance(v, (dict, list)):
                say(f"  {k}: {bold(str(v))}")
        return 0
    if run.status == "stopped":
        stop = run.stopped or {}
        say(yellow("Stopped.") + " " + str(stop.get("reason", "")))
        say(dim("  A stop is a result. The run did what it was built to do."))
        return 0
    if run.status == "abandoned":
        stop = run.stopped or {}
        say(red("Abandoned.") + f" {stop.get('op')} could not be executed by this build.")
        say(dim("  Supply a handler for it, or upgrade alphaengine."))
        return 1
    say(dim(f"Run left {run.status}. `alphaengine` can resume it: {run.run_id}"))
    return 1


NEWLINE = chr(10)


def _extract_flags(line: str) -> tuple[str, list[str]]:
    """Pull `--data x` / `--universe y` / `--project z` out of any line.

    Returns (prose, flags). The prose is what the agent sees, so
    `screen my book --universe sp100` asks about the book and loads sp100,
    rather than asking a model to interpret a flag as part of a question.

    Only the three data flags. Anything else stays in the prose: a sentence
    containing "--" is more likely to be a sentence than a mistyped flag, and
    stripping tokens a user meant to say is worse than passing one through.
    """
    parts = line.split()
    prose: list[str] = []
    flags: list[str] = []
    i = 0
    while i < len(parts):
        tok = parts[i]
        if tok in ("--data", "--universe", "--project") and i + 1 < len(parts):
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


def _apply_flags(flags: list[str], session: Any, data: Any, backtest_fn: Any) -> tuple[Any, Any]:
    """Load whatever `--data` / `--universe` / `--project` name, mid-session.

    THE SAME FLAGS THE SHELL TAKES, because a message that prints
    `--universe sp100` must mean the same thing wherever it is read. Anything
    else is a tool teaching two dialects of itself.
    """
    ns = argparse.Namespace(project=None, data=None, universe=None)
    i = 0
    unknown: list[str] = []
    while i < len(flags):
        f = flags[i]
        value = flags[i + 1] if i + 1 < len(flags) else None
        if f in ("--data", "--universe", "--project") and value:
            setattr(ns, f[2:], value)
            i += 2
            continue
        unknown.append(f)
        i += 1
    if unknown:
        raise ValueError(
            f"I do not know {' '.join(unknown)} in a session. "
            "Here `run` takes --data, --universe and --project."
        )
    loaded, fn = resolve_data(ns, session)
    return (loaded if loaded is not None else data), (fn or backtest_fn)


def _describe_source(flags: list[str]) -> str | None:
    for key in ("--universe", "--data", "--project"):
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
    head = gap.splitlines()[0]
    return (
        f"  {head}"
        + NEWLINE
        + NEWLINE
        + "  From here, load it first:"
        + NEWLINE
        + "      universe <name>            a universe you registered in the portal"
        + NEWLINE
        + "      data <file.csv>            a local file"
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
    """The short list, rendered from the one declaration.

    This was a hand-maintained string and it went stale the way every third copy
    of a list does: `--data` and `--universe` shipped and never appeared here,
    `demo` existed and was not mentioned. See `commands.py` for why there is now
    one source and four renderings.
    """
    from .commands import COMMANDS

    width = max(len(_signature(c)) for c in COMMANDS if c.scope in ("both", "repl"))
    lines = [""]
    for c in COMMANDS:
        if c.scope not in ("both", "repl"):
            continue
        lines.append(f"  {bold(_signature(c).ljust(width))}   {dim(c.purpose)}")
    lines += ["", "  " + dim("`commands` for the full directory, `commands <verb>` for one.")]
    return "\n".join(lines)


def _signature(c: Command) -> str:
    return f"{c.verb} {c.args}".strip()


def status_line(url: str, *, data: Any, project: str | None, keyed: bool) -> str:
    """One dim line above the prompt: what is loaded, and what can run.

    THE QUESTION THIS ANSWERS is the one asked most often inside a session and
    the one the boot screen answers once and then scrolls away: is my data in?
    Without it a user types `keys` or `status` purely to re-read state that was
    printed two minutes ago, and a run that silently had no data looked
    identical to one that did -- which is exactly how "the S&P 500 is a handful
    of names" happened.

    Deliberately ONE LINE and dim. It sits above every prompt, so anything
    louder becomes wallpaper and stops being read at all.
    """
    from .model import available_models

    bits: list[str] = []

    if data is None:
        bits.append(dim("no data"))
    else:
        try:
            n = len(data)
        except TypeError:
            n = None
        label = project or "loaded"
        bits.append(green(f"{label}" + (f" {DOT} {n} series" if n is not None else "")))

    bits.append(green("signed in") if keyed else dim("not signed in"))

    models = available_models()
    bits.append(dim(models[0][0]) if models else dim("no model"))

    return "  " + dim(f" {DOT} ").join(bits)


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
        say("  " + bold(title))
        width = max(len(_signature(c)) for c in rows)
        for c in rows:
            where = "" if c.scope == "both" else dim(f"  ({c.scope})")
            say(f"    {_signature(c).ljust(width)}   {dim(c.purpose)}{where}")
        say("")

    # THE DOORS GET THEIR OWN BLOCK. They are flags rather than verbs, and they
    # are the single thing people are most often hunting for when they open a
    # directory at all — "how do I give it my data" is not answerable from a
    # list of verbs.
    say("  " + bold("Getting your data in"))
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

    out = ["  " + dim("Where you are")]
    for live, name, does, note in rungs:
        mark = green(on) if live else dim(off)
        label = bold(name.ljust(14)) if live else dim(name.ljust(14))
        does_s = does.ljust(44) if live else dim(does.ljust(44))
        out.append(f"  {mark} {label}{does_s}{note}")

    # ONE ACTIONABLE LINE, and only when something is locked. The rungs
    # themselves stay in plain language — a menu that recites environment
    # variables reads like a manual — but a reader who cannot do the thing they
    # came for still needs to be told exactly how, without going to look it up.
    if not keyed:
        out += ["", "  " + dim("Type ") + bold("key quantos") + dim(f" to sign in, or export {_ENV_KEY}.")]
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
            + bold("key anthropic")
            + dim(" (or ")
            + bold("key openai")
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


def boot(url: str, *, project: str | None, data: Any, keyed: bool, session: Any = None) -> None:
    """The opening screen. State first, decoration second.

    Suppressed entirely when stdout is not a TTY: a CI log wants a transcript.
    """
    from ._version import __version__

    if not _tty():
        say(f"alphaengine {__version__}  {DOT}  {url}")
        return

    # The mark, the name, and what the mark means — three lines, and the third
    # is doing teaching rather than decoration.
    say("")
    say("  " + cyan(PLATEAU) + "   " + bold("alphaengine"))
    say("  " + dim(EDGE) + "   " + dim("Find out what survives."))
    say("")

    # Data is described by SHAPE, never printed. A boot screen that echoes the
    # first rows of somebody's price series into their scrollback — where it
    # will sit until the buffer rolls, and into any recording of the session —
    # is a boundary violation for the sake of a nicer banner.
    if data is None:
        loaded = dim("none") + dim(f"  {DOT}  pass --project to load yours")
    else:
        try:
            n = len(data)  # dict of series, DataFrame, or sequence
        except TypeError:
            n = None
        loaded = green("ready") + dim(f"  {DOT}  {n} series" if n is not None else "")

    # SIGNED IN IS A FACT THE USER SHOULD SEE, not infer from things working.
    # It also disambiguates the two ways a key can be present: one survives a
    # new terminal and one does not, and "why does it work here but not there"
    # is exactly the confusion that costs an afternoon.
    from .auth import load_stored

    stored = load_stored()
    if not keyed:
        signed = dim("no")
    elif "QUANTOS_API_KEY" in stored and not _shell_set("QUANTOS_API_KEY", stored):
        signed = green("yes") + dim(f"  {DOT}  stored on this machine")
    else:
        signed = green("yes") + dim(f"  {DOT}  from your shell")

    rows: list[tuple[str, str]] = [
        ("version", bold(__version__)),
        ("signed in", signed),
        ("project", (bold(project) if project else dim("none"))),
        ("data", loaded),
    ]
    for line in _boxed(rows):
        say(line)

    say("")
    for line in ladder_lines(keyed=keyed):
        say(line)

    # ONLY WHEN IT CHANGES THE ANSWER: signed in, and nothing already loaded
    # from disk. A run that knows its data does not need a directory of the
    # account's, and the call is not worth making to print something ignorable.
    if keyed and data is None:
        lines = account_data_lines(session)
        if lines:
            say("")
            for line in lines:
                say(line)

    say("")
    # THE ONE LINE THAT IS NOT STATUS. It is here because it is the thing most
    # often forgotten and the thing that makes the rest make sense.
    say("  " + dim("Your data stays on this machine. Only the findings ever travel."))
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
    say(yellow("  Not authenticated.") + dim("  Your portal login does not reach this process."))
    say(dim("  A browser session is a cookie; this needs a key. They are different"))
    say(dim("  credentials on purpose."))
    say("")
    say("  " + dim("Get one:") + f"  the portal {ARROW} Data {ARROW} API keys {ARROW} New key")
    say("  " + dim("Then:") + "     " + bold("key quantos") + dim(f"   (or export {_ENV_KEY})"))
    say("")


def offer_key(which: str = "quantos") -> bool:
    """Ask whether to enter a key now. True if one was entered.

    Only ever called on an interactive terminal — a prompt in a CI log is a
    hang, so callers check `_tty()` first.
    """
    if not _tty():
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


def enter_key(which: str) -> bool:
    """Prompt for a key and hold it FOR THIS SESSION ONLY.

    ── WHY THIS DOES NOT PERSIST ANYTHING ─────────────────────────────────────
    #
    The convenient version writes to a config file, and then a credential that
    can spend the user's money lives in plaintext on disk, survives the session,
    and gets copied with the project directory. We are not in the business of
    holding customer keys — that is the whole reason `AgentDriver` takes a
    callable and has no key field — and a dotfile would give that up for the
    sake of saving one `export`.

    So this sets the variable in THIS PROCESS. Close the terminal and it is
    gone. For a key you want every time, `export` it in your shell profile,
    which is a thing your shell already manages properly.

    Read with `getpass` so it does not echo and does not land in scrollback.
    """
    import getpass

    if which == "quantos":
        env, what = _ENV_KEY, "QuantOS key (ae_live_…) — created in the portal, on Data"
    elif which == "anthropic":
        env, what = "ANTHROPIC_API_KEY", "Anthropic key (sk-ant-…)"
    elif which == "openai":
        env, what = "OPENAI_API_KEY", "OpenAI key (sk-…)"
    else:
        say(red(f"unknown key {which!r} — try: quantos, anthropic, openai"))
        return False

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


def _ask(
    session: Any,
    url: str,
    request: str,
    *,
    data: Any,
    backtest_fn: Any,
    workspace_id: str | None = None,
) -> Any:
    """Natural language → a workflow chosen and driven by the user's own model.

    THE PART THAT IS EASY TO GET WRONG: the model picks the WORKFLOW from the
    server's catalogue and then picks each STEP from what the server permits. It
    never composes a sequence. Everything it is allowed to do is a choice among
    options somebody else produced, which is why this can be non-deterministic
    without being unsafe.
    """
    from .client import AgentDriver, AgentRefusal, Offline, ServerError
    from .model import NoModelConfigured, build_think

    try:
        think, label = build_think()
    except NoModelConfigured as exc:
        say(yellow(str(exc)))
        return None

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

    say(f"  {dim('agent  ' + DOT)}  {dim(label)}")

    # Choosing the workflow is the same shape as choosing a step: an index into
    # a list the server produced. Reusing `AgentDriver.pick` rather than writing
    # a second selector keeps one place where a model's answer is validated.
    driver = AgentDriver(
        think,
        goal=request,
        on_thought=lambda why: say(f"  {dim('agent  ' + DOT)}  {dim(why)}"),
    )
    options = [{"op": w.get("name", "?"), "params": {"agency": w.get("agency")}} for w in catalogue]
    try:
        # The single longest silence in the product: an inference call over the
        # user's own key, with no streaming and nothing to report until it
        # returns. Watched, because "is it thinking or is it hung" was
        # unanswerable from the screen.
        with Working(dim("reading the catalogue"), plain="reading the catalogue"):
            index = driver.pick(options)
        chosen = catalogue[index]
    except AgentRefusal as exc:
        say(red(f"The model did not choose a workflow: {exc}"))
        return None

    name = str(chosen.get("name"))

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
        return None

    repro = chosen.get("reproducible")
    say(
        f"  {dim('agent  ' + DOT)}  chose {bold(name)}  "
        + (green("reproducible") if repro else yellow("not reproducible"))
    )

    try:
        run = session.open(name, data=data, backtest_fn=backtest_fn, workspace_id=workspace_id)
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

    _report(run)
    if run.status == "closed":
        _publish_signals(session, run, workspace_id=workspace_id, label=None)
    _answer(request, run, think)
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


def _answer(request: str, run: Any, think: Callable[[str], str]) -> None:
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

    try:
        with Working(dim("composing the answer"), plain="composing the answer"):
            answer = synthesize(request, run, write=write)
    except UncitedFigure as exc:
        # NAMED, NOT SWALLOWED. This is the guard working, and a user who sees
        # nothing cannot tell it from a model that had nothing to say.
        say(yellow(f"  The answer was refused: {exc}"))
        return
    except Exception:  # noqa: BLE001
        say(dim("  No answer could be composed from this run."))
        return

    say("")
    for line in _wrap(answer.text):
        say(f"  {line}")
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


def cmd_repl(args: argparse.Namespace) -> int:
    """A session. The point is that a run is watchable and repeatable without
    retyping a command line each time."""
    from .client import Offline, ServerError

    session, url = _session(args.url, args.key)
    try:
        data, backtest_fn = resolve_data(args, session)
        workspace_id = _resolve_sleeve(session, getattr(args, "sleeve", None))
    except (ProjectError, ValueError) as exc:
        say(red(str(exc)))
        return 2

    boot(
        url,
        project=args.project,
        data=data,
        keyed=bool(args.key or os.environ.get(_ENV_KEY)),
        session=session,
    )

    # Named on the status line so "is my data in, and from where" is answerable
    # without running a command. `--universe` and `--data` are recorded too,
    # because "loaded" without a source is the half-answer that let a run with
    # no data look like a run with data.
    loaded_project = args.project or (
        f"universe:{args.universe}"
        if getattr(args, "universe", None)
        else (f"file:{pathlib.Path(args.data).name}" if getattr(args, "data", None) else None)
    )
    last = None
    while True:
        try:
            # Reprinted each turn rather than once at boot: a `project` or `key`
            # mid-session changes it, and a status line that can be stale is
            # worse than none.
            say(
                status_line(
                    url, data=data, project=loaded_project, keyed=bool(os.environ.get(_ENV_KEY) or args.key)
                )
            )
            line = input(bold("> ")).strip()
        except (EOFError, KeyboardInterrupt):
            say("")
            return 0
        if not line:
            continue
        # FLAGS FIRST, WHATEVER THE LINE IS. A command and a sentence are not
        # alternatives here: `screen my book --universe sp100` is both, and the
        # prompt used to force a choice between them.
        line, inline_flags = _extract_flags(line)
        if inline_flags:
            try:
                data, backtest_fn = _apply_flags(inline_flags, session, data, backtest_fn)
                loaded_project = _describe_source(inline_flags) or loaded_project
            except (ProjectError, ValueError) as exc:
                say(red(str(exc)))
                continue
            except Exception as exc:  # noqa: BLE001 - a server refusal, said plainly
                say(red(str(exc)))
                continue
            if not line.strip():
                continue  # the flags WERE the instruction

        # `alphaengine demo` at this prompt is somebody following the tool's own
        # instructions: every message it prints shows the shell form, because
        # that is the form that works in a shell. Stripping the binary name here
        # costs one line and removes a whole class of "I typed what it told me
        # to" and got a research question back.
        if line.split()[:1] == ["alphaengine"]:
            line = line.partition(" ")[2].strip()
            if not line:
                continue

        verb, _, rest = line.partition(" ")
        rest = rest.strip()

        if verb in ("quit", "exit"):
            return 0
        if verb == "help":
            say(_help_text())
            continue
        if verb == "demo":
            # A SHELL SUBCOMMAND ONLY, until now. So the single thing every
            # refusal recommends to a stuck user -- "see it work on the built-in
            # example first" -- was the one thing this prompt could not do.
            cmd_demo(args)
            continue
        if verb in ("commands", "?"):
            cmd_commands(argparse.Namespace(verb=rest))
            continue
        if verb == "workflows":
            cmd_workflows(args)
            continue
        if verb == "status":
            say(dim("no run yet") if last is None else f"{last.run_id}  {last.status}")
            continue
        if verb == "logout":
            cmd_logout(args)
            for name in ("QUANTOS_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
                os.environ.pop(name, None)
            session, url = _session(args.url, args.key)
            continue
        if verb in ("key", "keys"):
            if rest:
                if enter_key(rest):
                    # The session was built with the old key, so rebuild it or
                    # the newly-entered credential would not be used until
                    # restart — which looks exactly like the key not working.
                    session, url = _session(args.url, args.key)
            else:
                for ln in ladder_lines(keyed=bool(os.environ.get(_ENV_KEY) or args.key)):
                    say(ln)
                say("")
                say(dim("  `key quantos` · `key anthropic` · `key openai` to enter one now"))
            continue
        if verb in ("universe", "data"):
            if not rest:
                say(red(f"{verb} what? try `{verb} <name>`."))
                continue
            try:
                ns = argparse.Namespace(
                    project=None,
                    data=rest if verb == "data" else None,
                    universe=rest if verb == "universe" else None,
                )
                loaded, _ = resolve_data(ns, session)
                if loaded is not None:
                    data = loaded
                    loaded_project = f"{verb}:{rest}"
            except (ProjectError, ValueError) as exc:
                say(red(str(exc)))
            except Exception as exc:  # noqa: BLE001 - a server refusal, said plainly
                say(red(str(exc)))
            continue

        if verb == "project":
            try:
                data, backtest_fn = load_project(rest)
                loaded_project = rest
                say(dim(f"project: {rest}"))
            except ProjectError as exc:
                say(red(str(exc)))
            continue
        # A COMMAND WORD AT THE START OF A SENTENCE IS STILL A SENTENCE.
        #
        # `run a query into the S&P 500 and see which stocks are outperforming`
        # was parsed as `run` + a workflow named "a query into the S&P 500 …",
        # and the server answered "no workflow named …". The most natural way to
        # phrase an agentic request begins with a verb, so the one parse rule
        # that mattered was throwing every good request away.
        #
        # `run` means the scripted path ONLY when what follows is a single token
        # naming a real workflow. Anything else is a request, which is what the
        # user obviously meant.
        if verb == "run" and rest:
            name, flags = _split_run(rest)
            if name and _is_workflow(session, name):
                # FLAGS WORK IN HERE NOW. They are the same flags the shell
                # takes, because a message that says "--universe sp100" has to
                # mean the same thing wherever it is read -- and the preflight
                # message says exactly that.
                try:
                    if flags:
                        data, backtest_fn = _apply_flags(flags, session, data, backtest_fn)
                        loaded_project = _describe_source(flags) or loaded_project
                except (ProjectError, ValueError) as exc:
                    say(red(str(exc)))
                    continue

                gap = preflight(session.workflows(), name, data=data, backtest_fn=backtest_fn)
                if gap:
                    say(yellow(_repl_gap(gap, name)))
                    continue
                rest = name
            else:
                rest = rest if not flags else name

        if verb == "run" and rest and (" " not in rest) and _is_workflow(session, rest):
            try:
                last = session.open(rest, data=data, backtest_fn=backtest_fn)
                _drive(last)
                _report(last)
            except Offline:
                _explain_offline(url)
            except ServerError as exc:
                refused(exc)
                # A session holds its credential, so a key entered now does
                # nothing until the session is rebuilt — which is exactly the
                # bug that makes "I entered my key and it still says denied".
                if is_auth_error(exc) and offer_key():
                    session, url = _session(args.url, args.key)
                    say(dim("  Try that again."))
            continue

        if verb == "run" and not rest:
            say(red("run what? try `workflows`, or just describe what you want."))
            continue

        # A TYPO IS NOT A RESEARCH QUESTION. Falling through to the agent is
        # the right default, and it means one mistyped verb costs a model call
        # and returns something unrelated to what was asked. A single-word input
        # that is nearly a command is almost certainly that command.
        if " " not in line:
            near = _near_verb(line)
            if near:
                say(dim(f"  No command {line!r}. Did you mean `{near}`?"))
                say(dim("  Type it again as a sentence if you meant it as a question."))
                continue

        # ── ANYTHING ELSE IS A REQUEST, NOT A TYPO ─────────────────────────
        #
        # This used to answer "unknown command", which is the wrong default for
        # a research tool: the interesting input is a sentence about what you
        # are trying to find out, and the commands above are the shortcuts.
        # Falling through to the agent makes the prompt behave the way a quant
        # would expect after using any modern coding harness.
        #
        # THE SPLIT MATTERS AND IS VISIBLE. `run <workflow>` is SCRIPTED — the
        # workflow owns the sequence, two runs over the same data give the same
        # path, and the artifact is reproducible. A sentence is EXPLORATORY — a
        # model chooses each step from what the server permits, and two runs can
        # differ. Both are legitimate; conflating them is not, so the run says
        # which one it was.
        # A QUESTION MAY NAME ITS OWN DATA. "screen my sp100 universe" is a
        # complete instruction: the workflow is inferable and so is the data.
        # Requiring the user to load it first, in a separate command, is the
        # fork this whole change exists to remove.
        if data is None:
            named = _universe_named_in(line, session)
            if named:
                try:
                    data, backtest_fn = _apply_flags(["--universe", named], session, data, backtest_fn)
                    loaded_project = f"universe:{named}"
                except Exception as exc:  # noqa: BLE001
                    say(dim(f"  ({named} is registered and could not be loaded: {exc})"))

        last = (
            _ask(
                session,
                url,
                line,
                data=data,
                backtest_fn=backtest_fn,
                workspace_id=workspace_id,
            )
            or last
        )


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
        "--sleeve",
        help="name or id of a sleeve this run belongs to; binds it to that sleeve's risk budget",
    )

    p = argparse.ArgumentParser(
        prog="alphaengine",
        parents=[common],
        description="Run validated research workflows. Your data stays on this machine.",
    )
    sub = p.add_subparsers(dest="command")
    sub.add_parser("version", parents=[common], help="print the version")
    sub.add_parser("workflows", parents=[common], help="list what the server offers")
    sub.add_parser("demo", parents=[common], help="run the built-in example, offline")
    sub.add_parser("logout", parents=[common], help="remove stored credentials")
    d = sub.add_parser("commands", parents=[common], help="every command, grouped")
    d.add_argument("verb", nargs="?", help="expand one command in full")

    r = sub.add_parser("run", parents=[common], help="run one workflow to completion")
    r.add_argument("workflow")
    r.add_argument("--label", help="what to call the artifact")
    r.add_argument("--input", action="append", help="workflow input as key=value (repeatable)")
    r.add_argument("--quiet", action="store_true", help="only the result")
    return p


def cmd_demo(_args: argparse.Namespace) -> int:
    """The built-in example. No server, no account, no network, no repo.

    This exists because `pip install alphaengine` used to leave you with a CLI
    and nothing to point it at — the example lived in `examples/`, which the
    wheel does not carry, so the README's first instruction was `git clone`.
    Asking somebody to clone a repo to try a pip-installable tool is a first run
    most people do not finish.
    """
    from . import demo

    return demo.run()


def cmd_logout(_args: argparse.Namespace) -> int:
    """Delete the stored credentials. The file goes, not a flag inside it."""
    from .auth import clear_stored, config_path

    if clear_stored():
        say(green("Signed out.") + dim(f"  removed {config_path()}"))
    else:
        say(dim("Not signed in — there was nothing stored."))
    say(dim("Any key exported in your shell is untouched; unset it there."))
    return 0


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
    handler = {
        "version": cmd_version,
        "workflows": cmd_workflows,
        "demo": cmd_demo,
        "logout": cmd_logout,
        "commands": cmd_commands,
        "run": cmd_run,
        None: cmd_repl,  # bare `alphaengine` opens a session
    }[args.command]
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
