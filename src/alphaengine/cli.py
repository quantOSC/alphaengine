"""`alphaengine` — the terminal entry point.

    $ alphaengine                      # a session
    $ alphaengine workflows            # what the server offers
    $ alphaengine run validate_study --project research.momentum
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
import sys
from typing import Any

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


def say(*parts: str) -> None:
    print(*parts, flush=True)


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
        say(red(f"The server refused: {exc.detail}"))
        return 2

    if not rows:
        say("The server offers no workflows.")
        return 0

    for r in rows:
        # REPRODUCIBILITY IS STATED UP FRONT, not discovered when two runs of
        # the same question disagree. Same reasoning as a trial count that says
        # whether it was derived or asserted.
        repro = r.get("reproducible")
        tag = (
            green("reproducible")
            if repro
            else yellow(f"exploratory {DOT} two runs may differ")
            if repro is False
            else dim("")
        )
        say(f"  {bold(str(r.get('name')))}  {dim(str(r.get('version')))}  {tag}")
    return 0


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
            if not quiet:
                say(f"  {dim('server ' + ARROW)}  {bold(op)}")
            before = run.status
            run.step(step)
            n += 1

            still = any(p.get("step_id") == step.get("step_id") for p in run.permitted)
            if still and before == "open" and run.status == "open":
                key = str(step.get("step_id"))
                failures[key] = failures.get(key, 0) + 1
                if not quiet:
                    say(f"  {red('local  ' + DOT)}  {op} could not be executed here")
                if failures[key] >= 2:
                    run.status = "abandoned"
                    run.stopped = {"reason": "step_failed", "op": op}
                    return
            elif not quiet:
                say(f"  {dim('local  ' + DOT)}  done")

            if run.status != "open":
                return


def cmd_run(args: argparse.Namespace) -> int:
    from .client import Offline, ServerError

    try:
        data, backtest_fn = load_project(args.project)
    except ProjectError as exc:
        say(red(str(exc)))
        return 2

    session, url = _session(args.url, args.key)
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
        run = session.open(args.workflow, data=data, backtest_fn=backtest_fn, **inputs)
        _drive(run, quiet=args.quiet)
    except Offline:
        _explain_offline(url)
        return 2
    except ServerError as exc:
        say(red(f"The server refused: {exc.detail}"))
        return 2

    return _report(run)


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


HELP = """
  workflows            what the server offers, and which are reproducible
  run <name>           start one   (--project, --label, --input k=v)
  status               the current run
  help                 this
  quit                 leave
"""


def boot(url: str, *, project: str | None, data: Any, keyed: bool) -> None:
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
    say("  " + dim(EDGE) + "   " + dim("a plateau survives. a knife edge does not."))
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

    rows: list[tuple[str, str]] = [
        ("version", bold(__version__)),
        ("server", url if keyed else dim(url)),
        (
            "auth",
            green("key found") if keyed else yellow(f"none  {DOT}  set {_ENV_KEY}"),
        ),
        ("project", (bold(project) if project else dim("none"))),
        ("data", loaded),
    ]
    for line in _boxed(rows):
        say(line)

    say("")
    for cmd, what in (
        ("workflows", "what your workspace offers"),
        ("run <name>", "execute one, watching every step"),
        ("help", "everything else, and `quit` to leave"),
    ):
        say("  " + bold(cmd.ljust(12)) + dim(what))
    say("")
    # THE ONE LINE THAT IS NOT STATUS. It is here because it is the thing most
    # often forgotten and the thing that makes the rest make sense.
    say("  " + dim("The sequence runs on the server. The compute runs here, on your data."))
    say("")


def cmd_repl(args: argparse.Namespace) -> int:
    """A session. The point is that a run is watchable and repeatable without
    retyping a command line each time."""
    from .client import Offline, ServerError

    session, url = _session(args.url, args.key)
    data, backtest_fn = None, None
    if args.project:
        try:
            data, backtest_fn = load_project(args.project)
        except ProjectError as exc:
            say(red(str(exc)))
            return 2

    boot(url, project=args.project, data=data, keyed=bool(args.key or os.environ.get(_ENV_KEY)))

    last = None
    while True:
        try:
            line = input(bold("> ")).strip()
        except (EOFError, KeyboardInterrupt):
            say("")
            return 0
        if not line:
            continue
        verb, _, rest = line.partition(" ")
        rest = rest.strip()

        if verb in ("quit", "exit"):
            return 0
        if verb == "help":
            say(HELP)
            continue
        if verb == "workflows":
            cmd_workflows(args)
            continue
        if verb == "status":
            say(dim("no run yet") if last is None else f"{last.run_id}  {last.status}")
            continue
        if verb == "project":
            try:
                data, backtest_fn = load_project(rest)
                say(dim(f"project: {rest}"))
            except ProjectError as exc:
                say(red(str(exc)))
            continue
        if verb == "run":
            if not rest:
                say(red("run what? try `workflows`"))
                continue
            try:
                last = session.open(rest, data=data, backtest_fn=backtest_fn)
                _drive(last)
                _report(last)
            except Offline:
                _explain_offline(url)
            except ServerError as exc:
                say(red(f"The server refused: {exc.detail}"))
            continue

        say(dim(f"unknown command {verb!r} — `help`"))


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

    p = argparse.ArgumentParser(
        prog="alphaengine",
        parents=[common],
        description="Run validated research workflows. Your data stays on this machine.",
    )
    sub = p.add_subparsers(dest="command")
    sub.add_parser("version", parents=[common], help="print the version")
    sub.add_parser("workflows", parents=[common], help="list what the server offers")

    r = sub.add_parser("run", parents=[common], help="run one workflow to completion")
    r.add_argument("workflow")
    r.add_argument("--label", help="what to call the artifact")
    r.add_argument("--input", action="append", help="workflow input as key=value (repeatable)")
    r.add_argument("--quiet", action="store_true", help="only the result")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = {
        "version": cmd_version,
        "workflows": cmd_workflows,
        "run": cmd_run,
        None: cmd_repl,  # bare `alphaengine` opens a session
    }[args.command]
    try:
        return handler(args)
    except KeyboardInterrupt:
        say("")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
