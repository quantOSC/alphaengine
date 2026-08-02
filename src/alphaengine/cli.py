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
        refused(exc)
        if is_auth_error(exc) and offer_key():
            say(dim("  Key set. Run that again."))
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
        needs = r.get("requires") or []
        need_s = dim("  needs " + ", ".join(str(n) for n in needs)) if needs else dim("  needs nothing")
        say(f"  {bold(str(r.get('name')))}  {dim(str(r.get('version')))}  {tag}{need_s}")
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

    have = {"data": data is not None, "backtest_fn": backtest_fn is not None}
    missing = [r for r in (row.get("requires") or []) if not have.get(str(r), True)]
    if not missing:
        return None

    what = " and ".join(f"`{m}`" for m in missing)
    return (
        f"{name} needs {what} from your project module, and none was loaded.\n"
        f"  Point at a module of yours that defines them:\n"
        f"      alphaengine run {name} --project research.momentum\n"
        f"  Or try it on the built-in example first:\n"
        f"      alphaengine run {name} --project alphaengine.demo"
    )


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
        # ASK WHAT IT NEEDS BEFORE STARTING IT. One extra GET against a run that
        # would otherwise fail twice and abandon.
        gap = preflight(session.workflows(), args.workflow, data=data, backtest_fn=backtest_fn)
        if gap:
            say("")
            say(yellow(gap))
            return 2
        run = session.open(args.workflow, data=data, backtest_fn=backtest_fn, **inputs)
        _drive(run, quiet=args.quiet)
    except Offline:
        _explain_offline(url)
        return 2
    except ServerError as exc:
        refused(exc)
        if is_auth_error(exc) and offer_key():
            say(dim("  Key set. Run that again."))
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
  <anything else>      ASK IN PLAIN ENGLISH. Your model picks a workflow and
                       drives it, choosing each step from what the server
                       permits. Exploratory: two runs may differ.
                       Needs ANTHROPIC_API_KEY or OPENAI_API_KEY in this shell.

  run <name>           run one workflow exactly as written   (--label, --input k=v)
                       Scripted: the same data gives the same path.
  workflows            what the server offers, and which are reproducible
  project <module>     load `data` and `backtest_fn` from your module
  keys                 what is unlocked, and what would unlock the rest
  logout               remove stored credentials from this machine
  key <which>          enter one now (quantos | anthropic | openai), session only
  status               the current run
  help                 this
  quit                 leave
"""


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
    from .model import available_models

    models = available_models()
    on = "●" if _UNICODE_OK else "*"
    off = "○" if _UNICODE_OK else "-"

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


def _ask(session: Any, url: str, request: str, *, data: Any, backtest_fn: Any) -> Any:
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
        chosen = catalogue[driver.pick(options)]
    except AgentRefusal as exc:
        say(red(f"The model did not choose a workflow: {exc}"))
        return None

    name = str(chosen.get("name"))
    repro = chosen.get("reproducible")
    say(
        f"  {dim('agent  ' + DOT)}  chose {bold(name)}  "
        + (green("reproducible") if repro else yellow("not reproducible"))
    )

    try:
        run = session.open(name, data=data, backtest_fn=backtest_fn)
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
    # SAID AT THE END, WHERE IT TRAVELS WITH THE RESULT. A study whose path was
    # chosen by a model is a different object from one whose path was fixed in
    # advance, and the difference has to reach whoever reads the artifact.
    say(dim("  The model chose this path. Another run may take a different one."))
    return run


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
        if verb == "project":
            try:
                data, backtest_fn = load_project(rest)
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
        last = _ask(session, url, line, data=data, backtest_fn=backtest_fn) or last


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
    sub.add_parser("demo", parents=[common], help="run the built-in example, offline")
    sub.add_parser("logout", parents=[common], help="remove stored credentials")

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

    apply_stored()

    args = build_parser().parse_args(argv)
    handler = {
        "version": cmd_version,
        "workflows": cmd_workflows,
        "demo": cmd_demo,
        "logout": cmd_logout,
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
