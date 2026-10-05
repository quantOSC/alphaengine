"""One line of the session.

`run <workflow>` is scripted. Anything else is a sentence, and the model may
only choose among workflows and steps the server already permitted.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..cli import (
    QUESTION_VERBS,
    ProjectError,
    _apply_flags,
    _ask,
    _describe_source,
    _drive,
    _explain_offline,
    _extract_flags,
    _help_text,
    _is_workflow,
    _near_verb,
    _prior_line,
    _repl_gap,
    _report,
    _split_run,
    dim,
    load_project,
    match_thesis,
    pop_flag,
    preflight,
    project_grid,
    red,
    resolve_data,
    say,
    thesis_id_of,
    thesis_label,
    thesis_text,
    yellow,
)
from ..client import Offline, ServerError
from ..core.relations import KINDS as _CROSS

_OVERLAP = {"overlap", "check_overlap"}
_PORTFOLIO = {"portfolio", "allocate"}


@dataclass
class Desk:
    """What the session is holding. Mutated by `submit`."""

    session: Any
    url: str
    args: Any
    data: Any = None
    backtest_fn: Any = None
    loaded: str | None = None
    last: Any = None
    think: Callable[[str], str] | None = None
    keyed: bool = False
    pending_secret: str | None = None
    pending_url: str | None = None
    pending_provider: bool = False
    held_key: str | None = None
    account_dirty: bool = False
    pinned_model: str | None = None
    book: Any = None
    thesis: dict[str, Any] | None = None
    thesis_declined: bool = False
    matrix: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)
    portfolio: dict[str, Any] | None = None


def submit(desk: Desk, line: str) -> bool:
    """Handle one line. True means the session should close."""
    line = line.strip()
    if not line:
        return False

    line, inline_flags = _extract_flags(line)
    # `--symbol` on check_overlap must not collapse the universe before the
    # book is built. Every other workflow still wants that collapse.
    peeled_symbol = ""
    tokens = [tok.lower().replace("-", "_") for tok in line.split()]
    overlap_line = bool(tokens) and (
        tokens[0] in _OVERLAP or (len(tokens) > 1 and tokens[0] == "run" and tokens[1] in _OVERLAP)
    )
    if overlap_line and "--symbol" in inline_flags:
        kept: list[str] = []
        i = 0
        while i < len(inline_flags):
            if inline_flags[i] == "--symbol" and i + 1 < len(inline_flags):
                peeled_symbol = inline_flags[i + 1]
                i += 2
                continue
            kept.append(inline_flags[i])
            i += 1
        inline_flags = kept
    thesis_query, inline_flags = pop_flag(inline_flags, "--thesis")
    if thesis_query and not _pin_thesis(desk, thesis_query):
        return False
    if inline_flags:
        try:
            desk.data, desk.backtest_fn = _apply_flags(
                inline_flags, desk.session, desk.data, desk.backtest_fn
            )
            desk.loaded = _describe_source(inline_flags) or desk.loaded
        except (ProjectError, ValueError) as exc:
            say(red(str(exc)))
            return False
        except Exception as exc:  # noqa: BLE001 — a server refusal, said plainly
            say(red(str(exc)))
            return False
    if not line.strip():
        return False

    if line.split()[:1] == ["alphaengine"]:
        line = line.partition(" ")[2].strip()
        if not line:
            return False
    if line.startswith("/"):
        line = line[1:].lstrip()

    if line.lower() == "enter model key":
        desk.pending_secret = "MODEL_KEY"
        say(dim("Paste the model key. It stays hidden. The provider is read from the key."))
        return False

    verb, _, rest = line.partition(" ")
    if verb in QUESTION_VERBS and not rest:
        verb, rest = "run", QUESTION_VERBS[verb]
    rest = rest.strip()

    if verb in ("quit", "exit"):
        return True
    if verb == "help":
        say(_help_text())
        return False
    if verb == "demo":
        from ..cli import cmd_demo

        cmd_demo(desk.args if desk.args is not None else _ns(desk))
        return False
    if verb == "process":
        from ..cli import cmd_process

        cmd_process(
            _ns(desk, dgp=rest or "gbm", data=None, project=None, universe=None),
            data=desk.data,
        )
        return False
    if verb == "models":
        from ..cli import cmd_models

        cmd_models(_ns(desk))
        return False
    if verb == "model":
        _pin_model(desk, rest)
        return False
    if verb == "trace":
        from ..cli import cmd_trace

        cmd_trace(_ns(desk, run_id=rest or getattr(desk.last, "run_id", None)))
        return False
    if verb == "book":
        _book(desk, rest)
        return False
    if verb in ("thesis", "theses"):
        _thesis(desk, rest)
        return False
    if verb in ("commands", "?"):
        from ..cli import cmd_commands

        cmd_commands(_ns(desk, verb=rest))
        return False
    if verb == "workflows":
        from ..cli import cmd_workflows

        cmd_workflows(_ns(desk))
        return False
    if verb == "runs":
        from ..cli import cmd_runs

        cmd_runs(_ns(desk, limit=25))
        return False
    if verb == "gaps":
        from ..cli import cmd_gaps

        cmd_gaps(_ns(desk))
        return False
    if verb == "tonight":
        from ..cli import cmd_tonight

        cmd_tonight(_ns(desk, budget=0))
        return False
    if verb == "logout":
        _logout(desk)
        return False
    if verb in ("key", "keys", "login"):
        _key(desk, rest)
        return False
    if verb == "status":
        if desk.thesis:
            say(dim(f"thesis {thesis_label(desk.thesis)}"))
        if desk.last is None:
            say(dim("no run yet"))
        else:
            say(f"{desk.last.run_id}  {desk.last.status}")
        return False
    if verb == "load":
        _load_spec(desk, rest)
        return False
    if verb in ("universe", "data"):
        _load_named(desk, verb, rest)
        return False
    if verb == "project":
        try:
            desk.data, desk.backtest_fn = load_project(rest)
            desk.loaded = rest
            say(dim(f"project: {rest}"))
            _remember(desk, _loaded_note(desk))
        except ProjectError as exc:
            say(red(str(exc)))
        return False

    if verb in _PORTFOLIO:
        _portfolio(desk, rest)
        return False

    if verb in _CROSS or verb in _OVERLAP:
        if verb in _OVERLAP:
            _overlap(desk, "")
        else:
            _cross(desk, _CROSS[verb])
        return False

    if verb == "run" and rest:
        name, flags = _split_run(rest)
        key = name.lower().replace("-", "_")
        if key in _PORTFOLIO:
            _portfolio(desk, rest)
            return False
        if key in _CROSS:
            _cross(desk, _CROSS[key])
            return False
        if key in _OVERLAP:
            tail = " ".join(flags)
            if peeled_symbol and "--symbol" not in flags:
                tail = f"--symbol {peeled_symbol} {tail}".strip()
            _overlap(desk, tail)
            return False
        if name and " " not in name and _is_workflow(desk.session, name):
            _scripted(desk, name, flags)
            return False

    if verb == "run" and not rest:
        say(red("run what? try `workflows`, or just describe what you want."))
        return False

    if " " not in line:
        near = _near_verb(line)
        if near:
            say(dim(f"  No command {line!r}. Did you mean `{near}`?"))
            say(dim("  Type it again as a sentence if you meant it as a question."))
            return False

    _sentence(desk, line)
    return False


def _ns(desk: Desk, **extra: Any) -> Any:
    import argparse

    base = desk.args
    return argparse.Namespace(
        url=getattr(base, "url", None),
        key=getattr(base, "key", None),
        **extra,
    )


def _logout(desk: Desk) -> None:
    import os

    from ..auth import MANAGED
    from ..cli import _session, cmd_logout

    cmd_logout(_ns(desk))
    for name in MANAGED:
        os.environ.pop(name, None)
    desk.session, desk.url = _session(getattr(desk.args, "url", None), getattr(desk.args, "key", None))
    desk.keyed = False
    desk.pending_secret = None
    desk.pending_url = None
    desk.pending_provider = False
    desk.held_key = None
    desk.pinned_model = None
    desk.thesis = None
    desk.thesis_declined = False
    if str(desk.loaded or "").startswith("universe:"):
        desk.data = None
        desk.backtest_fn = None
        desk.loaded = None
    desk.account_dirty = True


def _key(desk: Desk, rest: str) -> None:
    from ..cli import _KEY_PROMPTS

    which = (rest.split() or [""])[0].lower()
    if not which:
        if not desk.keyed:
            which = "quantos"
        else:
            say(dim("Already signed in. Type `enter model key` and paste it."))
            return
    if which not in _KEY_PROMPTS:
        say(dim(f"Unknown {which!r}. Try: {', '.join(_KEY_PROMPTS)}"))
        return
    env, what = _KEY_PROMPTS[which]
    desk.pending_secret = env
    say(dim(f"Paste the {which} key. {what} It stays hidden, then this machine keeps it."))


def env_for_model_key(value: str) -> str | None:
    """Which env var this key belongs in, when the prefix says so."""
    text = value.strip()
    if text.startswith("sk-ant-"):
        return "ANTHROPIC_API_KEY"
    if text.startswith("sk-or-"):
        return "OPENROUTER_API_KEY"
    if text.startswith("gsk_"):
        return "GROQ_API_KEY"
    if text.startswith("AIza"):
        return "GEMINI_API_KEY"
    if text.startswith("sk-"):
        return "OPENAI_API_KEY"
    return None


def _pin_model(desk: Desk, rest: str) -> None:
    import os

    from ..model import available_models, pin_model

    if not rest:
        pinned = os.environ.get("ALPHAENGINE_PROVIDER") or os.environ.get("ALPHAENGINE_MODEL")
        say(dim("  pinned: " + (pinned or "none (first usable)")))
        for label, model in available_models():
            say(f"    {label}/{model}")
        return
    provider, model_name = pin_model(rest)
    desk.pinned_model = f"{provider or ''}/{model_name or rest}".strip("/")
    say(dim(f"model {desk.pinned_model}"))


def _book(desk: Desk, rest: str) -> None:
    from ..book import Book

    if desk.book is None:
        desk.book = Book()
    book = desk.book
    if rest in ("status", "monitor"):
        say(str(book.monitor()))
        return
    if rest:
        if isinstance(desk.data, dict) and rest in desk.data:
            book.add(rest, desk.data[rest])
            say(dim(f"sleeve {rest}"))
        else:
            say(yellow("load data first, then `book <name>`"))
        return
    names = book.names
    say(dim("empty book") if not names else "sleeves: " + ", ".join(names))


def _load_spec(desk: Desk, rest: str) -> None:
    from ..cli import _open_spec, classify_load

    if not rest:
        say(red("load what? a file, a project module, or a universe name."))
        return
    try:
        kind = classify_load(rest)
        desk.data, desk.backtest_fn, desk.loaded = _open_spec(
            kind, rest, desk.session, desk.data, desk.backtest_fn
        )
        say(dim(f"loaded {desk.loaded}"))
        _remember(desk, _loaded_note(desk))
    except (ProjectError, ValueError) as exc:
        say(red(str(exc)))
    except Exception as exc:  # noqa: BLE001
        say(red(str(exc)))


def _load_named(desk: Desk, verb: str, rest: str) -> None:
    import argparse

    if not rest:
        say(red(f"{verb} what? try `{verb} <name>`."))
        return
    try:
        ns = argparse.Namespace(
            project=None,
            data=rest if verb == "data" else None,
            universe=rest if verb == "universe" else None,
            symbol=None,
        )
        loaded, _ = resolve_data(ns, desk.session)
        if loaded is not None:
            desk.data = loaded
            desk.loaded = f"{verb}:{rest}"
            say(dim(f"loaded {desk.loaded}"))
            _remember(desk, _loaded_note(desk))
    except (ProjectError, ValueError) as exc:
        say(red(str(exc)))
    except Exception as exc:  # noqa: BLE001
        say(red(str(exc)))


def _thesis_rows(desk: Desk) -> list[dict[str, Any]] | None:
    try:
        rows = desk.session.theses()
    except Exception as exc:  # noqa: BLE001 — offline or unsigned, said plainly
        say(red(str(exc)))
        return None
    return [r for r in rows if isinstance(r, dict)]


def _pin_thesis(desk: Desk, query: str) -> bool:
    """Pin one thesis. False leaves the previous pin alone."""
    rows = _thesis_rows(desk)
    if rows is None:
        return False
    chosen, candidates = match_thesis(rows, query)
    if chosen is None:
        if candidates:
            say(yellow("Which thesis? " + ", ".join(thesis_label(r) for r in candidates)))
        else:
            say(yellow(f"No thesis named {query!r}."))
        return False
    if not thesis_id_of(chosen):
        say(yellow(f"{thesis_label(chosen)} has no id, so a run cannot be attached to it."))
        return False
    desk.thesis = chosen
    desk.thesis_declined = False
    say(dim(f"thesis {thesis_label(chosen)}"))
    text = thesis_text(chosen)
    if text:
        say(dim("  " + (text if len(text) <= 240 else text[:237] + "...")))
    return True


def _thesis(desk: Desk, rest: str) -> None:
    if rest.lower() in ("clear", "none", "off"):
        desk.thesis = None
        desk.thesis_declined = True
        say(dim("thesis cleared"))
        return
    if rest:
        _pin_thesis(desk, rest)
        return
    rows = _thesis_rows(desk)
    if rows is None:
        return
    if not rows:
        say(dim("No theses on this account."))
        return
    selected = thesis_id_of(desk.thesis) if desk.thesis else ""
    for row in rows:
        mark = "  selected" if selected and thesis_id_of(row) == selected else ""
        say(f"  {thesis_label(row)}{mark}")
    if desk.thesis is None and len(rows) > 1:
        say(dim("Type thesis <name> to use one."))


def _cross(desk: Desk, kind: str) -> None:
    """Correlation, covariance, or cointegration on the names already loaded."""
    from ..core.relations import as_closes, cointegration, correlation, covariance, summary_lines

    if not isinstance(desk.data, dict) or not desk.data:
        say(yellow("Load a universe first. These run on the names you already have."))
        return
    if kind == "cointegration":
        n = len(as_closes(desk.data))
        pairs = n * (n - 1) // 2
        if pairs > 20:
            say(dim(f"  screening {pairs} pairs across {n} names"))
    try:
        if kind == "correlation":
            result = correlation(desk.data)
        elif kind == "covariance":
            result = covariance(desk.data)
        else:
            result = cointegration(desk.data)
    except ValueError as exc:
        say(red(str(exc)))
        return
    desk.matrix = result
    for line in summary_lines(result):
        say(line)
    from ..context import result_note

    _remember(desk, result_note(result))
    from ..cli import _publish_panel

    _publish_panel(desk.session, result, thesis_id=_thesis_id(desk))


def _portfolio(desk: Desk, line: str) -> None:
    """Weights for the loaded names. Session notes travel with the result."""
    from ..context import result_note
    from ..core.portfolio import build_portfolio

    if not isinstance(desk.data, dict) or not desk.data:
        say(yellow("Load a universe first. A portfolio is built from the names you already have."))
        return
    low = line.lower()
    method = "risk_parity" if ("risk parity" in low or "equal risk" in low) else "hrp"
    measured: list[str] | None = None
    if desk.matrix and desk.matrix.get("kind") in ("correlation", "covariance", "cointegration"):
        measured = [str(name) for name in (desk.matrix.get("names") or [])]
    try:
        result = build_portfolio(desk.data, method=method, notes=desk.notes, names=measured or None)
    except ValueError as exc:
        say(red(str(exc)))
        return
    desk.portfolio = result
    desk.matrix = result
    say(f"  {result['method']}  {result['n_assets']} names  Ledoit-Wolf covariance")
    if result.get("n_skipped"):
        say(dim(f"  {result['n_skipped']} names skipped (too short to share the history)"))
    rows = list(result.get("rows") or [])
    for row in rows[:8]:
        say(f"  {row['name']}  {row['weight']}")
    if len(rows) > 8:
        say(dim(f"  {len(rows) - 8} more on the right"))
    if result.get("session"):
        say(dim("  session already recorded: " + ", ".join(result["session"])))
    _remember(desk, result_note(result))
    from ..cli import _publish_panel

    _publish_panel(desk.session, result, thesis_id=_thesis_id(desk))


def _overlap(desk: Desk, rest: str) -> None:
    """One name against the equal-weight of the other loaded names.

    `check_overlap` is not a correlation matrix. It asks whether a single
    candidate is the book again. With a universe loaded, the book is the
    other names. With no name named, the useful run is the full matrix.
    """
    from ..cli import _looks_like_universe
    from ..core.relations import overlap_against_the_rest

    symbol = ""
    flags = rest.split()
    if "--symbol" in flags:
        i = flags.index("--symbol")
        if i + 1 < len(flags):
            symbol = flags[i + 1]
    elif flags and not flags[0].startswith("--"):
        symbol = flags[0]

    say("  check_overlap measures one name against the equal-weight of the others.")
    if not symbol or not _looks_like_universe(desk.data):
        if _looks_like_universe(desk.data):
            say(dim("  No name was given, so this is every name against every other name."))
            _cross(desk, "correlation")
            return
        say(yellow("Load a universe, or name one series and a book."))
        return
    try:
        prepared = overlap_against_the_rest(desk.data, symbol)
    except ValueError as exc:
        say(red(str(exc)))
        return
    others = len(prepared["book_names"])
    say(dim(f"  {prepared['symbol']} against the equal-weight of {others} other names."))
    held = desk.data
    desk.data = {
        "returns": prepared["returns"],
        "book_returns": prepared["book_returns"],
        "symbol": prepared["symbol"],
        "book_names": prepared["book_names"],
    }
    try:
        _scripted(desk, "check_overlap", [])
    finally:
        desk.data = held


def _scripted(desk: Desk, name: str, flags: list[str]) -> None:
    desk.matrix = None
    thesis_query, flags = pop_flag(flags, "--thesis")
    if thesis_query and not _pin_thesis(desk, thesis_query):
        return
    try:
        if flags:
            desk.data, desk.backtest_fn = _apply_flags(flags, desk.session, desk.data, desk.backtest_fn)
            desk.loaded = _describe_source(flags) or desk.loaded
        gap = preflight(desk.session.workflows(), name, data=desk.data, backtest_fn=desk.backtest_fn)
        if gap:
            say(yellow(_repl_gap(gap, name)))
            return
        grid = project_grid()
        pinned = thesis_id_of(desk.thesis) if desk.thesis else ""
        if desk.thesis:
            say(dim(f"  thesis  {thesis_label(desk.thesis)}"))
        desk.last = desk.session.open(
            name,
            data=desk.data,
            backtest_fn=desk.backtest_fn,
            thesis_id=pinned or None,
            **({"grid": grid} if grid else {}),
        )
        _drive(desk.last)
        _report(desk.last)
        if desk.last is not None:
            _remember(desk, _prior_line(desk.last))
    except Offline:
        _explain_offline(desk.url)
    except ServerError as exc:
        from ..cli import refused

        refused(exc)
    except (ProjectError, ValueError) as exc:
        say(red(str(exc)))


def _sentence(desk: Desk, line: str) -> None:
    from ..cli import _apply_flags as apply_flags
    from ..cli import _stored_universes, _universe_named_in
    from ..context import consider

    if _local_request(desk, line):
        return

    if desk.data is None:
        named = _universe_named_in(line, desk.session)
        if not named:
            named, _ready = _stored_universes(desk.session)
        if named:
            try:
                desk.data, desk.backtest_fn = apply_flags(
                    ["--universe", named], desk.session, desk.data, desk.backtest_fn
                )
                desk.loaded = f"universe:{named}"
                _remember(desk, _loaded_note(desk))
            except Exception as exc:  # noqa: BLE001
                say(dim(f"  ({named} is registered and could not be loaded: {exc})"))
        if _local_request(desk, line):
            return

    # A generic question reads the session first. A reply of "workflow" means
    # the notes do not answer it, and the catalogue path below still runs.
    kind, text = consider(line, desk.notes, _thinker(desk))
    if kind == "answer":
        say(text)
        return
    if kind == "refuse":
        say(yellow(text))
        return

    pinned = desk.thesis
    run = _ask(
        desk.session,
        desk.url,
        line,
        data=desk.data,
        backtest_fn=desk.backtest_fn,
        think=desk.think,
        thesis_id=thesis_id_of(pinned) or None if pinned else None,
        thesis_name=thesis_label(pinned) if pinned else None,
        thesis_statement=thesis_text(pinned) if pinned else None,
        prior=_session_prior(desk),
    )
    if run is not None:
        desk.last = run
        _remember(desk, _prior_line(run))


def _local_request(desk: Desk, line: str) -> bool:
    """Run a calculation when the line names one and the data is already loaded.

    A question that does not name the calculation ("what was the strongest
    pair") falls through and is answered from the session notes.
    """
    from ..cli import _symbol_in
    from ..core.relations import as_closes

    if not isinstance(desk.data, dict) or not as_closes(desk.data):
        return False
    low = line.lower()
    if "portfolio" in low or "allocate" in low or "risk parity" in low:
        _portfolio(desk, line)
        return True
    if "cointegrat" in low:
        _cross(desk, "cointegration")
        return True
    if "covariance" in low:
        _cross(desk, "covariance")
        return True
    if "correlation" in low or "correlate" in low:
        _cross(desk, "correlation")
        return True
    if "overlap" in low:
        _overlap(desk, _symbol_in(line, desk.data) or "")
        return True
    return False


def _thesis_id(desk: Desk) -> str | None:
    if not desk.thesis:
        return None
    found = thesis_id_of(desk.thesis)
    return found or None


def _remember(desk: Desk, text: str) -> None:
    from ..context import remember

    desk.notes = remember(desk.notes, text)


def _loaded_note(desk: Desk) -> str:
    from ..context import loaded_note

    return loaded_note(desk.loaded or "data", desk.data)


def _session_prior(desk: Desk) -> str | None:
    """Every note, plus the last run if it is not already one of them."""
    parts = list(desk.notes)
    if desk.last is not None:
        line = _prior_line(desk.last)
        if line not in parts:
            parts.append(line)
    if not parts:
        return None
    return "\n".join(parts)


def _thinker(desk: Desk) -> Callable[[str], str] | None:
    """The session's model, built once. No key means the catalogue path explains it."""
    if desk.think is not None:
        return desk.think
    from ..model import NoModelConfigured, build_think

    try:
        think, _label = build_think()
    except NoModelConfigured:
        return None
    desk.think = think
    return think
