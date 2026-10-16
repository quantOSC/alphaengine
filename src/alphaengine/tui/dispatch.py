"""One line of the session.

A sentence is a stack of tool calls. The model chooses the tools. The math
runs locally. `run <workflow>` still opens a portal workflow by name, and
the workflow tool does the same thing from inside a sentence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..cli import (
    QUESTION_VERBS,
    ProjectError,
    _apply_flags,
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
    pending_report: Any = None
    trace_stack: list[str] = field(default_factory=list)
    question: str = ""
    bakeoff_done: bool = False


def submit(desk: Desk, line: str) -> bool:
    """Handle one line. True means the session should close."""
    line = line.strip()
    if not line:
        return False
    if desk.pending_report is not None:
        answer = line.lower()
        if answer in {"yes", "y", "save", "download"}:
            path = desk.pending_report.save()
            desk.pending_report = None
            say(f"  report  {path}")
            return False
        if answer in {"no", "n"}:
            desk.pending_report = None
            say(dim("  report left unwritten"))
            return False
        desk.pending_report = None

    desk.bakeoff_done = False

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
    """A sentence is a stack of tool calls. A bakeoff is that stack's closed room."""
    think = _thinker(desk)
    if think is None:
        say("A scenario is a stack of tool calls. Enter a model key and the harness will choose them.")
        return
    desk.question = line
    desk.trace_stack = []
    if _wants_bakeoff(line):
        _offer(desk, _begin_bakeoff(desk, think).trace)
        return

    from ..agent.sim import run_tools

    def on_call(call: Any) -> None:
        desk.trace_stack.append(call.line())
        say(f"  {len(desk.trace_stack):02d}  {call.line()}")

    trace = run_tools(
        line,
        think,
        lambda name, args: _invoke_tool(desk, name, args),
        context=_session_prior(desk) or "",
        on_call=on_call,
    )
    _offer(desk, trace)


def _wants_bakeoff(line: str) -> bool:
    low = line.lower().replace("-", " ")
    return "bakeoff" in low or "bake off" in low


def _bakeoff_seconds(value: Any) -> float:
    """A model may ask for a budget. It stays inside half a minute to a minute."""
    from ..agent.bakeoff import DEFAULT_SECONDS, HARD_CEILING

    number = _as_float(value, DEFAULT_SECONDS)
    return min(HARD_CEILING, max(30.0, number))


def _optional_rate(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_float(value: Any, default: float) -> float:
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _offer(desk: Desk, trace: Any) -> None:
    desk.pending_report = trace
    if getattr(trace, "answer", ""):
        say(trace.answer)
    say(dim("  Type yes to save this trace as a report on this machine."))


def _begin_bakeoff(desk: Desk, think: Callable[[str], str], *, seconds: float | None = None) -> Any:
    """One bakeoff for this sentence. A second request in the same sentence is refused."""
    from ..agent.bakeoff import DEFAULT_SECONDS, Bakeoff, run_bakeoff
    from ..agent.sim import Call, Trace

    if desk.bakeoff_done:
        trace = Trace(question=desk.question, answer="Referee. This sentence already ran a bakeoff.")
        return Bakeoff(trace=trace, figures={"mode": "bakeoff", "stopped": "already_ran", "n_trials": 0})
    desk.bakeoff_done = True

    def on_call(call: Call) -> None:
        desk.trace_stack.append(call.line())
        say(f"  {len(desk.trace_stack):02d}  {call.line()}")

    return run_bakeoff(
        desk.question,
        think,
        desk.data,
        seconds=DEFAULT_SECONDS if seconds is None else seconds,
        on_call=on_call,
    )


def _invoke_tool(desk: Desk, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """One deterministic tool. The model chose the name. This only runs it."""
    if name == "bakeoff":
        think = _thinker(desk)
        if think is None:
            return {"error": "a bakeoff needs a model key"}
        held = _begin_bakeoff(desk, think, seconds=_bakeoff_seconds(args.get("seconds")))
        raw = held.figures
        cleaned: dict[str, Any] = {}
        if isinstance(raw, dict):
            cleaned.update({str(key): value for key, value in raw.items()})
        return cleaned
    if name == "profile":
        from ..core.profile import profile_data

        if desk.data is None:
            return {"error": "nothing is loaded"}
        out = profile_data(desk.data)
        return {
            "n_names": out.get("n_names"),
            "n_obs_min": out.get("n_obs_min"),
            "n_obs_max": out.get("n_obs_max"),
            "n_ragged": out.get("n_ragged"),
            "n_spikes": out.get("n_spikes"),
            "dates_supplied": out.get("dates_supplied"),
        }
    if name == "align":
        from ..core.market import align_market

        return align_market(desk.data)
    if name == "regime":
        from ..core.regime import hamilton_regime

        series = _one_series(desk, args.get("name"))
        if series is None:
            return {"error": "name a series that is loaded"}
        return hamilton_regime(series)
    if name == "price":
        from ..core.options import black_scholes
        from ..core.rates import cash_rate

        rate, _, _ = cash_rate(_optional_rate(args.get("rate")))
        return black_scholes(
            spot=_as_float(args.get("spot"), 0.0),
            strike=_as_float(args.get("strike"), 0.0),
            expiry_years=_as_float(args.get("expiry") or args.get("expiry_years"), 0.0),
            vol=_as_float(args.get("vol"), 0.0),
            rate=_as_float(args.get("rate"), rate),
            call=str(args.get("right") or "call").lower() != "put",
        )
    if name == "reprice":
        from ..core.options import reprice
        from ..core.rates import cash_rate

        rate, _, _ = cash_rate(_optional_rate(args.get("rate")))
        return reprice(
            spot=_as_float(args.get("spot"), 0.0),
            strike=_as_float(args.get("strike"), 0.0),
            expiry_years=_as_float(args.get("expiry") or args.get("expiry_years"), 0.0),
            vol=_as_float(args.get("vol"), 0.0),
            rate=_as_float(args.get("rate"), rate),
            spot_shock=_as_float(args.get("spot_shock"), 0.0),
            vol_shock=_as_float(args.get("vol_shock"), 0.0),
            call=str(args.get("right") or "call").lower() != "put",
        )
    if name in {"correlation", "covariance", "cointegration"}:
        return _tool_cross(desk, name)
    if name == "portfolio":
        method = str(args.get("method") or "").replace("_", " ")
        _portfolio(desk, method)
        book = desk.portfolio or {}
        if not book:
            return {"error": "portfolio was not built"}
        return {
            "method": book.get("method"),
            "n_assets": book.get("n_assets"),
            "weight_sum": book.get("weight_sum"),
        }
    if name == "overlap":
        symbol = str(args.get("symbol") or "")
        _overlap(desk, symbol)
        return {"symbol": symbol or None, "note": "overlap against the rest of the book"}
    if name == "performance":
        from ..core.performance import performance_report

        series = _book_returns(desk)
        if series is None:
            return {"error": "need a return series or a book of prices"}
        return performance_report(series)
    if name == "var":
        from ..core.risk import compute_var_cvar

        series = _book_returns(desk)
        if series is None:
            return {"error": "need a return series or a book of prices"}
        return compute_var_cvar(series)
    if name == "process":
        from ..core.process import dgp_stress

        series = _one_series(desk, args.get("name"))
        if series is None:
            return {"error": "need a series to fit"}
        return dgp_stress(series, dgp=str(args.get("dgp") or "gbm"))
    if name == "workflow":
        workflow = str(args.get("name") or "").strip()
        if not workflow:
            return {"error": "name the workflow"}
        why = str(args.get("why") or "").strip()
        if why:
            say(dim(f"  {why}"))
        say(dim(f"  workflow  {workflow}"))
        _scripted(desk, workflow, [])
        status = getattr(desk.last, "status", None)
        return {"workflow": workflow, "status": status or "unopened"}
    return {"error": f"{name} is not wired"}


def _one_series(desk: Desk, name: Any) -> list[float] | None:
    from ..core.relations import _values

    data = desk.data
    if isinstance(data, (list, tuple)):
        return [float(v) for v in data]
    if not isinstance(data, dict) or not data:
        return None
    if name and str(name) in data:
        values = _values(data[str(name)])
        return values
    for series in data.values():
        values = _values(series)
        if values:
            return values
    return None


def _book_returns(desk: Desk) -> list[float] | None:
    from ..core.relations import as_returns

    data = desk.data
    if isinstance(data, (list, tuple)) and len(data) >= 30:
        return [float(v) for v in data]
    if not isinstance(data, dict):
        return None
    panel = as_returns(data)
    if not panel:
        return None
    columns = list(panel.values())
    width = min(len(col) for col in columns)
    if width < 2:
        return None
    tail = [col[-width:] for col in columns]
    return [sum(row) / len(row) for row in zip(*tail, strict=True)]


def _tool_cross(desk: Desk, kind: str) -> dict[str, Any]:
    _cross(desk, kind)
    matrix = desk.matrix or {}
    if matrix.get("kind") != kind:
        return {"error": f"{kind} did not run"}
    names = list(matrix.get("names") or [])
    return {"kind": kind, "n_names": len(names), "names": names[:12]}


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
