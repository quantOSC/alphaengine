"""What you can type, declared once.

── WHY THIS FILE EXISTS ───────────────────────────────────────────────────────

The command list lived in three hand-maintained places: the REPL's `HELP`
string, the README, and the docs page on the site. None of them agreed. `--data`
and `--universe` shipped and appeared in one of the three; `demo` appeared in
two; the docs page listed **no CLI commands at all** while describing the
library in fourteen sections.

That is not a documentation problem, it is a source problem. Three copies of a
list drift within a week and the drift is invisible — every copy looks
authoritative on its own, and a reader who finds the stale one concludes the
feature does not exist.

So: one declaration, four renderings.

    HELP in the REPL                  -> `help`
    the interactive directory          -> `commands`, `commands <verb>`
    the README's reference block       -> scripts/gen_docs.py
    the docs page table                -> the same generator

A test asserts the README block matches what this generates, so a verb added
here and forgotten there fails a build rather than misleading somebody.

── WHAT A GOOD ENTRY LOOKS LIKE ───────────────────────────────────────────────

`purpose` is ONE LINE and is what a directory shows. It answers "would I want
this", not "how does it work". `body` is what `commands <verb>` expands to, and
it is allowed to explain a distinction that matters — scripted versus
exploratory, which door supplies a simulator — because that is the thing a
reader cannot infer from a signature.

`examples` are REAL and copy-pasteable. An example with a placeholder nobody can
resolve (`--project <your module>`) teaches nothing; `--project research.momentum`
at least shows the shape.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Command", "COMMANDS", "GROUPS", "Flag", "FLAGS", "by_verb", "in_group"]


@dataclass(frozen=True)
class Flag:
    name: str
    takes: str
    purpose: str


@dataclass(frozen=True)
class Command:
    verb: str
    #: Which block of the directory it appears under. See `GROUPS`.
    group: str
    #: What follows the verb, as a reader would type it. "" for a bare verb.
    args: str
    #: ONE LINE. Answers "would I want this", not "how does it work".
    purpose: str
    #: Where it works. A few verbs exist only inside the session.
    scope: str = "both"  # cli | repl | both
    #: The longer explanation `commands <verb>` prints.
    body: str = ""
    examples: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()  # names, resolved against FLAGS


#: The four questions somebody has, in the order they have them. Grouping beats
#: alphabetical here: a flat list of thirteen verbs makes a reader scan for a
#: word they do not know yet.
GROUPS: tuple[tuple[str, str], ...] = (
    ("start", "Get started"),
    ("run", "Do the work"),
    ("data", "Bring your data"),
    ("session", "Session"),
)


FLAGS: dict[str, Flag] = {
    "url": Flag("--url", "URL", "workflow server (default $QUANTOS_API_URL or the public API)"),
    "key": Flag("--key", "KEY", "portal-issued ae_live_ key (default $QUANTOS_API_KEY)"),
    "project": Flag("--project", "MODULE", "a module exposing `data` and `backtest_fn`"),
    "data": Flag("--data", "FILE", "a local CSV: wide, long, or a single series"),
    "universe": Flag("--universe", "NAME", "a universe registered in the portal, with its stored closes"),
    "symbol": Flag(
        "--symbol", "TICKER", "one name out of a loaded universe; its closes become the return series"
    ),
    "label": Flag("--label", "TEXT", "what to call the artifact this run produces"),
    "input": Flag("--input", "K=V", "a workflow input; repeatable"),
    "quiet": Flag("--quiet", "", "only the result, no step narration"),
    "limit": Flag("--limit", "N", "how many rows to show (default 25)"),
    "budget": Flag("--budget", "N", "how many runs a night is worth (default 3)"),
}


#: The questions, as commands. Each is `run <workflow>` with the workflow
#: pre-chosen: same flags, same behaviour, one implementation. Declared from
#: one table so the parser, the docs and the shell cannot drift.
_QUESTIONS: tuple[tuple[str, str, str, str], ...] = (
    (
        "diagnose",
        "diagnose_data",
        "can I trust this data?",
        "Coverage, unadjusted-split candidates, stale and ragged series, named not dropped. "
        "Reports and never stops: a dirty file closing with its problems named IS the artifact.",
    ),
    (
        "screen",
        "screen_universe",
        "what is worth a look?",
        "Rank a universe, get a bounded shortlist, and the ranked file is FILED: dated, "
        "versioned, diffable against tomorrow's.",
    ),
    (
        "signal",
        "evaluate_signal",
        "does this signal carry information?",
        "IC with its t-stat, quantile spread, decay half-life. An implausibly high IC STOPS "
        "the run naming lookahead, because a leaked edge is worse than an overfit one.",
    ),
    (
        "validate",
        "validate_study",
        "is this real?",
        "Your simulator, our referee: the sweep, the deflated Sharpe, PBO, and a trial count "
        "that was counted rather than claimed.",
    ),
    (
        "stress",
        "stress_study",
        "where does it break?",
        "The cost ladder (the bps where it dies), Sharpe by subperiod, drawdown anatomy. "
        "Pass --input turnover=<one-way per period> or the cost curve reports absences.",
    ),
    (
        "overlap",
        "check_overlap",
        "is it new, or my book again?",
        "Correlation and beta of a candidate to the book, with the verdict said plainly: "
        "diversifier, related, or the book again.",
    ),
    (
        "size",
        "size_position",
        "how much?",
        "A target weight of capital with CVaR, drawdown and the minimum track record beside "
        "it. A record too short to size is refused, not sized small.",
    ),
    (
        "monitor",
        "monitor_sleeve",
        "still inside the lines?",
        "Four-way status: ok, breached, undetermined, unchecked. Nothing checked never reads as all clear.",
    ),
)


def _question_commands() -> tuple[Command, ...]:
    return tuple(
        Command(
            verb=verb,
            group="run",
            args="",
            purpose=purpose,
            body=body + f"\n\nThe long spelling is `run {workflow}`; the flags are identical.",
            examples=(f"alphaengine {verb} --universe <name>",)
            if verb in ("diagnose", "screen")
            else (f"alphaengine {verb}",),
        )
        for verb, workflow, purpose, body in _QUESTIONS
    )


COMMANDS: tuple[Command, ...] = _question_commands() + (
    # ── get started ────────────────────────────────────────────────────────
    Command(
        verb="demo",
        group="start",
        args="",
        scope="both",
        purpose="run the built-in example offline, with no account and no data",
        body=(
            "The fastest way to see the whole offline half work. A fixed seed, so it "
            "runs with the network off, gives the same figures on every machine and "
            "needs no data licence.\n\n"
            "It is a DEMONSTRATION OF THE WIRING and not a strategy: the example is a "
            "moving-average crossover on a random walk, it has no edge, and the "
            "verdict at the end says so. That is the example working rather than "
            "failing, a tool that only ever agrees with you is not a referee."
        ),
        examples=("alphaengine demo",),
    ),
    Command(
        verb="runs",
        group="start",
        args="[--limit N]",
        scope="both",
        purpose="your own week: what ran, what it decided, what it filed",
        body=(
            "The terminal could START work and could not show you any of it, so "
            '"what did I already try" was a question you had to open a browser to '
            "ask, from the one place you are least likely to want to leave.\n\n"
            "MOSTLY STOPS, AND THAT IS THE POINT. The dead ends are the majority "
            "of a research week and the part that evaporates everywhere else, so a "
            "stopped run prints its REASON here rather than only its token: the "
            "sentence is the most useful thing the run produced.\n\n"
            "Every row carries the honest trial count WITH its provenance, because "
            "a count that was derived from a grid that ran and one that was "
            "asserted are different claims about the same number."
        ),
        examples=("alphaengine runs", "alphaengine runs --limit 50"),
    ),
    Command(
        verb="gaps",
        group="start",
        args="",
        scope="both",
        purpose="what your record says is UNANSWERED, and what closes each one",
        body=(
            "`runs` shows what you already tried. This shows what you have not: "
            "the other half of the same question, and the one you want at 9am "
            "rather than at the end of a task.\n\n"
            "EVERY LINE NAMES ITS REMEDY. A gap that cannot name one is an "
            "observation rather than a next step, so the ones with no workflow "
            "behind them are not printed as work.\n\n"
            "Derived from the record, never asked of a model: a model invited to "
            "find gaps can decline to find them, and can invent them. What you "
            "see is a query over runs that actually happened."
        ),
        examples=("alphaengine gaps",),
    ),
    Command(
        verb="tonight",
        group="start",
        args="[--budget N]",
        scope="both",
        purpose="what would run unattended tonight, without running any of it",
        body=(
            "The same derivation the scheduler uses, so this is what WILL happen "
            "rather than a description of it.\n\n"
            "EVERYTHING SKIPPED IS PRINTED. A plan showing only what it kept "
            "reads as 'this is everything that mattered', which is the one thing "
            "a plan must never imply. Two reasons appear: past tonight's budget, "
            "and blocked on an input only you can state. A monitor's tolerances "
            "are not something the planner may invent."
        ),
        examples=("alphaengine tonight", "alphaengine tonight --budget 5"),
        flags=("budget",),
    ),
    Command(
        verb="workflows",
        group="start",
        args="",
        purpose="what the server offers, what each needs, and which reproduce",
        body=(
            "The modeling week as questions: can I trust this data, what is worth "
            "a look, does this signal carry information, is it real, where does it "
            "break, is it new or my book again, how much, is it still inside the "
            "lines. Only `validate_study` wants your simulator.\n\n"
            "`reproducible` is not decoration. A SCRIPTED workflow reaches the same "
            "place twice over the same data; an EXPLORATORY one may not, and its "
            "artifact must never be read as though it does. Knowing which BEFORE you "
            "start is the point of publishing it."
        ),
        examples=("alphaengine workflows",),
    ),
    Command(
        verb="key",
        group="start",
        args="[quantos | anthropic | openai]",
        scope="repl",
        purpose="enter a credential now, or see which rungs are unlocked",
        body=(
            "Three rungs and each is useful without the one above it:\n\n"
            "  the maths      yours, offline, always. No account.\n"
            "  workflows      a QuantOS `ae_live_` key. The loop, end to end.\n"
            "  ask anything   your OWN model key. Plain English in.\n\n"
            "Bare `key` shows which are lit. `key quantos` prompts for one and "
            "rebuilds the session, because a session holds its credential and a key "
            "entered without that looks exactly like a key that does not work.\n\n"
            "Nothing here stores a model key. It is read from the environment at call "
            "time and handed to the provider's own client."
        ),
        examples=("key", "key quantos", "key anthropic"),
    ),
    Command(
        verb="commands",
        group="start",
        args="[verb]",
        purpose="this directory, or one command in full",
        body=(
            "Bare `commands` groups every verb by the question it answers. "
            "`commands <verb>` expands one: full syntax, its flags, and real examples."
        ),
        examples=("commands", "commands run", "alphaengine commands"),
    ),
    # ── do the work ────────────────────────────────────────────────────────
    Command(
        verb="run",
        group="run",
        args="<workflow>",
        purpose="run one workflow exactly as written",
        body=(
            "SCRIPTED. The workflow owns the sequence, so two runs over the same data "
            "reach the same place and the artifact is reproducible. This is the path "
            "to use when you know which question you are asking.\n\n"
            'A STOP IS A RESULT AND EXITS 0. "This did not clear the bar" is the '
            "system working exactly as intended, and a non-zero exit would make every "
            "CI pipeline treat an honest refusal as a broken build, which is the "
            "pressure that gets honesty controls switched off."
        ),
        examples=(
            "alphaengine run screen_universe --universe sp500",
            "alphaengine run size_position --data returns.csv",
            "alphaengine run validate_study --project research.momentum",
        ),
        flags=("project", "data", "universe", "symbol", "label", "input", "quiet", "url", "key"),
    ),
    Command(
        verb="<anything else>",
        group="run",
        args="",
        scope="repl",
        purpose="ask in plain English; your model picks a workflow and drives it",
        body=(
            "EXPLORATORY. Your own model chooses a workflow from the catalogue, then "
            "chooses each step from what the server permits, it never composes a "
            "sequence, so this can be non-deterministic without being unsafe.\n\n"
            "Two runs of the same question may differ, and the run says so rather "
            "than presenting an exploratory result as a reproducible one.\n\n"
            "Needs ANTHROPIC_API_KEY or OPENAI_API_KEY in this shell, and the SDK "
            "installed: `pip install 'alphaengine[agents]'`."
        ),
        examples=(
            "which of my names are overbought on RSI?",
            "is the momentum study still standing up?",
        ),
    ),
    Command(
        verb="status",
        group="run",
        args="",
        scope="repl",
        purpose="the current run, and what is loaded",
        examples=("status",),
    ),
    # ── bring your data ────────────────────────────────────────────────────
    Command(
        verb="universe",
        group="data",
        args="<name>",
        scope="repl",
        purpose="load a universe you registered in the portal, with its closes",
        body=(
            "Brings the closes you stored with it. That is your own upload, "
            "decrypted back to your own account, not us fetching market data.\n\n"
            "Holds its names and not its prices? The portal's Data page marks it "
            '"this browser only"; press "Store on my account" there and it loads '
            "here. Or use `data` with a file instead.\n\n"
            "This verb exists because `project` was the only way to load anything "
            "mid-session, so somebody with a universe in the portal had to quit and "
            "restart with a flag to reach it."
        ),
        examples=("universe sp100", "run screen_universe --universe sp100"),
    ),
    Command(
        verb="data",
        group="data",
        args="<file.csv>",
        scope="repl",
        purpose="load a local CSV without leaving the session",
        body=(
            "Wide (date,AAPL,MSFT), long (date,symbol,close), or a single series. "
            "The shape is decided by the header and nothing else; a file it cannot "
            "place says so and names all three rather than guessing."
        ),
        examples=("data prices.csv", "run size_position --data returns.csv"),
    ),
    Command(
        verb="project",
        group="data",
        args="<module>",
        scope="repl",
        purpose="load `data` and `backtest_fn` from a module of yours",
        body=(
            "The ONLY door that can supply a simulator, because a simulator is code, "
            "which is why `validate_study` needs this one and the other three "
            "workflows do not.\n\n"
            "An ordinary Python module with your prices in `data` and, if the workflow "
            "sweeps, your backtest in `backtest_fn`. A module rather than a config "
            "file because a quant already has this module: it is the notebook cell "
            "they were going to run anyway."
        ),
        examples=("project research.momentum", "project alphaengine.demo_universe"),
    ),
    Command(
        verb="logout",
        group="session",
        args="",
        purpose="remove stored credentials from this machine",
        examples=("alphaengine logout",),
    ),
    Command(
        verb="version",
        group="session",
        args="",
        scope="cli",
        purpose="print the version",
        examples=("alphaengine version",),
    ),
    Command(
        verb="help",
        group="session",
        args="",
        scope="repl",
        purpose="the short list",
        examples=("help",),
    ),
    Command(
        verb="quit",
        group="session",
        args="",
        scope="repl",
        purpose="leave the session",
        examples=("quit",),
    ),
)


#: How data reaches a run. Not commands, but the thing people are most often
#: looking for when they open the directory, so the directory names them.
DATA_DOORS: tuple[tuple[str, str], ...] = (
    ("--data FILE", "a local CSV. Wide (date,AAPL,MSFT), long (date,symbol,close), or one series."),
    ("--universe NAME", "a universe you registered in the portal, with the closes you stored."),
    ("--project MODULE", "a Python module. The only door that can carry a simulator."),
)


def by_verb(verb: str) -> Command | None:
    return next((c for c in COMMANDS if c.verb == verb), None)


def in_group(group: str, scope: str | None = None) -> list[Command]:
    """Commands in one group, optionally filtered to where they work."""
    return [c for c in COMMANDS if c.group == group and (scope is None or c.scope in ("both", scope))]
