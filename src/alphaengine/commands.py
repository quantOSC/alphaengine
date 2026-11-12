"""What you can type, declared once.

── WHY THIS FILE EXISTS ───────────────────────────────────────────────────────

The command list lived in three hand-maintained places: the REPL's `HELP`
string, the README, and the docs page on the site. None of them agreed. `--data`
and `--universe` shipped and appeared in one of the three; `demo` appeared in
two; the docs page listed **no CLI commands at all** while describing the
library in fourteen sections.

So: one declaration (`COMMANDS` + `FLAGS`), driving the parser, the interactive
`commands` directory, and `scripts/gen_docs.py` (`QOS-169`).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "Command",
    "COMMANDS",
    "DATA_DOORS",
    "GROUPS",
    "Flag",
    "FLAGS",
    "access_word",
    "by_verb",
    "in_group",
    "slash_entries",
]


@dataclass(frozen=True)
class Flag:
    name: str
    takes: str
    purpose: str
    deprecated: bool = False


@dataclass(frozen=True)
class Command:
    verb: str
    #: Which block of the directory it appears under. See `GROUPS`.
    group: str
    #: What follows the verb, as a reader would type it. "" for a bare verb.
    args: str
    #: ONE LINE (<= 80 chars). Answers "would I want this", not "how does it work".
    purpose: str
    #: Where it works: `cli`, `repl`, or `both`.
    scope: str = "both"
    #: Execution boundary: `local`, `remote`, or `hybrid` (`QOS-169`).
    locality: str = "local"
    #: Whether `--json` emits a structured machine-readable JSON payload.
    supports_json: bool = False
    #: The longer explanation `commands <verb>` prints.
    body: str = ""
    examples: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()  # names, resolved against FLAGS


GROUPS: tuple[tuple[str, str], ...] = (
    ("start", "Get started"),
    ("run", "Do the work"),
    ("data", "Bring your data"),
    ("session", "Session"),
)


FLAGS: dict[str, Flag] = {
    "url": Flag("--url", "URL", "workflow server (default $ALPHAENGINE_API_URL or public API)"),
    "key": Flag(
        "--key",
        "KEY",
        "deprecated: exposes key in process list; use login or $QUANTOS_API_KEY",
        deprecated=True,
    ),
    "borrow_bps": Flag(
        "--borrow-bps",
        "BPS",
        "annual borrow rate in basis points accrued on short positions (default: 50.0 bps)",
    ),
    "signal": Flag(
        "--signal",
        "PATH_OR_MOD",
        "coded signal module or .py file exposing signal(data_asof=..., params=...)",
    ),
    "project": Flag(
        "--project",
        "MODULE",
        "a module exposing `data` and `backtest_fn` (user-owned P&L, not engine-verified)",
    ),
    "data": Flag("--data", "FILE", "a local CSV or Parquet file: wide, long, or a single series"),
    "universe": Flag("--universe", "NAME", "a universe registered in the portal, with its stored closes"),
    "thesis": Flag("--thesis", "NAME", "a thesis on this account; the run is attached to it"),
    "symbol": Flag(
        "--symbol", "TICKER", "one name out of a loaded universe; its closes become the return series"
    ),
    "label": Flag("--label", "TEXT", "what to call the artifact this run produces"),
    "input": Flag("--input", "K=V", "a workflow input; repeatable"),
    "quiet": Flag("--quiet", "", "only the result, no step narration"),
    "stream": Flag("--stream", "", "print the answer once the citation guard has passed"),
    "share_series": Flag(
        "--share-series",
        "",
        "opt in to uploading full series on run/sync payloads (default off)",
    ),
    "limit": Flag("--limit", "N", "maximum rows or items per page (default 25 on log/runs, 100 on pull)"),
    "budget": Flag("--budget", "N", "how many runs a night is worth (default 3)"),
    "purge": Flag(
        "--purge",
        "",
        "clear OS keyring, remove credentials.json, and unset keys in this process",
    ),
    "no_keyring": Flag(
        "--no-keyring",
        "",
        "store credential in mode-0600 credentials.json instead of the OS keyring",
    ),
    "name": Flag(
        "--name",
        "NAME",
        "name for the initialized project or registered dataset",
    ),
    "idea": Flag(
        "--idea",
        "ID",
        "attach the run or filter the log to a local hypothesis id",
    ),
    "family": Flag(
        "--family",
        "FAMILY",
        "strategy or experiment family for cumulative trial counting",
    ),
    "since": Flag(
        "--since",
        "CURSOR_OR_DATE",
        "filter project log on/after YYYY-MM-DD, or pull cursor token on pull",
    ),
    "verdict": Flag(
        "--verdict",
        "VERDICT",
        "filter project log by verdict (robust, marginal, likely_noise, unverified)",
    ),
    "asof_col": Flag(
        "--asof-col",
        "COL",
        "as-of column name or cutoff date for look-ahead checks in data add",
    ),
    "json": Flag(
        "--json",
        "",
        "emit structured machine-readable JSON to stdout",
    ),
    "stdin": Flag(
        "--stdin",
        "",
        "read an upstream JSON result envelope from standard input",
    ),
    "kill_rule": Flag(
        "--kill-rule",
        "RULE",
        "falsification rule for a hypothesis, e.g. 'DSR < 0.9'",
    ),
    "kill_if": Flag(
        "--kill-if",
        "RULE",
        "deprecated alias for --kill-rule",
        deprecated=True,
    ),
    "format": Flag(
        "--format",
        "FMT",
        "output format for report memo: md or pdf (default md)",
    ),
    "dry_run": Flag(
        "--dry-run",
        "",
        "print the sync boundary preview without opening a network socket",
    ),
    "show_payload": Flag(
        "--show-payload",
        "",
        "include the redacted sync_contract=2 JSON payload in the push preview",
    ),
    "yes": Flag(
        "--yes",
        "",
        "skip interactive confirmation prompt on push",
    ),
    "live": Flag(
        "--live",
        "FILE",
        "live return series CSV to compare against a recorded backtest run",
    ),
    "interval": Flag(
        "--interval",
        "SECONDS",
        "polling interval in seconds for watch (default 1.0)",
    ),
    "cycles": Flag(
        "--cycles",
        "N",
        "number of polling cycles for watch before exiting (0 means continuous)",
    ),
    "version": Flag(
        "--version",
        "",
        "print engine_version, sync_contract, and wheel_sha256",
    ),
    "allow_code": Flag(
        "--allow-code",
        "",
        "permit importing external user project modules during verify",
    ),
}


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
        "Your signal or simulator, our referee: the sweep, DSR probability, PBO, and a trial "
        "count that was counted per executed attempt rather than claimed.",
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
        "what sizing does the risk model imply?",
        "A model-implied weight of capital with CVaR, drawdown and the minimum track record beside "
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
    base_flags = (
        "signal",
        "project",
        "data",
        "universe",
        "symbol",
        "thesis",
        "label",
        "input",
        "quiet",
        "stream",
        "share_series",
        "json",
        "stdin",
        "idea",
        "family",
        "url",
    )
    return tuple(
        Command(
            verb=verb,
            group="run",
            args="",
            purpose=purpose,
            locality="hybrid",
            supports_json=True,
            body=body + f"\n\nThe long spelling is `run {workflow}`; the flags are identical.",
            examples=(f"alphaengine {verb} --universe sp500",)
            if verb in ("diagnose", "screen")
            else (f"alphaengine {verb}",),
            flags=base_flags + (("live",) if verb == "monitor" else ()),
        )
        for verb, workflow, purpose, body in _QUESTIONS
    )


COMMANDS: tuple[Command, ...] = _question_commands() + (
    # ── get started ────────────────────────────────────────────────────────
    Command(
        verb="init",
        group="start",
        args="[--name NAME]",
        scope="both",
        locality="local",
        purpose="initialize a local .alphaengine/ research project in this directory",
        body=(
            "Creates `.alphaengine/` in the current directory with `project.toml`, "
            "`ledger/`, `data/`, `studies/`, `ideas/`, `universes/`, and content-addressed "
            "`objects/`.\n\n"
            "Commands find the project by walking up from the current directory, the way "
            "git does. Outside a project, every command behaves as a standalone call."
        ),
        examples=("alphaengine init", "alphaengine init --name momentum"),
        flags=("name",),
    ),
    Command(
        verb="demo",
        group="start",
        args="[--json]",
        scope="both",
        locality="local",
        supports_json=True,
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
        flags=("json", "idea", "family"),
    ),
    Command(
        verb="process",
        group="start",
        args="[ou | gbm | jump | garch]",
        scope="both",
        locality="local",
        supports_json=True,
        purpose="fit a named process to a series and stress it, offline",
        body=(
            "Calibrate a named data-generating process on a close, a spread, or "
            "returns, then replay the book under Monte Carlo draws from that law.\n\n"
            "Default is GBM. `process ou` asks whether the path mean-reverts; "
            "`process garch` fits a GARCH(1,1) vol path you can then vol-target. "
            "Pass --data for a local file; with nothing loaded it uses the built-in "
            "demo walk, which has no edge, and the stress should say so.\n\n"
            "Figures only: scalars, a histogram, a sketched band. The simulated "
            "paths stay on this machine. Trial count is the path count, sourced "
            "as monte_carlo. This is local, like `demo`; it does not call the portal."
        ),
        examples=("alphaengine process", "alphaengine process ou", "process garch"),
        flags=("data", "json", "stdin", "idea", "family"),
    ),
    Command(
        verb="runs",
        group="start",
        args="[--limit N]",
        scope="both",
        locality="remote",
        purpose="your own week: what ran, what it decided, what it filed",
        body=(
            "Reads recent runs from your quantOS account.\n\n"
            "MOSTLY STOPS, AND THAT IS THE POINT. The dead ends are the majority "
            "of a research week and the part that evaporates everywhere else, so a "
            "stopped run prints its REASON here rather than only its token.\n\n"
            "Every row carries the honest trial count WITH its provenance."
        ),
        examples=("alphaengine runs", "alphaengine runs --limit 50"),
        flags=("limit", "url"),
    ),
    Command(
        verb="log",
        group="start",
        args="[--idea ID] [--family FAMILY] [--since DATE] [--verdict VERDICT]",
        scope="both",
        locality="local",
        supports_json=True,
        purpose="list local project ledger runs, trial counts, and verdicts",
        body=(
            "Reads `.alphaengine/ledger/runs.jsonl` and prints recorded runs newest "
            "first, showing both the single-run trial count and the cumulative trial "
            "count accumulated across executed attempts in the idea or family."
        ),
        examples=(
            "alphaengine log",
            "alphaengine log --family momentum --verdict likely_noise",
        ),
        flags=("idea", "family", "since", "verdict", "limit", "json"),
    ),
    Command(
        verb="show",
        group="start",
        args="<run> [--json]",
        scope="both",
        locality="local",
        supports_json=True,
        purpose="inspect one recorded project run, its hashes, trials, and figures",
        body=(
            "Prints the full ledger record for `<run>` (exact run id, prefix, or "
            "`HEAD`), including parameter hash, dataset hashes, manifest hash, "
            "single-run and cumulative trial counts, figures, and verdict."
        ),
        examples=("alphaengine show HEAD",),
        flags=("json",),
    ),
    Command(
        verb="diff",
        group="start",
        args="<run-a> <run-b> [--json]",
        scope="both",
        locality="local",
        supports_json=True,
        purpose="compare data, parameter, trial, and figure changes between two runs",
        body=(
            "Compares two runs from the project ledger side by side: dataset hash "
            "changes, parameter changes, trial count changes, and numeric figure "
            "deltas."
        ),
        examples=("alphaengine diff run_a run_b",),
        flags=("json",),
    ),
    Command(
        verb="gaps",
        group="start",
        args="",
        scope="both",
        locality="remote",
        purpose="what your record says is UNANSWERED, and what closes each one",
        body=(
            "`runs` shows what you already tried. This shows what you have not: "
            "the other half of the same question.\n\n"
            "EVERY LINE NAMES ITS REMEDY. Derived from the record, never asked of a model."
        ),
        examples=("alphaengine gaps",),
        flags=("url",),
    ),
    Command(
        verb="tonight",
        group="start",
        args="[--budget N]",
        scope="both",
        locality="remote",
        purpose="what would run unattended tonight, without running any of it",
        body=(
            "The same derivation the scheduler uses, so this is what WILL happen "
            "rather than a description of it. Everything skipped is printed with its reason."
        ),
        examples=("alphaengine tonight", "alphaengine tonight --budget 5"),
        flags=("budget", "url"),
    ),
    Command(
        verb="workflows",
        group="start",
        args="",
        locality="remote",
        purpose="what the server offers, what each needs, and which reproduce",
        body=(
            "Lists workflows published by the configured quantOS server, their input "
            "requirements, and whether each is reproducible."
        ),
        examples=("alphaengine workflows",),
        flags=("url",),
    ),
    Command(
        verb="login",
        group="start",
        args="[quantos | anthropic | openai | gemini | groq | azure | openrouter | gateway]",
        scope="both",
        locality="local",
        purpose="sign in, or login anthropic for a model key",
        body=(
            "Three rungs and each is useful without the one above it:\n\n"
            "  the maths      yours, offline, always. No account.\n"
            "  workflows      a QuantOS `ae_live_` key. The loop, end to end.\n"
            "  ask anything   your OWN model key. Plain English in.\n\n"
            "When you choose to stay signed in, credentials are stored in the OS "
            "keychain (`keyring`) when available; `credentials.json` (mode 0600) is "
            "written only when no OS keyring is available or `--no-keyring` is passed."
        ),
        examples=("login", "login anthropic", "alphaengine login"),
        flags=("no_keyring",),
    ),
    Command(
        verb="key",
        group="start",
        args="[quantos | anthropic | openai | gemini | groq | azure | openrouter | gateway]",
        scope="repl",
        locality="local",
        purpose="same as login: enter a credential, or see which rungs are lit",
        body=(
            "Kept because muscle memory. Prefer `login`. Bare `key` still shows "
            "the rungs; `key anthropic` still prompts."
        ),
        examples=("key", "key quantos", "key anthropic"),
    ),
    Command(
        verb="commands",
        group="start",
        args="[verb]",
        locality="local",
        purpose="this directory, or one command in full",
        body=(
            "Bare `commands` groups every verb by the question it answers and labels "
            "`[local]` vs `[remote]`. `commands <verb>` expands one in full."
        ),
        examples=("commands", "commands run", "alphaengine commands"),
    ),
    # ── do the work ────────────────────────────────────────────────────────
    Command(
        verb="run",
        group="run",
        args="<workflow>",
        locality="hybrid",
        supports_json=True,
        purpose="run one workflow or coded signal exactly as written",
        body=(
            "SCRIPTED. The workflow owns the sequence, so two runs over the same data "
            "reach the same place and the artifact is reproducible. This is the path "
            "to use when you know which question you are asking.\n\n"
            "Pass `--signal path/to/my_signal.py` to execute an engine-verified causal "
            "weight signal (`signal(data_asof=..., params=...)`) with mandatory costs "
            "and attempt-level trial counting. `--project` runs a user-owned `backtest_fn` "
            "(explicitly labeled not engine-verified).\n\n"
            'A STOP IS A RESULT AND EXITS 0. "This did not clear the bar" is the '
            "system working as intended."
        ),
        examples=(
            "alphaengine run screen_universe --universe sp500",
            "alphaengine run validate_study --signal research/momentum_signal.py --data prices.csv",
            "alphaengine run validate_study --project research.momentum",
            "alphaengine run examples/screen_validate_size.yaml",
        ),
        flags=(
            "signal",
            "project",
            "data",
            "universe",
            "symbol",
            "thesis",
            "label",
            "input",
            "quiet",
            "stream",
            "share_series",
            "json",
            "stdin",
            "idea",
            "family",
            "url",
        ),
    ),
    Command(
        verb="rerun",
        group="run",
        args="<run> [--json]",
        scope="both",
        locality="local",
        supports_json=True,
        purpose="re-execute a local run and check hash, figure, and provenance parity",
        body=(
            "Re-executes `<run>` from its recorded local inputs and reports three "
            "independent verification statuses: `hash_verified` (CAS integrity), "
            "`figures_reproduced` (every recorded scalar and curve/list within tolerance, "
            "failing if any key is missing), and `provenance_verified` (`RunManifest` "
            "engine/wheel/code/environment parity)."
        ),
        examples=("alphaengine rerun HEAD", "alphaengine rerun HEAD --json"),
        flags=("json",),
    ),
    Command(
        verb="watch",
        group="run",
        args="<pipeline.yaml> [--interval SECONDS]",
        scope="both",
        locality="local",
        purpose="re-run a declarative pipeline whenever a registered dataset file changes",
        body=(
            "Polls the pipeline file and every registered dataset in `.alphaengine/data/` "
            "using SHA-256 content fingerprints, re-executing the pipeline whenever a "
            "watched file changes on disk."
        ),
        examples=("alphaengine watch examples/screen_validate_size.yaml",),
        flags=("interval", "cycles"),
    ),
    Command(
        verb="seal",
        group="run",
        args="<study.json | run | idea>",
        scope="both",
        locality="local",
        supports_json=True,
        purpose="write canonical seal_version=2 content_sha256 into a study or run",
        body=(
            "Computes the deterministic SHA-256 over `sealed_study_projection(study)` "
            "(`seal_version = 2`, matching the exact sync wire projection) and writes "
            "`content_sha256` into the study file."
        ),
        examples=("alphaengine seal study.json", "alphaengine seal HEAD"),
        flags=("json",),
    ),
    Command(
        verb="verify",
        group="run",
        args="<study.json> [--data FILE] [--allow-code] [--json]",
        scope="both",
        locality="local",
        supports_json=True,
        purpose="verify a study.json seal, figure reproduction, and manifest provenance",
        body=(
            "Runs offline without an account or key. Reports three independent statuses: "
            "`hash_verified` (`seal_version = 2` canonical projection or `legacy_v1`), "
            "`figures_reproduced` (re-running against `--data` or local project record), "
            "and `provenance_verified` (`RunManifest` environment parity)."
        ),
        examples=(
            "alphaengine verify study.json",
            "alphaengine verify study.json --data prices.csv",
        ),
        flags=("data", "allow_code", "json"),
    ),
    Command(
        verb="report",
        group="run",
        args="<study.json> | memo <idea | run> [--format md | pdf]",
        scope="both",
        locality="hybrid",
        purpose="send a study to quantOS (/api/me/runs/ingest), or export a cited memo",
        body=(
            "Sends a saved `study.json` to your quantOS workspace (`POST /api/me/runs/ingest`).\n\n"
            "`report memo <idea|run> --format md|pdf` exports a local committee-style "
            "research memo where every figure cites its `[run_id]`."
        ),
        examples=(
            "alphaengine report study.json",
            "alphaengine report memo HEAD --format md",
        ),
        flags=("format", "url"),
    ),
    Command(
        verb="<anything else>",
        group="run",
        args="",
        scope="repl",
        locality="hybrid",
        purpose="ask in plain English; your model picks a workflow and drives it",
        body=(
            "EXPLORATORY. Your own model chooses a workflow from the catalogue, then "
            "chooses each step from what the server permits.\n\n"
            "Needs a model key in this shell and `pip install 'alphaengine[agents]'`."
        ),
        examples=(
            "which of my names rank highest on RSI?",
            "is the momentum study still standing up?",
        ),
    ),
    Command(
        verb="status",
        group="run",
        args="",
        scope="repl",
        locality="local",
        purpose="the current run, and what is loaded",
        examples=("status",),
    ),
    Command(
        verb="models",
        group="start",
        args="",
        locality="local",
        purpose="which model providers this machine can actually use",
        body=(
            "Lists every provider that has a key AND an SDK on this machine. A key "
            "without an SDK is reported separately."
        ),
        examples=("alphaengine models", "models"),
    ),
    Command(
        verb="model",
        group="start",
        args="[<provider[:name]>]",
        scope="repl",
        locality="local",
        purpose="pin the model for this session, or show the pin",
        body=(
            "Examples: `model anthropic`, `model openai/gpt-4o`, `model groq`. "
            "Sets ALPHAENGINE_PROVIDER / ALPHAENGINE_MODEL in this process only."
        ),
        examples=("model", "model anthropic", "model openai/gpt-4o"),
    ),
    Command(
        verb="trace",
        group="start",
        args="[run_id]",
        locality="local",
        purpose="local model/run events for a run, hashed prompts only",
        body=(
            "Reads ~/.local/share/alphaengine/runs/<id>.jsonl. Choices, why, answers "
            "after the citation guard, usage, prompt hashes. Never prices, keys, or raw prompts."
        ),
        examples=("alphaengine trace", "trace", "trace <run_id>"),
    ),
    Command(
        verb="book",
        group="data",
        args="[<name> | status]",
        scope="repl",
        locality="local",
        purpose="show or load sleeves on the multi-strategy book",
        body=(
            "A book is named return series, not an order pad. `book` lists sleeves. "
            "`book status` monitors each one."
        ),
        examples=("book", "book status"),
    ),
    # ── bring your data ────────────────────────────────────────────────────
    Command(
        verb="load",
        group="data",
        args="<file | module | universe>",
        scope="repl",
        locality="hybrid",
        purpose="a CSV, a project module, or a portal universe",
        body=(
            "One verb, three shapes. The token decides the door:\n\n"
            "  prices.csv              a local file (wide, long, or one series)\n"
            "  research.momentum       a module exposing `data` and `backtest_fn`\n"
            "  sp500                   a universe you registered in the portal\n\n"
            "`data`, `project` and `universe` still work. Prefer `load`."
        ),
        examples=("load prices.csv", "load research.momentum", "load sp500"),
    ),
    Command(
        verb="idea",
        group="data",
        args='new "<thesis>" --kill-rule "<rule>" [--family FAMILY] | list | status <id> <state> | show <id>',
        scope="both",
        locality="local",
        supports_json=True,
        purpose="create, list, update, and evaluate local research hypotheses",
        body=(
            "Tracks hypotheses as first-class objects in `.alphaengine/ideas/ideas.jsonl` "
            'with a falsification rule (`--kill-rule "DSR < 0.9"`; legacy `--kill-if` '
            "is accepted as an alias).\n\n"
            "Attach runs with `--idea <id>`. `idea show <id>` lists linked runs, "
            "cumulative trials, and the current verdict against the kill rule."
        ),
        examples=(
            'alphaengine idea new "Cross-sectional momentum holds" --kill-rule "DSR < 0.9" --family momentum',
            "alphaengine idea list",
            "alphaengine idea status idea_1 testing",
            "alphaengine idea show idea_1",
        ),
        flags=("kill_rule", "family", "json"),
    ),
    Command(
        verb="thesis",
        group="data",
        args="[<name> | clear]",
        scope="repl",
        locality="remote",
        purpose="choose a portal thesis for the next run",
        body=(
            "Lists the theses on this account, or pins one. A pinned thesis rides on "
            "every following run, question, and workflow as thesis_id."
        ),
        examples=(
            "thesis",
            "thesis momentum",
            "thesis clear",
            "alphaengine run screen_universe --thesis momentum",
        ),
    ),
    Command(
        verb="correlation",
        group="data",
        args="",
        scope="repl",
        locality="local",
        purpose="the correlation of every loaded name with every other name",
        body=(
            "Runs on the universe already loaded. Closes become returns. The full "
            "matrix stays on this machine."
        ),
        examples=("correlation", "run check_overlap --symbol MU"),
    ),
    Command(
        verb="covariance",
        group="data",
        args="",
        scope="repl",
        locality="local",
        purpose="the sample covariance of every loaded name with every other name",
        body=("Same panel as `correlation`: the universe already loaded, closes turned into returns."),
        examples=("covariance",),
    ),
    Command(
        verb="cointegration",
        group="data",
        args="",
        scope="repl",
        locality="local",
        purpose="which loaded names are cointegrated",
        body=(
            "Engle-Granger across the loaded closes, every pair. Needs the factors "
            "extra (`pip install 'alphaengine[factors]'`)."
        ),
        examples=("cointegration",),
    ),
    Command(
        verb="portfolio",
        group="data",
        args="",
        scope="repl",
        locality="local",
        purpose="weights for the names already loaded",
        body=(
            "Hierarchical risk parity on a Ledoit-Wolf covariance of the loaded "
            "returns. Say `portfolio risk parity` for equal risk contribution."
        ),
        examples=("portfolio", "portfolio risk parity"),
    ),
    Command(
        verb="universe",
        group="data",
        args="<name>",
        scope="repl",
        locality="remote",
        purpose="same as load: a universe registered in the portal",
        body=("Brings the closes you stored with it on your account, not us fetching market data."),
        examples=("universe sp100", "run screen_universe --universe sp100"),
    ),
    Command(
        verb="data",
        group="data",
        args="add <file> [--name NAME] [--asof-col COL] | list | show <name> | lineage <run>",
        scope="both",
        locality="local",
        supports_json=True,
        purpose="register, inspect, and trace local CSV/Parquet datasets",
        body=(
            "Manages the local `.alphaengine/data/` dataset registry, storing the "
            "full SHA-256, row count, date range, column schema, and hygiene checks."
        ),
        examples=(
            "alphaengine data add prices.csv --name equities",
            "alphaengine data list",
            "alphaengine data show equities",
            "alphaengine data lineage HEAD",
        ),
        flags=("name", "asof_col", "json"),
    ),
    Command(
        verb="project",
        group="data",
        args="<module>",
        scope="repl",
        locality="local",
        purpose="same as load: a module with data and backtest_fn",
        body=(
            "Loads a Python module exposing `data` and optional `backtest_fn` "
            "(user-owned P&L; use `--signal` for engine-verified weight simulation)."
        ),
        examples=("project research.momentum", "project alphaengine.demo_universe"),
    ),
    # ── session ────────────────────────────────────────────────────────────
    Command(
        verb="remote",
        group="session",
        args="add <name> [url] | remove <name> | list",
        scope="both",
        locality="local",
        purpose="configure, remove, or list remote sync endpoints in .alphaengine/project.toml",
        body=(
            "Registers or removes a clean named remote in `.alphaengine/project.toml` "
            "(`remote add quantos [url]`, `remote remove quantos`). "
            "Rejects URLs containing embedded credentials or query secrets."
        ),
        examples=(
            "alphaengine remote add quantos",
            "alphaengine remote remove quantos",
            "alphaengine remote list",
        ),
        flags=("url",),
    ),
    Command(
        verb="push",
        group="session",
        args="[remote] [--dry-run] [--show-payload] [--yes] [--share-series]",
        scope="both",
        locality="remote",
        supports_json=True,
        purpose="send this project's summaries to the firm workspace",
        body=(
            "Posts sealed studies (`seal_version = 2`), allowlisted ledger runs, ideas "
            "(`idea_id`, `kill_rule`), and idempotent `trial_events` to "
            "`POST /api/workspace/sync` (`sync_contract = 2`).\n\n"
            "Code and raw datasets stay local. Prints a boundary preview first; "
            "`--dry-run` prints the preview without opening a socket, and `--show-payload` "
            "includes the redacted JSON payload.\n\n"
            "Exit codes: `0` all accepted/duplicate, `2` partial rejection, `3` all rejected, "
            "`4` auth/entitlement error, `5` network/server error, `6` contract mismatch."
        ),
        examples=(
            "alphaengine push --dry-run",
            "alphaengine push --dry-run --show-payload",
            "alphaengine push --yes",
        ),
        flags=("dry_run", "show_payload", "yes", "share_series", "url", "json"),
    ),
    Command(
        verb="pull",
        group="session",
        args="[remote] [--since CURSOR] [--limit N]",
        scope="both",
        locality="remote",
        supports_json=True,
        purpose="copy team run, idea, and study summaries into this project",
        body=(
            "The inbound half of workspace sync. `push` sends this machine's summaries up; "
            "`pull` copies the team's runs, ideas, and sealed studies down from "
            "`GET /api/workspace/pull`, continuing from the cursor in "
            "`.alphaengine/project.toml`.\n\n"
            "Pulled records are marked read-only. Code and price files are not downloaded. "
            "Needs a firm `ae_live_` key. The portal website is not the sync API: "
            "`remote add quantos` with no URL stores the firm API."
        ),
        examples=("alphaengine pull", "alphaengine pull --limit 50"),
        flags=("since", "limit", "url", "json"),
    ),
    Command(
        verb="whoami",
        group="session",
        args="[--json]",
        scope="both",
        locality="local",
        supports_json=True,
        purpose="show credential source, masked prefix, keyring status, and remote URL",
        body=(
            "Runs locally without requiring a key. Reports whether `QUANTOS_API_KEY` "
            "and model provider keys come from `env`, `keyring`, `file`, or `none`, "
            "whether an OS keyring backend is usable, and the active remote URL. "
            "Never prints raw secret values."
        ),
        examples=("alphaengine whoami", "alphaengine whoami --json"),
        flags=("url", "json"),
    ),
    Command(
        verb="doctor",
        group="session",
        args="[--url URL] [--json]",
        scope="both",
        locality="hybrid",
        supports_json=True,
        purpose="check local project, keyring, credentials, and server version parity",
        body=(
            "Runs locally without requiring a key, and checks `GET /api/health` when "
            "`--url` or a configured remote is reachable. Verifies `engine_version` "
            "and `sync_contract` parity and prints the exact `pip install` command "
            "when client and server drift."
        ),
        examples=("alphaengine doctor", "alphaengine doctor --json"),
        flags=("url", "json"),
    ),
    Command(
        verb="plugin",
        group="session",
        args="list",
        scope="both",
        locality="local",
        supports_json=True,
        purpose="list installed loader, method, and report template plugins",
        body=(
            "Discovers plugins registered under the `alphaengine.plugins` entry-point "
            "group across `loader`, `method`, and `template` kinds."
        ),
        examples=("alphaengine plugin list",),
        flags=("json",),
    ),
    Command(
        verb="shell",
        group="session",
        args="",
        scope="cli",
        locality="local",
        purpose="open the interactive research shell with project completion and status",
        body=(
            "Opens the interactive research shell (`ae shell`) with tab completion "
            "over command verbs, run IDs, dataset names, and hypothesis IDs."
        ),
        examples=("alphaengine shell",),
    ),
    Command(
        verb="logout",
        group="session",
        args="[--purge]",
        scope="both",
        locality="local",
        purpose="remove stored credentials from OS keyring and credentials.json",
        body=(
            "Clears stored credentials from both the OS keychain (`keyring`) and "
            "`credentials.json`. Pass `--purge` to also remove empty config directories "
            "and unset managed keys in the current process."
        ),
        examples=("alphaengine logout", "alphaengine logout --purge"),
        flags=("purge",),
    ),
    Command(
        verb="version",
        group="session",
        args="[--json]",
        scope="both",
        locality="local",
        supports_json=True,
        purpose="print engine_version, wheel_sha256, sync_contract, and Python version",
        body=(
            "Prints `engine_info()` (`engine_version`, `wheel_sha256`, `sync_contract`, "
            "`manifest_version`) and the active Python runtime version."
        ),
        examples=("alphaengine version", "alphaengine version --json"),
        flags=("json",),
    ),
    Command(
        verb="help",
        group="session",
        args="",
        scope="repl",
        locality="local",
        purpose="the short list: demo, login, load, then a question",
        examples=("help",),
    ),
    Command(
        verb="quit",
        group="session",
        args="",
        scope="repl",
        locality="local",
        purpose="leave the session",
        examples=("quit",),
    ),
)


DATA_DOORS: tuple[tuple[str, str], ...] = (
    (
        "--data FILE",
        "a local CSV or Parquet. Wide (date,AAPL,MSFT), long (date,symbol,close), or one series.",
    ),
    (
        "--signal PATH",
        "a Python module or .py file exposing `signal(data_asof=..., params=...)` (engine-verified P&L).",
    ),
    (
        "--project MODULE",
        "a Python module exposing `data` and `backtest_fn` (user-owned P&L, not engine-verified).",
    ),
    ("--universe NAME", "a universe you registered in the portal, with the closes you stored."),
)


def access_word(locality: str) -> str:
    """One word for the directory and the `/` picker.

    `free` never calls the portal. `firm` does, and needs a signed-in key.
    `both` runs on this machine until a flag sends it to the portal.
    """
    if locality == "remote":
        return "firm"
    if locality == "hybrid":
        return "both"
    return "free"


_SLASH_FIRST = (
    "demo",
    "login",
    "load",
    "screen",
    "validate",
    "push",
    "pull",
    "commands",
    "help",
)


def slash_entries() -> tuple[tuple[str, str, str], ...]:
    """Rows for the session `/` picker: verb, access word, one-line purpose."""
    pool: dict[str, tuple[str, str, str]] = {}
    for command in COMMANDS:
        if command.scope == "cli" or command.verb == "<anything else>":
            continue
        pool[command.verb] = (command.verb, access_word(command.locality), command.purpose)
    ordered: list[tuple[str, str, str]] = []
    for verb in _SLASH_FIRST:
        row = pool.pop(verb, None)
        if row is not None:
            ordered.append(row)
    ordered.extend(pool.values())
    ordered.append(("view", "firm", "the books on this account"))
    ordered.append(("details", "free", "version and what stays on this machine"))
    return tuple(ordered)


def by_verb(verb: str) -> Command | None:
    return next((c for c in COMMANDS if c.verb == verb), None)


def in_group(group: str, scope: str | None = None) -> list[Command]:
    """Commands in one group, optionally filtered to where they work."""
    return [c for c in COMMANDS if c.group == group and (scope is None or c.scope in ("both", scope))]
