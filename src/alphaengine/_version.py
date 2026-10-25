"""Single source of the version.

Read by hatchling at build time (see [tool.hatch.version] in pyproject.toml), so
this file and the published artifact can never disagree.

SEMVER, WITH ONE LOCAL RULE: a MAJOR bump is required whenever a computed value
changes. Once this is on PyPI the numbers are a public contract, somebody's
saved study has to reproduce in two years, so a different result from the same
inputs is a breaking change even when the signature is untouched.

While the leading digit is 0, the MINOR position carries that rule: 0.1 -> 0.2
is what a changed figure costs. The API may still move underneath it.
"""

# 0.3.0 IS THE RULE ABOVE BEING OBEYED, not a marketing number. The sweep
# addressed its result matrix by TRIAL index while failed trials contribute no
# column, so any grid with a failure read the wrong survivor's column (or
# raised). Fixing it changes the figure a 0.2 install would have produced from
# the same inputs — which is exactly what the MINOR position costs here.
#
# 0.4.0 IS THE SAME RULE AGAIN, and it is the harder case to spot because it
# reads as "we added some charts". The wire's figure cap went 64 -> 512 and the
# per-period cap in `core.signals` went with it, so the SAME INPUTS NOW PRODUCE
# DIFFERENT OUTPUT: `ic_by_period` returns up to 512 readings where it returned
# 64, `best_curve` and `drawdown_curve` carry a curve rather than a sketch of
# one, a grid of up to 512 configurations records its full surface instead of
# reporting `trials_recorded: false`, and `compute.overlap` emits a rolling
# correlation and a scatter it did not emit before.
#
# Not one of those is a bug fix, and every one changes what a saved study
# reproduces. That costs the MINOR position while the leading digit is 0.
#
# 0.5.0 IS THE RULE READ THE OTHER WAY: no computed value moved, and the SURFACE
# broke. `--sleeve` and `--thesis` are removed along with `Session.sleeves()`,
# `.theses()` and `.propose_sleeve()`. Their three routes were part of the thesis
# object model, which came out of the platform with the agent desk it was built
# around, so the flags could only have failed at the END of a run — after the
# work, naming a server error rather than a missing feature.
#
# A removed flag is a breaking change even though every figure is byte-identical,
# and while the leading digit is 0 the MINOR position carries that too.
#
# Added in the same release, and the reason the removal is not a loss: `gaps` and
# `tonight`. The record already knew which workflow closes each open gap and
# nothing traversed those edges; these two read that derivation, so the terminal
# can now answer "what have I NOT tried" and "what will run while I sleep".
#
# 0.6.0 IS THE SAME RULE READ AS A FEATURE RELEASE. No computed value moved.
# The artefact grew a research OS around the maths that already shipped: any
# BYOK model (Anthropic, OpenAI-compatible, Gemini, Groq, OpenRouter, Azure,
# a private gateway), the workflow ops that were library-only (`backtest`,
# CPCV, factors, pairs, walk-forward, book overlap), allowlisted traces that
# stay on disk when the portal has no route, a session with history and slash
# verbs, and a screen that paints the parameter surface it is judging.
#
# A new published wheel is a new thing people `pip install`. While the leading
# digit is 0 the MINOR position carries that, the same way 0.5.0 carried a
# surface change that left every figure byte-identical.
#
# 0.6.1 IS THE SCREEN, NOT THE MATHS. No computed value moved. The session is a
# transcript now: a hue-cycling wait-star over a travelling plateau, cards for
# what a step found, a composer instead of a dashboard. Same rungs, same
# doors, same refusals. The published artefact is a new wheel because 0.6.0
# already left the building.
#
# 0.8.0 IS THE MODELING SUITE THE REST OF THE LIBRARY WAS MISSING. Existing
# DSR / PBO / performance / screen goldens stay byte-identical. The artefact
# grew named processes (OU, GBM, jumps, GARCH), DGP stress with an honest
# Monte Carlo trial count, Grinold alpha / breadth / TC, a frozen chart
# vocabulary, and a terminal boot that is the README banner in glyphs.
# New figures are a public contract from this release; old numbers did not move.
#
# 0.9.0: `run_backtest(..., fill_timing="next_open")` no longer fills the first
# bar with that bar's own signal. A signal is known at the close, and the open
# of the same bar printed before it. The first fill is now the next bar's open.
# Close fills are unchanged. No golden in tests/test_goldens.py moves. The
# same release makes the full-screen terminal the only session, and posts each
# math line's series on `/traces` without putting those series inside figures.
#
# 0.9.1 IS THE SESSION, NOT THE MATHS. No computed value moved. A signed-in
# account loads its one stored universe from the portal. `enter model key`
# takes whichever provider key is pasted. The left pane lists the commands.
#
# 0.9.2 IS THE SESSION, NOT THE MATHS. No computed value moved. A command
# shows that it is working, and a run writes each step as it happens. A thesis
# on the account can be pinned, and the run opened after that carries its id.
#
# 0.9.3 IS THE SESSION, NOT THE MATHS. No computed value moved. The math pane
# is narrower. A sentence is shown the workflow it is choosing and what a step
# already measured, and a finished step is not run again.
#
# 0.9.4 IS THE SESSION, NOT THE MATHS. No computed value moved. The session
# opens on the desk: what is loaded, the thesis on its own lines, and the
# next thing to type.
#
# 0.9.5 IS THE SESSION, NOT THE MATHS. No golden moved. Correlation, covariance,
# and cointegration run on the universe already loaded, and the square matrix
# is filed for the portal. check_overlap keeps that universe: one named symbol
# against the equal-weight of the others. The correlation and beta are the
# same numbers; the step now also names the candidate and the book.
#
# 0.10.0 MOVES ONE GOLDEN, ON PURPOSE. Parametric VaR is
# max(0, -mean*h + z*std*sqrt(h)). The 0.9 figure was z*std*sqrt(h), so the
# pinned seed goes 1.91 -> 1.84. Historical VaR and CVaR scale the same way:
# the mean over the horizon, the shock over its square root. At one day a
# negative percentile is still its absolute value. Daily vol is unchanged.
# Deflated Sharpe, PBO, and the performance golden stay: the default cash
# rate is 0, excess of zero, not a bill. The factor regression used to assume
# 4% when the caller passed nothing; it now uses 0, and reports that, plus
# whether the factors were already excess and that alpha is an arithmetic
# annualization. A supplied dollar equity curve is no longer read as a return
# by subtracting 1 from its last mark. A drawdown built from returns starts
# at unit capital, so a loss on the first bar counts. Grinold alpha is empty
# when no volatility is supplied, instead of treating every name as vol 1.
# Breadth does not imply an IR until the names are declared independent or a
# mean correlation is supplied. The same release answers a later question
# from the session record, and builds a portfolio (hierarchical risk parity,
# or risk parity when asked) from the loaded returns. Those two add no
# figure to the goldens.
#
# 1.0.0 IS THE FIRST RELEASE. Everything before 1.0 was the beta: the
# maths, the session, the airlock, the pinned 4.21% H.15 3-month Treasury
# bill rate (as of 2026-10-06, DGS3MO; pass 0 for excess of zero), automatic
# Newey-West HAC lag selection, and Benjamini-Hochberg FDR cointegration
# screening. 1.0 is the simulator: a sentence is a stack of tool calls, the
# trace is drawn while the tools run, and a report file is written on this
# machine only when the user accepts it.
#
# 1.0.1 IS THE REPORT AND THE HARNESS TOOLS. No golden in tests/test_goldens.py
# moves. The session adds `monte_carlo` and `scenarios` to the tool stack, fixes
# the sample-count divisor in Ledoit-Wolf shrinkage so hierarchical risk parity
# and risk parity differentiate weights across a 50-name panel, and writes a
# multi-page portfolio, scenario, Monte Carlo, and bakeoff report to Downloads.
#
# 1.0.2 IS THE FOUR-PASS GAN BAKEOFF. No golden in tests/test_goldens.py moves.
# The bakeoff tournament runs four adversarial passes (eight candidate proposals
# across `equal`, `hrp`, `risk_parity`, `inverse_vol`, `var_sorted`, `low_vol`,
# `momentum`, and `reversion` against eight regime, tail, rate, and crash-cluster
# attacks) under a hard twelve-call, sixty-second ceiling, feeding the referee's
# post-attack standing back to the board after every pass.
#
# 1.0.3 IS TRUST, SEALED STUDIES, AND THE N=1 DSR CORRECTION.
# `deflated_sharpe(..., n_trials=1)` now returns `psr_vs_zero` with
# `sr0_expected_max = 0.0` (the n_trials=1 golden moves from 0.6875 / 0.037 to
# 0.8434 / 0.0) instead of silently deflating as N=2. `sweep.verdict()` passes
# cross-trial Sharpe dispersion into `deflated_sharpe` per Bailey & López de
# Prado (2014). `combinatorial_oos_stability` replaces `cpcv_score` (kept as a
# deprecated alias). Study schema 1.2 adds `content_sha256` sealing, full 64-hex
# data hashes, `alphaengine seal`, `alphaengine report`, and
# `alphaengine.io` loaders (`from_quantconnect`, `from_vectorbt`). Series upload
# on `/traces` is opt-in via `--share-series` or `ALPHAENGINE_SHARE_SERIES=1`.
# 1.1.0 IS THE LOCAL-FIRST RESEARCH OS. No golden in tests/test_goldens.py
# moves. `ae init` creates `.alphaengine/` (`project.toml`, append-only JSONL
# ledgers, dataset registry, ideas, studies, and SHA-256 content-addressed
# objects). Commands walk up from the working directory to find the project,
# accumulate trial counts across runs on the same idea or family, emit and
# consume versioned `v1.0` JSON result envelopes (`--json` / `--stdin`), run
# declarative YAML pipelines, track hypotheses with `--kill-if` rules and
# prior-work warnings, export `[run_id]`-cited committee memos, discover
# `alphaengine.plugins` entry points, watch datasets, monitor live drift, and
# sync with quantOS on opt-in (`ae push` / `ae pull`).
__version__ = "1.1.0"

#: Default workflow/ingest API host until a dedicated qosai.app API host exists.
#: Overridable via ``ALPHAENGINE_API_URL`` (or ``QUANTOS_API_URL``).
DEFAULT_API_URL = "https://alpha-backend-production-51df.up.railway.app"
API_URL_ENV = "ALPHAENGINE_API_URL"
LEGACY_API_URL_ENV = "QUANTOS_API_URL"
