# AGENTS.md

## Cursor Cloud specific instructions

AlphaEngine is a pure-Python CLI + library (`alphaengine`) for validated quant
research. There is no windowed GUI: the product is a terminal session (an
animated parameter-surface canvas at boot and while Working) plus
`import alphaengine`.

### Environment

- Dependencies live in `.venv` (Python 3.12). Use `.venv/bin/<tool>` or
  `source .venv/bin/activate`. The console script is `.venv/bin/alphaengine`.
- Installed editable with `dev` and `factors` extras so statsmodels is present.

### Lint / test / build / run

- Lint: `.venv/bin/python -m ruff check src tests` and
  `.venv/bin/python -m ruff format --check src tests`
- Types: `.venv/bin/python -m mypy src`
- Tests: `.venv/bin/python -m pytest` and
  `.venv/bin/python -m pytest tests -m golden -q`
- Docs guard: `.venv/bin/python scripts/gen_docs.py --check`
  (regenerate with `--write`). A benign `SyntaxWarning: invalid escape sequence '\|'`
  is pre-existing.
- Offline app: `.venv/bin/alphaengine demo`

### Non-obvious gotchas

- **Three rungs.** `demo` and the importable library are fully offline. Workflow
  CLI verbs (`diagnose`, `screen`, …, `run <workflow>`) need a portal
  `ae_live_` key (`QUANTOS_API_KEY`). Plain-English mode also needs a **model**
  key. In a session: `login` (QuantOS) or `login anthropic`. `key` is the same
  verb. Any OpenAI-compatible key works: Anthropic, OpenAI (`OPENAI_BASE_URL` for
  gateways), Gemini, Groq, OpenRouter, Azure, or
  `ALPHAENGINE_API_KEY`+`ALPHAENGINE_BASE_URL`. `alphaengine models` lists what
  this machine can actually use. Keys may persist in
  `~/.config/alphaengine/credentials.json` (mode 0600) if the user says yes at
  `login`; env always wins. LLM keys are never sent to QuantOS.
- **One data verb.** `load prices.csv` / `load research.momentum` / `load sp500`
  unifies `--data` / `--project` / `--universe`. The old verbs still work.
- **Motion is a real TTY.** Boot and `Working` paint a tall parameter-surface
  canvas. Tests that monkeypatch `_tty` get colour without frame sleeps
  (`_live_tty` gates motion). Do not add `time.sleep` on the `_tty()` path.
  This image often has `NO_COLOR=1`; unset it to see the canvas
  (`env -u NO_COLOR COLORTERM=truecolor TERM=xterm-256color`).
- **A stop / `marginal` verdict exits 0.** Only a step that could not execute
  exits non-zero. Unauthenticated workflow calls exit 2.
- **Data never leaves.** Executor and study-report guards refuse lists longer
  than 512 by length. Model telemetry (choices, why, post-guard answers, prompt
  hashes) may POST to `/api/harness/runs/{id}/events` when logged in; a 404 is
  swallowed and the same allowlist is always written to
  `~/.local/share/alphaengine/runs/<id>.jsonl`. Never send prompts, series, or
  keys. `alphaengine trace` reads the local dump.
- **Goldens are a public contract.** Do not "fix" a golden to land a speedup;
  that is a version bump (`_version.py`).
- **Compute ops on the executor** include the 0.6 workflow set (`backtest`,
  `score_backtest`, `cpcv`, `factors`, `pairs`, `cointegrated_pairs`,
  `walk_forward`, `book_overlap`), the 0.7 CS / book set
  (`panel_transform`, `signal_icir`, `fama_macbeth`, `quantile_book`,
  `ewma_cov`, `denoise_cov`, `hrp`, `risk_parity`, `vol_target`), and the
  0.8 process / Grinold set (`ou_calibrate`, `ou_simulate`, `gbm_calibrate`,
  `jump_calibrate`, `garch`, `dgp_stress`, `grinold_alpha`, `breadth_ir`,
  `detone_cov`). Panels, covariance matrices, GARCH variance paths, OU paths
  and alpha vectors stay in the workspace; figures go over the wire. The
  portal must offer an op in a workflow graph or it sits unused; the CLI
  `process` command runs the DGP stress offline without a portal. Short names
  in a panel are skipped and counted, they do not shrink the rest of the book.
- `sweep(..., jobs=N)` defaults to 1. Greater than 1 is opt-in and must keep
  trial index identity, including failures.
- The `connectors` extra is lazy: parquet via `pyarrow`, HTTP via `httpx` to a
  URL the caller names. `import alphaengine` must not import httpx.
