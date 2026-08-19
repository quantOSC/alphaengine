# AGENTS.md

## Cursor Cloud specific instructions

AlphaEngine is a pure-Python CLI + library (`alphaengine`) for validated quant
research — parameter sweeps, deflated Sharpe / PBO overfitting detection, and
portable study artifacts. There is no GUI; all interaction is via the terminal
or `import alphaengine`.

### Environment

- Dependencies are installed into a virtualenv at `.venv` (Python 3.12). Run
  tools via `.venv/bin/<tool>` or activate with `source .venv/bin/activate`.
  The console script is `.venv/bin/alphaengine`; `.venv/bin/python -m alphaengine`
  always works too. The update script keeps this venv in sync.
- The package is installed editable with the `dev` and `factors` extras, so
  `statsmodels` (factor/cointegration path) is importable and type-checked.

### Lint / test / build / run

Standard commands are documented in `README.md` ("Development") and enforced in
`.github/workflows/ci.yml`. Run them via the venv, e.g.:

- Lint: `.venv/bin/python -m ruff check src tests` and
  `.venv/bin/python -m ruff format --check src tests`
- Types: `.venv/bin/python -m mypy src`
- Tests: `.venv/bin/python -m pytest` (full) and
  `.venv/bin/python -m pytest tests -m golden -q` (frozen public-contract figures)
- Docs guard: `.venv/bin/python scripts/gen_docs.py --check` (fails if the
  README command table drifts from `src/alphaengine/commands.py`; regenerate
  with `--write`). Note: this script prints a benign `SyntaxWarning: invalid
  escape sequence '\|'` — it is pre-existing and not a failure.
- Run the app offline: `.venv/bin/alphaengine demo` (full research loop, no
  account, no network).

### Non-obvious gotchas

- **Only the offline "maths" rung runs without credentials.** `alphaengine demo`
  and the importable library (`sweep`, `Study`, `core.*`) are fully offline. The
  workflow CLI verbs (`diagnose`, `screen`, `signal`, `validate`, `stress`,
  `overlap`, `size`, `monitor`, `run <workflow>`) call a remote QuantOS server
  and require a portal-issued `ae_live_` key via `QUANTOS_API_KEY` (or
  `key quantos`); without it they print "Not authenticated" and exit 2. Plain-English
  "ask anything" mode additionally needs your own model key (`ANTHROPIC_API_KEY`
  / `OPENAI_API_KEY`). These keys are not stored by the package.
- **A refusal / "Stop" exits 0 on purpose.** "This did not clear the bar" is the
  system working, not a failure. Only a run that could not execute a step exits
  non-zero (and "not authenticated" exits 2). Do not treat a `stop`/`marginal`
  verdict as a broken build.
- Golden tests (`-m golden`) pin computed values as a public contract; if one
  moves, that is intended to fail CI and requires a version bump — do not "fix"
  a golden to make it pass.
