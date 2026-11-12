# Architecture (`alphaengine` 2.0)

`quantOSC/alphaengine` is the **single canonical engine** across the entire QuantOS stack.

Every consumer — the PyPI library (`import alphaengine`), the `alphaengine` / `ae` CLI, the `quantOS` backend API, the isolated backtest/job runner, and the `quantOS` MCP server — installs and executes this exact package by version and wheel SHA-256. No sibling repository may vendor, fork, or maintain shadow copies of engine math, backtesting, validation, trial counting, or sync schemas.

---

## 1. Repository Ownership Boundary

| Capability | Canonical Owner | Rule |
|---|---|---|
| Backtesting (`run_backtest`, `score_backtest`, `walk_forward`, coded `signal()` adapter) | `quantOSC/alphaengine` | Pure Python (`numpy` + `scipy`), deterministic, zero network calls. |
| Statistical validation (`deflated_sharpe` / `dsr_probability`, `pbo_cscv`, `combinatorial_oos_stability`, `min_track_record_length`, `harvey_liu_haircut`, `trials_for_deflated_sharpe`) | `quantOSC/alphaengine` | Single `Verdict` vocabulary (`robust`, `marginal`, `likely_noise`, `unverified`) and single cash-rate policy (`core.rates`). |
| Cross-sectional, factor, covariance, allocation, process & Grinold primitives | `quantOSC/alphaengine` | Panels, covariance matrices, and return paths stay local; bounded summary figures cross the wire. |
| Reproducibility provenance (`RunManifest`, `Study` canonical SHA-256 seal, local `.alphaengine/` ledger & CAS) | `quantOSC/alphaengine` | Content-addressed JSON (`canonical_sha256`), 3-status verification (`hash_verified`, `figures_reproduced`, `provenance_verified`). |
| Workspace sync wire contract (`alphaengine.sync.schema`, `sync_contract = 2`) | `quantOSC/alphaengine` | Imported directly by both CLI client and `quantOS` backend (`routes/workspace_sync.py`). |
| Auth, multi-tenant Postgres persistence, billing, isolated container execution, LLM orchestration, MCP transport, web UI | `quantOSC/quantOS` | Consumes `alphaengine==2.0.x` via `--require-hashes` (`constraints/engine.txt`) and asserts `alphaengine.engine_info()` at boot. |

---

## 2. Core Design Invariants

1. **Two core runtime dependencies (`numpy`, `scipy`).** Optional capabilities live behind extras (`factors`, `agents`, `connectors`, `tui`, `auth`) so `pip install alphaengine` stays minimal and `import alphaengine` never opens a network socket.
2. **Every executed attempt is a trial (William's rule).** Every single run, identical rerun, and sweep/walk-forward grid point that begins execution — whether it succeeds or raises an error — receives a unique `attempt_id` (UUID4) and increments the trial count. Only pre-execution static/input validation failures record `0` trials. Ledger write failures raise immediately rather than falling back to `N = 1`.
3. **One verdict vocabulary & stated conventions.** Statistical validation emits `Verdict` (`robust`, `marginal`, `likely_noise`, `unverified`) and `dsr_probability` (with `deflated_sharpe` retained as a deprecated 2.0 alias). Single-trial or unrecorded-trial runs set `deflation_applied = False` and cannot reach `robust`. Every scored/validated output states `engine_version`, `annualization`, `risk_free`, and `costs_model`.
4. **Coded strategies emit weights, not P&L (`alphaengine.coded`).** User `signal(*, data_asof, params)` functions receive strictly causal `data_asof` slices and return dated, signed target weights; `run_backtest` simulates fills (`next_open`, mandatory `5/1 bps` default costs, `borrow_bps`, dated `adv_series`), capped at `400` grid combinations under `coded:<idea_id>`. Legacy `backtest_fn -> returns` remains supported locally and is explicitly labeled `"user-owned P&L, not engine-verified"`.
5. **Data and secrets stay safe.** Raw datasets and user strategy code never leave the local machine during `ae push` or workflow execution (series upload requires explicit `share_series=True` opt-in within wire bounds). Market data connectors/examples use `yfinance` only (no FMP). Credentials use the OS keyring first, never write `credentials.json` when a keyring works, and are redacted across all output streams.

---

## 3. Package Layout (`src/alphaengine/`)

- `_version.py`: Single version source (`__version__`), `SYNC_CONTRACT_VERSION`, `MANIFEST_VERSION`, and runtime `engine_info()`.
- `core/`: Pure quantitative primitives (`backtest.py`, `validation.py`, `conventions.py`, `performance.py`, `rates.py`, `walkforward.py`, `risk.py`, `panel.py`, `signals.py`, `cross_section.py`, `covariance.py`, `allocate.py`, `process.py`, `active.py`, `factors.py`, `pairs.py`, `screen.py`, `data_profile.py`).
- `validation.py`: Compatibility module re-exporting `alphaengine.core.validation` and `Verdict`.
- `coded.py`: `Signal` protocol, AST static validator (`check_signal_module`), and causal weight simulator (`run_signal`).
- `sweep/`: Parameter-grid runner (`runner.py`) with per-attempt UUID tracking and surface diagnostics.
- `manifest.py`: Versioned `RunManifest` (`manifest_version = 1`) for code, dataset, parameter, cost, and environment provenance.
- `ledger.py`, `project.py`, `datasets.py`, `ideas.py`, `memo.py`, `envelope.py`: Local `.alphaengine/` workspace, CAS store, append-only JSONL logs, and 3-status verification.
- `study/`: Portable `Study` artifact (`schema.py`, `seal_version = 2` canonical projection + `seal_version = 1` legacy verifier, `report.py`).
- `sync/`: Shared `sync_contract = 2` wire schemas (`schema.py`) and push/pull client (`__init__.py`).
- `methods.py`: Machine-readable methods catalogue (`method_id`, formula, literature citation, `engine_version`) for figure citations.
- `auth.py`, `commands.py`, `cli.py`, `repl.py`, `tui/`, `client/`: Keyring-first credential management, unified CLI command schema, interactive terminal session, and remote workflow executor.
