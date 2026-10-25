# Changelog

All notable changes to `alphaengine` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to Semantic Versioning (with computed golden values
treated as part of the public contract).

## [1.1.1] - 2026-10-24

### Added
- **Single firm-key choke point (`alphaengine.auth.require_firm_key`):** Used by every gated command (`run`, `show`, `monitor`/`watch`, `validate`, `screen`, `diagnose`, `signal`, `stress`, `overlap`, `size`, `workflows`, `runs`, `gaps`, `tonight`, and the CLI OS: `init`, `log`, `diff`, `rerun`, `data`, `idea`, `seal`, `report`, `remote`, `push`, `pull`, `plugin`, `shell`). Free and keyless forever: `import alphaengine`, `alphaengine demo`, `alphaengine process`, and `ae verify`.
- **OS keychain support & key redaction:** `auth.py` stores keys in the OS keychain (`keyring`) when available with `~/.config/alphaengine/credentials.json` (`0600`) as fallback, and `redact_secrets()` strips `ae_live_*` tokens from CLI output, error messages, tracebacks, and `--json` envelopes.
- **Canonical SHA-256 contract & golden fixture (`alphaengine/contracts/hash_golden.json`):** Unified `content_sha256`, study seals, `.alphaengine/objects/` CAS blobs, dataset hashes, and parameter hashes on `canonical_sha256()` (Schema 1.2 sorted-key JSON, `.8f` float formatting, 64-hex SHA-256). Added `verify_study_mode()` so studies sealed under the pre-1.2 legacy format still verify and are labeled `"legacy"`.
- **Free vs. Firm documentation (`docs/free-vs-firm.md`):** Documents the offline open-source library boundary, firm `ae_live_` Connect keys, and what uses the network.
- **Socket-blocked offline smoke test:** Replaced the import-only check in `tests/test_smoke.py` with a socket-blocked test (`socket.socket` and `socket.create_connection` patched to raise) that imports `alphaengine` and executes `deflated_sharpe` (DSR), `pbo_cscv` (PBO via CSCV), `combinatorial_oos_stability`, `performance_report`, `compute_var_cvar`, `sweep`, `load_csv`, `from_quantconnect`, and `from_vectorbt`.

### Changed
- **Cumulative project-wide trials & DSR everywhere:** Project-wide cumulative trial counts (`project_trials` across `.alphaengine/ledger/trials.jsonl`) now feed every DSR — run output, headline `deflated_sharpe` and `n_trials`, `ae push` payloads, and `ae seal` studies — with no per-run or per-family reset.
- **Identical re-run deduplication:** Re-running an identical configuration (`command`, `params_hash`, and `dataset_hashes` matching a prior run in `.alphaengine/ledger/runs.jsonl`) records `duplicate_of` and adds `0` new trials (`9 -> 9`), whereas distinct configurations always increment cumulative `N` (`9 -> 18`).
- **Removed bare `except Exception: pass` in `sweep/runner.py`:** `_maybe_record_project_sweep` now raises a descriptive `RuntimeError` when a discovered `.alphaengine/` project ledger cannot be written.
- **Metadata-driven `__version__` and naming cleanup:** `__version__` is loaded via `importlib.metadata.version("alphaengine")` with no hardcoded version strings; standardized `combinatorial_oos_stability` (with `cpcv_score` emitting `DeprecationWarning`) and `PBO via CSCV` across docs and top-level exports.

### Removed
- **Retail CLI path (`--retail` and `/api/retail/ledger/import`):** Removed `--retail` from `report` and CLI dispatch. Retail ledger import is performed via web upload in the quantOS portal.

## [1.1.0] - 2026-10-24

### Added
- **Project workspace (`ae init` and `ae` alias):** Initializes `.alphaengine/` (`project.toml`, `ledger/`, `data/`, `studies/`, `ideas/`, `universes/`, and SHA-256 content-addressed `objects/`). Commands walk up from the current directory to locate the project; documented in `docs/project-format.md`.
- **Project-wide ledger (`ae log`, `ae show`, `ae diff`, `ae rerun`):** Every run and `sweep()` inside a project writes an append-only run record and accumulates trial counts across runs on the same `idea_id` or `family`.
- **Dataset registry (`ae data add`, `ae data list`, `ae data show`, `ae data lineage`):** Registers CSV and Parquet datasets locally with full 64-hex SHA-256 hashes, row counts, date ranges, column schemas, and checks for duplicate timestamps, gaps, look-ahead rows (`--asof-col`), and survivorship hints.
- **Standard result envelope (`v1.0`), pipes, and declarative YAML pipelines:** Added `--json` and `--stdin` support across commands (`ae screen --json | ae validate --stdin | ae size --stdin`), `ae run <pipeline.yaml>`, two example pipelines in `examples/`, and `docs/result-format.md`.
- **Hypotheses as objects (`ae idea new`, `ae idea list`, `ae idea status`, `ae idea show`):** First-class hypothesis tracking with `--kill-if` rule evaluation, `--idea <id>` run linking, and pre-run prior-work checks.
- **Extended seal, verify, and committee memos:** `ae seal <run|idea>`, `ae verify <study.json> [--data <file>]`, and `ae report memo <idea|run> --format md|pdf` where every figure cites its `[run_id]`.
- **Research shell (`ae shell`):** Interactive shell with tab completion over verbs, run IDs, dataset names, and idea IDs, a project status line with cumulative trial count, inline ledger/idea commands, and automatic ledger recording of agent proposals.
- **Entry-point plugin system (`alphaengine.plugins`, `ae plugin list`):** Supports `loader`, `method`, and `template` plugins, with `quantconnect`, `vectorbt`, `csv`, `deflated_sharpe`, `combinatorial_oos_stability`, and `committee_memo` registered as built-ins (`docs/plugins.md`).
- **Local file watcher & live drift monitor:** `ae watch <pipeline.yaml>` re-runs pipelines when watched datasets change on disk; `ae monitor <run> --live <file>` compares live returns against the backtest's bootstrapped 95% intervals.
- **Opt-in workspace sync (`ae remote`, `ae push [--dry-run]`, `ae pull`):** Syncs sealed studies, allowlisted ledger runs, and ideas to `POST /api/workspace/sync` (with 404 fallback to `POST /api/me/runs/ingest`), enforcing the 512-element series cap and never uploading dataset rows.

## [1.0.3] - 2026-10-24

### Added
- **Sealed studies (Schema 1.2):** Canonical-JSON SHA-256 content hash (`content_sha256` with sorted keys and fixed 8-decimal float formatting), full 64-hex `data_hash`, and optional per-trial ledger export (`SweepResult.ledger()`, `include_ledger=True`).
- **CLI study verbs:** `alphaengine seal <study.json>` (writes `content_sha256` and verifies on re-run) and `alphaengine report <study.json> [--retail]` (reports to `/api/me/runs/ingest` by default or `/api/retail/ledger/import` with `--retail`).
- **Complete study ingest payload:** `study/report.py` `_payload()` now includes `selection` (PBO), `psr_vs_zero`, `sr0_expected_max`, `min_track_record_length`, `grid_keys`, `n_failed_trials`, `content_sha256`, and optional `assignment_id` and `thesis_id`.
- **Interop loaders (`alphaengine.io`):** `from_quantconnect(path)` and `from_vectorbt(portfolio_or_csv)` construct a `Study` with returns, performance metrics, data hash, and trial-count provenance (`"asserted"` when provided/present, `"not_recorded"` with verdict refusal when absent).
- **Credentials purge:** `alphaengine logout --purge` removes `~/.config/alphaengine/credentials.json`, cleans up its config directory, and clears managed keys from the current process.
- **API URL override:** `ALPHAENGINE_API_URL` environment variable and unified `DEFAULT_API_URL` constant across CLI and study reporting.
- **Security & governance:** Weekly Dependabot configuration (pip + GitHub Actions), Python CodeQL workflow, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and GitHub issue templates.

### Changed
- **Opt-in series sharing on `/traces`:** `_report_trace` sends scalar figures only by default. Uploading full series to `/api/harness/runs/{id}/traces` requires `--share-series` or `ALPHAENGINE_SHARE_SERIES=1`; any payload sent without opt-in is guarded at most 512 elements.
- **Deliberate math & golden updates:**
  - `deflated_sharpe(..., n_trials=1)` no longer silently deflates as `N=2`. At `n_trials=1`, `sr0_expected_max` is `0.0` (was `0.037` on the `rets(9, 200)` golden) and `deflated_sharpe` equals `psr_vs_zero` (`0.8434`, was `0.6875` on the golden), with an explicit note that no multiple-testing deflation applies.
  - `SweepResult.verdict()` passes the cross-trial Sharpe dispersion across valid grid trials into `deflated_sharpe` per Bailey & López de Prado (2014), falling back to single-series sampling error only when fewer than 2 valid trials exist.
  - Renamed `cpcv_score` to `combinatorial_oos_stability` (keeping `cpcv_score` as a deprecated alias) to accurately describe single-stream purged/embargoed sub-sample stability.
  - `pbo_cscv` validates `n_splits` (even integer $\ge 2$) and raises a clear `ValueError`.
- **Neutral framing:** Added *"Research tool, not investment advice"* disclaimer to README, `--help`, and the boot screen; renamed user-facing bakeoff `survivor` / `survivor_weights` to `top_ranked_config` / `model_weights` while preserving backward-compatible dictionary keys.
- **Release guard:** `.github/workflows/publish.yml` verifies that the git tag matches `v{__version__}` before building.

## [1.0.2] - 2026-10-20

### Added
- **Four-pass adversarial GAN bakeoff:** `run_bakeoff` and `core.bakeoff.referee` run four passes (eight candidate proposals across `equal`, `hrp`, `risk_parity`, `inverse_vol`, `var_sorted`, `low_vol`, `momentum`, and `reversion` against eight regime, tail, rate, and crash-cluster attacks) under a hard 12-call, 60-second ceiling.
- **Multi-pass bakeoff scorecard in PDF reports:** Cumulative wealth across passes, in-sample vs. post-attack Sharpe comparison, and candidate scorecard table.

## [1.0.1] - 2026-10-19

### Added
- **Harness simulation tools:** Added `monte_carlo` and `scenarios` to the local session tool stack.
- **Multi-page PDF & text reports:** Accepted session traces render multi-panel portfolio, scenario stress, Monte Carlo fan, and bakeoff reports to the user's `Downloads` folder.

### Fixed
- **Ledoit-Wolf sample-count divisor:** Fixed sample covariance scaling in `ledoit_wolf_cov` so `hrp_weights` and `risk_parity_weights` differentiate weights across wide panels.

## [1.0.0] - 2026-10-18

### Added
- **Interactive terminal simulator (`alphaengine`):** Full-screen Textual session where plain-English questions execute as a deterministic stack of local tool calls (`profile`, `align`, `regime`, `price`, `reprice`, `correlation`, `covariance`, `cointegration`, `portfolio`, `overlap`, `performance`, `var`, `process`, `workflow`, `bakeoff`) with citation guarding.
- **Pinned cash rate & statistical refinements:** Default risk-free rate pinned to the H.15 3-month Treasury constant maturity (`4.21%` as of `2026-10-06`, `DGS3MO`; pass `0` for excess of zero), automatic Newey-West HAC lag selection in factor regressions, and Benjamini-Hochberg FDR survivors in cointegration screening.
