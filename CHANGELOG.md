# Changelog

All notable changes to `alphaengine` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to Semantic Versioning (with computed golden values
treated as part of the public contract).

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
