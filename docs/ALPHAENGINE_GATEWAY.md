# AlphaEngine Gateway (`alphaengine` MCP Server)

This document provides the authoritative architectural specification and reference for the **AlphaEngine Model Context Protocol (MCP) Gateway** (`QOS-147`, `QOS-172`, `QOS-174`).

---

## 1. Gateway Architecture & Identity

- **Server Name:** `alphaengine`
- **Host / Deployment Endpoint:** `aemcp`
- **Engine Version:** `2.0.4`
- **Protocol:** Model Context Protocol (MCP) JSON-RPC over `stdio`, `sse`, or HTTP streaming.
- **Authentication:**
  - Deterministic calculations: 100% free, local, and keyless.
  - Account / Hosted tools: Authenticated via QuantOS firm API key (`ae_live_*`) or active session token verified server-side on every request (HTTP 401/403 on invalid keys). No API keys or tokens are ever logged or included in responses.
- **Data Source:** Market data references strictly use `yfinance` only. Zero third-party proprietary data vendors.

---

## 2. All 25 Audited MCP Tools

The AlphaEngine MCP gateway registers exactly **25 tools**: 22 deterministic quantitative computing tools and 3 workspace/firm account tools.

| Tool Name | Type | Inputs | Outputs | State & Storage | Description |
|:---|:---:|:---|:---|:---|:---|
| `backtest` | Deterministic | `signals, prices, slippage_bps=5.0, commission_bps=1.0, borrow_bps=50.0, fill_timing='close'` | `dates, equity_curve, returns, trades, final_equity, total_return_pct, cost_breakdown` | stateless (in-memory compute) | Deterministic weight-target portfolio simulator net of slippage, commission, borrow, and liquidation. |
| `score_backtest` | Deterministic | `bt (backtest dict), n_trials=None, pnl_matrix=None, risk_free_rate=None` | `performance, validation (dsr_probability, verdict, harvey_liu, min_track_record_length)` | stateless (in-memory compute) | Composes risk-adjusted performance with trial-adjusted deflation and returns canonical 2.0 Verdict. |
| `deflated_sharpe` | Deterministic | `returns, n_trials, trials_sharpe_std=None, risk_free_rate=None, matrix=None, n_eff=None` | `dsr_probability, verdict, sr0_expected_max, n_trials, n_eff, n_eff_method` | stateless (in-memory compute) | Deflated Sharpe Ratio probability for multiple testing and non-normal skew/kurtosis (Bailey & LdP). |
| `pbo_cscv` | Deterministic | `pnl_matrix (T x N), n_splits=10, max_combos=2000` | `pbo, n_partitions, n_configs, n_eff, n_eff_method, logit_mean, verdict` | stateless (in-memory compute) | Probability of Backtest Overfitting via Combinatorially Symmetric Cross-Validation (Bailey et al.). |
| `combinatorial_oos_stability` | Deterministic | `returns, n_groups=8, n_test_groups=2, purge=1, embargo=1, n_trials=1` | `sharpe_annualized (distribution), dsr_probability, verdict, pct_paths_positive` | stateless (in-memory compute) | Evaluates combinatorial out-of-sample stability with causal purge and embargo over held-out block paths. |
| `min_track_record_length` | Deterministic | `returns, sr_benchmark=0.0, confidence=0.95, risk_free_rate=None` | `min_track_record_length (bars), min_track_record_years, sufficient, shortfall_obs` | stateless (in-memory compute) | Calculates minimum track record length needed to distinguish skill from luck under skew and kurtosis. |
| `harvey_liu` | Deterministic | `sharpe_annualized, n_obs, n_trials, n_obs_effective=None, n_eff_participation=None` | `t_stat, p_value_single, bonferroni_p, holm_p, bhy_p, haircut_sharpe_holm, at_n_eff` | stateless (in-memory compute) | Harvey & Liu (2015) multiple-testing Sharpe ratio haircut under Bonferroni, Holm, and BHY adjustments. |
| `performance_report` | Deterministic | `returns, equity_curve=None, risk_free_rate=None, periods_per_year=252` | `sharpe_ratio, sharpe_annualized, sortino_ratio, calmar_ratio, max_drawdown_pct, CAGR` | stateless (in-memory compute) | Calculates standard risk-adjusted performance figures excess of the pinned 4.21% Treasury bill rate. |
| `var_cvar` | Deterministic | `returns, confidence=0.95 (or alpha=0.05), horizon_days=1, portfolio_value=100000` | `historical, parametric, cornish_fisher, cvar (all as loss pct), confidence, horizon_days` | stateless (in-memory compute) | Computes Value at Risk and Conditional Value at Risk (Expected Shortfall) as positive loss fractions. |
| `screen_universe` | Deterministic | `prices, universe=None, rank_by='return_pct' (or rule='momentum'), top_n=20, filters=None` | `rows (ranked shortlist), n_evaluated, n_passing, n_returned, n_insufficient` | stateless (in-memory compute) | Screens a universe of assets cross-sectionally without touching external networks. |
| `fama_macbeth` | Deterministic | `signal ({ticker: scores}), prices ({ticker: closes}), horizon=21` | `lambda_mean, lambda_by_date, t_stat, n_dates, n_names` | stateless (in-memory compute) | Two-pass Fama-MacBeth cross-sectional regression estimating factor risk premia and HAC t-statistics. |
| `quantile_book` | Deterministic | `signal ({ticker: scores}), prices ({ticker: closes}), horizon=21, quantiles=5` | `spread_pct, quantile returns, n_periods, n_names, turnover` | stateless (in-memory compute) | Sorts assets into quantile portfolios and evaluates bucket return spread and rank monotonicity. |
| `hrp` | Deterministic | `cov_matrix, names=None` | `weights, n_assets, max_weight, min_weight, condition_number` | stateless (in-memory compute) | Hierarchical Risk Parity portfolio allocation (Lopez de Prado 2016) via quasi-diagonalization. |
| `risk_parity` | Deterministic | `cov_matrix, names=None` | `weights, converged, n_assets, max_weight, min_weight, condition_number` | stateless (in-memory compute) | Equal risk contribution portfolio allocation balancing marginal volatility risk. |
| `vol_target` | Deterministic | `returns, target_vol=0.10, method='ewma', ewma_lambda=0.94, max_leverage=None` | `leverages, mean_leverage, n_obs, n_warmup, method` | stateless (in-memory compute) | Computes volatility-targeted dynamic leverage allocation scalar. |
| `denoise_cov` | Deterministic | `cov_matrix, n_obs=252 (or q=T/N)` | `cov (denoised), lambda_plus, n_signal_eigenvalues, n_noise_eigenvalues, q` | stateless (in-memory compute) | Marcenko-Pastur spectral truncation denoising of empirical sample covariance matrices. |
| `detone_cov` | Deterministic | `cov_matrix` | `detoned_cov` | stateless (in-memory compute) | Removes dominant market eigenvector mode from correlation matrix to isolate idiosyncratic risk. |
| `ewma_cov` | Deterministic | `returns_matrix (T x N), lambda_param=0.94` | `ewma_cov_matrix` | stateless (in-memory compute) | Exponentially weighted moving average covariance matrix estimation (RiskMetrics convention). |
| `dgp_stress` | Deterministic | `returns, dgp='gbm', n_paths=200 (or n_scenarios), seed=7` | `dgp, n_paths, sharpe_q10, sharpe_q50, sharpe_q90, frac_positive, sharpe_hist` | stateless (in-memory compute) | Data Generating Process parametric bootstrap and extreme scenario stress testing. |
| `ou_process` | Deterministic | `prices_or_spread, dt=1.0` | `kappa, theta, sigma, half_life, phi, mean_reverting` | stateless (in-memory compute) | Calibrates Ornstein-Uhlenbeck continuous-time mean-reverting process parameters via exact MLE. |
| `gbm_process` | Deterministic | `prices, kind=None` | `mu (drift), sigma (diffusion vol), n_obs` | stateless (in-memory compute) | Geometric Brownian Motion parameter calibration via log-return normal maximum likelihood. |
| `jump_process` | Deterministic | `returns, z=3.0, kind=None` | `mu, sigma, jump_lambda, jump_mu, jump_sigma, n_jumps_named` | stateless (in-memory compute) | Merton jump-diffusion process calibration identifying discontinuous market shock parameters. |
| `my_universe` | Account | `tickers=None (omitted to read, list of tickers to update), name='default'` | `tickers, count, updated_at` | persisted in researcher's workspace (.alphaengine/universes/) | Retrieves or updates the researcher's active asset universe. |
| `deliver_run` | Account | `run_id, summary (RunManifest dict)` | `run_id, content_sha256, delivered_at, status` | persisted in firm research ledger and audit trail | Uploads an allowlisted research summary and CAS manifest to the firm ledger. |
| `run_workflow` | Account | `workflow (or workflow_name), params=None` | `job_id, status, submitted_at` | persisted in firm execution queue and run history | Submits a validated quantOS research workflow to the hosted execution cluster. |

---

## 3. Storage & State Model (Stateless vs Account)

Contrary to outdated documentation claiming the entire gateway is "stateless":

1. **Deterministic Tools (`1–22`): 100% Stateless.**
   - All deterministic primitives (`backtest`, `score_backtest`, `deflated_sharpe`, `pbo_cscv`, `combinatorial_oos_stability`, `min_track_record_length`, `harvey_liu`, `performance_report`, `var_cvar`, `screen_universe`, `fama_macbeth`, `quantile_book`, `hrp`, `risk_parity`, `vol_target`, `denoise_cov`, `detone_cov`, `ewma_cov`, `dgp_stress`, `ou_process`, `gbm_process`, `jump_process`) execute pure in-memory computations on the host.
   - Nothing is persisted to disk, database, or external cache.

2. **Account Tools (`23–25`): Persisted Workspace Records.**
   - `my_universe`: Reads and writes the researcher's saved ticker list in `.alphaengine/universes/` in the researcher's workspace.
   - `deliver_run`: Uploads an allowlisted research summary and CAS `RunManifest` to the firm's shared research ledger (`.alphaengine/ledger/` and portal audit trail).
   - `run_workflow`: Submits an asynchronous workflow execution job to the hosted execution cluster and records the job ID in run history.

---

## 4. Engine Conventions & Output Standards

Every tool output returning numeric or qualitative evaluation conforms to AlphaEngine 2.0 standards:

1. **Live Engine Version Stamp:** Every figure and response object includes `"engine_version": "2.0.4"`.
2. **Unified Verdict Vocabulary:** Overfitting and validation tools emit the canonical 2.0 `Verdict` enum:
   - `robust` (DSR \(\ge 0.90\), PBO \(\le 0.20\), \(N > 1\))
   - `marginal` (\(0.50 \le \text{DSR} < 0.90\) or unassisted \(N \le 1\))
   - `likely_noise` (DSR \(< 0.50\) or PBO \(> 0.50\))
   - `unverified` (data insufficient for statistical evaluation)
3. **Cash-Rate Policy:** Excess returns subtract the pinned 4.21% annualized rate (DGS3MO bill) unless `risk_free_rate=0.0` is explicitly requested.
4. **Trial Accounting:** Effective independent trials (\(N_{\text{eff}}\)) via the eigenvalue participation ratio \((\sum \lambda_i)^2 / \sum \lambda_i^2\) are reported alongside raw \(N\) on correlated grids.

---

## 5. Scope Boundaries: What MCP Does NOT Do

- **No Python Source Upload:** The MCP gateway does not accept arbitrary Python code uploads or execute unvetted user code on the host.
- **Coded Strategy Redirection:** Coded target-weight strategies (`signal(*, data_asof, params)`) are evaluated locally through the AlphaEngine CLI (`ae run --signal`) or inside isolated, hardened execution containers.
