# AlphaEngine MCP Server

The AlphaEngine Model Context Protocol (MCP) server exposes validated quantitative financial analytics directly to LLM agents and clients (host `aemcp`, server name `alphaengine`).

For the comprehensive tool reference, architecture specification, and security invariants, see [ALPHAENGINE_GATEWAY.md](../docs/ALPHAENGINE_GATEWAY.md).

## Architecture Overview

* **Engine:** Powered canonically by `alphaengine==2.0.4`. All quantitative computations run directly through `alphaengine.core` and `alphaengine.mcp.registry`.
* **25 Tools Registered:**
  - **22 Deterministic In-Memory Tools:** Fully stateless compute engines (`backtest`, `score_backtest`, `deflated_sharpe`, `pbo_cscv`, `combinatorial_oos_stability`, `min_track_record_length`, `harvey_liu`, `performance_report`, `var_cvar`, `screen_universe`, `fama_macbeth`, `quantile_book`, `hrp`, `risk_parity`, `vol_target`, `denoise_cov`, `detone_cov`, `ewma_cov`, `dgp_stress`, `ou_process`, `gbm_process`, `jump_process`). Zero disk writes or session persistence.
  - **3 Firm Account Tools:** Connected to workspace persistence and firm ledger (`my_universe`, `deliver_run`, `run_workflow`). Stores universe tracking, delivered runs, and workflow job states under `.alphaengine/`.
* **Output Conventions:** Every figure-generating tool outputs standardized dictionary contracts containing `engine_version: "2.0.4"` and canonical `verdict` enums (`robust`, `marginal`, `likely_noise`, `unverified`).
* **Conventions & Risk-Free Rate:** Standardized DGS3MO benchmark rate across all Sharpe ratio calculations.
* **Data Source:** `yfinance` only.
* **Security & Non-Goals:**
  - No Python source upload or remote code execution.
  - Coded workflows are strictly directed to the local CLI (`ae run --signal`) or the isolated runner.
  - Zero secrets or client keys are transmitted or logged.

## Installation & Pinned Dependency

The MCP server environment installs `alphaengine` pinned with exact SHA256 wheel hashes:

```bash
pip install -r constraints/engine.txt
```

On server boot, the process verifies `alphaengine.__version__ == "2.0.4"` before registering tools.
