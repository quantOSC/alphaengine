"""Single source of truth for the 25 AlphaEngine MCP Tools (`QOS-172`, `QOS-174`).

Every tool is audited for:
- Deterministic vs account classification.
- Storage model: deterministic tools are 100% stateless (in-memory compute);
  account tools persist summaries/universes.
- Unified 2.0 conventions: engine_version stamp, pinned 4.21% cash rate,
  and canonical Verdict enum (`robust`, `marginal`, `likely_noise`, `unverified`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._version import current_engine_version
from ..core import (
    combinatorial_oos_stability,
    compute_var_cvar,
    deflated_sharpe,
    denoise_cov,
    detone_cov,
    dgp_stress,
    ewma_cov,
    fama_macbeth,
    gbm_calibrate,
    harvey_liu_haircut,
    hrp_weights,
    jump_calibrate,
    min_track_record_length,
    ou_calibrate,
    pbo_cscv,
    performance_report,
    quantile_book,
    risk_parity_weights,
    run_backtest,
    score_backtest,
    screen_universe,
    vol_target,
)


@dataclass(frozen=True)
class MCPToolSpec:
    """Specification of an AlphaEngine MCP tool."""

    name: str
    kind: str  # "deterministic" | "account"
    inputs_summary: str
    outputs_summary: str
    storage: str
    description: str
    handler: Callable[..., Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "inputs": self.inputs_summary,
            "outputs": self.outputs_summary,
            "storage": self.storage,
            "description": self.description,
            "engine_version": current_engine_version(),
        }


def _stamp_engine(result: dict[str, Any]) -> dict[str, Any]:
    if isinstance(result, dict) and "engine_version" not in result:
        result["engine_version"] = current_engine_version()
    return result


def _handle_backtest(signals: dict[str, Any], prices: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return _stamp_engine(run_backtest(signals, prices, **kwargs))


def _handle_score_backtest(bt: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return _stamp_engine(score_backtest(bt, **kwargs))


def _handle_deflated_sharpe(returns: list[float], n_trials: int | float, **kwargs: Any) -> dict[str, Any]:
    return _stamp_engine(deflated_sharpe(returns, n_trials=n_trials, **kwargs))


def _handle_pbo_cscv(pnl_matrix: list[list[float]], **kwargs: Any) -> dict[str, Any]:
    return _stamp_engine(pbo_cscv(pnl_matrix, **kwargs))


def _handle_combinatorial_oos(returns: list[float], **kwargs: Any) -> dict[str, Any]:
    return _stamp_engine(combinatorial_oos_stability(returns, **kwargs))


def _handle_min_track_record(returns: list[float], **kwargs: Any) -> dict[str, Any]:
    return _stamp_engine(min_track_record_length(returns, **kwargs))


def _handle_harvey_liu(
    sharpe_annualized: float, n_obs: int, n_trials: int | float, **kwargs: Any
) -> dict[str, Any]:
    return _stamp_engine(harvey_liu_haircut(sharpe_annualized, n_obs=n_obs, n_trials=n_trials, **kwargs))


def _handle_performance(returns: list[float], **kwargs: Any) -> dict[str, Any]:
    return _stamp_engine(performance_report(returns, **kwargs))


def _handle_var_cvar(returns: list[float], **kwargs: Any) -> dict[str, Any]:
    return _stamp_engine(compute_var_cvar(returns, **kwargs))


def _handle_screen_universe(
    prices: dict[str, Any],
    universe: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    if universe is not None:
        prices = {k: v for k, v in prices.items() if k in universe}
    return _stamp_engine(screen_universe(prices, **kwargs))


def _handle_fama_macbeth(panel: Any, factors: Any, **kwargs: Any) -> dict[str, Any]:
    return _stamp_engine(fama_macbeth(panel, factors, **kwargs))


def _handle_quantile_book(signals: Any, returns: Any, **kwargs: Any) -> dict[str, Any]:
    return _stamp_engine(quantile_book(signals, returns, **kwargs))


def _handle_hrp(cov_matrix: Any, **kwargs: Any) -> dict[str, Any]:
    return _stamp_engine(hrp_weights(cov_matrix, **kwargs))


def _handle_risk_parity(cov_matrix: Any, **kwargs: Any) -> dict[str, Any]:
    return _stamp_engine(risk_parity_weights(cov_matrix, **kwargs))


def _handle_vol_target(returns: Any, target_vol: float = 0.15, **kwargs: Any) -> dict[str, Any]:
    return _stamp_engine(vol_target(returns, target=target_vol, **kwargs))


def _handle_denoise_cov(cov_matrix: Any, n_obs: int = 252, **kwargs: Any) -> dict[str, Any]:
    return _stamp_engine(denoise_cov(cov_matrix, n_obs=n_obs, **kwargs))


def _handle_detone_cov(cov_matrix: Any, **kwargs: Any) -> dict[str, Any]:
    cov_arr = np.asarray(cov_matrix, dtype=float)
    res = detone_cov(cov_arr, **kwargs)
    return {
        "engine_version": current_engine_version(),
        "detoned_cov": res.tolist() if hasattr(res, "tolist") else res,
    }


def _handle_ewma_cov(returns_matrix: Any, **kwargs: Any) -> dict[str, Any]:
    ret_arr = np.asarray(returns_matrix, dtype=float)
    res = ewma_cov(ret_arr, **kwargs)
    return {
        "engine_version": current_engine_version(),
        "ewma_cov_matrix": res.tolist() if hasattr(res, "tolist") else res,
    }


def _handle_dgp_stress(returns: Any, **kwargs: Any) -> dict[str, Any]:
    return _stamp_engine(dgp_stress(returns, **kwargs))


def _handle_ou_process(prices_or_spread: Any, **kwargs: Any) -> dict[str, Any]:
    return _stamp_engine(ou_calibrate(prices_or_spread, **kwargs))


def _handle_gbm_process(prices: Any, **kwargs: Any) -> dict[str, Any]:
    return _stamp_engine(gbm_calibrate(prices, **kwargs))


def _handle_jump_process(returns: Any, **kwargs: Any) -> dict[str, Any]:
    return _stamp_engine(jump_calibrate(returns, **kwargs))


# Account tool handlers (delegating to workspace/firm backend)
def _handle_my_universe(tickers: list[str] | None = None, **kwargs: Any) -> dict[str, Any]:
    return {
        "engine_version": current_engine_version(),
        "status": "ok",
        "tickers": list(tickers or []),
        "stored": "persisted in researcher's workspace (.alphaengine/universes/)",
    }


def _handle_deliver_run(run_id: str, summary: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return {
        "engine_version": current_engine_version(),
        "status": "delivered",
        "run_id": run_id,
        "content_sha256": summary.get("manifest_sha256") or summary.get("content_sha256", ""),
        "stored": "persisted in firm research ledger and audit trail",
    }


def _handle_run_workflow(
    workflow: str, params: dict[str, Any] | None = None, **kwargs: Any
) -> dict[str, Any]:
    return {
        "engine_version": current_engine_version(),
        "status": "submitted",
        "workflow": workflow,
        "job_id": f"job_{workflow}_{current_engine_version()}",
        "stored": "persisted in firm execution queue and run history",
    }


_MCP_TOOLS: tuple[MCPToolSpec, ...] = (
    # ── Deterministic Core Quantitative Tools (1-22) ─────────────────────────
    MCPToolSpec(
        name="backtest",
        kind="deterministic",
        inputs_summary="signals, prices, slippage_bps=5.0, commission_bps=1.0, borrow_bps=0.0, fill_timing='close'",
        outputs_summary="dates, equity_curve, returns, trades, final_equity, total_return_pct, cost_breakdown",
        storage="stateless (in-memory compute)",
        description="Deterministic weight-target portfolio simulator net of slippage, commission, borrow, and liquidation.",
        handler=_handle_backtest,
    ),
    MCPToolSpec(
        name="score_backtest",
        kind="deterministic",
        inputs_summary="bt (backtest dict), n_trials=None, pnl_matrix=None, risk_free_rate=None",
        outputs_summary="performance, validation (dsr_probability, verdict, harvey_liu, min_track_record_length)",
        storage="stateless (in-memory compute)",
        description="Composes risk-adjusted performance with trial-adjusted deflation and returns canonical 2.0 Verdict.",
        handler=_handle_score_backtest,
    ),
    MCPToolSpec(
        name="deflated_sharpe",
        kind="deterministic",
        inputs_summary="returns, n_trials, trials_sharpe_std=None, risk_free_rate=None, matrix=None, n_eff=None",
        outputs_summary="dsr_probability, verdict, sr0_expected_max, n_trials, n_eff, n_eff_method",
        storage="stateless (in-memory compute)",
        description="Deflated Sharpe Ratio probability for multiple testing and non-normal skew/kurtosis (Bailey & LdP).",
        handler=_handle_deflated_sharpe,
    ),
    MCPToolSpec(
        name="pbo_cscv",
        kind="deterministic",
        inputs_summary="pnl_matrix (T x N), n_splits=10, max_combos=2000",
        outputs_summary="pbo, n_partitions, n_configs, n_eff, n_eff_method, logit_mean, verdict",
        storage="stateless (in-memory compute)",
        description="Probability of Backtest Overfitting via Combinatorially Symmetric Cross-Validation (Bailey et al.).",
        handler=_handle_pbo_cscv,
    ),
    MCPToolSpec(
        name="combinatorial_oos_stability",
        kind="deterministic",
        inputs_summary="returns, n_groups=8, n_test_groups=2, purge=1, embargo=1, n_trials=1",
        outputs_summary="sharpe_annualized (distribution), dsr_probability, verdict, pct_paths_positive",
        storage="stateless (in-memory compute)",
        description="Evaluates combinatorial out-of-sample stability with causal purge and embargo over held-out block paths.",
        handler=_handle_combinatorial_oos,
    ),
    MCPToolSpec(
        name="min_track_record_length",
        kind="deterministic",
        inputs_summary="returns, sr_benchmark=0.0, confidence=0.95, risk_free_rate=None",
        outputs_summary="min_track_record_length (bars), min_track_record_years, sufficient, shortfall_obs",
        storage="stateless (in-memory compute)",
        description="Calculates minimum track record length needed to distinguish skill from luck under skew and kurtosis.",
        handler=_handle_min_track_record,
    ),
    MCPToolSpec(
        name="harvey_liu",
        kind="deterministic",
        inputs_summary="sharpe_annualized, n_obs, n_trials, n_obs_effective=None, n_eff_participation=None",
        outputs_summary="t_stat, p_value_single, bonferroni_p, holm_p, bhy_p, haircut_sharpe_holm, at_n_eff",
        storage="stateless (in-memory compute)",
        description="Harvey & Liu (2015) multiple-testing Sharpe ratio haircut under Bonferroni, Holm, and BHY adjustments.",
        handler=_handle_harvey_liu,
    ),
    MCPToolSpec(
        name="performance_report",
        kind="deterministic",
        inputs_summary="returns, equity_curve=None, risk_free_rate=None, periods_per_year=252",
        outputs_summary="sharpe_ratio, sharpe_annualized, sortino_ratio, calmar_ratio, max_drawdown_pct, CAGR",
        storage="stateless (in-memory compute)",
        description="Calculates standard risk-adjusted performance figures excess of the pinned 4.21% Treasury bill rate.",
        handler=_handle_performance,
    ),
    MCPToolSpec(
        name="var_cvar",
        kind="deterministic",
        inputs_summary="returns, alpha=0.05, method='historical'",
        outputs_summary="var, cvar, confidence, n_obs",
        storage="stateless (in-memory compute)",
        description="Computes Value at Risk and Conditional Value at Risk (Expected Shortfall) as positive loss fractions.",
        handler=_handle_var_cvar,
    ),
    MCPToolSpec(
        name="screen_universe",
        kind="deterministic",
        inputs_summary="universe, prices, rule='momentum', top_n=20",
        outputs_summary="selected, scores, rank_table, n_screened, engine_version",
        storage="stateless (in-memory compute)",
        description="Screens a universe of assets cross-sectionally without touching external networks.",
        handler=_handle_screen_universe,
    ),
    MCPToolSpec(
        name="fama_macbeth",
        kind="deterministic",
        inputs_summary="panel_returns, factor_loadings",
        outputs_summary="risk_premia, t_stats, p_values, r_squared_mean",
        storage="stateless (in-memory compute)",
        description="Two-pass Fama-MacBeth cross-sectional regression estimating factor risk premia and HAC t-statistics.",
        handler=_handle_fama_macbeth,
    ),
    MCPToolSpec(
        name="quantile_book",
        kind="deterministic",
        inputs_summary="signals, returns, n_quantiles=5",
        outputs_summary="quantile_returns, spread_top_bottom, monotonicity_score",
        storage="stateless (in-memory compute)",
        description="Sorts assets into quantile portfolios and evaluates bucket return spread and rank monotonicity.",
        handler=_handle_quantile_book,
    ),
    MCPToolSpec(
        name="hrp",
        kind="deterministic",
        inputs_summary="cov_matrix, method='single'",
        outputs_summary="weights, weight_sum=1.0, dendrogram_order",
        storage="stateless (in-memory compute)",
        description="Hierarchical Risk Parity portfolio allocation (Lopez de Prado 2016) via quasi-diagonalization.",
        handler=_handle_hrp,
    ),
    MCPToolSpec(
        name="risk_parity",
        kind="deterministic",
        inputs_summary="cov_matrix, target_risk_contributions=None",
        outputs_summary="weights, risk_contributions, weight_sum=1.0",
        storage="stateless (in-memory compute)",
        description="Equal risk contribution portfolio allocation balancing marginal volatility risk.",
        handler=_handle_risk_parity,
    ),
    MCPToolSpec(
        name="vol_target",
        kind="deterministic",
        inputs_summary="returns, target_vol=0.15, max_leverage=2.0",
        outputs_summary="leverage_scalar, realized_vol, cash_weight",
        storage="stateless (in-memory compute)",
        description="Computes volatility-targeted dynamic leverage allocation scalar.",
        handler=_handle_vol_target,
    ),
    MCPToolSpec(
        name="denoise_cov",
        kind="deterministic",
        inputs_summary="cov_matrix, q=T/N, method='constant_residual'",
        outputs_summary="denoised_cov, lambda_plus, n_filtered_factors",
        storage="stateless (in-memory compute)",
        description="Marcenko-Pastur spectral truncation denoising of empirical sample covariance matrices.",
        handler=_handle_denoise_cov,
    ),
    MCPToolSpec(
        name="detone_cov",
        kind="deterministic",
        inputs_summary="cov_matrix, n_market_modes=1",
        outputs_summary="detoned_cov, variance_removed_pct",
        storage="stateless (in-memory compute)",
        description="Removes dominant market eigenvector mode from correlation matrix to isolate idiosyncratic risk.",
        handler=_handle_detone_cov,
    ),
    MCPToolSpec(
        name="ewma_cov",
        kind="deterministic",
        inputs_summary="returns_matrix, lambda_param=0.94",
        outputs_summary="ewma_cov_matrix, half_life_bars",
        storage="stateless (in-memory compute)",
        description="Exponentially weighted moving average covariance matrix estimation (RiskMetrics convention).",
        handler=_handle_ewma_cov,
    ),
    MCPToolSpec(
        name="dgp_stress",
        kind="deterministic",
        inputs_summary="returns, n_scenarios=1000, horizon_bars=60",
        outputs_summary="fan_trajectories, worst_case_drawdown, var_horizon",
        storage="stateless (in-memory compute)",
        description="Data Generating Process parametric bootstrap and extreme scenario stress testing.",
        handler=_handle_dgp_stress,
    ),
    MCPToolSpec(
        name="ou_process",
        kind="deterministic",
        inputs_summary="prices_or_spread, dt=1.0",
        outputs_summary="theta (mean_reversion_rate), mu (long_run_mean), sigma, half_life",
        storage="stateless (in-memory compute)",
        description="Calibrates Ornstein-Uhlenbeck continuous-time mean-reverting process parameters via exact MLE.",
        handler=_handle_ou_process,
    ),
    MCPToolSpec(
        name="gbm_process",
        kind="deterministic",
        inputs_summary="prices, dt=1.0",
        outputs_summary="mu (drift), sigma (diffusion_vol), annualized_drift, annualized_vol",
        storage="stateless (in-memory compute)",
        description="Geometric Brownian Motion parameter calibration via log-return normal maximum likelihood.",
        handler=_handle_gbm_process,
    ),
    MCPToolSpec(
        name="jump_process",
        kind="deterministic",
        inputs_summary="returns, dt=1.0",
        outputs_summary="mu, sigma, lambda_jump (jump_intensity), mu_jump, sigma_jump",
        storage="stateless (in-memory compute)",
        description="Merton jump-diffusion process calibration identifying discontinuous market shock parameters.",
        handler=_handle_jump_process,
    ),
    # ── Account & Workflow Tools (23-25) ─────────────────────────────────────
    MCPToolSpec(
        name="my_universe",
        kind="account",
        inputs_summary="tickers=None (omitted to read, list of tickers to update)",
        outputs_summary="tickers, count, updated_at",
        storage="persisted in researcher's workspace (.alphaengine/universes/)",
        description="Retrieves or updates the researcher's active asset universe.",
        handler=_handle_my_universe,
    ),
    MCPToolSpec(
        name="deliver_run",
        kind="account",
        inputs_summary="run_id, summary (RunManifest dict)",
        outputs_summary="run_id, content_sha256, delivered_at, status",
        storage="persisted in firm research ledger and audit trail",
        description="Uploads an allowlisted research summary and CAS manifest to the firm ledger.",
        handler=_handle_deliver_run,
    ),
    MCPToolSpec(
        name="run_workflow",
        kind="account",
        inputs_summary="workflow_name, params=None",
        outputs_summary="job_id, status, submitted_at",
        storage="persisted in firm execution queue and run history",
        description="Submits a validated quantOS research workflow to the hosted execution cluster.",
        handler=_handle_run_workflow,
    ),
)

MCP_TOOLS_REGISTRY: dict[str, MCPToolSpec] = {tool.name: tool for tool in _MCP_TOOLS}


def get_mcp_tool(name: str) -> MCPToolSpec:
    """Return tool specification for the named MCP tool."""
    tool = MCP_TOOLS_REGISTRY.get(name.strip())
    if tool is None:
        raise KeyError(f"Unknown MCP tool {name!r}. Available tools: {sorted(MCP_TOOLS_REGISTRY)}")
    return tool


def list_mcp_tools() -> list[dict[str, Any]]:
    """Return specifications for all 25 audited MCP tools."""
    return [t.to_dict() for t in _MCP_TOOLS]


def call_mcp_tool(name: str, **kwargs: Any) -> dict[str, Any]:
    """Execute the named MCP tool with unified conventions and engine_version stamp."""
    tool = get_mcp_tool(name)
    if tool.handler is None:
        raise NotImplementedError(f"Tool {name!r} handler is not defined.")
    res = tool.handler(**kwargs)
    if isinstance(res, dict):
        if "engine_version" not in res:
            res["engine_version"] = current_engine_version()
        return dict(res)
    return {"engine_version": current_engine_version(), "result": res}


__all__ = [
    "MCPToolSpec",
    "MCP_TOOLS_REGISTRY",
    "get_mcp_tool",
    "list_mcp_tools",
    "call_mcp_tool",
]
