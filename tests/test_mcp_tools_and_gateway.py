"""Tests for AlphaEngine MCP Tool Registry and Gateway Documentation (`QOS-172`, `QOS-174`)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from alphaengine._version import __version__, current_engine_version
from alphaengine.core import (
    compute_var_cvar,
    deflated_sharpe,
    gbm_calibrate,
    harvey_liu_haircut,
    jump_calibrate,
    min_track_record_length,
    ou_calibrate,
    performance_report,
    run_backtest,
    score_backtest,
)
from alphaengine.mcp.registry import (
    MCP_TOOLS_REGISTRY,
    call_mcp_tool,
    list_mcp_tools,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_mcp_tool_snapshot_count_is_exact_25() -> None:
    """Snapshot test verifying exactly 25 audited MCP tools."""
    tools = list_mcp_tools()
    assert len(tools) == 25, f"Expected exactly 25 tools, found {len(tools)}"

    deterministic = [t for t in tools if t["kind"] == "deterministic"]
    account = [t for t in tools if t["kind"] == "account"]

    assert len(deterministic) == 22, f"Expected 22 deterministic tools, found {len(deterministic)}"
    assert len(account) == 3, f"Expected 3 account tools, found {len(account)}"

    for tool in tools:
        assert tool["name"]
        assert tool["inputs"]
        assert tool["outputs"]
        assert tool["storage"]
        assert tool["description"]
        assert tool["engine_version"] == __version__
        assert tool["engine_version"] == "2.0.0"


def test_deterministic_tools_parity_with_alphaengine_core() -> None:
    """Parity test: deterministic MCP tools produce identical results to direct alphaengine calls."""
    rng = np.random.default_rng(42)
    sample_returns = rng.normal(0.0005, 0.01, 252).tolist()
    sample_prices = (100.0 * np.exp(np.cumsum(sample_returns))).tolist()

    # 1. Deflated Sharpe
    dsr_mcp = call_mcp_tool("deflated_sharpe", returns=sample_returns, n_trials=10)
    dsr_direct = deflated_sharpe(sample_returns, n_trials=10)
    assert dsr_mcp["dsr_probability"] == pytest.approx(dsr_direct["dsr_probability"])
    assert dsr_mcp["verdict"] == dsr_direct["verdict"]
    assert dsr_mcp["engine_version"] == current_engine_version()

    # 2. Performance Report
    perf_mcp = call_mcp_tool("performance_report", returns=sample_returns)
    perf_direct = performance_report(sample_returns)
    assert perf_mcp["sharpe_ratio"] == pytest.approx(perf_direct["sharpe_ratio"])
    assert perf_mcp["sharpe_annualized"] == pytest.approx(perf_direct["sharpe_annualized"])
    assert perf_mcp["engine_version"] == current_engine_version()

    # 3. VaR / CVaR
    var_mcp = call_mcp_tool("var_cvar", returns=sample_returns)
    var_direct = compute_var_cvar(sample_returns)
    assert var_mcp["confidence"] == pytest.approx(var_direct["confidence"])
    assert var_mcp["cvar"]["cvar_pct"] == pytest.approx(var_direct["cvar"]["cvar_pct"])
    assert var_mcp["parametric"]["var_pct"] == pytest.approx(var_direct["parametric"]["var_pct"])
    assert var_mcp["engine_version"] == current_engine_version()

    # 4. Minimum Track Record Length
    min_mcp = call_mcp_tool("min_track_record_length", returns=sample_returns)
    min_direct = min_track_record_length(sample_returns)
    assert min_mcp["min_track_record_length"] == pytest.approx(min_direct["min_track_record_length"])
    assert min_mcp["engine_version"] == current_engine_version()

    # 5. Harvey-Liu Haircut
    hl_mcp = call_mcp_tool("harvey_liu", sharpe_annualized=1.5, n_obs=252, n_trials=10)
    hl_direct = harvey_liu_haircut(1.5, n_obs=252, n_trials=10)
    assert hl_mcp["haircut_sharpe_holm"] == pytest.approx(hl_direct["haircut_sharpe_holm"])
    assert hl_mcp["t_stat"] == pytest.approx(hl_direct["t_stat"])
    assert hl_mcp["engine_version"] == current_engine_version()

    # 6. Backtest & Score Backtest
    signals = {"SPY": [0.5] * 60}
    prices = {"SPY": [100.0 + i for i in range(60)]}
    bt_mcp = call_mcp_tool("backtest", signals=signals, prices=prices)
    bt_direct = run_backtest(signals, prices)
    assert bt_mcp["final_equity"] == pytest.approx(bt_direct["final_equity"])
    assert bt_mcp["engine_version"] == current_engine_version()

    score_mcp = call_mcp_tool("score_backtest", bt=bt_mcp)
    score_direct = score_backtest(bt_direct)
    assert score_mcp["performance"]["sharpe_annualized"] == pytest.approx(
        score_direct["performance"]["sharpe_annualized"]
    )
    assert score_mcp["engine_version"] == current_engine_version()

    # 7. Processes (OU, GBM, Jump)
    ou_mcp = call_mcp_tool("ou_process", prices_or_spread=sample_prices)
    ou_direct = ou_calibrate(sample_prices)
    assert ou_mcp["theta"] == pytest.approx(ou_direct["theta"])
    assert ou_mcp["engine_version"] == current_engine_version()

    gbm_mcp = call_mcp_tool("gbm_process", prices=sample_prices)
    gbm_direct = gbm_calibrate(sample_prices)
    assert gbm_mcp["mu"] == pytest.approx(gbm_direct["mu"])
    assert gbm_mcp["sigma"] == pytest.approx(gbm_direct["sigma"])
    assert gbm_mcp["engine_version"] == current_engine_version()

    jump_mcp = call_mcp_tool("jump_process", returns=sample_returns)
    jump_direct = jump_calibrate(sample_returns)
    assert jump_mcp["mu"] == pytest.approx(jump_direct["mu"])
    assert jump_mcp["sigma"] == pytest.approx(jump_direct["sigma"])
    assert jump_mcp["engine_version"] == current_engine_version()


def test_account_tools_reporting_and_version_stamp() -> None:
    """Account tools stamp engine_version and document persistence destination."""
    univ = call_mcp_tool("my_universe", tickers=["AAPL", "MSFT"])
    assert univ["engine_version"] == "2.0.0"
    assert univ["tickers"] == ["AAPL", "MSFT"]
    assert "workspace" in univ["stored"]

    deliv = call_mcp_tool("deliver_run", run_id="run_123", summary={"manifest_sha256": "abc"})
    assert deliv["engine_version"] == "2.0.0"
    assert deliv["run_id"] == "run_123"
    assert "ledger" in deliv["stored"]

    wf = call_mcp_tool("run_workflow", workflow="momentum_sweep")
    assert wf["engine_version"] == "2.0.0"
    assert wf["workflow"] == "momentum_sweep"
    assert "queue" in wf["stored"]


def test_gateway_doc_matches_registry_without_drift() -> None:
    """Verify scripts/gen_gateway_doc.py --check passes cleanly."""
    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "gen_gateway_doc.py"), "--check"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"gen_gateway_doc.py --check failed: {res.stdout}\n{res.stderr}"


def test_gateway_doc_integrity_links_and_zero_secrets() -> None:
    """Verify ALPHAENGINE_GATEWAY.md and mcp-server/README.md resolve links and leak no secrets."""
    gw_path = REPO_ROOT / "ALPHAENGINE_GATEWAY.md"
    assert gw_path.exists(), "ALPHAENGINE_GATEWAY.md does not exist"
    content = gw_path.read_text(encoding="utf-8")

    # All 25 tool names must appear in markdown
    for tool_name in MCP_TOOLS_REGISTRY:
        assert f"`{tool_name}`" in content or tool_name in content, f"Missing {tool_name} in gateway doc"

    # Verify no hardcoded secrets or API key values
    forbidden_patterns = [
        r"ae_live_[a-zA-Z0-9]{16,}",
        r"sk-ant-[a-zA-Z0-9]{16,}",
        r"bearer\s+[a-zA-Z0-9_\-\.]{20,}",
    ]
    for pat in forbidden_patterns:
        match = re.search(pat, content, re.IGNORECASE)
        assert match is None, f"Found secret pattern {pat} in ALPHAENGINE_GATEWAY.md: {match.group(0)}"

    # Check mcp-server/README.md
    mcp_readme = REPO_ROOT / "mcp-server" / "README.md"
    assert mcp_readme.exists(), "mcp-server/README.md does not exist"
    mcp_content = mcp_readme.read_text(encoding="utf-8")
    assert "ALPHAENGINE_GATEWAY.md" in mcp_content
    assert "25 Tools" in mcp_content or "25 tools" in mcp_content
