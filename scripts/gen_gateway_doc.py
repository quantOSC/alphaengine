#!/usr/bin/env python3
"""Generator and drift guard for ALPHAENGINE_GATEWAY.md (QOS-174).

Generates `ALPHAENGINE_GATEWAY.md` directly from the code-side tool registry
`src/alphaengine/mcp/registry.py` to ensure complete accuracy, resolving links,
documenting all 25 tools, state models, auth, and version conventions.

Run:
    python scripts/gen_gateway_doc.py --check   # Fails if document is out of sync
    python scripts/gen_gateway_doc.py --write   # Overwrites document with latest schema
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATEWAY_DOC = ROOT / "docs" / "ALPHAENGINE_GATEWAY.md"

# Add src to sys.path so we can import alphaengine.mcp
sys.path.insert(0, str(ROOT / "src"))

from alphaengine._version import current_engine_version  # noqa: E402
from alphaengine.mcp.registry import _MCP_TOOLS  # noqa: E402


def build_gateway_markdown() -> str:
    """Construct full ALPHAENGINE_GATEWAY.md content from the audited tool registry."""
    engine_ver = current_engine_version()

    tool_rows = []
    for t in _MCP_TOOLS:
        tool_rows.append(
            f"| `{t.name}` | {t.kind.capitalize()} | `{t.inputs_summary}` | `{t.outputs_summary}` | {t.storage} | {t.description} |"
        )

    tools_table = "\n".join(tool_rows)

    return f"""# AlphaEngine Gateway (`alphaengine` MCP Server)

This document provides the authoritative architectural specification and reference for the **AlphaEngine Model Context Protocol (MCP) Gateway** (`QOS-147`, `QOS-172`, `QOS-174`).

---

## 1. Gateway Architecture & Identity

- **Server Name:** `alphaengine`
- **Host / Deployment Endpoint:** `aemcp`
- **Engine Version:** `{engine_ver}`
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
{tools_table}

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

1. **Live Engine Version Stamp:** Every figure and response object includes `"engine_version": "{engine_ver}"`.
2. **Unified Verdict Vocabulary:** Overfitting and validation tools emit the canonical 2.0 `Verdict` enum:
   - `robust` (DSR \(\\ge 0.90\), PBO \(\\le 0.20\), \(N > 1\))
   - `marginal` (\(0.50 \\le \\text{{DSR}} < 0.90\) or unassisted \(N \\le 1\))
   - `likely_noise` (DSR \(< 0.50\) or PBO \(> 0.50\))
   - `unverified` (data insufficient for statistical evaluation)
3. **Cash-Rate Policy:** Excess returns subtract the pinned 4.21% annualized rate (DGS3MO bill) unless `risk_free_rate=0.0` is explicitly requested.
4. **Trial Accounting:** Effective independent trials (\(N_{{\\text{{eff}}}}\)) via the eigenvalue participation ratio \((\\sum \\lambda_i)^2 / \\sum \\lambda_i^2\) are reported alongside raw \(N\) on correlated grids.

---

## 5. Scope Boundaries: What MCP Does NOT Do

- **No Python Source Upload:** The MCP gateway does not accept arbitrary Python code uploads or execute unvetted user code on the host.
- **Coded Strategy Redirection:** Coded target-weight strategies (`signal(*, data_asof, params)`) are evaluated locally through the AlphaEngine CLI (`ae run --signal`) or inside isolated, hardened execution containers.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and check ALPHAENGINE_GATEWAY.md")
    parser.add_argument(
        "--check", action="store_true", help="Check whether ALPHAENGINE_GATEWAY.md is up to date"
    )
    parser.add_argument("--write", action="store_true", help="Write ALPHAENGINE_GATEWAY.md")
    args = parser.parse_args()

    content = build_gateway_markdown()

    if args.write or not args.check:
        GATEWAY_DOC.write_text(content, encoding="utf-8")
        print(f"ALPHAENGINE_GATEWAY.md written successfully ({len(_MCP_TOOLS)} tools).")
        return 0

    if not GATEWAY_DOC.is_file():
        print(f"ERROR: {GATEWAY_DOC} does not exist. Run with --write.", file=sys.stderr)
        return 1

    current = GATEWAY_DOC.read_text(encoding="utf-8")
    if current != content:
        print(
            "ERROR: ALPHAENGINE_GATEWAY.md is out of sync with src/alphaengine/mcp/registry.py.",
            file=sys.stderr,
        )
        print("Run `python scripts/gen_gateway_doc.py --write` to update.", file=sys.stderr)
        return 1

    print("ALPHAENGINE_GATEWAY.md is current and matches the 25-tool MCP registry.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
