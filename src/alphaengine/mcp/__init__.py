"""AlphaEngine MCP Server module (`QOS-172`, `QOS-174`).

Exposes all 25 audited tools with unified 2.0 conventions (4.21% cash rate,
`Verdict` vocabulary, live `engine_version` stamp, and zero state on deterministic tools).
"""

from .registry import (
    MCP_TOOLS_REGISTRY,
    MCPToolSpec,
    call_mcp_tool,
    get_mcp_tool,
    list_mcp_tools,
)

__all__ = [
    "MCPToolSpec",
    "MCP_TOOLS_REGISTRY",
    "get_mcp_tool",
    "list_mcp_tools",
    "call_mcp_tool",
]
