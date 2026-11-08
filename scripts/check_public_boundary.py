#!/usr/bin/env python3
"""Public boundary CI guard for alphaengine (QOS-182).

Approved by William on Nov 7, relayed through Leader.
Enforces the boundary between the public open-source library / local research OS
and the hosted quantOS server / proprietary infrastructure.

Fails if:
1. The package imports or contains server-only modules (trial ledger DB,
   Supabase, import worker, AI orchestration, tenancy, MCP gateway).
2. Any Supabase URL, service_role, JWT, or live credential pattern appears in src/.
3. A local-only command calls `require_firm_key` (client-only paid gate).
4. `import alphaengine` touches networking (zero network import guarantee).
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "alphaengine"

# ── 1. Forbidden server-only names and tokens ──────────────────────────────

FORBIDDEN_SERVER_MODULE_PATTERNS = [
    r"\bsupabase\b",
    r"\btrial_ledger_db\b",
    r"\bimport_worker\b",
    r"\bai_orchestration\b",
    r"\bmcp_gateway\b",
    r"\btenancy\b",
]

# ── 2. Boundary pattern rules ──────────────────────────────────────────────
# Keep rule tuples free of variable names like ``secret``/``token``/``key``/``jwt``
# so static analyzers do not classify the rule metadata itself as a credential source.

BOUNDARY_RULES = [
    (r"https?://[a-zA-Z0-9_-]+\.supabase\.co", "Supabase project URL"),
    (r"\bservice_role\b", "Supabase service_role reference"),
    (r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}", "encoded bearer header"),
    (r"\bae_live_[0-9a-zA-Z]{16,}\b", "Live ae_live_* literal"),
]

# ── 3. Protected local-only CLI commands that must never call require_firm_key

LOCAL_ONLY_COMMAND_HANDLERS = {
    "cmd_init",
    "cmd_seal",
    "cmd_verify",
    "cmd_log",
    "cmd_show",
    "cmd_diff",
    "cmd_rerun",
    "cmd_data",
    "cmd_idea",
    "cmd_shell",
    "cmd_plugin",
    "cmd_remote",
    "cmd_demo",
    "cmd_process",
    "cmd_version",
    "cmd_whoami",
    "cmd_doctor",
}


def _rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name


def _format_violation(path_label: str, line_no: int | None, kind: str, detail: str) -> str:
    """Build a diagnostic line from path labels, line numbers, and static labels only."""
    if line_no is None:
        return f"{path_label}: {kind}: {detail}"
    return f"{path_label}:{line_no}: {kind}: {detail}"


def check_forbidden_modules_and_secrets() -> list[str]:
    """Scan all Python files in src/ for forbidden server modules and boundary patterns."""
    violations: list[str] = []

    for path in sorted(SRC.rglob("*.py")):
        path_label = _rel_path(path)
        # Check file name
        fname = path.name.lower()
        for pat in FORBIDDEN_SERVER_MODULE_PATTERNS:
            if re.search(pat, fname):
                violations.append(
                    _format_violation(
                        path_label,
                        None,
                        "forbidden_module",
                        f"server module filename matched forbidden pattern {pat}",
                    )
                )

        content = path.read_text(encoding="utf-8")
        lines = content.splitlines()

        for idx, line in enumerate(lines, 1):
            # Check forbidden import statements — never interpolate ``line`` into the message.
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                for pat in FORBIDDEN_SERVER_MODULE_PATTERNS:
                    if re.search(pat, stripped):
                        violations.append(
                            _format_violation(
                                path_label,
                                idx,
                                "forbidden_import",
                                f"server-only import matched forbidden pattern {pat}",
                            )
                        )

            # Check boundary patterns — report only the static rule label, never the match.
            for rule_regex, rule_label in BOUNDARY_RULES:
                if re.search(rule_regex, line):
                    violations.append(
                        _format_violation(
                            path_label,
                            idx,
                            "boundary_pattern",
                            f"disallowed pattern detected ({rule_label})",
                        )
                    )

    return violations


def check_local_commands_no_paid_gate() -> list[str]:
    """AST-inspect cli.py to guarantee no local-only command calls require_firm_key()."""
    violations: list[str] = []
    cli_file = SRC / "cli.py"
    if not cli_file.is_file():
        return [_format_violation("cli.py", None, "missing", "cli.py not found")]

    path_label = _rel_path(cli_file)
    tree = ast.parse(cli_file.read_text(encoding="utf-8"), filename=str(cli_file))

    class LocalCommandChecker(ast.NodeVisitor):
        def __init__(self) -> None:
            self.current_func: str | None = None

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            prev = self.current_func
            self.current_func = node.name
            self.generic_visit(node)
            self.current_func = prev

        def visit_Call(self, node: ast.Call) -> None:
            if self.current_func in LOCAL_ONLY_COMMAND_HANDLERS:
                call_name = ""
                if isinstance(node.func, ast.Name):
                    call_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    call_name = node.func.attr
                if call_name == "require_firm_key":
                    fname = self.current_func or "unknown"
                    violations.append(
                        _format_violation(
                            path_label,
                            int(node.lineno),
                            "paid_gate",
                            f"local-only command '{fname}' calls require_firm_key()",
                        )
                    )
            self.generic_visit(node)

    LocalCommandChecker().visit(tree)
    return violations


def check_import_no_networking() -> list[str]:
    """Ensure `import alphaengine` makes zero network socket calls."""
    script = (
        "import socket, sys\n"
        "def _blocked(*args, **kwargs):\n"
        "    raise RuntimeError('Network socket call during import!')\n"
        "socket.socket = _blocked\n"
        "socket.create_connection = _blocked\n"
        "try:\n"
        "    import alphaengine\n"
        "    print('IMPORT_OK')\n"
        "except Exception as exc:\n"
        "    print(f'IMPORT_FAILED: {exc}', file=sys.stderr)\n"
        "    sys.exit(1)\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        code = int(res.returncode)
        return [
            _format_violation(
                "import alphaengine",
                None,
                "network",
                f"network access attempted during import (subprocess exit {code}; stderr omitted)",
            )
        ]
    return []


def main() -> int:
    all_violations: list[str] = []

    # 1 & 2: Server modules and boundary patterns
    all_violations.extend(check_forbidden_modules_and_secrets())

    # 3: No local command calls require_firm_key
    all_violations.extend(check_local_commands_no_paid_gate())

    # 4: Zero networking on import
    all_violations.extend(check_import_no_networking())

    if all_violations:
        count = len(all_violations)
        print(
            f"PUBLIC BOUNDARY VIOLATIONS DETECTED (QOS-182): {count} violation(s)",
            file=sys.stderr,
        )
        return 1

    print("public boundary guard ok: no server code, disallowed patterns, or paid local gates found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
