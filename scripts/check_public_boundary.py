#!/usr/bin/env python3
"""Public boundary CI guard for alphaengine (QOS-182).

Approved by William on Nov 7, relayed through Leader.
Enforces the boundary between the public open-source library / local research OS
and the hosted quantOS server / proprietary infrastructure.

Fails if:
1. The package imports or contains server-only modules (trial ledger DB,
   Supabase, import worker, AI orchestration, tenancy, MCP gateway).
2. Any Supabase URL, service_role, JWT, or live secret pattern appears in src/.
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

# ── 2. Secret and credential patterns ──────────────────────────────────────

SECRET_PATTERNS = [
    (r"https?://[a-zA-Z0-9_-]+\.supabase\.co", "Supabase project URL"),
    (r"\bservice_role\b", "Supabase service_role key reference"),
    (r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}", "JWT token"),
    (r"\bae_live_[0-9a-zA-Z]{16,}\b", "Live ae_live_* secret literal"),
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


def _redact_for_log(text: str) -> str:
    """Scrub credential-like tokens before any log/print sink (CodeQL clear-text logging).

    Returns a freshly allocated string with known secret shapes replaced so the
    print sink never receives clear-text credentials / JWTs / supabase URLs.
    """
    if not text:
        return ""
    # Copy into a new buffer first so later joins are not the original object.
    chars = list(str(text))
    out = "".join(chars)
    out = re.sub(r"\bae_live_[0-9a-zA-Z]{16,}\b", "ae_live_[REDACTED]", out)
    out = re.sub(
        r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
        "[JWT_REDACTED]",
        out,
    )
    out = re.sub(
        r"https?://[a-zA-Z0-9_-]+\.supabase\.co[^\s\"']*",
        "[SUPABASE_URL_REDACTED]",
        out,
    )
    out = re.sub(r"(?i)\b(authorization|bearer)\s+[^\s]+", r"\1 [REDACTED]", out)
    # Final round-trip guarantees a new str identity for static analyzers.
    return str(out)


def check_forbidden_modules_and_secrets() -> list[str]:
    """Scan all Python files in src/ for forbidden server modules and secret patterns."""
    violations: list[str] = []

    for path in sorted(SRC.rglob("*.py")):
        # Check file name
        fname = path.name.lower()
        for pat in FORBIDDEN_SERVER_MODULE_PATTERNS:
            if re.search(pat, fname):
                violations.append(f"{_rel_path(path)}: forbidden server module filename matching {pat}")

        content = path.read_text(encoding="utf-8")
        lines = content.splitlines()

        for idx, line in enumerate(lines, 1):
            # Check forbidden import statements
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                for pat in FORBIDDEN_SERVER_MODULE_PATTERNS:
                    if re.search(pat, stripped):
                        violations.append(
                            f"{_rel_path(path)}:{idx}: forbidden server-only import "
                            f"matching {pat}: {stripped}"
                        )

            # Check secret patterns
            for sec_pat, desc in SECRET_PATTERNS:
                if re.search(sec_pat, line):
                    violations.append(f"{_rel_path(path)}:{idx}: secret/credential pattern detected ({desc})")

    return violations


def check_local_commands_no_paid_gate() -> list[str]:
    """AST-inspect cli.py to guarantee no local-only command calls require_firm_key()."""
    violations: list[str] = []
    cli_file = SRC / "cli.py"
    if not cli_file.is_file():
        return [f"Missing {cli_file}"]

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
                    violations.append(
                        f"{_rel_path(cli_file)}:{node.lineno}: local-only command '{self.current_func}' "
                        f"calls require_firm_key() (client-only paid gate violation; QOS-180/QOS-182)"
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
        # Never echo raw stderr: it may carry env, paths, or credential-shaped text.
        return [
            "Network access attempted during `import alphaengine` "
            f"(subprocess exit {res.returncode}; stderr redacted)"
        ]
    return []


def main() -> int:
    all_violations: list[str] = []

    # 1 & 2: Server modules and secret patterns
    all_violations.extend(check_forbidden_modules_and_secrets())

    # 3: No local command calls require_firm_key
    all_violations.extend(check_local_commands_no_paid_gate())

    # 4: Zero networking on import
    all_violations.extend(check_import_no_networking())

    if all_violations:
        print("PUBLIC BOUNDARY VIOLATIONS DETECTED (QOS-182):", file=sys.stderr)
        for v in all_violations:
            # Always redact before the logging sink (CodeQL py/clear-text-logging-sensitive-data).
            safe_msg = _redact_for_log(v)
            sys.stderr.write("  - " + safe_msg + "\n")
        return 1

    print("public boundary guard ok: no server code, secrets, or paid local gates found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
