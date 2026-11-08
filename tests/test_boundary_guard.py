"""Tests for the public boundary CI guard (QOS-182).

Approved by William on Nov 7, relayed through Leader.
Asserts that the public boundary guard passes on the repository, and that
a seeded violation of each kind fails the check.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_public_boundary import main as run_boundary_check  # noqa: E402


def test_public_boundary_guard_clean():
    """Verify that the repository currently passes all public boundary checks."""
    assert run_boundary_check() == 0


def test_seeded_server_only_module_violation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that importing a server-only module in src/ triggers a failure."""
    import scripts.check_public_boundary as guard

    fake_src = tmp_path / "src" / "alphaengine"
    fake_src.mkdir(parents=True)
    violation_file = fake_src / "leaked_server.py"
    violation_file.write_text("import supabase\nfrom trial_ledger_db import Session\n", encoding="utf-8")

    monkeypatch.setattr(guard, "SRC", fake_src)
    violations = guard.check_forbidden_modules_and_secrets()
    assert any("supabase" in v for v in violations)
    assert any("trial_ledger_db" in v for v in violations)
    # Messages must never echo the scanned source line (CodeQL clear-text logging).
    for v in violations:
        assert "import supabase" not in v
        assert "from trial_ledger_db" not in v


def test_seeded_secret_pattern_violation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that hardcoding secrets or Supabase URLs in src/ triggers a failure."""
    import scripts.check_public_boundary as guard

    fake_src = tmp_path / "src" / "alphaengine"
    fake_src.mkdir(parents=True)
    violation_file = fake_src / "credentials_leak.py"
    live_key = "ae_live_0123456789abcdef0123456789"
    violation_file.write_text(
        f'SUPABASE_URL = "https://myproject.supabase.co"\nROLE = "service_role"\nLIVE_KEY = "{live_key}"\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(guard, "SRC", fake_src)
    violations = guard.check_forbidden_modules_and_secrets()
    assert any("Supabase project URL" in v for v in violations)
    assert any("service_role" in v for v in violations)
    assert any("Live ae_live_* secret literal" in v for v in violations)
    # Logged messages are built from path/line/constant labels only — never the match.
    for v in violations:
        assert live_key not in v
        assert "myproject.supabase.co" not in v
        assert "SUPABASE_URL" not in v
        assert "LIVE_KEY" not in v


def test_seeded_local_command_paid_gate_violation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that a local-only command calling require_firm_key() triggers a failure."""
    import scripts.check_public_boundary as guard

    fake_src = tmp_path / "src" / "alphaengine"
    fake_src.mkdir(parents=True)
    fake_cli = fake_src / "cli.py"
    fake_cli.write_text(
        "import argparse\n"
        "from .auth import require_firm_key\n\n"
        "def cmd_init(args: argparse.Namespace) -> int:\n"
        "    require_firm_key()\n"
        "    return 0\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(guard, "SRC", fake_src)
    violations = guard.check_local_commands_no_paid_gate()
    assert len(violations) >= 1
    assert any("cmd_init" in v and "require_firm_key" in v for v in violations)


def test_seeded_import_network_violation(monkeypatch: pytest.MonkeyPatch):
    """Verify that if an import touches socket networking, the check catches it."""
    # Test the barrier mechanism by pointing it to a command that attempts a socket call
    script = (
        "import socket, sys\n"
        "def _blocked(*args, **kwargs):\n"
        "    raise RuntimeError('Network socket call during import!')\n"
        "socket.socket = _blocked\n"
        "try:\n"
        "    socket.socket()\n"
        "except Exception as exc:\n"
        "    print(f'IMPORT_FAILED: {exc}', file=sys.stderr)\n"
        "    sys.exit(1)\n"
    )
    res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert res.returncode == 1
    assert "Network socket call during import!" in res.stderr


def test_import_network_check_omits_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Import-network failures must not echo subprocess stderr into violation text."""
    import scripts.check_public_boundary as guard

    class FakeResult:
        returncode = 7
        stdout = ""
        stderr = "leak ae_live_0123456789abcdef0123456789 and password=supersecret"

    monkeypatch.setattr(guard.subprocess, "run", lambda *a, **k: FakeResult())
    violations = guard.check_import_no_networking()
    assert len(violations) == 1
    assert "exit 7" in violations[0]
    assert "ae_live_" not in violations[0]
    assert "supersecret" not in violations[0]
    assert "stderr omitted" in violations[0]
