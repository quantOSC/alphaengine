"""Tests for keyring-first auth, secret hygiene, and CLI UX (`QOS-155`, `QOS-168`, `QOS-169`)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from alphaengine import auth, cli
from alphaengine.ideas import create_idea
from alphaengine.project import init_project
from alphaengine.sync import remote_add


class _FakeVaultBackend:
    """In-memory fake OS keyring backend for deterministic unit testing."""


class _FakeKeyringModule(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("keyring")
        self.store: dict[tuple[str, str], str] = {}
        self._backend = _FakeVaultBackend()

    def get_keyring(self) -> Any:
        return self._backend

    def get_password(self, service: str, username: str) -> str | None:
        return self.store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        self.store.pop((service, username), None)


def test_keyring_only_storage_and_legacy_file_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.delenv("QUANTOS_API_KEY", raising=False)

    fake_kr = _FakeKeyringModule()
    monkeypatch.setitem(sys.modules, "keyring", fake_kr)
    monkeypatch.setattr(auth, "_ALLOW_KEYRING_IN_TEST", True)
    monkeypatch.setattr(auth, "_KEYRING", fake_kr)

    assert auth.keyring_available() is True

    # 1. Saving when keyring is available writes ONLY to keyring — `credentials.json` is NOT created!
    res = auth.save_credential("QUANTOS_API_KEY", "ae_live_keyring_secret_9999")
    assert res["storage"] == "keyring"
    assert res["file_written"] is False
    assert not auth.config_path().exists()
    assert fake_kr.get_password("alphaengine", "QUANTOS_API_KEY") == "ae_live_keyring_secret_9999"

    # 2. Legacy `credentials.json` migration moves secrets into keyring and deletes plaintext file
    legacy_path = auth.config_path()
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(
        json.dumps({"ANTHROPIC_API_KEY": "sk-ant-legacysecret123456"}),
        encoding="utf-8",
    )
    assert auth.has_legacy_file_credentials() is True
    mig = auth.migrate_file_to_keyring()
    assert mig["migrated"] == ["ANTHROPIC_API_KEY"]
    assert not legacy_path.exists()
    assert fake_kr.get_password("alphaengine", "ANTHROPIC_API_KEY") == "sk-ant-legacysecret123456"

    # 3. `purge_stored()` clears BOTH keyring entries and any file fallback
    purged = auth.purge_stored()
    assert purged is True
    assert fake_kr.store == {}


def test_plaintext_file_fallback_disclosure_and_no_keyring_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg_nokeyring"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata_nokeyring"))
    monkeypatch.delenv("QUANTOS_API_KEY", raising=False)
    monkeypatch.setattr(auth, "_ALLOW_KEYRING_IN_TEST", False)
    monkeypatch.setattr(auth, "_KEYRING", False)

    # Without a usable keyring, `login` discloses plaintext mode-0600 storage
    planted = "ae_live_SECRET_DO_NOT_LEAK_12345"
    monkeypatch.setattr("getpass.getpass", lambda _p="": planted)
    monkeypatch.setattr("builtins.input", lambda _p="": "y")
    rc = cli.main(["login", "--key", planted])
    assert rc == 0
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    # Raw planted secret MUST NEVER appear in stdout or stderr!
    assert planted not in combined
    assert "mode-0600" in combined or "plaintext" in combined.lower()
    # `--key` deprecation warning emitted on stderr
    assert "--key" in captured.err
    assert auth.config_path().is_file()

    # `logout --purge` removes stored credentials cleanly
    assert cli.main(["logout", "--purge"]) == 0
    capsys.readouterr()
    assert not auth.config_path().exists()


def test_validate_clean_url_rejects_embedded_secrets_and_userinfo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    proj = init_project(tmp_path, name="clean-url-test")

    assert auth.validate_clean_url("https://api.qosai.app") == "https://api.qosai.app"

    dirty_urls = [
        "https://user:secretpass@api.qosai.app",
        "https://ae_live_SECRET12345@api.qosai.app",
        "https://api.qosai.app/workspace?api_key=ae_live_SECRET12345",
        "https://api.qosai.app/workspace?token=my_secret_token",
        "ftp://api.qosai.app",
    ]
    for bad in dirty_urls:
        with pytest.raises(ValueError):
            auth.validate_clean_url(bad)
        with pytest.raises(ValueError):
            remote_add(proj, "quantos", bad)


def test_keyless_diagnostics_whoami_version_doctor_and_secret_redaction_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg_diag"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata_diag"))
    monkeypatch.delenv("QUANTOS_API_KEY", raising=False)
    monkeypatch.setattr(auth, "_ALLOW_KEYRING_IN_TEST", False)
    monkeypatch.setattr(auth, "_KEYRING", False)

    # 1. `ae version`, `ae whoami`, and `ae doctor` MUST succeed (`exit 0`) WITHOUT `QUANTOS_API_KEY`!
    assert cli.main(["version", "--json"]) == 0
    v_json = json.loads(capsys.readouterr().out)
    assert v_json["engine_version"] == "2.0.4"
    assert v_json["sync_contract"] == 2
    assert v_json["manifest_version"] == 1

    assert cli.main(["whoami", "--json"]) == 0
    w_json = json.loads(capsys.readouterr().out)
    assert w_json["quantos"]["configured"] is False

    assert cli.main(["doctor", "--json"]) == 0
    d_json = json.loads(capsys.readouterr().out)
    assert d_json["engine"]["engine_version"] == "2.0.4"

    # 2. Plant a live secret in env and sweep `whoami`, `doctor`, and error paths:
    # the raw secret MUST NEVER appear anywhere in stdout, stderr, or JSON!
    secret = "ae_live_SUPER_SECRET_NEVER_PRINT_8888"
    monkeypatch.setenv("QUANTOS_API_KEY", secret)
    monkeypatch.setenv(
        "ALPHAENGINE_API_URL",
        f"https://api.qosai.app/bad?token={secret}",
    )

    for argv in (["whoami"], ["whoami", "--json"], ["doctor"], ["doctor", "--json"]):
        cli.main(argv)
        out_err = capsys.readouterr()
        assert secret not in out_err.out
        assert secret not in out_err.err

    assert secret not in auth.redact_secrets(f"Authorization: Bearer {secret}")


def test_push_human_readable_preview_and_tty_confirmation_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    proj = init_project(tmp_path, name="push-ux-test")
    remote_add(proj, "quantos", "https://api.quantos.example")
    create_idea(proj, thesis="Dispersion carry", kill_rule="dsr_probability < 0.5")

    # 1. `ae push --dry-run` works offline WITHOUT a firm key (`exit 0`) and prints human-readable preview!
    monkeypatch.delenv("QUANTOS_API_KEY", raising=False)
    assert cli.main(["push", "--dry-run"]) == 0
    preview_out = capsys.readouterr().out
    assert "push preview" in preview_out
    assert "Code and raw datasets stay local." in preview_out

    # 2. Interactive TTY without `--yes`: declining confirmation (`"n"`) aborts cleanly before HTTP!
    monkeypatch.setenv("QUANTOS_API_KEY", "ae_live_ux_key_123456")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "n")
    http_called = False

    def forbid_http(*_a: Any, **_kw: Any) -> Any:
        nonlocal http_called
        http_called = True
        raise AssertionError("HTTP should not be called when user declines push confirmation")

    monkeypatch.setattr("urllib.request.urlopen", forbid_http)
    rc_cancel = cli.main(["push"])
    assert rc_cancel == 0
    assert http_called is False
    assert "cancelled" in capsys.readouterr().out.lower()
