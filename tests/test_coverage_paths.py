"""Coverage tests for network, auth, study reporting, CLI verbs, and session dispatch."""

from __future__ import annotations

import argparse
import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alphaengine import cli
from alphaengine.client import Offline, ServerError
from alphaengine.study import ReportError, Study, save
from alphaengine.study.report import _guard, _payload, _retail_payload, report
from alphaengine.tui import dispatch
from alphaengine.tui.dispatch import Desk, submit


class _FakeSession:
    def __init__(self, *, fail: Exception | None = None) -> None:
        self.fail = fail
        self.saved_signals: list[dict[str, Any]] = []
        self.filed_panels: list[dict[str, Any]] = []
        self.share_series = False

    def workflows(self) -> list[dict[str, Any]]:
        if self.fail:
            raise self.fail
        return [
            {
                "name": "screen_universe",
                "version": "1.0.0",
                "requires": ["universe"],
                "reproducible": True,
                "answers": "what is worth a look",
            },
            {
                "name": "size_position",
                "version": "1.0.0",
                "requires": ["returns"],
                "reproducible": True,
                "answers": "what sizing does the risk model imply",
            },
        ]

    def gaps(self) -> dict[str, Any]:
        if self.fail:
            raise self.fail
        return {
            "queued": [{"workflow": "screen_universe", "because": "universe has not been screened"}],
            "skipped": [{"workflow": "monitor_sleeve", "because": "no tolerances", "reason": "needs input"}],
        }

    def plan(self, budget: int = 0) -> dict[str, Any]:
        if self.fail:
            raise self.fail
        return {
            "queued": [
                {
                    "workflow": "screen_universe",
                    "inputs": {"universe": "sp500"},
                    "because": "due tonight",
                }
            ],
            "skipped": [{"workflow": "size_position", "reason": f"past budget {budget}"}],
        }

    def recent_runs(self, limit: int = 25) -> list[dict[str, Any]]:
        if self.fail:
            raise self.fail
        return [
            {
                "created_at": "2026-10-24T12:00:00Z",
                "kind": "screen_universe",
                "verdict": "completed",
                "honest_n": 25,
                "n_trials_source": "derived_from_grid",
                "signal_file": {"version": 2},
                "reason": "completed cleanly",
            },
            {
                "created_at": "2026-10-23T12:00:00Z",
                "kind": "validate_study",
                "verdict": "stopped",
                "reason": "knife edge surface",
            },
        ][:limit]

    def universes(self) -> list[dict[str, Any]]:
        if self.fail:
            raise self.fail
        return [
            {
                "id": "u1",
                "name": "sp500",
                "definition": {"symbols": ["AAPL", "MSFT"], "n_obs": 60, "cache_id": "c1"},
            },
            {
                "id": "u2",
                "name": "watch",
                "definition": {"symbols": ["NVDA"], "n_obs": 0},
            },
        ]

    def universe_series(self, universe_id: str, *, window: int = 0) -> dict[str, Any]:
        if self.fail:
            raise self.fail
        rng = np.random.default_rng(5)
        return {
            "prices": {
                "AAPL": (100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 60))).tolist(),
                "MSFT": (200 * np.cumprod(1 + rng.normal(0.0004, 0.01, 60))).tolist(),
            }
        }

    def theses(self) -> list[dict[str, Any]]:
        if self.fail:
            raise self.fail
        return [
            {"id": "th_1", "name": "momentum", "statement": "cross-sectional momentum holds"},
            {"id": "th_2", "name": "quality", "statement": "high ROE outperforms"},
        ]

    def sleeves(self) -> list[dict[str, Any]]:
        if self.fail:
            raise self.fail
        return [{"name": "core_book", "stage": "live", "universe_id": "u1"}]

    def file_panel(self, figures: dict[str, Any], *, thesis_id: str | None = None) -> dict[str, Any]:
        self.filed_panels.append({"figures": figures, "thesis_id": thesis_id})
        return {"filed": True, "id": "pnl_1"}

    def save_signals(self, **kwargs: Any) -> dict[str, Any]:
        self.saved_signals.append(kwargs)
        return {"label": kwargs.get("label", "signals"), "asof": kwargs.get("asof"), "version": 3}

    def open(self, workflow: str, **kwargs: Any) -> Any:
        if self.fail:
            raise self.fail

        class _ClosedRun:
            run_id = "r_scripted"
            status = "closed"
            permitted: list[dict[str, Any]] = []
            selection = "any"
            stopped = None
            artifact = {"workflow": workflow, "figures": {"s1": {"n_evaluated": 2}}}
            figures = {"s1": {"n_evaluated": 2}}

            def gaps(self) -> list[dict[str, Any]]:
                return []

            def filed_diff(self) -> None:
                return None

        return _ClosedRun()


def test_study_report_error_paths_and_nested_list_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    study = Study(
        label="test study",
        n_trials=5,
        process={"dgp": "gbm"},
        charts=[{"kind": "line", "key": "eq", "title": "Equity"}],
        trial_ledger=[{"index": 0, "params_hash": "abc", "sharpe": 0.5, "failed": None}],
    )
    study.seal()
    payload = _payload(study)
    assert payload["process"] == {"dgp": "gbm"}
    assert payload["charts"][0]["kind"] == "line"
    assert payload["trial_ledger"][0]["params_hash"] == "abc"
    assert _guard({"nested": [{"items": [1, 2, 3]}]})["nested"][0]["items"] == [1, 2, 3]

    # Retail fallback sharpe branches (sharpe_ratio then best_sharpe)
    s_ratio = Study(label="s1", performance={"sharpe_ratio": 0.42})
    assert _retail_payload(s_ratio)["sharpe"] == 0.42
    s_surf = Study(label="s2", surface={"best_sharpe": 1.15})
    assert _retail_payload(s_surf)["sharpe"] == 1.15

    # HTTP 401/403 -> rejected key
    def raise_401(req: Any, timeout: Any = None) -> Any:
        body = io.BytesIO(b'{"detail": "revoked token"}')
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, body)  # type: ignore[arg-type]

    monkeypatch.setattr(urllib.request, "urlopen", raise_401)
    with pytest.raises(ReportError, match="The key was rejected \\(401\\).*revoked token"):
        report(study, api_key="ae_live_bad", base_url="https://api.example.invalid")

    # HTTP 500 -> server refused
    def raise_500(req: Any, timeout: Any = None) -> Any:
        body = io.BytesIO(b"not json html error")
        raise urllib.error.HTTPError(req.full_url, 500, "Server Error", {}, body)  # type: ignore[arg-type]

    monkeypatch.setattr(urllib.request, "urlopen", raise_500)
    with pytest.raises(ReportError, match="The server refused the study \\(500\\)"):
        report(study, api_key="ae_live_ok", base_url="https://api.example.invalid")

    # URLError -> could not reach
    def raise_url_err(req: Any, timeout: Any = None) -> Any:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", raise_url_err)
    with pytest.raises(ReportError, match="Could not reach https://api.example.invalid"):
        report(study, api_key="ae_live_ok", base_url="https://api.example.invalid")


def test_cli_network_and_auth_commands_and_error_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = _FakeSession()
    monkeypatch.setattr(cli, "_session", lambda url, key, **kw: (fake, "https://api.example.invalid"))

    # workflows, gaps, tonight, runs
    assert cli.cmd_workflows(argparse.Namespace(url=None, key="ae_live_1")) == 0
    assert cli.cmd_gaps(argparse.Namespace(url=None, key="ae_live_1")) == 0
    assert cli.cmd_tonight(argparse.Namespace(url=None, key="ae_live_1", budget=2)) == 0
    assert cli.cmd_runs(argparse.Namespace(url=None, key="ae_live_1", limit=10)) == 0
    out = capsys.readouterr().out
    assert "screen_universe" in out
    assert "queued" in out
    assert "validate_study" in out

    # Account data & sleeve lines on boot
    data_lines = cli.account_data_lines(fake)
    sleeve_lines = cli.account_sleeve_lines(fake)
    assert any("sp500" in ln for ln in data_lines)
    assert any("prices not stored" in ln for ln in data_lines)
    assert any("core_book" in ln for ln in sleeve_lines)

    # Offline & ServerError(401) branches on workflows, gaps, tonight, runs
    offline_sess = _FakeSession(fail=Offline("offline"))
    monkeypatch.setattr(cli, "_session", lambda url, key, **kw: (offline_sess, "https://api.example.invalid"))
    assert cli.cmd_workflows(argparse.Namespace(url=None, key=None)) == 2
    assert cli.cmd_gaps(argparse.Namespace(url=None, key=None)) == 2
    assert cli.cmd_tonight(argparse.Namespace(url=None, key=None, budget=1)) == 2
    assert cli.cmd_runs(argparse.Namespace(url=None, key=None, limit=5)) == 2

    auth_sess = _FakeSession(fail=ServerError(401, "Unauthorized"))
    monkeypatch.setattr(cli, "_session", lambda url, key, **kw: (auth_sess, "https://api.example.invalid"))
    monkeypatch.setattr(cli, "offer_key", lambda *a, **k: False)
    assert cli.cmd_workflows(argparse.Namespace(url=None, key=None)) == 2
    assert cli.cmd_gaps(argparse.Namespace(url=None, key=None)) == 2
    assert cli.cmd_tonight(argparse.Namespace(url=None, key=None, budget=1)) == 2
    assert cli.cmd_runs(argparse.Namespace(url=None, key=None, limit=5)) == 2
    capsys.readouterr()

    # cmd_models, cmd_commands, cmd_trace
    assert cli.cmd_models(argparse.Namespace()) == 0
    assert cli.cmd_commands(argparse.Namespace(verb="")) == 0
    assert cli.cmd_commands(argparse.Namespace(verb="seal")) == 0
    assert cli.cmd_commands(argparse.Namespace(verb="nonexistent_verb")) == 2

    monkeypatch.setattr("alphaengine.events.data_dir", lambda: tmp_path)
    assert cli.cmd_trace(argparse.Namespace(run_id=None)) == 0
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    (runs_dir / "r123.jsonl").write_text('{"event": "step"}\n', encoding="utf-8")
    assert cli.cmd_trace(argparse.Namespace(run_id="r123")) == 0
    assert cli.cmd_trace(argparse.Namespace(run_id=None)) == 0
    capsys.readouterr()

    # cmd_seal and cmd_report_study error paths
    missing_path = tmp_path / "missing.json"
    assert cli.cmd_seal(argparse.Namespace(study_path=str(missing_path))) == 2
    assert cli.cmd_report_study(argparse.Namespace(study_path=str(missing_path))) == 2

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not valid json", encoding="utf-8")
    assert cli.cmd_seal(argparse.Namespace(study_path=str(bad_json))) == 2
    assert cli.cmd_report_study(argparse.Namespace(study_path=str(bad_json))) == 2

    tampered = tmp_path / "tampered.json"
    s = Study(label="t", n_trials=2)
    save(s, tampered)
    raw = json.loads(tampered.read_text(encoding="utf-8"))
    raw["n_trials"] = 99
    tampered.write_text(json.dumps(raw), encoding="utf-8")
    assert cli.cmd_report_study(argparse.Namespace(study_path=str(tampered), key="ae_live_1", url=None)) == 2

    valid_study = tmp_path / "valid.json"
    save(Study(label="ok", n_trials=2), valid_study)
    assert cli.cmd_report_study(argparse.Namespace(study_path=str(valid_study), key="", url=None)) == 2
    capsys.readouterr()


def test_cli_enter_key_and_render_helpers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "cfg"))

    assert cli.enter_key("unknown_provider") is False
    monkeypatch.setattr("getpass.getpass", lambda _p: "")
    assert cli.enter_key("quantos") is False

    monkeypatch.setattr("getpass.getpass", lambda _p: "ae_live_test_key")
    monkeypatch.setattr("builtins.input", lambda _p: "n")
    try:
        assert cli.enter_key("quantos") is True
        monkeypatch.setattr("builtins.input", lambda _p: "y")
        assert cli.enter_key("quantos") is True
    finally:
        cli.cmd_logout(argparse.Namespace(purge=True))

    # _render_artifact, _render_monitor_status, _say_diff, _say_next
    cli._render_artifact(
        {
            "figures": {
                "s1": {
                    "status": "breached",
                    "n_checked": 2,
                    "breaches": [
                        {
                            "limit": "max_drawdown_pct",
                            "observed": 25.0,
                            "tolerance": 20.0,
                            "direction": "above",
                        }
                    ],
                    "unknown_limits": ["mystery_metric"],
                    "verdict": "the_book_again",
                    "rows": [{"symbol": "AAPL", "score": 12.3, "volatility_pct": 18.4}],
                    "n_evaluated": 20,
                    "n_insufficient": 3,
                    "truncated": True,
                }
            }
        }
    )
    cli._render_monitor_status({"status": "ok", "n_checked": 1})
    cli._render_monitor_status({"status": "undetermined", "undetermined": [{"limit": "var_pct"}]})
    cli._render_monitor_status({"status": "unchecked"})

    class _DiffRun:
        run_id = "r_diff"
        status = "closed"
        artifact = {"figures": {"s1": {"deflated_sharpe": 0.8}}}

        def __init__(self, diff_data: dict[str, Any]) -> None:
            self._diff = diff_data

        def filed_diff(self) -> dict[str, Any]:
            return self._diff

        def gaps(self) -> list[dict[str, Any]]:
            return [{"says": "untested cost ladder", "closes_with": "stress_study"}]

    cli._say_diff(_DiffRun({"first_file": True}))
    cli._say_diff(_DiffRun({"compared_with": "2026-10-20", "held": ["AAPL"]}))
    cli._say_diff(
        _DiffRun(
            {
                "compared_with": "2026-10-20",
                "n_entered": 1,
                "n_exited": 1,
                "n_moved": 1,
                "held": ["MSFT"],
                "entered": [{"ticker": "NVDA"}],
                "exited": [{"ticker": "INTC"}],
            }
        )
    )
    cli._report(_DiffRun({}))
    out = capsys.readouterr().out
    assert "BREACHED" in out
    assert "NVDA" in out


def test_tui_dispatch_commands_and_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "cfg"))

    rng = np.random.default_rng(11)
    book = {
        "AAPL": (100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 90))).tolist(),
        "MSFT": (150 * np.cumprod(1 + rng.normal(0.0004, 0.012, 90))).tolist(),
        "NVDA": (200 * np.cumprod(1 + rng.normal(0.0006, 0.015, 90))).tolist(),
    }
    fake = _FakeSession()
    monkeypatch.setattr(cli, "_session", lambda url, key, **kw: (fake, "https://api.example.invalid"))

    desk = Desk(
        session=fake,
        url="https://api.example.invalid",
        args=argparse.Namespace(url=None, key="ae_live_1"),
        data=book,
        keyed=True,
    )

    assert submit(desk, "") is False
    assert submit(desk, "/help") is False
    assert submit(desk, "status") is False
    assert submit(desk, "view") is False
    assert submit(desk, "models") is False
    assert submit(desk, "model") is False
    assert submit(desk, "model anthropic/claude-3-5-sonnet") is False
    assert submit(desk, "commands") is False
    assert submit(desk, "workflows") is False
    assert submit(desk, "runs") is False
    assert submit(desk, "gaps") is False
    assert submit(desk, "tonight") is False
    assert submit(desk, "enter model key") is False
    assert desk.pending_secret == "MODEL_KEY"
    assert submit(desk, "key") is False
    assert submit(desk, "key anthropic") is False
    assert submit(desk, "key unknown_provider") is False

    # thesis & book commands
    assert submit(desk, "thesis") is False
    assert submit(desk, "thesis momentum") is False
    assert desk.thesis is not None
    assert submit(desk, "thesis clear") is False
    assert desk.thesis is None

    assert submit(desk, "book") is False
    assert submit(desk, "book AAPL") is False
    assert submit(desk, "book status") is False

    # cross-section & portfolio verbs
    assert submit(desk, "correlation") is False
    assert submit(desk, "covariance") is False
    assert submit(desk, "portfolio") is False
    assert submit(desk, "portfolio risk parity") is False
    assert submit(desk, "run correlation") is False
    assert submit(desk, "run portfolio") is False
    assert submit(desk, "run") is False
    assert submit(desk, "wrokflows") is False  # near verb suggestion

    # load / data / universe
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text("date,AAPL,MSFT\n2026-01-01,100,200\n2026-01-02,101,202\n", encoding="utf-8")
    assert submit(desk, "load") is False
    assert submit(desk, f"load {csv_path}") is False
    assert submit(desk, f"data {csv_path}") is False
    assert submit(desk, "universe sp500") is False
    assert submit(desk, "project missing_module_xyz") is False
    assert submit(desk, "trace") is False
    assert submit(desk, "process gbm") is False

    # Inline flags on submit (--data, --symbol on overlap, --share-series)
    assert submit(desk, f"run screen_universe --data {csv_path} --share-series") is False
    assert fake.share_series is True
    desk.data = book
    assert submit(desk, "overlap --symbol AAPL") is False

    # Sentence path with a stubbed think function
    desk.think = lambda prompt: (
        '{"tool":"profile","args":{}}'
        if "none yet" in prompt.split("Calls already made this turn:")[-1]
        else '{"tool":"done","args":{"answer":"Measured 3 names across 90 observations."}}'
    )
    assert submit(desk, "what is in this book?") is False
    assert desk.pending_report is not None

    # env_for_model_key prefixes
    assert dispatch.env_for_model_key("sk-ant-123") == "ANTHROPIC_API_KEY"
    assert dispatch.env_for_model_key("sk-or-123") == "OPENROUTER_API_KEY"
    assert dispatch.env_for_model_key("gsk_123") == "GROQ_API_KEY"
    assert dispatch.env_for_model_key("AIza123") == "GEMINI_API_KEY"
    assert dispatch.env_for_model_key("sk-123") == "OPENAI_API_KEY"
    assert dispatch.env_for_model_key("other") is None

    # _invoke_tool coverage across deterministic tools
    desk.data = book
    assert "n_names" in dispatch._invoke_tool(desk, "profile", {})
    assert "dates_supplied" in dispatch._invoke_tool(desk, "align", {})
    assert "sigma_high" in dispatch._invoke_tool(desk, "regime", {"name": "AAPL"})
    assert "price" in dispatch._invoke_tool(
        desk, "price", {"spot": 100, "strike": 100, "expiry": 1, "vol": 0.2}
    )
    assert "shocked_price" in dispatch._invoke_tool(
        desk,
        "reprice",
        {"spot": 100, "strike": 100, "expiry": 1, "vol": 0.2, "spot_shock": -0.05, "vol_shock": 0.05},
    )
    assert dispatch._invoke_tool(desk, "correlation", {})["kind"] == "correlation"
    assert dispatch._invoke_tool(desk, "portfolio", {"method": "hrp"})["method"] == "hrp"
    assert "sharpe_annualized" in dispatch._invoke_tool(desk, "performance", {})
    assert "var_pct" in dispatch._invoke_tool(desk, "var", {}).get("parametric", {})
    assert "n_paths" in dispatch._invoke_tool(desk, "process", {"name": "AAPL", "dgp": "gbm"})
    assert "error" in dispatch._invoke_tool(desk, "unknown_tool", {})

    # pending_report yes / no handling
    class _DummyReport:
        def save(self) -> Path:
            out_file = tmp_path / "report.txt"
            out_file.write_text("saved", encoding="utf-8")
            return out_file

    desk.pending_report = _DummyReport()
    assert submit(desk, "yes") is False
    desk.pending_report = _DummyReport()
    assert submit(desk, "no") is False

    # logout --purge in session
    assert submit(desk, "logout --purge") is False
    assert desk.keyed is False
    assert submit(desk, "quit") is True
    capsys.readouterr()
