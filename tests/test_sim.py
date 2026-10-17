"""The 1.0 simulator: tools, the trace, and the report the user accepts."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from alphaengine._version import __version__
from alphaengine.agent.sim import Call, Trace, run_tools
from alphaengine.core.market import align_market, describe_market
from alphaengine.core.options import black_scholes, implied_vol
from alphaengine.core.regime import hamilton_regime
from alphaengine.tui.stage import stage_lines


def test_this_is_version_one() -> None:
    assert __version__.startswith("1.0.")


def test_an_atm_call_matches_the_normal_cdf() -> None:
    # r = 0, vol = 0.2, T = 1, S = K = 100. d1 = 0.1, price = 100 * (2 N(0.1) - 1).
    n_d1 = 0.5 * (1.0 + math.erf(0.1 / math.sqrt(2.0)))
    expected = 100.0 * (2.0 * n_d1 - 1.0)
    got = black_scholes(spot=100, strike=100, expiry_years=1, vol=0.2, rate=0)
    assert got["price"] == round(expected, 6)
    assert abs(got["price"] - 7.965567) < 1e-4
    back = implied_vol(price=got["price"], spot=100, strike=100, expiry_years=1, rate=0)
    assert back["implied_vol"] == pytest.approx(0.2, abs=1e-4)


def test_a_calm_then_wild_path_ends_in_the_high_vol_state() -> None:
    rng = np.random.default_rng(3)
    calm = rng.normal(0.0, 0.01, 80)
    wild = rng.normal(0.0, 0.05, 80)
    fit = hamilton_regime(np.concatenate([calm, wild]))
    assert "error" not in fit
    assert fit["sigma_high"] > fit["sigma_low"]
    assert fit["prob_high_now"] > 0.5
    assert "_path" in fit


def test_dates_are_kept_and_a_gap_is_counted() -> None:
    days = [f"2026-01-{d:02d}" for d in range(1, 11)]
    prices = {day: 100.0 + i for i, day in enumerate(days)}
    macro = {day: 4.2 for day in days if day != "2026-01-05"}
    book = {"AAA": prices, "DGS10": macro}
    described = describe_market(book)
    assert "DGS10" in described["macro"]
    assert "AAA" in described["prices"]
    aligned = align_market(book)
    assert aligned["n_common"] == 9
    assert aligned["gaps"]["DGS10"] == 1
    assert aligned["n_macro"] == 1
    assert aligned["n_price"] == 1


def test_a_sentence_runs_two_tools_and_refuses_an_invented_number(tmp_path) -> None:
    seen: list[str] = []

    def think(prompt: str) -> str:
        calls = prompt.split("Calls already made this turn:")[-1]
        if "none yet" in calls:
            return '{"tool":"profile","args":{}}'
        if "profile" in calls and "var" not in calls:
            return '{"tool":"var","args":{}}'
        return '{"tool":"done","args":{"answer":"The book Sharpe is 9.99."}}'

    def invoke(name: str, args: dict) -> dict:
        seen.append(name)
        if name == "profile":
            return {"n_names": 3, "n_obs_min": 40, "note": "three names"}
        return {"var_95": 0.021}

    trace = run_tools("stress the book", think, invoke)
    assert seen == ["profile", "var"]
    assert trace.refused
    assert trace.answer.startswith("Refused")
    path = trace.save(tmp_path)
    text_path = path if path.suffix == ".txt" else path.with_suffix(".txt")
    text = text_path.read_text(encoding="utf-8")
    assert text.startswith("alphaengine 1.0")
    assert "profile" in text and "var" in text
    assert "Refused" in text


def test_a_saved_trace_is_a_charted_pdf(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    trace = Trace(
        question="stress the book",
        answer="The book stays inside the bands the tools reported.",
        calls=[Call(tool="var", args={}, figures={"var_95": 0.021, "cvar_95": 0.034}, note="var_95 0.021")],
    )
    trace.panel = {"AAPL": [10.0, 10.4, 10.1, 10.8, 11.0, 10.6, 11.2, 11.5, 11.1, 11.8]}
    path = trace.save(tmp_path)
    assert path.suffix == ".pdf"
    raw = path.read_bytes()
    assert raw.startswith(b"%PDF")
    assert len(raw) > 2000
    text = path.with_suffix(".txt").read_text(encoding="utf-8")
    assert text.startswith("alphaengine 1.0")
    assert "stress the book" in text
    assert "AAPL" not in text


def test_the_downloaded_report_has_a_short_name(tmp_path) -> None:
    import re

    path = Trace(question="construct the book").save(tmp_path)
    assert re.fullmatch(r"report-[0-9a-f]{6}", path.stem)
    assert path.with_suffix(".txt").is_file()


def test_a_constructed_book_has_weights_scenarios_and_a_simulation() -> None:
    from alphaengine.agent.brief import study_for

    rng = np.random.default_rng(3)
    panel = {}
    for name in ("AAPL", "MSFT", "XOM"):
        panel[name] = (100 * np.cumprod(1 + rng.normal(0.0004, 0.01 * (len(panel) + 1), 80))).tolist()
    trace = Trace(question="construct a portfolio looking at XOM", panel=panel)
    study = study_for(trace, deep=True)
    methods = {book["method"] for book in study["books"]}
    assert {"equal_weight", "hrp", "risk_parity", "var_sorted"} <= methods
    hrp_book = next(book for book in study["books"] if book["method"] == "hrp")
    weights = [w for _, w in hrp_book["weights"]]
    assert max(weights) > min(weights)
    assert study["focus_names"] == ["XOM"]
    assert study["correlation"]["names"]
    assert any(row["name"] == "volatility x1.5" for row in study["scenarios"])
    assert study["monte"]["n_paths"] == 64
    assert len(study["monte"]["fan"]["q50"]) >= 20
    assert study["bakeoff"]["survivor"]
    assert "Hierarchical risk parity" in study["findings"]
    assert "AAPL" not in Trace(question="construct a portfolio", panel=panel).render()


def test_monte_carlo_and_scenarios_are_harness_tools() -> None:
    from alphaengine.agent.sim import TOOLS
    from alphaengine.tui.dispatch import Desk, _invoke_tool

    rng = np.random.default_rng(7)
    book = {
        "AAA": (100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 90))).tolist(),
        "BBB": (80 * np.cumprod(1 + rng.normal(0.0002, 0.018, 90))).tolist(),
        "CCC": (60 * np.cumprod(1 + rng.normal(0.0003, 0.012, 90))).tolist(),
    }
    tool_names = {name for name, _ in TOOLS}
    assert "monte_carlo" in tool_names
    assert "scenarios" in tool_names
    desk = Desk(session=object(), url="fake://", args=None, data=book)
    mc = _invoke_tool(desk, "monte_carlo", {"dgp": "gbm", "n_paths": 32, "horizon": 60})
    sc = _invoke_tool(desk, "scenarios", {})
    assert mc["n_paths"] == 32
    assert "terminal_q50_pct" in mc
    assert sc["n_scenarios"] >= 5


def test_a_report_defaults_to_the_downloads_folder(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("alphaengine.agent.sim.downloads_dir", lambda: tmp_path)
    path = Trace(question="stress the book").save()
    assert path.parent == tmp_path
    assert path.name.startswith("report-")


def test_downloads_falls_back_to_home_when_the_known_folder_is_missing(monkeypatch, tmp_path) -> None:
    from alphaengine.agent.sim import downloads_dir

    monkeypatch.setattr("alphaengine.agent.sim._known_downloads", lambda: None)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    assert downloads_dir() == tmp_path / "Downloads"


def test_xdg_names_the_downloads_folder(monkeypatch, tmp_path) -> None:
    from alphaengine.agent.sim import _xdg_downloads

    config = tmp_path / "config"
    config.mkdir()
    (config / "user-dirs.dirs").write_text('XDG_DOWNLOAD_DIR="$HOME/Descargas"\n', encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    assert _xdg_downloads() == tmp_path / "home" / "Descargas"


def test_the_stage_draws_the_stack() -> None:
    frame = stage_lines(1.2, activity="regime", stack=["profile", "regime"], version="1.0")
    assert frame[0].startswith("AE 1.0")
    assert "regime" in frame[0]
    assert any("regime" in line for line in frame[-2:])
