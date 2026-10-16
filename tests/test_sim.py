"""The 1.0 simulator: tools, the trace, and the report the user accepts."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alphaengine._version import __version__
from alphaengine.agent.sim import Call, Trace, run_tools
from alphaengine.core.market import align_market, describe_market
from alphaengine.core.options import black_scholes, implied_vol
from alphaengine.core.regime import hamilton_regime
from alphaengine.tui.stage import stage_lines


def test_this_is_version_one() -> None:
    assert __version__ == "1.0.0"


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


def test_the_stage_draws_the_stack() -> None:
    frame = stage_lines(1.2, activity="regime", stack=["profile", "regime"], version="1.0")
    assert frame[0].startswith("AE 1.0")
    assert "regime" in frame[0]
    assert any("regime" in line for line in frame[-2:])
