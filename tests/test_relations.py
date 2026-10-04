"""Correlation, covariance, and cointegration on a loaded universe."""

from __future__ import annotations

import pytest

from alphaengine.core.relations import (
    correlation,
    covariance,
    overlap_against_the_rest,
    portal_figures,
)


def _prices() -> dict[str, list[float]]:
    # Two names that move together, and one that moves the other way.
    up = [10.0]
    down = [10.0]
    flat = [10.0]
    for _ in range(8):
        up.append(round(up[-1] * 1.01, 6))
        down.append(round(down[-1] * 0.99, 6))
        flat.append(round(flat[-1] * 1.01, 6))
    return {"AAA": up, "BBB": list(flat), "CCC": down}


def test_correlation_of_closes_is_a_full_matrix() -> None:
    result = correlation(_prices())
    names = result["names"]
    assert names == ["AAA", "BBB", "CCC"]
    assert result["n_obs"] == 8
    grid = result["values"]
    assert len(grid) == 3 and all(len(row) == 3 for row in grid)
    # AAA and BBB were built from the same growth. The diagonal is 1.
    i_aaa, i_bbb = names.index("AAA"), names.index("BBB")
    assert grid[i_aaa][i_aaa] == 1.0
    assert grid[i_aaa][i_bbb] == 1.0
    assert grid[i_aaa][names.index("CCC")] < 0


def test_return_series_are_not_treated_as_prices() -> None:
    data = {
        "AAA": [0.01, 0.02, -0.01, 0.03],
        "BBB": [0.02, 0.04, -0.02, 0.06],
    }
    result = correlation(data)
    assert result["values"][0][1] == 1.0
    assert result["n_obs"] == 4


def test_covariance_diagonal_is_the_sample_variance() -> None:
    data = {"AAA": [0.01, 0.02, 0.03], "BBB": [0.01, 0.02, 0.03]}
    result = covariance(data)
    # mean 0.02, sum of squared deviations 0.0002, divide by n-1.
    assert result["values"][0][0] == pytest.approx(0.0001, abs=1e-6)


def test_overlap_uses_the_other_names_as_the_book() -> None:
    data = {
        "AAA": [0.01 * i for i in range(30)],
        "BBB": [0.02] * 30,
        "CCC": [0.04] * 30,
    }
    prepared = overlap_against_the_rest(data, "aaa")
    assert prepared["symbol"] == "AAA"
    assert prepared["book_names"] == ["BBB", "CCC"]
    assert prepared["book_returns"][0] == pytest.approx(0.03)
    assert len(prepared["returns"]) == len(prepared["book_returns"]) == 30


def test_a_hundred_name_matrix_is_a_portal_figure() -> None:
    """The triangle of pairs does not fit the 512 cap. The square of rows does."""
    from alphaengine.client.executor import _guard

    data = {f"N{i:03d}": [0.01 * (i + 1), -0.02, 0.015 * ((i % 4) + 1), 0.005] for i in range(100)}
    figures = portal_figures(correlation(data))
    _guard(figures)
    assert figures["charts"][0] == {"kind": "matrix", "key": "matrix", "title": "correlation"}
    assert len(figures["names"]) == 100
    assert len(figures["matrix"]) == 100
    assert len(figures["matrix"][0]) == 100
    assert figures["truncated"] is False
    assert "returns" not in figures


def test_a_book_past_the_figure_cap_is_truncated_not_refused() -> None:
    from alphaengine.client.executor import _guard

    data = {f"N{i:04d}": [0.01, -0.02, 0.015, 0.005 * ((i % 3) + 1)] for i in range(513)}
    figures = portal_figures(correlation(data))
    _guard(figures)
    assert figures["n_names"] == 513
    assert figures["truncated"] is True
    assert len(figures["matrix"]) == 512
    assert len(figures["matrix"][0]) == 512


def test_overlap_figures_name_the_candidate_and_the_book() -> None:
    from alphaengine.client.executor import StepExecutor

    data = {
        "returns": [0.01 * ((i % 5) - 2) for i in range(40)],
        "book_returns": [0.004 * ((i % 3) - 1) for i in range(40)],
        "symbol": "MU",
        "book_names": ["AAPL", "MSFT"],
    }
    out = StepExecutor(data=data).execute("compute.overlap", {})
    assert out["symbol"] == "MU"
    assert out["n_book_names"] == 2
    assert out["book_names"] == ["AAPL", "MSFT"]
    assert "returns" not in out
    kinds = {hint["kind"] for hint in out["charts"]}
    assert "scatter" in kinds
    assert "curve" in kinds


def test_a_missing_panels_route_does_not_fail_the_math() -> None:
    from alphaengine.client.session import ServerError, Session

    session = Session(base_url="https://example.invalid", api_key="k")
    figures = portal_figures(correlation({"AAA": [0.01, 0.02, -0.01], "BBB": [0.02, 0.04, -0.02]}))
    seen: dict = {}

    def _post(path: str, body: dict) -> dict:
        seen["path"] = path
        seen["thesis"] = body.get("thesis_id")
        return {"id": "panel_1"}

    session._post = _post  # type: ignore[method-assign]
    filed = session.file_panel(figures, thesis_id="th_1")
    assert filed["filed"] is True
    assert seen["path"] == "/api/me/panels"
    assert seen["thesis"] == "th_1"

    def _missing(path: str, body: dict) -> dict:
        raise ServerError(404, "not found")

    session._post = _missing  # type: ignore[method-assign]
    assert session.file_panel(figures)["reason"] == "missing"


def test_cointegration_reports_pairs_without_importing_until_asked() -> None:
    pytest.importorskip("statsmodels")
    from alphaengine.core.relations import cointegration

    prices = {name: [10.0 + i for i in range(10)] for name in ("AAA", "BBB", "CCC")}
    result = cointegration(prices)
    assert result["kind"] == "cointegration"
    assert result["n_pairs"] == 3
    assert result["n_cointegrated"] == 0
