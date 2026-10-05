"""A portfolio is the covariance of the loaded returns, with the session attached."""

from __future__ import annotations

import pytest

from alphaengine.core.portfolio import build_portfolio, portfolio_figures


def _prices() -> dict[str, list[float]]:
    return {
        "AAA": [100.0 + i * 0.4 for i in range(40)],
        "BBB": [50.0 + (i % 5) * 0.3 for i in range(40)],
        "CCC": [80.0 + i * 0.1 for i in range(40)],
    }


def test_hrp_weights_sum_to_one() -> None:
    result = build_portfolio(_prices(), notes=["correlation: 3 names, 39 observations."])
    assert result["method"] == "hrp"
    assert result["covariance"] == "ledoit_wolf"
    assert result["n_assets"] == 3
    assert result["weight_sum"] == pytest.approx(1.0, abs=1e-4)
    assert result["session"] == ["correlation"]
    assert sum(result["weights"].values()) == pytest.approx(1.0, abs=1e-4)
    assert all(weight >= 0 for weight in result["weights"].values())


def test_risk_parity_is_requested_in_words() -> None:
    result = build_portfolio(_prices(), method="risk_parity")
    assert result["method"] == "risk_parity"
    assert result["weight_sum"] == pytest.approx(1.0, abs=1e-4)


def test_a_prior_calculation_limits_the_book_to_the_names_it_kept() -> None:
    result = build_portfolio(_prices(), names=["AAA", "BBB"])
    assert result["names"] == ["AAA", "BBB"]


def test_one_name_is_not_a_portfolio() -> None:
    with pytest.raises(ValueError):
        build_portfolio({"AAA": [100.0, 101.0, 102.0, 103.0]})


def test_the_panel_is_weights_not_the_series() -> None:
    figures = portfolio_figures(build_portfolio(_prices()))
    assert figures["kind"] == "portfolio"
    assert figures["charts"] == [{"kind": "rows", "key": "weights", "title": "hrp weights"}]
    assert len(figures["weights"]) == 3
    blob = str(figures)
    assert "100.0" not in blob
    for row in figures["weights"]:
        assert set(row) == {"name", "weight"}
