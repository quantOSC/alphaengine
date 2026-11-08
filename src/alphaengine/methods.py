"""Machine-readable catalogue of AlphaEngine statistical and simulation methods.

Supports ``QOS-91`` / ``QOS-126`` figure citations so portal methodology pages,
committee memos, and API citation objects derive formulas, assumptions, and
``engine_version`` directly from the installed engine rather than hand-edited copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._version import current_engine_version


@dataclass(frozen=True)
class MethodSpec:
    """Immutable specification of one quantitative method in AlphaEngine."""

    method_id: str
    title: str
    entrypoint: str
    formula: str
    citation: str
    assumptions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "title": self.title,
            "entrypoint": self.entrypoint,
            "formula": self.formula,
            "citation": self.citation,
            "assumptions": list(self.assumptions),
            "engine_version": current_engine_version(),
        }


_METHODS: tuple[MethodSpec, ...] = (
    MethodSpec(
        method_id="deflated_sharpe",
        title="Deflated Sharpe Ratio (DSR Probability)",
        entrypoint="alphaengine.core.validation:deflated_sharpe",
        formula="Phi((SR - SR_0) * sqrt(T - 1) / sqrt(1 - skew * SR + ((kurt - 1) / 4) * SR^2))",
        citation="Bailey & Lopez de Prado (2014), Journal of Portfolio Management 40(5), 94-107.",
        assumptions=(
            "Returns are stationary with finite fourth moment (Pearson kurtosis).",
            "Every executed attempt counts in n_trials (including identical reruns and failed grid points).",
            "At n_trials <= 1, deflation_applied is False and verdict cannot exceed marginal.",
        ),
    ),
    MethodSpec(
        method_id="probabilistic_sharpe_ratio",
        title="Probabilistic Sharpe Ratio (PSR)",
        entrypoint="alphaengine.core.validation:probabilistic_sharpe_ratio",
        formula="Phi((SR - SR*) * sqrt(T - 1) / sqrt(1 - skew * SR + ((kurt - 1) / 4) * SR^2))",
        citation="Bailey & Lopez de Prado (2012), Journal of Risk 15(2), 3-44.",
        assumptions=("Adjusts single-trial Sharpe standard error for sample skewness and kurtosis.",),
    ),
    MethodSpec(
        method_id="min_track_record_length",
        title="Minimum Track Record Length (MinTRL)",
        entrypoint="alphaengine.core.validation:min_track_record_length",
        formula="1 + (1 - skew * SR + ((kurt - 1) / 4) * SR^2) * (Z_alpha / (SR - SR*))^2",
        citation="Bailey & Lopez de Prado (2012), Journal of Risk 15(2), 3-44.",
        assumptions=("Requires observed per-period Sharpe strictly above sr_benchmark.",),
    ),
    MethodSpec(
        method_id="pbo_cscv",
        title="Probability of Backtest Overfitting via CSCV",
        entrypoint="alphaengine.core.validation:pbo_cscv",
        formula="mean(logit(rank_OOS(argmax_IS)) <= 0) across balanced (S, S/2) block partitions",
        citation=(
            "Bailey, Borwein, Lopez de Prado & Zhu (2017), Journal of Computational Finance 20(4), 39-69."
        ),
        assumptions=(
            "Requires T >= 2 * n_splits aligned observations across N >= 2 configurations.",
            "n_splits must be an even integer >= 2.",
        ),
    ),
    MethodSpec(
        method_id="combinatorial_oos_stability",
        title="Combinatorial Out-of-Sample Stability",
        entrypoint="alphaengine.core.validation:combinatorial_oos_stability",
        formula="Distribution of Sharpe and DSR across C(G, k) purged and embargoed held-out block paths",
        citation="Lopez de Prado (2018), Advances in Financial Machine Learning, ch. 7 & 12.",
        assumptions=(
            "Evaluates a pre-computed return stream with purge trimmed at block start "
            "and embargo at block end.",
        ),
    ),
    MethodSpec(
        method_id="harvey_liu_haircut",
        title="Harvey-Liu Multiple-Testing Sharpe Haircut",
        entrypoint="alphaengine.core.validation:harvey_liu_haircut",
        formula="Bonferroni, Holm step-down, and BHY step-up adjusted Student-t p-values mapped to Sharpe",
        citation="Harvey & Liu (2015), Journal of Portfolio Management 42(1), 13-28.",
        assumptions=("Uses AR(1) effective sample size n_obs_effective when supplied.",),
    ),
    MethodSpec(
        method_id="performance_report",
        title="Risk-Adjusted Performance Report",
        entrypoint="alphaengine.core.performance:performance_report",
        formula="Sharpe = mean(r - rf/ppy)/std(r, ddof=1)*sqrt(ppy); CAGR = prod(1+r)^(ppy/T) - 1",
        citation="Sortino & Price (1994); pinned DGS3MO cash rate via alphaengine.core.rates.",
        assumptions=(
            "Excess returns subtract annual_rate / periods_per_year unless risk_free_rate=0.0 is passed.",
            "Drawdown, VaR, and CVaR are reported as positive loss magnitudes.",
        ),
    ),
    MethodSpec(
        method_id="run_backtest",
        title="Deterministic Weight-Target Portfolio Simulator",
        entrypoint="alphaengine.core.backtest:run_backtest",
        formula=(
            "Causal close or next_open fills net of slippage_bps, commission_bps, "
            "borrow_bps, impact, and final liquidation"
        ),
        citation="Almgren et al. (2005) square-root impact when ADV or adv_series is supplied.",
        assumptions=(
            "next_open fills wait for a prior close; missing opens fall back to that bar's close.",
            "Short positions accrue borrow_bps / 10000 / periods_per_year per bar.",
            "Final residual positions liquidate at the final bar with full transaction costs.",
        ),
    ),
    MethodSpec(
        method_id="coded_signal_adapter",
        title="Causal Coded-Strategy Signal Adapter",
        entrypoint="alphaengine.coded:run_signal",
        formula=(
            "signal(data_asof=data[:t], params=params) -> validated dated weights "
            "-> run_backtest(next_open, costs on)"
        ),
        citation="AlphaEngine 2.0 Coded-Backtest Specification (QOS-154).",
        assumptions=(
            "data_asof contains bars strictly <= rebalance bar t (no future bars).",
            "Grid searches are capped at 400 combinations under family coded:<idea_id>.",
        ),
    ),
)

METHODS_CATALOGUE: dict[str, MethodSpec] = {m.method_id: m for m in _METHODS}


def get_method(method_id: str) -> dict[str, Any]:
    """Return one method specification stamped with the live ``engine_version``."""
    spec = METHODS_CATALOGUE.get(str(method_id).strip())
    if spec is None:
        raise KeyError(f"Unknown method_id {method_id!r}. Known: {sorted(METHODS_CATALOGUE)}")
    return spec.to_dict()


def list_methods() -> list[dict[str, Any]]:
    """Return all registered method specifications stamped with ``engine_version``."""
    return [m.to_dict() for m in _METHODS]


__all__ = ["METHODS_CATALOGUE", "MethodSpec", "get_method", "list_methods"]
