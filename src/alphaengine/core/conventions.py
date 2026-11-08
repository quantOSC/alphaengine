"""Canonical statistical verdict vocabulary and stated output conventions (2.0).

Every validation, scoring, backtest, walk-forward, and sweep entrypoint shares:
1. One statistical ``Verdict`` enum (``robust``, ``marginal``, ``likely_noise``,
   ``unverified``), with documented deprecated alias normalization for 1.x names
   (``edge -> robust``, ``inconclusive -> marginal``, ``acceptable -> marginal``,
   ``overfit -> likely_noise``, ``not_recorded -> unverified``).
2. Explicit, machine-readable ``annualization``, ``risk_free``, and
   ``costs_model`` convention dictionaries stamped onto outputs so identical
   returns through the library, CLI ``--json``, MCP server, and backend API
   always report identical numbers and definitions.
"""

from __future__ import annotations

import warnings
from enum import Enum
from typing import Any

from .rates import cash_rate

VOLATILITY_FLOOR: float = 1e-6


class Verdict(str, Enum):
    """Canonical statistical verdict across AlphaEngine 2.0."""

    ROBUST = "robust"
    MARGINAL = "marginal"
    LIKELY_NOISE = "likely_noise"
    UNVERIFIED = "unverified"

    @classmethod
    def normalize(cls, value: str | Verdict | None, *, warn: bool = False) -> Verdict | None:
        """Normalize a string or Verdict to a canonical Verdict enum member."""
        norm = normalize_verdict(value, warn=warn)
        if norm is None:
            return None
        try:
            return cls(norm)
        except ValueError:
            return None


DEPRECATED_VERDICT_ALIASES: dict[str, Verdict] = {
    "edge": Verdict.ROBUST,
    "inconclusive": Verdict.MARGINAL,
    "acceptable": Verdict.MARGINAL,
    "overfit": Verdict.LIKELY_NOISE,
    "not_recorded": Verdict.UNVERIFIED,
}


def normalize_verdict(value: str | Verdict | None, *, warn: bool = False) -> str | None:
    """Normalize a canonical or legacy 1.x verdict string into a 2.0 ``Verdict`` value.

    Non-statistical workflow status words (such as ``clean`` or ``ranked``) are
    returned unchanged if not part of the statistical verdict alias map.
    """
    if value is None:
        return None
    if isinstance(value, Verdict):
        return value.value
    key = str(value).strip().lower()
    if not key:
        return None
    for member in Verdict:
        if key == member.value:
            return member.value
    if key in DEPRECATED_VERDICT_ALIASES:
        mapped = DEPRECATED_VERDICT_ALIASES[key].value
        if warn:
            warnings.warn(
                f"Verdict {key!r} is deprecated in alphaengine 2.0 and will be removed in 2.1; "
                f"use {mapped!r} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return mapped
    return key


def verdict_from_dsr(
    dsr_probability: float | None,
    *,
    pbo: float | None = None,
    n_trials: int | float | None = None,
    n_trials_source: str = "asserted",
) -> tuple[str, bool]:
    """Compute canonical ``(verdict, deflation_applied)`` from DSR and PBO.

    Rules:
    - ``deflation_applied`` is ``True`` iff ``n_trials_source != "not_recorded"``
      and ``n_trials is not None`` and ``float(n_trials) > 1.0``.
    - Without deflation (``n_trials <= 1`` or ``not_recorded``), ``robust`` is
      structurally unreachable: a strong single-trial result caps at ``marginal``.
    - ``pbo > 0.5`` or ``dsr_probability < 0.5`` yields ``likely_noise``.
    - ``deflation_applied and dsr_probability >= 0.9 and (pbo is None or pbo <= 0.2)``
      yields ``robust``; all intermediate cases yield ``marginal``.
    """
    deflation_applied = bool(
        n_trials_source != "not_recorded" and n_trials is not None and float(n_trials) > 1.0
    )
    if dsr_probability is None:
        if n_trials_source == "not_recorded":
            return Verdict.UNVERIFIED.value, False
        return Verdict.MARGINAL.value, deflation_applied
    if pbo is not None and float(pbo) > 0.5:
        return Verdict.LIKELY_NOISE.value, deflation_applied
    if float(dsr_probability) < 0.5:
        return Verdict.LIKELY_NOISE.value, deflation_applied
    if not deflation_applied:
        return Verdict.MARGINAL.value, False
    if float(dsr_probability) >= 0.9 and (pbo is None or float(pbo) <= 0.2):
        return Verdict.ROBUST.value, True
    return Verdict.MARGINAL.value, True


def build_annualization_spec(periods_per_year: int = 252) -> dict[str, Any]:
    """Return the stated annualization convention dictionary."""
    ppy = max(int(periods_per_year), 1)
    return {
        "periods_per_year": ppy,
        "sharpe_scaling": "sqrt(periods_per_year)",
        "volatility_scaling": "sqrt(periods_per_year)",
        "return_compounding": "geometric",
        "alpha_scaling": "arithmetic",
    }


def build_risk_free_spec(
    risk_free_rate: float | None = None,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Return the stated risk-free rate convention dictionary."""
    ppy = max(int(periods_per_year), 1)
    annual, source, as_of = cash_rate(risk_free_rate)
    return {
        "annual_rate": float(annual),
        "per_period_rate": round(float(annual) / ppy, 10),
        "per_period_formula": "annual_rate / periods_per_year",
        "source": source,
        "as_of": as_of,
    }


def build_costs_model_spec(
    *,
    mode: str = "pre_netted_returns",
    slippage_bps: float | None = None,
    commission_bps: float | None = None,
    borrow_bps: float | None = None,
    fill_timing: str | None = None,
    adv_source: str | None = None,
    liquidate_final: bool | None = None,
) -> dict[str, Any]:
    """Return the stated transaction-cost model dictionary."""
    return {
        "mode": mode,
        "slippage_bps": float(slippage_bps) if slippage_bps is not None else None,
        "commission_bps": float(commission_bps) if commission_bps is not None else None,
        "borrow_bps": float(borrow_bps) if borrow_bps is not None else None,
        "fill_timing": fill_timing,
        "adv_source": adv_source,
        "liquidate_final": liquidate_final,
    }


__all__ = [
    "DEPRECATED_VERDICT_ALIASES",
    "VOLATILITY_FLOOR",
    "Verdict",
    "build_annualization_spec",
    "build_costs_model_spec",
    "build_risk_free_spec",
    "normalize_verdict",
    "verdict_from_dsr",
]
