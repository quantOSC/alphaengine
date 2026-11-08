"""Top-level ``alphaengine.validation`` compatibility module (``QOS-151``).

Re-exports all validation primitives from :mod:`alphaengine.core.validation`
and the canonical :class:`~alphaengine.core.conventions.Verdict` enum so
callers importing ``alphaengine.validation`` directly resolve the canonical 2.0
implementations.
"""

from __future__ import annotations

from .core.conventions import DEPRECATED_VERDICT_ALIASES, Verdict, normalize_verdict, verdict_from_dsr
from .core.validation import (
    VOL_FLOOR,
    combinatorial_oos_stability,
    cpcv_score,
    deflated_sharpe,
    dsr_probability,
    expected_max_sharpe,
    harvey_liu_haircut,
    min_track_record_length,
    n_eff_participation,
    pbo_cscv,
    probabilistic_sharpe_ratio,
    trials_for_deflated_sharpe,
)
from .core.walkforward import walk_forward

__all__ = [
    "DEPRECATED_VERDICT_ALIASES",
    "VOL_FLOOR",
    "Verdict",
    "combinatorial_oos_stability",
    "cpcv_score",
    "deflated_sharpe",
    "dsr_probability",
    "expected_max_sharpe",
    "harvey_liu_haircut",
    "min_track_record_length",
    "n_eff_participation",
    "normalize_verdict",
    "pbo_cscv",
    "probabilistic_sharpe_ratio",
    "trials_for_deflated_sharpe",
    "verdict_from_dsr",
    "walk_forward",
]
