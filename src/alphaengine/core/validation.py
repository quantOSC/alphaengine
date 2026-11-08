"""
Validation tools, the moat: tell the caller when an idea is probably noise.

Pure numpy/scipy:
  - Deflated Sharpe Ratio (DSR probability) + Probabilistic Sharpe Ratio (PSR),
    Bailey & López de Prado (2014). Corrects the Sharpe for the number of
    trials and for non-normal returns.
  - Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric
    Cross-Validation (CSCV), Bailey, Borwein, López de Prado, Zhu (2017).
  - Combinatorial Out-of-Sample Stability, López de Prado (2018).
  - Minimum Track Record Length (MinTRL), Bailey & López de Prado (2012).
  - Harvey & Liu (2015) multiple-testing Sharpe ratio haircuts.

No look-ahead, no LLM, no I/O. Deterministic given inputs on the pinned stack.
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import special, stats

from .._version import current_engine_version
from .conventions import (
    Verdict,
    build_annualization_spec,
    build_costs_model_spec,
    build_risk_free_spec,
    verdict_from_dsr,
)
from .rates import cash_rate, rate_fields
from .trace import record

_EULER = 0.5772156649015329

#: Volatility floor (per-period return std). Streams below 1e-6 are treated as
#: flat/degenerate rather than dividing by floating-point rounding noise.
VOL_FLOOR = 1e-6

_SESSION_LEDGER_PROVIDER: Callable[[], int | None] | None = None


def set_session_ledger_provider(provider: Callable[[], int | None] | None) -> None:
    """Register a provider that reports cumulative attempts executed in this session (`QOS-181`)."""
    global _SESSION_LEDGER_PROVIDER
    _SESSION_LEDGER_PROVIDER = provider


def get_session_ledger_floor() -> int | None:
    """Return the active session ledger trial count if registered, else None."""
    if _SESSION_LEDGER_PROVIDER is not None:
        try:
            val = _SESSION_LEDGER_PROVIDER()
            if val is not None and not isinstance(val, bool) and int(val) >= 1:
                return int(val)
        except Exception:  # noqa: BLE001
            return None
    return None


def trials_for_deflated_sharpe(
    n_trials: int | None = None,
    *,
    n_trials_source: str | None = None,
    sweep_trials: int | None = None,
    ledger_trials: int | None = None,
    ledger_runs: Sequence[Mapping[str, Any]] | int | None = None,
) -> tuple[int | None, str]:
    """Resolve ``(n_trials, n_trials_source)`` applying any recorded ledger floor.

    Rules (``QOS-151`` / ``QOS-153`` / ``QOS-181``):
    - Never invents ``1`` when unrecorded and never applies a synthetic baseline-120
      floor: only real executed attempts in ``sweep_trials``, ``n_trials``,
      ``ledger_trials``, ``ledger_runs``, or an active session hook contribute.
    - ``monte_carlo`` stress runs inside ``ledger_runs`` contribute ``0`` strategy
      trials; every other executed attempt contributes ``max(1, r["n_trials"])``.
    - When the ledger holds strictly more attempts than the single-run ``n_trials``
      (or when ``n_trials`` was omitted), the ledger floor wins with source
      ``"derived_from_ledger"``.
    """
    ledger_floor: int | None = None
    session_floor = get_session_ledger_floor()
    if session_floor is not None:
        ledger_floor = session_floor
    if isinstance(ledger_runs, int) and not isinstance(ledger_runs, bool):
        if ledger_runs >= 1:
            ledger_floor = int(ledger_runs)
    elif isinstance(ledger_runs, Sequence):
        counted = 0
        for row in ledger_runs:
            if not isinstance(row, Mapping):
                counted += 1
                continue
            src = str(row.get("n_trials_source") or "")
            if src == "monte_carlo":
                continue
            raw_nt = row.get("n_trials")
            if isinstance(raw_nt, (int, float)) and not isinstance(raw_nt, bool) and int(raw_nt) >= 1:
                counted += int(raw_nt)
            else:
                counted += 1
        if counted >= 1:
            ledger_floor = counted

    if ledger_trials is not None and not isinstance(ledger_trials, bool) and int(ledger_trials) >= 1:
        ledger_floor = max(ledger_floor or 0, int(ledger_trials))

    base: int | None = None
    base_src = "not_recorded"
    if sweep_trials is not None and not isinstance(sweep_trials, bool) and int(sweep_trials) >= 1:
        base = int(sweep_trials)
        base_src = "derived_from_grid"
    elif n_trials is not None and not isinstance(n_trials, bool) and int(n_trials) >= 1:
        base = int(n_trials)
        base_src = str(n_trials_source or "asserted")

    if ledger_floor is not None and ledger_floor >= 1 and (base is None or ledger_floor > base):
        return ledger_floor, "derived_from_ledger"
    return base, base_src


def _harmonic_number(m: float) -> float:
    """Harmonic number H_M = sum_{k=1}^M 1/k, extended smoothly to real M >= 1 via digamma."""
    m_f = max(float(m), 1.0)
    if m_f.is_integer() and m_f <= 10000:
        return float(sum(1.0 / k for k in range(1, int(m_f) + 1)))
    return float(special.digamma(m_f + 1.0) + _EULER)


def harvey_liu_haircut(
    sharpe_annualized: float,
    n_obs: int,
    n_trials: int | float,
    *,
    periods_per_year: int = 252,
    n_obs_effective: float | None = None,
    n_eff_participation: float | None = None,
    trial_pvalues: Sequence[float] | None = None,
    trial_rank: int = 1,
) -> dict[str, Any]:
    """Harvey & Liu (2015) 'Backtesting' multiple-testing Sharpe ratio haircut.

    Computes two-sided single-test p-value from the Student-t distribution with
    ``df = n_obs - 1`` degrees of freedom, adjusts it under Bonferroni, Holm
    (step-down), and BHY (Benjamini, Hochberg & Yekutieli 2001 step-up), and
    converts each adjusted p-value back into an adjusted t-statistic and
    haircutted annualized Sharpe ratio at both ``n_trials`` and
    ``n_eff_participation``.
    """
    ppy = max(int(periods_per_year), 1)
    t_obs = max(int(n_obs), 2)
    df = max(t_obs - 1, 1)
    n_eff_obs = (
        float(n_obs_effective) if (n_obs_effective is not None and n_obs_effective > 0) else float(t_obs)
    )
    sr_ann = float(sharpe_annualized)
    sr_pp = sr_ann / math.sqrt(ppy)
    t_stat = sr_pp * math.sqrt(n_eff_obs)
    p_single = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df))) if t_stat != 0.0 else 1.0

    def _at_multiplicity(m_val: float) -> dict[str, Any]:
        m_f = max(float(m_val), 1.0)
        c_m = _harmonic_number(m_f)

        # Bonferroni: p^Bonf = min(1, M * p_single)
        p_bonf = min(1.0, m_f * p_single)

        if trial_pvalues is not None and len(trial_pvalues) >= 2:
            p_sorted = sorted(max(0.0, min(1.0, float(p))) for p in trial_pvalues)
            k_len = len(trial_pvalues)
            idx = max(0, min(int(trial_rank) - 1, k_len - 1))
            if int(trial_rank) == 1:
                diffs = [abs(p - p_single) for p in p_sorted]
                idx = int(np.argmin(diffs))
            holm_steps = []
            for j in range(idx + 1):
                mult_j = max(1.0, m_f - j * (m_f - 1.0) / max(k_len - 1, 1))
                holm_steps.append(min(1.0, mult_j * p_sorted[j]))
            p_holm = max(holm_steps) if holm_steps else p_bonf

            bhy_steps = []
            for j in range(idx, k_len):
                rank_j = 1.0 + j * (m_f - 1.0) / max(k_len - 1, 1)
                bhy_steps.append(min(1.0, (m_f * c_m / rank_j) * p_sorted[j]))
            p_bhy = min(bhy_steps) if bhy_steps else min(1.0, m_f * c_m * p_single)
        else:
            m_int = max(int(round(m_f)), 1)
            if m_int <= 1:
                p_holm = p_single
                p_bhy = p_single
            else:
                p_sched = [
                    p_single if j == 0 else min(1.0, p_single + (j / m_f) * (1.0 - p_single))
                    for j in range(m_int)
                ]
                r_idx = max(0, min(int(trial_rank) - 1, m_int - 1))
                p_holm = min(1.0, max((m_f - j) * p_sched[j] for j in range(r_idx + 1)))
                p_bhy = min(
                    1.0,
                    min((m_f * c_m / (j + 1.0)) * p_sched[j] for j in range(r_idx, m_int)),
                )

        def _method_entry(p_adj: float) -> dict[str, Any]:
            if sr_ann <= 0.0 or t_stat <= 0.0 or p_adj >= 1.0:
                t_adj = 0.0
                sr_adj = 0.0
                hc_pct = 100.0 if sr_ann > 0.0 else 0.0
            else:
                t_adj = float(max(0.0, stats.t.ppf(1.0 - p_adj / 2.0, df=df)))
                sr_adj = float((t_adj / math.sqrt(n_eff_obs)) * math.sqrt(ppy))
                hc_pct = float(max(0.0, (1.0 - sr_adj / sr_ann) * 100.0))
            return {
                "p_value": round(float(p_adj), 6),
                "t_stat_adjusted": round(t_adj, 4),
                "haircut_sharpe": round(sr_adj, 4),
                "haircut_pct": round(hc_pct, 2),
            }

        m_out: int | float = int(m_f) if m_f.is_integer() else round(m_f, 4)
        return {
            "n_trials": m_out,
            "bonferroni": _method_entry(p_bonf),
            "holm": _method_entry(p_holm),
            "bhy": _method_entry(p_bhy),
        }

    at_n = _at_multiplicity(float(n_trials))
    neff_val = float(n_eff_participation) if n_eff_participation is not None else float(n_trials)
    at_neff = _at_multiplicity(neff_val)
    return {
        "engine_version": current_engine_version(),
        "sharpe_annualized": round(sr_ann, 4),
        "t_stat": round(t_stat, 4),
        "p_value_single": round(p_single, 6),
        "at_n_trials": at_n,
        "at_n_eff": at_neff,
        "bonferroni_p": at_n["bonferroni"]["p_value"],
        "holm_p": at_n["holm"]["p_value"],
        "bhy_p": at_n["bhy"]["p_value"],
        "haircut_sharpe_bonferroni": at_n["bonferroni"]["haircut_sharpe"],
        "haircut_sharpe_holm": at_n["holm"]["haircut_sharpe"],
        "haircut_sharpe_bhy": at_n["bhy"]["haircut_sharpe"],
    }


def _as_1d_float(values: ArrayLike | Sequence[float] | None) -> np.ndarray:
    """Convert any 1-D array-like (list, ndarray, pandas Series) to a 1-D float ndarray, dropping None."""
    if values is None:
        return np.empty(0, dtype=float)
    raw = np.asarray(values)
    if raw.ndim == 0:
        item = raw.item()
        return np.empty(0, dtype=float) if item is None else np.asarray([float(item)], dtype=float)
    if raw.dtype.kind in ("f", "i", "u"):
        return raw.astype(float, copy=False).ravel()
    return np.asarray([float(r) for r in raw.ravel() if r is not None], dtype=float)


# ── Sharpe statistics ───────────────────────────────────────────────────

# Sample volatility below this is zero for a Sharpe. Exactly 1e-6 still
# divides. A constant series is not always ``std == 0`` in float64, and
# dividing by that residual prints a ratio with no economic meaning. Same
# tolerance in ``core.performance``.
_ZERO_VOL = 1e-6


def _per_period_sharpe(returns: np.ndarray, *, risk_free_per_period: float = 0.0) -> float:
    """Mean excess over a constant per-period cash rate, divided by sample vol.

    Applies ``VOL_FLOOR = 1e-6`` so flat or near-flat rounding noise never
    produces an unbounded Sharpe ratio.
    """
    if returns.size < 2:
        return 0.0
    sd = float(returns.std(ddof=1))
    if not math.isfinite(sd) or sd < VOL_FLOOR:
        return 0.0
    return float((returns.mean() - float(risk_free_per_period)) / sd)


def probabilistic_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
    sr_benchmark: float = 0.0,
) -> float:
    """PSR: P(true per-period Sharpe > sr_benchmark). Bailey & LdP (2014).

    `kurtosis` is Pearson (normal = 3). Returns a probability in [0, 1].
    """
    if n_obs < 2 or not math.isfinite(sharpe):
        return 0.0
    sk = skew if math.isfinite(skew) else 0.0
    ku = kurtosis if math.isfinite(kurtosis) else 3.0
    denom = 1.0 - sk * sharpe + ((ku - 1.0) / 4.0) * sharpe**2
    if denom <= 0 or not math.isfinite(denom):
        denom = 1e-9
    z = (sharpe - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    if not math.isfinite(z):
        return 0.0
    return float(stats.norm.cdf(z))


def n_eff_participation(matrix_or_corr: ArrayLike) -> float:
    """Effective number of trials via the eigenvalue participation ratio.

    Computes ``(sum(lambda_i))^2 / sum(lambda_i^2)`` over the eigenvalues of
    the trial-return correlation matrix ``C``. Accepts either an ``(N, N)``
    correlation matrix directly (symmetric with unit diagonal) or a ``(T, N)``
    matrix of per-period trial returns.
    """
    arr = np.asarray(matrix_or_corr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return 1.0
    if (
        arr.shape[0] == arr.shape[1]
        and np.all(np.isfinite(arr))
        and np.allclose(arr, arr.T, atol=1e-8)
        and np.allclose(np.diag(arr), 1.0, atol=1e-6)
    ):
        corr = arr
    else:
        if arr.shape[0] < 2:
            return 1.0
        stds = arr.std(axis=0, ddof=1)
        valid = np.isfinite(stds) & (stds > 1e-15)
        n_valid = int(np.sum(valid))
        if n_valid < 2:
            return 1.0
        corr = np.corrcoef(arr[:, valid], rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 1.0)
    eigs = np.clip(np.linalg.eigvalsh(corr), 0.0, None)
    denom = float(np.sum(eigs**2))
    if denom <= 0.0:
        return 1.0
    return float((float(np.sum(eigs)) ** 2) / denom)


def expected_max_sharpe(n_trials: int | float, trials_sharpe_std: float) -> float:
    """E[max Sharpe] over N independent trials of noise (per-period units).

    SR0 = σ_SR · [ (1-γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ].
    The benchmark a *real* strategy must beat to be non-spurious.
    """
    n_f = float(n_trials)
    if n_f <= 1.0 or trials_sharpe_std <= 0:
        return 0.0
    a = stats.norm.ppf(1.0 - 1.0 / n_f)
    b = stats.norm.ppf(1.0 - 1.0 / (n_f * math.e))
    return float(trials_sharpe_std * ((1.0 - _EULER) * a + _EULER * b))


def deflated_sharpe(
    returns: ArrayLike | Sequence[float],
    *,
    n_trials: int | float,
    trials_sharpe_std: float | None = None,
    risk_free_rate: float | None = None,
    periods_per_year: int = 252,
    matrix: ArrayLike | None = None,
    n_eff: float | None = None,
    n_eff_method: str | None = None,
) -> dict[str, Any]:
    """Deflated Sharpe Ratio probability for a return stream.

    DSR = PSR(SR0), where SR0 = expected max Sharpe across ``n_trials``.
    Emits ``dsr_probability`` as the canonical [0, 1] probability field and keeps
    ``deflated_sharpe`` as a deprecated 2.0 alias. At ``n_trials <= 1``,
    ``deflation_applied`` is ``False`` and ``verdict`` cannot reach ``robust``.
    When ``matrix`` or ``n_eff`` is supplied (`QOS-181`), reports both raw
    ``n_trials`` and ``n_eff`` with ``n_eff_method`` labeled.
    """
    arr = _as_1d_float(returns)
    n = arr.size
    if n < 8:
        return {
            "error": "need >= 8 observations",
            "n_obs": int(n),
            "engine_version": current_engine_version(),
        }

    ppy = max(int(periods_per_year), 1)
    annual, _, _ = cash_rate(risk_free_rate)
    rf_per = annual / ppy
    sd = float(arr.std(ddof=1))
    vol_floored = bool(not math.isfinite(sd) or sd < VOL_FLOOR)

    n_f = float(n_trials)
    trials_out: int | float = int(n_f) if n_f.is_integer() else round(n_f, 4)

    if vol_floored:
        sr = 0.0
        skew = 0.0
        kurt = 3.0
        psr0 = 0.0
        sr0 = 0.0
        dsr = 0.0
        std_for_max = (
            float(trials_sharpe_std)
            if (trials_sharpe_std is not None and float(trials_sharpe_std) > 0)
            else 0.0
        )
        deflation_applied = bool(n_f > 1.0)
        verdict = Verdict.LIKELY_NOISE.value
        no_deflation_note: str | None = (
            f"volatility below VOL_FLOOR ({VOL_FLOOR:g}); Sharpe and DSR floored to 0.0."
        )
    else:
        sr = _per_period_sharpe(arr, risk_free_per_period=rf_per)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            raw_skew = float(stats.skew(arr, bias=False)) if n > 2 else 0.0
            raw_kurt = float(stats.kurtosis(arr, fisher=False, bias=False)) if n > 3 else 3.0
        skew = raw_skew if math.isfinite(raw_skew) else 0.0
        kurt = raw_kurt if math.isfinite(raw_kurt) else 3.0
        psr0 = probabilistic_sharpe_ratio(sr, n, skew, kurt, 0.0)

        sr_est_sd = (
            math.sqrt(max(0.0, 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2) / (n - 1)) if n > 1 else 0.0
        )
        std_for_max = (
            float(trials_sharpe_std)
            if (trials_sharpe_std is not None and float(trials_sharpe_std) > 0)
            else sr_est_sd
        )
        if n_f <= 1.0:
            sr0 = 0.0
            dsr = psr0
            no_deflation_note = (
                "n_trials=1: no multiple-testing deflation applies; dsr_probability equals psr_vs_zero "
                "and verdict cannot exceed marginal."
            )
        else:
            sr0 = expected_max_sharpe(n_f, std_for_max)
            dsr = probabilistic_sharpe_ratio(sr, n, skew, kurt, sr0)
            no_deflation_note = None
        verdict, deflation_applied = verdict_from_dsr(
            dsr,
            n_trials=n_f,
            n_trials_source="asserted",
        )

    dsr_rounded = round(dsr, 4)
    out: dict[str, Any] = {
        "engine_version": current_engine_version(),
        "n_obs": int(n),
        "n_trials": trials_out,
        "deflation_applied": deflation_applied,
        "vol_floored": vol_floored,
        "sharpe_per_period": round(sr, 4),
        "sharpe_annualized": round(sr * math.sqrt(ppy), 4),
        **rate_fields(risk_free_rate),
        "periods_per_year": ppy,
        "annualization": build_annualization_spec(ppy),
        "risk_free": build_risk_free_spec(risk_free_rate, ppy),
        "costs_model": build_costs_model_spec(mode="pre_netted_returns"),
        "skew": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "psr_vs_zero": round(psr0, 4),
        "sr0_expected_max": round(sr0, 4),
        "dsr_probability": dsr_rounded,
        "deflated_sharpe": dsr_rounded,
        "trials_sharpe_std": round(std_for_max, 4),
        # The headline shown INSTEAD of raw Sharpe.
        "verdict": verdict,
    }
    if no_deflation_note is not None:
        out["note"] = no_deflation_note

    if matrix is not None and n_eff is None:
        n_eff = round(n_eff_participation(matrix), 4)

    if n_eff is not None:
        neff_val = float(n_eff)
        neff_method = n_eff_method or "eigenvalue_participation_ratio"
        if vol_floored:
            dsr_neff = 0.0
        elif neff_val <= 1.0:
            dsr_neff = psr0
        else:
            sr0_neff = expected_max_sharpe(neff_val, std_for_max)
            dsr_neff = probabilistic_sharpe_ratio(sr, n, skew, kurt, sr0_neff)
        dsr_neff_rounded = round(dsr_neff, 4)
        neff_out: int | float = int(neff_val) if neff_val.is_integer() else round(neff_val, 4)
        out["n_eff"] = neff_out
        out["n_eff_participation"] = neff_out
        out["n_eff_method"] = neff_method
        out["deflated_sharpe_at_n_eff"] = dsr_neff_rounded
        out["dsr_probability_at_n_eff"] = dsr_neff_rounded
    record(
        "deflated_sharpe",
        "PSR(SR*) where SR* is the expected maximum Sharpe across n_trials",
        inputs={"n_obs": int(n), "n_trials": trials_out, "sharpe_per_period": round(sr, 4)},
        series={"returns": arr},
        result=dsr_rounded,
    )
    record(
        "psr_vs_zero",
        "norm.cdf( (SR - 0) * sqrt(n-1) / sqrt(1 - skew*SR + ((kurt-1)/4)*SR^2) )",
        inputs={"n_obs": int(n), "skew": round(skew, 4), "kurtosis": round(kurt, 4)},
        series={"returns": arr},
        result=round(psr0, 4),
    )
    record(
        "sr0_expected_max",
        "sigma_SR * ((1-gamma)*Phi^{-1}(1-1/N) + gamma*Phi^{-1}(1-1/(N*e)))",
        inputs={"n_trials": trials_out, "trials_sharpe_std": round(std_for_max, 4)},
        series={"returns": arr},
        result=round(sr0, 4),
    )
    return out


def min_track_record_length(
    returns: ArrayLike | Sequence[float],
    *,
    sr_benchmark: float = 0.0,
    confidence: float = 0.95,
    risk_free_rate: float | None = None,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Minimum Track Record Length (Bailey & López de Prado, 2012): the number of
    observations needed for the observed Sharpe to be statistically greater than
    `sr_benchmark` at `confidence`.

    MinTRL = 1 + [1 - γ3·SR + ((γ4-1)/4)·SR²] · (Z_α / (SR - SR*))²  (per-period).
    """
    arr = _as_1d_float(returns)
    n = arr.size
    if n < 8:
        return {
            "error": "need >= 8 observations",
            "n_obs": int(n),
            "engine_version": current_engine_version(),
        }
    ppy = max(int(periods_per_year), 1)
    annual, _, _ = cash_rate(risk_free_rate)
    sr = _per_period_sharpe(arr, risk_free_per_period=annual / ppy)
    if sr <= sr_benchmark:
        return {
            "engine_version": current_engine_version(),
            "n_obs": int(n),
            "sharpe_per_period": round(sr, 4),
            **rate_fields(risk_free_rate),
            "periods_per_year": ppy,
            "annualization": build_annualization_spec(ppy),
            "risk_free": build_risk_free_spec(risk_free_rate, ppy),
            "costs_model": build_costs_model_spec(mode="pre_netted_returns"),
            "min_track_record_length": None,
            "sufficient": False,
            "note": "observed Sharpe <= benchmark; not distinguishable at any length",
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        skew = float(stats.skew(arr, bias=False)) if n > 2 else 0.0
        kurt = float(stats.kurtosis(arr, fisher=False, bias=False)) if n > 3 else 3.0
    skew = skew if math.isfinite(skew) else 0.0
    kurt = kurt if math.isfinite(kurt) else 3.0
    z = float(stats.norm.ppf(confidence))
    mintrl = 1.0 + (1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2) * (z / (sr - sr_benchmark)) ** 2
    mintrl = max(1.0, float(mintrl))
    out = {
        "engine_version": current_engine_version(),
        "n_obs": int(n),
        "sharpe_per_period": round(sr, 4),
        "confidence": confidence,
        "min_track_record_length": round(mintrl, 1),
        "min_track_record_years": round(mintrl / ppy, 2),
        **rate_fields(risk_free_rate),
        "periods_per_year": ppy,
        "annualization": build_annualization_spec(ppy),
        "risk_free": build_risk_free_spec(risk_free_rate, ppy),
        "costs_model": build_costs_model_spec(mode="pre_netted_returns"),
        "sufficient": bool(n >= mintrl),
        "shortfall_obs": max(0, int(math.ceil(mintrl - n))),
    }
    record(
        "min_track_record_length",
        "1 + (1 - skew*SR + ((kurt-1)/4)*SR^2) * (Z_alpha / (SR - SR*))^2",
        inputs={"n_obs": int(n), "confidence": confidence, "sharpe_per_period": round(sr, 4)},
        series={"returns": arr},
        result=round(mintrl, 1),
    )
    return out


# ── PBO via CSCV ────────────────────────────────────────────────────────


def _block_sharpe(sums: np.ndarray, sqs: np.ndarray, counts: np.ndarray, blocks: list[int]) -> np.ndarray:
    """Per-config Sharpe over the selected blocks from precomputed moments."""
    s = sums[blocks].sum(axis=0)
    q = sqs[blocks].sum(axis=0)
    c = counts[blocks].sum()
    mean = s / c
    var = q / c - mean**2
    sd = np.sqrt(np.maximum(var, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd >= VOL_FLOOR, mean / sd, 0.0)


def pbo_cscv(pnl_matrix, n_splits: int = 10, max_combos: int = 2000) -> dict:
    """Probability of Backtest Overfitting via CSCV (Bailey et al. 2017).

    `pnl_matrix`: shape (T observations, N configurations) of per-period
    returns/PnL, one column per strategy variant. Splits T into `n_splits`
    contiguous blocks, enumerates balanced in-sample/out-of-sample partitions,
    and for each picks the IS-best config and measures its OOS rank. PBO =
    fraction of partitions where the IS-best config lands below the OOS median.

    Uses the unified 2.0 ``Verdict`` vocabulary:
    ``pbo > 0.5 -> "likely_noise"``, ``0.2 < pbo <= 0.5 -> "marginal"``,
    ``pbo <= 0.2 -> "robust"``.
    """
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2 or n_splits % 2 != 0:
        raise ValueError(
            f"n_splits must be an even integer >= 2 (got {n_splits!r}); "
            "CSCV partitions blocks into equal in-sample and out-of-sample halves."
        )
    if pnl_matrix is None:
        return {
            "error": "need (T>=2S observations, N>=2 configs)",
            "shape": [0],
            "engine_version": current_engine_version(),
        }
    M = np.asarray(pnl_matrix, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2 or M.shape[0] < n_splits * 2:
        return {
            "error": "need (T>=2S observations, N>=2 configs)",
            "shape": list(M.shape),
            "engine_version": current_engine_version(),
        }
    T, N = M.shape

    blocks = np.array_split(np.arange(T), n_splits)
    sums = np.vstack([M[b].sum(axis=0) for b in blocks])  # (S, N)
    sqs = np.vstack([(M[b] ** 2).sum(axis=0) for b in blocks])  # (S, N)
    counts = np.array([len(b) for b in blocks], dtype=float)  # (S,)

    block_ids = list(range(n_splits))
    half = n_splits // 2
    combos = list(itertools.combinations(block_ids, half))
    if len(combos) > max_combos:
        step = max(1, len(combos) // max_combos)
        combos = combos[::step][:max_combos]

    logits = []
    for is_blocks in combos:
        oos_blocks = [b for b in block_ids if b not in is_blocks]
        is_perf = _block_sharpe(sums, sqs, counts, list(is_blocks))
        oos_perf = _block_sharpe(sums, sqs, counts, oos_blocks)
        n_star = int(np.argmax(is_perf))
        order = np.argsort(np.argsort(oos_perf))  # ranks 0..N-1
        rank = order[n_star] + 1
        w = rank / (N + 1)
        w = min(max(w, 1e-6), 1 - 1e-6)
        logits.append(math.log(w / (1.0 - w)))

    logits_arr = np.asarray(logits)
    pbo = float(np.mean(logits_arr <= 0.0)) if logits_arr.size else float("nan")
    verdict = (
        Verdict.LIKELY_NOISE.value
        if pbo > 0.5
        else Verdict.MARGINAL.value
        if pbo > 0.2
        else Verdict.ROBUST.value
    )
    n_eff = round(n_eff_participation(M), 4)
    out = {
        "engine_version": current_engine_version(),
        "pbo": round(pbo, 4),
        "n_partitions": int(logits_arr.size),
        "n_configs": int(N),
        "n_eff": n_eff,
        "n_eff_participation": n_eff,
        "n_eff_method": "eigenvalue_participation_ratio",
        "n_splits": int(n_splits),
        "logit_mean": round(float(np.mean(logits_arr)), 4) if logits_arr.size else None,
        "verdict": verdict,
    }
    record(
        "pbo",
        "fraction of CSCV partitions where the in-sample best ranks below the OOS median",
        inputs={"n_configs": int(N), "n_splits": int(n_splits), "n_partitions": int(logits_arr.size)},
        series={"logit": logits_arr},
        result=round(pbo, 4),
    )
    return out


# ── Combinatorial OOS stability (single return stream) ───────────────────────


def combinatorial_oos_stability(
    returns: ArrayLike | Sequence[float],
    *,
    n_groups: int = 8,
    n_test_groups: int = 2,
    purge: int = 1,
    embargo: int = 1,
    n_trials: int = 1,
    max_paths: int = 2000,
    risk_free_rate: float | None = None,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Combinatorial out-of-sample stability over a single pre-computed return stream."""
    arr = _as_1d_float(returns)
    n = arr.size
    if n < 8:
        return {
            "error": "need >= 8 observations",
            "n_obs": int(n),
            "engine_version": current_engine_version(),
        }
    n_groups = int(n_groups)
    n_test_groups = int(n_test_groups)
    if n_groups < 2 or n_groups > n:
        return {
            "error": "need 2 <= n_groups <= n_obs",
            "n_obs": int(n),
            "n_groups": n_groups,
            "engine_version": current_engine_version(),
        }
    if not (1 <= n_test_groups < n_groups):
        return {
            "error": "need 1 <= n_test_groups < n_groups",
            "n_test_groups": n_test_groups,
            "engine_version": current_engine_version(),
        }
    purge = max(0, int(purge))
    embargo = max(0, int(embargo))
    ppy = max(int(periods_per_year), 1)
    annual, _, _ = cash_rate(risk_free_rate)

    groups = list(np.array_split(np.arange(n), n_groups))
    combos = list(itertools.combinations(range(n_groups), n_test_groups))
    if len(combos) > max_paths:
        step = max(1, len(combos) // max_paths)
        combos = combos[::step][:max_paths]

    sharpes: list[float] = []
    dsrs: list[float] = []
    n_short = 0
    for test_ids in combos:
        parts = []
        for gi in test_ids:
            block = groups[gi]
            if purge or embargo:
                hi = block.size - embargo
                block = block[purge:hi] if hi > purge else block[:0]
            if block.size:
                parts.append(block)
        if not parts:
            continue
        path = arr[np.sort(np.concatenate(parts))]
        if path.size < 2:
            continue
        sharpes.append(_per_period_sharpe(path, risk_free_per_period=annual / ppy) * math.sqrt(ppy))
        if path.size >= 8:
            try:
                d = deflated_sharpe(
                    path.tolist(),
                    n_trials=max(1, int(n_trials)),
                    risk_free_rate=risk_free_rate,
                    periods_per_year=ppy,
                )
            except (ValueError, FloatingPointError):
                d = {"error": "dsr undefined for this path"}
            if "error" not in d:
                dsrs.append(float(d["dsr_probability"]))
        else:
            n_short += 1

    if not sharpes:
        return {
            "error": "no evaluable CPCV paths (groups too small after purge/embargo)",
            "n_obs": int(n),
            "n_groups": n_groups,
            "engine_version": current_engine_version(),
        }

    sh = np.asarray(sharpes)
    deflation_applied = bool(int(n_trials) > 1)
    out: dict[str, Any] = {
        "engine_version": current_engine_version(),
        "n_obs": int(n),
        "n_groups": n_groups,
        "n_test_groups": n_test_groups,
        "purge": purge,
        "embargo": embargo,
        "n_trials": int(n_trials),
        "deflation_applied": deflation_applied,
        **rate_fields(risk_free_rate),
        "periods_per_year": ppy,
        "annualization": build_annualization_spec(ppy),
        "risk_free": build_risk_free_spec(risk_free_rate, ppy),
        "costs_model": build_costs_model_spec(mode="pre_netted_returns"),
        "n_paths": int(sh.size),
        "n_backtest_paths_theoretical": int(math.comb(n_groups, n_test_groups) * n_test_groups // n_groups),
        "sharpe_annualized": {
            "mean": round(float(sh.mean()), 4),
            "std": round(float(sh.std(ddof=1)) if sh.size > 1 else 0.0, 4),
            "min": round(float(sh.min()), 4),
            "p25": round(float(np.percentile(sh, 25)), 4),
            "median": round(float(np.percentile(sh, 50)), 4),
            "p75": round(float(np.percentile(sh, 75)), 4),
            "max": round(float(sh.max()), 4),
        },
        "pct_paths_positive": round(float(np.mean(sh > 0)), 4),
        "n_paths_short_for_dsr": int(n_short),
    }
    med_sr = float(np.median(sh))
    if dsrs:
        d = np.asarray(dsrs)
        dsr_dist = {
            "mean": round(float(d.mean()), 4),
            "min": round(float(d.min()), 4),
            "median": round(float(np.percentile(d, 50)), 4),
            "max": round(float(d.max()), 4),
            "pct_paths_robust": round(float(np.mean(d >= 0.9)), 4),
        }
        out["dsr_probability"] = dsr_dist
        out["deflated_sharpe"] = dict(dsr_dist)
        med_dsr = float(np.percentile(d, 50))
        out["verdict"] = (
            Verdict.LIKELY_NOISE.value
            if (med_sr <= 0 or med_dsr < 0.5)
            else Verdict.ROBUST.value
            if (deflation_applied and med_dsr >= 0.9)
            else Verdict.MARGINAL.value
        )
    else:
        out["verdict"] = Verdict.LIKELY_NOISE.value if med_sr <= 0 else Verdict.MARGINAL.value
    record(
        "cpcv_sharpe",
        "annualised Sharpe of each purged and embargoed held-out path",
        inputs={
            "n_groups": n_groups,
            "n_test_groups": n_test_groups,
            "purge": purge,
            "embargo": embargo,
            "n_paths": int(sh.size),
        },
        series={"sharpe_annualized": sh},
        result=out["sharpe_annualized"]["median"],
    )
    return out


def cpcv_score(
    returns: ArrayLike | Sequence[float],
    *,
    n_groups: int = 8,
    n_test_groups: int = 2,
    purge: int = 1,
    embargo: int = 1,
    n_trials: int = 1,
    max_paths: int = 2000,
    risk_free_rate: float | None = None,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Deprecated alias for :func:`combinatorial_oos_stability`."""
    warnings.warn(
        "cpcv_score is deprecated and will be removed in a future major release; "
        "use combinatorial_oos_stability instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return combinatorial_oos_stability(
        returns,
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        purge=purge,
        embargo=embargo,
        n_trials=n_trials,
        max_paths=max_paths,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )


#: Explicit probability-named alias (`QOS-156`): DSR in [0, 1] is a probability, not an annualized Sharpe.
dsr_probability = deflated_sharpe
