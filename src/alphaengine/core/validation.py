"""
Validation tools, the moat: tell the caller when an idea is probably noise.

Lifted verbatim (math-identical) from backend/quant/overfitting.py:
  - Deflated Sharpe Ratio (DSR) + Probabilistic Sharpe Ratio (PSR), Bailey & López de Prado (2014). Corrects the Sharpe for the number of
    trials and for non-normal returns.
  - Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric
    Cross-Validation (CSCV), Bailey, Borwein, López de Prado, Zhu (2017).

Pure numpy/scipy. No look-ahead, no LLM, no I/O. Deterministic given inputs on
the pinned stack. The public names match the build spec's 6-tool beta cut
(`deflated_sharpe`, `pbo_cscv`); behaviour is unchanged from the source.
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import special, stats

from .rates import cash_rate, rate_fields
from .trace import record

_EULER = 0.5772156649015329


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
            k_len = len(p_sorted)
            # Locate the target p-value rank (0-indexed) in p_sorted
            idx = max(0, min(int(trial_rank) - 1, k_len - 1))
            if int(trial_rank) == 1:
                diffs = [abs(p - p_single) for p in p_sorted]
                idx = int(np.argmin(diffs))
            # Step-down Holm and step-up BHY scaled to multiplicity m_f
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
            # Single-strategy summary statistics: Harvey & Liu (2015) eqs. (10)-(14)
            # Construct HL synthetic ordered p-value schedule across M tests when M >= 2
            m_int = max(int(round(m_f)), 1)
            if m_int <= 1:
                p_holm = p_single
                p_bhy = p_single
            else:
                # Synthetic uniform order-statistic p-values anchored at p_(1) = p_single
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


def _per_period_sharpe(returns: np.ndarray, *, risk_free_per_period: float = 0.0) -> float:
    """Mean excess over a constant per-period cash rate, divided by sample vol.

    An explicit 0 is excess of zero. The bill, when one is used, is subtracted
    by the caller before this function sees the series. Vol is the sample
    standard deviation of the returns (ddof=1); subtracting a constant does
    not change it, or the skew, or the kurtosis.
    """
    sd = returns.std(ddof=1)
    if sd == 0 or not math.isfinite(sd):
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
    if n_obs < 2:
        return 0.0
    denom = 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe**2
    if denom <= 0:
        denom = 1e-9
    z = (sharpe - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(denom)
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
) -> dict[str, Any]:
    """Deflated Sharpe Ratio for a return stream.

    DSR = PSR(SR0), where SR0 = expected max Sharpe across `n_trials`. A DSR
    near 1 means the result survives the multiple-testing correction; near 0
    means it's probably noise. `trials_sharpe_std` should come from the
    caller's hypothesis ledger; when absent we estimate it from the result's
    own sampling error (a conservative lower bound on selection variance).
    """
    arr = _as_1d_float(returns)
    n = arr.size
    if n < 8:
        return {"error": "need >= 8 observations", "n_obs": int(n)}

    ppy = max(int(periods_per_year), 1)
    annual, _, _ = cash_rate(risk_free_rate)
    rf_per = annual / ppy
    sr = _per_period_sharpe(arr, risk_free_per_period=rf_per)
    skew = float(stats.skew(arr, bias=False)) if n > 2 else 0.0
    kurt = float(stats.kurtosis(arr, fisher=False, bias=False)) if n > 3 else 3.0
    psr0 = probabilistic_sharpe_ratio(sr, n, skew, kurt, 0.0)

    # Sampling SD of the Sharpe estimator under the null (Lo, 2002), used as
    # fallback only when cross-trial Sharpe dispersion is not supplied (e.g.
    # fewer than 2 valid trials). When trials_sharpe_std > 0 is supplied, use
    # it directly so correlated trials are not over-penalized by the single-series SE floor.
    sr_est_sd = math.sqrt((1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2) / (n - 1)) if n > 1 else 0.0
    std_for_max = (
        float(trials_sharpe_std)
        if (trials_sharpe_std is not None and float(trials_sharpe_std) > 0)
        else sr_est_sd
    )
    n_f = float(n_trials)
    if n_f <= 1.0:
        sr0 = 0.0
        dsr = psr0
        no_deflation_note: str | None = (
            "n_trials=1: no multiple-testing deflation applies; deflated_sharpe equals psr_vs_zero."
        )
    else:
        sr0 = expected_max_sharpe(n_f, std_for_max)
        dsr = probabilistic_sharpe_ratio(sr, n, skew, kurt, sr0)
        no_deflation_note = None

    trials_out: int | float = int(n_f) if n_f.is_integer() else round(n_f, 4)
    out = {
        "n_obs": int(n),
        "n_trials": trials_out,
        "sharpe_per_period": round(sr, 4),
        "sharpe_annualized": round(sr * math.sqrt(ppy), 4),
        **rate_fields(risk_free_rate),
        "periods_per_year": ppy,
        "skew": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "psr_vs_zero": round(psr0, 4),
        "sr0_expected_max": round(sr0, 4),
        "deflated_sharpe": round(dsr, 4),
        "trials_sharpe_std": round(std_for_max, 4),
        # The headline shown INSTEAD of raw Sharpe.
        "verdict": ("likely_noise" if dsr < 0.5 else "marginal" if dsr < 0.9 else "robust"),
    }
    if no_deflation_note is not None:
        out["note"] = no_deflation_note
    record(
        "deflated_sharpe",
        "PSR(SR*) where SR* is the expected maximum Sharpe across n_trials",
        inputs={"n_obs": int(n), "n_trials": trials_out, "sharpe_per_period": round(sr, 4)},
        series={"returns": arr},
        result=round(dsr, 4),
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
    `sr_benchmark` at `confidence`. Answers "is this backtest long enough to trust,
    or do I need more history before believing the Sharpe?"

    MinTRL = 1 + [1 - γ3·SR + ((γ4-1)/4)·SR²] · (Z_α / (SR - SR*))²  (per-period).
    Returns the required count + whether the actual sample already clears it.
    """
    arr = _as_1d_float(returns)
    n = arr.size
    if n < 8:
        return {"error": "need >= 8 observations", "n_obs": int(n)}
    ppy = max(int(periods_per_year), 1)
    annual, _, _ = cash_rate(risk_free_rate)
    sr = _per_period_sharpe(arr, risk_free_per_period=annual / ppy)
    if sr <= sr_benchmark:
        return {
            "n_obs": int(n),
            "sharpe_per_period": round(sr, 4),
            **rate_fields(risk_free_rate),
            "periods_per_year": ppy,
            "min_track_record_length": None,
            "sufficient": False,
            "note": "observed Sharpe <= benchmark; not distinguishable at any length",
        }
    skew = float(stats.skew(arr, bias=False)) if n > 2 else 0.0
    kurt = float(stats.kurtosis(arr, fisher=False, bias=False)) if n > 3 else 3.0
    z = float(stats.norm.ppf(confidence))
    mintrl = 1.0 + (1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2) * (z / (sr - sr_benchmark)) ** 2
    mintrl = max(1.0, float(mintrl))
    out = {
        "n_obs": int(n),
        "sharpe_per_period": round(sr, 4),
        "confidence": confidence,
        "min_track_record_length": round(mintrl, 1),
        "min_track_record_years": round(mintrl / ppy, 2),
        **rate_fields(risk_free_rate),
        "periods_per_year": ppy,
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
        return np.where(sd > 0, mean / sd, 0.0)


def pbo_cscv(pnl_matrix, n_splits: int = 10, max_combos: int = 2000) -> dict:
    """Probability of Backtest Overfitting via CSCV (Bailey et al. 2017).

    `pnl_matrix`: shape (T observations, N configurations) of per-period
    returns/PnL, one column per strategy variant. Splits T into `n_splits`
    contiguous blocks, enumerates balanced in-sample/out-of-sample partitions,
    and for each picks the IS-best config and measures its OOS rank. PBO =
    fraction of partitions where the IS-best config lands below the OOS median.
    """
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2 or n_splits % 2 != 0:
        raise ValueError(
            f"n_splits must be an even integer >= 2 (got {n_splits!r}); "
            "CSCV partitions blocks into equal in-sample and out-of-sample halves."
        )
    if pnl_matrix is None:
        return {"error": "need (T>=2S observations, N>=2 configs)", "shape": [0]}
    M = np.asarray(pnl_matrix, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2 or M.shape[0] < n_splits * 2:
        return {"error": "need (T>=2S observations, N>=2 configs)", "shape": list(M.shape)}
    T, N = M.shape

    blocks = np.array_split(np.arange(T), n_splits)
    # Precompute per-block moments per config.
    sums = np.vstack([M[b].sum(axis=0) for b in blocks])  # (S, N)
    sqs = np.vstack([(M[b] ** 2).sum(axis=0) for b in blocks])  # (S, N)
    counts = np.array([len(b) for b in blocks], dtype=float)  # (S,)

    block_ids = list(range(n_splits))
    half = n_splits // 2
    combos = list(itertools.combinations(block_ids, half))
    if len(combos) > max_combos:
        # Deterministic subsample to bound cost on large S.
        step = max(1, len(combos) // max_combos)
        combos = combos[::step][:max_combos]

    logits = []
    for is_blocks in combos:
        oos_blocks = [b for b in block_ids if b not in is_blocks]
        is_perf = _block_sharpe(sums, sqs, counts, list(is_blocks))
        oos_perf = _block_sharpe(sums, sqs, counts, oos_blocks)
        n_star = int(np.argmax(is_perf))
        # Rank of the IS-best config OOS (1 = worst .. N = best).
        order = np.argsort(np.argsort(oos_perf))  # ranks 0..N-1
        rank = order[n_star] + 1
        w = rank / (N + 1)
        w = min(max(w, 1e-6), 1 - 1e-6)
        logits.append(math.log(w / (1.0 - w)))

    logits_arr = np.asarray(logits)
    pbo = float(np.mean(logits_arr <= 0.0)) if logits_arr.size else float("nan")
    out = {
        "pbo": round(pbo, 4),
        "n_partitions": int(logits_arr.size),
        "n_configs": int(N),
        "n_splits": int(n_splits),
        "logit_mean": round(float(np.mean(logits_arr)), 4) if logits_arr.size else None,
        "verdict": ("overfit" if pbo > 0.5 else "acceptable" if pbo > 0.2 else "robust"),
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
    """Combinatorial out-of-sample stability over a single pre-computed return stream.

    WHAT THIS MEASURES (AND HOW IT DIFFERS FROM FULL CPCV):
    True Combinatorial Purged Cross-Validation (López de Prado, 2018) fits a model
    on each training partition, purges/embargoes the training labels around test
    boundaries, predicts on the held-out test groups, and stitches test predictions
    into full-length backtest paths.

    Because AlphaEngine evaluates an already-computed return stream without a
    re-trainable model, this function measures **combinatorial out-of-sample
    stability**: it splits the `T` returns into `n_groups` contiguous blocks,
    enumerates `C(n_groups, n_test_groups)` held-out block combinations, trims
    `purge` observations from the leading edge and `embargo` observations from
    the trailing edge of each held-out block, concatenates the held-out blocks
    in time order, and computes the distribution of annualized Sharpe and
    deflated Sharpe (haircut by `n_trials`) across all held-out sub-sample
    combinations.
    """
    arr = _as_1d_float(returns)
    n = arr.size
    if n < 8:
        return {"error": "need >= 8 observations", "n_obs": int(n)}
    n_groups = int(n_groups)
    n_test_groups = int(n_test_groups)
    if n_groups < 2 or n_groups > n:
        return {"error": "need 2 <= n_groups <= n_obs", "n_obs": int(n), "n_groups": n_groups}
    if not (1 <= n_test_groups < n_groups):
        return {"error": "need 1 <= n_test_groups < n_groups", "n_test_groups": n_test_groups}
    purge = max(0, int(purge))
    embargo = max(0, int(embargo))

    groups = list(np.array_split(np.arange(n), n_groups))
    combos = list(itertools.combinations(range(n_groups), n_test_groups))
    if len(combos) > max_paths:  # deterministic subsample to bound cost
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
        ppy = max(int(periods_per_year), 1)
        annual, _, _ = cash_rate(risk_free_rate)
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
                d = {"error": "dsr undefined for this path"}  # e.g. degenerate/near-riskless path
            if "error" not in d:
                dsrs.append(d["deflated_sharpe"])
        else:
            n_short += 1

    if not sharpes:
        return {
            "error": "no evaluable CPCV paths (groups too small after purge/embargo)",
            "n_obs": int(n),
            "n_groups": n_groups,
        }

    sh = np.asarray(sharpes)
    out = {
        "n_obs": int(n),
        "n_groups": n_groups,
        "n_test_groups": n_test_groups,
        "purge": purge,
        "embargo": embargo,
        "n_trials": int(n_trials),
        **rate_fields(risk_free_rate),
        "periods_per_year": max(int(periods_per_year), 1),
        "n_paths": int(sh.size),
        # LdP's reconstructed-path count, reported for reference.
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
        out["deflated_sharpe"] = {
            "mean": round(float(d.mean()), 4),
            "min": round(float(d.min()), 4),
            "median": round(float(np.percentile(d, 50)), 4),
            "max": round(float(d.max()), 4),
            "pct_paths_robust": round(float(np.mean(d >= 0.9)), 4),
        }
        med_dsr = float(np.median(d))
        out["verdict"] = (
            "likely_noise"
            if (med_sr <= 0 or med_dsr < 0.5)
            else "robust"
            if med_dsr >= 0.9
            else "inconclusive"
        )
    else:
        out["verdict"] = "likely_noise" if med_sr <= 0 else "inconclusive"
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
    """Deprecated alias for :func:`combinatorial_oos_stability`.

    Measures combinatorial out-of-sample sub-sample stability on a single return
    stream (no model re-training, paths are held-out block combinations with
    purge/embargo trimmed from each test block).
    """
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
