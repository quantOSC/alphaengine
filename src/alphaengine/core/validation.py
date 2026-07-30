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

import numpy as np
from scipy import stats

_EULER = 0.5772156649015329


# ── Sharpe statistics ───────────────────────────────────────────────────


def _per_period_sharpe(returns: np.ndarray) -> float:
    sd = returns.std(ddof=1)
    if sd == 0 or not math.isfinite(sd):
        return 0.0
    return float(returns.mean() / sd)


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


def expected_max_sharpe(n_trials: int, trials_sharpe_std: float) -> float:
    """E[max Sharpe] over N independent trials of noise (per-period units).

    SR0 = σ_SR · [ (1-γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ].
    The benchmark a *real* strategy must beat to be non-spurious.
    """
    if n_trials < 2 or trials_sharpe_std <= 0:
        return 0.0
    a = stats.norm.ppf(1.0 - 1.0 / n_trials)
    b = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(trials_sharpe_std * ((1.0 - _EULER) * a + _EULER * b))


def deflated_sharpe(
    returns: list[float],
    *,
    n_trials: int,
    trials_sharpe_std: float | None = None,
) -> dict:
    """Deflated Sharpe Ratio for a return stream.

    DSR = PSR(SR0), where SR0 = expected max Sharpe across `n_trials`. A DSR
    near 1 means the result survives the multiple-testing correction; near 0
    means it's probably noise. `trials_sharpe_std` should come from the
    caller's hypothesis ledger; when absent we estimate it from the result's
    own sampling error (a conservative lower bound on selection variance).
    """
    arr = np.asarray([float(r) for r in (returns or []) if r is not None], dtype=float)
    n = arr.size
    if n < 8:
        return {"error": "need >= 8 observations", "n_obs": int(n)}

    sr = _per_period_sharpe(arr)
    skew = float(stats.skew(arr, bias=False)) if n > 2 else 0.0
    kurt = float(stats.kurtosis(arr, fisher=False, bias=False)) if n > 3 else 3.0
    psr0 = probabilistic_sharpe_ratio(sr, n, skew, kurt, 0.0)

    # Sampling SD of the Sharpe estimator under the null (Lo, 2002), a floor
    # for trial-Sharpe dispersion when the caller doesn't supply one.
    sr_est_sd = math.sqrt((1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2) / (n - 1)) if n > 1 else 0.0
    std_for_max = trials_sharpe_std if (trials_sharpe_std and trials_sharpe_std > 0) else sr_est_sd
    sr0 = expected_max_sharpe(max(2, n_trials), std_for_max)
    dsr = probabilistic_sharpe_ratio(sr, n, skew, kurt, sr0)

    return {
        "n_obs": int(n),
        "n_trials": int(n_trials),
        "sharpe_per_period": round(sr, 4),
        "sharpe_annualized": round(sr * math.sqrt(252), 4),
        "skew": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "psr_vs_zero": round(psr0, 4),
        "sr0_expected_max": round(sr0, 4),
        "deflated_sharpe": round(dsr, 4),
        "trials_sharpe_std": round(std_for_max, 4),
        # The headline shown INSTEAD of raw Sharpe.
        "verdict": ("likely_noise" if dsr < 0.5 else "marginal" if dsr < 0.9 else "robust"),
    }


def min_track_record_length(
    returns: list[float],
    *,
    sr_benchmark: float = 0.0,
    confidence: float = 0.95,
) -> dict:
    """Minimum Track Record Length (Bailey & López de Prado, 2012): the number of
    observations needed for the observed Sharpe to be statistically greater than
    `sr_benchmark` at `confidence`. Answers "is this backtest long enough to trust,
    or do I need more history before believing the Sharpe?"

    MinTRL = 1 + [1 - γ3·SR + ((γ4-1)/4)·SR²] · (Z_α / (SR - SR*))²  (per-period).
    Returns the required count + whether the actual sample already clears it.
    """
    arr = np.asarray([float(r) for r in (returns or []) if r is not None], dtype=float)
    n = arr.size
    if n < 8:
        return {"error": "need >= 8 observations", "n_obs": int(n)}
    sr = _per_period_sharpe(arr)
    if sr <= sr_benchmark:
        return {
            "n_obs": int(n),
            "sharpe_per_period": round(sr, 4),
            "min_track_record_length": None,
            "sufficient": False,
            "note": "observed Sharpe <= benchmark; not distinguishable at any length",
        }
    skew = float(stats.skew(arr, bias=False)) if n > 2 else 0.0
    kurt = float(stats.kurtosis(arr, fisher=False, bias=False)) if n > 3 else 3.0
    z = float(stats.norm.ppf(confidence))
    mintrl = 1.0 + (1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2) * (z / (sr - sr_benchmark)) ** 2
    mintrl = max(1.0, float(mintrl))
    return {
        "n_obs": int(n),
        "sharpe_per_period": round(sr, 4),
        "confidence": confidence,
        "min_track_record_length": round(mintrl, 1),
        "min_track_record_years": round(mintrl / 252.0, 2),
        "sufficient": bool(n >= mintrl),
        "shortfall_obs": max(0, int(math.ceil(mintrl - n))),
    }


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
    M = np.asarray(pnl_matrix, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2 or M.shape[0] < n_splits * 2:
        return {"error": "need (T>=2S observations, N>=2 configs)", "shape": list(M.shape)}
    T, N = M.shape
    if n_splits % 2 != 0:
        n_splits -= 1  # CSCV needs an even number of blocks to halve

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
    return {
        "pbo": round(pbo, 4),
        "n_partitions": int(logits_arr.size),
        "n_configs": int(N),
        "n_splits": int(n_splits),
        "logit_mean": round(float(np.mean(logits_arr)), 4) if logits_arr.size else None,
        "verdict": ("overfit" if pbo > 0.5 else "acceptable" if pbo > 0.2 else "robust"),
    }


# ── CPCV, Combinatorial Purged Cross-Validation (single return stream) ──────


def cpcv_score(
    returns: list[float],
    *,
    n_groups: int = 8,
    n_test_groups: int = 2,
    purge: int = 1,
    embargo: int = 1,
    n_trials: int = 1,
    max_paths: int = 2000,
) -> dict:
    """Combinatorial Purged Cross-Validation (López de Prado, 2018) on ONE return
    stream, the robustness read a single backtest Sharpe hides.

    Splits the T returns into `n_groups` contiguous groups and enumerates every
    C(n_groups, n_test_groups) way of holding out `n_test_groups` groups as an
    out-of-sample (OOS) test PATH. Each held-out block is PURGED (`purge` obs
    dropped at its leading edge, where a multi-bar return label could overlap the
    preceding training block) and EMBARGOED (`embargo` obs dropped at its trailing
    edge). For every path we measure the OOS Sharpe and, when the path is long
    enough (>= 8 obs), its deflated Sharpe (haircut by `n_trials`). The output is
    the DISTRIBUTION of those metrics across all paths.

    This reuses the combinatorial-block machinery of `pbo_cscv` (itertools
    combinations of contiguous blocks) but differs in purpose: pbo_cscv needs a
    matrix of competing configs to measure SELECTION overfitting; cpcv_score
    scores a SINGLE strategy's stability across many purged held-out partitions.
    A 2024 controlled study (ScienceDirect S0950705124011110) finds CPCV superior
    to K-Fold / Purged-K-Fold / Walk-Forward on both PBO and the DSR test stat.

    Purge/embargo are effectively no-ops for the non-overlapping 1-bar returns the
    backtester emits, but are applied so overlapping-label callers are covered.
    """
    arr = np.asarray([float(r) for r in (returns or []) if r is not None], dtype=float)
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
        sharpes.append(_per_period_sharpe(path) * math.sqrt(252))
        if path.size >= 8:
            try:
                d = deflated_sharpe(path.tolist(), n_trials=max(1, int(n_trials)))
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
    return out
