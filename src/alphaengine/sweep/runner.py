"""The grid runner.

WHAT THIS DELIBERATELY DOES NOT DO
    It does not backtest. `sweep()` takes YOUR function and calls it once per
    parameter combination. We orchestrate and measure; you simulate. Shipping a
    backtester would mean competing with the engine you already trust, and
    inheriting responsibility for its correctness, which is not a trade worth
    making for a library whose entire claim is that its numbers are reliable.

WHY THE TRIAL COUNT IS NOT A PARAMETER
    Because a parameter is a place to be optimistic. `n_trials` is `len(grid)`,
    computed from the grid actually iterated — including failed parameter
    combinations (`QOS-153`) — and there is no argument to override it. That is
    the single design decision this module exists for.
"""

from __future__ import annotations

import itertools
import math
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .._version import current_engine_version
from ..core import (
    deflated_sharpe,
    min_track_record_length,
    n_eff_participation,
    pbo_cscv,
    performance_report,
)
from ..core.conventions import (
    Verdict,
    build_annualization_spec,
    build_costs_model_spec,
    build_risk_free_spec,
)
from ..core.rates import cash_rate
from ..core.validation import VOL_FLOOR

__all__ = ["sweep", "SweepResult", "Trial"]

# A parameter set is a plain dict; a returns series is a sequence of floats.
Params = Mapping[str, Any]
Returns = Sequence[float]
BacktestFn = Callable[..., Returns]


def _expand(grid: Mapping[str, Iterable[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of the grid, in a deterministic order.

    Sorted by key so the same grid produces the same ordering on any machine and
    any Python version, the trial index is part of what gets recorded, so it
    cannot depend on dict iteration order.
    """
    keys = sorted(grid)
    if not keys:
        # itertools.product() with no iterables yields ONE empty tuple, so an
        # empty grid would quietly run once and report n_trials=1. That is
        # exactly the flattering count this module exists to prevent, so it has
        # to fail rather than pass.
        return []
    values = [list(grid[k]) for k in keys]
    empty = [k for k, v in zip(keys, values, strict=True) if not v]
    if empty:
        # One empty axis makes the whole product empty. A caller who wrote
        # {"fast": []} meant to sweep something and should hear about it.
        raise ValueError(f"grid axis has no values: {empty}")
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]


def _hash_data(data: Any) -> str:
    """Full 64-hex sha256 content hash of the input series via `canonical_sha256`."""
    from ..study.schema import canonical_sha256

    try:
        arr = np.asarray(data, dtype=float)
        return canonical_sha256(arr.tolist())
    except (TypeError, ValueError):
        return canonical_sha256(data)


def _hash_params(p: Params) -> str:
    from ..study.schema import canonical_sha256

    return canonical_sha256(dict(p))[:12]


def _sharpe(
    returns: npt.NDArray[np.float64],
    *,
    risk_free_rate: float | None = None,
    periods_per_year: int = 252,
) -> float:
    """Per-period excess Sharpe ratio aligned with `cash_rate()` and `VOL_FLOOR` (`QOS-156`)."""
    if returns.size <= 1:
        return 0.0
    ppy = max(int(periods_per_year), 1)
    annual_rf, _, _ = cash_rate(risk_free_rate)
    excess = returns - (annual_rf / ppy)
    sd = float(excess.std(ddof=1))
    return float(excess.mean() / sd) if sd >= VOL_FLOOR else 0.0


@dataclass
class Trial:
    """One executed parameter attempt and what it produced."""

    index: int
    params: dict[str, Any]
    params_hash: str
    n_obs: int
    sharpe: float
    sharpe_annualized: float
    total_return_pct: float
    failed: str | None = None
    attempt_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    executed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class SweepResult:
    """Everything the sweep saw. The trial matrix is the irreplaceable part."""

    trials: list[Trial]
    matrix: npt.NDArray[np.float64] = field(repr=False)  # (T observations, N_ok configurations)
    data_hash: str = ""
    grid_keys: list[str] = field(default_factory=list)
    store_params: bool = False
    run_id: str | None = None
    cumulative_trials: int | None = None
    project_trials: int | None = None
    risk_free_rate: float | None = None
    periods_per_year: int = 252

    @property
    def n_trials(self) -> int:
        """Derived, never supplied: all executed grid attempts, including failures (`QOS-153`)."""
        return len(self.trials)

    @property
    def best(self) -> Trial:
        return max((t for t in self.trials if t.failed is None), key=lambda t: t.sharpe)

    def verdict(
        self,
        *,
        risk_free_rate: float | None = None,
        periods_per_year: int | None = None,
    ) -> dict[str, Any]:
        """Deflate the best result for the full search that actually executed.

        Per William's trial counting rule (`QOS-153`), every executed grid
        combination — including failed ones — counts toward `n_trials`.
        Cross-trial Sharpe dispersion (`trials_sharpe_std`) is estimated over
        the surviving series when `>= 2` configurations succeeded.

        PBO is reported alongside but is NOT the gate. It answers a different
        question, "was the choice AMONG configurations informative?", and
        returns roughly 0.5 for genuinely near-tied top configs even when the
        underlying edge is real.
        """
        rf_eff = self.risk_free_rate if risk_free_rate is None else risk_free_rate
        ppy_eff = max(int(self.periods_per_year if periods_per_year is None else periods_per_year), 1)
        best = self.best
        ok = [t for t in self.trials if t.failed is None]
        col = self.matrix[:, ok.index(best)]
        trials_sharpe_std: float | None = None
        if len(ok) >= 2:
            dispersion = float(np.std([t.sharpe for t in ok], ddof=1))
            if math.isfinite(dispersion) and dispersion > 0:
                trials_sharpe_std = dispersion
        total_attempted = len(self.trials)
        n_ok = len(ok)
        n_failed = total_attempted - n_ok
        effective_n = (
            max(total_attempted, int(self.cumulative_trials))
            if self.cumulative_trials is not None
            else total_attempted
        )
        n_eff = round(n_eff_participation(self.matrix), 4)
        dsr = deflated_sharpe(
            col.tolist(),
            n_trials=effective_n,
            trials_sharpe_std=trials_sharpe_std,
            risk_free_rate=rf_eff,
            periods_per_year=ppy_eff,
        )
        dsr_neff = deflated_sharpe(
            col.tolist(),
            n_trials=n_eff,
            trials_sharpe_std=trials_sharpe_std,
            risk_free_rate=rf_eff,
            periods_per_year=ppy_eff,
        )
        out: dict[str, Any] = {
            "engine_version": current_engine_version(),
            "n_trials": effective_n,
            "run_n_trials": total_attempted,
            "n_ok_trials": n_ok,
            "n_failed_trials": n_failed,
            "n_trials_source": "derived_from_grid",
            "deflation_applied": dsr.get("deflation_applied", effective_n > 1),
            "n_eff": n_eff,
            "n_eff_participation": n_eff,
            "n_eff_method": "eigenvalue_participation_ratio",
            "deflated_sharpe_at_n_eff": dsr_neff.get("deflated_sharpe"),
            "dsr_probability_at_n_eff": dsr_neff.get("dsr_probability"),
            "n_eff_diagnostic_note": (
                "diagnostic only; headline dsr_probability uses n_trials with cross-trial trials_sharpe_std"
            ),
            "best_trial_index": best.index,
            "best_params_hash": best.params_hash,
            "data_hash": self.data_hash,
            "dsr_probability": dsr.get("dsr_probability"),
            "deflated_sharpe": dsr.get("deflated_sharpe"),
            "verdict": dsr.get("verdict"),
            "psr_vs_zero": dsr.get("psr_vs_zero"),
            "sr0_expected_max": dsr.get("sr0_expected_max"),
            "performance": performance_report(
                col.tolist(),
                risk_free_rate=rf_eff,
                periods_per_year=ppy_eff,
            ),
            "annualization": build_annualization_spec(ppy_eff),
            "risk_free": build_risk_free_spec(rf_eff, ppy_eff),
            "costs_model": build_costs_model_spec(mode="user_backtest_fn"),
        }
        if self.store_params:
            out["best_params"] = best.params
        if self.run_id is not None:
            out["run_id"] = self.run_id
        if self.cumulative_trials is not None:
            out["cumulative_trials"] = effective_n
            out["project_trials"] = self.project_trials
            out["cumulative_dsr_probability"] = dsr.get("dsr_probability")
            out["cumulative_deflated_sharpe"] = dsr.get("deflated_sharpe")
            out["cumulative_verdict"] = dsr.get("verdict")

        # Minimum track record length: how long this record must be before the
        # result is distinguishable from luck. Bites universally, and converts
        # into a dated obligation rather than a pass/fail insult.
        try:
            out["min_track_record_length"] = min_track_record_length(
                col.tolist(),
                risk_free_rate=rf_eff,
                periods_per_year=ppy_eff,
            )
        except Exception:  # noqa: BLE001 - never fail a verdict on an optional figure
            out["min_track_record_length"] = None

        # PBO needs at least two configurations; a single-point "sweep" cannot
        # say anything about selection, and should say so rather than emit a
        # number that looks like an answer.
        if self.matrix.shape[1] >= 2:
            out["selection"] = {
                "question": "was the choice among configurations informative?",
                "not_a_verdict_on_the_edge": True,
                **pbo_cscv(self.matrix.tolist()),
            }
        return out

    def ledger(self) -> list[dict[str, Any]]:
        """Per-trial ledger export: attempt_id + parameter hash + Sharpe + status per trial."""
        return [
            {
                "attempt_id": t.attempt_id,
                "executed_at": t.executed_at,
                "index": t.index,
                "params_hash": t.params_hash,
                "status": "failed" if t.failed is not None else "ok",
                "n_obs": t.n_obs,
                "sharpe": None if t.failed is not None else t.sharpe,
                "sharpe_annualized": None if t.failed is not None else t.sharpe_annualized,
                "total_return_pct": None if t.failed is not None else t.total_return_pct,
                "failed": t.failed,
            }
            for t in self.trials
        ]

    def save(
        self,
        path: str | Path = "study.json",
        *,
        label: str = "",
        data_description: str = "",
        notes: str = "",
        include_ledger: bool = False,
    ) -> Path:
        """Write the study to disk. Local file, no account, no upload."""
        from ..study import Study
        from ..study import save as _save

        study = Study.from_sweep(
            self,
            label=label,
            data_description=data_description,
            notes=notes,
            include_ledger=include_ledger,
        )
        return _save(study, path)

    def surface(self) -> dict[str, Any]:
        """The neighbourhood: plateau or knife edge.

        The payoff of running a grid, and the reason this is a coaching output
        rather than a refereeing one, a researcher ends the session knowing
        where to re-centre, not merely that their number was flattered.
        """
        ok = [t for t in self.trials if t.failed is None]
        if not ok:
            return {"shape": "empty", "n_ok": 0, "n_failed": len(self.trials)}

        sharpes = np.array([t.sharpe for t in ok], dtype=float)
        best = float(sharpes.max())
        median = float(np.median(sharpes))
        # Share of the grid that holds up near the best result. A broad plateau
        # is robustness; one spike is a result fitted to its own parameters.
        near = float((sharpes >= best * 0.8).mean()) if best > 0 else 0.0
        shape = "plateau" if near >= 0.30 else "ridge" if near >= 0.10 else "knife_edge"

        # Where the robust region sits, per parameter, the actionable half.
        centre: dict[str, Any] = {}
        if self.store_params:
            strong = [t for t in ok if best > 0 and t.sharpe >= best * 0.8]
            for k in self.grid_keys:
                vals = [t.params.get(k) for t in strong]
                numeric = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
                if numeric:
                    centre[k] = {"median": float(np.median(numeric)), "range": [min(numeric), max(numeric)]}

        return {
            "shape": shape,
            "n_ok": len(ok),
            "n_failed": len(self.trials) - len(ok),
            "best_sharpe": round(best, 4),
            "median_sharpe": round(median, 4),
            "share_within_20pct_of_best": round(near, 4),
            "plateau_centre": centre or None,
            "reading": {
                "plateau": "A broad region performs. The result does not depend on the exact parameters.",
                "ridge": "A narrow region performs. Sensitive to the parameters; treat with care.",
                "knife_edge": "One configuration performs and its neighbours do not. "
                "Usually a fitted result.",
            }[shape],
        }


def sweep(
    backtest_fn: BacktestFn,
    grid: Mapping[str, Iterable[Any]],
    *,
    data: Any = None,
    store_params: bool = False,
    on_error: str = "record",
    jobs: int = 1,
    risk_free_rate: float | None = None,
    periods_per_year: int = 252,
) -> SweepResult:
    """Run `backtest_fn` once per combination in `grid`.

    Args:
        backtest_fn: YOUR backtest. Called as ``backtest_fn(data=data, **params)``
            when `data` is given, otherwise ``backtest_fn(**params)``. Must
            return a sequence of per-period returns.
        grid: parameter name -> values to try. The cartesian product is the
            search, and its length is the trial count.
        data: passed through untouched. Only hashed, never inspected or stored.
        store_params: keep the parameter values in the result. OFF by default.
            THE GRID IS OFTEN BIGGER IP THAN THE RETURN SERIES: "it uploads my
            parameter search" ends a conversation faster than "it uploads my
            returns". Hashes are always kept, so runs remain comparable without
            the values leaving.
        on_error: "record" marks a failing combination and continues (the
            default, because one bad corner of a grid should not lose the
            other ninety-nine results); "raise" propagates.
        jobs: worker count. Default 1 preserves order and the golden figures.
            Greater than 1 runs combinations concurrently but still records
            each trial at its original index, including failures (0.3.0).
        risk_free_rate: annual cash rate subtracted per period (`annual / periods_per_year`);
            defaults to the pinned 3-month Treasury bill (`core.rates.cash_rate`).
        periods_per_year: trading periods per year (default 252).

    Returns:
        SweepResult, holding the full (T x N_ok) trial matrix that PBO needs and
        every executed `Trial` (including failures) for DSR trial accounting.
    """
    if not callable(backtest_fn):
        raise TypeError(f"backtest_fn must be callable, got {type(backtest_fn).__name__}")
    combos = _expand(grid)
    if not combos:
        raise ValueError("grid is empty: nothing to sweep")
    ppy = max(int(periods_per_year), 1)

    data_hash = _hash_data(data) if data is not None else ""

    def _run_one(i: int, params: dict[str, Any]) -> tuple[int, Trial, npt.NDArray[np.float64] | None]:
        attempt_id = str(uuid.uuid4())
        executed_at = datetime.now(timezone.utc).isoformat()
        try:
            raw = backtest_fn(data=data, **params) if data is not None else backtest_fn(**params)
            r = np.asarray(list(raw), dtype=float)
            if r.ndim != 1 or r.size == 0:
                raise ValueError(f"backtest_fn returned shape {r.shape}; expected a 1-D return series")
            if not np.all(np.isfinite(r)):
                r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as exc:  # noqa: BLE001
            failed_trial = Trial(
                index=i,
                params=dict(params),
                params_hash=_hash_params(params),
                n_obs=0,
                sharpe=0.0,
                sharpe_annualized=0.0,
                total_return_pct=0.0,
                failed=f"{type(exc).__name__}: {exc}",
                attempt_id=attempt_id,
                executed_at=executed_at,
            )
            if on_error == "raise":
                _maybe_record_failed_project_sweep(
                    [failed_trial],
                    grid,
                    data_hash=data_hash,
                    backtest_fn=backtest_fn,
                    risk_free_rate=risk_free_rate,
                    periods_per_year=ppy,
                )
                raise
            return (i, failed_trial, None)
        sr = _sharpe(r, risk_free_rate=risk_free_rate, periods_per_year=ppy)
        trial = Trial(
            index=i,
            params=dict(params),
            params_hash=_hash_params(params),
            n_obs=int(r.size),
            sharpe=round(sr, 6),
            sharpe_annualized=round(sr * math.sqrt(ppy), 4),
            total_return_pct=round((float(np.prod(1 + r)) - 1) * 100, 4),
            attempt_id=attempt_id,
            executed_at=executed_at,
        )
        return i, trial, r

    workers = max(1, int(jobs))
    slots: list[tuple[Trial, npt.NDArray[np.float64] | None] | None] = [None] * len(combos)
    if workers == 1:
        for i, params in enumerate(combos):
            _, trial, col = _run_one(i, params)
            slots[i] = (trial, col)
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_run_one, i, params) for i, params in enumerate(combos)]
            for fut in futs:
                i, trial, col = fut.result()
                slots[i] = (trial, col)

    ordered = [slot for slot in slots if slot is not None]
    trials = [t for t, _ in ordered]
    columns = [c for _, c in ordered if c is not None]

    if not columns:
        _maybe_record_failed_project_sweep(
            trials,
            grid,
            data_hash=data_hash,
            backtest_fn=backtest_fn,
            risk_free_rate=risk_free_rate,
            periods_per_year=ppy,
        )
        raise RuntimeError(
            f"every one of the {len(combos)} combinations failed. First error: {trials[0].failed}"
        )

    # Ragged output means the configurations are not comparable, and silently
    # truncating would produce a PBO over series that do not line up in time.
    lengths = {c.size for c in columns}
    if len(lengths) > 1:
        n = min(lengths)
        columns = [c[-n:] for c in columns]

    result = SweepResult(
        trials=trials,
        matrix=np.column_stack(columns),
        data_hash=data_hash,
        grid_keys=sorted(grid),
        store_params=store_params,
        risk_free_rate=risk_free_rate,
        periods_per_year=ppy,
    )
    _maybe_record_project_sweep(result, grid, backtest_fn=backtest_fn)
    return result


def _maybe_record_failed_project_sweep(
    trials: list[Trial],
    grid: Mapping[str, Iterable[Any]],
    *,
    data_hash: str,
    backtest_fn: Any = None,
    risk_free_rate: float | None = None,
    periods_per_year: int = 252,
) -> None:
    """Record executed failed trial attempts in `.alphaengine/` before raising (`QOS-153`)."""
    from ..ledger import LedgerError, is_auto_sweep_suppressed, record_run, record_trials
    from ..project import find_project

    if is_auto_sweep_suppressed():
        return
    proj = find_project()
    if proj is None:
        return
    grid_keys = sorted(str(k) for k in grid)
    grid_serializable = {str(k): list(v) for k, v in grid.items()}
    family = f"sweep:{','.join(grid_keys)}" if grid_keys else "sweep"
    trial_entries = [
        {
            "attempt_id": t.attempt_id,
            "executed_at": t.executed_at,
            "index": t.index,
            "params_hash": t.params_hash,
            "status": "failed",
            "n_obs": 0,
            "sharpe": None,
            "sharpe_annualized": None,
            "total_return_pct": None,
            "failed": t.failed,
        }
        for t in trials
    ]
    try:
        rec = record_run(
            proj,
            command="sweep",
            params={"grid": grid_serializable, "grid_keys": grid_keys},
            dataset_hashes=[data_hash] if data_hash else [],
            n_trials=len(trials),
            n_trials_source="derived_from_grid",
            figures={
                "n_trials": len(trials),
                "run_n_trials": len(trials),
                "n_ok_trials": 0,
                "n_failed_trials": len(trials),
                "n_trials_source": "derived_from_grid",
                "surface": {"shape": "empty", "n_ok": 0, "n_failed": len(trials)},
                "failed": trials[0].failed if trials else "all_failed",
            },
            verdict=Verdict.UNVERIFIED.value,
            family=family,
            code_target=backtest_fn,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )
        record_trials(
            proj,
            run_id=str(rec["run_id"]),
            trials=trial_entries,
            family=family,
            timestamp=str(rec["timestamp"]),
        )
    except LedgerError:
        raise
    except OSError as exc:
        raise LedgerError(
            f"Failed to record failed sweep in project ledger at {proj.dot_dir}: {exc}"
        ) from exc


def _maybe_record_project_sweep(
    result: SweepResult,
    grid: Mapping[str, Iterable[Any]],
    *,
    backtest_fn: Any = None,
) -> None:
    """When `sweep()` runs inside a `.alphaengine/` project, record the run and all attempted trials."""
    from dataclasses import asdict

    from ..ledger import LedgerError, is_auto_sweep_suppressed, record_run, record_trials
    from ..project import find_project

    if is_auto_sweep_suppressed():
        return
    proj = find_project()
    if proj is None:
        return
    try:
        grid_serializable = {str(k): list(v) for k, v in grid.items()}
        family = f"sweep:{','.join(result.grid_keys)}" if result.grid_keys else "sweep"
        best = result.best
        ok = [t for t in result.trials if t.failed is None]
        best_col = result.matrix[:, ok.index(best)].tolist()
        disp = float(np.std([t.sharpe for t in ok], ddof=1)) if len(ok) >= 2 else None
        trials_std = disp if (disp is not None and math.isfinite(disp) and disp > 0) else None
        v = result.verdict()
        matrix_sha = proj.put_object(
            {
                "trials": [asdict(t) for t in result.trials],
                "matrix": result.matrix.tolist(),
                "data_hash": result.data_hash,
                "grid_keys": result.grid_keys,
                "store_params": result.store_params,
                "risk_free_rate": result.risk_free_rate,
                "periods_per_year": result.periods_per_year,
            }
        )
        rec = record_run(
            proj,
            command="sweep",
            params={"grid": grid_serializable, "grid_keys": result.grid_keys},
            dataset_hashes=[result.data_hash] if result.data_hash else [],
            n_trials=len(result.trials),
            n_trials_source="derived_from_grid",
            figures={
                "n_trials": v["n_trials"],
                "run_n_trials": v["run_n_trials"],
                "n_ok_trials": v["n_ok_trials"],
                "n_failed_trials": v["n_failed_trials"],
                "n_trials_source": v["n_trials_source"],
                "deflation_applied": v.get("deflation_applied"),
                "n_eff_participation": v.get("n_eff_participation"),
                "deflated_sharpe_at_n_eff": v.get("deflated_sharpe_at_n_eff"),
                "n_eff_diagnostic_note": v.get("n_eff_diagnostic_note"),
                "dsr_probability": v.get("dsr_probability"),
                "deflated_sharpe": v.get("deflated_sharpe"),
                "psr_vs_zero": v.get("psr_vs_zero"),
                "sr0_expected_max": v.get("sr0_expected_max"),
                "performance": v.get("performance", {}),
                "surface": result.surface(),
            },
            verdict=v.get("verdict"),
            family=family,
            repro_spec={
                "kind": "sweep",
                "matrix_object_sha256": matrix_sha,
                "risk_free_rate": result.risk_free_rate,
                "periods_per_year": result.periods_per_year,
            },
            returns_for_deflation=best_col,
            trials_sharpe_std=trials_std,
            risk_free_rate=result.risk_free_rate,
            periods_per_year=result.periods_per_year,
            code_target=backtest_fn,
        )
        record_trials(
            proj,
            run_id=str(rec["run_id"]),
            trials=result.ledger(),
            family=family,
            timestamp=str(rec["timestamp"]),
        )
        result.run_id = str(rec["run_id"])
        result.cumulative_trials = int(rec["cumulative_trials"])
        result.project_trials = int(rec["project_trials"])
    except LedgerError:
        raise
    except OSError as exc:
        raise LedgerError(f"Failed to record sweep in project ledger at {proj.dot_dir}: {exc}") from exc
