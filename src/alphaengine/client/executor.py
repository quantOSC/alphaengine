"""The step executor: runs one operation locally and returns figures.

THE HALF OF THE SPLIT THAT LIVES ON YOUR MACHINE. A workflow server tells this
executor what to do; the executor does it against data that never leaves, and
returns derived figures. Your prices, your returns and your parameter grid stay
where they are.

WHAT THIS DELIBERATELY DOES NOT CONTAIN
    No graph, no router, no notion of what comes next. It executes what it is
    handed and reports. If you find yourself wanting to add "and then usually
    we..." to this file, that belongs to whoever is orchestrating, not here.
    The absence is the design: an executor that knew the sequence would be a
    second, worse copy of the server's job, and the two would drift.

WHAT COMES BACK IS FIGURES, NEVER SERIES
    Every handler returns scalars and small structures. A server that asked for
    a return series would be refused, and none does: the whole arrangement rests
    on the data staying put, so the client enforces its own half rather than
    trusting the other end.

THE WORKSPACE
    Some operations read what an earlier one produced: a deflated Sharpe needs
    the sweep's trial matrix. That intermediate stays HERE, in memory, keyed by
    op name. The server sees the figure, never the matrix.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

import numpy as np

from ..charts import chart
from ..core import (
    breadth_ir,
    compute_var_cvar,
    cost_ladder,
    cpcv_score,
    cs_rank,
    cs_winsorize,
    cs_zscore,
    deflated_sharpe,
    denoise_cov,
    detone_cov,
    dgp_stress,
    drawdown_anatomy,
    ewma_cov,
    fama_macbeth,
    garch_calibrate,
    gbm_calibrate,
    grinold_alpha,
    hrp_weights,
    information_coefficient,
    jump_calibrate,
    min_track_record_length,
    neutralize,
    ou_calibrate,
    ou_simulate,
    overlap_stats,
    pbo_cscv,
    performance_report,
    profile_data,
    quantile_book,
    quantile_returns,
    risk_parity_weights,
    run_backtest,
    score_backtest,
    screen_universe,
    series_values,
    signal_decay,
    signal_icir,
    subperiod_stability,
    technical_features,
    variance_explained,
    vol_target,
)
from ..core.allocate import cov_from_returns
from ..core.covariance import cov_diagnostics, returns_matrix, triangle
from ..core.walkforward import walk_forward
from ..sweep import sweep as run_sweep

__all__ = ["StepExecutor", "UnsupportedOp", "MAX_FIGURE_LIST", "Handler"]

# A figures payload: derived values, never a series. `Workspace` holds the
# intermediates that stay on this machine (the trial matrix, most importantly).
Figures = dict[str, Any]
Workspace = dict[str, Any]
Handler = Callable[[Figures, Workspace], Figures]

# Mirrors the server's guard (`harness/vocabulary.MAX_LIST_LEN`). Enforced on
# our side too, because "the data never leaves" should not depend on the other
# end remembering to check. THE TWO MUST MOVE TOGETHER: a client cap above the
# server's turns every long figure into a rejected step, and one below it
# silently truncates work the server would have accepted.
MAX_FIGURE_LIST = 512

# ── the story figures, bounded by construction ─────────────────────────────
# A run's detail page can only draw what travels, and what travels is a DERIVED
# SUMMARY of the caller's own backtest — never their input series.
#
# RAISED 2026-08-08 with the wire (64 -> 512). At the old bound these curves
# arrived pre-bucketed to 64 points, which is a sketch of a curve rather than
# the curve: a 500-day equity path collapsed to 64 buckets loses every drawdown
# shorter than eight sessions, and the drawdowns are the part a reader is
# looking for. The bucketing still happens HERE, on the machine that holds the
# data, so a longer history is still summarised rather than shipped whole.
CURVE_POINTS = 512  # the best configuration's path, and its drawdown
IC_POINTS = 256  # per-period ICs
COST_POINTS = 64  # cost rungs — still a handful of levels in practice


def _bucketed(
    values: list[float], max_points: int, agg: Callable[[list[float]], float]
) -> list[tuple[int, float]]:
    """Contiguous, near-equal slices by POSITION — profile.py's bucketing.

    Returns (end-of-bucket index, aggregated value) per bucket. Fewer points
    than buckets means one point per bucket: a passthrough, so a short series
    is reported exactly rather than resampled into something it is not.
    """
    n = len(values)
    if n == 0:
        return []
    k = min(max_points, n)
    bounds = [round(i * n / k) for i in range(k + 1)]
    out: list[tuple[int, float]] = []
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        if hi <= lo:
            continue
        out.append((hi - 1, float(agg(values[lo:hi]))))
    return out


def _last(segment: list[float]) -> float:
    return segment[-1]


def _mean(segment: list[float]) -> float:
    return sum(segment) / len(segment)


def _trial_column(result: Any, trial_index: int) -> list[float]:
    """The matrix column for a TRIAL index.

    The matrix holds only the trials that RAN — a failed trial records no
    column — so a trial index has to be mapped through the survivors before it
    can address the matrix. `matrix[:, best.index]` read the wrong
    configuration's returns (or fell off the end) the moment any earlier trial
    failed, which is exactly the run where nobody is double-checking.
    """
    survivors = [t.index for t in result.trials if t.failed is None]
    column: list[float] = result.matrix[:, survivors.index(trial_index)].tolist()
    return column


class UnsupportedOp(LookupError):
    """This build cannot execute that operation.

    Raised rather than silently skipped. A skipped step reports success on work
    that never happened, and the figure it did not produce becomes a gap
    somewhere downstream that nobody can trace back to here.
    """


def _hash(data: Any) -> str:
    try:
        arr = np.asarray(data, dtype=float)
        return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]
    except (TypeError, ValueError):
        blob = json.dumps(data, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def _engine_version() -> str:
    """Which build produced the figures. Stamped so a consumer can reproduce or
    refuse, which is the whole reason the version is a public contract."""
    from .._version import __version__

    return f"alphaengine@{__version__}"


def _schema_version() -> str:
    from ..study import SCHEMA_VERSION

    return str(SCHEMA_VERSION)


def _n_obs(data: Any) -> int:
    # A mapping is a CONTAINER of series, not a series: {"close": [...]},
    # {"returns": [...]}, {symbol: rows}. Its observation count is the longest
    # series it holds. `len(dict)` is its KEY count — which read the demo's
    # 1,200-observation close series as one observation, and the resolve gate
    # then refused an honest series as too short.
    #
    # UNLESS the mapping IS a series: {date: close}, the shape a dated universe
    # carries per symbol. Its values are scalars, so the recursion below scored
    # every dated symbol as zero observations — the same bug, one level down.
    if isinstance(data, dict):
        dated = series_values(data)[0]
        if dated.size:
            return int(dated.size)
        return max((_n_obs(v) for v in data.values()), default=0)
    try:
        return int(np.asarray(data, dtype=float).shape[0])
    except (TypeError, ValueError, IndexError):
        try:
            return len(data)
        except TypeError:
            return 0


def _as_returns(data: Any) -> list[float] | None:
    """Pull a return series out of whatever the caller passed as `data`.

    Accepts a bare sequence of numbers, or a mapping carrying one under
    `returns` / `pnl` — the two spellings a research module actually uses. Any
    other shape returns None, which the caller turns into a message naming what
    was expected. Guessing harder than this would mean picking a column out of a
    frame on the caller's behalf and being wrong silently.
    """
    if data is None:
        return None
    if isinstance(data, dict):
        for key in ("returns", "pnl"):
            inner = data.get(key)
            if inner is not None:
                return _as_returns(inner)
        # A {date: value} mapping is a dated return series, not a container:
        # the shared reader sorts it by key. A universe ({symbol: series})
        # still reads as empty here and stays refused.
        dated = series_values(data)[0]
        if dated.ndim == 1 and dated.size:
            dated_out: list[float] = dated.tolist()
            return dated_out
        return None
    try:
        arr = np.asarray(data, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 1 or arr.size == 0:
        return None
    out: list[float] = arr.tolist()
    return out


def _guard(figures: dict[str, Any]) -> dict[str, Any]:
    """Refuse to send anything series-shaped, whatever it is called."""

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, (list, tuple)):
            if len(node) > MAX_FIGURE_LIST:
                raise ValueError(
                    f"refusing to send {path}: {len(node)} elements is a series, "
                    "not a figure. Your data stays on your machine."
                )
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(figures, "figures")
    return figures


class StepExecutor:
    """Executes ops against local data.

    Args:
        data: whatever your steps operate on. Passed to your backtest function
            untouched and never transmitted.
        backtest_fn: your own backtest, for `compute.sweep`. We orchestrate and
            measure; you simulate.
        handlers: extra or overriding op handlers, as {op: callable(params,
            workspace) -> figures}. This is the extension point for `data.*` and
            `record.*` ops that only your environment can answer.
    """

    def __init__(
        self,
        *,
        data: Any = None,
        backtest_fn: Callable[..., Any] | None = None,
        handlers: dict[str, Handler] | None = None,
    ) -> None:
        self.data = data
        self.backtest_fn = backtest_fn
        self.workspace: dict[str, Any] = {}
        self._handlers: dict[str, Handler] = {
            "data.resolve": self._resolve,
            "data.describe": self._resolve,
            "compute.sweep": self._sweep,
            "compute.deflated_sharpe": self._deflated,
            "compute.pbo_cscv": self._pbo,
            "compute.min_track_record_length": self._mintrl,
            "compute.performance_report": self._performance,
            "compute.compute_var_cvar": self._var,
            "compute.technical_features": self._technical,
            "compute.screen": self._screen,
            # The modeling week beyond the sweep: signal evaluation, data
            # hygiene, stress, and overlap. Same contract as everything above:
            # figures out, series stay here.
            "compute.signal_ic": self._signal_ic,
            "compute.signal_quantiles": self._signal_quantiles,
            "compute.signal_decay": self._signal_decay,
            "data.profile": self._profile,
            "compute.subperiods": self._subperiods,
            "compute.cost_ladder": self._cost_ladder,
            "compute.drawdown_anatomy": self._drawdown_anatomy,
            "compute.overlap": self._overlap,
            "compute.backtest": self._backtest,
            "compute.score_backtest": self._score_backtest,
            "compute.cpcv": self._cpcv,
            "compute.factors": self._factors,
            "compute.pairs": self._pairs,
            "compute.cointegrated_pairs": self._cointegrated_pairs,
            "compute.walk_forward": self._walk_forward,
            "compute.book_overlap": self._book_overlap,
            "compute.panel_transform": self._panel_transform,
            "compute.signal_icir": self._signal_icir,
            "compute.fama_macbeth": self._fama_macbeth,
            "compute.quantile_book": self._quantile_book,
            "compute.ewma_cov": self._ewma_cov,
            "compute.denoise_cov": self._denoise_cov,
            "compute.hrp": self._hrp,
            "compute.risk_parity": self._risk_parity,
            "compute.vol_target": self._vol_target,
            "compute.ou_calibrate": self._ou_calibrate,
            "compute.ou_simulate": self._ou_simulate,
            "compute.gbm_calibrate": self._gbm_calibrate,
            "compute.jump_calibrate": self._jump_calibrate,
            "compute.garch": self._garch,
            "compute.dgp_stress": self._dgp_stress,
            "compute.grinold_alpha": self._grinold_alpha,
            "compute.breadth_ir": self._breadth_ir,
            "compute.detone_cov": self._detone_cov,
            # `emit.*` and `record.*` were vocabulary strings the server could
            # issue and this executor had no handler for, so a run that reached
            # one could never produce the artifact it existed to produce. The
            # consumer was built before the producer.
            "emit.study": self._emit_study,
            "emit.screen": self._emit_echo,
            "emit.monitor": self._emit_echo,
            "emit.sizing_decision": self._emit_echo,
            "emit.signal_report": self._emit_echo,
            "emit.health": self._emit_echo,
            "emit.stress": self._emit_echo,
            "emit.overlap": self._emit_echo,
            "emit.process_report": self._emit_echo,
            "record.note": self._record,
            "record.decision": self._record,
            "record.approval": self._record,
        }
        if handlers:
            self._handlers.update(handlers)

    def supports(self, op: str) -> bool:
        return op in self._handlers

    def execute(self, op: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        handler = self._handlers.get(op)
        if handler is None:
            raise UnsupportedOp(
                f"{op!r} is not executable by this build. Supply a handler for it, or upgrade alphaengine."
            )
        return _guard(handler(dict(params or {}), self.workspace))

    # ── handlers ───────────────────────────────────────────────────────────
    def _resolve(self, params: Figures, ws: Workspace) -> Figures:
        """Identify the data without disclosing it.

        A content hash rather than a name: a label can be changed to escape a
        history and an array cannot, so this is what makes a run reproducible
        without the series ever crossing.

        ── NOTHING IS NOT AN EMPTY DATASET ────────────────────────────────────

        This used to hash `None` and return `{n_obs: 0, hash: "74999fd..."}`, a
        content hash for absent data. That made "you supplied nothing"
        indistinguishable from "you supplied an empty set" everywhere
        downstream — and a gate reading `n_obs: 0` answered with the sentence
        written for a SMALL universe: "that is a handful of names rather than a
        universe", when there had been no universe at all.

        A refusal that names the wrong cause is worse than no refusal. So this
        refuses, and the message is the one the user can act on.
        """
        if self.data is None:
            raise UnsupportedOp(
                "no data was loaded for this run, so there is nothing to identify. "
                "Load some and try again:\n"
                "  --data prices.csv        a local file\n"
                "  --universe <name>        a universe you registered in the portal\n"
                "  --project <module>       a Python module exposing `data`"
            )
        return {
            "ref": params.get("ref") or "local",
            "hash": _hash(self.data),
            "n_obs": _n_obs(self.data),
        }

    def _sweep(self, params: Figures, ws: Workspace) -> Figures:
        if self.backtest_fn is None:
            raise UnsupportedOp(
                "compute.sweep needs your backtest function. Pass backtest_fn to "
                "StepExecutor: we orchestrate and measure, you simulate."
            )
        grid = params.get("grid") or {}
        if not grid:
            # The grid IS the trial count — the denominator every deflated
            # figure divides by — so it must come from the caller's project,
            # never be invented here. Refuse with the fix rather than letting
            # `sweep` raise a bare ValueError up through the run loop.
            raise UnsupportedOp(
                "compute.sweep got an empty grid, and the grid is the trial count. "
                'Declare one in your project module, e.g. GRID = {"fast": [5, 10, 20], '
                '"slow": [50, 100, 200]}. Every combination is one trial.'
            )
        jobs = int(params.get("jobs") or 1)
        result = run_sweep(self.backtest_fn, grid, data=self.data, jobs=jobs)
        ws["sweep"] = result  # the trial matrix stays here

        surface = result.surface()
        figures: Figures = {
            "n_trials": result.n_trials,
            "n_trials_source": "derived_from_grid",
            "data_hash": result.data_hash,
            "shape": surface["shape"],
            "share_within_20pct_of_best": surface["share_within_20pct_of_best"],
            "best_sharpe": surface["best_sharpe"],
            "n_ok": surface["n_ok"],
            "n_failed": surface["n_failed"],
        }

        # The parameter surface, per trial — the artifact a sweep exists to
        # produce, and a derived statistic per configuration, so recording it
        # crosses no data boundary. Sent only when the WHOLE grid fits the
        # wire's cap: a sampled surface would read as the full one, which is
        # exactly the misrepresentation the cap exists to prevent. Failed trials
        # are omitted so a configuration that did not run renders as a hole in
        # the surface, never as a number.
        #
        # BOUND TO THE CAP RATHER THAN REPEATING IT. This was a literal 64 while
        # the cap was 64, so raising one silently left the other — and the
        # symptom would have been a 300-configuration surface still reporting
        # `trials_recorded: false` for no visible reason.
        if result.n_trials <= MAX_FIGURE_LIST:
            figures["trials"] = [
                {**t.params, "sharpe": t.sharpe_annualized} for t in result.trials if t.failed is None
            ]
        else:
            figures["trials_recorded"] = False

        # THE STORY FIGURES. The best configuration's own path, and where it
        # drew down — computed from the column the sweep already holds, bucketed
        # to the wire's bound BEFORE anything travels. A Sharpe is believed with
        # its path; the drawdown is the risk that number hides. The drawdown
        # takes each bucket's WORST reading rather than its endpoint, because
        # sampling across a trough understates the risk in exactly the
        # direction that flatters.
        best = _trial_column(result, result.best.index)
        equity = np.cumprod(1.0 + np.asarray(best, dtype=float))
        cum: list[float] = (equity - 1.0).tolist()
        figures["best_curve"] = [{"i": i, "v": round(v, 6)} for i, v in _bucketed(cum, CURVE_POINTS, _last)]
        peak = np.maximum.accumulate(equity)
        with np.errstate(divide="ignore", invalid="ignore"):
            dd: list[float] = np.where(peak > 0, equity / peak - 1.0, 0.0).tolist()
        figures["drawdown_curve"] = [{"i": i, "v": round(v, 6)} for i, v in _bucketed(dd, CURVE_POINTS, min)]
        return figures

    def _best_column(self, ws: Workspace) -> list[float]:
        result = ws.get("sweep")
        if result is None:
            raise UnsupportedOp("no sweep in this run's workspace; that figure has nothing to read.")
        return _trial_column(result, result.best.index)

    def _returns_column(self, ws: Workspace) -> list[float]:
        """The return series this run is measuring.

        Two sources and the order matters. A sweep's best column when one ran:
        that IS what the run is about, and reading anything else would measure a
        different thing than the one being validated. Otherwise the returns the
        caller supplied, because a workflow that does not sweep — a monitor, a
        sizing decision — is still measuring something real and should not have
        to fabricate a one-cell grid to be allowed to.
        """
        if ws.get("sweep") is not None:
            return self._best_column(ws)

        supplied = _as_returns(self.data)
        if supplied is None:
            raise UnsupportedOp(
                "that figure needs a return series and this run has neither a sweep "
                "nor one it can find. Pass returns as `data` — a sequence of "
                "per-period returns, or a mapping with them under `returns`."
            )
        return supplied

    def _deflated(self, params: Figures, ws: Workspace) -> Figures:
        """Deflate a Sharpe by the number of configurations actually tried.

        THE DENOMINATOR DECIDES WHAT THIS FIGURE MEANS, so where it came from
        travels beside it and is never invented here:

            derived_from_grid   a sweep ran and this is its trial count
            asserted            no sweep; the caller stated a count
            not_recorded        neither, so there is no deflated Sharpe to give

        The third case used to read `int(params.get("n_trials") or 1)`, which is
        the most generous denominator arithmetic can produce and was removed from
        the core function on 2026-08-01 for exactly that reason. It survived here
        because `_best_column` raised first and made it unreachable — and it would
        have gone live the moment a workflow measured a supplied series without
        sweeping, which is now a thing workflows do. A default that is wrong and
        currently unreachable is still wrong.
        """
        result = ws.get("sweep")
        if result is not None:
            n_trials, source = int(result.n_trials), "derived_from_grid"
        elif params.get("n_trials") is not None:
            n_trials, source = int(params["n_trials"]), "asserted"
        else:
            # No count, so no deflation. Reported as absent-with-a-reason rather
            # than as a number, because a DSR standing on an invented denominator
            # looks exactly like one standing on a counted grid.
            figures: Figures = {
                "deflated_sharpe": None,
                "n_trials": None,
                "n_trials_source": "not_recorded",
                "verdict": None,
                "note": (
                    "no trial count was recorded for this run, so the Sharpe cannot "
                    "be deflated. Sweep the grid, or state the count you searched."
                ),
            }
            ws["verdict"] = figures
            return figures

        out = deflated_sharpe(self._returns_column(ws), n_trials=n_trials)
        figures = {
            "deflated_sharpe": out.get("deflated_sharpe"),
            "psr_vs_zero": out.get("psr_vs_zero"),
            "sr0_expected_max": out.get("sr0_expected_max"),
            "verdict": out.get("verdict"),
            "n_trials": n_trials,
            "n_trials_source": source,
        }
        # Kept so `emit.study` can carry the verdict this run actually produced
        # rather than deriving it a second time. Two derivations of one figure
        # can disagree, and the second is the one nobody looks at.
        ws["verdict"] = figures
        return figures

    def _pbo(self, params: Figures, ws: Workspace) -> Figures:
        result = ws.get("sweep")
        if result is None or result.matrix.shape[1] < 2:
            # Honest rather than a number that looks like an answer: one
            # configuration means the choice among configurations was not a
            # choice.
            return {"pbo": None, "note": "needs at least two configurations"}
        out = pbo_cscv(result.matrix.tolist())
        return {"pbo": out.get("pbo"), "verdict": out.get("verdict"), "n_configs": out.get("n_configs")}

    def _mintrl(self, params: Figures, ws: Workspace) -> Figures:
        return dict(min_track_record_length(self._returns_column(ws), **params))

    def _performance(self, params: Figures, ws: Workspace) -> Figures:
        return dict(performance_report(self._returns_column(ws), **params))

    def _var(self, params: Figures, ws: Workspace) -> Figures:
        out = compute_var_cvar(self._returns_column(ws), **params)
        return {
            "n_obs": out.get("n_obs"),
            "confidence": out.get("confidence"),
            "horizon_days": out.get("horizon_days"),
            # A sample too small for the tail to mean anything. Travels with the
            # number rather than being left for the reader to work out.
            "low_sample": out.get("low_sample"),
            "parametric": out.get("parametric"),
            "cornish_fisher": out.get("cornish_fisher"),
            "historical": out.get("historical"),
            # CVaR is the figure a sizing gate should read: it is the average
            # loss BEYOND the VaR point, so it says what the bad day costs rather
            # than only how often it arrives.
            "cvar": out.get("cvar"),
        }

    def _universe(self) -> dict[str, Any]:
        """The caller's universe, or a message naming the shape that was wanted.

        `{symbol: series}` is what both screening ops read. A caller who passed a
        bare return series gets told so, instead of an AttributeError from three
        frames down that names `.items` and explains nothing.
        """
        if not isinstance(self.data, dict) or not self.data:
            raise UnsupportedOp(
                "that operation reads a universe and this run's data is not one. "
                "Pass `data` as {symbol: prices} — prices being closes, "
                "{date: close}, or OHLC rows."
            )
        return self.data

    def _technical(self, params: Figures, ws: Workspace) -> Figures:
        out = technical_features(self._universe(), **params)
        # Only the scalar readings travel; any embedded series is dropped rather
        # than transmitted.
        return {k: v for k, v in out.items() if not isinstance(v, (list, tuple))}

    def _screen(self, params: Figures, ws: Workspace) -> Figures:
        """Rank the universe locally and send back the shortlist.

        The universe stays here and a bounded number of rows leave, which is the
        same split as everywhere else in this file: we orchestrate and measure,
        your data does not move. Note this returns LESS than
        `compute.technical_features` does on the same input, and that is the
        point — a feature table grows with the universe, a shortlist does not.
        """
        return dict(screen_universe(self._universe(), **params))

    # ── the modeling week beyond the sweep ─────────────────────────────────

    def _signal_panels(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """The `{signal, prices}` pair a signal evaluation runs on.

        A signal is scored AGAINST something, so the data for these ops is a
        mapping carrying both panels. Anything else is refused with the shape
        named — a panel guessed out of position ranks the wrong thing silently.
        """
        data = self.data
        if (
            isinstance(data, dict)
            and isinstance(data.get("signal"), dict)
            and isinstance(data.get("prices"), dict)
        ):
            return data["signal"], data["prices"]
        raise UnsupportedOp(
            "signal evaluation needs `data` shaped {'signal': {symbol: values}, "
            "'prices': {symbol: closes}} — the signal panel and what it is "
            "scored against. Expose both from a --project module."
        )

    def _signal_ic(self, params: Figures, ws: Workspace) -> Figures:
        signal, prices = self._signal_panels()
        out = dict(information_coefficient(signal, prices, horizon=int(params.get("horizon") or 21)))
        # The story figure: the per-period sequence, bucketed. `p` is the
        # end-of-bucket period ordinal (1-based, oldest first) so the axis
        # reads as time; the value is the bucket's MEAN, because a stretch of
        # periods is a claim about its average sign, not its endpoint.
        ics = out.get("ic_by_period")
        if isinstance(ics, list):
            out["ic_by_period"] = [
                {"p": p + 1, "ic": round(v, 6)}
                for p, v in _bucketed([float(x) for x in ics], IC_POINTS, _mean)
            ]
        return out

    def _signal_quantiles(self, params: Figures, ws: Workspace) -> Figures:
        signal, prices = self._signal_panels()
        return dict(
            quantile_returns(
                signal,
                prices,
                horizon=int(params.get("horizon") or 21),
                quantiles=int(params.get("quantiles") or 5),
            )
        )

    def _signal_decay(self, params: Figures, ws: Workspace) -> Figures:
        signal, prices = self._signal_panels()
        horizons = params.get("horizons") or (1, 5, 21, 63)
        return dict(signal_decay(signal, prices, horizons=tuple(int(h) for h in horizons)))

    def _profile(self, params: Figures, ws: Workspace) -> Figures:
        """Health-check the loaded universe. Reports; repairs nothing."""
        data = self.data
        if isinstance(data, dict) and isinstance(data.get("prices"), dict):
            data = data["prices"]
        if not isinstance(data, dict):
            raise UnsupportedOp(
                "data.profile inspects a universe: {symbol: closes}. Load one with --universe or --data."
            )
        return dict(profile_data(data))

    def _subperiods(self, params: Figures, ws: Workspace) -> Figures:
        return dict(subperiod_stability(self._returns_column(ws), segments=int(params.get("segments") or 4)))

    def _cost_ladder(self, params: Figures, ws: Workspace) -> Figures:
        # Turnover is the CALLER'S number; absent, every rung is None. A
        # guessed turnover would produce a confident curve about a strategy
        # nobody runs.
        turnover = params.get("turnover")
        out = dict(
            cost_ladder(
                self._returns_column(ws),
                turnover=None if turnover is None else float(turnover),
            )
        )
        # The story figure: the descent itself, level by level. Only the rungs
        # that were MEASURED — with no turnover every rung is None, and the
        # honest record of that is the key's absence, never a flat line.
        curve = [
            {"bps": float(b), "sharpe": round(float(s), 6)}
            for b, s in zip(out.get("bps_levels") or [], out.get("sharpe_at_bps") or [], strict=True)
            if s is not None
        ][:COST_POINTS]
        if curve:
            out["cost_curve"] = curve
        return out

    def _drawdown_anatomy(self, params: Figures, ws: Workspace) -> Figures:
        return dict(drawdown_anatomy(self._returns_column(ws)))

    def _overlap(self, params: Figures, ws: Workspace) -> Figures:
        """The candidate against the book. Both series are the caller's own."""
        book = self.data.get("book_returns") if isinstance(self.data, dict) else None
        if book is None:
            raise UnsupportedOp(
                "compute.overlap needs `data` carrying both series: {'returns': "
                "[...], 'book_returns': [...]} — the candidate and what the book "
                "already holds."
            )
        candidate = _as_returns(self.data)
        if candidate is None:
            raise UnsupportedOp(
                "compute.overlap found `book_returns` but no candidate. Put the "
                "idea's own series under `returns`."
            )
        out = dict(overlap_stats(candidate, book))

        # THE STORY FIGURES. A correlation and a beta are two numbers standing
        # in for a relationship, and a desk cannot act on the summary alone:
        #
        #   the SCATTER is the joint distribution the correlation summarises,
        #   and it is where you see that a 0.2 reading is really a cloud plus
        #   four shared crashes;
        #
        #   ROLLING CORRELATION is whether the reading is stable. An idea that
        #   averages 0.2 to the book but runs at 0.9 in every drawdown is the
        #   book again exactly when it matters, and the average says the
        #   opposite. This is the figure that changes a sizing decision.
        #
        # Both are derived from the caller's own two series, on this machine,
        # and bounded before anything travels.
        if out.get("correlation") is not None:
            c = np.asarray(candidate, dtype=float)
            b = np.asarray(book, dtype=float)
            depth = int(out.get("n_obs") or min(c.size, b.size))
            c, b = c[-depth:], b[-depth:]

            stride = max(1, depth // CURVE_POINTS)
            out["overlap_scatter"] = [
                {"x": round(float(b[i]), 6), "y": round(float(c[i]), 6)} for i in range(0, depth, stride)
            ]

            # A quarter of the history, floored so the window is a measurement
            # rather than a coincidence, and skipped entirely when the series
            # is too short to roll — a rolling reading over 12 points would be
            # noise drawn as a trend.
            window = max(20, depth // 4)
            if depth >= window * 2:
                rolling: list[float] = []
                for end in range(window, depth + 1):
                    cw, bw = c[end - window : end], b[end - window : end]
                    sc, sb = float(cw.std(ddof=1)), float(bw.std(ddof=1))
                    if sc == 0 or sb == 0:
                        rolling.append(float("nan"))
                        continue
                    cov = float(np.mean((cw - cw.mean()) * (bw - bw.mean())))
                    rolling.append(cov / (sc * sb) * window / (window - 1))
                clean = [v for v in rolling if v == v]
                if clean:
                    out["rolling_correlation"] = [
                        {"i": i + window, "v": round(v, 6)}
                        for i, v in _bucketed(rolling, CURVE_POINTS, _last)
                        if v == v
                    ]
                    out["rolling_window"] = window
                    out["max_rolling_correlation"] = round(max(clean), 6)
                    out["min_rolling_correlation"] = round(min(clean), 6)
        return out

    # ── the rest of the modeling week, previously library-only ─────────────

    def _backtest(self, params: Figures, ws: Workspace) -> Figures:
        """Run the built-in simulator. Series stay in the workspace."""
        data = self.data
        if not (isinstance(data, dict) and "signals" in data and "prices" in data):
            raise UnsupportedOp(
                "compute.backtest needs `data` shaped {'signals': {ticker: [...]}, "
                "'prices': {ticker: [...]} }."
            )
        cfg = {
            k: params[k]
            for k in (
                "slippage_bps",
                "commission_bps",
                "fill_timing",
                "initial_capital",
                "max_position_pct",
                "adv",
                "impact_coef",
                "max_participation",
            )
            if k in params
        }
        bt = run_backtest(data["signals"], data["prices"], **cfg)
        ws["backtest"] = bt
        if isinstance(bt.get("returns"), list):
            ws.setdefault("returns", bt["returns"])
        return {
            "n_bars": bt.get("n_bars"),
            "n_trades": bt.get("n_trades"),
            "final_equity": bt.get("final_equity"),
            "total_return_pct": bt.get("total_return_pct"),
            "cost_breakdown": bt.get("cost_breakdown"),
            "capacity": bt.get("capacity"),
            "config": bt.get("config"),
        }

    def _score_backtest(self, params: Figures, ws: Workspace) -> Figures:
        bt = ws.get("backtest")
        if bt is None:
            raise UnsupportedOp("compute.score_backtest needs compute.backtest first in this run.")
        n_trials = params.get("n_trials")
        source = params.get("n_trials_source")
        sweep = ws.get("sweep")
        if sweep is not None and n_trials is None:
            n_trials, source = int(sweep.n_trials), "derived_from_grid"
        scored = score_backtest(
            bt,
            n_trials=None if n_trials is None else int(n_trials),
            n_trials_source=source,
            cpcv=bool(params.get("cpcv")),
        )
        ws["score"] = scored
        return {
            "performance": scored.get("performance"),
            "validation": scored.get("validation"),
            "trade_summary": scored.get("trade_summary"),
            "costs": scored.get("costs"),
            "not_computed": scored.get("not_computed"),
        }

    def _cpcv(self, params: Figures, ws: Workspace) -> Figures:
        returns = self._returns_column(ws)
        n_trials = params.get("n_trials")
        sweep = ws.get("sweep")
        if sweep is not None and n_trials is None:
            n_trials = int(sweep.n_trials)
        return dict(
            cpcv_score(
                returns,
                n_groups=int(params.get("n_groups") or 8),
                n_test_groups=int(params.get("n_test_groups") or 2),
                purge=int(params.get("purge") or 1),
                embargo=int(params.get("embargo") or 1),
                n_trials=int(n_trials) if n_trials is not None else 1,
            )
        )

    def _factors(self, params: Figures, ws: Workspace) -> Figures:
        try:
            from ..core.factors import decompose_factors
        except ModuleNotFoundError as exc:
            raise UnsupportedOp(str(exc)) from exc
        data = self.data
        if not (isinstance(data, dict) and "factor_returns" in data):
            raise UnsupportedOp(
                "compute.factors needs `data` shaped {'returns': [...], 'factor_returns': {name: [...]}}."
            )
        portfolio = _as_returns(data) or data.get("returns")
        if portfolio is None:
            raise UnsupportedOp("compute.factors found factor_returns but no portfolio returns.")
        kwargs = {}
        if params.get("risk_free_rate") is not None:
            kwargs["risk_free_rate"] = float(params["risk_free_rate"])
        return dict(decompose_factors(list(portfolio), data["factor_returns"], **kwargs))

    def _pairs(self, params: Figures, ws: Workspace) -> Figures:
        try:
            from ..core.pairs import compute_spread_signal
        except ModuleNotFoundError as exc:
            raise UnsupportedOp(str(exc)) from exc
        data = self.data
        if not isinstance(data, dict) or len(data) < 2:
            raise UnsupportedOp(
                "compute.pairs needs two close series. Pass `data` as {A: closes, B: closes} "
                "or {'a': ..., 'b': ...}."
            )
        if "a" in data and "b" in data:
            a, b = data["a"], data["b"]
            sa = str(params.get("symbol_a") or "A")
            sb = str(params.get("symbol_b") or "B")
        else:
            skip = ("signal", "prices", "returns", "book_returns", "factor_returns")
            names = [k for k in data if k not in skip]
            if len(names) < 2:
                raise UnsupportedOp("compute.pairs needs two named close series.")
            sa, sb = names[0], names[1]
            a, b = data[sa], data[sb]
        out = dict(compute_spread_signal(a, b, symbol_a=sa, symbol_b=sb))
        # A z-score path is a series. Keep the last reading only.
        if isinstance(out.get("zscore_series"), list):
            out["zscore_last"] = out["zscore_series"][-1] if out["zscore_series"] else None
            del out["zscore_series"]
        return out

    def _cointegrated_pairs(self, params: Figures, ws: Workspace) -> Figures:
        try:
            from ..core.pairs import find_cointegrated_pairs
        except ModuleNotFoundError as exc:
            raise UnsupportedOp(str(exc)) from exc
        universe = self._universe()
        # Cap the all-pairs search so a 500-name universe does not become an
        # accidental O(n²) overnight job. Caller can pass an explicit list.
        symbols = list(universe)[: int(params.get("max_symbols") or 40)]
        prices = {k: universe[k] for k in symbols}
        only = bool(params.get("cointegrated_only", True))
        out = dict(find_cointegrated_pairs(prices, cointegrated_only=only))
        pairs = out.get("pairs") or []
        if len(pairs) > MAX_FIGURE_LIST:
            out["pairs"] = pairs[:MAX_FIGURE_LIST]
            out["truncated"] = True
        return out

    def _walk_forward(self, params: Figures, ws: Workspace) -> Figures:
        if self.backtest_fn is None:
            raise UnsupportedOp(
                "compute.walk_forward needs your backtest function, the same as compute.sweep."
            )
        grid = params.get("grid") or {}
        if not grid:
            raise UnsupportedOp("compute.walk_forward got an empty grid, and the grid is the trial count.")
        out = walk_forward(
            self.backtest_fn,
            grid,
            data=self.data,
            n_windows=int(params.get("n_windows") or 4),
            oos_fraction=float(params.get("oos_fraction") or 0.25),
        )
        ws["walk_forward"] = out
        return out

    def _book_overlap(self, params: Figures, ws: Workspace) -> Figures:
        from ..book import Book

        data = self.data
        book = ws.get("book")
        if book is None:
            book = Book()
            sleeves = data.get("sleeves") if isinstance(data, dict) else None
            if isinstance(sleeves, dict):
                for name, series in list(sleeves.items())[:16]:
                    book.add(str(name), series)
            elif isinstance(data, dict) and "book_returns" in data:
                book.add("book", data["book_returns"])
                candidate = _as_returns(data)
                if candidate:
                    book.add("candidate", candidate)
            else:
                raise UnsupportedOp(
                    "compute.book_overlap needs `data` as {'sleeves': {name: returns, ...}} "
                    "or the overlap shape {'returns': [...], 'book_returns': [...]}."
                )
            ws["book"] = book
        candidate = params.get("candidate")
        return book.overlap_matrix(candidate)

    # ── daily CS + book construction ───────────────────────────────────────

    def _panel_and_controls(self) -> tuple[dict[str, Any], Any]:
        data = self.data
        if isinstance(data, dict) and isinstance(data.get("panel"), dict):
            return data["panel"], data.get("controls")
        if isinstance(data, dict) and data:
            return data, None
        raise UnsupportedOp(
            "compute.panel_transform needs a {symbol: series} panel, or "
            "{'panel': {...}, 'controls': {...}} for neutralize."
        )

    def _panel_transform(self, params: Figures, ws: Workspace) -> Figures:
        panel, controls = self._panel_and_controls()
        method = str(params.get("method") or "zscore").lower()
        if method == "rank":
            out = cs_rank(panel)
        elif method == "winsorize":
            out = cs_winsorize(
                panel,
                lower=float(params.get("lower") or 0.01),
                upper=float(params.get("upper") or 0.99),
            )
        elif method == "neutralize":
            out = neutralize(panel, controls if controls is not None else params.get("controls"))
        elif method == "zscore":
            out = cs_zscore(panel)
        else:
            raise UnsupportedOp("compute.panel_transform method is rank, zscore, winsorize, or neutralize.")
        ws["panel"] = out.get("panel")
        figures = {k: v for k, v in out.items() if k != "panel"}
        return figures

    def _signal_icir(self, params: Figures, ws: Workspace) -> Figures:
        signal, prices = self._signal_panels()
        out = dict(
            signal_icir(
                signal,
                prices,
                horizon=int(params.get("horizon") or 21),
                method=str(params.get("method") or "spearman"),
            )
        )
        ics = out.get("ic_by_period")
        if isinstance(ics, list):
            out["ic_by_period"] = [
                {"p": p + 1, "ic": round(v, 6)}
                for p, v in _bucketed([float(x) for x in ics], IC_POINTS, _mean)
            ]
        return out

    def _fama_macbeth(self, params: Figures, ws: Workspace) -> Figures:
        signal, prices = self._signal_panels()
        out = dict(fama_macbeth(signal, prices, horizon=int(params.get("horizon") or 21)))
        series = out.get("lambda_by_date")
        if isinstance(series, list):
            out["lambda_by_date"] = [
                {"p": p + 1, "lambda": round(v, 6)}
                for p, v in _bucketed([float(x) for x in series], IC_POINTS, _mean)
            ]
        return out

    def _quantile_book(self, params: Figures, ws: Workspace) -> Figures:
        signal, prices = self._signal_panels()
        return dict(
            quantile_book(
                signal,
                prices,
                horizon=int(params.get("horizon") or 21),
                quantiles=int(params.get("quantiles") or 5),
            )
        )

    def _returns_panel(self) -> tuple[Any, list[str], list[str]]:
        data = self.data
        if isinstance(data, dict) and isinstance(data.get("returns"), dict):
            data = data["returns"]
        if not isinstance(data, dict) or not data:
            raise UnsupportedOp(
                "covariance and allocation need `data` as {symbol: returns}, "
                "or {'returns': {symbol: returns}}."
            )
        R, names, skipped = returns_matrix(data)
        if R.shape[0] < 2 or R.shape[1] < 2:
            raise UnsupportedOp("need at least two names and two observations to build a covariance.")
        return R, names, skipped

    def _ewma_cov(self, params: Figures, ws: Workspace) -> Figures:
        R, names, skipped = self._returns_panel()
        cov = ewma_cov(R, lam=float(params.get("lam") or 0.94))
        ws["cov"] = cov
        ws["cov_names"] = names
        out = cov_diagnostics(cov)
        out["method"] = "ewma"
        out["lam"] = float(params.get("lam") or 0.94)
        out["n_obs"] = int(R.shape[0])
        out["n_skipped"] = len(skipped)
        tri = triangle(cov, names)
        if tri is not None:
            out["triangle"] = tri
        return out

    def _denoise_cov(self, params: Figures, ws: Workspace) -> Figures:
        R, names, skipped = self._returns_panel()
        sample = cov_from_returns(R, method=str(params.get("estimator") or "sample"))
        out = denoise_cov(sample, n_obs=int(R.shape[0]))
        cov = out.pop("cov")
        ws["cov"] = cov
        ws["cov_names"] = names
        out.update(cov_diagnostics(cov))
        out["n_skipped"] = len(skipped)
        tri = triangle(cov, names)
        if tri is not None:
            out["triangle"] = tri
        explained = variance_explained(cov)
        if explained:
            out["variance_explained"] = explained
        hints = []
        if tri is not None:
            hints.append(chart("triangle", "triangle", "denoised covariance"))
        if explained:
            hints.append(chart("rows", "variance_explained", "eigenvalue share"))
        if hints:
            out["charts"] = hints
        return out

    def _hrp(self, params: Figures, ws: Workspace) -> Figures:
        R, names, skipped = self._returns_panel()
        method = str(params.get("cov") or "sample")
        if method in ("denoise", "mp"):
            raw = cov_from_returns(R, method="sample")
            cov = denoise_cov(raw, n_obs=int(R.shape[0]))["cov"]
        else:
            cov = cov_from_returns(R, method=method, lam=float(params.get("lam") or 0.94))
        ws["cov"] = cov
        out = hrp_weights(cov, names=names)
        out["n_skipped"] = len(skipped)
        out["n_obs"] = int(R.shape[0])
        out["n_trials"] = len(names)
        out["n_trials_source"] = "derived_from_names"
        return out

    def _risk_parity(self, params: Figures, ws: Workspace) -> Figures:
        R, names, skipped = self._returns_panel()
        cov = cov_from_returns(R, method=str(params.get("cov") or "lw"))
        ws["cov"] = cov
        out = risk_parity_weights(cov, names=names)
        out["n_skipped"] = len(skipped)
        out["n_obs"] = int(R.shape[0])
        return out

    def _vol_target(self, params: Figures, ws: Workspace) -> Figures:
        returns = _as_returns(self.data)
        if returns is None:
            raise UnsupportedOp("compute.vol_target needs a return series.")
        how = str(params.get("method") or "ewma").lower()
        gvar = ws.get("garch_var")
        return dict(
            vol_target(
                returns,
                target=float(params.get("target") or 0.10),
                ewma_lambda=float(params.get("ewma_lambda") or 0.94),
                periods_per_year=int(params.get("periods_per_year") or 252),
                method=how,
                garch_var=gvar,
            )
        )

    def _series_or_refuse(self, why: str) -> list[float]:
        returns = _as_returns(self.data)
        if returns is not None:
            return returns
        if isinstance(self.data, dict):
            for key in ("close", "spread", "series"):
                inner = self.data.get(key)
                if inner is not None:
                    got = _as_returns(inner) if not isinstance(inner, dict) else None
                    if got:
                        return got
                    from ..core.process import as_series

                    arr = as_series(inner)
                    if arr.size:
                        return [float(v) for v in arr.tolist()]
        from ..core.process import as_series

        arr = as_series(self.data)
        if arr.size:
            return [float(v) for v in arr.tolist()]
        raise UnsupportedOp(why)

    def _ou_calibrate(self, params: Figures, ws: Workspace) -> Figures:
        series = self._series_or_refuse("compute.ou_calibrate needs a spread or return series.")
        out = dict(ou_calibrate(series, dt=float(params.get("dt") or 1.0)))
        ws["ou"] = out
        figures = {k: v for k, v in out.items() if k != "paths"}
        figures["charts"] = [chart("curve", "path_sketch", "calibrated spread")]
        return figures

    def _ou_simulate(self, params: Figures, ws: Workspace) -> Figures:
        cal = ws.get("ou") if isinstance(ws.get("ou"), dict) else None
        series = None
        try:
            series = self._series_or_refuse("compute.ou_simulate needs a series when no OU is calibrated.")
        except UnsupportedOp:
            series = None
        if cal is None or cal.get("kappa") is None:
            if series is None:
                raise UnsupportedOp("compute.ou_simulate needs a calibrated OU or a series to fit.")
            cal = ou_calibrate(series)
            ws["ou"] = cal
        if cal.get("kappa") is None:
            raise UnsupportedOp("the series is not mean-reverting; OU simulate refused.")
        n = int(params.get("n_obs") or cal.get("n_obs") or 120)
        sim = ou_simulate(
            kappa=float(cal["kappa"]),
            theta=float(cal.get("theta") or 0.0),
            sigma=float(cal.get("sigma") or 0.01),
            n_obs=n,
            n_paths=int(params.get("n_paths") or 200),
            x0=float(series[-1]) if series else None,
            seed=int(params.get("seed") or 7),
        )
        ws["ou_paths"] = sim.pop("paths")
        sim["charts"] = [
            chart("curve", "mean_path", "mean OU path"),
            chart("band", "band", "OU quantile band"),
        ]
        return sim

    def _gbm_calibrate(self, _params: Figures, ws: Workspace) -> Figures:
        series = self._series_or_refuse("compute.gbm_calibrate needs a price or return series.")
        out = dict(gbm_calibrate(series))
        ws["gbm"] = out
        return out

    def _jump_calibrate(self, params: Figures, ws: Workspace) -> Figures:
        series = self._series_or_refuse("compute.jump_calibrate needs a price or return series.")
        out = dict(jump_calibrate(series, z=float(params.get("z") or 3.0)))
        ws["jump"] = out
        return out

    def _garch(self, _params: Figures, ws: Workspace) -> Figures:
        series = self._series_or_refuse("compute.garch needs a return series.")
        out = dict(garch_calibrate(series))
        ws["garch_var"] = out.pop("var_path")
        out["charts"] = [chart("curve", "vol_path", "GARCH vol")]
        return out

    def _dgp_stress(self, params: Figures, ws: Workspace) -> Figures:
        series = self._series_or_refuse("compute.dgp_stress needs a return or price series.")
        dgp = str(params.get("dgp") or "gbm").lower()
        out = dict(
            dgp_stress(
                series,
                dgp=dgp,
                n_paths=int(params.get("n_paths") or 200),
                seed=int(params.get("seed") or 7),
                backtest_fn=self.backtest_fn,
            )
        )
        out["charts"] = [chart("hist", "sharpe_hist", "Sharpe under the DGP")]
        return out

    def _grinold_alpha(self, params: Figures, ws: Workspace) -> Figures:
        panel, _ = self._panel_and_controls()
        ic = params.get("ic")
        if ic is None:
            raise UnsupportedOp("compute.grinold_alpha needs params.ic, the information coefficient.")
        vols = params.get("vols") if isinstance(params.get("vols"), dict) else None
        out = dict(grinold_alpha(panel, ic=float(ic), vols=vols))
        ws["alpha"] = out.pop("alpha")
        return out

    def _breadth_ir(self, params: Figures, ws: Workspace) -> Figures:
        ic = params.get("ic")
        n_names = params.get("n_names")
        if ic is None:
            raise UnsupportedOp("compute.breadth_ir needs params.ic.")
        if n_names is None:
            try:
                panel, _ = self._panel_and_controls()
                n_names = len(panel)
            except UnsupportedOp:
                n_names = 0
        holdings = params.get("holdings") if isinstance(params.get("holdings"), dict) else None
        ideal = params.get("ideal") if isinstance(params.get("ideal"), dict) else ws.get("alpha")
        if isinstance(ideal, dict) and not holdings:
            ideal = None
        return dict(
            breadth_ir(
                ic=float(ic),
                n_names=int(n_names or 0),
                holdings=holdings,
                ideal=ideal if isinstance(ideal, dict) else None,
            )
        )

    def _detone_cov(self, params: Figures, ws: Workspace) -> Figures:
        R, names, skipped = self._returns_panel()
        sample = cov_from_returns(R, method=str(params.get("estimator") or "sample"))
        cov = detone_cov(sample)
        ws["cov"] = cov
        ws["cov_names"] = names
        out = cov_diagnostics(cov)
        out["method"] = "detone"
        out["n_skipped"] = len(skipped)
        out["n_obs"] = int(R.shape[0])
        explained = variance_explained(cov)
        if explained:
            out["variance_explained"] = explained
        tri = triangle(cov, names)
        if tri is not None:
            out["triangle"] = tri
        hints = []
        if tri is not None:
            hints.append(chart("triangle", "triangle", "detoned covariance"))
        if explained:
            hints.append(chart("rows", "variance_explained", "eigenvalue share"))
        if hints:
            out["charts"] = hints
        return out

    # ── emit / record ──────────────────────────────────────────────────────
    #
    # THESE RETURN FIGURES; THEY DO NOT POST ANYWHERE. The server already
    # receives every step result and is the only party that can write to your
    # workspace, so having the client open a second connection with a second
    # credential to deliver something the server is about to be handed anyway
    # would be two ways to do one thing — and the two would drift.
    #
    # `report()` remains the OFFLINE path: a notebook with no run, no server and
    # no workflow. Different caller, different problem, deliberately separate.

    def _emit_study(self, params: Figures, ws: Workspace) -> Figures:
        """Assemble the study this run produced, from what actually ran.

        THE TRIAL COUNT IS TAKEN FROM THE SWEEP, NEVER FROM `params`. A count
        arriving in a directive is a count somebody upstream could have chosen;
        a count read off the grid that executed is one that was counted. Since
        alphaengine 0.2.0 an unrecorded denominator cannot reach an `edge`
        verdict at all, which is precisely why the distinction has to be made
        HERE, at the only point in the system that knows the truth.
        """
        result = ws.get("sweep")
        if result is None:
            raise UnsupportedOp(
                "emit.study has no sweep to describe. Run compute.sweep first, or "
                "this run has nothing to emit."
            )

        study: Figures = {
            "label": params.get("label") or "Study",
            "engine_version": _engine_version(),
            "schema_version": _schema_version(),
            "n_trials": result.n_trials,
            "n_trials_source": "derived_from_grid",
            "data_hash": result.data_hash,
        }
        if params.get("data_description"):
            study["data_description"] = params["data_description"]
        if params.get("notes"):
            study["notes"] = params["notes"]

        # Whatever the run already computed rides along rather than being
        # recomputed: a study whose verdict was derived twice can disagree with
        # itself, and the second derivation is the one nobody looks at.
        surface = result.surface()
        study["surface"] = {
            k: surface[k]
            for k in ("shape", "share_within_20pct_of_best", "best_sharpe", "n_ok", "n_failed")
            if k in surface
        }
        verdict = ws.get("verdict") or {}
        if verdict:
            study["verdict"] = verdict.get("verdict")
            study["deflated_sharpe"] = verdict.get("deflated_sharpe")
        return study

    def _emit_echo(self, params: Figures, ws: Workspace) -> Figures:
        """Seal an artifact the SERVER assembled, and stamp which build sealed it.

        THE ASYMMETRY WITH `emit.study` IS DELIBERATE. A study's trial count has
        to be read off the sweep that ran, because a count is the one figure this
        machine knows and the server can only be told. A shortlist, a sizing
        decision and a monitor reading are the opposite: they are the workflow's
        own conclusion drawn from figures it already holds, so re-deriving them
        here would be a second opinion nobody asked for and the two could
        disagree.

        So this echoes the values and adds only what the server cannot know:
        which build produced the figures underneath, which is what a consumer
        needs to reproduce or refuse.
        """
        out: Figures = dict(params)
        out["engine_version"] = _engine_version()
        out["schema_version"] = _schema_version()
        return out

    def _record(self, params: Figures, ws: Workspace) -> Figures:
        """A note, a decision, an approval: values only, echoed for the ledger.

        VALUES, NOT FREE TEXT PARSED FOR MEANING. The one time this codebase
        parsed prose to decide something it inverted an entire slate to short.
        Whatever the server wants recorded it names as a parameter, and this
        hands it back so the durable step record carries it.
        """
        return dict(params)
