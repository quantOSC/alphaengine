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

from ..core import (
    compute_var_cvar,
    deflated_sharpe,
    min_track_record_length,
    pbo_cscv,
    performance_report,
    screen_universe,
    technical_features,
)
from ..sweep import sweep as run_sweep

__all__ = ["StepExecutor", "UnsupportedOp", "MAX_FIGURE_LIST", "Handler"]

# A figures payload: derived values, never a series. `Workspace` holds the
# intermediates that stay on this machine (the trial matrix, most importantly).
Figures = dict[str, Any]
Workspace = dict[str, Any]
Handler = Callable[[Figures, Workspace], Figures]

# Mirrors the server's guard. Enforced on our side too, because "the data never
# leaves" should not depend on the other end remembering to check.
MAX_FIGURE_LIST = 64


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
            # `emit.*` and `record.*` were vocabulary strings the server could
            # issue and this executor had no handler for, so a run that reached
            # one could never produce the artifact it existed to produce. The
            # consumer was built before the producer.
            "emit.study": self._emit_study,
            "emit.screen": self._emit_echo,
            "emit.monitor": self._emit_echo,
            "emit.sizing_decision": self._emit_echo,
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
        """
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
        result = run_sweep(self.backtest_fn, params.get("grid") or {}, data=self.data)
        ws["sweep"] = result  # the trial matrix stays here

        surface = result.surface()
        return {
            "n_trials": result.n_trials,
            "n_trials_source": "derived_from_grid",
            "data_hash": result.data_hash,
            "shape": surface["shape"],
            "share_within_20pct_of_best": surface["share_within_20pct_of_best"],
            "best_sharpe": surface["best_sharpe"],
            "n_ok": surface["n_ok"],
            "n_failed": surface["n_failed"],
        }

    def _best_column(self, ws: Workspace) -> list[float]:
        result = ws.get("sweep")
        if result is None:
            raise UnsupportedOp("no sweep in this run's workspace; that figure has nothing to read.")
        column: list[float] = result.matrix[:, result.best.index].tolist()
        return column

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
