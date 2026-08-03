"""Connect to a workflow server and drive a run.

    from alphaengine import sweep                 # works offline, always
    from alphaengine.client import connect        # needs a server

    session = connect("https://...", api_key="...")
    run = session.open("validate_study", data=prices, backtest_fn=my_backtest,
                       grid={"fast": [5, 10], "slow": [50, 100]})
    run.drive()
    run.artifact

OFFLINE IS A FIRST-CLASS STATE, NOT AN ERROR PATH
    Every primitive in this package runs with the network off. Workflows do not,
    because the sequencing lives on the server. When there is no connection the
    honest answer is "workflows are unavailable, the maths still works", said
    plainly and once. What must never happen is a silent downgrade to some
    lesser path the caller did not choose: a quietly degraded run that still
    produces an artifact is worse than no run, because the artifact looks the
    same as a real one.

A STOP IS A RESULT
    When a gate refuses, `drive()` returns normally with `run.stopped` set. It
    does not raise. "This did not clear the bar" is the system working, and
    routing it through the exception channel would put it alongside connection
    failures, which need the opposite handling.

NO GRAPH LIVES HERE
    This module holds a loop: ask what is permitted, execute it, report, repeat.
    It has no opinion about order, because it is never told one.
"""

from __future__ import annotations

import contextlib
import json
import logging
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any

from .executor import Figures, Handler, StepExecutor

logger = logging.getLogger(__name__)

__all__ = ["connect", "Session", "Run", "Offline", "ServerError", "connect_or_offline"]

_TIMEOUT = 120


class Offline(RuntimeError):
    """No workflow server is reachable.

    Deliberately its own type. Callers routinely want to catch exactly this and
    fall back to running primitives by hand, which is a supported way to work
    rather than a failure.
    """


class ServerError(RuntimeError):
    """The server refused. `detail` carries what it said."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


@dataclass
class Run:
    """One workflow run. Holds a run id and nothing else of substance.

    All the state is server side, which is what makes a dropped connection cost
    a round trip rather than the work: `resume()` picks up where this left off
    and nothing re-executes.
    """

    session: Session
    run_id: str
    executor: StepExecutor
    status: str = "open"
    permitted: list[Figures] = field(default_factory=list)
    selection: str = "any"
    stopped: Figures | None = None
    artifact: Figures | None = None

    #: WHAT THIS RUN HAS PRODUCED SO FAR, and it has to live here.
    #:
    #: Three consumers read `run.figures` — the agent's step choice
    #: (`AgentDriver._ask`), the numbers an answer may cite (`figures_of`) and
    #: the caveats attached to it (`caveats_of`) — and NOTHING DEFINED IT. All
    #: three quietly received `{}` from `getattr(run, "figures", {})`. So the
    #: agent chose every step blind, and the synthesis pass was handed no
    #: numbers at all, reached for one in the question ("the s&p 500") and was
    #: refused by the citation guard. The guard was the only part of that chain
    #: working, and it read to the user as the product breaking.
    #:
    #: FILLED FROM THE STEPS THIS MACHINE EXECUTED, never from the server. The
    #: directive deliberately carries no figures — `Directive`'s absences are
    #: load-bearing, and figures are keyed by stage, which is the shape of the
    #: graph. But the client COMPUTES these: `executor.execute` returns them.
    #: Keying by `op` rather than by stage keeps it that way, because an op is
    #: public vocabulary and a stage name is not.
    figures: Figures = field(default_factory=dict)

    #: What the last failed step said, for a caller that renders failures.
    #: Cleared on the next success — this is display state, not the record;
    #: the record is the server-side trace, which keeps every attempt.
    last_error: str | None = None

    def _absorb(self, directive: Figures) -> None:
        self.status = directive.get("status", "open")
        self.permitted = list(directive.get("permitted") or [])
        self.selection = directive.get("selection") or "any"
        self.stopped = directive.get("stop")
        self.artifact = directive.get("artifact")
        # The sealed artifact carries the run's figures as the SERVER assembled
        # them — including values it derived that this machine never saw. Merged
        # at close so the final answer cites the complete record rather than
        # only the half this process happened to compute.
        if isinstance(self.artifact, dict):
            sealed = self.artifact.get("figures")
            if isinstance(sealed, dict):
                self.figures.update(sealed)

    def resume(self) -> Run:
        self._absorb(self.session._get(f"/api/harness/runs/{self.run_id}"))
        return self

    def trace(self) -> list[Figures]:
        """The run's own record. Yours, including the steps that failed."""
        out = self.session._get(f"/api/harness/runs/{self.run_id}/trace")
        return list(out.get("events") or [])

    def step(self, step: Figures) -> Figures:
        """Execute one permitted step locally and report the figures."""
        attempt_id = uuid.uuid4().hex
        try:
            figures = self.executor.execute(step["op"], step.get("params"))
            # RECORDED BEFORE THE ROUND TRIP, so an agent asked to choose the
            # next step sees what the last one produced. That ordering is the
            # whole point: a chooser handed an empty dict is not choosing.
            if isinstance(figures, dict):
                self.figures[str(step["op"])] = figures
            body = {"step_id": step["step_id"], "attempt_id": attempt_id, "figures": figures}
            self.last_error = None
        except Exception as exc:  # noqa: BLE001 — reported, never swallowed; see below
            # Report the failure rather than dropping it — OR crashing. A
            # silently skipped step is a gap the server cannot see, and an
            # exception that unwinds the whole loop turns one bad step into a
            # traceback where a directive belongs. `UnsupportedOp` is the
            # designed refusal; anything else is the caller's own code failing
            # (their backtest_fn, their data's shape) and gets the same honest
            # treatment: named, recorded in the trace, and re-offered once.
            body = {
                "step_id": step["step_id"],
                "attempt_id": attempt_id,
                "ok": False,
                "error": str(exc),
            }
            self.last_error = str(exc)
        directive = self.session._post(f"/api/harness/runs/{self.run_id}/steps", body)
        self._absorb(directive)
        return directive

    def drive(self, *, max_steps: int = 200, max_attempts: int = 2) -> Run:
        """Run to completion: execute what is permitted until stopped or closed.

        `max_steps` is a backstop against a server that keeps issuing work, not
        an expected limit. Hitting it is a bug somewhere, so it raises rather
        than returning a half-finished run that looks finished.

        ── WHY `max_attempts` EXISTS ──────────────────────────────────────────

        A step this build cannot execute reports `ok: False`, and the server
        answers by permitting THE SAME STEP AGAIN. That is correct for a
        transient failure and catastrophic for a permanent one: an op with no
        handler fails identically every time, so the loop spun the same
        impossible directive two hundred times and then raised a
        `max_steps` error naming a limit that had nothing to do with the
        problem. Survivable in a script that nobody watches. In an interactive
        session it is two hundred round trips and a message that points at the
        wrong thing.

        A step that fails twice in a row is a step that will not succeed, so the
        run stops with the REASON — the op and what it said — rather than
        exhausting a counter. That is the same instinct as `Stop` being a 200:
        "this cannot proceed" is a result, and it should read like one.
        """
        n = 0
        failures: dict[str, int] = {}
        while self.status == "open":
            if n >= max_steps:
                raise RuntimeError(f"run {self.run_id} exceeded {max_steps} steps without finishing")
            if not self.permitted:
                self.resume()
                if self.status != "open" or not self.permitted:
                    break

            # selection "all": everything in the set, they do not branch.
            # selection "any": pick one. Without an agent, first is as good a
            # choice as any and is at least deterministic.
            batch = self.permitted if self.selection == "all" else self.permitted[:1]
            for s in batch:
                before = self.status
                self.step(s)
                n += 1
                if self.status != "open":
                    return self

                # Did the server hand back the same step? Then the report we
                # just filed was a failure it wants retried.
                still_offered = any(p.get("step_id") == s.get("step_id") for p in self.permitted)
                if still_offered and before == "open":
                    key = str(s.get("step_id"))
                    failures[key] = failures.get(key, 0) + 1
                    if failures[key] >= max_attempts:
                        op = s.get("op")
                        self.status = "abandoned"
                        self.stopped = {
                            "reason": "step_failed",
                            "op": op,
                            "step_id": key,
                            "attempts": failures[key],
                            "detail": (
                                f"{op!r} failed {failures[key]} times and the server keeps "
                                f"offering it. This build cannot execute it: supply a handler "
                                f"for {op!r}, or upgrade alphaengine."
                            ),
                        }
                        return self
                else:
                    failures.pop(str(s.get("step_id")), None)
        return self


@dataclass
class Session:
    base_url: str
    api_key: str | None = None

    def open(
        self,
        workflow: str,
        *,
        version: str | None = None,
        data: Any = None,
        backtest_fn: Any = None,
        handlers: dict[str, Handler] | None = None,
        workspace_id: str | None = None,
        **inputs: Any,
    ) -> Run:
        """Open a run. `data` and `backtest_fn` stay on this machine.

        `workspace_id` names the SLEEVE this run belongs to. Optional, and
        supplying it is what connects a run to the object model: the server
        resolves the risk budget that binds the sleeve, and the result rolls
        into the pod's book. Without it the run is complete, correct, and
        attached to nothing — which is why several finished mechanisms on the
        platform had never once received input.
        """
        body: Figures = {"workflow": workflow, "version": version, "inputs": inputs}
        if workspace_id:
            body["workspace_id"] = workspace_id
        directive = self._post("/api/harness/runs", body)
        run = Run(
            session=self,
            run_id=directive["run_id"],
            executor=StepExecutor(data=data, backtest_fn=backtest_fn, handlers=handlers),
        )
        run._absorb(directive)
        return run

    def workflows(self) -> list[Figures]:
        """Names and versions. That is all a client is given, and all it needs."""
        return list(self._get("/api/harness/workflows").get("workflows") or [])

    def universes(self) -> list[Figures]:
        """Universes you registered in the portal: names, and the SYMBOLS in them.

        A universe is a DEFINITION — which names, and how you source them — so
        this call moves no market data. `universe_series` is the separate call
        for the case where you asked the portal to hold your closes too.
        """
        return list(self._get("/api/me/universes").get("universes") or [])

    def universe_series(self, universe_id: str, *, window: int = 0) -> Figures:
        """The closes you stored WITH a universe, decrypted back to you.

        ── WHY THIS IS NOT A §9 VIOLATION, AND THE DISTINCTION IS THE WHOLE POINT

        §9 is that WE never fetch market data on your behalf and never hold a
        series as our own. This is neither: it is data you uploaded, encrypted
        at rest, returned to the same account that put it there. The portal's
        own endpoint calls it "owner-only decrypt… lets another device of the
        SAME account run on a universe registered elsewhere" — and the CLI is
        that other device. Nothing new crosses a boundary; a thing you already
        own moves between two of your own surfaces.

        The alternative is worse and is what shipped first: register an S&P
        universe in the portal, then be told to find the same CSV on disk before
        the OS will look at it. That is not a data boundary, it is a missing
        function wearing one.

        Raises `ServerError` with a 404 when the universe was registered
        browser-only — symbols and no stored series, which is a real and common
        choice rather than a fault.
        """
        path = f"/api/me/universes/{universe_id}/series"
        if window:
            path += f"?window={int(window)}"
        return self._get(path)

    def save_signals(
        self,
        *,
        asof: str,
        rows: list[Figures],
        label: str = "signals",
        workspace_id: str | None = None,
        engine_version: str | None = None,
        notes: list[str] | None = None,
        source_run_id: str | None = None,
    ) -> Figures:
        """Record a ranked file: ticker, rank, score, target weight. Dated.

        ── THE ONLY THING THIS PACKAGE WRITES, AND THAT IS THE POINT ───────────

        Everything else here reads. This is the one call that leaves an object
        behind, and it exists because a run that produces a shortlist and then
        prints it has produced nothing: the desk that wanted the list has to
        copy it out of a terminal, and next week nobody can say what yesterday's
        looked like or what moved.

        WHAT MAY BE IN A ROW, AND WHAT MAY NEVER BE. Ticker, rank, score, target
        weight. Never a share count, never a per-account quantity, never an
        order. That boundary is permanent and it is enforced on the server by an
        allowlist rather than by this docstring — a row carrying `shares` is
        REFUSED, not quietly stripped, because a caller who sent quantities
        believes they were stored.

        `notes` travels WITH the file on purpose. A ranked list read without the
        caveats the run attached to it is the exact failure every honesty control
        in this product exists to prevent, and the caveats are least likely to
        survive the step where the list gets handed to somebody else.

        Idempotent on (label, asof): re-running a screen on the same day
        REPLACES that day's file and bumps its version rather than accumulating
        near-duplicates that leave a reader choosing between versions of the
        truth.
        """
        body: Figures = {"asof": asof, "rows": rows, "label": label}
        if workspace_id:
            body["workspace_id"] = workspace_id
        if engine_version:
            body["engine_version"] = engine_version
        if notes:
            body["notes"] = notes
        if source_run_id:
            body["source_run_id"] = source_run_id
        return self._post("/api/me/signals", body)

    def theses(self) -> list[Figures]:
        """The claims on your account, including drafts with no sleeve yet.

        A draft is a thesis somebody wrote and has not acted on. It is the input
        to the only verb in this package that creates anything, so the CLI has
        to be able to see one.
        """
        return list(self._get("/api/me/theses").get("theses") or [])

    def propose_sleeve(
        self,
        *,
        thesis_id: str,
        rows: list[Figures],
        rationale: str,
        caveats: list[str] | None = None,
        payload: Figures | None = None,
        run_id: str | None = None,
        book_id: str | None = None,
    ) -> Figures:
        """File a proposed sleeve for a thesis. THE ONLY THING THIS MAKES.

        ── WHAT THIS IS AND IS NOT ────────────────────────────────────────────

        It creates a PROPOSAL, never a sleeve and never a position. The screen
        ran here, on this machine, against data that never moved; what crosses
        is the shortlist and the argument for it. Somebody then reads it and
        decides — in the portal, because deciding is a human act and a
        long-lived key pressing accept would be the machine approving its own
        work through one extra hop.

        THE CAVEATS ARE NOT OPTIONAL. They travel with the proposal because the
        handoff is exactly where they get dropped, and a shortlist read without
        what the run could not establish is the failure every control in this
        product exists to prevent.
        """
        body: Figures = {
            "thesis_id": thesis_id,
            "rows": rows,
            "rationale": rationale,
            "caveats": list(caveats or []),
        }
        if payload:
            body["payload"] = payload
        if run_id:
            body["source_run_id"] = run_id
        if book_id:
            body["book_id"] = book_id
        return self._post("/api/me/proposals/from-os", body)

    def sleeves(self) -> list[Figures]:
        """The sleeves you can run against, if your account is in a pod.

        A sleeve is a workspace with a pod, a stage and a PM. Naming one on a
        run is what lets the platform resolve the risk budget that binds it and
        roll the result into the pod's book — without it every run is orphaned
        from the object model and every sleeve reports zero.

        An account with no pod has no sleeves, and that is a working product
        rather than an error: a solo quant is a first-class user here.
        """
        return list(self._get("/api/me/sleeves").get("sleeves") or [])

    # ── transport ──────────────────────────────────────────────────────────
    def _request(self, method: str, path: str, body: Figures | None = None) -> Figures:
        url = f"{self.base_url.rstrip('/')}{path}"
        payload = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=payload, method=method)
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")

        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                out: Figures = json.loads(resp.read().decode() or "{}")
                return out
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            with contextlib.suppress(ValueError, AttributeError):
                detail = json.loads(detail).get("detail", detail)
            raise ServerError(exc.code, str(detail)) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise Offline(
                f"no workflow server at {self.base_url}. The maths in this package "
                "still runs offline; workflows need a connection."
            ) from exc

    def _get(self, path: str) -> Figures:
        return self._request("GET", path)

    def _post(self, path: str, body: Figures) -> Figures:
        return self._request("POST", path, body)


def connect(base_url: str, *, api_key: str | None = None) -> Session:
    """A session against a workflow server. Makes no network call by itself."""
    return Session(base_url=base_url, api_key=api_key)


def connect_or_offline(base_url: str, *, api_key: str | None = None) -> Session | None:
    """Connect, or return None if no server answers.

    For the script that should keep working either way. Returning None rather
    than a degraded Session is on purpose: the caller has to decide what to do
    without workflows, and a stub that silently does less would make that
    decision for them and hide it.
    """
    session = connect(base_url, api_key=api_key)
    try:
        session.workflows()
    except Offline:
        logger.info("no workflow server at %s; primitives only", base_url)
        return None
    except ServerError:
        # Reachable but unhappy (auth, most likely). That is a real server and
        # the caller should see the error, not an offline story.
        raise
    return session
