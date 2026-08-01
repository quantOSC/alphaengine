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

from .executor import Figures, Handler, StepExecutor, UnsupportedOp

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

    def _absorb(self, directive: Figures) -> None:
        self.status = directive.get("status", "open")
        self.permitted = list(directive.get("permitted") or [])
        self.selection = directive.get("selection") or "any"
        self.stopped = directive.get("stop")
        self.artifact = directive.get("artifact")

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
            body = {"step_id": step["step_id"], "attempt_id": attempt_id, "figures": figures}
        except UnsupportedOp as exc:
            # Report the failure rather than dropping it. A silently skipped step
            # is a gap the server cannot see and the trace cannot explain.
            body = {
                "step_id": step["step_id"],
                "attempt_id": attempt_id,
                "ok": False,
                "error": str(exc),
            }
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
        **inputs: Any,
    ) -> Run:
        """Open a run. `data` and `backtest_fn` stay on this machine."""
        directive = self._post(
            "/api/harness/runs",
            {"workflow": workflow, "version": version, "inputs": inputs},
        )
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
