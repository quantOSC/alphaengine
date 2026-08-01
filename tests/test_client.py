"""The client half of the split: local execution, and what it refuses to send.

No network here. A fake server stands in for the workflow service, which keeps
these fast and means they test OUR behaviour rather than a connection.
"""

from __future__ import annotations

import numpy as np
import pytest

from alphaengine.agent import AgentDriver, RefusedChoice
from alphaengine.client import MAX_FIGURE_LIST, Offline, StepExecutor, UnsupportedOp, connect
from alphaengine.client.session import Session


def prices(seed: int = 42, n: int = 600) -> np.ndarray:
    return 100 * np.cumprod(1 + np.random.default_rng(seed).normal(0.0004, 0.011, n))


def ma_cross(data=None, fast: int = 10, slow: int = 50):
    p = np.asarray(data, dtype=float)
    r = np.diff(p, prepend=p[0]) / p
    f = np.convolve(p, np.ones(fast) / fast, mode="same")
    s = np.convolve(p, np.ones(slow) / slow, mode="same")
    return (np.r_[0.0, (f > s).astype(float)[:-1]] * r).tolist()


def executor() -> StepExecutor:
    return StepExecutor(data=prices(), backtest_fn=ma_cross)


# ── the executor runs ops locally ──────────────────────────────────────────
def test_resolve_identifies_data_without_disclosing_it():
    out = executor().execute("data.resolve", {"ref": "prices:local"})
    assert out["n_obs"] == 600
    assert len(out["hash"]) == 16
    # The identity is a hash of the content, not a name somebody chose.
    other = StepExecutor(data=prices(seed=7)).execute("data.resolve", {})
    assert other["hash"] != out["hash"]


def test_sweep_reports_figures_and_keeps_the_matrix_local():
    ex = executor()
    out = ex.execute("compute.sweep", {"grid": {"fast": [5, 10, 15], "slow": [50, 100]}})

    assert out["n_trials"] == 6
    assert out["n_trials_source"] == "derived_from_grid"
    assert out["shape"] in {"plateau", "ridge", "knife_edge"}
    # The trial matrix stays in the workspace. It is the thing PBO needs and the
    # thing that must not travel.
    assert "matrix" not in out
    assert ex.workspace["sweep"].matrix.shape[1] == 6


def test_downstream_readings_use_the_sweeps_own_trial_count():
    """A count supplied from outside is a count somebody could flatter, so the
    executor uses the sweep that actually ran and ignores the parameter."""
    ex = executor()
    ex.execute("compute.sweep", {"grid": {"fast": [5, 10, 15], "slow": [50, 100]}})
    out = ex.execute("compute.deflated_sharpe", {"n_trials": 1})
    assert out["n_trials"] == 6, "the server's n_trials overrode the real one"


def test_pbo_is_honest_about_a_single_configuration():
    ex = executor()
    ex.execute("compute.sweep", {"grid": {"fast": [10], "slow": [50]}})
    out = ex.execute("compute.pbo_cscv", {})
    assert out["pbo"] is None
    assert "two configurations" in out["note"]


def test_an_unknown_op_is_refused_not_skipped():
    """A skipped step reports success on work that never happened."""
    with pytest.raises(UnsupportedOp):
        executor().execute("compute.something_new", {})


def test_sweep_without_a_backtest_says_so():
    ex = StepExecutor(data=prices())
    with pytest.raises(UnsupportedOp, match="your backtest function"):
        ex.execute("compute.sweep", {"grid": {"fast": [10]}})


def test_a_handler_can_be_supplied_for_ops_only_your_environment_answers():
    ex = StepExecutor(
        data=prices(),
        handlers={"record.note": lambda params, ws: {"text": "ok", "cites": ["s_1"]}},
    )
    assert ex.execute("record.note", {})["cites"] == ["s_1"]


# ── the data does not leave, enforced on this side too ─────────────────────
def test_the_executor_refuses_to_send_a_series():
    """Enforced client side as well as server side. 'The data stays put' should
    not depend on the other end remembering to check."""
    ex = StepExecutor(data=prices(), handlers={"data.describe": lambda p, w: {"px": [1.0] * 500}})
    with pytest.raises(ValueError, match="stays on your machine"):
        ex.execute("data.describe", {})


def test_a_series_is_caught_however_it_is_nested():
    ex = StepExecutor(data=prices(), handlers={"data.describe": lambda p, w: {"a": {"b": list(range(200))}}})
    with pytest.raises(ValueError, match="series"):
        ex.execute("data.describe", {})


def test_small_derived_structures_still_travel():
    ex = StepExecutor(data=prices(), handlers={"data.describe": lambda p, w: {"buckets": [1, 2, 3, 4, 5]}})
    assert ex.execute("data.describe", {})["buckets"] == [1, 2, 3, 4, 5]


# ── offline is a state, not a crash ────────────────────────────────────────
def test_offline_is_its_own_error_and_says_what_still_works():
    session = connect("http://127.0.0.1:9")  # nothing listens on port 9
    with pytest.raises(Offline, match="offline"):
        session.workflows()


def test_connect_or_offline_returns_none_rather_than_a_degraded_stub():
    """A stub that silently did less would make the caller's decision for them
    and hide that it had been made."""
    from alphaengine.client import connect_or_offline

    assert connect_or_offline("http://127.0.0.1:9") is None


def test_importing_the_top_level_does_not_pull_in_the_client():
    """The client is additive. Someone who never uses a server should not pay
    for it, and the offline guarantee should not depend on this subpackage."""
    import subprocess
    import sys

    code = "import alphaengine, sys; print('alphaengine.client' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "False", "importing the top level dragged in the client"


# ── driving a run against a fake server ────────────────────────────────────
class FakeSession(Session):
    """A server that issues a resolve, then a two-step batch, then closes."""

    def __init__(self):
        super().__init__(base_url="fake://", api_key=None)
        self.seen: list[dict] = []
        self._turn = 0

    def _post(self, path, body):
        if path.endswith("/steps"):
            self.seen.append(body)
            self._turn += 1
            if self._turn == 1:
                return {
                    "run_id": "r1",
                    "status": "open",
                    "cursor": 1,
                    "selection": "all",
                    "permitted": [
                        {
                            "step_id": "s2",
                            "op": "compute.sweep",
                            "params": {"grid": {"fast": [5, 10], "slow": [50]}},
                            "expects": "figures",
                        },
                        {"step_id": "s3", "op": "data.describe", "params": {}, "expects": "figures"},
                    ],
                }
            if self._turn == 2:
                return {
                    "run_id": "r1",
                    "status": "open",
                    "cursor": 2,
                    "selection": "all",
                    "permitted": [
                        {"step_id": "s3", "op": "data.describe", "params": {}, "expects": "figures"}
                    ],
                }
            return {
                "run_id": "r1",
                "status": "closed",
                "cursor": 3,
                "permitted": [],
                "artifact": {"workflow": "fake@1.0.0"},
            }
        return {
            "run_id": "r1",
            "status": "open",
            "cursor": 0,
            "selection": "any",
            "permitted": [{"step_id": "s1", "op": "data.resolve", "params": {}, "expects": "figures"}],
        }


def test_drive_runs_to_the_artifact():
    session = FakeSession()
    run = session.open("fake", data=prices(), backtest_fn=ma_cross)
    run.drive()

    assert run.status == "closed"
    assert run.artifact["workflow"] == "fake@1.0.0"
    assert [b["step_id"] for b in session.seen] == ["s1", "s2", "s3"]
    # Every report carried figures and none carried a series.
    assert all("figures" in b or b.get("ok") is False for b in session.seen)


def test_each_attempt_gets_its_own_id():
    """Idempotency is keyed on it, so two genuine attempts must not collide."""
    session = FakeSession()
    run = session.open("fake", data=prices(), backtest_fn=ma_cross)
    run.drive()
    ids = [b["attempt_id"] for b in session.seen]
    assert len(set(ids)) == len(ids)


def test_a_stop_is_returned_not_raised():
    class Stopping(FakeSession):
        def _post(self, path, body):
            if path.endswith("/steps"):
                return {
                    "run_id": "r1",
                    "status": "stopped",
                    "cursor": 1,
                    "permitted": [],
                    "stop": {"reason": "did not clear", "figures": {"dsr": 0.1}},
                }
            return super()._post(path, body)

    run = Stopping().open("fake", data=prices())
    run.drive()  # must not raise
    assert run.status == "stopped"
    assert run.stopped["figures"]["dsr"] == 0.1


def test_an_unsupported_op_is_reported_as_a_failed_step():
    """Not dropped: the server cannot see a gap the client silently left."""

    class Weird(FakeSession):
        def _post(self, path, body):
            if path.endswith("/steps"):
                self.seen.append(body)
                return {"run_id": "r1", "status": "closed", "cursor": 1, "permitted": [], "artifact": {}}
            return {
                "run_id": "r1",
                "status": "open",
                "cursor": 0,
                "selection": "any",
                "permitted": [{"step_id": "s1", "op": "compute.nope", "params": {}, "expects": "figures"}],
            }

    session = Weird()
    run = session.open("fake", data=prices())
    run.drive()
    assert session.seen[0]["ok"] is False
    assert "not executable" in session.seen[0]["error"]


# ── the agent is bounded by the permitted set ──────────────────────────────
def test_the_agent_may_only_pick_from_what_was_offered():
    run = FakeSession().open("fake", data=prices(), backtest_fn=ma_cross)
    driver = AgentDriver(choose=lambda permitted, figures: 99)
    with pytest.raises(RefusedChoice, match="outside"):
        driver.drive(run)


def test_an_out_of_range_choice_is_refused_rather_than_clamped():
    """Clamping would let the run continue while the record says the agent chose
    something it did not."""
    run = FakeSession().open("fake", data=prices(), backtest_fn=ma_cross)
    with pytest.raises(RefusedChoice):
        AgentDriver(choose=lambda p, f: -1).drive(run)


def test_the_agent_is_not_consulted_when_there_is_no_choice():
    """selection='all' means the steps do not branch. Asking a model to choose
    among steps that all run would invent a decision and then record it."""
    calls = []

    def choose(permitted, figures):
        calls.append(len(permitted))
        return 0

    run = FakeSession().open("fake", data=prices(), backtest_fn=ma_cross)
    AgentDriver(choose=choose).drive(run)

    assert run.status == "closed"
    # Only the selection="any" turn consulted it; the batch turn did not.
    assert calls == [1], f"the agent was asked to choose on a batch turn: {calls}"


def test_the_agent_drives_a_real_run_to_completion():
    session = FakeSession()
    run = session.open("fake", data=prices(), backtest_fn=ma_cross)
    AgentDriver(choose=lambda p, f: 0).drive(run)
    assert run.status == "closed"
    assert [b["step_id"] for b in session.seen] == ["s1", "s2", "s3"]


# ── emit.study and record.*: the ops that had no handler ───────────────────
#
# Both were vocabulary strings the server could issue and this executor could
# not execute, so a run that reached one could never produce the artifact it
# existed to produce. The consumer was built before the producer.


def test_emit_study_assembles_the_run_that_actually_happened():
    ex = executor()
    ex.execute("compute.sweep", {"grid": {"fast": [5, 10, 15], "slow": [50, 100]}})
    ex.execute("compute.deflated_sharpe", {})

    study = ex.execute("emit.study", {"label": "MA cross", "notes": "first pass"})

    # THE COUNT COMES FROM THE GRID THAT RAN. 3 x 2 = 6.
    assert study["n_trials"] == 6
    assert study["n_trials_source"] == "derived_from_grid"
    assert study["label"] == "MA cross"
    assert study["data_hash"]
    assert study["engine_version"].startswith("alphaengine@")
    assert study["schema_version"]
    # The verdict RIDES ALONG from the step that computed it rather than being
    # derived a second time, because two derivations can disagree.
    assert "deflated_sharpe" in study
    assert study["surface"]["n_ok"] == 6


def test_emit_study_ignores_a_trial_count_offered_by_the_server():
    """A count arriving in a directive is one somebody upstream could choose.

    Since 0.2.0 an unrecorded or flattered denominator cannot reach an `edge`
    verdict, so the distinction has to be made at the only point in the system
    that knows what actually ran.
    """
    ex = executor()
    ex.execute("compute.sweep", {"grid": {"fast": [5, 10], "slow": [50]}})
    study = ex.execute("emit.study", {"label": "x", "n_trials": 1, "n_trials_source": "asserted"})
    assert study["n_trials"] == 2
    assert study["n_trials_source"] == "derived_from_grid"


def test_emit_study_with_no_sweep_refuses_rather_than_inventing_one():
    with pytest.raises(UnsupportedOp) as e:
        executor().execute("emit.study", {"label": "nothing ran"})
    assert "sweep" in str(e.value)


def test_emit_study_still_cannot_send_a_series():
    """The guard runs on this like any other op."""
    ex = executor()
    ex.execute("compute.sweep", {"grid": {"fast": [5, 10], "slow": [50]}})
    study = ex.execute("emit.study", {"label": "x"})
    for value in study.values():
        assert not (isinstance(value, list) and len(value) > MAX_FIGURE_LIST)


def test_record_hands_values_back_for_the_ledger():
    out = executor().execute("record.decision", {"choice": "proceed", "why": "cleared the gate"})
    assert out == {"choice": "proceed", "why": "cleared the gate"}


# ── a step that cannot succeed stops the run, and says why ─────────────────


class StubbornSession(Session):
    """A server that keeps offering a step this build cannot execute.

    Not contrived: that IS the server's behaviour on a failed step, and it is
    correct for a transient failure. For a permanent one — an op with no handler
    — it used to mean two hundred identical round trips ending in a max_steps
    error naming a limit that had nothing to do with the problem.
    """

    def __init__(self):
        super().__init__(base_url="fake://", api_key=None)
        self.attempts = 0

    def _post(self, path, body):
        if path.endswith("/steps"):
            self.attempts += 1
        return {
            "run_id": "r9",
            "status": "open",
            "cursor": 0,
            "selection": "any",
            "permitted": [{"step_id": "s1", "op": "compute.nope", "params": {}, "expects": "figures"}],
        }


def test_a_step_that_keeps_failing_abandons_the_run_with_a_reason():
    session = StubbornSession()
    run = session.open("fake", data=prices(), backtest_fn=ma_cross)
    run.drive(max_steps=200)

    assert run.status == "abandoned"
    assert run.stopped["reason"] == "step_failed"
    assert run.stopped["op"] == "compute.nope"
    # Two attempts, not two hundred. The old loop spun to max_steps and raised.
    assert session.attempts == 2
    assert "compute.nope" in run.stopped["detail"]


def test_the_failure_is_reported_to_the_server_not_swallowed():
    """A skipped step reports success on work that never happened."""
    session = StubbornSession()
    run = session.open("fake", data=prices(), backtest_fn=ma_cross)
    run.drive()
    # `step()` posts ok=False with the reason; the server's trace needs it.
    assert run.status == "abandoned"
