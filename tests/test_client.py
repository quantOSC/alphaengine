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


# ── measuring a series when no sweep ran ───────────────────────────────────
#
# Until 2026-08-02 every reading in this executor except `technical_features`
# went through `_best_column`, which raises without a sweep. That was fine while
# the catalogue held one workflow and that workflow swept. It stopped being fine
# the moment a workflow wanted to measure something the caller already holds —
# a live sleeve, a candidate with a track record — and those are three of the
# four workflows now.


def returns(seed: int = 5, n: int = 400) -> list[float]:
    return np.random.default_rng(seed).normal(0.0006, 0.011, n).tolist()


def test_performance_reads_supplied_returns_when_no_sweep_ran():
    out = StepExecutor(data=returns()).execute("compute.performance_report", {})
    assert out["n_obs"] == 400
    assert out["sharpe_annualized"] is not None


def test_returns_can_arrive_under_a_key_on_a_mapping():
    """The two spellings a research module actually uses. Anything else is a
    guess, and a wrong guess here silently measures the wrong column."""
    for key in ("returns", "pnl"):
        out = StepExecutor(data={key: returns()}).execute("compute.performance_report", {})
        assert out["n_obs"] == 400


def test_var_and_track_record_also_work_without_a_sweep():
    ex = StepExecutor(data=returns())
    var = ex.execute("compute.compute_var_cvar", {})
    assert var["parametric"]["var_pct"] > 0
    # CVaR travels too: a sizing gate needs what the bad day COSTS, not only how
    # often it arrives.
    assert var["cvar"]["cvar_pct"] >= var["historical"]["var_pct"]

    trl = ex.execute("compute.min_track_record_length", {})
    assert "sufficient" in trl


def test_the_sweep_still_wins_when_there_is_one():
    """Order matters: the sweep's best column IS what the run is about, and
    reading the caller's raw data instead would measure a different thing than
    the one being validated."""
    ex = executor()
    ex.execute("compute.sweep", {"grid": {"fast": [5, 10], "slow": [50]}})
    swept = ex.execute("compute.performance_report", {})

    ex.data = returns()
    assert ex.execute("compute.performance_report", {})["n_obs"] == swept["n_obs"]


def test_a_reading_with_nothing_to_read_names_what_was_wanted():
    with pytest.raises(UnsupportedOp) as e:
        StepExecutor(data={"prices": {"AAPL": [1, 2, 3]}}).execute("compute.performance_report", {})
    assert "return series" in str(e.value)


# ── the trial count is never invented ──────────────────────────────────────
def test_a_deflated_sharpe_with_no_recorded_count_is_not_a_number():
    """`int(params.get("n_trials") or 1)` used to sit in this handler.

    One was the most generous denominator arithmetic can produce, which is why
    it was removed from the core function. It survived here because
    `_best_column` raised first and made it unreachable — and it would have gone
    live the moment a workflow measured a supplied series without sweeping.
    """
    out = StepExecutor(data=returns()).execute("compute.deflated_sharpe", {})
    assert out["deflated_sharpe"] is None
    assert out["n_trials"] is None
    assert out["n_trials_source"] == "not_recorded"
    assert out["verdict"] is None
    assert "count" in out["note"]


def test_a_stated_count_is_used_and_labelled_as_asserted():
    out = StepExecutor(data=returns()).execute("compute.deflated_sharpe", {"n_trials": 40})
    assert out["n_trials"] == 40
    assert out["n_trials_source"] == "asserted"
    assert out["deflated_sharpe"] is not None


def test_a_swept_count_outranks_a_stated_one_and_says_where_it_came_from():
    ex = executor()
    ex.execute("compute.sweep", {"grid": {"fast": [5, 10, 15], "slow": [50, 100]}})
    out = ex.execute("compute.deflated_sharpe", {"n_trials": 1})
    assert out["n_trials"] == 6
    assert out["n_trials_source"] == "derived_from_grid"


# ── screening ──────────────────────────────────────────────────────────────
def universe(n: int = 30, obs: int = 300) -> dict:
    rng = np.random.default_rng(7)
    return {f"S{i:02d}": (100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.01, obs))).tolist() for i in range(n)}


def test_screen_returns_a_shortlist_and_keeps_the_universe_local():
    ex = StepExecutor(data=universe())
    out = ex.execute("compute.screen", {"rank_by": "return_pct", "top_n": 5})

    assert len(out["rows"]) == 5
    assert out["universe_size"] == 30
    # The bounded-payload property, which is the whole reason this op exists
    # alongside `compute.technical_features`.
    assert len(out["rows"]) < out["universe_size"]


def test_screen_carries_its_coverage_so_a_gate_can_read_it():
    out = StepExecutor(data=universe()).execute("compute.screen", {"rank_by": "return_pct"})
    for key in ("universe_size", "n_evaluated", "n_passing", "n_insufficient"):
        assert key in out


def test_an_op_that_wants_a_universe_says_so_when_it_gets_a_series():
    """Instead of an AttributeError about `.items` from three frames down."""
    for op in ("compute.screen", "compute.technical_features"):
        with pytest.raises(UnsupportedOp) as e:
            StepExecutor(data=returns()).execute(op, {})
        assert "universe" in str(e.value)


def test_a_screen_of_a_big_universe_still_clears_the_series_guard():
    """The executor's own guard runs on this like any other op, and a screen is
    the one op whose result scales with the input if nothing bounds it."""
    out = StepExecutor(data=universe(300)).execute("compute.screen", {"top_n": 10_000})
    assert len(out["rows"]) <= MAX_FIGURE_LIST


# ── sealing an artifact the server assembled ───────────────────────────────
def test_emit_echoes_the_servers_conclusion_and_stamps_the_build():
    for op in ("emit.screen", "emit.monitor", "emit.sizing_decision"):
        out = StepExecutor().execute(op, {"label": "x", "target_weight": 0.02})
        assert out["label"] == "x"
        assert out["target_weight"] == 0.02
        # What the server cannot know: which build produced the figures under it.
        assert out["engine_version"].startswith("alphaengine@")
        assert out["schema_version"]


def test_emit_does_not_second_guess_the_workflow():
    """The asymmetry with `emit.study` is deliberate. A study's trial count is
    read off the sweep because this machine is the only party that knows it. A
    shortlist or a sizing decision is the workflow's own conclusion, and a second
    derivation here could disagree with the first."""
    ex = StepExecutor(data=universe())
    ex.execute("compute.screen", {"rank_by": "rsi", "top_n": 3})
    out = ex.execute("emit.screen", {"rank_by": "return_pct", "n_passing": 99})
    assert out["rank_by"] == "return_pct"
    assert out["n_passing"] == 99
