"""The agentic half: what the model is allowed to decide, and what it is not.

THE SAFETY PROPERTY IS THE RETURN TYPE. `pick()` returns an INDEX into a list
the server produced, so a model physically cannot express an operation that was
not offered. Every test here is really testing that one idea from a different
angle: a hallucinated op, a sloppy reply, an out-of-range number, and a refusal
that must not silently become "the first option".
"""

from __future__ import annotations

import pytest

from alphaengine.client import AgentDriver, AgentRefusal

PERMITTED = [
    {"op": "compute.sweep", "step_id": "s1"},
    {"op": "compute.deflated_sharpe", "step_id": "s2"},
    {"op": "emit.study", "step_id": "s3"},
]


def driver(reply: str) -> AgentDriver:
    return AgentDriver(lambda prompt: reply, goal="validate a momentum idea")


# ── the model chooses among what was offered ───────────────────────────────


def test_a_clean_json_choice_is_taken():
    assert driver('{"choice": 1, "why": "deflate before believing it"}').pick(PERMITTED) == 1


def test_json_wrapped_in_prose_is_still_read():
    """Models narrate. That is not an error and must not be treated as one."""
    reply = 'Sure — I think the sweep first.\n\n{"choice": 0, "why": "need the grid"}\nHope that helps!'
    assert driver(reply).pick(PERMITTED) == 0


def test_json_in_a_code_fence_is_still_read():
    reply = '```json\n{"choice": 2, "why": "emit it"}\n```'
    assert driver(reply).pick(PERMITTED) == 2


def test_a_bare_integer_is_accepted():
    assert driver("1").pick(PERMITTED) == 1


def test_a_single_option_costs_no_model_call():
    """Not an optimisation — a call that cannot go wrong should not be made."""

    def explode(prompt: str) -> str:
        raise AssertionError("the model was consulted about a list of one")

    d = AgentDriver(explode, goal="anything")
    assert d.pick([PERMITTED[0]]) == 0


# ── and cannot choose anything else ────────────────────────────────────────


def test_an_out_of_range_choice_is_refused():
    with pytest.raises(AgentRefusal) as e:
        driver('{"choice": 9}').pick(PERMITTED)
    assert "not offered" in str(e.value)


def test_a_negative_choice_is_refused():
    with pytest.raises(AgentRefusal):
        driver('{"choice": -1}').pick(PERMITTED)


def test_an_invented_op_cannot_be_expressed():
    """THE WHOLE POINT. The model names an operation that does not exist; the
    only channel it has is an index, so the invention has nowhere to go."""
    with pytest.raises(AgentRefusal):
        driver('{"op": "compute.make_me_money"}').pick(PERMITTED)


def test_an_unparseable_reply_is_refused_and_not_defaulted():
    """Quietly taking option 0 would produce a run that LOOKS agent-driven and
    is not — the worst of both, since it is neither reproducible nor chosen."""
    with pytest.raises(AgentRefusal) as e:
        driver("I'm not sure, could you clarify?").pick(PERMITTED)
    assert "no choice found" in str(e.value)


def test_an_empty_permitted_list_is_refused():
    with pytest.raises(AgentRefusal):
        driver("0").pick([])


# ── the narrator ───────────────────────────────────────────────────────────


def test_the_reason_is_surfaced_so_the_loop_can_be_watched():
    seen: list[str] = []
    d = AgentDriver(
        lambda p: '{"choice": 0, "why": "the grid has to run first"}',
        goal="g",
        on_thought=seen.append,
    )
    d.pick(PERMITTED)
    assert seen == ["the grid has to run first"]


def test_history_accumulates_so_later_choices_see_earlier_ones():
    d = driver('{"choice": 0, "why": "first"}')
    d.pick(PERMITTED)
    d.pick(PERMITTED)
    assert len(d.history) == 2
    assert "compute.sweep" in d.history[0]


# ── the prompt carries no workflow knowledge ───────────────────────────────


def test_the_prompt_contains_no_ordering_or_thresholds():
    """`the loop must contain no workflow knowledge` — if this string ever
    starts saying "usually run the sweep first", the client has become a second,
    worse copy of the workflow the server owns."""
    from alphaengine.client.agent import _PROMPT

    lowered = _PROMPT.lower()
    for banned in ("first", "then", "usually", "threshold", "sharpe", "pbo", "0."):
        assert banned not in lowered, f"the prompt leaked workflow knowledge: {banned!r}"


def test_as_choice_is_the_index_bound_primitive():
    from alphaengine.agent.driver import AgentDriver as IndexDriver

    d = AgentDriver(lambda p: '{"choice": 1, "why": "x"}', goal="g")
    assert d.as_choice()(PERMITTED, {}) == 1
    assert callable(IndexDriver)


def test_measured_scalars_reach_the_model_and_series_do_not():
    """A later choice that cannot see what the last step found picks the same
    step again. The scalars go in. A list is a series and stays out."""
    seen: list[str] = []

    def think(prompt: str) -> str:
        seen.append(prompt)
        return '{"choice": 1, "why": "the sweep already reported"}'

    AgentDriver(think, goal="g").pick(
        PERMITTED,
        {"compute.sweep": {"sharpe": 0.4, "curve": [1, 2, 3], "verdict": "stop"}},
    )
    assert "compute.sweep" in seen[0]
    assert "sharpe=0.4" in seen[0]
    assert "verdict=stop" in seen[0]
    assert "[1, 2, 3]" not in seen[0]


def test_a_finished_step_is_not_executed_again():
    """A new step id with the same op and the same params is the same work.
    Running it again is how a sentence loops."""

    class _Replay:
        def __init__(self) -> None:
            self.status = "open"
            self.selection = "any"
            self.permitted = [{"step_id": "s1", "op": "compute.screen", "params": {}}]
            self.figures: dict = {}
            self.calls = 0
            self.stopped = None

        def resume(self) -> None:
            return None

        def step(self, step: dict) -> None:
            self.calls += 1
            self.figures["compute.screen"] = {"sharpe": 1.25}
            self.permitted = [{"step_id": "s9", "op": "compute.screen", "params": {}}]
            self.status = "open"

    run = _Replay()
    AgentDriver(lambda _p: '{"choice": 0}', goal="g").drive(run)
    assert run.calls == 1
    assert run.status == "stopped"
    assert run.stopped["reason"] == "already finished"


def test_a_question_that_names_one_workflow_is_not_left_to_guess():
    from alphaengine.client.agent import prefer_index

    options = [
        {"op": "screen_universe", "answers": "what in my universe is worth a look"},
        {"op": "validate_study", "answers": "does this result hold up once you count the tries"},
    ]
    assert prefer_index("screen my universe", options) == 0
    assert prefer_index("which of my names are overbought", options) is None
    assert prefer_index("look at the universe and validate the study", options) is None


def test_params_are_truncated_before_reaching_the_model():
    """A step's params can carry a parameter grid, which is frequently bigger
    intellectual property than the returns. No reason to ship all of it to a
    third party to label a menu item."""
    from alphaengine.client.agent import _brief

    got = _brief({"grid": {"fast": list(range(500))}})
    assert len(got) <= 121
