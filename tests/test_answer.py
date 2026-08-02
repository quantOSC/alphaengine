"""The synthesis pass: an answer, or a refusal, and never an invented number.

WHAT THESE ARE ACTUALLY GUARDING. `pick()` returning an index is the safety
property of the driving loop; this file holds the equivalent property for the
answering call. The model may write freely and may quote nothing the run did not
produce, checked after it replies and REFUSED rather than repaired — because an
answer with the invented figure quietly removed still reads as though the system
knew something, and prose is the format in which a made-up number is most likely
to be believed.

The second half is the caveats, and they are derived from the figures rather than
asked of the model. A caveat the model was invited to include is one it can
decline to include.
"""

from __future__ import annotations

import pytest

from alphaengine.agent import Answer, UncitedFigure, caveats_of, figures_of, synthesize
from alphaengine.agent.answer import check_citations


class Run:
    """Anything exposing `figures` and optionally `stopped` will do."""

    def __init__(self, figures, stopped=None):
        self.figures = figures
        self.stopped = stopped


def echo(text: str):
    return lambda _question, _figures: text


SWEPT = {
    "resolve": {"n_obs": 600, "hash": "abc"},
    "sweep": {
        "n_trials": 240,
        "n_trials_source": "derived_from_grid",
        "share_within_20pct_of_best": 0.42,
        "best_sharpe": 1.37,
    },
    "validate:0": {"deflated_sharpe": 0.61, "verdict": "marginal"},
}


# ── flattening what the run produced ───────────────────────────────────────


def test_figures_are_flattened_to_stage_and_key():
    got = figures_of(Run(SWEPT))
    assert got["sweep.n_trials"] == 240.0
    assert got["validate:0.deflated_sharpe"] == 0.61


def test_only_numbers_are_citable():
    """A model that misquotes a verdict is wrong in a way a reader can see. One
    that misquotes a Sharpe is not."""
    got = figures_of(Run(SWEPT))
    assert "sweep.n_trials_source" not in got
    assert "resolve.hash" not in got


def test_booleans_are_not_figures():
    got = figures_of(Run({"screen": {"truncated": True, "n_passing": 31}}))
    assert "screen.truncated" not in got
    assert got["screen.n_passing"] == 31.0


# ── the citation check ─────────────────────────────────────────────────────


def test_an_answer_may_quote_what_the_run_recorded():
    known = figures_of(Run(SWEPT))
    check_citations("The deflated Sharpe is 0.61 across 240 trials.", known)


def test_an_invented_figure_is_refused_not_trimmed():
    """THE ONE THAT MATTERS."""
    known = figures_of(Run(SWEPT))
    with pytest.raises(UncitedFigure) as e:
        check_citations("The deflated Sharpe is 0.94, which clears the bar.", known)
    assert "0.94" in str(e.value)


def test_rounding_is_not_treated_as_invention():
    """Quoting 0.55 for a figure recorded as 0.5478 is quoting the run. Refusing
    it is pedantry that teaches callers to turn the check off."""
    known = figures_of(Run({"v": {"deflated_sharpe": 0.5478}}))
    check_citations("A deflated Sharpe of 0.55.", known)


def test_percent_and_fraction_are_the_same_figure():
    """Both spellings appear across these figures by design — max_drawdown_pct
    is a percent, target_weight is a fraction."""
    known = figures_of(Run({"m": {"max_drawdown_pct": 9.12}}))
    check_citations("A drawdown of 0.0912.", known)

    known = figures_of(Run({"s": {"target_weight": 0.04}}))
    check_citations("Hold 4.0 percent.", known)


def test_small_bare_integers_are_prose():
    """A model writing "3 of the 5 names held up" is doing arithmetic on the
    run's own results, and refusing that sentence makes this unusable. The trade
    is documented in the module rather than papered over."""
    known = figures_of(Run(SWEPT))
    check_citations("3 of the 5 configurations held up.", known)


def test_a_large_bare_integer_must_still_be_traceable():
    """Where the gap closes again: 100 and up looks like a figure, not a count
    somebody did in their head."""
    known = figures_of(Run(SWEPT))
    check_citations("Across 240 trials.", known)
    with pytest.raises(UncitedFigure):
        check_citations("Across 9999 trials.", known)


def test_a_thousands_separator_does_not_smuggle_a_number_through():
    known = figures_of(Run({"s": {"n": 1200.0}}))
    check_citations("1,200 observations.", known)
    with pytest.raises(UncitedFigure):
        check_citations("4,321 observations.", known)


# ── caveats are derived, never requested ───────────────────────────────────


def test_an_unrecorded_trial_count_caps_the_verdict():
    run = Run({"validate": {"n_trials": None, "n_trials_source": "not_recorded", "deflated_sharpe": None}})
    caveats, ceiling = caveats_of(run)
    assert ceiling == "inconclusive"
    assert any("edge" in c for c in caveats)


def test_an_asserted_count_is_named_as_stated_rather_than_counted():
    caveats, ceiling = caveats_of(Run({"v": {"n_trials": 40, "n_trials_source": "asserted"}}))
    assert ceiling == "marginal"
    assert any("stated rather than counted" in c for c in caveats)


def test_a_derived_count_carries_no_caveat():
    caveats, ceiling = caveats_of(Run(SWEPT))
    assert ceiling is None
    assert not any("trial count" in c for c in caveats)


def test_a_monitor_that_checked_nothing_does_not_read_as_clean():
    """`breaches: []` meaning "nobody was watching" is this product's own failure
    mode arriving inside the product."""
    caveats, _ = caveats_of(Run({"package": {"status": "unchecked", "n_breaches": 0, "n_checked": 0}}))
    assert any("not a clean bill" in c for c in caveats)


def test_names_that_could_not_be_measured_are_named():
    caveats, _ = caveats_of(Run({"screen": {"n_insufficient": 40, "universe_size": 500, "n_passing": 31}}))
    assert any("40 of 500" in c and "ranked low" in c for c in caveats)


def test_null_is_reported_as_not_recorded_and_never_as_zero():
    caveats, _ = caveats_of(Run({"measure": {"sharpe_annualized": None, "n_obs": 400}}))
    assert any("Absent, not zero" in c for c in caveats)
    assert any("sharpe_annualized" in c for c in caveats)


def test_the_same_gap_in_two_stages_is_one_fact_not_two():
    caveats, _ = caveats_of(
        Run(
            {
                "a": {"status": "unchecked"},
                "b": {"status": "unchecked"},
            }
        )
    )
    assert len([c for c in caveats if "clean bill" in c]) == 1


# ── the pass ───────────────────────────────────────────────────────────────


def test_an_answer_comes_back_with_its_caveats_attached():
    run = Run({"v": {"n_trials": None, "n_trials_source": "not_recorded", "sharpe_annualized": 1.4}})
    answer = synthesize("did it work?", run, write=echo("The annualised Sharpe is 1.4."))
    assert isinstance(answer, Answer)
    assert answer.text == "The annualised Sharpe is 1.4."
    assert answer.verdict_ceiling == "inconclusive"
    assert answer.caveats, "the caveats did not travel with the answer"


def test_a_stop_is_the_answer_and_the_model_is_not_asked_to_rephrase_it():
    """A refusal is already in plain words written by somebody who knew why.
    Handing it to a model risks softening it into a hedge, which is the one
    direction this product cannot afford to drift."""
    run = Run(
        {"resolve": {"n_obs": 30}}, stopped={"reason": "The series is too short for a deflated result."}
    )

    def never(_q, _f):
        raise AssertionError("the model was asked to rewrite a refusal")

    answer = synthesize("is it good?", run, write=never)
    assert answer.stopped
    assert "too short" in answer.text


def test_the_pass_refuses_an_answer_that_invented_a_number():
    with pytest.raises(UncitedFigure):
        synthesize("how did it do?", Run(SWEPT), write=echo("A Sharpe of 3.99."))


def test_the_writer_sees_the_figures_and_not_the_workflow():
    """It is not given the workflow because neither is this process."""
    seen = {}

    def capture(question, figures):
        seen["q"], seen["f"] = question, figures
        return "240 trials ran."

    synthesize("what happened?", Run(SWEPT), write=capture)
    assert seen["q"] == "what happened?"
    assert seen["f"]["sweep.n_trials"] == 240.0
    assert not any("stage" in k or "gate" in k for k in seen["f"])


def test_the_writer_cannot_mutate_the_runs_figures():
    def meddle(_q, figures):
        figures["sweep.n_trials"] = 1.0
        return "240 trials ran."

    answer = synthesize("what happened?", Run(SWEPT), write=meddle)
    assert answer.cited["sweep.n_trials"] == 240.0


def test_a_run_shaped_as_a_plain_dict_works_too():
    answer = synthesize("?", {"figures": SWEPT}, write=echo("240 trials."))
    assert answer.cited["sweep.n_trials"] == 240.0
