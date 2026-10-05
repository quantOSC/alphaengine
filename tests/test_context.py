"""A later question may quote the session. It may not invent a figure."""

from __future__ import annotations

from alphaengine.context import consider, loaded_note, remember, result_note


def test_the_window_keeps_the_recent_notes() -> None:
    notes: list[str] = []
    for i in range(20):
        notes = remember(notes, f"loaded set {i}: 3 names")
    assert len(notes) == 12
    assert notes[0].startswith("loaded set 8")
    assert notes[-1].startswith("loaded set 19")


def test_a_loaded_book_is_a_shape_not_a_series() -> None:
    note = loaded_note("book", {"AAA": [10.0, 11.0, 12.0], "BBB": [1.0, 2.0]})
    assert "2 names" in note
    assert "10.0" not in note


def test_an_answer_may_quote_a_recorded_figure() -> None:
    notes = ["correlation: 3 names, 39 observations. Strongest AAA BBB 0.82."]

    def think(prompt: str) -> str:
        assert "0.82" in prompt
        assert "AAA" in prompt
        return '{"answer": "The strongest pair is AAA and BBB at 0.82."}'

    kind, text = consider("what stands out", notes, think)
    assert kind == "answer"
    assert "0.82" in text


def test_an_invented_figure_is_refused() -> None:
    notes = ["correlation: 3 names, 39 observations. Strongest AAA BBB 0.82."]

    def think(_prompt: str) -> str:
        return '{"answer": "The Sharpe is 4.25."}'

    kind, text = consider("how good is it", notes, think)
    assert kind == "refuse"
    assert "4.25" in text


def test_a_question_the_notes_cannot_answer_falls_through() -> None:
    def think(_prompt: str) -> str:
        return '{"workflow": true}'

    kind, text = consider("which names are overbought", ["loaded book: 3 names"], think)
    assert kind == "workflow"
    assert text == ""


def test_an_unparseable_reply_falls_through() -> None:
    kind, _text = consider("what now", ["loaded book: 3 names"], lambda _prompt: "not json")
    assert kind == "workflow"


def test_no_notes_does_not_call_the_model() -> None:
    def think(_prompt: str) -> str:
        raise AssertionError("the model should not be asked when the desk is empty")

    kind, _text = consider("what stands out", [], think)
    assert kind == "workflow"


def test_a_portfolio_note_keeps_the_largest_weight() -> None:
    note = result_note(
        {
            "kind": "portfolio",
            "method": "hrp",
            "n_assets": 3,
            "weight_sum": 1.0,
            "weights": {"AAA": 0.2, "BBB": 0.5, "CCC": 0.3},
        }
    )
    assert "0.5" in note
    assert "BBB" in note
