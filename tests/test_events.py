"""Event allowlist: hashes not prompts, figures not series, 404 never fails a run."""

from __future__ import annotations

from alphaengine.events import EventSink, allowlist, make_event


def test_allowlist_drops_unknown_and_series_keys():
    raw = {
        "kind": "pick",
        "choice": 1,
        "why": "deflate it",
        "prompt": "the full prompt must never leave",
        "returns": list(range(800)),
        "api_key": "sk-nope",
    }
    out = allowlist(raw)
    assert "prompt" not in out
    assert "returns" not in out
    assert "api_key" not in out
    assert out["choice"] == 1
    assert out["why"] == "deflate it"


def test_make_event_hashes_nothing_it_was_not_given():
    event = make_event("answer", run_id="r1", answer_text="sharpe 1.2", answer_caveats=["not recorded"])
    assert event["kind"] == "answer"
    assert event["run_id"] == "r1"
    assert "prompt" not in event


def test_sink_writes_jsonl_and_swallows_a_missing_portal_route(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

    class FakeSession:
        def post_event(self, run_id: str, event: dict) -> dict:
            raise RuntimeError("404: not found")

    sink = EventSink(session=FakeSession(), run_id="abc")
    sink.emit(make_event("pick", run_id="abc", choice=0, why="first"))
    path = tmp_path / "alphaengine" / "runs" / "abc.jsonl"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert '"choice": 0' in text
    assert "first" in text
    # A second emit must not raise either.
    sink.emit(make_event("thought", run_id="abc", why="still going"))
