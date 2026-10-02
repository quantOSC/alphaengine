"""The full-screen session: boot, a scripted run, a sentence, one math line."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

textual = pytest.importorskip("textual")
assert textual is not None

from textual.widgets import DataTable, Input  # noqa: E402

from alphaengine.client.session import Session  # noqa: E402
from alphaengine.tui.app import QuantOSApp  # noqa: E402


def _returns() -> list[float]:
    return np.random.default_rng(8).normal(0.0005, 0.01, 40).tolist()


class _DeskSession(Session):
    def __init__(self) -> None:
        super().__init__(base_url="fake://", api_key=None)
        self.posts: list[tuple[str, dict]] = []

    def _get(self, path: str):
        if path.endswith("/workflows"):
            return {
                "workflows": [
                    {
                        "name": "screen_universe",
                        "requires": [],
                        "reproducible": False,
                        "agency": "exploratory",
                    },
                    {"name": "measure", "requires": [], "reproducible": True, "agency": "scripted"},
                ]
            }
        if path.endswith("/gaps") or path.endswith("/trace"):
            return {"gaps": [], "events": []}
        if "signals" in path:
            return {"signals": []}
        if path.endswith("/universes"):
            return {"universes": [{"name": "sp500"}]}
        return {"run_id": "r1", "status": "closed", "permitted": []}

    def _post(self, path, body):
        self.posts.append((path, body))
        if path.endswith("/traces"):
            return {}
        if path.endswith("/steps"):
            return {
                "run_id": "r1",
                "status": "closed",
                "permitted": [],
                "artifact": {"workflow": body.get("step_id", "measure")},
            }
        return {
            "run_id": "r1",
            "status": "open",
            "selection": "any",
            "permitted": [{"step_id": "s1", "op": "compute.performance_report", "params": {}}],
        }


def _think(prompt: str) -> str:
    if '"choice"' in prompt:
        return '{"choice": 0, "why": "the question is a screen"}'
    return "The run did not measure that."


async def _wait(pilot: object, app: QuantOSApp) -> None:
    pause = pilot.pause  # type: ignore[attr-defined]
    for _ in range(80):
        await pause(0.05)
        if not app.busy:
            return
    raise AssertionError("the line never finished")


def test_boot_shows_the_ladder_and_a_stored_universe() -> None:
    async def body() -> None:
        app = QuantOSApp(session=_DeskSession(), url="fake://", data=_returns(), keyed=True, think=_think)
        async with app.run_test(size=(140, 42)) as pilot:
            await pilot.pause()
            assert "enter model key" in app.rail_text
            assert "load <name>" in app.rail_text
            assert "the maths" in app.rail_text
            assert "sp500" in app.rail_text
            prompt = app.query_one(Input)
            assert str(prompt.placeholder).startswith("Ask")

    asyncio.run(body())


def test_a_scripted_run_opens_the_sharpe_line_at_full_length() -> None:
    async def body() -> None:
        app = QuantOSApp(session=_DeskSession(), url="fake://", data=_returns(), keyed=True, think=_think)
        async with app.run_test(size=(140, 42)) as pilot:
            await pilot.pause()
            field = app.query_one(Input)
            field.focus()
            field.value = "run measure"
            await pilot.press("enter")
            await _wait(pilot, app)
            assert app.desk.last is not None
            assert app.desk.last.status == "closed"
            assert "sqrt" in app.formula_text
            table = app.query_one("#series", DataTable)
            assert table.row_count == 40

    asyncio.run(body())


class _StoredBook(Session):
    """One portal universe whose closes were stored with the upload."""

    def __init__(self, names: list[str]) -> None:
        super().__init__(base_url="fake://", api_key="ae_live_test")
        self._names = names

    def _get(self, path: str):
        if path.endswith("/series"):
            return {"prices": {"AAPL": [10.0, 11.0, 12.0, 13.0]}}
        if path.endswith("/universes"):
            return {
                "universes": [
                    {
                        "id": f"u{i}",
                        "name": name,
                        "definition": {"cache_id": f"c{i}", "symbols": ["AAPL"]},
                    }
                    for i, name in enumerate(self._names)
                ]
            }
        if path.endswith("/workflows"):
            return {"workflows": []}
        return {}

    def _post(self, path, body):
        return {}


def test_a_signed_in_account_loads_its_one_stored_universe() -> None:
    async def body() -> None:
        app = QuantOSApp(session=_StoredBook(["sp500"]), url="fake://", keyed=True, think=_think)
        async with app.run_test(size=(140, 42)) as pilot:
            await pilot.pause()
            assert app.desk.loaded == "universe:sp500"
            assert "AAPL" in app.desk.data
            assert "stored" in app.rail_text
            assert any("portal" in line for line in app.transcript)

    asyncio.run(body())


def test_several_stored_universes_are_named_and_not_guessed() -> None:
    async def body() -> None:
        app = QuantOSApp(session=_StoredBook(["sp500", "ndx"]), url="fake://", keyed=True, think=_think)
        async with app.run_test(size=(140, 42)) as pilot:
            await pilot.pause()
            assert app.desk.data is None
            text = "\n".join(app.transcript)
            assert "sp500" in text and "ndx" in text

    asyncio.run(body())


def test_login_pastes_a_quantos_key_and_login_gemini_pastes_that_provider() -> None:
    async def body() -> None:
        app = QuantOSApp(session=_DeskSession(), url="fake://", keyed=False, think=_think)
        async with app.run_test(size=(140, 42)) as pilot:
            await pilot.pause()
            field = app.query_one(Input)
            field.focus()
            field.value = "login"
            await pilot.press("enter")
            await _wait(pilot, app)
            assert app.desk.pending_secret == "QUANTOS_API_KEY"
            assert app.query_one(Input).password is True

            app.desk.pending_secret = None
            app.desk.keyed = True
            app.query_one(Input).password = False
            field.value = "enter model key"
            await pilot.press("enter")
            await _wait(pilot, app)
            assert app.desk.pending_secret == "MODEL_KEY"
            assert app.query_one(Input).password is True

    asyncio.run(body())


def test_a_model_key_prefix_names_the_provider() -> None:
    from alphaengine.tui.dispatch import env_for_model_key

    assert env_for_model_key("sk-ant-abc") == "ANTHROPIC_API_KEY"
    assert env_for_model_key("sk-or-abc") == "OPENROUTER_API_KEY"
    assert env_for_model_key("gsk_abc") == "GROQ_API_KEY"
    assert env_for_model_key("AIzaabc") == "GEMINI_API_KEY"
    assert env_for_model_key("sk-abc") == "OPENAI_API_KEY"
    assert env_for_model_key("not-a-known-prefix") is None


def test_a_sentence_uses_the_fake_model_to_choose_a_workflow() -> None:
    async def body() -> None:
        app = QuantOSApp(session=_DeskSession(), url="fake://", data=_returns(), keyed=True, think=_think)
        async with app.run_test(size=(140, 42)) as pilot:
            await pilot.pause()
            field = app.query_one(Input)
            field.focus()
            field.value = "which of my names are overbought"
            await pilot.press("enter")
            await _wait(pilot, app)
            assert any("screen_universe" in line for line in app.transcript)

    asyncio.run(body())
