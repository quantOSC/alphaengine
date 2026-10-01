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
