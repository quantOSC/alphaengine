"""The quantOS session. One full-screen terminal: the conversation, then the math."""

from __future__ import annotations

import argparse
import os
import re
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Input, ListItem, ListView, RichLog, Sparkline, Static

from ..cli import narration_to, resolve_data
from .dispatch import Desk, submit

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

_CSS = """
Screen { background: #0b0d10; }
#status {
    dock: top;
    height: 1;
    background: #12151a;
    color: #8b949e;
    padding: 0 1;
}
#body { height: 1fr; }
#rail {
    width: 26;
    min-width: 26;
    background: #101318;
    color: #c5cdd6;
    padding: 1 1;
    overflow: hidden;
}
#transcript {
    width: 1fr;
    height: 1fr;
    background: #0b0d10;
    padding: 0 1;
}
#inspector {
    width: 44;
    min-width: 36;
    background: #101318;
    padding: 1 1;
    overflow: hidden;
    display: none;
}
#inspector-title { color: #8b949e; height: 1; }
#lines { height: 8; }
#formula { height: auto; max-height: 6; color: #9ecbff; }
#inputs { height: auto; max-height: 5; color: #c5cdd6; }
#spark { height: 3; margin: 1 0 0 0; }
#series { height: 1fr; }
#hint {
    dock: bottom;
    height: 1;
    background: #0b0d10;
    color: #6b7380;
    padding: 0 1;
}
#prompt { dock: bottom; margin: 0 1; }
"""


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


def _fit(text: str, width: int = 22) -> str:
    text = " ".join(str(text).split())
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


class QuantOSApp(App[None]):
    """Conversation first. The math pane opens when a run has a line to show."""

    CSS = _CSS
    TITLE = "quantOS"

    def __init__(
        self,
        *,
        session: Any,
        url: str,
        args: Any = None,
        data: Any = None,
        backtest_fn: Any = None,
        loaded: str | None = None,
        keyed: bool = False,
        think: Any = None,
    ) -> None:
        super().__init__()
        self.desk = Desk(
            session=session,
            url=url,
            args=args,
            data=data,
            backtest_fn=backtest_fn,
            loaded=loaded,
            think=think,
            keyed=keyed,
        )
        self._universes: list[str] = []
        self._lines: list[dict[str, Any]] = []
        self.busy = False
        self.rail_text = ""
        self.formula_text = ""
        self.transcript: list[str] = []

    def compose(self) -> ComposeResult:
        yield Static("", id="status")
        with Horizontal(id="body"):
            yield Static("", id="rail")
            yield RichLog(id="transcript", highlight=False, markup=False, wrap=True)
            with Vertical(id="inspector"):
                yield Static("math", id="inspector-title")
                yield ListView(id="lines")
                yield Static("", id="formula")
                yield Static("", id="inputs")
                yield Sparkline([], id="spark")
                yield DataTable(id="series", zebra_stripes=True)
        yield Static("enter send   ·   help   ·   quit", id="hint")
        yield Input(placeholder="Ask in plain English, or run <workflow>", id="prompt")

    def on_mount(self) -> None:
        self.query_one("#series", DataTable).cursor_type = "row"
        self._refresh_rail()
        log = self.query_one("#transcript", RichLog)
        log.write("Ask in plain English.")
        log.write("A sentence picks a workflow.  run <name> follows it exactly.")
        log.write("")
        log.write("help lists what you can type.  quit leaves.")
        self.query_one(Input).focus()
        try:
            rows = self.desk.session.universes()
        except Exception:  # noqa: BLE001 — the ladder still stands without the list
            rows = []
        self._universes = [
            str(row.get("name") or row.get("id") or "") for row in rows if isinstance(row, dict)
        ]
        self._universes = [name for name in self._universes if name]
        self._refresh_rail()

    def _refresh_rail(self) -> None:
        from ..model import available_models, keys_without_sdk

        keyed = self.desk.keyed
        models = available_models()
        stranded = keys_without_sdk()
        ask_on = keyed and bool(models)
        if ask_on:
            ask_note = _fit(models[0][0])
        elif keyed and stranded:
            ask_note = "sdk missing"
        elif keyed:
            ask_note = "add a model key"
        else:
            ask_note = "sign in first"

        on, off = "●", "○"
        if self.desk.loaded:
            loaded = _fit(self.desk.loaded)
        elif self.desk.data is not None:
            loaded = "series"
        else:
            loaded = "nothing"

        lines = [
            f"{on}  the maths",
            "    offline, always",
            "",
            f"{on if keyed else off}  workflows",
            "    ready" if keyed else "    sign in",
            "",
            f"{on if ask_on else off}  ask",
            f"    {ask_note}",
            "",
            "loaded",
            f"  {loaded}",
            "",
            "universes",
        ]
        if self._universes:
            lines.extend(f"  {_fit(name, 20)}" for name in self._universes[:12])
        else:
            lines.append("  none in reach")
        if not keyed:
            lines += ["", "key quantos", "signs you in"]
        self.rail_text = "\n".join(lines)
        self.query_one("#rail", Static).update(self.rail_text)

        bits = ["quantOS", "signed in" if keyed else "not signed in"]
        if self.desk.loaded:
            bits.append(self.desk.loaded)
        elif self.desk.data is not None:
            bits.append("data loaded")
        else:
            bits.append("no data")
        if ask_on:
            bits.append(models[0][0])
        self.query_one("#status", Static).update("   ·   ".join(bits))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text or self.busy:
            return
        if self.desk.pending_secret:
            self._accept_secret(text)
            return
        self.busy = True
        self.query_one("#transcript", RichLog).write(f"> {text}")

        def go() -> None:
            self._run_line(text)

        self.run_worker(go, thread=True, exclusive=True)

    def _accept_secret(self, value: str) -> None:
        env = self.desk.pending_secret
        self.desk.pending_secret = None
        prompt = self.query_one(Input)
        prompt.password = False
        prompt.placeholder = "Ask in plain English, or run <workflow>"
        log = self.query_one("#transcript", RichLog)
        if env is None or not value:
            log.write("nothing entered.")
            return
        os.environ[env] = value
        if env == "QUANTOS_API_KEY":
            from ..cli import _session

            args = self.desk.args
            self.desk.session, self.desk.url = _session(
                getattr(args, "url", None), getattr(args, "key", None)
            )
            self.desk.keyed = True
        try:
            from ..auth import save_key

            path = save_key(env, value)
            log.write(f"{env} accepted.  stored at {path}")
        except (OSError, ValueError) as exc:
            log.write(f"{env} accepted for this process.  ({exc})")
        self._refresh_rail()

    def _run_line(self, text: str) -> None:
        chunks: list[str] = []

        def sink(chunk: str) -> None:
            chunks.append(chunk)

        quitting = False
        try:
            with narration_to(sink):
                quitting = submit(self.desk, text)
        except Exception as exc:  # noqa: BLE001 — the session stays up
            chunks.append(str(exc))
        self.call_from_thread(self._finish, chunks, quitting)

    def _finish(self, chunks: list[str], quitting: bool) -> None:
        log = self.query_one("#transcript", RichLog)
        for chunk in chunks:
            plain = _plain(chunk)
            if plain.strip():
                self.transcript.append(plain)
                log.write(plain)
        if self.desk.pending_secret:
            prompt = self.query_one(Input)
            prompt.password = True
            prompt.placeholder = "paste the key (hidden)"
        self._refresh_rail()
        self._load_lines()
        self.busy = False
        if quitting:
            self.exit()

    def _load_lines(self) -> None:
        run = self.desk.last
        collected: list[dict[str, Any]] = []
        for payload in getattr(run, "traces", None) or []:
            op = str(payload.get("op") or "")
            for line in payload.get("lines") or []:
                if isinstance(line, dict):
                    collected.append({**line, "op": op})
        self._lines = collected
        inspector = self.query_one("#inspector", Vertical)
        view = self.query_one("#lines", ListView)
        for child in list(view.children):
            child.remove()
        if not self._lines:
            inspector.display = False
            self.formula_text = ""
            return
        inspector.display = True
        for index, line in enumerate(self._lines):
            label = _fit(f"{line.get('id', '?')}  {line.get('op', '')}", 40)
            view.append(ListItem(Static(label), name=str(index)))
        self._show_line(0)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.list_view.id != "lines":
            return
        index = event.list_view.index
        if index is not None and 0 <= index < len(self._lines):
            self._show_line(index)

    def _show_line(self, index: int) -> None:
        line = self._lines[index]
        formula = str(line.get("formula") or "")
        self.formula_text = formula
        self.query_one("#formula", Static).update(formula or " ")
        inputs = line.get("inputs") or {}
        rendered = "  ".join(f"{key}={value}" for key, value in inputs.items())
        result = line.get("result")
        self.query_one("#inputs", Static).update(f"result  {result}\n{rendered}".rstrip())

        series = line.get("series") or {}
        table = self.query_one("#series", DataTable)
        table.clear(columns=True)
        names = [name for name, values in series.items() if isinstance(values, list)]
        spark_source: list[float] = []
        spark = self.query_one("#spark", Sparkline)
        if names:
            table.display = True
            spark.display = True
            table.add_column("i", key="i")
            for name in names:
                table.add_column(name, key=name)
            length = max(len(series[name]) for name in names)
            for i in range(length):
                row: list[str] = [str(i)]
                for name in names:
                    values = series[name]
                    cell = values[i] if i < len(values) else None
                    row.append("" if cell is None else str(cell))
                    if name == names[0] and isinstance(cell, (int, float)):
                        spark_source.append(float(cell))
                table.add_row(*row)
            spark.data = spark_source or [0.0]
        else:
            table.display = False
            spark.display = False


def launch(args: argparse.Namespace) -> int:
    """Open the session from the bare `alphaengine` command."""
    from ..cli import ProjectError, _session, say

    session, url = _session(getattr(args, "url", None), getattr(args, "key", None))
    data: Any = None
    backtest_fn: Any = None
    loaded: str | None = None
    try:
        data, backtest_fn = resolve_data(args, session)
    except (ProjectError, ValueError) as exc:
        say(str(exc))
    if getattr(args, "project", None):
        loaded = str(args.project)
    elif getattr(args, "universe", None):
        loaded = f"universe:{args.universe}"
    elif getattr(args, "data", None):
        loaded = f"file:{args.data}"
    keyed = bool(getattr(args, "key", None) or os.environ.get("QUANTOS_API_KEY"))
    QuantOSApp(
        session=session,
        url=url,
        args=args,
        data=data,
        backtest_fn=backtest_fn,
        loaded=loaded,
        keyed=keyed,
    ).run()
    return 0
