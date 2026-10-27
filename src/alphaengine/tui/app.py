"""The quantOS session. A status line, the trace, and a prompt."""

from __future__ import annotations

import argparse
import os
import re
import time
from typing import Any

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import DataTable, Input, ListItem, ListView, RichLog, Static

from .._version import __version__
from ..cli import narration_to, resolve_data, thesis_id_of
from .dispatch import Desk, submit
from .mark import details_text, seal
from .typeface import SPIN, body_line, meta_line, spin_line, tool_line, user_line

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_TOOL_LINE = re.compile(r"^(\d{2})\s+(\S+)(?:\s+(.*))?$")

_URL_AFTER = {
    "AZURE_OPENAI_API_KEY": "AZURE_OPENAI_ENDPOINT",
    "ALPHAENGINE_API_KEY": "ALPHAENGINE_BASE_URL",
}
_CSS = """
Screen { background: #0b0d10; }
#status {
    dock: top;
    height: 1;
    background: #12151a;
    color: #8b949e;
    content-align: center middle;
    text-align: center;
}
#column {
    height: 1fr;
    width: 100%;
    background: #0b0d10;
    padding: 0 2;
}
#empty {
    width: 100%;
    height: 1fr;
    content-align: center middle;
    text-align: center;
    color: #6b7380;
}
#transcript {
    width: 100%;
    height: 1fr;
    background: #0b0d10;
    padding: 1 0 0 0;
    display: none;
}
#inspector {
    width: 100%;
    height: auto;
    max-height: 16;
    background: #0b0d10;
    padding: 0 0 1 0;
    display: none;
}
#lines { display: none; height: 1; }
#formula { height: auto; max-height: 4; color: #9ecbff; }
#inputs { height: auto; max-height: 3; color: #8b949e; }
#series { height: 8; }
#spinner {
    height: 1;
    background: #0b0d10;
    color: #9ecbff;
    padding: 0 0 0 0;
    display: none;
}
#prompt-row {
    dock: bottom;
    height: 7;
    background: #0b0d10;
    padding: 1 2;
}
#prompt {
    width: 100%;
    height: 5;
    padding: 1 2;
    border: tall #3d7ea6;
    background: #12151a;
    color: #e6edf3;
}
Details { align: center middle; background: #0b0d1080; }
#card {
    width: 64;
    height: auto;
    background: #10141a;
    border: tall #3d7ea6;
    padding: 1 2;
}
#card-title { color: #9ecbff; text-align: center; height: 1; }
#card-body { height: auto; color: #c5cdd6; }
#card-foot { color: #6b7380; text-align: center; height: 1; }
"""


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


def _series_shape(data: Any) -> tuple[int, int] | None:
    """Names and observations on the machine. None when nothing countable is loaded."""
    if isinstance(data, dict):
        series = [values for values in data.values() if isinstance(values, list)]
        if series:
            return len(series), max(len(values) for values in series)
    if isinstance(data, list) and data and isinstance(data[0], (int, float)):
        return 1, len(data)
    return None


def _books_from(rows: Any) -> list[dict[str, Any]]:
    books: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return books
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or row.get("id") or "")
        if not name:
            continue
        raw_definition = row.get("definition")
        definition = raw_definition if isinstance(raw_definition, dict) else {}
        symbols = definition.get("symbols") or []
        raw_obs = definition.get("n_obs")
        obs = int(raw_obs) if isinstance(raw_obs, (int, float)) and not isinstance(raw_obs, bool) else 0
        books.append(
            {
                "name": name,
                "names": len(symbols) if isinstance(symbols, list) else 0,
                "obs": obs,
            }
        )
    return books


def _only_stored(rows: Any) -> str | None:
    """The account's single stored book. Several books is a choice, not a guess."""
    if not isinstance(rows, list):
        return None
    ready: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "")
        raw_definition = row.get("definition")
        definition = raw_definition if isinstance(raw_definition, dict) else {}
        if name and definition.get("cache_id"):
            ready.append(name)
    return ready[0] if len(ready) == 1 else None


def _styled(plain: str) -> Text:
    stripped = plain.strip()
    if stripped.startswith(">"):
        return user_line(stripped.lstrip("> ").strip())
    match = _TOOL_LINE.match(stripped)
    if match:
        return tool_line(match.group(1), match.group(2), match.group(3) or "")
    lower = stripped.lower()
    if lower.startswith("type yes") or stripped.startswith("report ") or stripped.startswith("workflow "):
        return meta_line(stripped)
    return body_line(stripped)


def _fit(text: str, width: int = 22) -> str:
    text = " ".join(str(text).split())
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


class Details(ModalScreen[None]):
    """Housekeeping. Esc, enter, or q closes it."""

    BINDINGS = [("escape", "close", "Close")]

    def compose(self) -> ComposeResult:
        with Vertical(id="card"):
            yield Static("details", id="card-title")
            yield Static(details_text(), id="card-body")
            yield Static("esc closes", id="card-foot")

    def action_close(self) -> None:
        self.dismiss()

    def on_key(self, event: events.Key) -> None:
        if event.key in {"enter", "escape", "q"}:
            self.dismiss()
            event.stop()


class QuantOSApp(App[None]):
    """Status, then the trace, then the prompt. The column sits in the middle."""

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
        self._books: list[dict[str, Any]] = []
        self._lines: list[dict[str, Any]] = []
        self.busy = False
        self.activity = ""
        self._started = 0.0
        self._stage_timer: Any = None
        self.status_text = ""
        self.empty_text = ""
        self.formula_text = ""
        self.transcript: list[str] = []
        self.turns = 0
        self.loading = False
        self.load_frac = 0.0
        self.load_label = ""
        self.spin_label = ""
        self._spin_i = 0

    def compose(self) -> ComposeResult:
        yield Static("", id="status")
        with Vertical(id="column"):
            yield Static("", id="empty")
            yield RichLog(id="transcript", highlight=False, markup=False, wrap=True)
            yield Static("", id="spinner")
            with Vertical(id="inspector"):
                yield ListView(id="lines")
                yield Static("", id="formula")
                yield Static("", id="inputs")
                yield DataTable(id="series", zebra_stripes=True)
        with Horizontal(id="prompt-row"):
            yield Input(placeholder="Ask", id="prompt")

    def on_mount(self) -> None:
        self.query_one("#series", DataTable).cursor_type = "row"
        self.query_one(Input).focus()
        self._paint_empty()
        self._paint_status()
        self._stage_timer = self.set_interval(0.16, self._tick)
        self._sync_account()

    def on_unmount(self) -> None:
        timer = getattr(self, "_stage_timer", None)
        if timer is not None:
            timer.stop()

    def _refresh_rail(self) -> None:
        self._paint_status()
        self._paint_empty()
        prompt = self.query_one(Input)
        if not (self.desk.pending_secret or self.desk.pending_provider or self.desk.pending_url):
            prompt.placeholder = "Ask"

    def _project_trials(self, proj: Any) -> int:
        """Ledger size is read off the UI tick. Cache it so the spinner is not blocked on disk."""
        now = time.monotonic()
        cached = getattr(self, "_trials_cache", None)
        if isinstance(cached, tuple) and cached[0] == proj.root and now - float(cached[1]) < 2.0:
            return int(cached[2])
        from ..ledger import cumulative_trials_for

        _family, total = cumulative_trials_for(proj)
        self._trials_cache = (proj.root, now, int(total))
        return int(total)

    def _paint_status(self, *, working: bool = False) -> None:
        from ..model import available_models

        keyed = self.desk.keyed
        models = available_models()
        parts: list[tuple[str, str]] = [
            ("quantOS", "bold #9ecbff"),
            (__version__, "#6b7380"),
            ("signed in" if keyed else "not signed in", "#c5cdd6" if keyed else "#e6b450"),
        ]
        detail = self._account_detail()
        if detail:
            parts.append((detail, "#e6edf3"))
        from ..project import find_project

        proj = find_project()
        if proj is not None:
            parts.append((f"project:{proj.name} · {self._project_trials(proj)} trials", "#7ec8c4"))
        if models:
            parts.append((models[0][0], "#8b949e"))
        if self.loading and not self.busy:
            filled = max(0, min(12, int(round(self.load_frac * 12))))
            bar_s = "█" * filled + "░" * (12 - filled)
            label = _fit(self.load_label or "account", 18)
            parts.append((f"{bar_s}  {label}", "#7ec8c4"))
        elif working or self.busy:
            elapsed = time.monotonic() - self._started
            label = _fit(self.activity or "working", 42)
            parts.append((f"{label}  {elapsed:.1f}s", "#9ecbff"))
        bar = Text(justify="center")
        for index, (text, style) in enumerate(parts):
            if index:
                bar.append("    ")
            bar.append(text, style=style)
        self.status_text = bar.plain
        status = self.query("#status")
        node = status.first() if status else None
        if isinstance(node, Static):
            node.update(bar)

    def _paint_empty(self) -> None:
        empty = self.query("#empty")
        node = empty.first() if empty else None
        if not isinstance(node, Static):
            return
        show = self.turns == 0 and not self.transcript
        node.display = show
        log = self.query("#transcript")
        transcript = log.first() if log else None
        if isinstance(transcript, RichLog):
            transcript.display = not show
        if not show:
            return
        mark = seal(time.monotonic())
        self.empty_text = mark.plain
        node.update(mark)

    def _account_detail(self) -> str:
        """The book the portal handed over, and the shape now on this machine."""
        loaded = str(self.desk.loaded or "")
        name = loaded.split(":", 1)[1] if loaded.startswith("universe:") else ""
        book = next((row for row in self._books if row["name"] == name), None)
        shape = _series_shape(self.desk.data)
        n = shape[0] if shape else (int(book["names"]) if book else 0)
        obs = shape[1] if shape else (int(book["obs"]) if book else 0)
        if loaded.startswith("universe:"):
            bits = [loaded]
            if n:
                bits.append("1 name" if n == 1 else f"{n} names")
            if obs:
                bits.append(f"{obs} obs")
            return " · ".join(bits)
        bits = []
        if shape:
            bits.append("1 name" if n == 1 else f"{n} names")
            bits.append(f"{obs} obs")
        elif self.desk.data is not None:
            bits.append("data loaded")
        if self._books:
            shown = []
            for row in self._books[:4]:
                piece = str(row["name"])
                count = int(row["names"])
                if count:
                    piece += " · 1 name" if count == 1 else f" · {count} names"
                shown.append(piece)
            bits.append("  ".join(shown))
        return " · ".join(bits)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text or self.busy:
            return
        if self.desk.pending_secret:
            self._accept_secret(text)
            return
        if self.desk.pending_provider:
            self._accept_provider(text)
            return
        if self.desk.pending_url:
            self._accept_url(text)
            return
        if text.lower() in {"/details", "details"}:
            self.push_screen(Details())
            return
        self.turns += 1
        self.busy = True
        self.activity = text
        self._started = time.monotonic()
        prompt = self.query_one(Input)
        prompt.disabled = True
        self._reveal_trace()
        self._paint_status(working=True)
        self.spin_label = "working"
        self._paint_spinner()
        self.query_one("#transcript", RichLog).write(user_line(text))

        def go() -> None:
            self._run_line(text)

        self.run_worker(go, thread=True, exclusive=True)

    def _accept_secret(self, value: str) -> None:
        env = self.desk.pending_secret
        self.desk.pending_secret = None
        prompt = self.query_one(Input)
        prompt.password = False
        prompt.placeholder = "Ask"
        log = self.query_one("#transcript", RichLog)
        self._reveal_trace()
        if env is None or not value:
            log.write("nothing entered.")
            return
        if env == "MODEL_KEY":
            self._take_model_key(value, log, prompt)
            return
        self._store_credential(env, value, log, prompt)

    def _take_model_key(self, value: str, log: RichLog, prompt: Input) -> None:
        from .dispatch import env_for_model_key

        found = env_for_model_key(value)
        if found is None:
            self.desk.held_key = value
            self.desk.pending_provider = True
            prompt.password = False
            prompt.placeholder = "anthropic, openai, gemini, groq, openrouter, azure, gateway"
            log.write("Which provider is this key for?")
            return
        self._store_credential(found, value, log, prompt)

    def _accept_provider(self, text: str) -> None:
        from ..cli import _KEY_PROMPTS

        which = text.strip().lower()
        prompt = self.query_one(Input)
        log = self.query_one("#transcript", RichLog)
        if which not in _KEY_PROMPTS or which == "quantos":
            log.write("Try anthropic, openai, gemini, groq, openrouter, azure, or gateway.")
            return
        value = self.desk.held_key or ""
        self.desk.held_key = None
        self.desk.pending_provider = False
        prompt.placeholder = "Ask"
        if not value:
            log.write("nothing entered.")
            return
        self._store_credential(_KEY_PROMPTS[which][0], value, log, prompt)

    def _store_credential(self, env: str, value: str, log: RichLog, prompt: Input) -> None:
        from ..model import _clean_key, provider_for_env

        value = _clean_key(value)
        if not value:
            log.write("nothing entered.")
            return
        os.environ[env] = value
        prov = provider_for_env(env)
        if prov:
            os.environ["ALPHAENGINE_PROVIDER"] = prov
            os.environ.pop("ALPHAENGINE_MODEL", None)
        if self.desk.think_from_key:
            self.desk.think = None
            self.desk.think_from_key = False
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
            if prov:
                save_key("ALPHAENGINE_PROVIDER", prov)
            log.write(f"{env} accepted.  stored at {path}")
        except (OSError, ValueError) as exc:
            log.write(f"{env} accepted for this process.  ({exc})")
        follow = _URL_AFTER.get(env)
        if follow and not os.environ.get(follow):
            self.desk.pending_url = follow
            prompt.placeholder = f"paste {follow}"
            log.write(f"Now paste {follow}. It is not hidden.")
        elif env != "QUANTOS_API_KEY":
            self._say_model_ready(log)
        if env == "QUANTOS_API_KEY":
            self._sync_account()
        else:
            self._refresh_rail()

    def _accept_url(self, value: str) -> None:
        env = self.desk.pending_url
        self.desk.pending_url = None
        prompt = self.query_one(Input)
        prompt.placeholder = "Ask"
        log = self.query_one("#transcript", RichLog)
        self._reveal_trace()
        if env is None or not value:
            log.write("nothing entered.")
            return
        os.environ[env] = value
        try:
            from ..auth import save_key

            save_key(env, value)
        except (OSError, ValueError):
            pass
        log.write(f"{env} set.")
        self._say_model_ready(log)
        self._refresh_rail()

    def _say_model_ready(self, log: RichLog) -> None:
        from ..model import _SDK, available_models, keys_without_sdk

        stranded = keys_without_sdk()
        ready = available_models()
        if stranded:
            installs = " or ".join(f"pip install {_SDK[label]}" for label in stranded)
            log.write(f"Key stored. The SDK is not installed yet: {installs}")
            return
        if ready:
            label, model = ready[0]
            log.write(f"Model ready: {label}/{model}.")

    def _reveal_trace(self) -> None:
        empty = self.query("#empty")
        node = empty.first() if empty else None
        if isinstance(node, Static):
            node.display = False
        log = self.query("#transcript")
        transcript = log.first() if log else None
        if isinstance(transcript, RichLog):
            transcript.display = True

    def _sync_account(self) -> None:
        """Paint first, then fetch the account off the UI thread."""
        if not self.desk.keyed:
            self.loading = False
            self._refresh_rail()
            return
        if self.loading:
            return
        self.loading = True
        self.load_frac = 0.08
        self.load_label = "account"
        self._paint_status()
        self._paint_spinner()
        self.run_worker(self._load_account, thread=True, group="boot", exclusive=False)

    def _load_account(self) -> None:
        from ..cli import _narrow_to_universe, is_auth_error
        from ..client.session import request_timeout

        books: list[dict[str, Any]] = []
        data: Any = None
        loaded: str | None = None
        thesis: dict[str, Any] | None = None
        notes: list[str] = []
        auth_failed = False
        session = self.desk.session
        try:
            with request_timeout(10):
                try:
                    rows = session.universes()
                except Exception as exc:  # noqa: BLE001 — offline or unsigned still boots
                    if is_auth_error(exc):
                        auth_failed = True
                    rows = []
                books = _books_from(rows)
                self.call_from_thread(self._set_load, 0.45, "account")
                if not auth_failed and self.desk.data is None and self.desk.keyed:
                    wanted = _only_stored(rows)
                    if wanted:
                        self.call_from_thread(self._set_load, 0.72, wanted)

                        def sink(_chunk: str) -> None:
                            return

                        with narration_to(sink):
                            try:
                                data = _narrow_to_universe(session, wanted, None, rows=rows)
                                loaded = f"universe:{wanted}"
                            except Exception as exc:  # noqa: BLE001
                                notes.append(str(exc))
                if (
                    not auth_failed
                    and self.desk.keyed
                    and self.desk.thesis is None
                    and not self.desk.thesis_declined
                ):
                    self.call_from_thread(self._set_load, 0.9, "account")
                    try:
                        theses = [r for r in session.theses() if isinstance(r, dict) and thesis_id_of(r)]
                    except Exception:  # noqa: BLE001 — offline or unsigned still boots
                        theses = []
                    if len(theses) == 1:
                        thesis = theses[0]
        except Exception as exc:  # noqa: BLE001 — the session still opens
            notes.append(str(exc))
        finally:
            self.call_from_thread(self._apply_account, books, data, loaded, thesis, notes, auth_failed)

    def _set_load(self, frac: float, label: str) -> None:
        self.load_frac = frac
        self.load_label = label
        self._paint_status()

    def _apply_account(
        self,
        books: list[dict[str, Any]],
        data: Any,
        loaded: str | None,
        thesis: dict[str, Any] | None,
        notes: list[str],
        auth_failed: bool = False,
    ) -> None:
        if auth_failed:
            from ..auth import remove_key

            self.desk.keyed = False
            os.environ.pop("QUANTOS_API_KEY", None)
            remove_key("QUANTOS_API_KEY")
        self._books = books
        if data is not None and self.desk.data is None:
            self.desk.data = data
            if loaded:
                self.desk.loaded = loaded
        if thesis is not None and self.desk.thesis is None and not self.desk.thesis_declined:
            self.desk.thesis = thesis
        self.loading = False
        self.load_label = ""
        kept = [_plain(note).strip() for note in notes if _plain(note).strip()]
        if kept:
            self._reveal_trace()
            log = self.query("#transcript")
            node = log.first() if log else None
            for plain in kept:
                self.transcript.append(plain)
                if isinstance(node, RichLog):
                    node.write(_styled(plain))
        self._refresh_rail()
        self._paint_spinner()

    def _tick(self) -> None:
        if self.loading:
            if self.load_frac < 0.82:
                self.load_frac = min(0.82, self.load_frac + 0.015)
            self._paint_status()
            self._paint_spinner()
            if self.turns == 0 and not self.transcript:
                self._paint_empty()
            return
        if self.busy or self.spin_label:
            self._paint_spinner()
            if self.busy:
                self._paint_status(working=True)
            return
        self._paint_spinner()
        if self.turns == 0 and not self.transcript:
            self._paint_empty()

    def _paint_working(self) -> None:
        if self.busy:
            self._paint_status(working=True)

    def _paint_spinner(self) -> None:
        nodes = self.query("#spinner")
        node = nodes.first() if nodes else None
        if not isinstance(node, Static):
            return
        show = bool(self.loading or self.busy or self.spin_label)
        node.display = show
        if not show:
            node.update("")
            return
        self._spin_i = (self._spin_i + 1) % len(SPIN)
        if self.busy and self.spin_label:
            label = self.spin_label
        elif self.loading and not self.busy:
            label = self.load_label or "account"
        elif self.spin_label:
            label = self.spin_label
        else:
            label = "working"
        node.update(spin_line(SPIN[self._spin_i], label))

    def _live(self, chunk: str) -> None:
        # A working label updates the status line and does not join the trace.
        # The step lines are the trace; a second copy of the op name is noise.
        if chunk.startswith("\x1e"):
            label = _plain(chunk[1:]).strip()
            if label:
                self.activity = label
                self.spin_label = label
            if self.busy:
                self._paint_working()
            self._paint_spinner()
            return
        if chunk.startswith("\x1f"):
            self.spin_label = _plain(chunk[1:]).strip()
            self._paint_spinner()
            return
        plain = _plain(chunk).strip()
        if not plain:
            return
        self.activity = plain
        self.transcript.append(plain)
        self._reveal_trace()
        log = self.query("#transcript")
        node = log.first() if log else None
        if isinstance(node, RichLog):
            node.write(_styled(plain))
        if self.busy:
            self._paint_working()

    def _run_line(self, text: str) -> None:
        quitting = False

        def sink(chunk: str) -> None:
            self.call_from_thread(self._live, chunk)

        try:
            with narration_to(sink):
                quitting = submit(self.desk, text)
        except Exception as exc:  # noqa: BLE001 — the session stays up
            self.call_from_thread(self._live, str(exc))
        self.call_from_thread(self._finish, quitting)

    def _finish(self, quitting: bool) -> None:
        self.busy = False
        self.activity = ""
        self.spin_label = ""
        self._paint_spinner()
        prompt = self.query_one(Input)
        prompt.disabled = False
        if self.desk.pending_secret:
            prompt.password = True
            prompt.placeholder = "paste the key (hidden)"
        elif self.desk.pending_provider:
            prompt.password = False
            prompt.placeholder = "anthropic, openai, gemini, groq, openrouter, azure, gateway"
        elif self.desk.pending_url:
            prompt.password = False
            prompt.placeholder = f"paste {self.desk.pending_url}"
        else:
            prompt.password = False
            self._refresh_rail()
        if self.desk.account_dirty:
            self.desk.account_dirty = False
            self._sync_account()
        else:
            self._refresh_rail()
        self._load_lines()
        if not quitting:
            prompt.focus()
        if quitting:
            self.exit()

    def _load_lines(self) -> None:
        matrix = getattr(self.desk, "matrix", None)
        if matrix:
            self._show_matrix(matrix)
            return
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
        view.display = False
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

    def _show_matrix(self, result: dict[str, Any]) -> None:
        """The full name-by-name matrix. It does not fit a figure, so it stays here."""
        inspector = self.query_one("#inspector", Vertical)
        inspector.display = True
        kind = str(result.get("kind") or "matrix")
        names = list(result.get("names") or [])
        self.formula_text = f"{kind} across {len(names)} names"
        self.query_one("#formula", Static).update(self.formula_text)
        n_obs = result.get("n_obs")
        detail = f"{len(names)} names"
        if n_obs:
            detail += f"   {n_obs} observations"
        self.query_one("#inputs", Static).update(detail)
        self.query_one("#lines", ListView).display = False
        table = self.query_one("#series", DataTable)
        table.display = True
        table.clear(columns=True)
        if kind == "portfolio":
            self.formula_text = f"{result.get('method') or 'hrp'} on a Ledoit-Wolf covariance"
            self.query_one("#formula", Static).update(self.formula_text)
            for column in ("name", "weight"):
                table.add_column(column)
            for row in (result.get("rows") or [])[:40]:
                table.add_row(str(row.get("name") or ""), str(row.get("weight")))
            return
        if kind == "cointegration":
            for column in ("a", "b", "p", "half-life"):
                table.add_column(column)
            for row in (result.get("rows") or [])[:40]:
                if row.get("p") is None and not row.get("cointegrated"):
                    continue
                table.add_row(
                    str(row.get("a") or ""),
                    str(row.get("b") or ""),
                    "" if row.get("p") is None else str(row.get("p")),
                    "" if row.get("half_life") is None else str(row.get("half_life")),
                )
            return
        table.add_column("")
        for name in names:
            table.add_column(str(name))
        values = result.get("values") or []
        for i, name in enumerate(names):
            cells = values[i] if i < len(values) else []
            table.add_row(str(name), *["" if cell is None else str(cell) for cell in cells])

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
        if names:
            table.display = True
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
                table.add_row(*row)
        else:
            table.display = False


def launch(args: argparse.Namespace) -> int:
    """Open the session from the bare `alphaengine` command."""
    from ..cli import ProjectError, _session, say

    session, url = _session(
        getattr(args, "url", None),
        getattr(args, "key", None),
        share_series=bool(getattr(args, "share_series", False)),
    )
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
