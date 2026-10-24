"""The simulator. A sentence becomes a stack of tool calls, and the stack is the product.

The model chooses the next tool. The tool is deterministic. The trace is what
gets shown, and what a report file is made of when the user accepts one.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

Invoke = Callable[[str, dict[str, Any]], Any]
Think = Callable[[str], str]

MAX_CALLS = 8

TOOLS: tuple[tuple[str, str], ...] = (
    ("profile", "what is wrong with the loaded file"),
    ("align", "shared dates across stocks and macro; gaps counted, never filled"),
    ("regime", "two-state Hamilton filter on one named series"),
    ("price", "Black-Scholes price and Greeks for one contract"),
    ("reprice", "full reprice of that contract under a spot and vol shock"),
    ("correlation", "correlation of the loaded names"),
    ("covariance", "covariance of the loaded names"),
    ("cointegration", "cointegrated pairs, FDR survivors"),
    ("portfolio", "weights; args.method is hrp, risk_parity, inverse_vol, or var_sorted"),
    ("overlap", "one name against the equal-weight of the others; args.symbol"),
    ("performance", "Sharpe, drawdown, and the cash rate on the constructed book"),
    ("var", "value at risk and expected shortfall on the constructed book"),
    ("scenarios", "stress the constructed book across regime, inflation, rate, and volatility shocks"),
    (
        "monte_carlo",
        "Monte Carlo wealth and Sharpe simulation on the book; args.dgp, args.n_paths, args.horizon",
    ),
    ("process", "fit ou, gbm, jump, or garch and stress it; args.dgp"),
    ("workflow", "a portal workflow by name, when the account offers it"),
    ("bakeoff", "generators propose, an adversary attacks, the referee names the top-ranked config"),
    ("done", "stop. args.answer may cite only numbers the tools returned"),
)

_PROMPT = """You are the simulator inside alphaengine 1.0.

The user asked a question. You answer it by calling tools, as many as the
scenario needs, one per reply. The math is already written. You decide the
order. You do not invent a figure.
When constructing a portfolio or analyzing a scenario or simulation, build the
portfolio (`portfolio`), run `scenarios` and `monte_carlo` (and `var` or `bakeoff`
when relevant), and then finish with `done`.

Question:
@@QUESTION@@

What is already loaded and already known:
@@CONTEXT@@

Tools:
@@TOOLS@@

Calls already made this turn:
@@CALLS@@

Reply with ONE JSON object and nothing else.
{"tool": "<name>", "args": {}}
{"tool": "done", "args": {"answer": "two or three sentences citing only tool numbers"}}
"""


@dataclass
class Call:
    tool: str
    args: dict[str, Any]
    figures: dict[str, Any] = field(default_factory=dict)
    note: str = ""
    error: str | None = None

    def line(self) -> str:
        if self.error:
            return f"{self.tool}  refused  {self.error}"
        return f"{self.tool}  {self.note}".rstrip()


@dataclass
class Trace:
    """The run, as a stack. This is the artifact. The answer is the last line of it."""

    question: str
    calls: list[Call] = field(default_factory=list)
    answer: str = ""
    refused: bool = False
    # Local chart inputs. They stay off the model prompt and off the wire.
    panel: dict[str, list[float]] | None = None
    matrix: dict[str, Any] | None = None
    boards: list[dict[str, Any]] = field(default_factory=list)
    bakeoff: dict[str, Any] | None = None
    monte: dict[str, Any] | None = None
    study: dict[str, Any] | None = None
    findings: str = ""
    reading: str = ""

    def render(self) -> str:
        lines = ["alphaengine 1.0", f"question  {self.question}", ""]
        if not self.calls:
            lines.append("no tools ran")
        for i, call in enumerate(self.calls, start=1):
            lines.append(f"{i:02d}  {call.line()}")
            for key, value in _public(call.figures).items():
                lines.append(f"    {key}  {value}")
        lines.append("")
        if self.answer:
            lines.append(self.answer)
        return "\n".join(lines)

    def save(self, directory: Path | None = None) -> Path:
        """Write the text trace and, when matplotlib is installed, the charted PDF.

        The PDF is the report. It goes to the machine's Downloads folder.
        The text file stays beside it so a machine without matplotlib still
        has the stack.
        """
        folder = directory or downloads_dir()
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        digest = hashlib.sha256(f"{stamp}|{self.question}".encode()).hexdigest()[:6]
        stem = f"report-{digest}"
        text_path = folder / f"{stem}.txt"
        text_path.write_text(self.render() + "\n", encoding="utf-8")
        pdf_path = folder / f"{stem}.pdf"
        try:
            from .chart import write_report

            write_report(self, pdf_path)
        except ImportError:
            return text_path
        if pdf_path.is_file() and pdf_path.stat().st_size > 0:
            return pdf_path
        return text_path


# The Windows known-folder id for Downloads. A redirected folder (OneDrive,
# a moved library) lives here, not at a guessed path under the profile.
_DOWNLOADS_GUID = "{374DE290-123F-4565-9164-39C4925E467B}"


def downloads_dir() -> Path:
    """The machine's Downloads folder. Reports land here, not under AppData."""
    found = _known_downloads()
    if found is not None:
        return found
    return Path.home() / "Downloads"


def _known_downloads() -> Path | None:
    if os.name == "nt":
        return _windows_downloads()
    return _xdg_downloads()


def _windows_downloads() -> Path | None:
    """The Windows Downloads folder. Looked up by attribute so Linux type checks."""
    try:
        import winreg
    except ImportError:
        return None
    open_key = getattr(winreg, "OpenKey", None)
    query = getattr(winreg, "QueryValueEx", None)
    hive = getattr(winreg, "HKEY_CURRENT_USER", None)
    if open_key is None or query is None or hive is None:
        return None
    try:
        with open_key(
            hive,
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders",
        ) as key:
            raw, _ = query(key, _DOWNLOADS_GUID)
    except OSError:
        return None
    if not isinstance(raw, str) or not raw.strip():
        return None
    return Path(os.path.expandvars(raw))


def _xdg_downloads() -> Path | None:
    config = os.environ.get("XDG_CONFIG_HOME")
    path = (Path(config) if config else Path.home() / ".config") / "user-dirs.dirs"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    home = str(Path.home())
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("XDG_DOWNLOAD_DIR"):
            continue
        raw = stripped.partition("=")[2].strip().strip('"').strip("'")
        raw = raw.replace("${HOME}", home).replace("$HOME", home)
        expanded = os.path.expandvars(raw).strip()
        if expanded:
            return Path(expanded)
    return None


def run_tools(
    question: str,
    think: Think,
    invoke: Invoke,
    *,
    context: str = "",
    max_calls: int = MAX_CALLS,
    on_call: Callable[[Call], None] | None = None,
) -> Trace:
    """Call tools until the model says done, or the stack is full."""
    trace = Trace(question=question)
    known: dict[str, float] = {}
    for _ in range(max_calls):
        raw = think(_fill(question, context, trace))
        parsed = parse_call(raw)
        if parsed is None:
            trace.answer = "The model did not return a tool call."
            trace.refused = True
            return trace
        name = str(parsed.get("tool") or "")
        raw_args = parsed.get("args")
        args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
        if name == "done":
            answer = str(args.get("answer") or parsed.get("answer") or "").strip()
            trace.answer = _guard_answer(answer, known, question)
            trace.refused = trace.answer.startswith("Refused")
            return trace
        if name not in {tool for tool, _ in TOOLS}:
            call = Call(tool=name or "?", args=args, error="that tool is not in the stack")
            trace.calls.append(call)
            if on_call:
                on_call(call)
            continue
        try:
            got: Any = invoke(name, args)
        except Exception as exc:  # noqa: BLE001 — the stack records the failure
            got = {"error": str(exc)}
        figures: dict[str, Any] = got if isinstance(got, dict) else {"result": got}
        error = str(figures["error"]) if figures.get("error") else None
        note = str(figures.get("note") or _note(figures))
        call = Call(tool=name, args=args, figures=_public(figures), note=note, error=error)
        trace.calls.append(call)
        known.update(_numbers(call.figures))
        if on_call:
            on_call(call)
    trace.answer = f"Stopped after {max_calls} tool calls. The stack above is the run."
    return trace


def parse_call(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "tool" not in obj:
        return None
    return obj


def _fill(question: str, context: str, trace: Trace) -> str:
    calls = "\n".join(call.line() for call in trace.calls) if trace.calls else "none yet"
    tools = "\n".join(f"- {name}: {purpose}" for name, purpose in TOOLS)
    return (
        _PROMPT.replace("@@QUESTION@@", question)
        .replace("@@CONTEXT@@", context or "nothing loaded yet")
        .replace("@@TOOLS@@", tools)
        .replace("@@CALLS@@", calls)
    )


def _public(figures: dict[str, Any]) -> dict[str, Any]:
    """Scalars and short lists. A path is not a figure."""
    out: dict[str, Any] = {}
    for key, value in figures.items():
        if str(key).startswith("_"):
            continue
        if isinstance(value, dict):
            continue
        if isinstance(value, (list, tuple)):
            if len(value) > 12:
                continue
            out[str(key)] = list(value)
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        out[str(key)] = value
    return out


def _note(figures: dict[str, Any]) -> str:
    bits = []
    for key, value in _public(figures).items():
        if key in {"method", "note"}:
            continue
        bits.append(f"{key} {value}")
        if len(bits) == 4:
            break
    return "  ".join(bits)


def _numbers(figures: dict[str, Any]) -> dict[str, float]:
    found: dict[str, float] = {}

    def walk(node: Any, prefix: str) -> None:
        if isinstance(node, bool):
            return
        if isinstance(node, (int, float)) and math.isfinite(float(node)):
            found[prefix or "n"] = float(node)
            return
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{prefix}.{key}" if prefix else str(key))

    walk(figures, "")
    return found


_NARRATE = """You are writing the portfolio construction and risk reading for a quantitative report.
Cite only numbers written below. Quote a decimal exactly as it appears.
Do not invent a weight, a Sharpe, a date, or a name count.
Write two to three paragraphs (eight to twelve sentences). Cover:
1) What the universe, regime, and any named tickers in the question show.
2) How the constructed model weights (hrp, risk_parity, var_sorted, or bakeoff top-ranked config)
   allocate capital and how their Sharpe, volatility, drawdown, and VaR compare.
3) What the scenario shocks and Monte Carlo simulation paths imply for the book's risk profile.

Question:
@@QUESTION@@

Tools and figures:
@@CALLS@@

Reply with the reading and nothing else.
"""


def narrate(trace: Trace, think: Think, extra: dict[str, float] | None = None) -> str:
    """One closing note. A number that is not in the stack is refused, and then dropped."""
    lines: list[str] = []
    known: dict[str, float] = {}
    for call in trace.calls:
        lines.append(call.line())
        for key, value in call.figures.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if isinstance(value, float) and not math.isfinite(value):
                continue
            lines.append(f"    {call.tool}.{key} {value}")
            known[f"{call.tool}.{key}"] = float(value)
            known[str(key)] = float(value)
    for key, value in (extra or {}).items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        lines.append(f"    {key} {value}")
        known[str(key)] = float(value)
        tail = str(key).rsplit(".", 1)[-1]
        known.setdefault(tail, float(value))
    if not lines:
        return ""
    prompt = _NARRATE.replace("@@QUESTION@@", trace.question).replace("@@CALLS@@", "\n".join(lines))
    try:
        raw = think(prompt)
    except Exception:  # noqa: BLE001 — a missing note does not discard the stack
        return ""
    text = str(raw or "").strip()
    if text.startswith("{"):
        parsed = parse_call(text)
        if isinstance(parsed, dict):
            text = str(parsed.get("answer") or parsed.get("args", {}).get("answer") or "").strip()
    guarded = _guard_answer(text, known, trace.question)
    if not guarded or guarded.startswith("Refused") or guarded.startswith("The model did not"):
        return ""
    return guarded


def _guard_answer(answer: str, known: dict[str, float], question: str) -> str:
    if not answer:
        return "The stack above is the run."
    from .answer import UncitedFigure, check_citations

    try:
        check_citations(answer, known, question)
    except UncitedFigure as exc:
        return f"Refused. {exc}"
    return answer
