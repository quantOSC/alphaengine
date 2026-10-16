"""The simulator. A sentence becomes a stack of tool calls, and the stack is the product.

The model chooses the next tool. The tool is deterministic. The trace is what
gets shown, and what a report file is made of when the user accepts one.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..events import data_dir

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
    ("portfolio", "weights; args.method is hrp or risk_parity"),
    ("overlap", "one name against the equal-weight of the others; args.symbol"),
    ("performance", "Sharpe, drawdown, and the cash rate on the loaded book"),
    ("var", "value at risk on the loaded book"),
    ("process", "fit ou, gbm, jump, or garch and stress it; args.dgp"),
    ("workflow", "a portal workflow by name, when the account offers it"),
    ("bakeoff", "generators propose, an adversary attacks, the referee names the survivor"),
    ("done", "stop. args.answer may cite only numbers the tools returned"),
)

_PROMPT = """You are the simulator inside alphaengine 1.0.

The user asked a question. You answer it by calling tools, as many as the
scenario needs, one per reply. The math is already written. You decide the
order. You do not invent a figure.

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

        The PDF is the report. The text file stays beside it so a machine
        without matplotlib still has the stack.
        """
        folder = directory or (data_dir() / "reports")
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        text_path = folder / f"trace-{stamp}.txt"
        text_path.write_text(self.render() + "\n", encoding="utf-8")
        pdf_path = folder / f"trace-{stamp}.pdf"
        try:
            from .chart import write_report

            write_report(self, pdf_path)
        except ImportError:
            return text_path
        if pdf_path.is_file() and pdf_path.stat().st_size > 0:
            return pdf_path
        return text_path


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


def _guard_answer(answer: str, known: dict[str, float], question: str) -> str:
    if not answer:
        return "The stack above is the run."
    from .answer import UncitedFigure, check_citations

    try:
        check_citations(answer, known, question)
    except UncitedFigure as exc:
        return f"Refused. {exc}"
    return answer
