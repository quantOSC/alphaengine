"""What this session has already done, in a form a model may quote.

A generic question should not start a new workflow when the desk already
holds the answer. The notes are short on purpose: counts, a few pairs, a
few weights. A return series never goes into a note, and an answer that
cites a number the notes do not contain is refused.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["remember", "loaded_note", "result_note", "consider"]

_MAX_NOTES = 12

_PROMPT = """You are answering from a research session that is already in progress.

The question:
{question}

What this session has already recorded. These are the only facts you may use:
{notes}

Reply with ONE JSON object and nothing else.
{"answer": "two or three sentences"} when the notes already contain the answer.
{"workflow": true} when the question asks for a calculation the notes do not contain.

Quote only numbers that appear in the notes. If the notes do not answer the
question, reply {"workflow": true}.
"""


def remember(notes: list[str], text: str) -> list[str]:
    """Append one note. The oldest falls off so the window stays the recent desk."""
    clean = " ".join(str(text).split())
    if not clean:
        return list(notes)
    return [*notes, clean][-_MAX_NOTES:]


def loaded_note(label: str, data: Any) -> str:
    """The shape of what was loaded, never the series."""
    from .core.relations import as_closes

    closes = as_closes(data) if isinstance(data, dict) else {}
    if len(closes) >= 1:
        longest = max(len(values) for values in closes.values())
        return f"loaded {label}: {len(closes)} names, about {longest} observations"
    if isinstance(data, (list, tuple)):
        return f"loaded {label}: {len(data)} returns"
    return f"loaded {label}"


def result_note(result: dict[str, Any]) -> str:
    """One line a later question can quote."""
    kind = str(result.get("kind") or "")
    if kind in ("correlation", "covariance"):
        names = result.get("names") or []
        top = ""
        pairs = result.get("pairs") or []
        if pairs:
            pair = pairs[0]
            top = f" Strongest {pair.get('a')} {pair.get('b')} {pair.get('value')}."
        return f"{kind}: {len(names)} names, {result.get('n_obs')} observations.{top}"
    if kind == "cointegration":
        return (
            f"cointegration: {result.get('n_cointegrated')} of {result.get('n_pairs')} pairs"
            f" across {len(result.get('names') or [])} names."
        )
    if kind == "portfolio":
        weights = result.get("weights") or {}
        largest = ""
        if weights:
            name, weight = max(weights.items(), key=lambda item: float(item[1]))
            largest = f" Largest weight {weight} on {name}."
        return (
            f"portfolio {result.get('method')}: {result.get('n_assets')} names,"
            f" weights sum to {result.get('weight_sum')}.{largest}"
        )
    return kind


def consider(question: str, notes: list[str], think: Any) -> tuple[str, str]:
    """("answer", text), ("refuse", reason), or ("workflow", "").

    "workflow" means the notes do not answer it, so the caller should run one.
    A missing model is "workflow" too: the catalogue path already says what is missing.
    """
    if not notes or think is None:
        return "workflow", ""
    brief = "\n".join(f"- {note}" for note in notes)
    raw = think(_PROMPT.replace("{question}", question).replace("{notes}", brief))
    parsed = _parse(raw)
    if not isinstance(parsed, dict):
        return "workflow", ""
    if parsed.get("workflow") is True:
        return "workflow", ""
    answer = str(parsed.get("answer") or "").strip()
    if not answer:
        return "workflow", ""
    bad = _uncited(answer, brief)
    if bad:
        return "refuse", f"The answer cited {bad}, which this session has not recorded."
    return "answer", answer


def _parse(raw: str) -> dict[str, Any] | None:
    text = str(raw or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _uncited(answer: str, notes: str) -> str | None:
    from .agent.answer import _numbers_in, _quotes

    recorded = _numbers_in(notes)
    for cited, decimals in _numbers_in(answer):
        if any(_quotes(cited, decimals, value) for value, _decimals in recorded):
            continue
        return str(cited)
    return None
