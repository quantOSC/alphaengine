"""Bring your own model.

`AgentDriver` was named in three documents for weeks and existed in zero lines
of code. This is it, and the shape it was always described as having is the
shape that matters:

    IT TAKES A CALLABLE. There is no `api_key` parameter on anything here, no
    provider enum, no `import anthropic`. That is not minimalism — it is how
    "runs under the customer's own model key" is satisfied STRUCTURALLY rather
    than promised. A field we could put a key in is a field somebody eventually
    puts a key in, and then we are storing customer model credentials, which is
    more convenient and strictly worse. There is nowhere to put one.

    THE PACKAGE DEPENDS ON NO LLM SDK. `think` is `(prompt: str) -> str`. Wrap
    Anthropic, OpenAI, a local llama.cpp, a company gateway, or a function that
    returns a canned answer for a test. The package cannot tell and must not
    care.

── WHAT THE MODEL IS AND IS NOT ALLOWED TO DO ─────────────────────────────────

The model picks the NEXT STEP FROM `directive.permitted`. It does not author
steps, invent ops, set thresholds or decide when a run is finished. Those live
on the server, in the workflow, and the client is forbidden from knowing them —
"the loop must contain no workflow knowledge" is the standing rule this file has
to be careful not to break.

So `pick()` returns AN INDEX into a list the server produced. That return type
is the enforcement: an index cannot express an op the server did not offer. A
model that hallucinates `compute.make_me_money` produces an out-of-range integer
and gets refused, rather than producing a plausible dictionary that flows on.

── WHY A RUN DRIVEN THIS WAY IS MARKED NOT REPRODUCIBLE ───────────────────────

Two runs of the same exploratory workflow over the same data can take different
paths, because a model chose. That is the point of exploratory mode and it is
also a fact that must travel with the artifact — a study whose PATH was chosen
by a model is a different epistemic object from one whose path was fixed in
advance, and collapsing them would be exactly the kind of quiet overstatement
the trial-count work exists to prevent.

The server already carries the vocabulary (`Workflow.agency`, SCRIPTED vs
EXPLORATORY, with a `reproducible` property). This half just has to be honest
about which one ran.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from typing import Any, Protocol

__all__ = ["AgentDriver", "Think", "AgentRefusal"]

Figures = dict[str, Any]


class Think(Protocol):
    """Any callable that takes a prompt and returns text.

    Deliberately the weakest possible interface. Everything else in this module
    is built so that this is all a caller has to supply.

    THE `/` IS LOad-BEARING. Without it the protocol requires the parameter to
    be *named* `prompt`, so an ordinary `lambda p: ...` or a
    `Callable[[str], str]` fails to satisfy it — mypy caught exactly that. We do
    not care what the caller names their argument; positional-only says so.
    """

    def __call__(self, prompt: str, /) -> str: ...


class AgentRefusal(RuntimeError):
    """The model produced something that is not a choice among what was offered."""


# The prompt is a TEMPLATE OVER THE SERVER'S OWN WORDS. It contains no workflow
# knowledge — no op names, no ordering, no thresholds — because everything it
# describes is read out of the directive at call time. If this string ever
# starts saying "usually you should run the sweep first", the rule has been
# broken and the client has become a second, worse copy of the workflow.
_PROMPT = """\
You are choosing the next step in a quantitative research run.

The goal, in the researcher's words:
{goal}

What has happened so far:
{history}

What has already been measured:
{measured}

You may choose exactly ONE of these permitted steps. Do not choose a step the
history says has already finished.
{options}

Reply with ONLY a JSON object:
  {{"choice": <integer index>, "why": "<one short sentence>"}}

Choose the index of the step that best advances the goal. You may not invent a
step, and you may not choose anything not listed above.
"""

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "with",
        "from",
        "into",
        "over",
        "once",
        "your",
        "you",
        "are",
        "was",
        "were",
        "been",
        "have",
        "has",
        "had",
        "does",
        "did",
        "can",
        "what",
        "which",
        "where",
        "when",
        "who",
        "how",
        "and",
        "or",
        "not",
        "for",
        "its",
        "about",
        "just",
        "than",
        "then",
        "them",
        "they",
        "my",
        "our",
        "of",
        "to",
        "in",
        "on",
        "is",
        "it",
        "be",
        "as",
        "at",
        "by",
        "if",
        "so",
        "we",
    }
)


class AgentDriver:
    """Drives a run by asking a model which permitted step to take next.

    Usage is deliberately three lines:

        driver = AgentDriver(think=my_model, goal="validate this momentum idea")
        run = session.open("validate_study", data=data, backtest_fn=fn)
        driver.drive(run)
    """

    def __init__(
        self,
        think: Think,
        *,
        goal: str,
        on_thought: Callable[[str], None] | None = None,
        on_step: Callable[[str], None] | None = None,
        max_steps: int = 60,
        sink: Any | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        self.think = think
        self.goal = goal
        # A narrator, so the terminal can show WHY the model chose what it chose.
        # A loop you cannot watch is a loop you cannot trust, and that applies
        # doubly when something non-deterministic is making the choices.
        self.on_thought = on_thought or (lambda _s: None)
        # A second narrator for the step itself, so a watched run can show the
        # op while it is still running and the duration once it returns.
        self.on_step = on_step or (lambda _s: None)
        self.max_steps = max_steps
        self.history: list[str] = []
        self.sink = sink
        self.provider = provider
        self.model = model

    def as_choice(self) -> Any:
        """Index-bound `choose` for `alphaengine.agent.AgentDriver`.

        The library driver is the primitive: it takes `(permitted, figures) -> int`
        and cannot name an op. This wrapper is the prompt+goal layer that USES it.
        """

        def choose(permitted: list[Figures], figures: Figures) -> int:
            return self.pick(permitted, figures)

        return choose

    # ── the one decision ───────────────────────────────────────────────────

    def pick(self, permitted: list[Figures], figures: Any = None) -> int:
        """Return an INDEX into `permitted`. Never a step, never an op name.

        The index is the safety property. A model cannot name an operation the
        server did not offer, because the only thing it can return is a position
        in the server's own list.
        """
        if not permitted:
            raise AgentRefusal("nothing is permitted; there is no choice to make")
        if len(permitted) == 1:
            # No sense spending a model call on a list of one, and no sense
            # letting one be got wrong.
            return 0

        options = "\n".join(
            f"  {i}. {p.get('op', '?')}" + (f"  {_brief(p.get('params'))}" if p.get("params") else "")
            for i, p in enumerate(permitted)
        )
        prompt = _PROMPT.format(
            goal=self.goal,
            history="\n".join(f"  - {h}" for h in self.history[-12:]) or "  (nothing yet)",
            measured=summarize_figures(figures),
            options=options,
        )
        raw = self.think(prompt)
        idx, why = _parse_choice(raw, len(permitted))
        if why:
            self.on_thought(why)
            self.history.append(f"chose {permitted[idx].get('op')}: {why}")
        else:
            self.history.append(f"chose {permitted[idx].get('op')}")
        if self.sink is not None:
            import hashlib

            from ..events import make_event

            self.sink.emit(
                make_event(
                    "pick",
                    provider=self.provider,
                    model=self.model,
                    choice=idx,
                    why=why,
                    workflow=str(permitted[idx].get("op") or ""),
                    agency="exploratory",
                    prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:16],
                    prompt_chars=len(prompt),
                )
            )
        return idx

    # ── the loop ───────────────────────────────────────────────────────────

    def drive(self, run: Any) -> Any:
        """Run to completion, asking the model at each fork.

        The structure mirrors `Run.drive` deliberately, including the
        two-identical-failures stop: a step this build cannot execute fails the
        same way every time, and the server correctly re-offers it, so without a
        counter an agentic run spins exactly as the scripted one used to.
        """
        n = 0
        failures: dict[str, int] = {}
        done_ids: set[str] = set()
        done_sigs: set[str] = set()
        while run.status == "open" and n < self.max_steps:
            if not run.permitted:
                run.resume()
                if run.status != "open" or not run.permitted:
                    break

            fresh = [p for p in run.permitted if not _already_done(p, done_ids, done_sigs)]
            if not fresh:
                op = str((run.permitted[0] or {}).get("op") or "?")
                self.on_step(f"stop     {op} already finished")
                self.history.append(f"{op} already finished; not running it again")
                run.status = "stopped"
                run.stopped = {"reason": "already finished", "op": op}
                return run

            if getattr(run, "selection", None) == "all":
                batch = list(fresh)
            elif len(fresh) == 1:
                batch = fresh
            else:
                batch = [fresh[self.pick(fresh, getattr(run, "figures", None))]]

            for step in batch:
                op = str(step.get("op") or "?")
                self.on_step(f"step {n + 1:<2}  {op}")
                before = run.status
                started = time.monotonic()
                run.step(step)
                took = time.monotonic() - started
                n += 1

                still = any(p.get("step_id") == step.get("step_id") for p in run.permitted)
                if still and before == "open" and run.status == "open":
                    key = str(step.get("step_id"))
                    failures[key] = failures.get(key, 0) + 1
                    self.history.append(f"{op} could not be executed here")
                    self.on_step(f"         could not be executed   {took:.1f}s")
                    if failures[key] >= 2:
                        run.status = "abandoned"
                        run.stopped = {"reason": "step_failed", "op": step.get("op")}
                        return run
                else:
                    sid = str(step.get("step_id") or "")
                    if sid:
                        done_ids.add(sid)
                    done_sigs.add(_signature(step))
                    found = _finding(getattr(run, "figures", None), op)
                    note = f"{op} finished"
                    if found:
                        note += f" ({found})"
                    self.history.append(note)
                    detail = f"         {found}   {took:.1f}s" if found else f"         {took:.1f}s"
                    self.on_step(detail)
                if run.status != "open":
                    return run

        if run.status == "open" and n >= self.max_steps:
            raise AgentRefusal(
                f"the run was still open after {self.max_steps} steps; "
                "stopping rather than continuing to spend model calls"
            )
        return run


# ── parsing, which has to assume the model is sloppy ───────────────────────


def summarize_figures(figures: Any, *, limit: int = 8) -> str:
    """Scalars the run already produced. Lists stay out: a list is a series."""
    if not isinstance(figures, dict) or not figures:
        return "(nothing yet)"
    lines: list[str] = []
    for op, blob in figures.items():
        if not isinstance(blob, dict):
            continue
        bits = _scalar_bits(blob)
        if bits:
            lines.append(f"  {op}: " + ", ".join(bits))
        if len(lines) >= limit:
            break
    return "\n".join(lines) or "(nothing yet)"


def _scalar_bits(blob: dict[str, Any], limit: int = 4) -> list[str]:
    bits: list[str] = []
    for key, value in blob.items():
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, float):
            bits.append(f"{key}={value:.4g}")
        elif isinstance(value, int):
            bits.append(f"{key}={value}")
        elif isinstance(value, str) and value.strip() and len(value) <= 80:
            bits.append(f"{key}={value.strip()}")
        if len(bits) >= limit:
            break
    return bits


def _finding(figures: Any, op: str) -> str:
    if not isinstance(figures, dict):
        return ""
    blob = figures.get(op)
    if not isinstance(blob, dict):
        return ""
    return ", ".join(_scalar_bits(blob))


def _signature(step: dict[str, Any]) -> str:
    """Same op and same params is the same step, even with a new id."""
    return f"{step.get('op') or ''}|{_brief(step.get('params') or {})}"


def _already_done(step: dict[str, Any], done_ids: set[str], done_sigs: set[str]) -> bool:
    sid = str(step.get("step_id") or "")
    if sid and sid in done_ids:
        return True
    return _signature(step) in done_sigs


def prefer_index(question: str, options: list[dict[str, Any]]) -> int | None:
    """The one option whose own words the question uses.

    None when the question does not single one out. There is no default and
    no preferred order: a tie, or a weak overlap, goes to the model.
    """
    if not options:
        return None
    question_l = question.lower()
    q = _content_tokens(question)
    q_words = set(re.findall(r"[a-z0-9]+", question_l))
    scores: list[int] = []
    for opt in options:
        answers = opt.get("answers")
        if not isinstance(answers, str):
            params = opt.get("params")
            answers = params.get("answers") if isinstance(params, dict) else ""
        score = len(q & _content_tokens(str(answers or "")))
        name = str(opt.get("op") or "").lower()
        if name and name.replace("_", " ") in question_l:
            score += 3
        for piece in name.split("_"):
            if len(piece) >= 4 and piece in q_words:
                score += 2
        scores.append(score)
    best = max(scores)
    if best < 2:
        return None
    winners = [i for i, score in enumerate(scores) if score == best]
    return winners[0] if len(winners) == 1 else None


def _content_tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 4 and w not in _STOPWORDS}


def _brief(params: Any) -> str:
    """A short, SAFE rendering of a step's params for the prompt.

    Truncated hard. Params can carry a grid, and a grid is frequently bigger
    intellectual property than the returns — there is no reason to send all of
    it to a third-party model just to label a menu item.
    """
    try:
        s = json.dumps(params, default=str)
    except (TypeError, ValueError):
        s = str(params)
    return s[:120] + ("…" if len(s) > 120 else "")


def _parse_choice(raw: str, n: int) -> tuple[int, str]:
    """Pull `{"choice": i, "why": "..."}` out of whatever the model actually said.

    Models wrap JSON in prose, in code fences, or answer with a bare number.
    Each of those is a real thing they do, so each is handled — but an answer
    that is not a valid index is REFUSED rather than defaulted to 0. Silently
    taking the first option when the model said something unparseable would
    produce a run that looks agent-driven and is not.
    """
    text = (raw or "").strip()

    obj: Any = None
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
        except ValueError:
            obj = None

    if isinstance(obj, dict) and "choice" in obj:
        try:
            idx = int(obj["choice"])
        except (TypeError, ValueError):
            raise AgentRefusal(f"choice was not an integer: {obj['choice']!r}") from None
        why = str(obj.get("why") or "").strip()
    else:
        bare = re.search(r"-?\d+", text)
        if not bare:
            raise AgentRefusal(f"no choice found in the model's reply: {raw[:200]!r}")
        idx, why = int(bare.group(0)), ""

    if not 0 <= idx < n:
        raise AgentRefusal(
            f"the model chose step {idx}, which was not offered (0..{n - 1}). It cannot invent a step."
        )
    return idx, why
