"""The synthesis pass: turn a finished run into an answer, or refuse to.

    from alphaengine.agent import AgentDriver, synthesize

    run = driver.drive(run)
    answer = synthesize(question, run, write=my_model_fn)
    print(answer.text)
    for caveat in answer.caveats:
        print(" ", caveat)

WHY THIS IS A SEPARATE CALL AND NOT A RICHER RETURN FROM `pick()`

    `AgentDriver.pick()` returns an INDEX into the permitted set, and that is the
    safety property: the model's action space is what the server offered this
    turn, nothing it can name. An unparseable reply raises rather than defaulting
    to step 0, because silently choosing the first permitted step is exactly the
    failure the design exists to prevent.

    The consequence was that THE AGENT NEVER ANSWERED. A question in plain
    English produced step narration and a verdict and no answer, because there
    was no call whose job was to say anything. Making `pick()` return prose would
    have fixed that by widening the one interface whose narrowness is the whole
    guarantee.

    So: different call, different contract. This one may write freely and may
    cite NOTHING the run did not produce, which is enforced here the same way the
    index bound is enforced there — structurally, after the model replies, with a
    refusal rather than a repair.

THE CONTRACT, PRECISELY

    An answer may reference only figures the run recorded. A number that appears
    in the text and not in the run is an `UncitedFigure` and the answer is
    refused, not trimmed. Trimming would leave a fluent answer with a hole in it,
    and a fluent answer is the format in which a made-up number is most likely to
    be believed.

    Every honesty control on those figures travels with them. `None` stays "not
    recorded" and never becomes 0. A `not_recorded` trial count still caps the
    verdict. A monitor that checked nothing does not get to read as all-clear.
    These arrive as `caveats`, and a caller that renders `text` without them has
    reintroduced the failure this product exists to fix.

YOUR MODEL, YOUR KEY, STILL

    `write` is a callable you supply, exactly like `choose`. This package has no
    model dependency and nowhere to put a key.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["synthesize", "Answer", "UncitedFigure", "Writer", "figures_of", "caveats_of"]

#: Given the question and a flattened view of what the run recorded, return
#: prose. Nothing else: not a figure it invented, not a recommendation the run
#: did not support.
Writer = Callable[[str, dict[str, Any]], str]


class UncitedFigure(ValueError):
    """The answer contained a number the run did not produce.

    Raised rather than repaired. An answer with the invented figure quietly
    removed still reads as though the system knew something, and prose is the
    format in which a made-up number is most likely to be believed.
    """


@dataclass
class Answer:
    """What the run supports, and everything that qualifies it."""

    text: str
    #: The run's figures, flattened to `stage.key` — what the text was allowed to
    #: draw on, kept so a reader can check any sentence against a number.
    cited: dict[str, float] = field(default_factory=dict)
    #: Honesty controls that MUST be rendered with the text. Not decoration: each
    #: one is a claim the figures cannot support without it.
    caveats: list[str] = field(default_factory=list)
    #: The strongest verdict these figures can carry, or None when nothing caps
    #: it. A `not_recorded` denominator caps at "inconclusive" however good the
    #: deflated Sharpe looks.
    verdict_ceiling: str | None = None
    #: Set when the run stopped. A stop is a RESULT and is the answer.
    stopped: str | None = None


# ── reading what the run produced ──────────────────────────────────────────


def figures_of(run: Any) -> dict[str, float]:
    """Flatten a run's figures to `stage.key` -> number.

    Only NUMBERS. Strings, verdicts and notes are context for the writer but not
    citable quantities, and the check below is about quantities: a model that
    misquotes a verdict is wrong in a way a reader can see, and one that
    misquotes a Sharpe is not.
    """
    raw = getattr(run, "figures", None)
    if raw is None and isinstance(run, dict):
        raw = run.get("figures")
    out: dict[str, float] = {}
    _walk(raw or {}, "", out)
    return out


def _walk(node: Any, path: str, out: dict[str, float]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            _walk(value, f"{path}.{key}" if path else str(key), out)
        return
    if isinstance(node, (list, tuple)):
        for i, value in enumerate(node):
            _walk(value, f"{path}[{i}]", out)
        return
    # bool is an int in Python and "True" is not a figure anybody cites.
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        out[path] = float(node)


def caveats_of(run: Any) -> tuple[list[str], str | None]:
    """Every honesty control these figures carry, and the verdict ceiling.

    DERIVED FROM THE FIGURES, never asked of the model. A caveat the model was
    invited to include is a caveat it can decline to include, and the whole
    differentiator is that these survive to the screen whether or not anything
    downstream wants them to.
    """
    figures = getattr(run, "figures", None)
    if figures is None and isinstance(run, dict):
        figures = run.get("figures")
    figures = figures or {}

    caveats: list[str] = []
    ceiling: str | None = None

    for stage, blob in figures.items():
        if not isinstance(blob, dict):
            continue

        source = blob.get("n_trials_source")
        if source == "not_recorded":
            caveats.append(
                "No trial count was recorded, so the Sharpe could not be deflated. "
                "Nothing here can be called an edge."
            )
            ceiling = "inconclusive"
        elif source == "asserted":
            caveats.append("The trial count was stated rather than counted from a search that ran.")
            ceiling = ceiling or "marginal"

        if blob.get("status") == "unchecked":
            caveats.append("No tolerances were given, so nothing was checked. This is not a clean bill.")
        elif blob.get("status") == "undetermined":
            caveats.append("A stated limit could not be measured, so it is unknown whether it holds.")

        if blob.get("truncated") is True:
            caveats.append("The list was cut to the requested length; more names qualified.")

        insufficient = blob.get("n_insufficient")
        universe = blob.get("universe_size")
        if isinstance(insufficient, int) and insufficient > 0 and isinstance(universe, int) and universe:
            caveats.append(
                f"{insufficient} of {universe} names could not be measured and are absent "
                "from the ranking rather than ranked low."
            )

        if blob.get("low_sample") is True:
            caveats.append("The sample is small enough that the tail figures are indicative only.")

        # `null` is "not recorded" and never 0. Reported per stage rather than
        # per key so a run with several gaps does not bury the text in caveats.
        missing = sorted(k for k, v in blob.items() if v is None)
        if missing:
            caveats.append(f"Not recorded in {stage}: {', '.join(missing)}. Absent, not zero.")

    # Dedupe, keep first-seen order: two stages reporting the same gap is one
    # fact about the run, not two. Written as a loop rather than the usual
    # set-in-a-comprehension trick because that trick relies on `set.add`
    # returning None, which is true and is also the kind of cleverness a type
    # checker is right to object to.
    seen: set[str] = set()
    ordered: list[str] = []
    for caveat in caveats:
        if caveat not in seen:
            seen.add(caveat)
            ordered.append(caveat)
    return ordered, ceiling


# ── checking what came back ────────────────────────────────────────────────

#: Numbers that must be traceable to the run.
#:
#: THE THRESHOLD IS A DELIBERATE TRADE AND IT IS WORTH STATING PLAINLY. A model
#: writing "3 of the 5 names held up" is doing arithmetic on the run's own
#: results, and refusing that sentence would make this unusable. A model writing
#: "a Sharpe of 1.42" is making a claim that either came from the run or did not.
#:
#: So: anything with a DECIMAL POINT, and any integer of 100 or more, must be
#: traceable. Bare integers under 100 are treated as prose. That leaves a real
#: gap — a fabricated "12 trials" passes — and the gap is named here rather than
#: papered over, because a guard whose limits are undocumented gets trusted past
#: them.
#:
#: The "-" is a SIGN only when it does not follow a digit or a dot: in
#: "returns ranged 146-124" the dash is a range, and reading it as negation
#: refused a true sentence for citing a -124 nobody wrote.
_NUMBER = re.compile(r"(?<![\d.])-?\d[\d,]*\.?\d*")
_BARE_INT_FLOOR = 100

#: Rounding slack. A model quoting 0.55 for a figure recorded as 0.5478 is
#: quoting the run, and refusing that is pedantry that teaches callers to turn
#: the check off.
_TOLERANCE = 0.005


def _numbers_in(text: str) -> list[tuple[float, int]]:
    """Each number in the text, with the DECIMALS THE TOKEN ITSELF CLAIMS.

    The precision travels because quoting is lossy on purpose: "146" for a
    recorded 146.76 is the run's own figure at integer precision, and telling
    it apart from an invented 146 requires knowing the writer claimed no
    decimals.
    """
    out: list[tuple[float, int]] = []
    for match in _NUMBER.finditer(text or ""):
        token = match.group().rstrip(".").replace(",", "")
        if not token or token in ("-",):
            continue
        try:
            value = float(token)
        except ValueError:
            continue
        if "." not in token and abs(value) < _BARE_INT_FLOOR:
            continue
        decimals = len(token.split(".")[1]) if "." in token else 0
        out.append((value, decimals))
    return out


def _quotes(cited: float, decimals: int, recorded: float) -> bool:
    """Is `cited` the recorded figure, quoted at the cited token's precision?

    A model that writes "146" for a recorded 146.76 rounded or truncated a
    figure it was quoting — refusing that teaches callers to turn the check
    off, which is strictly worse than admitting precision-lossy quotes of REAL
    figures. What is admitted is exactly that and nothing wider: the cited
    value must equal the recorded one under round or trunc at the precision
    the token claims, so a cited 145 still fails against 146.76, and a figure
    the run never produced still fails against everything.

    Percent and fraction are the same figure wearing different units, and both
    spellings appear across these figures by design.
    """
    for rec in (recorded, recorded * 100.0, recorded / 100.0):
        if abs(rec - cited) <= _TOLERANCE:
            return True
        scale = 10.0**decimals
        if abs(round(rec, decimals) - cited) <= _TOLERANCE:
            return True
        if abs(math.trunc(rec * scale) / scale - cited) <= _TOLERANCE:
            return True
    return False


def _traceable(value: float, decimals: int, known: dict[str, float]) -> bool:
    return any(_quotes(value, decimals, recorded) for recorded in known.values())


def check_citations(text: str, known: dict[str, float], question: str = "") -> None:
    """Raise if the text quotes a number the run did not produce.

    ── A NUMBER THE ASKER SUPPLIED IS NOT AN INVENTED FIGURE ───────────────────

    Asked about "the s&p 500", a model naturally writes "the S&P 500" back, and
    this refused the entire answer for citing 500. The guard was right that 500
    was not a figure and wrong that it was a claim: repeating the name of the
    thing you were asked about is not quoting a result.

    So numbers appearing VERBATIM IN THE QUESTION are traceable to the question.
    The guard stays exactly as strict for everything the model produced itself,
    which is the only place it was ever protecting anything — an invented Sharpe
    is a claim about the world, and "S&P 500" is a noun.

    Deliberately the question's numbers and not a general allowlist of round
    ones: 500 is a perfectly plausible fabricated observation count, and it
    should still be refused when nobody asked about 500 of anything.
    """
    from_question = {v for v, _ in _numbers_in(question)} if question else set()
    invented = [
        v
        for v, decimals in _numbers_in(text)
        if not _traceable(v, decimals, known) and not any(abs(q - v) <= _TOLERANCE for q in from_question)
    ]
    if invented:
        raise UncitedFigure(
            f"the answer cited {invented} and this run produced no such figure. "
            "An answer may only quote numbers the run recorded."
        )


# ── the pass itself ────────────────────────────────────────────────────────


def synthesize(question: str, run: Any, *, write: Writer) -> Answer:
    """Answer `question` from what `run` recorded, or refuse.

    Args:
        question: what was actually asked, in the asker's own words.
        run: a finished run — open, stopped or closed. Anything exposing
            `figures` and optionally `stopped` / `artifact`.
        write: your model. Receives the question and the run's figures, returns
            prose. It is never given the workflow, because neither is this
            process.

    Returns:
        An `Answer` whose `caveats` are derived from the figures rather than
        requested from the model.

    Raises:
        UncitedFigure: the reply quoted a number the run did not produce.
    """
    known = figures_of(run)
    caveats, ceiling = caveats_of(run)

    stop = getattr(run, "stopped", None)
    if isinstance(stop, dict):
        stop = stop.get("reason") or stop.get("detail")

    if stop:
        # A STOP IS THE ANSWER, and it is already in plain words written by
        # somebody who knew why. Handing it to a model to rephrase risks
        # softening a refusal into a hedge, which is the one direction this
        # product cannot afford to drift.
        return Answer(
            text=str(stop), cited=known, caveats=caveats, verdict_ceiling=ceiling, stopped=str(stop)
        )

    text = str(write(question, dict(known)))
    check_citations(text, known, question)
    return Answer(text=text, cited=known, caveats=caveats, verdict_ceiling=ceiling)
