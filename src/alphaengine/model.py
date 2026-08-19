"""Your model, from your environment. Optional, and lazy.

`AgentDriver` takes a callable and knows nothing about providers — that is what
makes "runs under your own key" structural rather than promised. But somebody
sitting at a terminal should not have to write an adapter to type a sentence, so
this module builds that callable out of environment variables the user already
has set for every other tool they use.

── THE RULES THIS FILE MUST NOT BREAK ─────────────────────────────────────────

  NOTHING HERE IS IMPORTED AT PACKAGE IMPORT TIME. `import alphaengine` still
    makes no network call and pulls in no HTTP client; two tests assert it. Every
    provider import below happens inside a function, on the call that needs it.

  NO KEY IS EVER STORED, LOGGED, OR SENT ANYWHERE BUT THE PROVIDER. It is read
    from the environment at call time and handed to the provider's own client.
    Persistence of keys, if the user asks, lives in `auth.py` (mode 0600 under
    the user config dir) — never in this module, never on a field, never to
    QuantOS.

  NO PROVIDER IS A DEPENDENCY. `pip install alphaengine` still installs numpy and
    scipy and nothing else. If you want the agentic path you already have an SDK
    installed, because you already had a key; if you do not, the error says
    exactly which one to install.

── WHY ENVIRONMENT VARIABLES AND NOT A FLAG ───────────────────────────────────

A key on a command line is a key in shell history, in `ps` output on a shared
box, and in any terminal recording. The env-var convention is what every other
tool in this ecosystem uses, so it is also the one already set on the machine.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = [
    "available_models",
    "build_think",
    "keys_without_sdk",
    "pin_model",
    "parse_model_spec",
    "provider_labels",
    "NoModelConfigured",
    "ThinkEvent",
]


class NoModelConfigured(RuntimeError):
    """No provider key in the environment, so there is nothing to think with."""


ThinkEvent = dict[str, Any]


@dataclass(frozen=True)
class _Spec:
    """One way a key on this machine can become a `think` callable."""

    label: str
    env_keys: tuple[str, ...]
    default_model: str
    sdk: str
    kind: str  # anthropic | openai_compat | azure
    default_base: str | None = None
    base_env: str | None = None


#: Preference order. First usable wins unless ALPHAENGINE_PROVIDER / a pin says
#: otherwise. OpenAI-compatible gateways share the `openai` SDK on purpose: one
#: install lights Groq, OpenRouter, Gemini's OpenAI surface, Azure, vLLM, and a
#: private company gateway.
_SPECS: tuple[_Spec, ...] = (
    _Spec("anthropic", ("ANTHROPIC_API_KEY",), "claude-sonnet-4-5", "anthropic", "anthropic"),
    _Spec("openai", ("OPENAI_API_KEY",), "gpt-4o", "openai", "openai_compat", None, "OPENAI_BASE_URL"),
    _Spec(
        "azure",
        ("AZURE_OPENAI_API_KEY",),
        "gpt-4o",
        "openai",
        "azure",
        None,
        "AZURE_OPENAI_ENDPOINT",
    ),
    _Spec(
        "gemini",
        ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "gemini-2.0-flash",
        "openai",
        "openai_compat",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_BASE_URL",
    ),
    _Spec(
        "groq",
        ("GROQ_API_KEY",),
        "llama-3.3-70b-versatile",
        "openai",
        "openai_compat",
        "https://api.groq.com/openai/v1",
        "GROQ_BASE_URL",
    ),
    _Spec(
        "openrouter",
        ("OPENROUTER_API_KEY",),
        "openai/gpt-4o",
        "openai",
        "openai_compat",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_BASE_URL",
    ),
    _Spec(
        "gateway",
        ("ALPHAENGINE_API_KEY",),
        "gpt-4o",
        "openai",
        "openai_compat",
        None,
        "ALPHAENGINE_BASE_URL",
    ),
)

#: Back-compat alias used by older tests that iterate (env, label, default).
_PROVIDERS: list[tuple[str, str, str]] = [
    (spec.env_keys[0], spec.label, spec.default_model) for spec in _SPECS
]


def provider_labels() -> tuple[str, ...]:
    """Every label `key <label>` and `model <label>` will accept."""
    return tuple(s.label for s in _SPECS)


def _spec_by_label(label: str) -> _Spec | None:
    wanted = label.strip().lower()
    for spec in _SPECS:
        if spec.label == wanted:
            return spec
    return None


def _key_for(spec: _Spec) -> str | None:
    for name in spec.env_keys:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _base_for(spec: _Spec) -> str | None:
    if spec.base_env:
        override = os.environ.get(spec.base_env)
        if override:
            return override
    return spec.default_base


#: Which import each provider needs, checked BEFORE a closure is handed out.
_SDK = {spec.label: spec.sdk for spec in _SPECS}


def _sdk_present(label: str) -> bool:
    """Is the provider's SDK importable, without importing it?

    `find_spec` rather than a try/import: this runs on every boot of the
    interactive shell and importing a large SDK to discover it exists is a
    visible pause for no result.
    """
    import importlib.util  # noqa: PLC0415

    try:
        return importlib.util.find_spec(_SDK[label]) is not None
    except (ImportError, ValueError, KeyError):  # pragma: no cover - malformed install
        return False


def available_models() -> list[tuple[str, str]]:
    """Which providers this machine can ACTUALLY use, as (label, model).

    A key with no SDK behind it does not count. The boot screen lights its third
    rung from this, and a rung that lights on a provider which then raises is a
    boot screen that lied — the specific failure the capability ladder exists to
    prevent, since its whole job is telling a user what they cannot do and why.

    ALL usable providers are returned, not only the first. `/model` picks among
    them; `build_think` still defaults to the first unless a pin is set.
    """
    out: list[tuple[str, str]] = []
    for spec in _SPECS:
        if _key_for(spec) and _sdk_present(spec.label):
            pinned = os.environ.get("ALPHAENGINE_PROVIDER")
            model = spec.default_model
            if pinned == spec.label:
                model = os.environ.get("ALPHAENGINE_MODEL") or spec.default_model
            out.append((spec.label, model))
    return out


def keys_without_sdk() -> list[str]:
    """Providers the user has a key for and cannot use yet.

    Reported separately so the boot screen can say "one install away" rather
    than "not configured", which are different situations and want different
    next steps.
    """
    return [spec.label for spec in _SPECS if _key_for(spec) and not _sdk_present(spec.label)]


def parse_model_spec(spec: str) -> tuple[str | None, str | None]:
    """`openai/gpt-4o`, `openai:gpt-4o`, `anthropic`, or a bare model name.

    Returns `(provider, model)` with either side optional. A bare model name
    leaves the provider unset so `build_think` keeps the current one.
    """
    text = (spec or "").strip()
    if not text:
        return None, None
    for sep in ("/", ":"):
        if sep in text:
            left, right = text.split(sep, 1)
            if _spec_by_label(left):
                return left.strip().lower(), right.strip() or None
    if _spec_by_label(text):
        return text.lower(), None
    return None, text


def pin_model(spec: str) -> tuple[str | None, str | None]:
    """Pin provider/model for THIS PROCESS. Returns the resolved (label, model)."""
    provider, model = parse_model_spec(spec)
    if provider:
        os.environ["ALPHAENGINE_PROVIDER"] = provider
    if model:
        os.environ["ALPHAENGINE_MODEL"] = model
    return provider, model


def build_think(
    model: str | None = None,
    on_token: Callable[[str], None] | None = None,
    on_event: Callable[[ThinkEvent], None] | None = None,
    provider: str | None = None,
) -> tuple[Callable[[str], str], str]:
    """Return `(think, label)` for a provider that is actually USABLE.

    `think` is `(prompt) -> text` and nothing more, because that is the whole
    interface `AgentDriver` accepts.

    ── `on_token` STREAMS, AND IT DOES NOT WIDEN THE SEAM ──────────────────

    `on_token` is called with each chunk of text as it arrives. It is a
    parameter of this BUILDER, not of `think`, on purpose: the public contract
    is `(prompt: str, /) -> str` and it has to stay that, because a user's own
    callable is passed straight into `AgentDriver` and adding a second parameter
    would break every one of them. A model somebody else supplies simply does
    not stream, which is correct — we cannot stream a function we do not own.

    ── `on_event` IS TELEMETRY, NEVER THE ANSWER ───────────────────────────

    Called with an allowlisted dict (provider, model, prompt hash, usage). The
    prompt text itself is never in the event. A sink that POSTs this to QuantOS
    still cannot recover the prompt, the key, or the book.

    ── THE BUG THIS SHAPE FIXES ────────────────────────────────────────────

    This used to return a closure whose SDK import happened on first call. So a
    machine with `ANTHROPIC_API_KEY` set and the SDK missing got a valid-looking
    `think`, and the failure surfaced inside `driver.pick()` — three frames past
    the `except NoModelConfigured` in `cli._ask` that exists to handle exactly
    this. **A callable that cannot work is not a callable.**
    """
    wanted_provider = (provider or os.environ.get("ALPHAENGINE_PROVIDER") or "").strip().lower() or None
    wanted_model = model or os.environ.get("ALPHAENGINE_MODEL") or None
    if wanted_model and "/" in wanted_model and wanted_provider is None:
        parsed_p, parsed_m = parse_model_spec(wanted_model)
        if parsed_p:
            wanted_provider, wanted_model = parsed_p, parsed_m

    unusable: list[str] = []
    usable: list[tuple[str, Callable[[str], str], str]] = []

    for spec in _SPECS:
        key = _key_for(spec)
        if not key:
            continue
        if not _sdk_present(spec.label):
            unusable.append(spec.label)
            continue
        if wanted_provider in (None, spec.label):
            picked_model = wanted_model or spec.default_model
        else:
            picked_model = spec.default_model
        think = _build_for(spec, key, picked_model, on_token)
        label = f"{spec.label}/{picked_model}"
        usable.append((spec.label, _wrap_events(think, spec.label, picked_model, on_event), label))

    if wanted_provider:
        if wanted_provider not in {s.label for s in _SPECS}:
            raise NoModelConfigured(
                f"unknown provider {wanted_provider!r}. Try: {', '.join(provider_labels())}."
            )
        for label, think, tag in usable:
            if label == wanted_provider:
                return think, tag
        extra = ""
        if wanted_provider in unusable:
            extra = f"\n    pip install {_SDK[wanted_provider]}"
        raise NoModelConfigured(
            f"No usable {wanted_provider} key on this machine.{extra}\n"
            "The scripted path (`run <workflow>`) needs no model at all and works now."
        )

    if usable:
        _think, tag = usable[0][1], usable[0][2]
        return _think, tag

    if unusable:
        which = " or ".join(f"pip install {_SDK[label]}" for label in unusable)
        raise NoModelConfigured(
            f"A key is set for {' and '.join(unusable)}, and the SDK is not installed.\n"
            f"    {which}\n"
            "    or, for everything the agent path needs: pip install 'alphaengine[agents]'\n"
            "The scripted path (`run <workflow>`) needs no model at all and works now."
        )

    raise NoModelConfigured(
        "No model key found. The agentic path runs under YOUR account, so it "
        "needs your key in this shell:\n"
        "    ANTHROPIC_API_KEY=...   or   OPENAI_API_KEY=...\n"
        "    or GEMINI_API_KEY / GROQ_API_KEY / OPENROUTER_API_KEY /\n"
        "    AZURE_OPENAI_API_KEY / ALPHAENGINE_API_KEY+ALPHAENGINE_BASE_URL\n"
        "Nothing here stores it, and the scripted path (`run <workflow>`) needs "
        "no model at all."
    )


def _wrap_events(
    think: Callable[[str], str],
    provider: str,
    model: str,
    on_event: Callable[[ThinkEvent], None] | None,
) -> Callable[[str], str]:
    if on_event is None:
        return think

    def wrapped(prompt: str) -> str:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        text = think(prompt)
        on_event(
            {
                "kind": "completion",
                "provider": provider,
                "model": model,
                "prompt_hash": digest,
                "prompt_chars": len(prompt),
                "completion_chars": len(text or ""),
            }
        )
        return text

    return wrapped


def _build_for(
    spec: _Spec,
    key: str,
    model: str,
    on_token: Callable[[str], None] | None,
) -> Callable[[str], str]:
    if spec.kind == "anthropic":
        return _anthropic(key, model, on_token)
    if spec.kind == "azure":
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT") or ""
        version = os.environ.get("AZURE_OPENAI_API_VERSION") or "2024-06-01"
        return _azure(key, model, endpoint, version, on_token)
    return _openai_compat(key, model, _base_for(spec), on_token)


def _anthropic(key: str, model: str, on_token: Callable[[str], None] | None = None) -> Callable[[str], str]:
    def think(prompt: str) -> str:
        try:
            # No inline ignore. The SDK is deliberately NOT a dependency, so on
            # a machine without it mypy reports import-not-found and on a
            # machine WITH it any ignore is UNUSED and strict mode errors on
            # that. One annotation cannot be right in both states, so the answer
            # is configuration: see `ignore_missing_imports` for these two
            # modules in pyproject.toml.
            import anthropic  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on the machine
            raise NoModelConfigured(
                "ANTHROPIC_API_KEY is set but the SDK is not installed: pip install anthropic"
            ) from exc
        client = anthropic.Anthropic(api_key=key)
        # `Any` because the two calls return different concrete types that the
        # SDK does not unify — `Message` and `ParsedMessage[None]` — and the
        # only thing read off either is the text blocks, which both carry.
        msg: Any
        if on_token is not None:
            # `messages.stream` accumulates the message for us, so the streamed
            # chunks are a SIDE CHANNEL for the screen and never the source of
            # the returned text. Rebuilding the reply from chunks would make a
            # dropped chunk a silently wrong answer rather than a visible error.
            with client.messages.stream(
                model=model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for chunk in stream.text_stream:
                    on_token(chunk)
                msg = stream.get_final_message()
        else:
            msg = client.messages.create(
                model=model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
        # `getattr` rather than `b.text`. A response block is a union of a dozen
        # types and only TextBlock carries `.text` — reading the attribute
        # directly is a type error on every other member, and the runtime filter
        # in front of it is invisible to a checker. This says what the filter
        # already guarantees, in a form that survives the SDK adding a
        # fourteenth block type next month.
        return "".join(str(getattr(b, "text", "")) for b in msg.content if getattr(b, "type", None) == "text")

    return think


def _openai_compat(
    key: str,
    model: str,
    base_url: str | None,
    on_token: Callable[[str], None] | None = None,
) -> Callable[[str], str]:
    def think(prompt: str) -> str:
        try:
            import openai  # noqa: PLC0415  # see the anthropic import above
        except ImportError as exc:  # pragma: no cover - depends on the machine
            raise NoModelConfigured(
                "An OpenAI-compatible key is set but the SDK is not installed: pip install openai"
            ) from exc
        kwargs: dict[str, Any] = {"api_key": key}
        if base_url:
            kwargs["base_url"] = base_url
        client = openai.OpenAI(**kwargs)
        return _chat(client, model, prompt, on_token)

    return think


def _azure(
    key: str,
    model: str,
    endpoint: str,
    api_version: str,
    on_token: Callable[[str], None] | None = None,
) -> Callable[[str], str]:
    def think(prompt: str) -> str:
        try:
            import openai  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise NoModelConfigured(
                "AZURE_OPENAI_API_KEY is set but the SDK is not installed: pip install openai"
            ) from exc
        if not endpoint:
            raise NoModelConfigured(
                "AZURE_OPENAI_API_KEY is set and AZURE_OPENAI_ENDPOINT is not. "
                "Export the Azure resource endpoint, then retry."
            )
        client = openai.AzureOpenAI(api_key=key, azure_endpoint=endpoint, api_version=api_version)
        return _chat(client, model, prompt, on_token)

    return think


def _chat(client: Any, model: str, prompt: str, on_token: Callable[[str], None] | None) -> str:
    if on_token is not None:
        # No accumulating helper here, so the chunks ARE the answer and the
        # join is the return value rather than a second source of truth.
        parts: list[str] = []
        for event in client.chat.completions.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        ):
            piece = (event.choices[0].delta.content or "") if event.choices else ""
            if piece:
                parts.append(piece)
                on_token(piece)
        return "".join(parts)
    out = client.chat.completions.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return out.choices[0].message.content or ""
