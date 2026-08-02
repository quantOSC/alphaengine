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
    There is no config file, no keyring, and no field on any of our objects to
    put one in.

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

import os
from collections.abc import Callable

__all__ = ["available_models", "build_think", "NoModelConfigured"]


class NoModelConfigured(RuntimeError):
    """No provider key in the environment, so there is nothing to think with."""


#: (env var, provider label, default model). Order is preference order.
_PROVIDERS: list[tuple[str, str, str]] = [
    ("ANTHROPIC_API_KEY", "anthropic", "claude-sonnet-4-5"),
    ("OPENAI_API_KEY", "openai", "gpt-4o"),
]


def available_models() -> list[tuple[str, str]]:
    """Which providers this machine could use, as (label, model)."""
    return [(label, model) for env, label, model in _PROVIDERS if os.environ.get(env)]


def build_think(model: str | None = None) -> tuple[Callable[[str], str], str]:
    """Return `(think, label)` for the first provider with a key present.

    `think` is `(prompt) -> text` and nothing more, because that is the whole
    interface `AgentDriver` accepts.
    """
    for env, label, default in _PROVIDERS:
        key = os.environ.get(env)
        if not key:
            continue
        chosen = model or os.environ.get("ALPHAENGINE_MODEL") or default
        if label == "anthropic":
            return _anthropic(key, chosen), f"{label}/{chosen}"
        if label == "openai":
            return _openai(key, chosen), f"{label}/{chosen}"

    raise NoModelConfigured(
        "No model key found. The agentic path runs under YOUR account, so it "
        "needs your key in this shell:\n"
        "    ANTHROPIC_API_KEY=...   or   OPENAI_API_KEY=...\n"
        "Nothing here stores it, and the scripted path (`run <workflow>`) needs "
        "no model at all."
    )


def _anthropic(key: str, model: str) -> Callable[[str], str]:
    def think(prompt: str) -> str:
        try:
            import anthropic  # imported here: never at package import time
        except ImportError as exc:  # pragma: no cover - depends on the machine
            raise NoModelConfigured(
                "ANTHROPIC_API_KEY is set but the SDK is not installed: "
                "pip install anthropic"
            ) from exc
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")

    return think


def _openai(key: str, model: str) -> Callable[[str], str]:
    def think(prompt: str) -> str:
        try:
            import openai  # imported here: never at package import time
        except ImportError as exc:  # pragma: no cover - depends on the machine
            raise NoModelConfigured(
                "OPENAI_API_KEY is set but the SDK is not installed: pip install openai"
            ) from exc
        client = openai.OpenAI(api_key=key)
        out = client.chat.completions.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return out.choices[0].message.content or ""

    return think
