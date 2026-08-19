"""BYOK provider router: any key on this machine can light the third rung."""

from __future__ import annotations

import pytest

from alphaengine.model import (
    NoModelConfigured,
    available_models,
    build_think,
    keys_without_sdk,
    parse_model_spec,
    provider_labels,
)


def test_parse_model_spec_splits_provider_and_name():
    assert parse_model_spec("openai/gpt-4o") == ("openai", "gpt-4o")
    assert parse_model_spec("anthropic") == ("anthropic", None)
    assert parse_model_spec("gpt-4o") == (None, "gpt-4o")


def test_available_models_returns_every_usable_provider(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-a")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-o")
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setattr(
        "alphaengine.model._sdk_present",
        lambda label: label in {"anthropic", "openai", "groq"},
    )
    labels = [p for p, _m in available_models()]
    assert labels == ["anthropic", "openai", "groq"]


def test_build_think_honours_a_provider_pin(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-a")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-o")
    monkeypatch.setattr("alphaengine.model._sdk_present", lambda _label: True)
    monkeypatch.setenv("ALPHAENGINE_PROVIDER", "openai")
    monkeypatch.setenv("ALPHAENGINE_MODEL", "gpt-4o-mini")
    _think, label = build_think()
    assert label == "openai/gpt-4o-mini"


def test_openai_compat_base_url_is_read_not_hardcoded(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-o")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("alphaengine.model._sdk_present", lambda label: label == "openai")
    _think, label = build_think()
    assert label.startswith("openai/")


def test_unknown_provider_pin_is_a_message(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-o")
    monkeypatch.setattr("alphaengine.model._sdk_present", lambda _label: True)
    monkeypatch.setenv("ALPHAENGINE_PROVIDER", "not-a-vendor")
    with pytest.raises(NoModelConfigured, match="unknown provider"):
        build_think()


def test_events_never_include_the_prompt_or_the_key(monkeypatch):
    seen: list[dict] = []
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-should-not-leak")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("alphaengine.model._sdk_present", lambda label: label == "openai")

    def fake_build(spec, key, model, on_token):
        assert key == "sk-secret-should-not-leak"

        def think(prompt: str) -> str:
            return "ok"

        return think

    monkeypatch.setattr("alphaengine.model._build_for", fake_build)
    think, _label = build_think(on_event=seen.append)
    think("a prompt that must not be logged")
    assert seen
    blob = str(seen)
    assert "sk-secret" not in blob
    assert "a prompt that must not" not in blob
    assert seen[0]["prompt_hash"]
    assert seen[0]["prompt_chars"] == len("a prompt that must not be logged")


def test_provider_labels_cover_the_gateways():
    labels = provider_labels()
    for name in ("anthropic", "openai", "azure", "gemini", "groq", "openrouter", "gateway"):
        assert name in labels


def test_keys_without_sdk_names_every_stranded_provider(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setattr("alphaengine.model._sdk_present", lambda _label: False)
    assert "gemini" in keys_without_sdk()
