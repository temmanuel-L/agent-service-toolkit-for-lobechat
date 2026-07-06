import os
from unittest.mock import patch

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import FakeListChatModel
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from core.llm import get_model
from schema.models import (
    AnthropicModelName,
    FakeModelName,
    GroqModelName,
    OllamaModelName,
    OpenAICompatibleName,
    OpenAIModelName,
)


def test_get_model_openai():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
        model = get_model(OpenAIModelName.GPT_5_NANO)
        assert isinstance(model, ChatOpenAI)
        assert model.model_name == "gpt-5-nano"
        assert model.streaming is True


def test_get_model_openai_fast():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
        model = get_model(OpenAIModelName.GPT_5_NANO, fast=True)
        assert isinstance(model, ChatOpenAI)
        assert model.reasoning_effort == "minimal"


def test_get_model_openai_compatible_fast():
    with patch("core.settings.settings.COMPATIBLE_BASE_URL", "http://localhost/v1"), patch(
        "core.settings.settings.COMPATIBLE_API_KEY", "test_key"
    ):
        model = get_model(OpenAICompatibleName.OPENAI_NAME, fast=True)
        assert isinstance(model, ChatOpenAI)
        assert model.model_name == "MiniMax-M3"
        assert model.extra_body == {"thinking": {"type": "disabled"}}


def test_get_model_openai_compatible_default_disables_thinking():
    with patch("core.settings.settings.COMPATIBLE_BASE_URL", "http://localhost/v1"), patch(
        "core.settings.settings.COMPATIBLE_API_KEY", "test_key"
    ):
        model = get_model(OpenAICompatibleName.OPENAI_NAME)
        assert model.extra_body == {"thinking": {"type": "disabled"}}


def test_get_model_anthropic():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
        model = get_model(AnthropicModelName.HAIKU_45)
        assert isinstance(model, ChatAnthropic)
        assert model.model == "claude-haiku-4-5"
        assert model.temperature == 0.5
        assert model.streaming is True


def test_get_model_groq():
    with patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}):
        model = get_model(GroqModelName.LLAMA_31_8B)
        assert isinstance(model, ChatGroq)
        assert model.model_name == "llama-3.1-8b"
        assert model.temperature == 0.5


def test_get_model_groq_guard():
    with patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}):
        model = get_model(GroqModelName.LLAMA_GUARD_4_12B)
        assert isinstance(model, ChatGroq)
        assert model.model_name == "meta-llama/llama-guard-4-12b"
        assert model.temperature < 0.01


def test_get_model_ollama():
    with patch("core.settings.settings.OLLAMA_MODEL", "llama3.3"):
        model = get_model(OllamaModelName.OLLAMA_GENERIC)
        assert isinstance(model, ChatOllama)
        assert model.model == "llama3.3"
        assert model.temperature == 0.5


def test_get_model_fake():
    model = get_model(FakeModelName.FAKE)
    assert isinstance(model, FakeListChatModel)
    assert model.responses == ["This is a test response from the fake model."]


def test_get_model_invalid():
    with pytest.raises(ValueError, match="Unsupported model:"):
        # Using type: ignore since we're intentionally testing invalid input
        get_model("invalid_model")  # type: ignore
