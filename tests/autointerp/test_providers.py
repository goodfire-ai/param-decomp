import asyncio
from typing import Any

import httpx
import pytest
from pydantic import TypeAdapter, ValidationError

from spd.autointerp.providers import (
    AnthropicHaiku45LLMConfig,
    AnthropicOpus46LLMConfig,
    AnthropicProvider,
    AnthropicSonnet46LLMConfig,
    GoogleAILLMConfig,
    GoogleAIProvider,
    LLMConfig,
)


def test_google_ai_llm_config_roundtrip() -> None:
    adapter = TypeAdapter(LLMConfig)
    cfg = adapter.validate_python(
        {
            "type": "google_ai",
            "model": "gemini-3-flash-preview",
            "thinking_level": "minimal",
        }
    )
    assert isinstance(cfg, GoogleAILLMConfig)
    assert cfg.model == "gemini-3-flash-preview"
    assert cfg.thinking_level == "minimal"


def test_google_ai_provider_generate_content_payload() -> None:
    provider = GoogleAIProvider(api_key="test-key", model="gemini-3-flash-preview")

    captured: dict[str, Any] = {}

    class DummyClient:
        async def post(self, url: str, json: dict[str, Any], timeout: float) -> httpx.Response:
            nonlocal captured
            _ = timeout
            captured = json
            request = httpx.Request(
                "POST", f"https://generativelanguage.googleapis.com/v1beta/{url}"
            )
            return httpx.Response(
                200,
                request=request,
                json={
                    "candidates": [
                        {
                            "content": {"parts": [{"text": '{"label":"x","reasoning":"y"}'}]},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 3,
                        "candidatesTokenCount": 7,
                    },
                },
            )

        async def aclose(self) -> None:
            return None

    provider._client = DummyClient()  # pyright: ignore[reportAttributeAccessIssue]

    schema = {
        "type": "object",
        "properties": {"label": {"type": "string"}},
        "required": ["label"],
        "additionalProperties": False,
    }
    response = asyncio.run(
        provider.chat(
            prompt="hi",
            max_tokens=256,
            response_schema=schema,
            timeout_ms=5000,
        )
    )

    assert response.content == '{"label":"x","reasoning":"y"}'
    assert response.input_tokens == 3
    assert response.output_tokens == 7
    assert captured["generationConfig"]["responseMimeType"] == "application/json"
    assert captured["generationConfig"]["responseJsonSchema"] == {
        "type": "object",
        "properties": {"label": {"type": "string"}},
        "required": ["label"],
    }


def test_google_ai_provider_includes_thinking_config_when_requested() -> None:
    provider = GoogleAIProvider(
        api_key="test-key",
        model="gemini-3-flash-preview",
        thinking_level="low",
    )

    captured: dict[str, Any] = {}

    class DummyClient:
        async def post(self, url: str, json: dict[str, Any], timeout: float) -> httpx.Response:
            nonlocal captured
            _ = timeout
            captured = json
            request = httpx.Request(
                "POST", f"https://generativelanguage.googleapis.com/v1beta/{url}"
            )
            return httpx.Response(
                200,
                request=request,
                json={
                    "candidates": [
                        {
                            "content": {"parts": [{"text": '{"label":"x","reasoning":"y"}'}]},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 3,
                        "candidatesTokenCount": 7,
                    },
                },
            )

        async def aclose(self) -> None:
            return None

    provider._client = DummyClient()  # pyright: ignore[reportAttributeAccessIssue]

    asyncio.run(
        provider.chat(
            prompt="hi",
            max_tokens=256,
            response_schema={"type": "object"},
            timeout_ms=5000,
        )
    )

    assert captured["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}


def test_anthropic_llm_config_accepts_only_valid_model_specific_shapes() -> None:
    adapter = TypeAdapter(LLMConfig)

    sonnet = adapter.validate_python(
        {"type": "anthropic", "model": "claude-sonnet-4-6", "effort": "high"}
    )
    assert isinstance(sonnet, AnthropicSonnet46LLMConfig)

    opus = adapter.validate_python(
        {"type": "anthropic", "model": "claude-opus-4-6", "effort": "max"}
    )
    assert isinstance(opus, AnthropicOpus46LLMConfig)

    haiku = adapter.validate_python(
        {
            "type": "anthropic",
            "model": "claude-haiku-4-5-20251001",
            "thinking_budget": 2048,
        }
    )
    assert isinstance(haiku, AnthropicHaiku45LLMConfig)


@pytest.mark.parametrize(
    "data",
    [
        {"type": "anthropic", "model": "claude-sonnet-4-6", "thinking_budget": 1024},
        {"type": "anthropic", "model": "claude-sonnet-4-6", "effort": "max"},
        {"type": "anthropic", "model": "claude-opus-4-6", "thinking_budget": 1024},
        {"type": "anthropic", "model": "claude-haiku-4-5-20251001", "effort": "low"},
    ],
)
def test_anthropic_llm_config_rejects_invalid_model_specific_shapes(data: dict[str, Any]) -> None:
    adapter = TypeAdapter(LLMConfig)
    with pytest.raises(ValidationError):
        adapter.validate_python(data)


def test_anthropic_provider_uses_structured_outputs_for_46_models() -> None:
    provider = AnthropicProvider(
        api_key="test-key",
        config=AnthropicSonnet46LLMConfig(effort="medium"),
    )

    captured_json: dict[str, Any] = {}

    class DummyClient:
        async def post(self, url: str, json: dict[str, Any], timeout: float) -> httpx.Response:
            nonlocal captured_json
            _ = timeout
            captured_json = json
            request = httpx.Request("POST", f"https://api.anthropic.com{url}")
            return httpx.Response(
                200,
                request=request,
                json={
                    "content": [
                        {
                            "type": "text",
                            "text": '{"label":"foo","reasoning":"bar"}',
                        }
                    ],
                    "usage": {"input_tokens": 12, "output_tokens": 34},
                },
            )

        async def aclose(self) -> None:
            return None

    provider._client = DummyClient()  # pyright: ignore[reportAttributeAccessIssue]

    response = asyncio.run(
        provider.chat(
            prompt="prompt",
            max_tokens=100,
            response_schema={"type": "object"},
            timeout_ms=1000,
        )
    )

    assert captured_json["thinking"] == {"type": "adaptive"}
    assert captured_json["output_config"] == {
        "format": {
            "type": "json_schema",
            "schema": {"type": "object", "additionalProperties": False},
        },
        "effort": "medium",
    }
    assert response.content == '{"label":"foo","reasoning":"bar"}'


def test_anthropic_provider_uses_manual_thinking_for_haiku() -> None:
    provider = AnthropicProvider(
        api_key="test-key",
        config=AnthropicHaiku45LLMConfig(thinking_budget=1024),
    )

    captured_json: dict[str, Any] = {}

    class DummyClient:
        async def post(self, url: str, json: dict[str, Any], timeout: float) -> httpx.Response:
            nonlocal captured_json
            _ = timeout
            captured_json = json
            request = httpx.Request("POST", f"https://api.anthropic.com{url}")
            return httpx.Response(
                200,
                request=request,
                json={
                    "content": [
                        {
                            "type": "text",
                            "text": '{"label":"foo","reasoning":"bar"}',
                        }
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 2},
                },
            )

        async def aclose(self) -> None:
            return None

    provider._client = DummyClient()  # pyright: ignore[reportAttributeAccessIssue]

    asyncio.run(
        provider.chat(
            prompt="prompt",
            max_tokens=100,
            response_schema={"type": "object"},
            timeout_ms=1000,
        )
    )

    assert captured_json["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert captured_json["output_config"] == {
        "format": {
            "type": "json_schema",
            "schema": {"type": "object", "additionalProperties": False},
        }
    }
