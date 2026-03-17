"""LLM provider abstraction for multi-provider autointerp.

LLMConfig discriminated union determines which API to call:
  - OpenRouterLLMConfig → OpenRouter API (any model via vendor/model ID)
  - AnthropicLLMConfig → first-party Anthropic API (tool_use for structured output)
  - OpenAILLMConfig → first-party OpenAI API (json_schema response format)
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated, Any, Literal, override

import httpx
from pydantic import Field

from spd.base_config import BaseConfig
from spd.log import logger

ReasoningEffort = Literal["none", "low", "medium", "high"]

ProviderName = Literal["openrouter", "anthropic", "openai"]

# ---------------------------------------------------------------------------
# LLM config (discriminated union)
# ---------------------------------------------------------------------------


class OpenRouterLLMConfig(BaseConfig):
    type: Literal["openrouter"] = "openrouter"
    model: str = "google/gemini-3-flash-preview"
    reasoning_effort: ReasoningEffort = "low"


class AnthropicLLMConfig(BaseConfig):
    type: Literal["anthropic"] = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    thinking_budget: int | None = None


class OpenAILLMConfig(BaseConfig):
    type: Literal["openai"] = "openai"
    model: str
    reasoning_effort: ReasoningEffort = "none"


LLMConfig = Annotated[
    OpenRouterLLMConfig | AnthropicLLMConfig | OpenAILLMConfig,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Provider internals
# ---------------------------------------------------------------------------

_PROVIDER_ENV_VARS: dict[ProviderName, str] = {
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

_ANTHROPIC_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-20250514": (3.0 / 1_000_000, 15.0 / 1_000_000),
    "claude-opus-4-20250514": (15.0 / 1_000_000, 75.0 / 1_000_000),
    "claude-haiku-4-5-20251001": (0.80 / 1_000_000, 4.0 / 1_000_000),
}

_OPENAI_PRICING: dict[str, tuple[float, float]] = {
    # Add GPT-5 series pricing here when available
}


def _get_api_key(provider_name: ProviderName) -> str:
    env_var = _PROVIDER_ENV_VARS[provider_name]
    key = os.environ.get(env_var)
    assert key, f"{env_var} not set"
    return key


@dataclass
class ChatResponse:
    content: str
    input_tokens: int
    output_tokens: int


class RetryableAPIError(Exception):
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


def _parse_retry_after_header(resp: httpx.Response) -> float | None:
    val = resp.headers.get("retry-after")
    if val is None:
        return None
    try:
        return float(val)
    except ValueError:
        return None


class LLMProvider(ABC):
    @abstractmethod
    async def chat(
        self,
        prompt: str,
        max_tokens: int,
        response_schema: dict[str, Any],
        timeout_ms: int,
    ) -> ChatResponse: ...

    @abstractmethod
    async def get_pricing(self) -> tuple[float, float]:
        """Returns (input_price_per_token, output_price_per_token)."""

    @abstractmethod
    async def close(self) -> None: ...


class OpenRouterProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, reasoning_effort: ReasoningEffort):
        self.model = model
        self._reasoning_effort = reasoning_effort
        self._client = httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1",
            headers={"Authorization": f"Bearer {api_key}"},
        )

    @override
    async def chat(
        self,
        prompt: str,
        max_tokens: int,
        response_schema: dict[str, Any],
        timeout_ms: int,
    ) -> ChatResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {**response_schema, "additionalProperties": False},
                    "strict": True,
                },
            },
        }
        if self._reasoning_effort != "none":
            body["reasoning"] = {"effort": self._reasoning_effort}

        try:
            resp = await self._client.post(
                "/chat/completions", json=body, timeout=timeout_ms / 1000
            )
        except httpx.TransportError as e:
            raise RetryableAPIError(str(e)) from e

        if resp.status_code in (429, 502, 503, 504, 500, 408):
            retry_after = _parse_retry_after_header(resp)
            raise RetryableAPIError(
                f"HTTP {resp.status_code}: {resp.text[:200]}", retry_after=retry_after
            )

        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            msg = data["error"].get("message", str(data["error"]))
            raise RetryableAPIError(msg)

        choice = data["choices"][0]
        content = choice["message"]["content"]
        assert isinstance(content, str)
        usage = data["usage"]

        if choice.get("finish_reason") == "length":
            logger.warning(f"Response truncated at {max_tokens} tokens")

        return ChatResponse(
            content=content,
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
        )

    @override
    async def get_pricing(self) -> tuple[float, float]:
        resp = await self._client.get("/models")
        resp.raise_for_status()
        for model in resp.json()["data"]:
            if model["id"] == self.model:
                return float(model["pricing"]["prompt"]), float(model["pricing"]["completion"])
        raise ValueError(f"Model {self.model} not found on OpenRouter")

    @override
    async def close(self) -> None:
        await self._client.aclose()


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, thinking_budget: int | None):
        self.model = model
        self._thinking_budget = thinking_budget
        self._client = httpx.AsyncClient(
            base_url="https://api.anthropic.com",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )

    @override
    async def chat(
        self,
        prompt: str,
        max_tokens: int,
        response_schema: dict[str, Any],
        timeout_ms: int,
    ) -> ChatResponse:
        effective_max_tokens = max_tokens
        if self._thinking_budget is not None:
            effective_max_tokens = max_tokens + self._thinking_budget

        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": effective_max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [
                {
                    "name": "respond",
                    "description": "Respond with the structured output",
                    "input_schema": response_schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": "respond"},
        }
        if self._thinking_budget is not None:
            body["thinking"] = {"type": "enabled", "budget_tokens": self._thinking_budget}

        try:
            resp = await self._client.post("/v1/messages", json=body, timeout=timeout_ms / 1000)
        except httpx.TransportError as e:
            raise RetryableAPIError(str(e)) from e

        if resp.status_code in (429, 500, 502, 503, 529):
            retry_after = _parse_retry_after_header(resp)
            raise RetryableAPIError(
                f"HTTP {resp.status_code}: {resp.text[:200]}", retry_after=retry_after
            )

        resp.raise_for_status()
        data = resp.json()

        tool_input = None
        for block in data["content"]:
            if block["type"] == "tool_use" and block["name"] == "respond":
                tool_input = block["input"]
                break
        assert tool_input is not None, f"No tool_use response in: {data['content']}"

        content = json.dumps(tool_input)
        usage = data["usage"]
        return ChatResponse(
            content=content,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
        )

    @override
    async def get_pricing(self) -> tuple[float, float]:
        return _ANTHROPIC_PRICING.get(self.model, (3.0 / 1_000_000, 15.0 / 1_000_000))

    @override
    async def close(self) -> None:
        await self._client.aclose()


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, reasoning_effort: ReasoningEffort):
        self.model = model
        self._reasoning_effort = reasoning_effort
        self._client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {api_key}"},
        )

    @override
    async def chat(
        self,
        prompt: str,
        max_tokens: int,
        response_schema: dict[str, Any],
        timeout_ms: int,
    ) -> ChatResponse:
        is_reasoning_model = self.model.startswith(("o1-", "o3-", "o4-"))
        token_key = "max_completion_tokens" if is_reasoning_model else "max_tokens"
        body: dict[str, Any] = {
            "model": self.model,
            token_key: max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {**response_schema, "additionalProperties": False},
                    "strict": True,
                },
            },
        }
        if self._reasoning_effort != "none" and is_reasoning_model:
            body["reasoning_effort"] = self._reasoning_effort

        try:
            resp = await self._client.post(
                "/chat/completions", json=body, timeout=timeout_ms / 1000
            )
        except httpx.TransportError as e:
            raise RetryableAPIError(str(e)) from e

        if resp.status_code in (429, 500, 502, 503):
            retry_after = _parse_retry_after_header(resp)
            raise RetryableAPIError(
                f"HTTP {resp.status_code}: {resp.text[:200]}", retry_after=retry_after
            )

        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        content = choice["message"]["content"]
        assert isinstance(content, str)
        usage = data["usage"]

        if choice.get("finish_reason") == "length":
            logger.warning(f"Response truncated at {max_tokens} tokens")

        return ChatResponse(
            content=content,
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
        )

    @override
    async def get_pricing(self) -> tuple[float, float]:
        return _OPENAI_PRICING.get(self.model, (5.0 / 1_000_000, 15.0 / 1_000_000))

    @override
    async def close(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_provider(
    config: OpenRouterLLMConfig | AnthropicLLMConfig | OpenAILLMConfig,
) -> LLMProvider:
    match config:
        case OpenRouterLLMConfig():
            api_key = _get_api_key("openrouter")
            return OpenRouterProvider(api_key, config.model, config.reasoning_effort)
        case AnthropicLLMConfig():
            api_key = _get_api_key("anthropic")
            return AnthropicProvider(api_key, config.model, config.thinking_budget)
        case OpenAILLMConfig():
            api_key = _get_api_key("openai")
            return OpenAIProvider(api_key, config.model, config.reasoning_effort)
