"""LLM provider implementations for multi-provider autointerp.

Provider routing: model string determines which API to call.
  - Contains "/" → OpenRouter (e.g. "google/gemini-3.1-pro-preview", "anthropic/claude-sonnet-4")
  - Starts with "claude-" → first-party Anthropic API
  - Starts with "gpt-", "o1-", "o3-", "o4-", "chatgpt-" → first-party OpenAI API
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, override

import httpx

from spd.log import logger

ReasoningEffort = Literal["none", "low", "medium", "high"]

ProviderName = Literal["openrouter", "anthropic", "openai"]

_PROVIDER_ENV_VARS: dict[ProviderName, str] = {
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

# Per-token pricing (input, output). Approximate — used for cost tracking, not billing.
_ANTHROPIC_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-20250514": (3.0 / 1_000_000, 15.0 / 1_000_000),
    "claude-opus-4-20250514": (15.0 / 1_000_000, 75.0 / 1_000_000),
    "claude-haiku-4-5-20251001": (0.80 / 1_000_000, 4.0 / 1_000_000),
}

_OPENAI_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50 / 1_000_000, 10.0 / 1_000_000),
    "gpt-4o-mini": (0.15 / 1_000_000, 0.60 / 1_000_000),
    "gpt-4.1": (2.0 / 1_000_000, 8.0 / 1_000_000),
    "gpt-4.1-mini": (0.40 / 1_000_000, 1.60 / 1_000_000),
    "gpt-4.1-nano": (0.10 / 1_000_000, 0.40 / 1_000_000),
    "o3": (2.0 / 1_000_000, 8.0 / 1_000_000),
    "o3-mini": (1.10 / 1_000_000, 4.40 / 1_000_000),
    "o4-mini": (1.10 / 1_000_000, 4.40 / 1_000_000),
}


def infer_provider(model: str) -> ProviderName:
    if "/" in model:
        return "openrouter"
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith(("gpt-", "o1-", "o3-", "o4-", "chatgpt-")):
        return "openai"
    raise ValueError(
        f"Cannot infer provider for model '{model}'. "
        "Use 'vendor/model' for OpenRouter, 'claude-*' for Anthropic, 'gpt-*'/'o*-*' for OpenAI."
    )


def get_api_key_for_model(model: str) -> str:
    provider = infer_provider(model)
    env_var = _PROVIDER_ENV_VARS[provider]
    key = os.environ.get(env_var)
    assert key, f"{env_var} not set (required for model '{model}')"
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
    def __init__(self, api_key: str, model: str, reasoning_effort: ReasoningEffort):
        self.model = model
        self._reasoning_effort = reasoning_effort
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
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
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


def create_provider(model: str, reasoning_effort: ReasoningEffort) -> LLMProvider:
    """Create a provider from model string, auto-resolving the API key from env."""
    api_key = get_api_key_for_model(model)
    match infer_provider(model):
        case "openrouter":
            return OpenRouterProvider(api_key, model, reasoning_effort)
        case "anthropic":
            return AnthropicProvider(api_key, model, reasoning_effort)
        case "openai":
            return OpenAIProvider(api_key, model, reasoning_effort)
