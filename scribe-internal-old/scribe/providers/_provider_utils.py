"""AI CLI providers for Scribe.

This module provides a unified interface for different AI CLI tools (Claude, Gemini)
while preserving provider-specific functionality and behavior.
"""

from typing import List, Dict, Type

from .base import AICLIProvider
from .claude import ClaudeProvider
from .gemini import GeminiProvider
from .codex import CodexProvider


# Registry of available providers
PROVIDER_REGISTRY: Dict[str, Type[AICLIProvider]] = {
    "claude": ClaudeProvider,
    "gemini": GeminiProvider,
    "codex": CodexProvider,
}


def detect_available_providers() -> List[str]:
    """Detect which AI CLI providers are available on the system."""
    available = []

    for name, provider_class in PROVIDER_REGISTRY.items():
        try:
            provider = provider_class()
            if provider.is_available():
                available.append(name)
        except Exception:
            # Skip providers that fail to instantiate
            pass

    return available


def get_provider(provider_name: str) -> AICLIProvider:
    provider_name = provider_name.lower()

    if provider_name not in PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown AI provider: {provider_name}. Available: {list(PROVIDER_REGISTRY.keys())}"
        )

    provider = PROVIDER_REGISTRY[provider_name]()

    if not provider.is_available():
        available = detect_available_providers()
        raise ValueError(
            f"AI provider '{provider_name}' is not available. "
            f"Available providers: {available}. "
            f"Please install the required CLI tool."
        )

    return provider


def list_providers() -> List[str]:
    """List all registered providers (available or not)."""
    return list(PROVIDER_REGISTRY.keys())
