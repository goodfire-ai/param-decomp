"""
Model API utilities for calling OpenAI GPT-5
"""

import os
from typing import Optional
from openai import AsyncOpenAI


async def call_openai(
    prompt: str,
    model: str = "gpt-5",
    system_prompt: Optional[str] = None,
    api_key: Optional[str] = None,
    verbosity: Optional[str] = "low",
    reasoning_effort: Optional[str] = "medium",
) -> str:
    """
    Call GPT-5 model via OpenAI API using the responses.create endpoint.

    Args:
        prompt: User prompt
        system_prompt: Optional system prompt
        api_key: Optional API key (will use OPENAI_API_KEY env var if not provided)
        api_kwargs: Optional dictionary of additional arguments to pass to the API
    Returns:
        Model response as string
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
        )

    client = AsyncOpenAI(api_key=api_key)

    # Combine system prompt and user prompt into a single input
    if system_prompt:
        full_input = f"{system_prompt}\n\n{prompt}"
    else:
        full_input = prompt

    # Use responses.create as shown in the docs - only model and input are supported
    response = await client.responses.create(
        model=model,
        input=full_input,
        reasoning={"effort": reasoning_effort},
        text={"verbosity": verbosity},
    )

    return response.output_text
