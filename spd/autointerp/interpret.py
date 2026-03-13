import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import httpx
from openrouter import OpenRouter
from openrouter.components import JSONSchemaConfig, MessageTypedDict, ResponseFormatJSONSchema
from openrouter.errors import (
    BadGatewayResponseError,
    ChatError,
    EdgeNetworkTimeoutResponseError,
    ProviderOverloadedResponseError,
    RequestTimeoutResponseError,
    ServiceUnavailableResponseError,
    TooManyRequestsResponseError,
)
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.app.backend.compute import get_model_n_blocks
from spd.autointerp.prompt_template import INTERPRETATION_SCHEMA, format_prompt_template
from spd.autointerp.schemas import ArchitectureInfo, InterpretationResult
from spd.configs import LMTaskConfig
from spd.harvest.analysis import TokenPRLift, get_input_token_stats, get_output_token_stats
from spd.harvest.harvest import HarvestResult
from spd.harvest.schemas import ComponentData
from spd.harvest.storage import TokenStatsStorage
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo

# Retry config
MAX_RETRIES = 8
BASE_DELAY_S = 0.5
MAX_DELAY_S = 60.0
JITTER_FACTOR = 0.5
MAX_CONCURRENT_REQUESTS = 50
MAX_REQUESTS_PER_MINUTE = 300  # Gemini flash has 400 RPM limit


class RateLimiter:
    """Sliding window rate limiter for async code."""

    def __init__(self, max_requests: int, period_seconds: float):
        self.max_requests = max_requests
        self.period = period_seconds
        self.timestamps: list[float] = []
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self.lock:
            now = time.monotonic()
            self.timestamps = [t for t in self.timestamps if now - t < self.period]

            if len(self.timestamps) >= self.max_requests:
                sleep_time = self.timestamps[0] + self.period - now
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                self.timestamps = self.timestamps[1:]

            self.timestamps.append(time.monotonic())


RETRYABLE_ERRORS = (
    TooManyRequestsResponseError,
    ProviderOverloadedResponseError,
    ServiceUnavailableResponseError,
    BadGatewayResponseError,
    RequestTimeoutResponseError,
    EdgeNetworkTimeoutResponseError,
    ChatError,
    httpx.TransportError,  # Low-level network errors (ReadError, ConnectError, etc.)
)


class OpenRouterModelName(StrEnum):
    GEMINI_3_FLASH_PREVIEW = "google/gemini-3-flash-preview"


@dataclass
class CostTracker:
    input_tokens: int = 0
    output_tokens: int = 0
    input_price_per_token: float = 0.0
    output_price_per_token: float = 0.0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def cost_usd(self) -> float:
        return (
            self.input_tokens * self.input_price_per_token
            + self.output_tokens * self.output_price_per_token
        )


async def chat_with_retry(
    client: OpenRouter,
    model: str,
    messages: list[MessageTypedDict],
    response_format: ResponseFormatJSONSchema,
    max_tokens: int,
    context_label: str,
) -> tuple[str, int, int]:
    """Send chat request with exponential backoff retry. Returns (content, input_tokens, output_tokens)."""
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.chat.send_async(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
                response_format=response_format,
            )
            choice = response.choices[0]
            message = choice.message
            assert isinstance(message.content, str)
            assert response.usage is not None

            if choice.finish_reason == "length":
                logger.warning(f"{context_label}: Response truncated at {max_tokens} tokens")

            return (
                message.content,
                int(response.usage.prompt_tokens),
                int(response.usage.completion_tokens),
            )
        except RETRYABLE_ERRORS as e:
            last_error = e
            if attempt == MAX_RETRIES - 1:
                break

            delay = min(BASE_DELAY_S * (2**attempt), MAX_DELAY_S)
            jitter = delay * JITTER_FACTOR * random.random()
            total_delay = delay + jitter

            tqdm_asyncio.write(
                f"[retry {attempt + 1}/{MAX_RETRIES}] ({context_label}) "
                f"{type(e).__name__}, backing off {total_delay:.1f}s"
            )
            await asyncio.sleep(total_delay)

    assert last_error is not None
    raise RuntimeError(f"Max retries exceeded for {context_label}: {last_error}")


async def get_model_pricing(client: OpenRouter, model_id: str) -> tuple[float, float]:
    """Returns (input_price, output_price) per token."""
    response = await client.models.list_async()
    for model in response.data:
        if model.id == model_id:
            return float(model.pricing.prompt), float(model.pricing.completion)
    raise ValueError(f"Model {model_id} not found")


async def interpret_component(
    client: OpenRouter,
    model: str,
    component: ComponentData,
    arch: ArchitectureInfo,
    tokenizer: PreTrainedTokenizerBase,
    input_token_stats: TokenPRLift,
    output_token_stats: TokenPRLift,
) -> tuple[InterpretationResult, int, int] | None:
    """Returns (result, input_tokens, output_tokens), or None on failure."""
    prompt = format_prompt_template(
        component=component,
        arch=arch,
        tokenizer=tokenizer,
        input_token_stats=input_token_stats,
        output_token_stats=output_token_stats,
    )

    try:
        raw, in_tok, out_tok = await chat_with_retry(
            client=client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format=ResponseFormatJSONSchema(
                json_schema=JSONSchemaConfig(
                    name="interpretation",
                    schema_={**INTERPRETATION_SCHEMA, "additionalProperties": False},
                    strict=True,
                )
            ),
            max_tokens=1500,
            context_label=component.component_key,
        )
    except RuntimeError as e:
        logger.error(str(e))
        return None

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON: `{raw}`")
        return None

    assert len(parsed) == 3, f"Expected 3 fields, got {len(parsed)}"
    label = parsed["label"]
    confidence = parsed["confidence"]
    reasoning = parsed["reasoning"]
    assert isinstance(label, str) and isinstance(confidence, str) and isinstance(reasoning, str)

    return (
        InterpretationResult(
            component_key=component.component_key,
            label=label,
            confidence=confidence,
            reasoning=reasoning,
            raw_response=raw,
            prompt=prompt,
        ),
        in_tok,
        out_tok,
    )


async def interpret_all(
    components: list[ComponentData],
    arch: ArchitectureInfo,
    openrouter_api_key: str,
    interpreter_model: str,
    output_path: Path,
    token_stats: TokenStatsStorage,
    limit: int | None = None,
) -> list[InterpretationResult]:
    """Interpret all components with maximum parallelism. Rate limits handled via exponential backoff."""
    results: list[InterpretationResult] = []
    completed = set[str]()

    if output_path.exists():
        print(f"Resuming: {output_path} exists")
        with open(output_path) as f:
            for line in f:
                data = json.loads(line)
                results.append(InterpretationResult(**data))
                completed.add(data["component_key"])
        print(f"Resuming: {len(completed)} already completed")

    components_sorted = sorted(components, key=lambda c: c.mean_ci, reverse=True)
    remaining = [c for c in components_sorted if c.component_key not in completed]
    if limit is not None:
        remaining = remaining[:limit]
    print(f"Interpreting {len(remaining)} components")
    start_idx = len(results)

    output_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE, period_seconds=60.0)

    tokenizer = AutoTokenizer.from_pretrained(arch.tokenizer_name)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)

    async def process_one(
        component: ComponentData,
        index: int,
        client: OpenRouter,
        cost_tracker: CostTracker,
    ) -> None:
        await rate_limiter.acquire()
        async with semaphore:
            try:
                # Compute token stats for this component
                input_stats = get_input_token_stats(
                    token_stats, component.component_key, tokenizer, top_k=20
                )
                output_stats = get_output_token_stats(
                    token_stats, component.component_key, tokenizer, top_k=50
                )
                assert input_stats is not None, (
                    f"No input token stats for {component.component_key}"
                )
                assert output_stats is not None, (
                    f"No output token stats for {component.component_key}"
                )

                res = await interpret_component(
                    client=client,
                    model=interpreter_model,
                    component=component,
                    arch=arch,
                    tokenizer=tokenizer,
                    input_token_stats=input_stats,
                    output_token_stats=output_stats,
                )
                if res is None:
                    logger.error(f"Failed to interpret {component.component_key}")
                    return
                result, in_tok, out_tok = res

                async with output_lock:
                    results.append(result)
                    cost_tracker.add(in_tok, out_tok)
                    line = json.dumps(asdict(result)) + "\n"
                    log_progress = index % 100 == 0
                    progress_msg = (
                        f"[{index}] ${cost_tracker.cost_usd():.2f} ({cost_tracker.input_tokens:,} in, {cost_tracker.output_tokens:,} out)"
                        if log_progress
                        else ""
                    )
                with open(output_path, "a") as f:
                    f.write(line)

                if log_progress:
                    tqdm_asyncio.write(progress_msg)
            except Exception as e:
                logger.error(f"Fatal error on {component.component_key}: {type(e).__name__}: {e}")
                raise

    async with OpenRouter(api_key=openrouter_api_key) as client:
        input_price, output_price = await get_model_pricing(client, interpreter_model)
        cost_tracker = CostTracker(
            input_price_per_token=input_price, output_price_per_token=output_price
        )
        print(f"Pricing: ${input_price * 1e6:.2f}/M input, ${output_price * 1e6:.2f}/M output")

        await tqdm_asyncio.gather(
            *[
                process_one(c, i, client, cost_tracker)
                for i, c in enumerate(remaining, start=start_idx)
            ],
            desc="Interpreting",
        )

    print(f"Final cost: ${cost_tracker.cost_usd():.2f}")
    return results


def get_architecture_info(wandb_path: str, activation_contexts_dir: Path) -> ArchitectureInfo:
    run_info = SPDRunInfo.from_path(wandb_path)
    model = ComponentModel.from_run_info(run_info)
    n_blocks = get_model_n_blocks(model.target_model)
    config = run_info.config
    task_config = config.task_config
    assert isinstance(task_config, LMTaskConfig)
    assert config.tokenizer_name is not None

    harvest_config_path = activation_contexts_dir / "config.json"
    assert harvest_config_path.exists(), f"harvest config not found: {harvest_config_path}"
    harvest_config = json.loads(harvest_config_path.read_text())

    return ArchitectureInfo(
        n_blocks=n_blocks,
        c_per_layer=model.module_to_c,
        model_class=config.pretrained_model_class,
        dataset_name=task_config.dataset_name,
        tokenizer_name=config.tokenizer_name,
        seq_len=task_config.max_seq_len,
        context_tokens_per_side=harvest_config["activation_context_tokens_per_side"],
    )


def run_interpret(
    wandb_path: str,
    openrouter_api_key: str,
    interpreter_model: str,
    activation_contexts_dir: Path,
    correlations_dir: Path,
    autointerp_dir: Path,
    limit: int | None = None,
) -> list[InterpretationResult]:
    arch = get_architecture_info(wandb_path, activation_contexts_dir)
    components = HarvestResult.load_components(activation_contexts_dir)
    output_path = autointerp_dir / "results.jsonl"

    # Load token stats
    token_stats_path = correlations_dir / "token_stats.pt"
    assert token_stats_path.exists(), (
        f"token_stats.pt not found at {token_stats_path}. Run harvest first."
    )
    token_stats = TokenStatsStorage.load(token_stats_path)

    results = asyncio.run(
        interpret_all(
            components=components,
            arch=arch,
            openrouter_api_key=openrouter_api_key,
            interpreter_model=interpreter_model,
            output_path=output_path,
            token_stats=token_stats,
            limit=limit,
        )
    )

    print(f"Completed {len(results)} interpretations -> {output_path}")
    return results
