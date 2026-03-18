"""LLM API utilities: batch concurrent calls with rate limiting, retry, and cost tracking."""

import asyncio
import contextlib
import json
import random
import time
from collections.abc import AsyncGenerator, Iterable, Sized
from dataclasses import dataclass, field
from typing import Any

from aiolimiter import AsyncLimiter

from spd.autointerp.providers import LLMProvider, RetryableAPIError
from spd.log import logger

_MAX_RETRIES = 8
_BASE_DELAY_S = 0.5
_MAX_DELAY_S = 60.0
_JITTER_FACTOR = 0.5
_REQUEST_TIMEOUT_MS = 120_000
_JSON_PARSE_RETRIES = 3
_MAX_BACKOFF_S = 600.0


@dataclass
class LLMJob:
    prompt: str
    key: str


@dataclass
class LLMResult:
    job: LLMJob
    parsed: dict[str, Any]
    raw: str


@dataclass
class LLMError:
    job: LLMJob
    error: Exception


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass
class CostTracker:
    input_tokens: int = 0
    output_tokens: int = 0
    input_price_per_token: float = 0.0
    output_price_per_token: float = 0.0
    limit_usd: float | None = None
    _budget_exceeded: asyncio.Event = field(default_factory=asyncio.Event)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def add(self, input_tokens: int, output_tokens: int) -> None:
        async with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            if self.limit_usd is not None and self.cost_usd() >= self.limit_usd:
                self._budget_exceeded.set()

    def over_budget(self) -> bool:
        return self._budget_exceeded.is_set()

    def cost_usd(self) -> float:
        return (
            self.input_tokens * self.input_price_per_token
            + self.output_tokens * self.output_price_per_token
        )


class _BudgetExceededError(Exception):
    pass


class _GlobalBackoff:
    """Shared backoff that pauses all coroutines when the API pushes back."""

    def __init__(self) -> None:
        self._resume_at = 0.0
        self._lock = asyncio.Lock()

    async def set_backoff(self, seconds: float) -> None:
        assert seconds <= _MAX_BACKOFF_S, (
            f"Server requested {seconds:.0f}s backoff, exceeds {_MAX_BACKOFF_S:.0f}s cap"
        )
        async with self._lock:
            self._resume_at = max(self._resume_at, time.monotonic() + seconds)

    async def wait(self) -> None:
        delay = self._resume_at - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def map_llm_calls(
    provider: LLMProvider,
    jobs: Iterable[LLMJob],
    max_tokens: int,
    max_concurrent: int,
    max_requests_per_minute: int,
    cost_limit_usd: float | None,
    response_schema: dict[str, Any],
    n_total: int | None = None,
    cost_tracker: CostTracker | None = None,
) -> AsyncGenerator[LLMResult | LLMError]:
    """Fan out LLM calls concurrently, yielding results as they complete.

    Handles rate limiting, retry with exponential backoff, JSON parsing,
    cost tracking, and progress logging. Yields LLMResult on success,
    LLMError on failure. Silently stops remaining jobs on budget exceeded.

    Jobs can be a lazy iterable (e.g. a generator). Prompt building in the
    generator body naturally interleaves with async HTTP calls.

    Pass a shared CostTracker to accumulate costs across multiple calls.
    """
    if n_total is None and isinstance(jobs, Sized):
        n_total = len(jobs)

    assert not (cost_tracker is not None and cost_limit_usd is not None), (
        "Pass cost_limit_usd or cost_tracker, not both"
    )

    input_price, output_price = await provider.get_pricing()
    if cost_tracker is not None:
        cost = cost_tracker
        cost.input_price_per_token = input_price
        cost.output_price_per_token = output_price
    else:
        cost = CostTracker(
            input_price_per_token=input_price,
            output_price_per_token=output_price,
            limit_usd=cost_limit_usd,
        )
    rate_limiter = AsyncLimiter(max_rate=max_requests_per_minute, time_period=60)
    backoff = _GlobalBackoff()

    async def chat(prompt: str, context_label: str) -> str:
        if cost.over_budget():
            raise _BudgetExceededError(f"${cost.cost_usd():.2f}")

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            await backoff.wait()
            async with rate_limiter:
                try:
                    response = await provider.chat(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        response_schema=response_schema,
                        timeout_ms=_REQUEST_TIMEOUT_MS,
                    )
                    await cost.add(response.input_tokens, response.output_tokens)
                    return response.content
                except RetryableAPIError as e:
                    last_error = e
                    if attempt == _MAX_RETRIES - 1:
                        break

                    if e.retry_after is not None:
                        await backoff.set_backoff(e.retry_after)
                        delay = e.retry_after
                    else:
                        delay = min(_BASE_DELAY_S * (2**attempt), _MAX_DELAY_S)
                        jitter = delay * _JITTER_FACTOR * random.random()
                        delay = delay + jitter

                    logger.warning(
                        f"[retry {attempt + 1}/{_MAX_RETRIES}] ({context_label}) "
                        f"{type(e).__name__}: {e}, backing off {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)

        assert last_error is not None
        raise RuntimeError(f"Max retries exceeded for {context_label}: {last_error}")

    queue: asyncio.Queue[LLMResult | LLMError | None] = asyncio.Queue()

    n_done = 0
    budget_exceeded = False

    async def process_one(job: LLMJob) -> None:
        nonlocal n_done, budget_exceeded
        if budget_exceeded:
            return

        try:
            raw = ""
            parsed = None
            for attempt in range(_JSON_PARSE_RETRIES):
                raw = await chat(job.prompt, job.key)
                try:
                    parsed = json.loads(raw)
                    break
                except json.JSONDecodeError:
                    if attempt == _JSON_PARSE_RETRIES - 1:
                        raise
                    logger.warning(
                        f"{job.key}: invalid JSON "
                        f"(attempt {attempt + 1}/{_JSON_PARSE_RETRIES}), retrying"
                    )
            assert parsed is not None
            await queue.put(LLMResult(job=job, parsed=parsed, raw=raw))
        except _BudgetExceededError:
            budget_exceeded = True
            return
        except Exception as e:
            await queue.put(LLMError(job=job, error=e))

        n_done += 1
        total_str = f"/{n_total}" if n_total is not None else ""
        logger.info(
            f"[{n_done}{total_str}] ${cost.cost_usd():.2f} "
            f"({cost.input_tokens:,} in, {cost.output_tokens:,} out)"
        )

    async def run_all() -> None:
        job_queue: asyncio.Queue[LLMJob | None] = asyncio.Queue(maxsize=max_concurrent)

        async def worker() -> None:
            while (job := await job_queue.get()) is not None:
                await process_one(job)

        workers = [asyncio.create_task(worker()) for _ in range(max_concurrent)]
        try:
            for n_queued, job in enumerate(jobs, 1):
                if budget_exceeded:
                    break
                await job_queue.put(job)
                if n_queued % 500 == 0:
                    logger.info(f"Queued {n_queued} jobs")
            for _ in workers:
                await job_queue.put(None)
            await asyncio.gather(*workers)
        finally:
            await queue.put(None)

    task = asyncio.create_task(run_all())
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
    finally:
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        logger.info(
            f"Final cost: ${cost.cost_usd():.2f} "
            f"({cost.input_tokens:,} in, {cost.output_tokens:,} out)"
        )
