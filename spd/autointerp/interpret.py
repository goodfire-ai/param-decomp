import asyncio
import json
from collections.abc import Iterable
from pathlib import Path

from openrouter import OpenRouter
from openrouter.components import Effort, Reasoning

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.autointerp.config import StrategyConfig
from spd.autointerp.db import InterpDB
from spd.autointerp.llm_api import (
    LLMError,
    LLMJob,
    LLMResult,
    make_response_format,
    map_llm_calls,
)
from spd.autointerp.schemas import InterpretationResult, ModelMetadata
from spd.autointerp.strategies.dispatch import INTERPRETATION_SCHEMA, format_prompt
from spd.harvest.analysis import TokenPRLift, get_input_token_stats, get_output_token_stats
from spd.harvest.repo import HarvestRepo
from spd.harvest.schemas import ComponentData
from spd.log import logger

MAX_CONCURRENT = 50


async def interpret_component(
    api: OpenRouter,
    model: str,
    reasoning_effort: Effort,
    strategy: StrategyConfig,
    component: ComponentData,
    model_metadata: ModelMetadata,
    app_tok: AppTokenizer,
    input_token_stats: TokenPRLift,
    output_token_stats: TokenPRLift,
    context_tokens_per_side: int,
) -> InterpretationResult:
    """Interpret a single component. Used by the app for on-demand interpretation."""
    prompt = format_prompt(
        strategy=strategy,
        component=component,
        model_metadata=model_metadata,
        app_tok=app_tok,
        input_token_stats=input_token_stats,
        output_token_stats=output_token_stats,
        context_tokens_per_side=context_tokens_per_side,
    )

    schema = INTERPRETATION_SCHEMA
    response_format = make_response_format("interpretation", schema)

    response = await api.chat.send_async(
        model=model,
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
        response_format=response_format,
        reasoning=Reasoning(effort=reasoning_effort),
    )

    choice = response.choices[0]
    assert isinstance(choice.message.content, str)
    raw = choice.message.content
    parsed = json.loads(raw)

    assert len(parsed) == 3, f"Expected 3 fields, got {parsed}"
    label = parsed["label"]
    confidence = parsed["confidence"]
    reasoning_text = parsed["reasoning"]
    assert (
        isinstance(label, str) and isinstance(confidence, str) and isinstance(reasoning_text, str)
    )

    return InterpretationResult(
        component_key=component.component_key,
        label=label,
        confidence=confidence,
        reasoning=reasoning_text,
        raw_response=raw,
        prompt=prompt,
    )


def run_interpret(
    openrouter_api_key: str,
    model: str,
    reasoning_effort: Effort,
    limit: int | None,
    cost_limit_usd: float | None,
    max_requests_per_minute: int,
    max_concurrent: int,
    model_metadata: ModelMetadata,
    template_strategy: StrategyConfig,
    harvest: HarvestRepo,
    db_path: Path,
    tokenizer_name: str,
) -> list[InterpretationResult]:
    summary = harvest.get_summary()
    logger.info(f"Loaded summary for {len(summary)} components")

    token_stats = harvest.get_token_stats()
    assert token_stats is not None, "token_stats.pt not found. Run harvest first."

    harvest_config = harvest.get_config()
    raw = harvest_config["activation_context_tokens_per_side"]
    assert isinstance(raw, int), f"expected int, got {type(raw)}"
    context_tokens_per_side = raw

    app_tok = AppTokenizer.from_pretrained(tokenizer_name)

    eligible_keys = sorted(summary, key=lambda k: summary[k].firing_density, reverse=True)

    if limit is not None:
        eligible_keys = eligible_keys[:limit]

    async def _run() -> list[InterpretationResult]:
        db = InterpDB(db_path)

        try:
            completed = db.get_completed_keys()
            if completed:
                logger.info(f"Resuming: {len(completed)} already completed")

            remaining_keys = [k for k in eligible_keys if k not in completed]
            logger.info(f"Interpreting {len(remaining_keys)} components")

            schema = INTERPRETATION_SCHEMA

            def build_jobs() -> Iterable[LLMJob]:
                for key in remaining_keys:
                    component = harvest.get_component(key)
                    assert component is not None, f"Component {key} not found in harvest"
                    input_stats = get_input_token_stats(token_stats, key, app_tok, top_k=20)
                    output_stats = get_output_token_stats(token_stats, key, app_tok, top_k=50)
                    assert input_stats is not None
                    assert output_stats is not None
                    prompt = format_prompt(
                        strategy=template_strategy,
                        component=component,
                        model_metadata=model_metadata,
                        app_tok=app_tok,
                        input_token_stats=input_stats,
                        output_token_stats=output_stats,
                        context_tokens_per_side=context_tokens_per_side,
                    )
                    yield LLMJob(prompt=prompt, schema=schema, key=key)

            results: list[InterpretationResult] = []
            n_errors = 0

            async for outcome in map_llm_calls(
                openrouter_api_key=openrouter_api_key,
                model=model,
                reasoning_effort=reasoning_effort,
                jobs=build_jobs(),
                max_tokens=8000,
                max_concurrent=max_concurrent,
                max_requests_per_minute=max_requests_per_minute,
                cost_limit_usd=cost_limit_usd,
                response_schema=schema,
                n_total=len(remaining_keys),
            ):
                match outcome:
                    case LLMResult(job=job, parsed=parsed, raw=raw):
                        assert len(parsed) == 3, f"Expected 3 fields, got {len(parsed)}"
                        label = parsed["label"]
                        confidence = parsed["confidence"]
                        reasoning_text = parsed["reasoning"]
                        assert (
                            isinstance(label, str)
                            and isinstance(confidence, str)
                            and isinstance(reasoning_text, str)
                        )
                        result = InterpretationResult(
                            component_key=job.key,
                            label=label,
                            confidence=confidence,
                            reasoning=reasoning_text,
                            raw_response=raw,
                            prompt=job.prompt,
                        )
                        results.append(result)
                        db.save_interpretation(result)
                    case LLMError(job=job, error=e):
                        n_errors += 1
                        logger.error(f"Skipping {job.key}: {type(e).__name__}: {e}")

                error_rate = n_errors / (n_errors + len(results))
                # 10 is a magic number - just trying to avoid low sample size causing this to false alarm
                if error_rate > 0.2 and n_errors > 10:
                    raise RuntimeError(
                        f"Error rate {error_rate:.0%} ({n_errors}/{len(remaining_keys)}) exceeds 20% threshold"
                    )

        finally:
            db.close()

        db.mark_done()
        logger.info(f"Completed {len(results)} interpretations -> {db_path}")
        return results

    return asyncio.run(_run())
