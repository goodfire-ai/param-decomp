"""Main three-phase graph interpretation execution.

Structure:
    output_labels = scan(layers_reversed, step)
    input_labels  = scan(layers_forward,  step)
    unified       = map(output_labels + input_labels, unify)

Each scan folds over layers. Within a layer, components are labeled in parallel
via async LLM calls. The fold accumulator (labels_so_far) lets each component's
prompt include labels from previously-processed layers.
"""

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable
from pathlib import Path
from typing import Literal

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.autointerp.llm_api import CostTracker, LLMError, LLMJob, LLMResult, map_llm_calls
from spd.autointerp.schemas import ModelMetadata
from spd.dataset_attributions.storage import (
    AttrMetric,
    DatasetAttributionEntry,
    DatasetAttributionStorage,
)
from spd.graph_interp import graph_context
from spd.graph_interp.config import GraphInterpConfig
from spd.graph_interp.db import GraphInterpDB
from spd.graph_interp.graph_context import RelatedComponent, get_related_components
from spd.graph_interp.ordering import group_and_sort_by_layer
from spd.graph_interp.prompts import (
    LABEL_SCHEMA,
    format_input_prompt,
    format_output_prompt,
    format_unification_prompt,
)
from spd.graph_interp.schemas import LabelResult, PromptEdge
from spd.harvest.analysis import TokenPRLift, get_input_token_stats, get_output_token_stats
from spd.harvest.repo import HarvestRepo
from spd.harvest.schemas import ComponentData
from spd.harvest.storage import CorrelationStorage, TokenStatsStorage
from spd.log import logger

GetRelated = Callable[[str, dict[str, LabelResult]], list[RelatedComponent]]
Step = Callable[[list[str], dict[str, LabelResult]], Awaitable[dict[str, LabelResult]]]
MakePrompt = Callable[["ComponentData", "TokenPRLift", list[RelatedComponent]], str]


def run_graph_interp(
    openrouter_api_key: str,
    config: GraphInterpConfig,
    harvest: HarvestRepo,
    attribution_storage: DatasetAttributionStorage,
    correlation_storage: CorrelationStorage,
    token_stats: TokenStatsStorage,
    model_metadata: ModelMetadata,
    db_path: Path,
    tokenizer_name: str,
) -> None:
    logger.info("Loading tokenizer...")
    app_tok = AppTokenizer.from_pretrained(tokenizer_name)

    logger.info("Loading component summaries...")
    summaries = harvest.get_summary()
    alive = {k: s for k, s in summaries.items() if s.firing_density > 0.0}
    all_keys = sorted(alive, key=lambda k: alive[k].firing_density, reverse=True)
    if config.limit is not None:
        all_keys = all_keys[: config.limit]

    layers = group_and_sort_by_layer(all_keys, model_metadata.layer_descriptions)
    total = len(all_keys)
    logger.info(f"Graph interp: {total} components across {len(layers)} layers")

    # -- Injected behaviours ---------------------------------------------------

    shared_cost = CostTracker(limit_usd=config.cost_limit_usd)

    async def llm_map(
        jobs: Iterable[LLMJob], n_total: int | None = None
    ) -> AsyncGenerator[LLMResult | LLMError]:
        async for result in map_llm_calls(
            openrouter_api_key=openrouter_api_key,
            model=config.model,
            reasoning_effort=config.reasoning_effort,
            jobs=jobs,
            max_tokens=8000,
            max_concurrent=config.max_concurrent,
            max_requests_per_minute=config.max_requests_per_minute,
            cost_limit_usd=None,
            response_schema=LABEL_SCHEMA,
            n_total=n_total,
            cost_tracker=shared_cost,
        ):
            yield result

    concrete_to_canon = model_metadata.layer_descriptions
    canon_to_concrete = {v: k for k, v in concrete_to_canon.items()}

    def _translate_entries(entries: list[DatasetAttributionEntry]) -> list[DatasetAttributionEntry]:
        for e in entries:
            if e.layer in canon_to_concrete:
                e.layer = canon_to_concrete[e.layer]
                e.component_key = f"{e.layer}:{e.component_idx}"
        return entries

    def _to_canon(concrete_key: str) -> str:
        layer, idx = concrete_key.rsplit(":", 1)
        return f"{concrete_to_canon[layer]}:{idx}"

    def _make_get_attributed(
        method: Callable[..., list[DatasetAttributionEntry]], metric: AttrMetric
    ) -> "graph_context.GetAttributed":
        def get(
            key: str, k: int, sign: Literal["positive", "negative"]
        ) -> list[DatasetAttributionEntry]:
            return _translate_entries(method(_to_canon(key), k=k, sign=sign, metric=metric))

        return get

    def _get_related(get_attributed: "graph_context.GetAttributed") -> GetRelated:
        def get(key: str, labels_so_far: dict[str, LabelResult]) -> list[RelatedComponent]:
            return get_related_components(
                key,
                get_attributed,
                correlation_storage,
                labels_so_far,
                config.top_k_attributed,
            )

        return get

    # -- Layer processor (shared for output and input passes) --------------------

    def _make_process_layer(
        get_related: GetRelated,
        save_label: Callable[[LabelResult], None],
        pass_name: Literal["output", "input"],
        get_token_stats: Callable[[str], TokenPRLift | None],
        make_prompt: MakePrompt,
    ) -> Step:
        async def process(
            pending: list[str],
            labels_so_far: dict[str, LabelResult],
        ) -> dict[str, LabelResult]:
            def jobs() -> Iterable[LLMJob]:
                for key in pending:
                    component = harvest.get_component(key)
                    assert component is not None, f"Component {key} not found in harvest DB"
                    stats = get_token_stats(key)
                    assert stats is not None, f"No {pass_name} token stats for {key}"

                    related = get_related(key, labels_so_far)
                    db.save_prompt_edges(
                        [
                            PromptEdge(
                                component_key=key,
                                related_key=r.component_key,
                                pass_name=pass_name,
                                attribution=r.attribution,
                                related_label=r.label,
                                related_confidence=r.confidence,
                            )
                            for r in related
                        ]
                    )
                    yield LLMJob(
                        prompt=make_prompt(component, stats, related),
                        schema=LABEL_SCHEMA,
                        key=key,
                    )

            return await _collect_labels(llm_map, jobs(), len(pending), save_label)

        return process

    # -- Scan (fold over layers) -----------------------------------------------

    async def scan(
        layer_order: list[tuple[str, list[str]]],
        initial: dict[str, LabelResult],
        step: Step,
    ) -> dict[str, LabelResult]:
        labels = dict(initial)
        if labels:
            logger.info(f"Resuming, {len(labels)} already completed")

        completed_so_far = 0
        for layer, keys in layer_order:
            pending = [k for k in keys if k not in labels]
            if not pending:
                completed_so_far += len(keys)
                continue

            new_labels = await step(pending, labels)
            labels.update(new_labels)

            completed_so_far += len(keys)
            logger.info(f"Completed layer {layer} ({completed_so_far}/{total})")

        return labels

    # -- Map (parallel over all components) ------------------------------------

    async def map_unify(
        output_labels: dict[str, LabelResult],
        input_labels: dict[str, LabelResult],
    ) -> None:
        completed = db.get_completed_unified_keys()
        keys = [k for k in all_keys if k not in completed]
        if not keys:
            logger.info("Unification: all labels already completed")
            return
        if completed:
            logger.info(f"Unification: resuming, {len(completed)} already completed")

        unifiable_keys = [k for k in keys if k in output_labels and k in input_labels]
        n_skipped = len(keys) - len(unifiable_keys)

        def jobs() -> Iterable[LLMJob]:
            for key in unifiable_keys:
                component = harvest.get_component(key)
                assert component is not None, f"Component {key} not found in harvest DB"
                prompt = format_unification_prompt(
                    output_label=output_labels[key],
                    input_label=input_labels[key],
                    component=component,
                    model_metadata=model_metadata,
                    app_tok=app_tok,
                    label_max_words=config.label_max_words,
                    max_examples=config.max_examples,
                )
                yield LLMJob(prompt=prompt, schema=LABEL_SCHEMA, key=key)

        if n_skipped:
            logger.warning(f"Skipping {n_skipped} components missing output or input labels")
        logger.info(f"Unifying {len(unifiable_keys)} components")
        new_labels = await _collect_labels(
            llm_map, jobs(), len(unifiable_keys), db.save_unified_label
        )
        logger.info(f"Unification: completed {len(new_labels)}/{len(keys)}")

    # -- Run -------------------------------------------------------------------

    logger.info("Initializing DB and building scan steps...")
    db = GraphInterpDB(db_path)

    metric = config.attr_metric
    get_targets = _make_get_attributed(attribution_storage.get_top_targets, metric)
    get_sources = _make_get_attributed(attribution_storage.get_top_sources, metric)

    def _output_prompt(
        component: ComponentData, stats: TokenPRLift, related: list[RelatedComponent]
    ) -> str:
        return format_output_prompt(
            component=component,
            model_metadata=model_metadata,
            app_tok=app_tok,
            output_token_stats=stats,
            related=related,
            label_max_words=config.label_max_words,
            max_examples=config.max_examples,
        )

    def _input_prompt(
        component: ComponentData, stats: TokenPRLift, related: list[RelatedComponent]
    ) -> str:
        return format_input_prompt(
            component=component,
            model_metadata=model_metadata,
            app_tok=app_tok,
            input_token_stats=stats,
            related=related,
            label_max_words=config.label_max_words,
            max_examples=config.max_examples,
        )

    label_output = _make_process_layer(
        _get_related(get_targets),
        db.save_output_label,
        "output",
        lambda key: get_output_token_stats(token_stats, key, app_tok, top_k=50),
        _output_prompt,
    )
    label_input = _make_process_layer(
        _get_related(get_sources),
        db.save_input_label,
        "input",
        lambda key: get_input_token_stats(token_stats, key, app_tok, top_k=20),
        _input_prompt,
    )

    async def _run() -> None:
        logger.section("Phase 1: Output pass (late → early)")
        output_labels = await scan(list(reversed(layers)), db.get_all_output_labels(), label_output)

        logger.section("Phase 2: Input pass (early → late)")
        input_labels = await scan(list(layers), db.get_all_input_labels(), label_input)

        logger.section("Phase 3: Unification")
        await map_unify(output_labels, input_labels)

        logger.info(
            f"Completed: {db.get_label_count('output_labels')} output, "
            f"{db.get_label_count('input_labels')} input, "
            f"{db.get_label_count('unified_labels')} unified labels -> {db_path}"
        )
        db.mark_done()

    try:
        asyncio.run(_run())
    finally:
        db.close()


# -- Shared LLM call machinery ------------------------------------------------


async def _collect_labels(
    llm_map: Callable[[Iterable[LLMJob], int | None], AsyncGenerator[LLMResult | LLMError]],
    jobs: Iterable[LLMJob],
    n_total: int,
    save_label: Callable[[LabelResult], None],
) -> dict[str, LabelResult]:
    """Run LLM jobs, parse results, save to DB, return new labels."""
    new_labels: dict[str, LabelResult] = {}
    n_errors = 0

    async for outcome in llm_map(jobs, n_total):
        match outcome:
            case LLMResult(job=job, parsed=parsed, raw=raw):
                result = _parse_label(job.key, parsed, raw, job.prompt)
                save_label(result)
                new_labels[job.key] = result
            case LLMError(job=job, error=e):
                n_errors += 1
                logger.error(f"Skipping {job.key}: {type(e).__name__}: {e}")
        _check_error_rate(n_errors, len(new_labels))

    return new_labels


def _parse_label(key: str, parsed: dict[str, object], raw: str, prompt: str) -> LabelResult:
    assert len(parsed) == 3, f"Expected 3 fields, got {len(parsed)}"
    label = parsed["label"]
    confidence = parsed["confidence"]
    reasoning = parsed["reasoning"]
    assert isinstance(label, str) and isinstance(confidence, str) and isinstance(reasoning, str)
    return LabelResult(
        component_key=key,
        label=label,
        confidence=confidence,
        reasoning=reasoning,
        raw_response=raw,
        prompt=prompt,
    )


def _check_error_rate(n_errors: int, n_done: int) -> None:
    total = n_errors + n_done
    if total > 10 and n_errors / total > 0.05:
        raise RuntimeError(
            f"Error rate {n_errors / total:.0%} ({n_errors}/{total}) exceeds 5% threshold"
        )
