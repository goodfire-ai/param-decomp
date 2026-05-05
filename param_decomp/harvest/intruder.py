"""Intruder detection scoring.

Tests whether a component's activating examples are coherent by asking an LLM
to identify an "intruder" example drawn from a different component. No labels needed.

Based on: "Evaluating SAE interpretability without explanations" (2025).

Usage:
    python -m param_decomp.autointerp.scoring.scripts.run_intruder <wandb_path> --limit 100
"""

import bisect
import json
import random
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass

from param_decomp.app.backend.app_tokenizer import AppTokenizer
from param_decomp.autointerp.llm_api import LLMError, LLMJob, LLMResult, map_llm_calls
from param_decomp.autointerp.providers import LLMProvider
from param_decomp.harvest.config import IntruderEvalConfig
from param_decomp.harvest.db import HarvestDB
from param_decomp.harvest.schemas import ActivationExample
from param_decomp.log import logger

INTRUDER_SCHEMA = {
    "type": "object",
    "properties": {
        "intruder": {
            "type": "integer",
            "description": "1-indexed example number of the intruder",
        },
        "reasoning": {"type": "string", "description": "Brief explanation"},
    },
    "required": ["intruder", "reasoning"],
}


@dataclass
class IntruderTrial:
    correct_answer: int
    predicted: int
    is_correct: bool
    reasoning: str


@dataclass
class IntruderResult:
    component_key: str
    score: float
    trials: list[IntruderTrial]
    n_errors: int


class DensityIndex:
    """Index of components sorted by density for efficient similar-density lookup.

    Lightweight: stores only (key, density) pairs, not full component data.
    """

    def __init__(self, entries: list[tuple[str, float]]) -> None:
        entries.sort(key=lambda e: e[1])
        self._keys = [k for k, _ in entries]
        self._densities = [d for _, d in entries]
        self._key_to_idx = {k: i for i, k in enumerate(self._keys)}

    def sample_similar_key(
        self,
        target_key: str,
        rng: random.Random,
        tolerance: float,
    ) -> str | None:
        assert target_key in self._key_to_idx
        target_density = self._densities[self._key_to_idx[target_key]]

        lo = bisect.bisect_left(self._densities, target_density - tolerance)
        hi = bisect.bisect_right(self._densities, target_density + tolerance)

        candidates = [self._keys[i] for i in range(lo, hi) if self._keys[i] != target_key]

        if not candidates:
            return None

        return rng.choice(candidates)


def _format_example_xml(
    example: ActivationExample,
    app_tok: AppTokenizer,
) -> str:
    spans = app_tok.get_spans(example.token_ids)
    raw = "".join(spans)
    annotated_parts: list[str] = []
    for span, firing in zip(spans, example.firings, strict=True):
        if firing:
            annotated_parts.append(f"[[[{span}]]]")
        else:
            annotated_parts.append(span)
    annotated = "".join(annotated_parts)
    return f"<raw>{raw}</raw>\n<annotated>{annotated}</annotated>"


def _build_prompt(
    real_examples: list[ActivationExample],
    intruder: ActivationExample,
    intruder_position: int,
    app_tok: AppTokenizer,
) -> str:
    all_examples = list(real_examples)
    all_examples.insert(intruder_position, intruder)
    n_total = len(all_examples)
    n_real = len(real_examples)

    examples_text = ""
    for i, ex in enumerate(all_examples):
        examples_text += (
            f"<example_{i + 1}>\n{_format_example_xml(ex, app_tok)}\n</example_{i + 1}>\n\n"
        )

    return f"""\
Below are {n_total} text snippets from a neural network's training data. {n_real} come from contexts \
where the SAME component fires strongly. One is an INTRUDER from a DIFFERENT component.

Each example is shown in two views: <raw> is the literal text, and <annotated> is the same text \
with firing tokens wrapped in [[[brackets]]]. Compare the two to see which tokens are highlighted.

{examples_text}\
Which example is the intruder? Identify what pattern the majority share, then pick \
the example that does not fit.

Respond as JSON: {{"intruder": <int 1-{n_total}>, "reasoning": "<brief explanation>"}}"""


@dataclass
class _TrialGroundTruth:
    component_key: str
    correct_answer: int


def _build_trials(
    remaining_keys: list[str],
    db: HarvestDB,
    density_index: DensityIndex,
    n_real: int,
    n_trials: int,
    density_tolerance: float,
    app_tok: AppTokenizer,
) -> Iterator[tuple[LLMJob, _TrialGroundTruth]]:
    """Lazily build trial prompts, fetching examples from DB one component at a time."""
    rng = random.Random()
    n_skipped = 0
    for i, ck in enumerate(remaining_keys):
        if i > 0 and i % 1000 == 0:
            logger.info(
                f"Building trials: {i}/{len(remaining_keys)} components ({n_skipped} skipped)"
            )
        component = db.get_component(ck)
        assert component is not None, f"Component {ck} not found in DB"

        if density_index.sample_similar_key(ck, rng, density_tolerance) is None:
            n_skipped += 1
            continue

        for trial_idx in range(n_trials):
            real_examples = rng.sample(component.activation_examples, n_real)

            donor_key = density_index.sample_similar_key(ck, rng, density_tolerance)
            assert donor_key is not None
            donor = db.get_component(donor_key)
            assert donor is not None
            intruder = rng.choice(donor.activation_examples)

            intruder_pos = rng.randint(0, n_real)
            correct_answer = intruder_pos + 1

            key = f"{ck}/trial{trial_idx}"
            job = LLMJob(
                prompt=_build_prompt(real_examples, intruder, intruder_pos, app_tok),
                key=key,
            )
            gt = _TrialGroundTruth(component_key=ck, correct_answer=correct_answer)
            yield job, gt


async def run_intruder_scoring(
    db: HarvestDB,
    provider: LLMProvider,
    tokenizer_name: str,
    eval_config: IntruderEvalConfig,
    limit: int | None,
    cost_limit_usd: float | None,
) -> list[IntruderResult]:
    n_real = eval_config.n_real
    n_trials = eval_config.n_trials
    density_tolerance = eval_config.density_tolerance

    app_tok = AppTokenizer.from_pretrained(tokenizer_name)

    logger.info("Loading component index...")
    entries = db.get_component_densities(min_examples=n_real + 1)
    eligible_keys = [k for k, _ in entries]
    logger.info(f"Found {len(eligible_keys)} eligible components")

    if limit is not None:
        eligible_keys = eligible_keys[:limit]

    density_index = DensityIndex(entries)

    existing_scores = db.get_scores("intruder")
    completed = set(existing_scores.keys())
    if completed:
        logger.info(f"Resuming: {len(completed)} already scored")

    remaining_keys = [k for k in eligible_keys if k not in completed]
    n_trials_total = len(remaining_keys) * n_trials
    logger.info(f"Scoring {len(remaining_keys)} components ({n_trials_total} trials)")

    ground_truth: dict[str, _TrialGroundTruth] = {}

    def jobs_iter() -> Iterator[LLMJob]:
        for job, gt in _build_trials(
            remaining_keys, db, density_index, n_real, n_trials, density_tolerance, app_tok
        ):
            ground_truth[job.key] = gt
            yield job

    component_trials: defaultdict[str, list[IntruderTrial]] = defaultdict(list)
    component_errors: defaultdict[str, int] = defaultdict(int)
    results: list[IntruderResult] = []

    def _try_save(ck: str) -> None:
        n_done = len(component_trials[ck]) + component_errors.get(ck, 0)
        if n_done < n_trials:
            return
        if component_errors.get(ck, 0) > 0:
            return
        trials = component_trials[ck]
        correct = sum(1 for t in trials if t.is_correct)
        score = correct / len(trials) if trials else 0.0
        result = IntruderResult(component_key=ck, score=score, trials=trials, n_errors=0)
        results.append(result)
        db.save_score(ck, "intruder", score, json.dumps(asdict(result)))

    async for outcome in map_llm_calls(
        provider=provider,
        jobs=jobs_iter(),
        max_tokens=4000,
        max_concurrent=eval_config.llm.max_concurrent,
        max_requests_per_minute=eval_config.llm.max_requests_per_minute,
        cost_limit_usd=cost_limit_usd,
        response_schema=INTRUDER_SCHEMA,
        n_total=n_trials_total,
    ):
        match outcome:
            case LLMResult(job=job, parsed=parsed):
                gt = ground_truth[job.key]
                predicted = int(parsed["intruder"])
                component_trials[gt.component_key].append(
                    IntruderTrial(
                        correct_answer=gt.correct_answer,
                        predicted=predicted,
                        is_correct=predicted == gt.correct_answer,
                        reasoning=parsed.get("reasoning", ""),
                    )
                )
                db.save_intruder_prompt(job.key, job.prompt)
                _try_save(gt.component_key)
            case LLMError(job=job, error=e):
                gt = ground_truth[job.key]
                component_errors[gt.component_key] += 1
                logger.error(f"{job.key}: {type(e).__name__}: {e}")
                _try_save(gt.component_key)

    logger.info(f"Scored {len(results)} components")
    return results
