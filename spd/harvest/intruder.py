"""Intruder detection scoring.

Tests whether a component's activating examples are coherent by asking an LLM
to identify an "intruder" example drawn from a different component. No labels needed.

Based on: "Evaluating SAE interpretability without explanations" (2025).

Usage:
    python -m spd.autointerp.scoring.scripts.run_intruder <wandb_path> --limit 100
"""

import bisect
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.app.backend.utils import delimit_tokens
from spd.autointerp.llm_api import LLMError, LLMJob, LLMResult, map_llm_calls
from spd.autointerp.providers import LLMProvider
from spd.harvest.config import IntruderEvalConfig
from spd.harvest.db import HarvestDB
from spd.harvest.schemas import ActivationExample, ComponentData
from spd.log import logger

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


@dataclass
class IntruderResult:
    component_key: str
    score: float
    trials: list[IntruderTrial]
    n_errors: int


class DensityIndex:
    """Index of components sorted by bold density for efficient similar-density lookup."""

    def __init__(self, components: list[ComponentData], min_examples: int) -> None:
        eligible = [c for c in components if len(c.activation_examples) >= min_examples]
        eligible.sort(key=lambda c: c.firing_density)
        self._components = eligible
        self._densities = [c.firing_density for c in eligible]
        self._key_to_idx = {c.component_key: i for i, c in enumerate(self._components)}

    def sample_similar(
        self,
        target: ComponentData,
        rng: random.Random,
        tolerance: float,
    ) -> ComponentData:
        """Sample a different component with similar bold density."""
        assert target.component_key in self._key_to_idx
        target_density = self._densities[self._key_to_idx[target.component_key]]

        lo = bisect.bisect_left(self._densities, target_density - tolerance)
        hi = bisect.bisect_right(self._densities, target_density + tolerance)

        candidates = [
            self._components[i]
            for i in range(lo, hi)
            if self._components[i].component_key != target.component_key
        ]

        # Widen search if no candidates in tolerance band
        if not candidates:
            candidates = [c for c in self._components if c.component_key != target.component_key]

        return rng.choice(candidates)


def _format_example(
    example: ActivationExample,
    app_tok: AppTokenizer,
) -> str:
    spans = app_tok.get_spans(example.token_ids)
    tokens = [(span, firing) for span, firing in zip(spans, example.firings, strict=True)]
    return delimit_tokens(tokens)


def _sample_intruder(
    target: ComponentData,
    density_index: DensityIndex,
    rng: random.Random,
    density_tolerance: float,
) -> ActivationExample:
    """Sample an intruder example from a component with similar bold density."""
    donor = density_index.sample_similar(target, rng, tolerance=density_tolerance)
    return rng.choice(donor.activation_examples)


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
        examples_text += f"Example {i + 1}: {_format_example(ex, app_tok)}\n\n"

    return f"""\
Below are {n_total} text snippets from a neural network's training data. {n_real} come from contexts \
where the SAME component fires strongly. One is an INTRUDER from a DIFFERENT component.

Tokens between <<delimiters>> are where the component fires most strongly.

{examples_text}\
Which example is the intruder? Identify what pattern the majority share, then pick \
the example that does not fit.

Respond with the intruder example number (1-{n_total}) and brief reasoning."""


@dataclass
class _TrialGroundTruth:
    component_key: str
    correct_answer: int


async def run_intruder_scoring(
    components: list[ComponentData],
    provider: LLMProvider,
    tokenizer_name: str,
    score_db: HarvestDB,
    eval_config: IntruderEvalConfig,
    limit: int | None,
    cost_limit_usd: float | None,
) -> list[IntruderResult]:
    n_real = eval_config.n_real
    n_trials = eval_config.n_trials
    density_tolerance = eval_config.density_tolerance

    app_tok = AppTokenizer.from_pretrained(tokenizer_name)

    eligible = [c for c in components if len(c.activation_examples) >= n_real + 1]
    if limit is not None:
        eligible = eligible[:limit]

    density_index = DensityIndex(components, min_examples=n_real + 1)

    existing_scores = score_db.get_scores("intruder")
    completed = set(existing_scores.keys())
    if completed:
        logger.info(f"Resuming: {len(completed)} already scored")

    remaining = [c for c in eligible if c.component_key not in completed]
    logger.info(f"Scoring {len(remaining)} components ({len(remaining) * n_trials} trials)")

    rng = random.Random()
    jobs: list[LLMJob] = []
    ground_truth: dict[str, _TrialGroundTruth] = {}

    for i, component in enumerate(remaining):
        if i > 0 and i % 1000 == 0:
            logger.info(f"Building trials: {i}/{len(remaining)} components")
        for trial_idx in range(n_trials):
            real_examples = rng.sample(component.activation_examples, n_real)
            intruder = _sample_intruder(component, density_index, rng, density_tolerance)
            intruder_pos = rng.randint(0, n_real)
            correct_answer = intruder_pos + 1

            key = f"{component.component_key}/trial{trial_idx}"
            jobs.append(
                LLMJob(
                    prompt=_build_prompt(real_examples, intruder, intruder_pos, app_tok),
                    key=key,
                )
            )
            ground_truth[key] = _TrialGroundTruth(
                component_key=component.component_key,
                correct_answer=correct_answer,
            )
    logger.info(f"Built {len(jobs)} trials")

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
        score_db.save_score(ck, "intruder", score, json.dumps(asdict(result)))

    async for outcome in map_llm_calls(
        provider=provider,
        jobs=jobs,
        max_tokens=300,
        max_concurrent=eval_config.max_concurrent,
        max_requests_per_minute=eval_config.max_requests_per_minute,
        cost_limit_usd=cost_limit_usd,
        response_schema=INTRUDER_SCHEMA,
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
                    )
                )
                _try_save(gt.component_key)
            case LLMError(job=job, error=e):
                gt = ground_truth[job.key]
                component_errors[gt.component_key] += 1
                logger.error(f"{job.key}: {type(e).__name__}: {e}")
                _try_save(gt.component_key)

    logger.info(f"Scored {len(results)} components")
    return results
