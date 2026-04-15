"""Detection scoring.

Tests whether a component's interpretation label is predictive of its activations by asking
an LLM to classify plain text examples as activating or non-activating.

Based on: EleutherAI's sae-auto-interp (https://blog.eleuther.ai/autointerp/).
"""

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.app.backend.utils import delimit_tokens
from spd.autointerp.config import DetectionEvalConfig
from spd.autointerp.db import InterpDB
from spd.autointerp.llm_api import LLMError, LLMJob, LLMResult, map_llm_calls
from spd.autointerp.providers import LLMProvider
from spd.autointerp.repo import InterpRepo
from spd.harvest.schemas import ActivationExample, ComponentData
from spd.log import logger

DETECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "activating": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "1-indexed example numbers that activate the component",
        },
    },
    "required": ["activating"],
}


@dataclass
class DetectionTrial:
    predicted_activating: list[int]  # 1-indexed example numbers the LLM said activate
    actual_activating: list[int]  # ground truth 1-indexed
    tpr: float
    tnr: float
    balanced_acc: float


@dataclass
class DetectionResult:
    component_key: str
    score: float  # mean balanced accuracy across trials
    trials: list[DetectionTrial]
    n_errors: int


def _format_example_with_center_token(
    example: ActivationExample,
    app_tok: AppTokenizer,
) -> str:
    """Format an example with the center token marked with <<delimiters>>.

    Harvest windows are centered on the firing position, so the center token
    is always the one that triggered collection. We mark center for both
    activating and non-activating examples to avoid positional leakage.
    """
    valid_ids = [tid for tid in example.token_ids if tid >= 0]
    center = len(valid_ids) // 2
    spans = app_tok.get_spans(valid_ids)
    tokens = [(span, i == center) for i, span in enumerate(spans)]
    return delimit_tokens(tokens)


def _sample_non_activating_examples(
    target_component: ComponentData,
    all_components: list[ComponentData],
    n: int,
    rng: random.Random,
) -> list[ActivationExample]:
    """Sample non-activating examples from other components."""
    other_components = [
        c
        for c in all_components
        if c.component_key != target_component.component_key and len(c.activation_examples) >= 1
    ]
    assert other_components, "No other components available for non-activating sampling"

    sampled: list[ActivationExample] = []
    for _ in range(n):
        donor = rng.choice(other_components)
        sampled.append(rng.choice(donor.activation_examples))
    return sampled


def _build_detection_prompt(
    label: str,
    examples_with_labels: list[tuple[str, bool]],
) -> str:
    n_total = len(examples_with_labels)

    examples_text = ""
    for i, (text, _) in enumerate(examples_with_labels):
        examples_text += f"Example {i + 1}: {text}\n\n"

    return f"""\
A neural network component has been labeled as: "{label}"

Below are {n_total} text snippets. In each, one token is marked between <<delimiters>>. \
For some examples, the marked token is one where this component fires. \
For others, the marked token is random.

{examples_text}\
Based on the label, in which examples is the <<marked>> token one where this component fires?

Respond with the list of activating example numbers."""


@dataclass
class _TrialGroundTruth:
    component_key: str
    actual_activating: set[int]
    actual_non_activating: set[int]


async def run_detection_scoring(
    components: list[ComponentData],
    interp_repo: InterpRepo,
    score_db: InterpDB,
    provider: LLMProvider,
    tokenizer_name: str,
    config: DetectionEvalConfig,
    max_concurrent: int,
    max_requests_per_minute: int,
    limit: int | None,
    target_component_keys: list[str] | None,
    seed: int,
    cost_limit_usd: float | None,
) -> list[DetectionResult]:
    app_tok = AppTokenizer.from_pretrained(tokenizer_name)

    labels = {key: result.label for key, result in interp_repo.get_all_interpretations().items()}

    if target_component_keys is not None:
        eligible_by_key = {
            c.component_key: c
            for c in components
            if len(c.activation_examples) >= config.n_activating and c.component_key in labels
        }
        missing = [key for key in target_component_keys if key not in eligible_by_key]
        if missing:
            logger.warning(
                "Skipping target component keys missing labels or enough activation examples "
                f"for detection: {missing[:10]} ({len(missing)} missing)"
            )
        eligible = [eligible_by_key[key] for key in target_component_keys if key in eligible_by_key]
    else:
        eligible = [
            c
            for c in components
            if c.component_key in labels and len(c.activation_examples) >= config.n_activating
        ]
    if limit is not None:
        eligible = eligible[:limit]

    existing_scores = score_db.get_scores("detection")
    completed = set(existing_scores.keys())
    if completed:
        logger.info(f"Resuming: {len(completed)} already scored")

    remaining = [c for c in eligible if c.component_key not in completed]
    logger.info(f"Scoring {len(remaining)} components ({len(remaining) * config.n_trials} trials)")

    rng = random.Random(seed)
    jobs: list[LLMJob] = []
    ground_truth: dict[str, _TrialGroundTruth] = {}

    for component in remaining:
        label = labels[component.component_key]
        for trial_idx in range(config.n_trials):
            activating = (
                list(component.activation_examples)
                if len(component.activation_examples) <= config.n_activating
                else rng.sample(component.activation_examples, config.n_activating)
            )

            non_activating = _sample_non_activating_examples(
                component, components, config.n_non_activating, rng
            )

            formatted: list[tuple[str, bool]] = []
            for ex in activating:
                formatted.append((_format_example_with_center_token(ex, app_tok), True))
            for ex in non_activating:
                formatted.append((_format_example_with_center_token(ex, app_tok), False))
            rng.shuffle(formatted)

            key = f"{component.component_key}/trial{trial_idx}"
            actual_act = {i + 1 for i, (_, is_act) in enumerate(formatted) if is_act}
            actual_non_act = {i + 1 for i, (_, is_act) in enumerate(formatted) if not is_act}
            jobs.append(
                LLMJob(
                    prompt=_build_detection_prompt(label, formatted),
                    key=key,
                )
            )
            ground_truth[key] = _TrialGroundTruth(
                component_key=component.component_key,
                actual_activating=actual_act,
                actual_non_activating=actual_non_act,
            )

    component_trials: defaultdict[str, list[DetectionTrial]] = defaultdict(list)
    component_errors: defaultdict[str, int] = defaultdict(int)
    results: list[DetectionResult] = []

    def _try_save(ck: str) -> None:
        n_done = len(component_trials[ck]) + component_errors.get(ck, 0)
        if n_done < config.n_trials:
            return
        if component_errors.get(ck, 0) > 0:
            return
        trials = component_trials[ck]
        score = sum(t.balanced_acc for t in trials) / len(trials) if trials else 0.0
        result = DetectionResult(component_key=ck, score=score, trials=trials, n_errors=0)
        results.append(result)
        score_db.save_score(ck, "detection", score, json.dumps(asdict(result)))

    async for outcome in map_llm_calls(
        provider=provider,
        jobs=jobs,
        max_tokens=8000,
        max_concurrent=max_concurrent,
        max_requests_per_minute=max_requests_per_minute,
        cost_limit_usd=cost_limit_usd,
        response_schema=DETECTION_SCHEMA,
        n_total=len(jobs),
    ):
        match outcome:
            case LLMResult(job=job, parsed=parsed):
                gt = ground_truth[job.key]
                predicted = {int(x) for x in parsed["activating"]}
                tp = len(predicted & gt.actual_activating)
                tn = len(gt.actual_non_activating - predicted)
                tpr = tp / len(gt.actual_activating) if gt.actual_activating else 0.0
                tnr = tn / len(gt.actual_non_activating) if gt.actual_non_activating else 0.0
                component_trials[gt.component_key].append(
                    DetectionTrial(
                        predicted_activating=sorted(predicted),
                        actual_activating=sorted(gt.actual_activating),
                        tpr=tpr,
                        tnr=tnr,
                        balanced_acc=(tpr + tnr) / 2,
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
