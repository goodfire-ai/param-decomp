"""Compare intruder detection scores between two harvest subruns.

Guarantees the same set of component keys is scored in both subruns, making it
a fair A/B comparison of activation pattern coherence.

Usage:
    python -m spd.harvest.scripts.run_intruder_comparison \
        s-55ea3f9b \
        --subrun_a h-20260227_010249 \
        --subrun_b h-20260318_223737 \
        --limit 200
"""

import asyncio
import random
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from spd.adapters import adapter_from_id
from spd.autointerp.providers import OpenRouterLLMConfig, create_provider
from spd.harvest.config import IntruderEvalConfig
from spd.harvest.db import HarvestDB
from spd.harvest.intruder import run_intruder_scoring
from spd.harvest.repo import HarvestRepo
from spd.log import logger


def main(
    decomposition_id: str,
    subrun_a: str,
    subrun_b: str,
    limit: int = 200,
    n_trials: int = 10,
    seed: int = 42,
) -> None:
    load_dotenv()

    eval_config = IntruderEvalConfig(
        llm=OpenRouterLLMConfig(reasoning_effort="none"),
        n_trials=n_trials,
    )
    min_examples = eval_config.n_real + 1

    repo_a = HarvestRepo(decomposition_id, subrun_id=subrun_a, readonly=True)
    repo_b = HarvestRepo(decomposition_id, subrun_id=subrun_b, readonly=True)

    logger.info("Getting eligible component keys...")
    eligible_a = set(repo_a.get_eligible_component_keys(min_examples))
    eligible_b = set(repo_b.get_eligible_component_keys(min_examples))
    shared = sorted(eligible_a & eligible_b)
    logger.info(f"Eligible: A={len(eligible_a)}, B={len(eligible_b)}, shared={len(shared)}")
    assert len(shared) >= limit, f"Only {len(shared)} shared eligible components, need {limit}"

    rng = random.Random(seed)
    target_keys = rng.sample(shared, limit)
    logger.info(f"Selected {limit} component keys for comparison")

    logger.info(f"Loading {limit} components from subrun A ({subrun_a})...")
    components_a = list(repo_a.get_components_bulk(target_keys).values())
    logger.info(f"Loading {limit} components from subrun B ({subrun_b})...")
    components_b = list(repo_b.get_components_bulk(target_keys).values())
    assert len(components_a) == len(components_b) == limit

    tokenizer_name = adapter_from_id(decomposition_id).tokenizer_name
    provider = create_provider(eval_config.llm)

    async def run_both() -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scratch_db_a = HarvestDB(Path(tmpdir) / "scratch_a.db")
            scratch_db_b = HarvestDB(Path(tmpdir) / "scratch_b.db")

            logger.info(f"Scoring subrun A ({subrun_a})...")
            results_a = await run_intruder_scoring(
                components=components_a,
                provider=provider,
                tokenizer_name=tokenizer_name,
                score_db=scratch_db_a,
                eval_config=eval_config,
                limit=None,
                cost_limit_usd=eval_config.cost_limit_usd,
            )

            logger.info(f"Scoring subrun B ({subrun_b})...")
            results_b = await run_intruder_scoring(
                components=components_b,
                provider=provider,
                tokenizer_name=tokenizer_name,
                score_db=scratch_db_b,
                eval_config=eval_config,
                limit=None,
                cost_limit_usd=eval_config.cost_limit_usd,
            )

        scores_a = {r.component_key: r.score for r in results_a}
        scores_b = {r.component_key: r.score for r in results_b}
        shared_scored = set(scores_a) & set(scores_b)

        mean_a = sum(scores_a[k] for k in shared_scored) / len(shared_scored)
        mean_b = sum(scores_b[k] for k in shared_scored) / len(shared_scored)

        print("\n" + "=" * 60)
        print("INTRUDER DETECTION COMPARISON")
        print("=" * 60)
        print(f"  Subrun A: {subrun_a}  (mean={mean_a:.3f})")
        print(f"  Subrun B: {subrun_b}  (mean={mean_b:.3f})")
        print(f"  Components scored: {len(shared_scored)}")
        print(f"  Trials per component: {n_trials}")
        print()

        n_a_better = sum(1 for k in shared_scored if scores_a[k] > scores_b[k])
        n_b_better = sum(1 for k in shared_scored if scores_b[k] > scores_a[k])
        n_tied = sum(1 for k in shared_scored if scores_a[k] == scores_b[k])
        print(f"  A better: {n_a_better}")
        print(f"  B better: {n_b_better}")
        print(f"  Tied:     {n_tied}")
        print(f"  Delta (B - A): {mean_b - mean_a:+.3f}")
        print("=" * 60)

    asyncio.run(run_both())


if __name__ == "__main__":
    import fire

    fire.Fire(main)
