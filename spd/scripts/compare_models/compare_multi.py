"""Multi-model pairwise comparison script.

Computes all pairwise geometric and CI similarity metrics for a set of SPD models,
saves each pairwise result, and generates a summary report with means across pairs.

Usage:
    python spd/scripts/compare_models/compare_multi.py spd/scripts/compare_models/compare_multi_config.yaml
"""

import itertools
from pathlib import Path

import fire
from pydantic import Field

from spd.base_config import BaseConfig
from spd.log import logger
from spd.scripts.compare_models.compare_models import (
    METRIC_PREFIXES,
    CompareModelsConfig,
    _extract_layer_names,
    _metric_summary_table,
    _per_layer_table,
    model_id_from_path,
    resolve_output_dir,
    run_and_save_pair,
)
from spd.utils.run_utils import save_file


class MultiCompareConfig(BaseConfig):
    model_paths: list[str]
    mean_ci_threshold: float = Field(..., ge=0.0, le=1.0)
    n_eval_steps: int
    eval_batch_size: int
    shuffle_data: bool
    output_dir: str | None = None


def format_summary_markdown(
    pairwise_results: dict[tuple[str, str], dict[str, float]],
    config: MultiCompareConfig,
) -> str:
    lines: list[str] = []
    lines.append("# Multi-Model Comparison Summary\n")

    lines.append("**Models:**\n")
    for path in config.model_paths:
        lines.append(f"- `{path}`")
    lines.append("")

    lines.append(f"- **Mean CI threshold**: {config.mean_ci_threshold}")
    lines.append(f"- **Eval steps**: {config.n_eval_steps}")
    lines.append(f"- **Batch size**: {config.eval_batch_size}")
    lines.append(f"- **Number of pairs**: {len(pairwise_results)}\n")

    # Average all pairwise results
    all_keys: set[str] = set()
    for result in pairwise_results.values():
        all_keys.update(result.keys())

    averaged: dict[str, float] = {}
    for key in sorted(all_keys):
        values = [r[key] for r in pairwise_results.values() if key in r]
        if values:
            averaged[key] = sum(values) / len(values)

    lines.append("## Mean across all pairs (all layers)\n")
    lines.extend(_metric_summary_table(averaged, "all_layers"))
    lines.append("")

    # Per-pair summary
    lines.append("## Per-pair results (all_layers mean)\n")
    lines.append("| Pair | Rank-1 | U | V | CI |")
    lines.append("|------|-------:|--:|--:|---:|")
    for (id_a, id_b), result in pairwise_results.items():
        vals = [result.get(f"{p}_cosine_mean/all_layers") for p, _ in METRIC_PREFIXES]
        cells = [f"{x:.4f}" if x is not None else "N/A" for x in vals]
        lines.append(f"| {id_a} vs {id_b} | {' | '.join(cells)} |")
    lines.append("")

    # Per-layer averaged table
    layer_names = _extract_layer_names(averaged)
    if layer_names:
        lines.append("## Per-layer breakdown (averaged across pairs)\n")
        for prefix, label in METRIC_PREFIXES:
            lines.append(f"### {label}\n")
            lines.extend(_per_layer_table(averaged, prefix, layer_names))
            lines.append("")

    return "\n".join(lines)


def main(config_path: Path | str) -> None:
    config = MultiCompareConfig.from_file(config_path)
    assert len(config.model_paths) >= 2, "Need at least 2 models to compare"

    output_dir = resolve_output_dir(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = list(itertools.combinations(range(len(config.model_paths)), 2))
    logger.info(f"Comparing {len(config.model_paths)} models ({len(pairs)} pairs)")

    pairwise_results: dict[tuple[str, str], dict[str, float]] = {}

    for idx, (i, j) in enumerate(pairs):
        path_a, path_b = config.model_paths[i], config.model_paths[j]
        id_a, id_b = model_id_from_path(path_a), model_id_from_path(path_b)

        logger.info(f"Pair {idx + 1}/{len(pairs)}: {id_a} vs {id_b}")

        pair_config = CompareModelsConfig(
            current_model_path=path_a,
            reference_model_path=path_b,
            mean_ci_threshold=config.mean_ci_threshold,
            n_eval_steps=config.n_eval_steps,
            eval_batch_size=config.eval_batch_size,
            shuffle_data=config.shuffle_data,
        )

        pair_dir = output_dir / f"{id_a}_vs_{id_b}"
        similarities, _ = run_and_save_pair(pair_config, pair_dir)
        pairwise_results[(id_a, id_b)] = similarities

        logger.info(f"  Saved to {pair_dir}")

    # Save summary
    summary_data = {
        "model_paths": config.model_paths,
        "pairwise": {f"{a}_vs_{b}": v for (a, b), v in pairwise_results.items()},
    }
    save_file(summary_data, output_dir / "multi_summary.json")
    (output_dir / "multi_summary.md").write_text(format_summary_markdown(pairwise_results, config))

    logger.info(f"All comparisons complete! Results saved to {output_dir}")
    for (id_a, id_b), result in pairwise_results.items():
        rank1 = result.get("rank1_cosine_mean/all_layers", float("nan"))
        ci = result.get("ci_cosine_mean/all_layers", float("nan"))
        logger.info(f"  {id_a} vs {id_b}: rank1={rank1:.4f}, ci={ci:.4f}")


if __name__ == "__main__":
    fire.Fire(main)
