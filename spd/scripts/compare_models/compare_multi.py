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
    CompareModelsConfig,
    ModelComparator,
    format_results_markdown,
)
from spd.utils.run_utils import save_file


class MultiCompareConfig(BaseConfig):
    model_paths: list[str] = Field(..., description="List of model paths to compare pairwise")

    mean_ci_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Minimum mean causal importance for components to be included in comparison",
    )
    n_eval_steps: int = Field(
        ..., description="Number of evaluation steps to compute mean causal importances"
    )

    eval_batch_size: int = Field(..., description="Batch size for evaluation data loading")
    shuffle_data: bool = Field(..., description="Whether to shuffle the evaluation data")
    output_dir: str | None = Field(
        default=None,
        description="Directory to save results (defaults to 'out' directory relative to script location)",
    )


def _model_id(path: str) -> str:
    return path.rstrip("/").split("/")[-1]


def format_summary_markdown(
    pairwise_results: dict[tuple[str, str], dict[str, float]],
    config: MultiCompareConfig,
) -> str:
    """Format a summary report averaging across all pairwise comparisons."""
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

    prefixes = [
        ("rank1", "Rank-1 (V@U)"),
        ("u", "U vectors"),
        ("v", "V vectors"),
        ("ci", "CI profiles"),
    ]
    stats = ["mean", "std", "min", "max"]

    # Summary table (all_layers averages)
    lines.append("## Mean across all pairs (all layers)\n")
    lines.append("| Metric | Mean | Std | Min | Max |")
    lines.append("|--------|-----:|----:|----:|----:|")
    for prefix, label in prefixes:
        vals = [averaged.get(f"{prefix}_cosine_{s}/all_layers") for s in stats]
        if vals[0] is not None:
            lines.append(
                f"| {label} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {vals[3]:.4f} |"
            )
    lines.append("")

    # Per-pair summary (just the all_layers mean for each metric type)
    lines.append("## Per-pair results (all_layers mean)\n")
    lines.append("| Pair | Rank-1 | U | V | CI |")
    lines.append("|------|-------:|--:|--:|---:|")
    for (id_a, id_b), result in pairwise_results.items():
        rank1 = result.get("rank1_cosine_mean/all_layers")
        u = result.get("u_cosine_mean/all_layers")
        v = result.get("v_cosine_mean/all_layers")
        ci = result.get("ci_cosine_mean/all_layers")
        cells = [f"{x:.4f}" if x is not None else "N/A" for x in (rank1, u, v, ci)]
        lines.append(f"| {id_a} vs {id_b} | {' | '.join(cells)} |")
    lines.append("")

    # Per-layer averaged table
    layer_names: list[str] = []
    for key in averaged:
        if "/" not in key:
            continue
        layer = key.split("/", 1)[1]
        if layer != "all_layers" and layer not in layer_names:
            layer_names.append(layer)

    if layer_names:
        lines.append("## Per-layer breakdown (averaged across pairs)\n")
        for prefix, label in prefixes:
            lines.append(f"### {label}\n")
            lines.append("| Layer | Mean | Std | Min | Max |")
            lines.append("|-------|-----:|----:|----:|----:|")
            for layer in layer_names:
                vals = [averaged.get(f"{prefix}_cosine_{s}/{layer}") for s in stats]
                if vals[0] is not None:
                    lines.append(
                        f"| {layer} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {vals[3]:.4f} |"
                    )
            lines.append("")

    return "\n".join(lines)


def main(config_path: Path | str) -> None:
    config = MultiCompareConfig.from_file(config_path)
    assert len(config.model_paths) >= 2, "Need at least 2 models to compare"

    if config.output_dir is None:
        output_dir = Path(__file__).parent / "out"
    else:
        output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = list(itertools.combinations(range(len(config.model_paths)), 2))
    logger.info(f"Comparing {len(config.model_paths)} models ({len(pairs)} pairs)")

    pairwise_results: dict[tuple[str, str], dict[str, float]] = {}

    for idx, (i, j) in enumerate(pairs):
        path_a = config.model_paths[i]
        path_b = config.model_paths[j]
        id_a = _model_id(path_a)
        id_b = _model_id(path_b)

        logger.info(f"Pair {idx + 1}/{len(pairs)}: {id_a} vs {id_b}")

        pair_config = CompareModelsConfig(
            current_model_path=path_a,
            reference_model_path=path_b,
            mean_ci_threshold=config.mean_ci_threshold,
            n_eval_steps=config.n_eval_steps,
            eval_batch_size=config.eval_batch_size,
            shuffle_data=config.shuffle_data,
        )

        comparator = ModelComparator(pair_config)
        eval_iterator = comparator.create_eval_data_loader()
        similarities = comparator.run_comparison(eval_iterator)

        pairwise_results[(id_a, id_b)] = similarities

        stem = f"{id_a}_vs_{id_b}"
        save_file(similarities, output_dir / f"{stem}.json")
        pair_report = format_results_markdown(similarities, pair_config)
        (output_dir / f"{stem}.md").write_text(pair_report)

        logger.info(f"  Saved {stem}.json and {stem}.md")

        # Free GPU memory before next pair
        del comparator

    # Save summary
    summary_data = {
        "model_paths": config.model_paths,
        "pairwise": {f"{a}_vs_{b}": v for (a, b), v in pairwise_results.items()},
    }
    save_file(summary_data, output_dir / "multi_summary.json")

    summary_report = format_summary_markdown(pairwise_results, config)
    (output_dir / "multi_summary.md").write_text(summary_report)

    logger.info(f"All comparisons complete! Results saved to {output_dir}")

    # Print summary table
    for (id_a, id_b), result in pairwise_results.items():
        rank1 = result.get("rank1_cosine_mean/all_layers", float("nan"))
        ci = result.get("ci_cosine_mean/all_layers", float("nan"))
        logger.info(f"  {id_a} vs {id_b}: rank1={rank1:.4f}, ci={ci:.4f}")


if __name__ == "__main__":
    fire.Fire(main)
