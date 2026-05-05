"""Plot component activations vs component ID for high-CI datapoints.

Creates scatter plots (one per layer) where:
- X-axis: Component rank (ordered by median normalized activation, or by firing frequency)
- Y-axis: Component activation (normalized per-component to [0, 1])
- Filter: Only plots datapoints where CI > threshold

Usage:
    python -m param_decomp.scripts.plot_component_activations.plot_component_activations \
        wandb:goodfire/spd/runs/<run_id>
    python -m param_decomp.scripts.plot_component_activations.plot_component_activations \
        wandb:goodfire/spd/runs/<run_id> --ci_threshold 0.0
"""

import time
from collections import defaultdict
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from numpy.typing import NDArray

from param_decomp.harvest.repo import HarvestRepo
from param_decomp.harvest.schemas import ComponentData
from param_decomp.log import logger
from param_decomp.param_decomp_types import ModelPath
from param_decomp.utils.wandb_utils import parse_wandb_run_path

SCRIPT_DIR = Path(__file__).parent


def _extract_activations(
    components: list[ComponentData],
    ci_threshold: float,
) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, list[float]]]]:
    """Extract component activations, separating all vs above-threshold.

    Returns:
        - all_activations:      layer -> component_key -> all activation values
        - filtered_activations: layer -> component_key -> activations where CI > threshold
    """
    all_activations: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    filtered_activations: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for component_data in tqdm.tqdm(components, desc="Extracting activations", unit="comp"):
        layer = component_data.layer
        component_key = component_data.component_key
        for example in component_data.activation_examples:
            ci_vals = example.activations["causal_importance"]
            act_vals = example.activations["component_activation"]
            for ci_val, act_val in zip(ci_vals, act_vals, strict=True):
                all_activations[layer][component_key].append(act_val)
                if ci_val > ci_threshold:
                    filtered_activations[layer][component_key].append(act_val)

    return dict(all_activations), dict(filtered_activations)


def _normalize_per_component(
    all_activations: dict[str, list[float]],
    filtered_activations: dict[str, list[float]],
) -> dict[str, NDArray[np.floating]]:
    """Normalize filtered activations to [0, 1] using min-max from all activations."""
    normalized: dict[str, NDArray[np.floating]] = {}
    for key, filtered_acts in filtered_activations.items():
        if not filtered_acts:
            continue
        all_acts = np.array(all_activations[key])
        filtered_arr = np.array(filtered_acts)
        min_val = all_acts.min()
        max_val = all_acts.max()
        if max_val > min_val:
            normalized[key] = (filtered_arr - min_val) / (max_val - min_val)
        else:
            normalized[key] = np.full_like(filtered_arr, 0.5)
    return normalized


def _order_by_median(normalized: dict[str, NDArray[np.floating]]) -> list[str]:
    """Order component keys by median of their normalized activations (descending)."""
    medians = [(key, float(np.median(acts))) for key, acts in normalized.items()]
    medians.sort(key=lambda x: x[1], reverse=True)
    return [key for key, _ in medians]


def _order_by_frequency(
    normalized: dict[str, NDArray[np.floating]], firing_counts: dict[str, int]
) -> list[str]:
    """Order component keys by pre-calculated firing counts (descending)."""
    freqs = [(key, firing_counts.get(key, 0)) for key in normalized]
    freqs.sort(key=lambda x: x[1], reverse=True)
    return [key for key, _ in freqs]


def _create_layer_scatter_plot(
    normalized_by_key: dict[str, NDArray[np.floating]],
    ordered_keys: list[str],
    layer_name: str,
    run_id: str,
    output_path: Path,
    x_label: str,
    y_label: str,
) -> None:
    """Create scatter plot for a single layer."""
    x_vals: list[int] = []
    y_vals: list[float] = []
    for rank, key in enumerate(ordered_keys):
        acts = normalized_by_key[key]
        x_vals.extend([rank] * len(acts))
        y_vals.extend(acts.tolist())

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.scatter(x_vals, y_vals, alpha=0.3, s=1, marker=".")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"Layer: {layer_name}   |||   Run id: {run_id}")

    n_components = len(ordered_keys)
    n_points = len(x_vals)
    ax.text(
        0.02,
        0.98,
        f"Components: {n_components}\nDatapoints: {n_points}",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    fig.tight_layout()
    logger.info(f"  Saving {output_path.name} ({n_components} components, {n_points} points)...")
    t0 = time.time()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pil_kwargs={"optimize": True})
    plt.close(fig)
    logger.info(f"    saved in {time.time() - t0:.1f}s")


def plot_component_activations(
    wandb_path: ModelPath,
    ci_threshold: float = 0.1,
) -> None:
    _entity, _project, run_id = parse_wandb_run_path(str(wandb_path))

    out_dir = SCRIPT_DIR / "out" / run_id
    out_dir_median = out_dir / "order-by-median"
    out_dir_freq = out_dir / "order-by-freq"
    out_dir_median.mkdir(parents=True, exist_ok=True)
    out_dir_freq.mkdir(parents=True, exist_ok=True)

    repo = HarvestRepo.open_most_recent(decomposition_id=run_id, readonly=True)
    assert repo is not None, f"No harvest data for {run_id}"

    logger.info(f"Loading components for run {run_id}...")
    t0 = time.time()
    components = repo.get_all_components()
    logger.info(f"Loaded {len(components)} components in {time.time() - t0:.1f}s")

    logger.info("Loading firing counts...")
    t0 = time.time()
    token_stats = repo.get_token_stats()
    assert token_stats is not None, f"No token stats found for run {run_id}"
    firing_counts = {
        key: int(count)
        for key, count in zip(token_stats.component_keys, token_stats.firing_counts, strict=True)
    }
    logger.info(f"Loaded firing counts in {time.time() - t0:.1f}s")

    t0 = time.time()
    all_by_layer, filtered_by_layer = _extract_activations(components, ci_threshold)
    logger.info(f"Extraction took {time.time() - t0:.1f}s")

    n_layers = len(filtered_by_layer)
    n_total = sum(sum(len(v) for v in layer.values()) for layer in filtered_by_layer.values())
    logger.info(f"Found {n_total} datapoints across {n_layers} layers with CI > {ci_threshold}")
    assert n_total > 0, "No datapoints found above threshold. Try lowering ci_threshold."

    logger.info(f"Creating per-layer plots (ordered by median) in {out_dir_median}/...")
    for layer_name in sorted(all_by_layer.keys()):
        all_acts = all_by_layer[layer_name]
        filtered_acts = filtered_by_layer.get(layer_name, {})
        normalized = _normalize_per_component(all_acts, filtered_acts)
        if not normalized:
            continue
        ordered_keys = _order_by_median(normalized)
        safe_name = layer_name.replace(".", "_")
        _create_layer_scatter_plot(
            normalized,
            ordered_keys,
            layer_name,
            run_id,
            out_dir_median / f"{safe_name}.png",
            x_label="Component Rank (by median activation)",
            y_label="Normalized Component Activation",
        )

    logger.info(f"Creating per-layer plots (ordered by frequency) in {out_dir_freq}/...")
    for layer_name in sorted(all_by_layer.keys()):
        all_acts = all_by_layer[layer_name]
        filtered_acts = filtered_by_layer.get(layer_name, {})
        normalized = _normalize_per_component(all_acts, filtered_acts)
        if not normalized:
            continue
        abs_from_midpoint = {key: np.abs(acts - 0.5) for key, acts in normalized.items()}
        ordered_keys = _order_by_frequency(abs_from_midpoint, firing_counts)
        safe_name = layer_name.replace(".", "_")
        _create_layer_scatter_plot(
            abs_from_midpoint,
            ordered_keys,
            layer_name,
            run_id,
            out_dir_freq / f"{safe_name}.png",
            x_label="Component Rank (by firing frequency)",
            y_label="|Normalized Component Activation - 0.5|",
        )

    logger.info(f"All plots saved to {out_dir}")


if __name__ == "__main__":
    fire.Fire(plot_component_activations)
