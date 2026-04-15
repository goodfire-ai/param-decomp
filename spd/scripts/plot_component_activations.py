"""Plot component activations vs component ID for high-CI datapoints.

Creates scatter plots (one per layer) where:
- X-axis: Component rank (ordered by median normalized activation)
- Y-axis: Component activation (normalized per-component to [0, 1])
- Filter: Only plots datapoints where CI > threshold

Usage:
    python -m spd.scripts.plot_component_activations s-7884efcc
    python -m spd.scripts.plot_component_activations s-7884efcc --ci-threshold 0.0
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spd.harvest.repo import HarvestRepo
from spd.harvest.schemas import ComponentData


def extract_activations(
    components: list[ComponentData],
    ci_threshold: float,
) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, list[float]]]]:
    """Extract component activations, separating all vs above-threshold.

    Returns:
        Tuple of:
        - all_activations: layer -> component_key -> all activation values (for normalization)
        - filtered_activations: layer -> component_key -> activations where CI > threshold
    """
    all_activations: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    filtered_activations: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for component_data in components:
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


def normalize_per_component(
    all_activations: dict[str, list[float]],
    filtered_activations: dict[str, list[float]],
) -> dict[str, np.ndarray]:
    """Normalize filtered activations to [0, 1] using min-max from all activations."""
    normalized = {}
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


def order_by_median(normalized: dict[str, np.ndarray]) -> list[str]:
    """Order component keys by median of their normalized activations (descending)."""
    medians = [(key, np.median(acts)) for key, acts in normalized.items()]
    medians.sort(key=lambda x: x[1], reverse=True)
    return [key for key, _ in medians]


def order_by_frequency(
    normalized: dict[str, np.ndarray], firing_counts: dict[str, int]
) -> list[str]:
    """Order component keys by pre-calculated firing counts (descending)."""
    freqs = [(key, firing_counts.get(key, 0)) for key in normalized]
    freqs.sort(key=lambda x: x[1], reverse=True)
    return [key for key, _ in freqs]


def create_layer_scatter_plot(
    normalized_by_key: dict[str, np.ndarray],
    ordered_keys: list[str],
    layer_name: str,
    run_id: str,
    output_path: Path,
    x_label: str = "Component Rank (by median activation)",
    y_label: str = "Normalized Component Activation",
) -> None:
    """Create scatter plot for a single layer."""
    x_vals = []
    y_vals = []
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
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="WandB run ID (e.g., 's-7884efcc')")
    parser.add_argument(
        "--ci-threshold",
        type=float,
        default=0.1,
        help="Minimum CI value to include (default: 0.1)",
    )
    args = parser.parse_args()

    base_output_dir = Path(__file__).parent / "out" / args.run_id / "component-act-scatter"
    output_dir_median = base_output_dir / "order-by-median"
    output_dir_freq = base_output_dir / "order-by-freq"
    output_dir_median.mkdir(parents=True, exist_ok=True)
    output_dir_freq.mkdir(parents=True, exist_ok=True)

    repo = HarvestRepo.open_most_recent(decomposition_id=args.run_id, readonly=True)
    assert repo is not None, f"No harvest data for {args.run_id}"

    print(f"Loading components for run {args.run_id}...")
    components = repo.get_all_components()
    print(f"Loaded {len(components)} components")

    print("Loading firing counts...")
    token_stats = repo.get_token_stats()
    assert token_stats is not None, f"No token stats found for run {args.run_id}"
    firing_counts = {
        key: int(count)
        for key, count in zip(token_stats.component_keys, token_stats.firing_counts, strict=True)
    }

    print("Extracting activations...")
    all_by_layer, filtered_by_layer = extract_activations(components, args.ci_threshold)

    n_layers = len(filtered_by_layer)
    n_total = sum(sum(len(v) for v in layer.values()) for layer in filtered_by_layer.values())
    print(f"Found {n_total} datapoints across {n_layers} layers with CI > {args.ci_threshold}")

    if n_total == 0:
        print("No datapoints found above threshold. Try lowering --ci-threshold.")
        return

    print(f"Creating per-layer plots (ordered by median) in {output_dir_median}/...")
    for layer_name in sorted(all_by_layer.keys()):
        all_acts = all_by_layer[layer_name]
        filtered_acts = filtered_by_layer.get(layer_name, {})
        normalized = normalize_per_component(all_acts, filtered_acts)
        if not normalized:
            continue
        ordered_keys = order_by_median(normalized)
        safe_name = layer_name.replace(".", "_")
        output_path = output_dir_median / f"{safe_name}.png"
        create_layer_scatter_plot(normalized, ordered_keys, layer_name, args.run_id, output_path)
        print(f"  {output_path}")

    print(f"Creating per-layer plots (ordered by frequency) in {output_dir_freq}/...")
    for layer_name in sorted(all_by_layer.keys()):
        all_acts = all_by_layer[layer_name]
        filtered_acts = filtered_by_layer.get(layer_name, {})
        normalized = normalize_per_component(all_acts, filtered_acts)
        if not normalized:
            continue
        abs_from_midpoint = {key: np.abs(acts - 0.5) for key, acts in normalized.items()}
        ordered_keys = order_by_frequency(abs_from_midpoint, firing_counts)
        safe_name = layer_name.replace(".", "_")
        output_path = output_dir_freq / f"{safe_name}.png"
        create_layer_scatter_plot(
            abs_from_midpoint,
            ordered_keys,
            layer_name,
            args.run_id,
            output_path,
            x_label="Component Rank (by firing frequency)",
            y_label="|Normalized Component Activation - 0.5|",
        )
        print(f"  {output_path}")

    print("Done!")


if __name__ == "__main__":
    main()
