"""Plot 2D heatmap of CI value density vs component index (ordered by firing density).

Tests whether rare components have non-binary CI distributions.

X axis: component rank ordered by firing density (left=rarest, right=densest)
Y axis: CI value (0–1)
Color: density of (component, CI value) observations

Collects raw CI values via a forward pass (unbiased).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from pathlib import Path

from spd.clustering.activations import collect_activations
from spd.clustering.dataset import create_clustering_dataloader
from spd.harvest.db import HarvestDB
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.settings import SPD_OUT_DIR

WANDB_PATH = "wandb:goodfire/spd/runs/s-55ea3f9b"
HARVEST_PATH = SPD_OUT_DIR / "harvest/s-55ea3f9b/h-20260227_010249/harvest.db"
OUT_PATH = Path(__file__).parent / "ci_density.png"

N_TOKENS = 1_000_000
N_TOKENS_PER_SEQ = 10
BATCH_SIZE = 64
N_X_BINS = 300
N_Y_BINS = 100


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model = ComponentModel.from_pretrained(WANDB_PATH).to(device)

    print("Building dataloader...")
    dataloader = create_clustering_dataloader(
        model_path=WANDB_PATH,
        task_name="lm",
        batch_size=BATCH_SIZE,
        seed=0,
    )

    print(f"Collecting CI activations ({N_TOKENS:,} tokens)...")
    activations_dict = collect_activations(
        model=model,
        dataloader=dataloader,
        n_tokens=N_TOKENS,
        n_tokens_per_seq=N_TOKENS_PER_SEQ,
        device=device,
        seed=0,
    )

    # activations_dict: layer -> (n_tokens, n_components)
    all_acts = torch.cat(list(activations_dict.values()), dim=1).cpu().numpy()
    print(f"Activations shape: {all_acts.shape}")

    # Sort components by firing density from harvest
    print("Loading firing densities...")
    db = HarvestDB(HARVEST_PATH)
    summaries = db.get_summary()

    col_keys = [
        f"{layer}:{i}"
        for layer, acts in activations_dict.items()
        for i in range(acts.shape[1])
    ]
    firing_densities = np.array([
        summaries[k].firing_density if k in summaries else 0.0
        for k in col_keys
    ])
    sorted_indices = np.argsort(firing_densities)
    sorted_densities = firing_densities[sorted_indices]
    all_acts = all_acts[:, sorted_indices]
    n_components = all_acts.shape[1]

    print("Building heatmap...")
    heatmap = np.zeros((N_Y_BINS, N_X_BINS), dtype=np.float64)
    for comp_rank in range(n_components):
        x_bin = min(int(comp_rank / n_components * N_X_BINS), N_X_BINS - 1)
        ci_vals = all_acts[:, comp_rank]
        nonzero = ci_vals[ci_vals > 0]
        if len(nonzero) == 0:
            continue
        y_bins = np.minimum((nonzero * N_Y_BINS).astype(int), N_Y_BINS - 1)
        np.add.at(heatmap[:, x_bin], y_bins, 1)

    col_sums = heatmap.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    heatmap_norm = heatmap / col_sums

    print("Plotting...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    tick_positions = np.linspace(0, n_components, 6)
    tick_labels = [
        f"{sorted_densities[min(int(p), n_components - 1)]:.3f}"
        for p in tick_positions
    ]

    im0 = axes[0].imshow(
        np.log1p(heatmap), aspect="auto", origin="lower",
        extent=[0, n_components, 0, 1], cmap="viridis",
    )
    plt.colorbar(im0, ax=axes[0], label="log(count + 1)")
    axes[0].set_title("CI density (log counts)")
    axes[0].set_xlabel("Component rank (firing density →)")
    axes[0].set_ylabel("CI value")
    axes[0].set_xticks(tick_positions)
    axes[0].set_xticklabels(tick_labels, fontsize=8)

    im1 = axes[1].imshow(
        heatmap_norm, aspect="auto", origin="lower",
        extent=[0, n_components, 0, 1], cmap="plasma",
        norm=mcolors.PowerNorm(gamma=0.4),
    )
    plt.colorbar(im1, ax=axes[1], label="fraction of CI values")
    axes[1].set_title("CI distribution per component rank (column-normalised)")
    axes[1].set_xlabel("Component rank (firing density →)")
    axes[1].set_ylabel("CI value")
    axes[1].set_xticks(tick_positions)
    axes[1].set_xticklabels(tick_labels, fontsize=8)

    fig.suptitle(
        f"s-55ea3f9b — CI value distribution vs firing density rank ({N_TOKENS:,} tokens)\n"
        "(left=rarest, right=densest; only CI > 0 shown)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
