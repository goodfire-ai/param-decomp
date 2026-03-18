"""Scatter plot of component L2 norms vs firing density rank.

X axis: component rank ordered by firing density (same as ci_density plot)
Y axis: Frobenius norm of component (||V[:, i]|| * ||U[i, :]||)
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

from spd.harvest.db import HarvestDB
from spd.models.component_model import ComponentModel
from spd.models.components import LinearComponents
from spd.settings import SPD_OUT_DIR

WANDB_PATH = "wandb:goodfire/spd/runs/s-55ea3f9b"
HARVEST_PATH = SPD_OUT_DIR / "harvest/s-55ea3f9b/h-20260227_010249/harvest.db"
OUT_PATH = Path(__file__).parent / "component_norms.png"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model = ComponentModel.from_pretrained(WANDB_PATH).to(device)

    print("Computing norms...")
    norms: dict[str, torch.Tensor] = {}
    for layer_name, module in model.named_modules():
        if isinstance(module, LinearComponents):
            harvest_layer = layer_name.removeprefix("_components.").replace("-", ".")
            v_norms = module.V.norm(dim=0)  # (C,)
            u_norms = module.U.norm(dim=1)  # (C,)
            component_norms = (v_norms * u_norms).detach().cpu()
            for i, norm in enumerate(component_norms):
                norms[f"{harvest_layer}:{i}"] = norm.item()

    print(f"{len(norms)} components")

    print("Loading firing densities...")
    db = HarvestDB(HARVEST_PATH)
    summaries = db.get_summary()

    sorted_keys = sorted(norms.keys(), key=lambda k: summaries[k].firing_density if k in summaries else 0.0)
    sorted_densities = [summaries[k].firing_density if k in summaries else 0.0 for k in sorted_keys]
    sorted_norms = [norms[k] for k in sorted_keys]

    ranks = np.arange(len(sorted_keys))
    norm_vals = np.array(sorted_norms)
    density_vals = np.array(sorted_densities)

    print("Plotting...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Left: scatter coloured by density
    sc = axes[0].scatter(ranks, norm_vals, s=0.5, alpha=0.3, c=np.log1p(density_vals), cmap="plasma")
    plt.colorbar(sc, ax=axes[0], label="log(1 + firing_density)")
    axes[0].set_xlabel("Component rank (firing density →)")
    axes[0].set_ylabel("Frobenius norm (||V[:,i]|| · ||U[i,:]||)")
    axes[0].set_title("Component norms vs firing density rank")

    tick_positions = np.linspace(0, len(sorted_keys), 6).astype(int)
    tick_labels = [f"{density_vals[min(p, len(density_vals)-1)]:.3f}" for p in tick_positions]
    axes[0].set_xticks(tick_positions)
    axes[0].set_xticklabels(tick_labels, fontsize=8)

    # Right: norm vs firing density (log scale x)
    nonzero_mask = density_vals > 0
    axes[1].scatter(
        density_vals[nonzero_mask], norm_vals[nonzero_mask],
        s=0.5, alpha=0.3, color="steelblue"
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Firing density (log scale)")
    axes[1].set_ylabel("Frobenius norm")
    axes[1].set_title("Component norm vs firing density")

    fig.suptitle("s-55ea3f9b — Component Frobenius norms", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
