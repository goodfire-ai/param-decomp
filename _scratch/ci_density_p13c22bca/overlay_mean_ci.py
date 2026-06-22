"""Overlay the sorted-descending mean-CI curves across the token-sweep snapshots, one
colored line per token count, per site. Reads the per-snapshot mean_cis_*.npz the sweep
render wrote — no GPU / checkpoint restore needed."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("/mnt/home/oli/claude-slack/data/workspaces/3229052cd398/mean_ci_out")
SNAPSHOTS = [
    (8192, "8k"),
    (32768, "32k"),
    (131072, "131k"),
    (524288, "524k"),
    (2097152, "2.1M"),
    (8388608, "8.4M"),
    (33554432, "33.6M"),
]
SITE_KEYS = ["layers_18_mlp_gate_proj", "layers_18_mlp_up_proj", "layers_18_mlp_down_proj"]
SITE_TITLES = {k: k.replace("_", ".") for k in SITE_KEYS}

snapshots = [(label, np.load(OUT_DIR / f"mean_cis_{label}.npz")) for _, label in SNAPSHOTS]
colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.9, len(SNAPSHOTS)))


def render(log_y: bool) -> None:
    fig, axs = plt.subplots(len(SITE_KEYS), 1, figsize=(9, 3.2 * len(SITE_KEYS)), squeeze=False)
    for ax, key in zip(axs[:, 0], SITE_KEYS, strict=True):
        for (_, label), data, color in zip(SNAPSHOTS, snapshots, colors, strict=True):
            sorted_desc = np.sort(data[1][key])[::-1]
            ax.plot(range(sorted_desc.size), sorted_desc, color=color, lw=1.2, label=f"{label} tok")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Component")
        ax.set_ylabel("mean CI")
        ax.set_title(SITE_TITLES[key], fontsize=10)
        ax.legend(fontsize=8, title="tokens")
    fig.suptitle("mean CI per component — token sweep overlay", fontsize=12)
    fig.tight_layout()
    suffix = "log" if log_y else "linear"
    out = OUT_DIR / f"ci_mean_overlay_{suffix}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


render(log_y=True)
render(log_y=False)
