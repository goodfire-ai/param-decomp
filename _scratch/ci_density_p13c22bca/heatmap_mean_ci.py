"""Density-heatmap version of the sorted mean-CI-per-component plot.

For each site: components are sorted descending by mean CI (same "density ordering" as the
scatter), then binned into NX rank-columns. The y axis is mean CI on a log scale, binned
into NY log-spaced bands. cell(x, y) = fraction of the components in rank-band x whose mean
CI falls in CI-band y (each column normalised over its own components). Where the sorted
curve is flat (the sampling-floor plateau) a column's components pile into one CI band and
the cell is bright; where it is steep (the head) they spread thin across many bands.

Reads the per-snapshot mean_cis_*.npz the sweep render wrote — no GPU / checkpoint restore.
Exact-zero components (never active at this token count) have no log-CI band and are dropped
from their column's normalisation; the per-panel title reports how many were nonzero."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

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

NX = 256
NY = 180
CI_FLOOR = 1e-9
CI_CEIL = 1.0
x_edges = np.linspace(0, 49152, NX + 1)
y_edges = np.logspace(np.log10(CI_FLOOR), np.log10(CI_CEIL), NY + 1)


def column_density(sorted_desc: np.ndarray) -> np.ndarray:
    """(NX, NY) per-column-normalised density of CI values over log CI bands."""
    rank = np.arange(sorted_desc.size)
    counts, _, _ = np.histogram2d(rank, sorted_desc, bins=[x_edges, y_edges])
    col_totals = np.full(NX, 49152 / NX)
    return counts / col_totals[:, None]


def render(n_tokens: int, label: str) -> None:
    data = np.load(OUT_DIR / f"mean_cis_{label}.npz")
    fig, axs = plt.subplots(
        len(SITE_KEYS), 1, figsize=(9, 3.4 * len(SITE_KEYS)), squeeze=False, constrained_layout=True
    )
    mesh = None
    for ax, key in zip(axs[:, 0], SITE_KEYS, strict=True):
        v = data[key]
        sorted_desc = np.sort(v)[::-1]
        density = column_density(sorted_desc)
        masked = np.ma.masked_where(density == 0, density)
        mesh = ax.pcolormesh(
            x_edges, y_edges, masked.T, cmap="magma", norm=LogNorm(vmin=1e-3, vmax=1.0), shading="flat"
        )
        ax.set_yscale("log")
        ax.set_ylim(CI_FLOOR, CI_CEIL)
        ax.set_xlabel("Component (sorted desc by mean CI)")
        ax.set_ylabel("mean CI")
        ax.set_title(f"{SITE_TITLES[key]}   ({int((v > 0).sum())}/{v.size} nonzero)", fontsize=10)
    fig.colorbar(mesh, ax=axs[:, 0], label="fraction of component band", shrink=0.6)
    fig.suptitle(f"mean CI density heatmap — {n_tokens:,} tokens", fontsize=12)
    out = OUT_DIR / f"ci_mean_heatmap_{label}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out)


for n_tokens, label in SNAPSHOTS:
    render(n_tokens, label)
