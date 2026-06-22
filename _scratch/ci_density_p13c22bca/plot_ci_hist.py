"""CI-density heatmap from the unreduced per-token CI histogram (harvest_ci_hist.py).

x = component, density-ordered (sorted desc by mean CI), binned into NX columns.
y = per-token CI, log scale, bounded to [Y_LO, 1] (the active region; the exact-0
    underflow band and the near-empty 1e-9..Y_LO continuum sit below the axis).
cell(x, y) = density of that component-band's per-token CI observations in CI-band y.

Overlaid: the sorted per-component mean CI (sum / n_tokens), per-column-averaged, on a
twin axis spanning [1e-9, 1] so the sampling-floor tail (~1e-7) stays on-scale even though
the density axis starts at Y_LO.

Two normalisations, since ~99.8% of all (token, component) observations are inactive:
  - "full": per-column over ALL observations (the literal density; the inactive band carries
    almost all mass, so colour is log-scaled to expose the active fan).
  - "active": per-column over ACTIVE observations only (CI > 0) — the shape of the
    CI-when-active distribution.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

OUT_DIR = Path("/mnt/home/oli/claude-slack/data/workspaces/3229052cd398/mean_ci_out")
LABELS = ["131k", "2.1M", "33.6M"]
SORT_LABEL = "33.6M"  # rank components by mean CI at the largest token count (stable tail order)
SITE_KEYS = ["layers_18_mlp_gate_proj", "layers_18_mlp_up_proj", "layers_18_mlp_down_proj"]
SITE_TITLES = {k: k.replace("_", ".") for k in SITE_KEYS}
NX = 256
Y_LO = 1e-6  # density-axis lower bound (active region); overlaid mean-CI uses [MEAN_LO, 1]
MEAN_LO = 1e-9
MEAN_COLOR = "#34d8eb"

_ref = np.load(OUT_DIR / f"ci_hist_{SORT_LABEL}.npz")
SORT_ORDER = {k: np.argsort(_ref[f"sum__{k}"])[::-1] for k in SITE_KEYS}


def column_bin(counts_sorted: np.ndarray) -> np.ndarray:
    """(C, N_BINS) sorted by mean CI desc -> (NX, N_BINS) summed over equal rank-blocks."""
    c = counts_sorted.shape[0]
    edges = np.linspace(0, c, NX + 1).astype(int)
    return np.stack([counts_sorted[edges[i] : edges[i + 1]].sum(0) for i in range(NX)])


def column_mean(values_sorted: np.ndarray) -> np.ndarray:
    """(C,) sorted desc -> (NX,) averaged over equal rank-blocks."""
    c = values_sorted.shape[0]
    edges = np.linspace(0, c, NX + 1).astype(int)
    return np.array([values_sorted[edges[i] : edges[i + 1]].mean() for i in range(NX)])


def render(label: str, mode: str, color_scale: str) -> None:
    data = np.load(OUT_DIR / f"ci_hist_{label}.npz")
    y_edges = data["y_edges"]
    n_tokens = int(data["n_tokens"])
    ci_floor = float(data["ci_floor"])
    plot_y_edges = np.concatenate([[ci_floor / 10], y_edges])
    x_edges = np.linspace(0, 49152, NX + 1)
    visible_row = plot_y_edges[1:] > Y_LO  # rows whose upper edge is inside the [Y_LO, 1] view

    fig, axs = plt.subplots(len(SITE_KEYS), 1, figsize=(9, 3.6 * len(SITE_KEYS)), squeeze=False, constrained_layout=True)
    mesh = None
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    for ax, key in zip(axs[:, 0], SITE_KEYS, strict=True):
        order = SORT_ORDER[key]
        col = column_bin(data[f"counts__{key}"][order]).astype(np.float64)
        active_frac = 1.0 - col[:, 0] / col.sum(1)
        mean_ci_col = column_mean(data[f"sum__{key}"][order] / n_tokens)
        if mode == "full":
            density = col / col.sum(1, keepdims=True)
            vmin = 1e-7
        else:
            active = col[:, 1:]
            density = np.divide(active, active.sum(1, keepdims=True), out=np.zeros_like(active), where=active.sum(1, keepdims=True) > 0)
            density = np.concatenate([np.zeros((NX, 1)), density], axis=1)
            vmin = 1e-3
        cmap = matplotlib.colormaps["magma"].copy()
        cmap.set_bad(cmap(0.0))
        if color_scale == "log":
            plot_density = density
            norm = LogNorm(vmin=vmin, vmax=1.0)
        else:
            col_vis_max = np.where(visible_row, density, 0.0).max(axis=1, keepdims=True)
            plot_density = np.divide(density, col_vis_max, out=np.zeros_like(density), where=col_vis_max > 0)
            norm = Normalize(vmin=0.0, vmax=1.0)
        masked = np.ma.masked_where(plot_density <= 0, plot_density)
        mesh = ax.pcolormesh(x_edges, plot_y_edges, masked.T, cmap=cmap, norm=norm, shading="flat")
        ax.set_yscale("log")
        ax.set_ylim(Y_LO, 1.0)
        tw = ax.twinx()
        tw.plot(xc, mean_ci_col, color=MEAN_COLOR, lw=1.0)
        tw.set_yscale("log")
        tw.set_ylim(MEAN_LO, 1.0)
        tw.set_ylabel("mean CI (sorted)", color=MEAN_COLOR, fontsize=8)
        tw.tick_params(labelsize=7, colors=MEAN_COLOR)
        ax.set_xlabel(f"Component (sorted desc by mean CI @ {SORT_LABEL})")
        ax.set_ylabel("per-token CI")
        ax.set_title(f"{SITE_TITLES[key]}   (mean active frac {active_frac.mean():.4f})", fontsize=10)
    if color_scale == "linear":
        cbar_label = "per-column density, rescaled so col max = 1"
    else:
        cbar_label = "density over all obs (col-norm)" if mode == "full" else "density over active obs (col-norm)"
    fig.colorbar(mesh, ax=axs[:, 0], label=cbar_label, shrink=0.6)
    title = "per-token CI density" + (" — active-conditional" if mode == "active" else "")
    scale_note = "log color" if color_scale == "log" else "linear color (per-column max = 1)"
    fig.suptitle(f"{title} — {n_tokens:,} tokens   (cyan = sorted mean CI, right axis {MEAN_LO:g}–1; {scale_note})", fontsize=12)
    suffix = "" if color_scale == "log" else "_lin"
    out = OUT_DIR / f"ci_hist_heatmap_{mode}{suffix}_{label}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out)


for label in LABELS:
    for mode in ("full", "active"):
        for color_scale in ("log", "linear"):
            render(label, mode, color_scale)
