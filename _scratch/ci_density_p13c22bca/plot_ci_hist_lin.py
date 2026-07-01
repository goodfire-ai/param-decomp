"""CI-density heatmap from the LINEAR-binned per-token CI histogram (harvest_ci_hist.py).

Same layout as plot_ci_hist.py, but the y-axis bins are uniform 0.025-wide bands over
[0, 1] (not log-spaced), rendered on a LINEAR y-axis — so the CI-when-active shape is read
directly, without the low-CI decades compressing against 0.

x = component, density-ordered (sorted desc by mean CI), binned into NX columns.
y = per-token CI, linear [0, 1], 40 bins of 0.025 (top bin includes CI = 1).
cell(x, y) = per-column density in CI-band y, rescaled so each column's max = 1.

Overlaid: the sorted per-component mean CI (sum / n_tokens), per-column-averaged, on a twin
log axis [1e-9, 1] so the sampling-floor tail stays on-scale.

Two normalisations:
  - "full": per-column over ALL observations (bin 0 = [0, 0.025) is dominated by the ~99.8%
    exact-0 inactive mass).
  - "active": per-column over ACTIVE observations only (CI > 0) — the exact-0 count is
    subtracted from bin 0 using the harvested `zero__<site>` counter.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

OUT_DIR = Path("/mnt/home/oli/claude-slack/data/workspaces/3229052cd398/mean_ci_out")
LABELS = ["131k", "2.1M", "33.6M"]
SORT_LABEL = "33.6M"
SITE_KEYS = ["layers_18_mlp_gate_proj", "layers_18_mlp_up_proj", "layers_18_mlp_down_proj"]
SITE_TITLES = {k: k.replace("_", ".") for k in SITE_KEYS}
NX = 256
MEAN_LO = 1e-9
MEAN_COLOR = "#34d8eb"

_ref = np.load(OUT_DIR / f"ci_hist_lin_{SORT_LABEL}.npz")
SORT_ORDER = {k: np.argsort(_ref[f"sum__{k}"])[::-1] for k in SITE_KEYS}


def column_bin(counts_sorted: np.ndarray) -> np.ndarray:
    """(C, N_BINS) sorted by mean CI desc -> (NX, N_BINS) summed over equal rank-blocks."""
    c = counts_sorted.shape[0]
    edges = np.linspace(0, c, NX + 1).astype(int)
    return np.stack([counts_sorted[edges[i] : edges[i + 1]].sum(0) for i in range(NX)])


def column_sum(values_sorted: np.ndarray) -> np.ndarray:
    """(C,) sorted desc -> (NX,) summed over equal rank-blocks."""
    c = values_sorted.shape[0]
    edges = np.linspace(0, c, NX + 1).astype(int)
    return np.array([values_sorted[edges[i] : edges[i + 1]].sum() for i in range(NX)])


def column_mean(values_sorted: np.ndarray) -> np.ndarray:
    """(C,) sorted desc -> (NX,) averaged over equal rank-blocks."""
    c = values_sorted.shape[0]
    edges = np.linspace(0, c, NX + 1).astype(int)
    return np.array([values_sorted[edges[i] : edges[i + 1]].mean() for i in range(NX)])


def render(label: str, mode: str) -> None:
    data = np.load(OUT_DIR / f"ci_hist_lin_{label}.npz")
    y_edges = data["y_edges"]
    n_tokens = int(data["n_tokens"])
    x_edges = np.linspace(0, 49152, NX + 1)

    fig, axs = plt.subplots(len(SITE_KEYS), 1, figsize=(9, 3.6 * len(SITE_KEYS)), squeeze=False, constrained_layout=True)
    mesh = None
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    for ax, key in zip(axs[:, 0], SITE_KEYS, strict=True):
        order = SORT_ORDER[key]
        col = column_bin(data[f"counts__{key}"][order]).astype(np.float64)
        zero_col = column_sum(data[f"zero__{key}"][order]).astype(np.float64)
        mean_ci_col = column_mean(data[f"sum__{key}"][order] / n_tokens)
        if mode == "active":
            assert (col[:, 0] >= zero_col - 1e-6).all(), "exact-0 count exceeds bin-0 count"
            col = col.copy()
            col[:, 0] = np.maximum(col[:, 0] - zero_col, 0.0)
        active_frac = 1.0 - zero_col.sum() / column_bin(data[f"counts__{key}"][order]).sum()
        density = np.divide(col, col.sum(1, keepdims=True), out=np.zeros_like(col), where=col.sum(1, keepdims=True) > 0)
        col_max = density.max(axis=1, keepdims=True)
        plot_density = np.divide(density, col_max, out=np.zeros_like(density), where=col_max > 0)
        cmap = matplotlib.colormaps["magma"].copy()
        cmap.set_bad(cmap(0.0))
        masked = np.ma.masked_where(plot_density <= 0, plot_density)
        mesh = ax.pcolormesh(x_edges, y_edges, masked.T, cmap=cmap, norm=Normalize(0.0, 1.0), shading="flat")
        ax.set_ylim(0.0, 1.0)
        tw = ax.twinx()
        tw.plot(xc, mean_ci_col, color=MEAN_COLOR, lw=1.0)
        tw.set_yscale("log")
        tw.set_ylim(MEAN_LO, 1.0)
        tw.set_ylabel("mean CI (sorted)", color=MEAN_COLOR, fontsize=8)
        tw.tick_params(labelsize=7, colors=MEAN_COLOR)
        ax.set_xlabel(f"Component (sorted desc by mean CI @ {SORT_LABEL})")
        ax.set_ylabel("per-token CI")
        ax.set_title(f"{SITE_TITLES[key]}   (mean active frac {active_frac.mean():.4f})", fontsize=10)
    cbar_label = "per-column density, rescaled so col max = 1"
    fig.colorbar(mesh, ax=axs[:, 0], label=cbar_label, shrink=0.6)
    title = "per-token CI density (linear bins)" + (" — active-conditional" if mode == "active" else "")
    fig.suptitle(
        f"{title} — {n_tokens:,} tokens   (cyan = sorted mean CI, right axis {MEAN_LO:g}–1; "
        f"linear y, 40 bins × 0.025, linear color per-column max = 1)",
        fontsize=11,
    )
    out = OUT_DIR / f"ci_hist_heatmap_{mode}_linbins_{label}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out)


for label in LABELS:
    for mode in ("full", "active"):
        render(label, mode)
