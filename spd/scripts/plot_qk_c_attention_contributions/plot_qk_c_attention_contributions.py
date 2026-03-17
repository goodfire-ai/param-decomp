"""Plot weight-only attention contribution heatmaps between q and k subcomponents.

For each layer and relative position offset, produces a single grid containing:
  - First cell: mean (across heads) q·k attention contributions
  - Remaining cells: one per query head

U vectors are scaled by sign(mean_component_activation) * ||V|| to correct for components
whose activations are typically negative. Contributions are then z-scored per head (subtract
mean, divide by std across all offsets and pairs), making heads directly comparable.

Usage:
    python -m spd.scripts.plot_qk_c_attention_contributions.plot_qk_c_attention_contributions \
        wandb:goodfire/spd/runs/<run_id>
"""

import math
from dataclasses import dataclass
from pathlib import Path

import fire
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from spd.harvest.repo import HarvestRepo
from spd.harvest.schemas import ComponentSummary
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import LinearComponents
from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from spd.scripts.rope_aware_qk import compute_qk_rope_coefficients, evaluate_qk_at_offsets
from spd.spd_types import ModelPath
from spd.utils.wandb_utils import parse_wandb_run_path

SCRIPT_DIR = Path(__file__).parent
MIN_MEAN_CI = 0.01
DEFAULT_OFFSETS = tuple(range(17))


def _get_alive_indices(
    summary: dict[str, ComponentSummary], module_path: str, min_mean_ci: float
) -> list[int]:
    """Return component indices for a module sorted by CI descending, filtered by threshold."""
    components = [
        (s.component_idx, s.mean_activations["causal_importance"])
        for s in summary.values()
        if s.layer == module_path and s.mean_activations["causal_importance"] > min_mean_ci
    ]
    components.sort(key=lambda t: t[1], reverse=True)
    return [idx for idx, _ in components]


def _compute_per_head_attention_contributions(
    q_component: LinearComponents,
    k_component: LinearComponents,
    q_alive: list[int],
    k_alive: list[int],
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
    rotary_cos: torch.Tensor,
    rotary_sin: torch.Tensor,
    offsets: tuple[int, ...],
    summary: dict[str, ComponentSummary],
    q_module_path: str,
    k_module_path: str,
) -> NDArray[np.floating]:
    """Compute (n_offsets, n_q_heads, n_q_alive, n_k_alive) per-head attention contributions.

    Each head's contributions are z-scored (subtract mean, divide by std) across all
    (offset, q, k) values, making heads comparable.

    U vectors are scaled by sign(mean_activation) * ||V|| so that the weight-only dot
    product reflects the typical sign of the actual data-dependent contribution.
    """
    V_q_norms = torch.linalg.norm(q_component.V[:, q_alive], dim=0).float()  # (n_q_alive,)
    V_k_norms = torch.linalg.norm(k_component.V[:, k_alive], dim=0).float()  # (n_k_alive,)

    # Sign-correct: scale by sign of mean component activation so weight-only dot products
    # reflect the typical sign of the actual (data-dependent) contribution.
    q_signs = torch.tensor(
        [
            1.0
            if summary[f"{q_module_path}:{c}"].mean_activations["component_activation"] >= 0
            else -1.0
            for c in q_alive
        ]
    )
    k_signs = torch.tensor(
        [
            1.0
            if summary[f"{k_module_path}:{c}"].mean_activations["component_activation"] >= 0
            else -1.0
            for c in k_alive
        ]
    )

    U_q = q_component.U[q_alive].float() * (V_q_norms * q_signs)[:, None]
    U_q = U_q.reshape(len(q_alive), n_q_heads, head_dim)

    U_k = k_component.U[k_alive].float() * (V_k_norms * k_signs)[:, None]
    U_k = U_k.reshape(len(k_alive), n_kv_heads, head_dim)

    g = n_q_heads // n_kv_heads
    U_k_expanded = U_k.repeat_interleave(g, dim=1)  # (n_k_alive, n_q_heads, head_dim)

    head_results = []
    for h in range(n_q_heads):
        A, B = compute_qk_rope_coefficients(U_q[:, h, :], U_k_expanded[:, h, :])
        W_h = evaluate_qk_at_offsets(A, B, rotary_cos, rotary_sin, offsets)
        head_results.append(W_h)  # (n_offsets, n_q, n_k)

    # (n_heads, n_offsets, n_q, n_k) -> (n_offsets, n_heads, n_q, n_k)
    W = torch.stack(head_results).permute(1, 0, 2, 3)

    # Z-score per head: pool all (offset, q, k) values for each head, then subtract
    # mean and divide by std. This makes contributions comparable across heads (measured
    # in standard deviations above/below the head's average pair contribution).
    mean_h = W.mean(dim=(0, 2, 3), keepdim=True)  # (1, n_heads, 1, 1)
    std_h = W.std(dim=(0, 2, 3), keepdim=True)  # (1, n_heads, 1, 1)
    W_zscore = (W - mean_h) / std_h

    return W_zscore.cpu().numpy()


def _plot_heatmaps(
    W_per_head: NDArray[np.floating],
    q_alive: list[int],
    k_alive: list[int],
    n_q_heads: int,
    layer_idx: int,
    run_id: str,
    out_dir: Path,
    offset: int,
    vmax: float,
) -> None:
    n_cells = n_q_heads + 1  # mean + per-head
    n_cols = math.ceil(math.sqrt(n_cells))
    n_rows = math.ceil(n_cells / n_cols)
    n_q, n_k = W_per_head.shape[1], W_per_head.shape[2]

    W_mean = W_per_head.mean(axis=0)

    cell_w = max(4, n_k * 0.15)
    cell_h = max(3, n_q * 0.15)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(cell_w * n_cols, cell_h * n_rows), squeeze=False
    )

    all_data = [W_mean] + [W_per_head[h] for h in range(n_q_heads)]
    titles = ["Mean"] + [f"H{h}" for h in range(n_q_heads)]

    for cell, (data, title) in enumerate(zip(all_data, titles, strict=True)):
        row, col = divmod(cell, n_cols)
        ax = axes[row, col]
        im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        ax.set_title(title, fontsize=10, fontweight="bold" if cell == 0 else "normal")
        ax.set_yticks(range(n_q))
        ax.set_yticklabels([f"Q C{idx}" for idx in q_alive], fontsize=5)
        ax.set_xticks(range(n_k))
        ax.set_xticklabels([f"K C{idx}" for idx in k_alive], fontsize=5, rotation=90)

    # Hide unused cells
    for i in range(n_cells, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        f"{run_id}  |  Layer {layer_idx} \u2014 q\u00b7k attention contributions"
        f" (\u0394={offset})  (ci>{MIN_MEAN_CI})",
        fontsize=14,
        fontweight="bold",
    )
    fig.subplots_adjust(hspace=0.3, wspace=0.4)

    path = out_dir / f"layer{layer_idx}_qk_attention_contributions_offset{offset}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def _plot_diff_heatmaps(
    D_per_head: NDArray[np.floating],
    q_alive: list[int],
    k_alive: list[int],
    n_q_heads: int,
    layer_idx: int,
    run_id: str,
    out_dir: Path,
    offset: int,
    vmax: float,
) -> None:
    """Plot W(Δ=offset) - W(Δ=0) heatmaps."""
    n_cells = n_q_heads + 1
    n_cols = math.ceil(math.sqrt(n_cells))
    n_rows = math.ceil(n_cells / n_cols)
    n_q, n_k = D_per_head.shape[1], D_per_head.shape[2]

    D_mean = D_per_head.mean(axis=0)

    cell_w = max(4, n_k * 0.15)
    cell_h = max(3, n_q * 0.15)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(cell_w * n_cols, cell_h * n_rows), squeeze=False
    )

    all_data = [D_mean] + [D_per_head[h] for h in range(n_q_heads)]
    titles = ["Mean"] + [f"H{h}" for h in range(n_q_heads)]

    for cell, (data, title) in enumerate(zip(all_data, titles, strict=True)):
        row, col = divmod(cell, n_cols)
        ax = axes[row, col]
        im = ax.imshow(data, aspect="auto", cmap="PiYG", vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        ax.set_title(title, fontsize=10, fontweight="bold" if cell == 0 else "normal")
        ax.set_yticks(range(n_q))
        ax.set_yticklabels([f"Q C{idx}" for idx in q_alive], fontsize=5)
        ax.set_xticks(range(n_k))
        ax.set_xticklabels([f"K C{idx}" for idx in k_alive], fontsize=5, rotation=90)

    for i in range(n_cells, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        f"{run_id}  |  Layer {layer_idx} \u2014 q\u00b7k attention diff"
        f" (\u0394={offset} \u2212 \u0394=0)  (ci>{MIN_MEAN_CI})",
        fontsize=14,
        fontweight="bold",
    )
    fig.subplots_adjust(hspace=0.3, wspace=0.4)

    path = out_dir / f"layer{layer_idx}_qk_attention_diff_offset{offset}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def _plot_heatmaps_per_head(
    W: NDArray[np.floating],
    q_alive: list[int],
    k_alive: list[int],
    n_q_heads: int,
    layer_idx: int,
    run_id: str,
    out_dir: Path,
    offsets: tuple[int, ...],
    vmax: float,
) -> None:
    """For each head (and Mean), plot a grid of heatmaps across all offsets."""
    n_offsets = len(offsets)
    n_cols = math.ceil(math.sqrt(n_offsets))
    n_rows = math.ceil(n_offsets / n_cols)
    n_q, n_k = len(q_alive), len(k_alive)

    per_head_dir = out_dir / "heatmap_offsets_per_head"
    per_head_dir.mkdir(parents=True, exist_ok=True)

    cell_w = max(4, n_k * 0.15)
    cell_h = max(3, n_q * 0.15)

    # W shape: (n_offsets, n_q_heads, n_q, n_k)
    # Build list of (label, data) where data is (n_offsets, n_q, n_k)
    head_series: list[tuple[str, NDArray[np.floating]]] = [
        ("Mean", W.mean(axis=1)),
    ] + [(f"H{h}", W[:, h]) for h in range(n_q_heads)]

    for label, data in head_series:
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(cell_w * n_cols, cell_h * n_rows), squeeze=False
        )

        for cell, offset in enumerate(offsets):
            row, col = divmod(cell, n_cols)
            ax = axes[row, col]
            im = ax.imshow(data[cell], aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            ax.set_title(f"\u0394={offset}", fontsize=10)
            ax.set_yticks(range(n_q))
            ax.set_yticklabels([f"Q C{idx}" for idx in q_alive], fontsize=5)
            ax.set_xticks(range(n_k))
            ax.set_xticklabels([f"K C{idx}" for idx in k_alive], fontsize=5, rotation=90)

        for i in range(n_offsets, n_rows * n_cols):
            row, col = divmod(i, n_cols)
            axes[row, col].set_visible(False)

        fig.suptitle(
            f"{run_id}  |  Layer {layer_idx} {label} \u2014 q\u00b7k attention contributions"
            f"  (ci>{MIN_MEAN_CI})",
            fontsize=14,
            fontweight="bold",
        )
        fig.subplots_adjust(hspace=0.3, wspace=0.4)

        path = per_head_dir / f"layer{layer_idx}_qk_attention_{label.lower()}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {path}")


HEAD_CMAPS = [
    "Reds",
    "Blues",
    "Greens",
    "Oranges",
    "Purples",
    "Greys",
    "YlOrBr",
    "BuPu",
    "PuRd",
    "GnBu",
    "OrRd",
    "YlGn",
]


def _plot_head_vs_sum_scatter(
    W: NDArray[np.floating],
    n_q_heads: int,
    layer_idx: int,
    run_id: str,
    out_dir: Path,
    offsets: tuple[int, ...],
) -> None:
    """Scatter: x = mean-across-heads contribution, y = per-head contribution.

    Each head uses a distinct sequential colormap; within that colormap each
    offset maps to a different shade (darker = larger offset index).
    """
    # W shape: (n_offsets, n_q_heads, n_q, n_k)
    W_mean = W.mean(axis=1)  # (n_offsets, n_q, n_k)
    n_offsets = len(offsets)

    # Map offset indices to colormap values in [0.3, 0.9] so no shade is too light
    norm = mcolors.Normalize(vmin=-0.5, vmax=n_offsets - 0.5)
    offset_vals = [0.3 + 0.6 * norm(i) for i in range(n_offsets)]

    fig, ax = plt.subplots(figsize=(8, 8))

    for h in range(n_q_heads):
        cmap = plt.get_cmap(HEAD_CMAPS[h % len(HEAD_CMAPS)])
        for oi, offset in enumerate(offsets):
            x = W_mean[oi].ravel()
            y = W[oi, h].ravel()
            color = cmap(offset_vals[oi])
            ax.scatter(
                x,
                y,
                s=6,
                alpha=0.5,
                color=color,
                label=f"H{h} \u0394={offset}",
                edgecolors="none",
                rasterized=True,
            )

    # x=y reference line
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.5, alpha=0.4)
    ax.set_xlim((lo, hi))
    ax.set_ylim((lo, hi))

    ax.set_xlabel("Mean (all heads) attention contribution")
    ax.set_ylabel("Per-head attention contribution")
    ax.set_aspect("equal")

    # Legend: one entry per head (color patch at mid-shade), one per offset (grey shades)
    head_handles = []
    for h in range(n_q_heads):
        cmap = plt.get_cmap(HEAD_CMAPS[h % len(HEAD_CMAPS)])
        head_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=cmap(0.6),
                markersize=6,
                label=f"H{h}",
            )
        )
    offset_handles = []
    for oi, offset in enumerate(offsets):
        grey = str(1.0 - offset_vals[oi])  # darker for higher offset index
        offset_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=grey,
                markersize=6,
                label=f"\u0394={offset}",
            )
        )

    leg1 = ax.legend(
        handles=head_handles,
        loc="upper left",
        fontsize=7,
        title="Head",
        title_fontsize=8,
        framealpha=0.8,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=offset_handles,
        loc="lower right",
        fontsize=7,
        title="Offset",
        title_fontsize=8,
        framealpha=0.8,
    )

    fig.suptitle(
        f"{run_id}  |  Layer {layer_idx} \u2014 per-head vs mean q\u00b7k contributions"
        f"  (ci>{MIN_MEAN_CI})",
        fontsize=12,
        fontweight="bold",
    )

    scatter_dir = out_dir / "scatter_head_vs_sum"
    scatter_dir.mkdir(parents=True, exist_ok=True)
    path = scatter_dir / f"layer{layer_idx}_head_vs_sum_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def _plot_pair_lines(
    W_mean: NDArray[np.floating],
    offsets: tuple[int, ...],
    q_alive: list[int],
    k_alive: list[int],
    layer_idx: int,
    run_id: str,
    out_dir: Path,
    top_n_pairs: int,
) -> None:
    """Line plot of attention contribution vs offset for top (q_c, k_c) pairs.

    Top-N pairs are plotted in color; a wider set of background pairs is plotted
    in faint gray for context.
    """
    _n_offsets, n_q, n_k = W_mean.shape
    peak_abs = np.abs(W_mean).max(axis=0)  # (n_q, n_k)
    flat_ranked = np.argsort(peak_abs.ravel())[::-1]

    top_pairs = [divmod(int(idx), n_k) for idx in flat_ranked[:top_n_pairs]]
    top_pair_set = set(top_pairs)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = list(offsets)

    plotted_gray = False
    for qi in range(n_q):
        for ki in range(n_k):
            if (qi, ki) in top_pair_set:
                continue
            label = "other" if not plotted_gray else None
            ax.plot(x, W_mean[:, qi, ki], color="0.80", linewidth=0.8, alpha=0.45, label=label)
            plotted_gray = True

    for qi, ki in top_pairs:
        ax.plot(
            x,
            W_mean[:, qi, ki],
            marker="o",
            markersize=3,
            label=f"Q C{q_alive[qi]} \u2192 K C{k_alive[ki]}",
        )

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.set_xlabel("Offset (\u0394)")
    ax.set_ylabel("Attention contribution (mean across heads)")
    ax.set_xticks(x)
    ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.02, 0.5))

    fig.suptitle(
        f"{run_id}  |  Layer {layer_idx} \u2014 q\u00b7k pair contributions vs offset"
        f"  (top {len(top_pairs)} pairs, ci>{MIN_MEAN_CI})",
        fontsize=12,
        fontweight="bold",
    )

    lines_dir = out_dir / "lines"
    lines_dir.mkdir(parents=True, exist_ok=True)
    path = lines_dir / f"layer{layer_idx}_qk_pair_lines.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def _plot_pair_lines_per_head(
    W: NDArray[np.floating],
    offsets: tuple[int, ...],
    q_alive: list[int],
    k_alive: list[int],
    layer_idx: int,
    run_id: str,
    out_dir: Path,
    top_n: int,
) -> None:
    """Line plot of per-head attention contribution vs offset for top (head, q_c, k_c) triples."""
    peak_abs = np.abs(W).max(axis=0)  # (n_q_heads, n_q, n_k)

    flat_indices = np.argsort(peak_abs.ravel())[::-1][:top_n]
    n_k = len(k_alive)
    triples = []
    for idx in flat_indices:
        h, rem = divmod(int(idx), len(q_alive) * n_k)
        qi, ki = divmod(rem, n_k)
        triples.append((h, qi, ki))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = list(offsets)

    for h, qi, ki in triples:
        y = W[:, h, qi, ki]
        ax.plot(
            x,
            y,
            marker="o",
            markersize=3,
            label=f"H{h}: Q C{q_alive[qi]} \u2192 K C{k_alive[ki]}",
        )

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.set_xlabel("Offset (\u0394)")
    ax.set_ylabel("Attention contribution (single head)")
    ax.set_xticks(x)
    ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.02, 0.5))

    fig.suptitle(
        f"{run_id}  |  Layer {layer_idx} \u2014 per-head q\u00b7k pair contributions vs offset"
        f"  (top {len(triples)}, ci>{MIN_MEAN_CI})",
        fontsize=12,
        fontweight="bold",
    )

    lines_dir = out_dir / "lines_per_head"
    lines_dir.mkdir(parents=True, exist_ok=True)
    path = lines_dir / f"layer{layer_idx}_qk_pair_lines_per_head.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def _plot_pair_lines_single_head(
    W: NDArray[np.floating],
    offsets: tuple[int, ...],
    q_alive: list[int],
    k_alive: list[int],
    n_q_heads: int,
    layer_idx: int,
    run_id: str,
    out_dir: Path,
    top_n: int,
) -> None:
    """2x3 grid of per-head line plots with consistent pair colors across heads.

    Global top-K pairs (ranked by peak |W| across all heads and offsets) are plotted in
    color; each head's remaining local top-K pairs are plotted in faint gray.
    """
    # W shape: (n_offsets, n_q_heads, n_q, n_k)
    n_k = len(k_alive)
    x = list(offsets)

    # Global top-K: rank (q, k) pairs by peak absolute value across all heads and offsets
    global_peak = np.abs(W).max(axis=(0, 1))  # (n_q, n_k)
    global_flat = np.argsort(global_peak.ravel())[::-1][:top_n]
    global_pairs = [divmod(int(idx), n_k) for idx in global_flat]
    global_pair_set = set(global_pairs)

    cmap = plt.get_cmap("tab20")
    pair_colors = {pair: cmap(i % 20) for i, pair in enumerate(global_pairs)}

    n_cols = 3
    n_rows = math.ceil(n_q_heads / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 5 * n_rows), squeeze=False)

    for h in range(n_q_heads):
        row, col = divmod(h, n_cols)
        ax = axes[row, col]

        W_h = W[:, h]  # (n_offsets, n_q, n_k)
        n_q = W_h.shape[1]

        plotted_gray = False
        for qi in range(n_q):
            for ki in range(n_k):
                if (qi, ki) in global_pair_set:
                    continue
                label = "other" if not plotted_gray else None
                ax.plot(x, W_h[:, qi, ki], color="0.80", linewidth=0.8, alpha=0.45, label=label)
                plotted_gray = True

        # Plot global top-K pairs in color
        for qi, ki in global_pairs:
            ax.plot(
                x,
                W_h[:, qi, ki],
                color=pair_colors[(qi, ki)],
                marker="o",
                markersize=3,
                label=f"Q C{q_alive[qi]} \u2192 K C{k_alive[ki]}",
            )

        total = W_h.sum(axis=(1, 2))  # (n_offsets,)
        ax.plot(x, total, color="black", linewidth=2, label="sum (all pairs)")

        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.set_xlabel("Offset (\u0394)")
        ax.set_ylabel("Attention contribution")
        ax.set_title(f"H{h}", fontsize=11, fontweight="bold")
        ax.set_xticks(x)

    # Hide unused cells
    for i in range(n_q_heads, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].set_visible(False)

    # Shared legend from first subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=6,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        ncol=1,
    )

    fig.suptitle(
        f"{run_id}  |  Layer {layer_idx} \u2014 per-head q\u00b7k pair contributions vs offset"
        f"  (top {top_n}, ci>{MIN_MEAN_CI})",
        fontsize=13,
        fontweight="bold",
    )
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    lines_dir = out_dir / "lines_single_head"
    lines_dir.mkdir(parents=True, exist_ok=True)
    path = lines_dir / f"layer{layer_idx}_qk_pair_lines_grid.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def _plot_pair_lines_combined(
    W: NDArray[np.floating],
    offsets: tuple[int, ...],
    q_alive: list[int],
    k_alive: list[int],
    n_q_heads: int,
    layer_idx: int,
    out_dir: Path,
    top_n: int,
) -> None:
    """Combined mean + per-head line plots in a single figure.

    Top row (spanning 2 columns): mean across heads.
    Rows 1-3 (2 columns each): one subplot per head.
    Global top-N pairs share consistent colors across all subplots.
    """
    n_k = len(k_alive)
    x = list(offsets)
    W_mean = W.mean(axis=1)  # (n_offsets, n_q, n_k)
    n_q = W_mean.shape[1]

    # Global top-N: rank (q, k) pairs by peak |W| across all heads and offsets
    global_peak = np.abs(W).max(axis=(0, 1))  # (n_q, n_k)
    global_flat = np.argsort(global_peak.ravel())[::-1][:top_n]
    global_pairs = [divmod(int(idx), n_k) for idx in global_flat]
    global_pair_set = set(global_pairs)

    cmap = plt.get_cmap("tab20")
    pair_colors = {pair: cmap(i % 20) for i, pair in enumerate(global_pairs)}

    fig = plt.figure(figsize=(12, 17))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.12)
    gs_top = outer[0].subgridspec(1, 2, wspace=0.12)
    gs_bottom = outer[1].subgridspec(3, 2, hspace=0.15, wspace=0.12)
    ax_mean = fig.add_subplot(gs_top[0, 0])
    ax_legend = fig.add_subplot(gs_top[0, 1])
    head_axes = [fig.add_subplot(gs_bottom[h // 2, h % 2]) for h in range(n_q_heads)]
    all_axes = [ax_mean] + head_axes

    # --- Mean subplot ---
    plotted_gray = False
    for qi in range(n_q):
        for ki in range(n_k):
            if (qi, ki) in global_pair_set:
                continue
            label = "Other" if not plotted_gray else None
            ax_mean.plot(x, W_mean[:, qi, ki], color="0.80", linewidth=0.8, alpha=0.45, label=label)
            plotted_gray = True

    for qi, ki in global_pairs:
        ax_mean.plot(
            x,
            W_mean[:, qi, ki],
            color=pair_colors[(qi, ki)],
            marker="o",
            markersize=3,
            label=f"q.{q_alive[qi]} \u2192 k.{k_alive[ki]}",
        )

    total_mean = W_mean.sum(axis=(1, 2))
    ax_mean.plot(x, total_mean, color="black", linewidth=2)

    ax_mean.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax_mean.set_xlabel("Offset")
    ax_mean.set_ylabel("Standardized Attention Contribution")
    ax_mean.set_title("All Heads", fontsize=11, fontweight="bold")
    ax_mean.set_xticks(x)

    # Legend in dedicated cell (0, 1): reorder so "other" is last, add "sum" entry
    handles, labels = ax_mean.get_legend_handles_labels()
    pair_entries = [(ha, la) for ha, la in zip(handles, labels, strict=True) if la != "Other"]
    other_entries = [(ha, la) for ha, la in zip(handles, labels, strict=True) if la == "Other"]
    sum_handle = plt.Line2D([0], [0], color="black", linewidth=2)
    ordered_handles = (
        [ha for ha, _ in pair_entries] + [ha for ha, _ in other_entries] + [sum_handle]
    )
    ordered_labels = (
        [la for _, la in pair_entries] + [la for _, la in other_entries] + ["Sum of All Pairs"]
    )
    ax_legend.axis("off")
    ax_legend.legend(ordered_handles, ordered_labels, fontsize=11, loc="center left", frameon=False)

    # --- Per-head subplots ---
    for h, ax in enumerate(head_axes):
        W_h = W[:, h]  # (n_offsets, n_q, n_k)

        for qi in range(n_q):
            for ki in range(n_k):
                if (qi, ki) in global_pair_set:
                    continue
                ax.plot(x, W_h[:, qi, ki], color="0.80", linewidth=0.8, alpha=0.45)

        for qi, ki in global_pairs:
            ax.plot(
                x,
                W_h[:, qi, ki],
                color=pair_colors[(qi, ki)],
                marker="o",
                markersize=3,
            )

        total = W_h.sum(axis=(1, 2))
        ax.plot(x, total, color="black", linewidth=2, label="sum (all pairs)")

        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.set_title(f"H{h}", fontsize=11, fontweight="bold")
        ax.set_xticks(x)

        # X-axis label and tick labels only on bottom row
        if h >= n_q_heads - 2:
            ax.set_xlabel("Offset")
        else:
            ax.set_xticklabels([])

        # Y-axis label only on left column
        if h % 2 == 0:
            ax.set_ylabel("Standardized Attention Contribution")

    # Shared y-axis limits across all subplots
    all_ylims = [ax.get_ylim() for ax in all_axes]
    ymin = min(lo for lo, _ in all_ylims)
    ymax = max(hi for _, hi in all_ylims)
    for ax in all_axes:
        ax.set_ylim(ymin, ymax)

    lines_dir = out_dir / "lines_combined"
    lines_dir.mkdir(parents=True, exist_ok=True)
    path = lines_dir / f"layer{layer_idx}_qk_pair_lines_combined.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


PLOT_TYPES = (
    "heatmaps",
    "heatmaps_per_head",
    "scatter",
    "diffs",
    "lines",
    "lines_per_head",
    "lines_single_head",
    "lines_combined",
)


@dataclass
class _LayerCache:
    W: NDArray[np.floating]
    q_alive: list[int]
    k_alive: list[int]
    offsets: tuple[int, ...]
    n_q_heads: int


def _cache_path(out_dir: Path, layer_idx: int) -> Path:
    return out_dir / "cache" / f"layer{layer_idx}.npz"


def _save_layer_cache(out_dir: Path, layer_idx: int, cache: _LayerCache) -> None:
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        _cache_path(out_dir, layer_idx),
        W=cache.W,
        q_alive=np.array(cache.q_alive),
        k_alive=np.array(cache.k_alive),
        offsets=np.array(cache.offsets),
        n_q_heads=np.array(cache.n_q_heads),
    )


def _load_layer_cache(out_dir: Path, layer_idx: int) -> _LayerCache | None:
    path = _cache_path(out_dir, layer_idx)
    if not path.exists():
        return None
    data = np.load(path)
    return _LayerCache(
        W=data["W"],
        q_alive=data["q_alive"].tolist(),
        k_alive=data["k_alive"].tolist(),
        offsets=tuple(data["offsets"].tolist()),
        n_q_heads=int(data["n_q_heads"]),
    )


def _compute_and_cache_all_layers(
    wandb_path: ModelPath,
    offsets: tuple[int, ...],
    out_dir: Path,
    run_id: str,
) -> None:
    run_info = SPDRunInfo.from_path(wandb_path)
    model = ComponentModel.from_run_info(run_info)
    model.eval()

    repo = HarvestRepo.open_most_recent(run_id)
    assert repo is not None, f"No harvest data found for {run_id}"
    summary = repo.get_summary()

    target_model = model.target_model
    assert isinstance(target_model, LlamaSimpleMLP)
    blocks = target_model._h
    assert not blocks[0].attn.rotary_adjacent_pairs, "RoPE math assumes non-adjacent pairs layout"
    head_dim = blocks[0].attn.head_dim
    n_q_heads = blocks[0].attn.n_head
    n_kv_heads = blocks[0].attn.n_key_value_heads
    n_layers = len(blocks)
    logger.info(
        f"Model: {n_layers} layers, head_dim={head_dim}, "
        f"n_q_heads={n_q_heads}, n_kv_heads={n_kv_heads}"
    )

    with torch.no_grad():
        for layer_idx in range(n_layers):
            q_path = f"h.{layer_idx}.attn.q_proj"
            k_path = f"h.{layer_idx}.attn.k_proj"

            q_component = model.components[q_path]
            k_component = model.components[k_path]
            assert isinstance(q_component, LinearComponents)
            assert isinstance(k_component, LinearComponents)

            q_alive = _get_alive_indices(summary, q_path, MIN_MEAN_CI)
            k_alive = _get_alive_indices(summary, k_path, MIN_MEAN_CI)
            logger.info(
                f"Layer {layer_idx}: {len(q_alive)} q components, {len(k_alive)} k components"
            )

            if not q_alive or not k_alive:
                logger.info(f"Layer {layer_idx}: skipping (no alive q or k components)")
                continue

            rotary_cos = blocks[layer_idx].attn.rotary_cos
            rotary_sin = blocks[layer_idx].attn.rotary_sin
            assert isinstance(rotary_cos, torch.Tensor)
            assert isinstance(rotary_sin, torch.Tensor)

            W = _compute_per_head_attention_contributions(
                q_component,
                k_component,
                q_alive,
                k_alive,
                n_q_heads,
                n_kv_heads,
                head_dim,
                rotary_cos,
                rotary_sin,
                offsets,
                summary,
                q_path,
                k_path,
            )

            cache = _LayerCache(
                W=W, q_alive=q_alive, k_alive=k_alive, offsets=offsets, n_q_heads=n_q_heads
            )
            _save_layer_cache(out_dir, layer_idx, cache)
            logger.info(f"Cached layer {layer_idx}")


def _get_layer_caches(
    wandb_path: ModelPath,
    offsets: tuple[int, ...],
    out_dir: Path,
    run_id: str,
    recompute: bool,
) -> list[tuple[int, _LayerCache]]:
    """Load caches if they exist and offsets match, otherwise recompute all layers."""
    if not recompute:
        caches: list[tuple[int, _LayerCache]] = []
        layer_idx = 0
        while True:
            cached = _load_layer_cache(out_dir, layer_idx)
            if cached is None:
                break
            if cached.offsets == offsets:
                caches.append((layer_idx, cached))
            else:
                logger.info(f"Cache offsets mismatch at layer {layer_idx}, recomputing all")
                caches = []
                break
            layer_idx += 1

        if caches:
            logger.info(f"Loaded {len(caches)} layers from cache")
            return caches

    _compute_and_cache_all_layers(wandb_path, offsets, out_dir, run_id)

    caches = []
    layer_idx = 0
    while True:
        cached = _load_layer_cache(out_dir, layer_idx)
        if cached is None:
            break
        caches.append((layer_idx, cached))
        layer_idx += 1
    return caches


def _plot_layer(
    cache: _LayerCache,
    layer_idx: int,
    run_id: str,
    out_dir: Path,
    top_n_pairs: int,
    plots: set[str],
) -> None:
    W = cache.W
    q_alive = cache.q_alive
    k_alive = cache.k_alive
    offsets = cache.offsets
    n_q_heads = cache.n_q_heads

    W_mean_all = W.mean(axis=1)  # (n_offsets, n_q, n_k)
    vmax = float(max(np.abs(W_mean_all).max(), np.abs(W).max())) or 1.0

    if "heatmaps" in plots:
        for offset_idx, offset in enumerate(offsets):
            _plot_heatmaps(
                W[offset_idx],
                q_alive,
                k_alive,
                n_q_heads,
                layer_idx,
                run_id,
                out_dir,
                offset,
                vmax,
            )

    if "heatmaps_per_head" in plots:
        _plot_heatmaps_per_head(
            W, q_alive, k_alive, n_q_heads, layer_idx, run_id, out_dir, offsets, vmax
        )

    if "scatter" in plots:
        _plot_head_vs_sum_scatter(W, n_q_heads, layer_idx, run_id, out_dir, offsets)

    if "diffs" in plots:
        assert offsets[0] == 0, "First offset must be 0 for diff computation"
        W_base = W[0]
        non_zero_offsets = [(idx, o) for idx, o in enumerate(offsets) if o != 0]
        if non_zero_offsets:
            diffs = np.stack([W[idx] - W_base for idx, _ in non_zero_offsets])
            D_mean_all = diffs.mean(axis=1)
            diff_vmax = float(max(np.abs(D_mean_all).max(), np.abs(diffs).max())) or 1.0

            diff_dir = out_dir / "diffs"
            diff_dir.mkdir(parents=True, exist_ok=True)
            for i, (_, offset) in enumerate(non_zero_offsets):
                _plot_diff_heatmaps(
                    diffs[i],
                    q_alive,
                    k_alive,
                    n_q_heads,
                    layer_idx,
                    run_id,
                    diff_dir,
                    offset,
                    diff_vmax,
                )

    if "lines" in plots:
        _plot_pair_lines(
            W_mean_all, offsets, q_alive, k_alive, layer_idx, run_id, out_dir, top_n_pairs
        )

    if "lines_per_head" in plots:
        _plot_pair_lines_per_head(
            W, offsets, q_alive, k_alive, layer_idx, run_id, out_dir, top_n_pairs
        )

    if "lines_single_head" in plots:
        _plot_pair_lines_single_head(
            W,
            offsets,
            q_alive,
            k_alive,
            n_q_heads,
            layer_idx,
            run_id,
            out_dir,
            top_n_pairs,
        )

    if "lines_combined" in plots:
        _plot_pair_lines_combined(
            W,
            offsets,
            q_alive,
            k_alive,
            n_q_heads,
            layer_idx,
            out_dir,
            top_n_pairs,
        )


def plot_qk_c_attention_contributions(
    wandb_path: ModelPath,
    offsets: tuple[int, ...] = DEFAULT_OFFSETS,
    top_n_pairs: int = 10,
    plots: str = "all",
    recompute: bool = False,
) -> None:
    """Plot weight-only attention contribution analyses.

    Args:
        wandb_path: WandB run path.
        offsets: Relative position offsets to evaluate.
        top_n_pairs: Number of top (q, k) pairs to highlight in line plots.
        plots: Comma-separated plot types, or "all". Options: heatmaps, heatmaps_per_head,
            scatter, diffs, lines, lines_per_head, lines_single_head.
        recompute: Force recomputation even if cached data exists.
    """
    plot_set: set[str] = (
        set(PLOT_TYPES) if plots == "all" else {s.strip() for s in plots.split(",")}
    )
    unknown = plot_set - set(PLOT_TYPES)
    assert not unknown, f"Unknown plot types: {unknown}. Valid: {PLOT_TYPES}"

    _entity, _project, run_id = parse_wandb_run_path(str(wandb_path))
    out_dir = SCRIPT_DIR / "out" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_caches = _get_layer_caches(wandb_path, offsets, out_dir, run_id, recompute)

    for layer_idx, cache in layer_caches:
        _plot_layer(cache, layer_idx, run_id, out_dir, top_n_pairs, plot_set)

    logger.info(f"All plots saved to {out_dir}")


if __name__ == "__main__":
    fire.Fire(plot_qk_c_attention_contributions)
