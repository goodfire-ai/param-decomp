"""Visualize which residual stream directions each head's value matrix reads from.

Computes SVD of both the residual stream activations and the per-head W_V matrices,
then measures how much each head reads from each principal data axis. t-SNE clusters
axes read by similar head profiles. A bubble plot shows the result: bubble size = data
singular value, bubble alpha = head's mean attention at offset 1, bubble color = head ID.

Usage:
    python -m spd.scripts.plot_value_reading_tsne.plot_value_reading_tsne \
        wandb:goodfire/spd/runs/<run_id> --layer 1
"""

import math
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.manifold import TSNE

from spd.configs import LMTaskConfig
from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.models.component_model import SPDRunInfo
from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from spd.spd_types import ModelPath
from spd.utils.wandb_utils import parse_wandb_run_path

SCRIPT_DIR = Path(__file__).parent
ATTN_PROFILES_DIR = Path(__file__).parent.parent / "plot_attention_offset_profiles" / "out"
N_BATCHES = 100
BATCH_SIZE = 32


def _collect_post_rmsnorm_activations(
    model: LlamaSimpleMLP,
    loader: "torch.utils.data.DataLoader[dict[str, torch.Tensor]]",
    column_name: str,
    layer: int,
    n_batches: int,
    device: torch.device,
) -> torch.Tensor:
    """Collect post-RMSNorm residual stream activations at a layer.

    Returns: (total_tokens, d_model)
    """
    seq_len = model.config.n_ctx
    all_acts: list[torch.Tensor] = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            input_ids = batch[column_name][:, :seq_len].to(device)
            x = model.wte(input_ids)

            for layer_idx, block in enumerate(model._h):
                if layer_idx == layer:
                    attn_input = block.rms_1(x).float().cpu()  # (B, T, d_model)
                    all_acts.append(attn_input.reshape(-1, attn_input.shape[-1]))
                    break
                # Run full block to advance residual stream
                attn_input = block.rms_1(x)
                attn = block.attn
                q = (
                    attn.q_proj(attn_input)
                    .view(x.shape[0], x.shape[1], attn.n_head, attn.head_dim)
                    .transpose(1, 2)
                )
                k = (
                    attn.k_proj(attn_input)
                    .view(x.shape[0], x.shape[1], attn.n_key_value_heads, attn.head_dim)
                    .transpose(1, 2)
                )
                v = (
                    attn.v_proj(attn_input)
                    .view(x.shape[0], x.shape[1], attn.n_key_value_heads, attn.head_dim)
                    .transpose(1, 2)
                )

                position_ids = torch.arange(x.shape[1], device=device).unsqueeze(0)
                cos = attn.rotary_cos[position_ids].to(q.dtype)  # pyright: ignore[reportIndexIssue]
                sin = attn.rotary_sin[position_ids].to(q.dtype)  # pyright: ignore[reportIndexIssue]
                q, k = attn.apply_rotary_pos_emb(q, k, cos, sin)

                if attn.repeat_kv_heads > 1:
                    k = k.repeat_interleave(attn.repeat_kv_heads, dim=1)
                    v = v.repeat_interleave(attn.repeat_kv_heads, dim=1)

                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(attn.head_dim))
                att = att.masked_fill(
                    attn.bias[:, :, : x.shape[1], : x.shape[1]] == 0,  # pyright: ignore[reportIndexIssue]
                    float("-inf"),
                )
                att = torch.nn.functional.softmax(att, dim=-1)
                y = att @ v
                y = y.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], attn.n_embd)
                y = attn.o_proj(y)
                x = x + y
                x = x + block.mlp(block.rms_2(x))

            if (i + 1) % 25 == 0:
                logger.info(f"Collected {i + 1}/{n_batches} batches")

    return torch.cat(all_acts, dim=0)  # (total_tokens, d_model)


def _compute_reading_strengths(
    v_weight_head: NDArray[np.floating],
    data_svectors: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute |ψ_i| for one head: how much W_V^h reads from each data axis.

    Args:
        v_weight_head: (head_dim, d_model) — one head's V weight matrix
        data_svectors: (d_model, d_model) — rows are data singular vectors z_i

    Returns: (d_model,) — |ψ_i| for each data axis
    """
    # SVD of W_V^h: (head_dim, d_model) = U_h @ diag(s_h) @ Vh_T
    _, s_h, Vh_T = np.linalg.svd(v_weight_head, full_matrices=False)
    # Vh_T: (head_dim, d_model) — rows are right singular vectors of W_V^h
    # s_h: (head_dim,)

    # Inner products: (d_model, head_dim) — <z_i, v_j^h> for each i, j
    inner_products = data_svectors @ Vh_T.T  # (d_model, head_dim)

    # L2 norm: ψ_i = ||W_V^h z_i|| = ||S_h V_h^T z_i|| = sqrt(Σ_j (s_j * <z_i, v_j>)²)
    psi = np.sqrt(((inner_products * s_h[None, :]) ** 2).sum(axis=1))  # (d_model,)
    return psi


def _plot_wv_subspace_overlap(
    v_weight_per_head: NDArray[np.floating],
    layer: int,
    out_dir: Path,
) -> None:
    """Heatmap of pairwise W_V subspace overlap (Frobenius cosine similarity)."""
    n_heads = v_weight_per_head.shape[0]

    # For each head, compute Gram matrix M_h = W_V^h^T W_V^h (PSD, d_model x d_model)
    M: list[NDArray[np.floating]] = []
    for h in range(n_heads):
        w = v_weight_per_head[h]  # (head_dim, d_model)
        M.append(w.T @ w)  # (d_model, d_model)

    M_norms = [float(np.linalg.norm(m, "fro")) for m in M]

    # Cosine similarity of Gram matrices: tr(M_a M_b) / (||M_a||_F * ||M_b||_F)
    overlap = np.zeros((n_heads, n_heads))
    for a in range(n_heads):
        for b in range(n_heads):
            overlap[a, b] = float(np.trace(M[a] @ M[b])) / (M_norms[a] * M_norms[b])

    mask = np.tri(n_heads, dtype=bool)
    overlap_masked = np.where(mask, overlap, np.nan)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(overlap_masked, cmap="Purples", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="Subspace overlap")

    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.set_yticks(range(n_heads))
    ax.set_yticklabels([f"H{h}" for h in range(n_heads)])

    for i in range(n_heads):
        for j in range(i + 1):
            color = "white" if overlap[i, j] > 0.7 else "black"
            ax.text(j, i, f"{overlap[i, j]:.2f}", ha="center", va="center", fontsize=9, color=color)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    path = out_dir / f"layer{layer}_wv_subspace_overlap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def _plot_wv_strength_weighted_overlap(
    v_weight_per_head: NDArray[np.floating],
    layer: int,
    out_dir: Path,
) -> None:
    """Heatmap of pairwise W_V overlap weighted by joint reading strength.

    strength_weighted_overlap(a, b) = cos(M_a, M_b) * sqrt(||M_a||_F * ||M_b||_F)
    """
    n_heads = v_weight_per_head.shape[0]

    M: list[NDArray[np.floating]] = []
    for h in range(n_heads):
        w = v_weight_per_head[h]  # (head_dim, d_model)
        M.append(w.T @ w)  # (d_model, d_model)

    M_norms = [float(np.linalg.norm(m, "fro")) for m in M]

    overlap = np.zeros((n_heads, n_heads))
    for a in range(n_heads):
        for b in range(n_heads):
            cosine = float(np.trace(M[a] @ M[b])) / (M_norms[a] * M_norms[b])
            joint_strength = math.sqrt(M_norms[a] * M_norms[b])
            overlap[a, b] = cosine * joint_strength

    mask = np.tri(n_heads, dtype=bool)
    overlap_masked = np.where(mask, overlap, np.nan)

    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = float(np.nanmax(overlap_masked))
    im = ax.imshow(overlap_masked, cmap="Purples", vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="Strength-weighted overlap")

    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.set_yticks(range(n_heads))
    ax.set_yticklabels([f"H{h}" for h in range(n_heads)])

    for i in range(n_heads):
        for j in range(i + 1):
            color = "white" if overlap[i, j] > 0.7 * vmax else "black"
            ax.text(j, i, f"{overlap[i, j]:.2f}", ha="center", va="center", fontsize=9, color=color)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    path = out_dir / f"layer{layer}_wv_strength_weighted_overlap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def _plot_wv_data_weighted_overlap(
    v_weight_per_head: NDArray[np.floating],
    data_svectors: NDArray[np.floating],
    singular_values: NDArray[np.floating],
    layer: int,
    out_dir: Path,
) -> None:
    """Heatmap of W_V subspace overlap weighted by data activation magnitude.

    Transforms each head's W_V into the data-weighted space:
        W_eff^h = W_V^h @ Z @ diag(s)
    where Z has columns z_i (right singular vectors of X), s = data singular values.
    Then computes Frobenius cosine similarity of the resulting Gram matrices.
    """
    n_heads = v_weight_per_head.shape[0]

    # W_eff^h = W_V^h @ Z @ diag(s)  — (head_dim, d_model)
    Z_diag_s = data_svectors.T * singular_values[None, :]  # (d_model, d_model)
    M: list[NDArray[np.floating]] = []
    for h in range(n_heads):
        w_eff = v_weight_per_head[h] @ Z_diag_s  # (head_dim, d_model)
        M.append(w_eff.T @ w_eff)  # (d_model, d_model)

    M_norms = [float(np.linalg.norm(m, "fro")) for m in M]

    overlap = np.zeros((n_heads, n_heads))
    for a in range(n_heads):
        for b in range(n_heads):
            overlap[a, b] = float(np.trace(M[a] @ M[b])) / (M_norms[a] * M_norms[b])

    mask = np.tri(n_heads, dtype=bool)
    overlap_masked = np.where(mask, overlap, np.nan)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(overlap_masked, cmap="Purples", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="Data-weighted subspace overlap")

    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.set_yticks(range(n_heads))
    ax.set_yticklabels([f"H{h}" for h in range(n_heads)])

    for i in range(n_heads):
        for j in range(i + 1):
            color = "white" if overlap[i, j] > 0.7 else "black"
            ax.text(j, i, f"{overlap[i, j]:.2f}", ha="center", va="center", fontsize=9, color=color)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    path = out_dir / f"layer{layer}_wv_data_weighted_overlap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def _plot_wv_variance_weighted_overlap(
    v_weight_per_head: NDArray[np.floating],
    var_svectors: NDArray[np.floating],
    var_singular_values: NDArray[np.floating],
    layer: int,
    out_dir: Path,
) -> None:
    """Heatmap of W_V subspace overlap weighted by data variance (mean-centered SVD).

    Same as data-weighted overlap but uses singular vectors/values from
    mean-centered activations, so directions are weighted by variance rather
    than raw magnitude.
    """
    n_heads = v_weight_per_head.shape[0]

    Z_diag_s = var_svectors.T * var_singular_values[None, :]  # (d_model, d_model)
    M: list[NDArray[np.floating]] = []
    for h in range(n_heads):
        w_eff = v_weight_per_head[h] @ Z_diag_s  # (head_dim, d_model)
        M.append(w_eff.T @ w_eff)  # (d_model, d_model)

    M_norms = [float(np.linalg.norm(m, "fro")) for m in M]

    overlap = np.zeros((n_heads, n_heads))
    for a in range(n_heads):
        for b in range(n_heads):
            overlap[a, b] = float(np.trace(M[a] @ M[b])) / (M_norms[a] * M_norms[b])

    mask = np.tri(n_heads, dtype=bool)
    overlap_masked = np.where(mask, overlap, np.nan)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(overlap_masked, cmap="Purples", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="Variance-weighted subspace overlap")

    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.set_yticks(range(n_heads))
    ax.set_yticklabels([f"H{h}" for h in range(n_heads)])

    for i in range(n_heads):
        for j in range(i + 1):
            color = "white" if overlap[i, j] > 0.7 else "black"
            ax.text(j, i, f"{overlap[i, j]:.2f}", ha="center", va="center", fontsize=9, color=color)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    path = out_dir / f"layer{layer}_wv_variance_weighted_overlap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def _plot_wv_data_strength_weighted_overlap(
    v_weight_per_head: NDArray[np.floating],
    data_svectors: NDArray[np.floating],
    singular_values: NDArray[np.floating],
    layer: int,
    out_dir: Path,
) -> None:
    """Heatmap of W_V overlap weighted by both data activation magnitude and joint reading strength.

    combined(a, b) = cos(M_a^data, M_b^data) * sqrt(||M_a^data||_F * ||M_b^data||_F)
    """
    n_heads = v_weight_per_head.shape[0]

    Z_diag_s = data_svectors.T * singular_values[None, :]  # (d_model, d_model)
    M: list[NDArray[np.floating]] = []
    for h in range(n_heads):
        w_eff = v_weight_per_head[h] @ Z_diag_s  # (head_dim, d_model)
        M.append(w_eff.T @ w_eff)  # (d_model, d_model)

    M_norms = [float(np.linalg.norm(m, "fro")) for m in M]

    overlap = np.zeros((n_heads, n_heads))
    for a in range(n_heads):
        for b in range(n_heads):
            cosine = float(np.trace(M[a] @ M[b])) / (M_norms[a] * M_norms[b])
            joint_strength = math.sqrt(M_norms[a] * M_norms[b])
            overlap[a, b] = cosine * joint_strength

    # Normalize by max value for readable display
    max_val = overlap.max()
    overlap_norm = overlap / max_val

    mask = np.tri(n_heads, dtype=bool)
    overlap_masked = np.where(mask, overlap_norm, np.nan)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(overlap_masked, cmap="Purples", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="Data + strength weighted overlap (rel.)")

    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.set_yticks(range(n_heads))
    ax.set_yticklabels([f"H{h}" for h in range(n_heads)])

    for i in range(n_heads):
        for j in range(i + 1):
            color = "white" if overlap_norm[i, j] > 0.7 else "black"
            ax.text(
                j, i, f"{overlap_norm[i, j]:.2f}", ha="center", va="center", fontsize=9, color=color
            )

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    path = out_dir / f"layer{layer}_wv_data_strength_weighted_overlap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def _plot_psi_correlation(
    psi: NDArray[np.floating],
    layer: int,
    out_dir: Path,
    suffix: str = "",
) -> None:
    """Heatmap of pairwise Pearson correlation between heads' ψ profiles."""
    n_heads = psi.shape[0]
    corr = np.array(np.corrcoef(psi))  # (n_heads, n_heads)

    # Mask upper triangle
    mask = np.tri(n_heads, dtype=bool)
    corr_masked = np.where(mask, corr, np.nan)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_masked, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="Pearson correlation")

    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.set_yticks(range(n_heads))
    ax.set_yticklabels([f"H{h}" for h in range(n_heads)])

    for i in range(n_heads):
        for j in range(i + 1):
            color = "white" if abs(corr[i, j]) > 0.7 else "black"
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=9, color=color)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    path = out_dir / f"layer{layer}_psi_correlation{suffix}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def _plot_psi_scatter_grid(
    psi: NDArray[np.floating],
    layer: int,
    out_dir: Path,
    suffix: str = "",
) -> None:
    """Grid of pairwise scatter plots of ψ profiles between heads (lower triangle)."""
    n_heads = psi.shape[0]

    fig, axes = plt.subplots(
        n_heads, n_heads, figsize=(2.5 * n_heads, 2.5 * n_heads), squeeze=False
    )

    for i in range(n_heads):
        for j in range(n_heads):
            ax = axes[i, j]
            if i > j:
                ax.scatter(psi[j], psi[i], s=1, alpha=0.3, color="black", rasterized=True)
                corr = float(np.corrcoef(psi[i], psi[j])[0, 1])
                ax.text(
                    0.05,
                    0.95,
                    f"r={corr:.2f}",
                    transform=ax.transAxes,
                    fontsize=8,
                    va="top",
                    ha="left",
                )
            elif i == j:
                ax.hist(psi[i], bins=50, color="gray", edgecolor="none")
            else:
                ax.set_visible(False)

            if j == 0 and i > 0:
                ax.set_ylabel(f"H{i}", fontsize=10)
            if i == n_heads - 1 and j < i:
                ax.set_xlabel(f"H{j}", fontsize=10)

            ax.set_xticks([])
            ax.set_yticks([])

    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    path = out_dir / f"layer{layer}_psi_scatter_grid{suffix}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def _plot_singular_value_histogram(
    singular_values: NDArray[np.floating],
    layer: int,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(singular_values)), singular_values, color="steelblue", width=1.0)
    ax.set_xlabel("Singular vector index (sorted by singular value)")
    ax.set_ylabel("Singular value")
    ax.set_xlim(-0.5, len(singular_values) - 0.5)
    fig.tight_layout()
    path = out_dir / f"layer{layer}_singular_values_histogram.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def _plot_bubble(
    coords: NDArray[np.floating],
    singular_values: NDArray[np.floating],
    psi: NDArray[np.floating],
    attn_at_offset1: NDArray[np.floating],
    layer: int,
    method: str,
    out_dir: Path,
) -> None:
    """Render bubble plot from 2D coordinates (t-SNE or PCA).

    Args:
        coords: (d_model, 2) — 2D positions
        singular_values: (d_model,) — data singular values (bubble size)
        psi: (n_heads, d_model) — reading strengths
        attn_at_offset1: (n_heads,) — mean attention at offset 1 per head (bubble alpha)
        layer: layer index
        method: "tsne" or "pca" — used for filename and title
        out_dir: output directory
    """
    n_heads = psi.shape[0]
    d_model = psi.shape[1]
    cmap = plt.get_cmap("tab10")
    head_colors = [cmap(h) for h in range(n_heads)]

    sv_normalized = singular_values / singular_values.max()
    max_bubble_size = 200
    min_bubble_size = 5
    bubble_sizes = min_bubble_size + sv_normalized * (max_bubble_size - min_bubble_size)

    angles = np.linspace(0, 2 * np.pi, n_heads, endpoint=False)

    fig, ax = plt.subplots(figsize=(12, 10))

    for h in range(n_heads):
        alpha = float(np.clip(attn_at_offset1[h], 0.05, 1.0))
        color = (*head_colors[h][:3], alpha)

        radius = np.sqrt(bubble_sizes) * 0.05
        dx = radius * np.cos(angles[h])
        dy = radius * np.sin(angles[h])

        ax.scatter(
            coords[:, 0] + dx,
            coords[:, 1] + dy,
            s=bubble_sizes,
            c=[color] * d_model,
            edgecolors="none",
            label=f"H{h} (attn@1={attn_at_offset1[h]:.3f})",
            rasterized=True,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.set_title(method.upper(), fontsize=13, fontweight="bold")

    fig.tight_layout()
    path = out_dir / f"layer{layer}_value_reading_{method}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def _optimal_axis_ordering(psi: NDArray[np.floating]) -> NDArray[np.intp]:
    """Permute data axes so those with similar head-reading profiles are adjacent.

    Uses hierarchical clustering with optimal leaf ordering on the ψ profiles.
    """
    from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
    from scipy.spatial.distance import pdist

    # psi: (n_heads, d_model) — transpose so each row is one axis's head profile
    distances = pdist(psi.T, metric="cosine")
    Z = linkage(distances, method="ward")
    Z_optimal = optimal_leaf_ordering(Z, distances)
    return leaves_list(Z_optimal)


def _plot_grid_squares(
    singular_values: NDArray[np.floating],
    psi: NDArray[np.floating],
    layer: int,
    out_dir: Path,
    suffix: str = "",
) -> None:
    """3x2 grid where each head's subplot shows a grid of squares.

    Each square = one data singular vector. Size ∝ singular value,
    alpha ∝ ψ_i^h (normalized per head). Axes are ordered by hierarchical
    clustering so that axes read by similar heads are adjacent.
    """
    n_heads = psi.shape[0]
    d_model = psi.shape[1]
    n_cols = math.ceil(math.sqrt(d_model))  # 28

    order = _optimal_axis_ordering(psi)
    singular_values = singular_values[order]
    psi = psi[:, order]

    grid_x = np.arange(d_model) % n_cols
    grid_y = np.arange(d_model) // n_cols
    grid_y = grid_y.max() - grid_y

    fig, axes = plt.subplots(3, 2, figsize=(12, 18), squeeze=False)

    for h in range(n_heads):
        row, col = divmod(h, 2)
        ax = axes[row, col]

        # Size = ψ_i^h normalized per-head
        head_max = psi[h].max()
        psi_normalized = psi[h] / head_max if head_max > 0 else psi[h]
        max_marker_size = 120
        min_marker_size = 5
        marker_sizes = min_marker_size + psi_normalized * (max_marker_size - min_marker_size)

        ax.scatter(
            grid_x,
            grid_y,
            s=marker_sizes,
            c="black",
            marker="s",
            edgecolors="none",
            rasterized=True,
        )

        ax.set_title(f"H{h}", fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_ylim(-0.5, grid_y.max() + 0.5)
        ax.set_aspect("equal")

    fig.subplots_adjust(hspace=0.08, wspace=0.02)
    path = out_dir / f"layer{layer}_value_reading_grid{suffix}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_value_reading_tsne(
    wandb_path: ModelPath,
    layer: int = 1,
    n_batches: int = N_BATCHES,
    tsne_perplexity: float = 100.0,
    tsne_seed: int = 42,
) -> None:
    _entity, _project, run_id = parse_wandb_run_path(str(wandb_path))
    run_info = SPDRunInfo.from_path(wandb_path)

    out_dir = SCRIPT_DIR / "out" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    config = run_info.config
    assert config.pretrained_model_name is not None
    target_model = LlamaSimpleMLP.from_pretrained(config.pretrained_model_name)
    target_model.eval()
    for block in target_model._h:
        block.attn.flash_attention = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_model = target_model.to(device)

    n_heads = target_model._h[0].attn.n_head
    head_dim = target_model._h[0].attn.head_dim
    d_model = target_model.config.n_embd
    logger.info(f"Model: d_model={d_model}, n_heads={n_heads}, head_dim={head_dim}")

    # 1. Collect post-RMSNorm activations
    task_config = config.task_config
    assert isinstance(task_config, LMTaskConfig)
    dataset_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=config.tokenizer_name,
        split=task_config.eval_data_split,
        n_ctx=task_config.max_seq_len,
        is_tokenized=task_config.is_tokenized,
        streaming=task_config.streaming,
        column_name=task_config.column_name,
        shuffle_each_epoch=False,
    )
    loader, _ = create_data_loader(
        dataset_config=dataset_config, batch_size=BATCH_SIZE, buffer_size=1000
    )

    logger.info(f"Collecting post-RMSNorm activations at layer {layer}...")
    activations = _collect_post_rmsnorm_activations(
        target_model, loader, task_config.column_name, layer, n_batches, device
    )
    logger.info(f"Activations shape: {activations.shape}")

    # 2. SVD of residual stream (not mean-centered)
    logger.info("Computing residual stream SVD...")
    _, singular_values_t, Vt = torch.linalg.svd(activations, full_matrices=False)
    singular_values = singular_values_t.numpy()  # (d_model,)
    data_svectors = Vt.numpy()  # (d_model, d_model) — rows are z_i

    # 2b. SVD of mean-centered residual stream (variance weighting)
    logger.info("Computing mean-centered residual stream SVD...")
    activations_centered = activations - activations.mean(dim=0, keepdim=True)
    _, var_singular_values_t, var_Vt = torch.linalg.svd(activations_centered, full_matrices=False)
    var_singular_values = var_singular_values_t.numpy()  # (d_model,)
    var_svectors = var_Vt.numpy()  # (d_model, d_model) — rows are principal components

    # 3. Per-head W_V SVD and reading strengths
    logger.info("Computing per-head reading strengths...")
    v_weight = target_model._h[layer].attn.v_proj.weight.detach().float().cpu().numpy()
    # v_weight shape: (n_heads * head_dim, d_model)
    v_weight_per_head = v_weight.reshape(n_heads, head_dim, d_model)

    psi = np.zeros((n_heads, d_model))
    for h in range(n_heads):
        psi[h] = _compute_reading_strengths(v_weight_per_head[h], data_svectors) * singular_values

    # 4. Dimensionality reduction
    logger.info(f"Running t-SNE (perplexity={tsne_perplexity})...")
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=tsne_seed)
    tsne_coords = tsne.fit_transform(psi.T)  # (d_model, 2)

    logger.info("Computing PCA...")
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(psi.T)  # (d_model, 2)

    # 5. Load attention at offset 1
    attn_profile_path = ATTN_PROFILES_DIR / run_id / "dataset" / "attn_mean.npy"
    assert attn_profile_path.exists(), (
        f"Run plot_attention_offset_profiles first: {attn_profile_path}"
    )
    attn_mean = np.load(attn_profile_path)  # (n_layers, n_heads, n_offsets)
    attn_at_offset1 = attn_mean[layer, :, 1]  # (n_heads,)
    logger.info(f"Attention at offset 1: {attn_at_offset1}")

    # Save intermediate data
    np.save(out_dir / f"layer{layer}_singular_values.npy", singular_values)
    np.save(out_dir / f"layer{layer}_data_svectors.npy", data_svectors)
    np.save(out_dir / f"layer{layer}_psi.npy", psi)
    np.save(out_dir / f"layer{layer}_tsne_coords.npy", tsne_coords)
    np.save(out_dir / f"layer{layer}_pca_coords.npy", pca_coords)

    # 6. Unit-basis variant: ψ using standard basis instead of data singular vectors
    logger.info("Computing unit-basis reading strengths...")
    identity = np.eye(d_model)
    psi_unit = np.zeros((n_heads, d_model))
    for h in range(n_heads):
        psi_unit[h] = _compute_reading_strengths(v_weight_per_head[h], identity)
    np.save(out_dir / f"layer{layer}_psi_unit.npy", psi_unit)

    # 7. Plots
    _plot_wv_subspace_overlap(v_weight_per_head, layer, out_dir)
    _plot_wv_strength_weighted_overlap(v_weight_per_head, layer, out_dir)
    _plot_wv_data_weighted_overlap(
        v_weight_per_head, data_svectors, singular_values, layer, out_dir
    )
    _plot_wv_variance_weighted_overlap(
        v_weight_per_head, var_svectors, var_singular_values, layer, out_dir
    )
    _plot_wv_data_strength_weighted_overlap(
        v_weight_per_head, data_svectors, singular_values, layer, out_dir
    )
    _plot_psi_correlation(psi, layer, out_dir)
    _plot_psi_correlation(psi_unit, layer, out_dir, suffix="_unit_basis")
    _plot_psi_scatter_grid(psi, layer, out_dir)
    _plot_psi_scatter_grid(psi_unit, layer, out_dir, suffix="_unit_basis")
    _plot_singular_value_histogram(singular_values, layer, out_dir)
    _plot_bubble(tsne_coords, singular_values, psi, attn_at_offset1, layer, "tsne", out_dir)
    _plot_bubble(pca_coords, singular_values, psi, attn_at_offset1, layer, "pca", out_dir)
    _plot_grid_squares(singular_values, psi, layer, out_dir)
    _plot_grid_squares(singular_values, psi_unit, layer, out_dir, suffix="_unit_basis")

    logger.info(f"All outputs saved to {out_dir}")


if __name__ == "__main__":
    fire.Fire(plot_value_reading_tsne)
