"""Visualize pairwise W_OV subspace overlap between attention heads.

Computes read, write, and raw Frobenius cosine similarity of W_OV = W_O @ W_V
matrices between heads, optionally weighted by the data distribution via PCA.
See README.md in this directory for detailed equations and output structure.

Usage:
    python -m param_decomp.scripts.plot_wv_subspace_overlap.plot_wv_subspace_overlap \
        wandb:goodfire/spd/runs/<run_id> --layer 1
"""

import math
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray

from param_decomp.configs import LMTaskConfig
from param_decomp.data import DatasetConfig, create_data_loader
from param_decomp.log import logger
from param_decomp.models.component_model import ComponentModel, ParamDecompRunInfo
from param_decomp.param_decomp_types import ModelPath
from param_decomp.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from param_decomp.utils.wandb_utils import parse_wandb_run_path

SCRIPT_DIR = Path(__file__).parent
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


def _plot_component_head_amplification(
    v_weight_per_head: NDArray[np.floating],
    component_V: NDArray[np.floating],
    layer: int,
    out_dir: Path,
) -> None:
    """Heatmap of how much each head's W_V amplifies each value component's read direction.

    amplification[c, h] = ||W_V^h @ v_hat_c|| where v_hat_c = V[:, c] / ||V[:, c]||

    Args:
        v_weight_per_head: (n_heads, head_dim, d_model)
        component_V: (d_model, n_components) — right vectors from PD
    """
    n_heads = v_weight_per_head.shape[0]

    # Normalize V to unit length so we measure directional amplification, not V magnitude
    v_norms = np.linalg.norm(component_V, axis=0, keepdims=True).clip(min=1e-10)
    component_V_normed = component_V / v_norms

    # (n_heads, head_dim, d_model) @ (d_model, n_components) -> (n_heads, head_dim, n_components)
    projected = np.einsum("hdi,ic->hic", v_weight_per_head, component_V_normed)
    # L2 norm over head_dim -> (n_heads, n_components)
    amplification = np.sqrt((projected**2).sum(axis=1))  # (n_heads, n_components)
    amplification = amplification.T  # (n_components, n_heads)

    # Sort components by max amplification across heads (descending)
    sort_idx = np.argsort(-amplification.max(axis=1))
    amplification = amplification[sort_idx]

    fig, ax = plt.subplots(figsize=(5, 10))
    im = ax.imshow(amplification, aspect="auto", cmap="viridis", interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02, label=r"$\|W_V^h \mathbf{v}_c\|$")

    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.set_xlabel("Head")
    ax.set_ylabel("Value component (sorted by max amplification)")

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    path = out_dir / f"layer{layer}_component_head_amplification.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def _compute_overlap_matrix(
    gram_matrices: list[NDArray[np.floating]],
) -> NDArray[np.floating]:
    """Frobenius cosine similarity between all pairs of Gram matrices."""
    n = len(gram_matrices)
    norms = [float(np.linalg.norm(m, "fro")) for m in gram_matrices]
    overlap = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            overlap[a, b] = float(np.trace(gram_matrices[a] @ gram_matrices[b])) / (
                norms[a] * norms[b]
            )
    return overlap


def _compute_raw_fcs(matrices: list[NDArray[np.floating]]) -> NDArray[np.floating]:
    """Frobenius cosine similarity between raw matrices (not their Grams).

    Captures whether the full linear maps are similar, not just their read/write subspaces.
    """
    n = len(matrices)
    flat = [m.ravel() for m in matrices]
    norms = [float(np.linalg.norm(f)) for f in flat]
    overlap = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            overlap[a, b] = float(flat[a] @ flat[b]) / (norms[a] * norms[b])
    return overlap


def _analytical_fcs_baseline(m: int, n: int, write_side: bool = False, raw: bool = False) -> float:
    """Analytical expected FCS between random (m x n) i.i.d. N(0,1) matrices.

    For Gram FCS (raw=False):
        Read Gram M = W^T W: baseline ≈ m / (m+n+1)
        Write Gram M = W W^T: baseline ≈ n / (m+n+1)
    For raw FCS (raw=True):
        E[<W_a, W_b>_F] = 0 since entries are independent zero-mean, so baseline ≈ 0.
    """
    if raw:
        return 0.0
    if write_side:
        return n / (m + n + 1)
    return m / (m + n + 1)


def _compute_random_fcs_baseline(
    d_head: int,
    d_model: int,
    n_trials: int = 1000,
    projection: NDArray[np.floating] | None = None,
    write_side: bool = False,
    raw: bool = False,
) -> tuple[float, float]:
    """Empirical FCS baseline between random matrices.

    Also asserts consistency with the analytical formula (when no projection is applied).

    Args:
        d_head: Number of rows in each random matrix (head dimension).
        d_model: Number of columns (model dimension).
        n_trials: Number of random pairs to average over.
        projection: If provided, each random W is right-multiplied by this matrix
            before computing the Gram matrix. Shape (d_model, k) for arbitrary k.
        write_side: If True, compute W W^T (write Gram) instead of W^T W (read Gram).
        raw: If True, compute FCS of the raw matrices (not their Grams).

    Returns:
        (mean, std) of the FCS across trials.
    """
    rng = np.random.default_rng(seed=42)
    vals = np.empty(n_trials)
    for i in range(n_trials):
        wa = rng.standard_normal((d_head, d_model))
        wb = rng.standard_normal((d_head, d_model))
        if projection is not None:
            wa = wa @ projection
            wb = wb @ projection
        if raw:
            fa, fb = wa.ravel(), wb.ravel()
            vals[i] = float(fa @ fb) / (float(np.linalg.norm(fa)) * float(np.linalg.norm(fb)))
        elif write_side:
            ma = wa @ wa.T
            mb = wb @ wb.T
            vals[i] = float(np.trace(ma @ mb)) / (
                float(np.linalg.norm(ma, "fro")) * float(np.linalg.norm(mb, "fro"))
            )
        else:
            ma = wa.T @ wa
            mb = wb.T @ wb
            vals[i] = float(np.trace(ma @ mb)) / (
                float(np.linalg.norm(ma, "fro")) * float(np.linalg.norm(mb, "fro"))
            )
    empirical_mean = float(vals.mean())

    if projection is None:
        analytical = _analytical_fcs_baseline(d_head, d_model, write_side=write_side, raw=raw)
        assert abs(empirical_mean - analytical) < 0.02, (
            f"Empirical FCS baseline {empirical_mean:.4f} doesn't match "
            f"analytical {analytical:.4f} for ({d_head}, {d_model}, write={write_side}, raw={raw})"
        )

    return empirical_mean, float(vals.std())


def _render_overlap_heatmap(
    ax: plt.Axes,
    overlap: NDArray[np.floating],
    title: str,
    random_baseline: tuple[float, float] | None = None,
) -> "plt.cm.ScalarMappable":
    """Render a lower-triangular overlap heatmap on the given axes.

    Args:
        random_baseline: If provided, (mean, std) of the random baseline FCS.
            Rendered as a dashed horizontal line on the colorbar and annotated
            in the title.
    """
    n_heads = overlap.shape[0]
    mask = np.tri(n_heads, dtype=bool)
    overlap_masked = np.where(mask, overlap, np.nan)

    im = ax.imshow(overlap_masked, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.set_yticks(range(n_heads))
    ax.set_yticklabels([f"H{h}" for h in range(n_heads)])

    for i in range(n_heads):
        for j in range(i + 1):
            color = "white" if abs(overlap[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{overlap[i, j]:.2f}", ha="center", va="center", fontsize=9, color=color)

    for spine in ax.spines.values():
        spine.set_visible(False)

    if random_baseline is not None:
        mean, _std = random_baseline
        title = f"{title}\n(Random matrix baseline: {mean:.3f})"

    ax.set_title(title, fontsize=11, fontweight="bold")
    return im


def _plot_combined_paper_figure(
    weight_per_head: NDArray[np.floating],
    var_svectors: NDArray[np.floating],
    var_singular_values: NDArray[np.floating],
    layer: int,
    out_dir: Path,
    title_prefix: str = "",
) -> None:
    """Side-by-side read overlap: unweighted (left) and variance-weighted (right)."""
    n_heads = weight_per_head.shape[0]
    d_row = weight_per_head.shape[1]
    d_model = weight_per_head.shape[2]

    # Read Gram: W^T W
    M_unweighted = [weight_per_head[h].T @ weight_per_head[h] for h in range(n_heads)]
    overlap_unweighted = _compute_overlap_matrix(M_unweighted)

    Z_diag_s = var_svectors.T * var_singular_values[None, :]
    M_var = [
        (weight_per_head[h] @ Z_diag_s).T @ (weight_per_head[h] @ Z_diag_s) for h in range(n_heads)
    ]
    overlap_var = _compute_overlap_matrix(M_var)

    logger.info("Computing random baselines for read FCS...")
    baseline_unweighted = _compute_random_fcs_baseline(d_row, d_model)
    baseline_data_weighted = _compute_random_fcs_baseline(d_row, d_model, projection=Z_diag_s)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    prefix = f"{title_prefix} " if title_prefix else ""
    _render_overlap_heatmap(
        ax_left,
        overlap_unweighted,
        f"{prefix}Read overlap",
        random_baseline=baseline_unweighted,
    )
    im = _render_overlap_heatmap(
        ax_right,
        overlap_var,
        f"{prefix}Read overlap (data-weighted)",
        random_baseline=baseline_data_weighted,
    )

    fig.colorbar(im, ax=[ax_left, ax_right], shrink=0.8, pad=0.04, label="Cosine Similarity")

    path = out_dir / f"layer{layer}_read_overlap_combined.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def _plot_combined_write_figure(
    weight_per_head: NDArray[np.floating],
    var_svectors: NDArray[np.floating],
    var_singular_values: NDArray[np.floating],
    layer: int,
    out_dir: Path,
    title_prefix: str = "",
) -> None:
    """Side-by-side write overlap: unweighted (left) and variance-weighted (right)."""
    n_heads = weight_per_head.shape[0]
    d_row = weight_per_head.shape[1]
    d_model = weight_per_head.shape[2]

    # Write Gram: W W^T
    M_unweighted = [weight_per_head[h] @ weight_per_head[h].T for h in range(n_heads)]
    overlap_unweighted = _compute_overlap_matrix(M_unweighted)

    Z_diag_s = var_svectors.T * var_singular_values[None, :]
    M_var = [
        (weight_per_head[h] @ Z_diag_s) @ (weight_per_head[h] @ Z_diag_s).T for h in range(n_heads)
    ]
    overlap_var = _compute_overlap_matrix(M_var)

    logger.info("Computing random baselines for write FCS...")
    baseline_unweighted = _compute_random_fcs_baseline(d_row, d_model, write_side=True)
    baseline_data_weighted = _compute_random_fcs_baseline(
        d_row, d_model, projection=Z_diag_s, write_side=True
    )

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    prefix = f"{title_prefix} " if title_prefix else ""
    _render_overlap_heatmap(
        ax_left,
        overlap_unweighted,
        f"{prefix}Write overlap",
        random_baseline=baseline_unweighted,
    )
    im = _render_overlap_heatmap(
        ax_right,
        overlap_var,
        f"{prefix}Write overlap (data-weighted)",
        random_baseline=baseline_data_weighted,
    )

    fig.colorbar(im, ax=[ax_left, ax_right], shrink=0.8, pad=0.04, label="Cosine Similarity")

    path = out_dir / f"layer{layer}_write_overlap_combined.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def _plot_ov_paper_figure(
    weight_per_head: NDArray[np.floating],
    var_svectors: NDArray[np.floating] | None,
    var_singular_values: NDArray[np.floating] | None,
    layer: int,
    out_dir: Path,
    data_label: str = "Data weighted",
    filename_suffix: str = "",
) -> None:
    """1x3 paper figure: read overlap, write overlap, and raw FCS.

    When var_svectors/var_singular_values are provided, applies data weighting.
    When None, uses raw (unweighted) matrices.
    """
    n_heads = weight_per_head.shape[0]
    d_row = weight_per_head.shape[1]
    d_model = weight_per_head.shape[2]

    if var_svectors is not None and var_singular_values is not None:
        Z_diag_s = var_svectors.T * var_singular_values[None, :]
        W_eff = [weight_per_head[h] @ Z_diag_s for h in range(n_heads)]
        projection = Z_diag_s
    else:
        W_eff = [weight_per_head[h] for h in range(n_heads)]
        projection = None

    # Read Gram: W^T W
    M_read = [w.T @ w for w in W_eff]
    overlap_read = _compute_overlap_matrix(M_read)

    # Write Gram: W W^T
    M_write = [w @ w.T for w in W_eff]
    overlap_write = _compute_overlap_matrix(M_write)

    # Raw FCS
    overlap_raw = _compute_raw_fcs(W_eff)

    logger.info("Computing random baselines for OV paper figure...")
    baseline_read = _compute_random_fcs_baseline(d_row, d_model, projection=projection)
    baseline_write = _compute_random_fcs_baseline(
        d_row, d_model, projection=projection, write_side=True
    )
    baseline_raw = _compute_random_fcs_baseline(d_row, d_model, projection=projection, raw=True)

    fig, (ax_read, ax_write, ax_raw) = plt.subplots(1, 3, figsize=(19, 5), constrained_layout=True)

    _render_overlap_heatmap(
        ax_read,
        overlap_read,
        f"Read subspace cosine sim ({data_label})",
        random_baseline=baseline_read,
    )
    _render_overlap_heatmap(
        ax_write,
        overlap_write,
        f"Write subspace cosine sim ({data_label})",
        random_baseline=baseline_write,
    )
    im = _render_overlap_heatmap(
        ax_raw,
        overlap_raw,
        f"Raw cosine sim ({data_label})",
        random_baseline=baseline_raw,
    )

    fig.colorbar(
        im, ax=[ax_read, ax_write, ax_raw], shrink=0.8, pad=0.02, label="Cosine Similarity"
    )

    path = out_dir / f"layer{layer}_ov_paper_figure{filename_suffix}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


FAST_N_BATCHES = 10


def _collect_k_filtered_activations(
    component_model: "ComponentModel",
    loader: "torch.utils.data.DataLoader[dict[str, torch.Tensor]]",
    column_name: str,
    layer: int,
    seq_len: int,
    k_idx: int,
    n_batches: int,
    device: torch.device,
    ci_threshold: float = 0.5,
) -> torch.Tensor:
    """Collect post-RMSNorm activations filtered to tokens where a K component is active.

    The OV circuit reads from the residual stream at key positions (the attended-to
    tokens). Filtering to positions where the K component fires gives us the input
    distribution the OV circuit actually sees when this K component is active.

    Returns: (n_filtered_tokens, d_model)
    """
    k_path = f"h.{layer}.attn.k_proj"

    all_acts: list[torch.Tensor] = []
    total_tokens = 0
    filtered_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            input_ids = batch[column_name][:, :seq_len].to(device)
            out = component_model(input_ids, cache_type="input")
            ci = component_model.calc_causal_importances(
                out.cache, sampling="continuous", detach_inputs=True
            )

            k_ci = ci.lower_leaky[k_path][:, :, k_idx]  # (batch, seq)
            mask = k_ci > ci_threshold  # (batch, seq)

            # The input cache for k_proj IS the post-RMSNorm activation
            acts = out.cache[k_path].float().cpu()  # (batch, seq, d_model)
            mask_cpu = mask.cpu()

            total_tokens += mask_cpu.numel()
            filtered_tokens += mask_cpu.sum().item()
            all_acts.append(acts[mask_cpu])  # (n_passing, d_model)

            if (i + 1) % 25 == 0:
                logger.info(
                    f"K filter: {i + 1}/{n_batches} batches, "
                    f"{filtered_tokens}/{total_tokens} tokens pass ({filtered_tokens / total_tokens:.1%})"
                )

    result = torch.cat(all_acts, dim=0)
    logger.info(
        f"K filter complete: {filtered_tokens}/{total_tokens} tokens pass "
        f"({filtered_tokens / total_tokens:.1%}), shape {result.shape}"
    )
    return result


def _compute_variance_svd(
    activations: torch.Tensor,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute mean-centered SVD of activations. Returns (singular_values, svectors)."""
    activations_centered = activations - activations.mean(dim=0, keepdim=True)
    _, sv_t, Vt = torch.linalg.svd(activations_centered, full_matrices=False)
    return sv_t.numpy(), Vt.numpy()


def plot_wv_subspace_overlap(
    wandb_path: ModelPath,
    layer: int = 1,
    n_batches: int = N_BATCHES,
    fast: bool = False,
    k_filter: int | None = None,
) -> None:
    if fast:
        n_batches = FAST_N_BATCHES
        logger.info(f"Fast mode: n_batches={n_batches}, skipping individual plots")

    _entity, _project, run_id = parse_wandb_run_path(str(wandb_path))
    run_info = ParamDecompRunInfo.from_path(wandb_path)

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
        target_model,
        loader,
        task_config.column_name,
        layer,
        n_batches,
        device,
    )
    logger.info(f"Activations shape: {activations.shape}")

    # 2. SVD of residual stream (not mean-centered)
    logger.info("Computing residual stream SVD...")
    _, singular_values_t, Vt = torch.linalg.svd(activations, full_matrices=False)
    singular_values = singular_values_t.numpy()  # (d_model,)
    data_svectors = Vt.numpy()  # (d_model, d_model) — rows are z_i

    # 2b. SVD of mean-centered residual stream (variance weighting)
    logger.info("Computing mean-centered residual stream SVD...")
    var_singular_values, var_svectors = _compute_variance_svd(activations)

    # 3. Extract per-head W_V
    v_weight = target_model._h[layer].attn.v_proj.weight.detach().float().cpu().numpy()
    # v_weight shape: (n_heads * head_dim, d_model)
    v_weight_per_head = v_weight.reshape(n_heads, head_dim, d_model)

    # 3b. Extract per-head W_OV = W_O_head @ W_V_head
    o_weight = target_model._h[layer].attn.o_proj.weight.detach().float().cpu().numpy()
    # o_weight: (d_model, n_heads * head_dim) — heads partition the input (columns)
    ov_weight_per_head = np.zeros((n_heads, d_model, d_model))
    for h in range(n_heads):
        o_h = o_weight[:, h * head_dim : (h + 1) * head_dim]  # (d_model, head_dim)
        ov_weight_per_head[h] = o_h @ v_weight_per_head[h]  # (d_model, d_model)

    # 4. Load PD component model
    logger.info("Loading PD component model...")
    component_model = ComponentModel.from_pretrained(wandb_path)
    component_model.to(device).eval()
    v_component = component_model.components[f"h.{layer}.attn.v_proj"]
    component_V = v_component.V.detach().float().cpu().numpy()  # (d_model, n_components)
    logger.info(f"Value components: {component_V.shape[1]}")

    # 5. Save intermediate data for downstream analysis scripts
    ov_out_dir = out_dir / "ov"
    ov_out_dir.mkdir(exist_ok=True)
    np.save(ov_out_dir / f"layer{layer}_ov_weight_per_head.npy", ov_weight_per_head)
    np.save(ov_out_dir / f"layer{layer}_var_svectors.npy", var_svectors)
    np.save(ov_out_dir / f"layer{layer}_var_singular_values.npy", var_singular_values)
    logger.info(f"Saved intermediate data to {ov_out_dir}")

    # 6. Generate plots
    def _run_all_plots(
        weight_per_head: NDArray[np.floating],
        plot_out_dir: Path,
        title_prefix: str,
        is_ov: bool = False,
    ) -> None:
        plot_out_dir.mkdir(exist_ok=True)
        if not fast:
            _plot_wv_subspace_overlap(weight_per_head, layer, plot_out_dir)
            _plot_wv_strength_weighted_overlap(weight_per_head, layer, plot_out_dir)
            _plot_wv_data_weighted_overlap(
                weight_per_head, data_svectors, singular_values, layer, plot_out_dir
            )
            _plot_wv_variance_weighted_overlap(
                weight_per_head, var_svectors, var_singular_values, layer, plot_out_dir
            )
            _plot_wv_data_strength_weighted_overlap(
                weight_per_head, data_svectors, singular_values, layer, plot_out_dir
            )
        _plot_combined_paper_figure(
            weight_per_head,
            var_svectors,
            var_singular_values,
            layer,
            plot_out_dir,
            title_prefix=title_prefix,
        )
        _plot_combined_write_figure(
            weight_per_head,
            var_svectors,
            var_singular_values,
            layer,
            plot_out_dir,
            title_prefix=title_prefix,
        )
        _plot_component_head_amplification(weight_per_head, component_V, layer, plot_out_dir)
        if is_ov:
            _plot_ov_paper_figure(
                weight_per_head,
                var_svectors,
                var_singular_values,
                layer,
                plot_out_dir,
            )
            _plot_ov_paper_figure(
                weight_per_head,
                None,
                None,
                layer,
                plot_out_dir,
                data_label="Unweighted",
                filename_suffix="_unweighted",
            )

    logger.info("Generating W_OV plots...")
    _run_all_plots(ov_weight_per_head, out_dir / "ov", title_prefix="W_OV", is_ov=True)

    # 7. K-filtered OV analysis
    if k_filter is not None:
        k_idx = int(k_filter)
        logger.info(f"Collecting K-filtered activations (k={k_idx})...")

        # Need a fresh loader since the previous one was consumed
        filtered_loader, _ = create_data_loader(
            dataset_config=dataset_config, batch_size=BATCH_SIZE, buffer_size=1000
        )
        filtered_acts = _collect_k_filtered_activations(
            component_model,
            filtered_loader,
            task_config.column_name,
            layer,
            task_config.max_seq_len,
            k_idx,
            n_batches,
            device,
        )

        logger.info("Computing K-filtered variance SVD...")
        filt_var_sv, filt_var_svectors = _compute_variance_svd(filtered_acts)

        k_out_dir = out_dir / "ov" / f"k_{k_idx}"
        k_out_dir.mkdir(parents=True, exist_ok=True)
        np.save(k_out_dir / f"layer{layer}_var_svectors.npy", filt_var_svectors)
        np.save(k_out_dir / f"layer{layer}_var_singular_values.npy", filt_var_sv)
        logger.info(f"Saved K-filtered SVD data to {k_out_dir}")
        _plot_ov_paper_figure(
            ov_weight_per_head,
            filt_var_svectors,
            filt_var_sv,
            layer,
            k_out_dir,
            data_label=f"Data weighted: K.{k_idx}",
            filename_suffix=f"_k_{k_idx}",
        )

    logger.info(f"All outputs saved to {out_dir}")


if __name__ == "__main__":
    fire.Fire(plot_wv_subspace_overlap)
