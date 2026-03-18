"""Visualize pairwise W_V subspace overlap between attention heads.

Computes five variants of overlap metrics between heads' value projection
matrices, optionally weighted by the data distribution. See the companion
LaTeX writeup (out/wv_overlap_writeup.tex) for detailed equations.

Usage:
    python -m spd.scripts.plot_wv_subspace_overlap.plot_wv_subspace_overlap \
        wandb:goodfire/spd/runs/<run_id> --layer 1
"""

import math
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray

from spd.configs import LMTaskConfig
from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from spd.spd_types import ModelPath
from spd.utils.wandb_utils import parse_wandb_run_path

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
        component_V: (d_model, n_components) — right vectors from SPD decomposition
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


def _render_overlap_heatmap(
    ax: plt.Axes,
    overlap: NDArray[np.floating],
    title: str,
) -> "plt.cm.ScalarMappable":
    """Render a lower-triangular overlap heatmap on the given axes."""
    n_heads = overlap.shape[0]
    mask = np.tri(n_heads, dtype=bool)
    overlap_masked = np.where(mask, overlap, np.nan)

    im = ax.imshow(overlap_masked, cmap="Purples", vmin=0, vmax=1)

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

    ax.set_title(title, fontsize=11, fontweight="bold")
    return im


def _plot_combined_paper_figure(
    v_weight_per_head: NDArray[np.floating],
    var_svectors: NDArray[np.floating],
    var_singular_values: NDArray[np.floating],
    layer: int,
    out_dir: Path,
) -> None:
    """Side-by-side: unweighted subspace overlap (left) and variance-weighted (right)."""
    n_heads = v_weight_per_head.shape[0]

    # Unweighted Gram matrices
    M_unweighted = [v_weight_per_head[h].T @ v_weight_per_head[h] for h in range(n_heads)]
    overlap_unweighted = _compute_overlap_matrix(M_unweighted)

    # Variance-weighted Gram matrices
    Z_diag_s = var_svectors.T * var_singular_values[None, :]
    M_var = [
        (v_weight_per_head[h] @ Z_diag_s).T @ (v_weight_per_head[h] @ Z_diag_s)
        for h in range(n_heads)
    ]
    overlap_var = _compute_overlap_matrix(M_var)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    _render_overlap_heatmap(ax_left, overlap_unweighted, "Subspace overlap")
    im = _render_overlap_heatmap(ax_right, overlap_var, "Data-weighted subspace overlap")

    fig.colorbar(im, ax=[ax_left, ax_right], shrink=0.8, pad=0.04, label="Cosine Similarity")

    path = out_dir / f"layer{layer}_wv_overlap_combined.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_wv_subspace_overlap(
    wandb_path: ModelPath,
    layer: int = 1,
    n_batches: int = N_BATCHES,
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

    # 3. Extract per-head W_V
    v_weight = target_model._h[layer].attn.v_proj.weight.detach().float().cpu().numpy()
    # v_weight shape: (n_heads * head_dim, d_model)
    v_weight_per_head = v_weight.reshape(n_heads, head_dim, d_model)

    # 4. Plots
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

    # 5. Combined paper figure
    _plot_combined_paper_figure(
        v_weight_per_head, var_svectors, var_singular_values, layer, out_dir
    )

    # 6. Component-head amplification
    logger.info("Loading SPD component model...")
    component_model = ComponentModel.from_pretrained(wandb_path)
    v_component = component_model.components[f"h.{layer}.attn.v_proj"]
    component_V = v_component.V.detach().float().cpu().numpy()  # (d_model, n_components)
    logger.info(f"Value components: {component_V.shape[1]}")
    _plot_component_head_amplification(v_weight_per_head, component_V, layer, out_dir)

    logger.info(f"All outputs saved to {out_dir}")


if __name__ == "__main__":
    fire.Fire(plot_wv_subspace_overlap)
