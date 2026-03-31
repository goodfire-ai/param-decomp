"""Data-specific QK component contribution plots for dataset samples.

For each (sample, query_pos), decomposes the pre-softmax attention logits into
per-(q_component, k_component) contributions at each key position, and overlays the
sum with ground-truth logits from the target model.

Two modes:
  - weighted: contribution scaled by actual component activations at each position
  - binary: contribution counted only if both components' per-token CI exceeds a threshold

The sum over ALL components (not just alive ones) is used for validation against ground truth.
Alive filtering only controls which pairs get individual lines in the plot.

Usage:
    python -m spd.scripts.plot_qk_c_datapoint.plot_qk_c_datapoint \
        wandb:goodfire/spd/runs/<run_id> \
        --sample_indices 0 1 2 \
        --query_positions 5 4 3 \
        --layer 1
"""

import math
from dataclasses import dataclass
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from transformers import AutoTokenizer

from spd.configs import LMTaskConfig
from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import LinearComponents
from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from spd.scripts.collect_attention_patterns import collect_attention_patterns_with_logits
from spd.scripts.rope_aware_qk import compute_qk_rope_coefficients, evaluate_qk_at_offsets
from spd.spd_types import ModelPath
from spd.utils.wandb_utils import parse_wandb_run_path

SCRIPT_DIR = Path(__file__).parent


@dataclass
class DatapointContributions:
    """Per-head QK component contributions for a single query position."""

    # (n_q_heads, C_q, C_k, n_key_positions) — contributions for ALL components
    contributions: NDArray[np.floating]
    # (n_q_heads, n_key_positions) — actual pre-softmax logits
    ground_truth: NDArray[np.floating]
    tokens: list[str]
    query_pos: int


def _compute_datapoint_contributions(
    model: ComponentModel,
    target_model: LlamaSimpleMLP,
    input_ids: torch.Tensor,
    query_pos: int,
    layer_idx: int,
    mode: str,
    ci_threshold: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute per-head (q_c, k_c) contributions at each key position for ALL components.

    Returns:
        contributions: (n_q_heads, C_q, C_k, query_pos+1)
        ground_truth: (n_q_heads, query_pos+1)
    """
    with torch.no_grad():
        out = model(input_ids, cache_type="input")

    q_path = f"h.{layer_idx}.attn.q_proj"
    k_path = f"h.{layer_idx}.attn.k_proj"

    component_acts = model.get_all_component_acts(out.cache)
    q_acts_all = component_acts[q_path][0].float()  # (seq_len, C_q)
    k_acts_all = component_acts[k_path][0].float()  # (seq_len, C_k)

    q_acts = q_acts_all[query_pos]  # (C_q,)
    k_acts = k_acts_all[: query_pos + 1]  # (n_key_pos, C_k)

    # For binary mode, get per-token CI
    q_mask: torch.Tensor | None = None
    k_mask: torch.Tensor | None = None
    if mode == "binary":
        ci = model.calc_causal_importances(
            pre_weight_acts=out.cache,
            sampling="continuous",
            detach_inputs=True,
        ).lower_leaky
        q_ci = ci[q_path][0, query_pos]  # (C_q,)
        k_ci = ci[k_path][0, : query_pos + 1]  # (n_key_pos, C_k)
        q_mask = (q_ci > ci_threshold).float()
        k_mask = (k_ci > ci_threshold).float()

    q_component = model.components[q_path]
    k_component = model.components[k_path]
    assert isinstance(q_component, LinearComponents)
    assert isinstance(k_component, LinearComponents)

    C_q = q_component.U.shape[0]
    C_k = k_component.U.shape[0]

    block = target_model._h[layer_idx]
    n_q_heads = block.attn.n_head
    n_kv_heads = block.attn.n_key_value_heads
    head_dim = block.attn.head_dim
    g = n_q_heads // n_kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    # ALL U vectors reshaped to per-head
    U_q = q_component.U.float().reshape(C_q, n_q_heads, head_dim)
    U_k = k_component.U.float().reshape(C_k, n_kv_heads, head_dim)
    U_k_expanded = U_k.repeat_interleave(g, dim=1)  # (C_k, n_q_heads, head_dim)

    rotary_cos = block.attn.rotary_cos
    rotary_sin = block.attn.rotary_sin
    assert isinstance(rotary_cos, torch.Tensor)
    assert isinstance(rotary_sin, torch.Tensor)

    offsets = tuple(query_pos - t_k for t_k in range(query_pos + 1))

    W_all_heads = []
    for h in range(n_q_heads):
        A, B = compute_qk_rope_coefficients(U_q[:, h, :], U_k_expanded[:, h, :])
        W_h = evaluate_qk_at_offsets(A, B, rotary_cos, rotary_sin, offsets)
        W_all_heads.append(W_h)  # (n_offsets, C_q, C_k)

    # (n_q_heads, n_key_pos, C_q, C_k)
    W = torch.stack(W_all_heads)

    if mode == "weighted":
        # q_acts: (C_q,), k_acts: (n_key_pos, C_k)
        act_weights = q_acts[None, None, :, None] * k_acts[None, :, None, :]
        contributions = (scale * W * act_weights.to(W.device)).detach().cpu().numpy()
    else:
        assert q_mask is not None and k_mask is not None
        mask_weights = q_mask[None, None, :, None] * k_mask[None, :, None, :]
        contributions = (scale * W * mask_weights.to(W.device)).detach().cpu().numpy()

    # Transpose to (n_q_heads, C_q, C_k, n_key_pos)
    contributions = contributions.transpose(0, 2, 3, 1)

    # Ground truth: actual pre-softmax logits from target model
    for blk in target_model._h:
        blk.attn.flash_attention = False

    with torch.no_grad():
        results = collect_attention_patterns_with_logits(target_model, input_ids)

    _, logits = results[layer_idx]
    ground_truth = logits[0, :, query_pos, : query_pos + 1].float().cpu().numpy()

    return contributions, ground_truth


def _plot_combined_lines(
    result: DatapointContributions,
    run_id: str,
    layer_idx: int,
    mode: str,
    out_dir: Path,
    top_n: int,
    sample_idx: int,
) -> None:
    """4x2 grid: mean + legend on top, per-head subplots below.

    Top-N pairs are selected by peak absolute contribution on this specific datapoint
    (across all heads and key positions). The sum line includes ALL components.
    """
    W = result.contributions  # (n_q_heads, C_q, C_k, n_key_pos)
    gt = result.ground_truth  # (n_q_heads, n_key_pos)
    tokens = result.tokens
    query_pos = result.query_pos
    n_q_heads = W.shape[0]
    C_k = W.shape[2]
    n_key_pos = W.shape[3]

    x = list(range(n_key_pos))

    # Rank ALL (q, k) pairs by peak absolute contribution on this datapoint
    global_peak = np.abs(W).max(axis=(0, 3))  # (C_q, C_k)
    global_flat = np.argsort(global_peak.ravel())[::-1][:top_n]
    top_pairs = [divmod(int(idx), C_k) for idx in global_flat]

    # Use tab20 + tab20b for up to 40 distinct colors
    cmap_a = plt.get_cmap("tab20")
    cmap_b = plt.get_cmap("tab20b")
    pair_colors = {
        pair: cmap_a(i % 20) if i < 20 else cmap_b(i % 20) for i, pair in enumerate(top_pairs)
    }

    # Mean across heads
    W_mean = W.mean(axis=0)  # (C_q, C_k, n_key_pos)
    gt_mean = gt.mean(axis=0)  # (n_key_pos,)
    total_all_mean = W_mean.sum(axis=(0, 1))  # (n_key_pos,)

    fig = plt.figure(figsize=(12, 17))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.12)
    gs_top = outer[0].subgridspec(1, 2, wspace=0.12)
    gs_bottom = outer[1].subgridspec(3, 2, hspace=0.15, wspace=0.12)
    ax_mean = fig.add_subplot(gs_top[0, 0])
    ax_legend = fig.add_subplot(gs_top[0, 1])
    head_axes = [fig.add_subplot(gs_bottom[h // 2, h % 2]) for h in range(n_q_heads)]
    all_axes = [ax_mean] + head_axes

    # --- Mean subplot ---
    # Only plot top-N pairs as individual lines (skip "other" gray lines for clarity
    # since C_q * C_k can be huge)
    for qi, ki in top_pairs:
        ax_mean.plot(
            x,
            W_mean[qi, ki],
            color=pair_colors[(qi, ki)],
            marker="o",
            markersize=2,
            linewidth=1,
            label=f"q.{qi} -> k.{ki}",
        )

    ax_mean.plot(x, total_all_mean, color="black", linewidth=2)
    # Ground truth only meaningful in weighted mode (same units as component sum)
    show_gt = mode == "weighted"
    if show_gt:
        ax_mean.plot(x, gt_mean, color="red", linewidth=2, linestyle="--")

    ax_mean.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
    y_label = "Pre-softmax logit" if show_gt else "Weight-only contribution"
    ax_mean.set_ylabel(y_label)
    ax_mean.set_title("All Heads", fontsize=11, fontweight="bold")
    ax_mean.set_xticks(x)
    ax_mean.set_xticklabels(tokens[:n_key_pos], rotation=45, ha="right", fontsize=6)

    # Legend
    handles, labels = ax_mean.get_legend_handles_labels()
    sum_handle = plt.Line2D([0], [0], color="black", linewidth=2)
    ordered_handles = list(handles) + [sum_handle]
    ordered_labels = list(labels) + ["Sum (all components)"]
    if show_gt:
        gt_handle = plt.Line2D([0], [0], color="red", linewidth=2, linestyle="--")
        ordered_handles.append(gt_handle)
        ordered_labels.append("Ground truth")
    ax_legend.axis("off")
    n_legend_cols = 2 if len(ordered_labels) > 12 else 1
    ax_legend.legend(
        ordered_handles,
        ordered_labels,
        fontsize=8,
        loc="center left",
        frameon=False,
        ncol=n_legend_cols,
    )

    # --- Per-head subplots ---
    # Per-head visibility threshold: skip a pair in a head if its peak absolute
    # contribution in that head is < 5% of the head's total logit range
    HEAD_VIS_FRAC = 0.05

    for h, ax in enumerate(head_axes):
        total_h = W[h].sum(axis=(0, 1))
        head_range = float(total_h.max() - total_h.min()) or 1.0
        head_thresh = HEAD_VIS_FRAC * head_range

        for qi, ki in top_pairs:
            if np.abs(W[h, qi, ki]).max() < head_thresh:
                continue
            ax.plot(
                x,
                W[h, qi, ki],
                color=pair_colors[(qi, ki)],
                marker="o",
                markersize=2,
                linewidth=1,
            )

        ax.plot(x, total_h, color="black", linewidth=2)
        if show_gt:
            ax.plot(x, gt[h], color="red", linewidth=2, linestyle="--")

        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
        ax.set_title(f"H{h}", fontsize=11, fontweight="bold")
        ax.set_xticks(x)

        if h >= n_q_heads - 2:
            ax.set_xticklabels(tokens[:n_key_pos], rotation=45, ha="right", fontsize=6)
            ax.set_xlabel("Key position")
        else:
            ax.set_xticklabels([])

        if h % 2 == 0:
            ax.set_ylabel(y_label)

    # Shared y-axis
    all_ylims = [ax.get_ylim() for ax in all_axes]
    ymin = min(lo for lo, _ in all_ylims)
    ymax = max(hi for _, hi in all_ylims)
    for ax in all_axes:
        ax.set_ylim(ymin, ymax)

    query_token = tokens[query_pos]
    fig.suptitle(
        f"{run_id}  |  Layer {layer_idx}  |  sample {sample_idx}"
        f"  |  query pos {query_pos} ({query_token!r})  |  {mode}",
        fontsize=12,
        fontweight="bold",
    )

    layer_dir = out_dir / f"layer{layer_idx}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    path = layer_dir / f"sample{sample_idx}_pos{query_pos}_{mode}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_qk_c_datapoint(
    wandb_path: ModelPath,
    layer: int,
    sample_indices: int | list[int] = 0,
    query_positions: int | list[int] = 5,
    mode: str = "weighted",
    ci_threshold: float = 0.01,
    top_n_pairs: int = 40,
) -> None:
    """Plot data-specific QK component contributions for dataset samples.

    Args:
        wandb_path: WandB run path (e.g. wandb:goodfire/spd/runs/<run_id>).
        layer: Layer index.
        sample_indices: Dataset sample index or list of indices.
        query_positions: Query token position(s) (0-indexed). Single int applied to all
            samples, or list matched 1:1 with sample_indices.
        mode: "weighted" (scale by activations) or "binary" (CI threshold gating).
        ci_threshold: Per-token CI threshold for binary mode.
        top_n_pairs: Number of top (q, k) pairs to highlight (ranked by actual contribution
            on each datapoint).
    """
    assert mode in ("weighted", "binary"), f"mode must be 'weighted' or 'binary', got {mode!r}"

    if isinstance(sample_indices, int):
        sample_indices = [sample_indices]
    if isinstance(query_positions, int):
        query_positions = [query_positions] * len(sample_indices)
    assert len(sample_indices) == len(query_positions), (
        f"Got {len(sample_indices)} sample_indices but {len(query_positions)} query_positions"
    )

    _entity, _project, run_id = parse_wandb_run_path(str(wandb_path))
    run_info = SPDRunInfo.from_path(wandb_path)
    config = run_info.config

    out_dir = SCRIPT_DIR / "out" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    assert config.tokenizer_name is not None
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    model = ComponentModel.from_run_info(run_info)
    model.eval()

    target_model = model.target_model
    assert isinstance(target_model, LlamaSimpleMLP)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load dataset
    task_config = config.task_config
    assert isinstance(task_config, LMTaskConfig)
    seq_len = target_model.config.n_ctx
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
        dataset_config=dataset_config,
        batch_size=1,
        buffer_size=1000,
    )

    # Collect the requested samples
    max_idx = max(sample_indices)
    samples: dict[int, torch.Tensor] = {}
    for i, batch in enumerate(loader):
        if i > max_idx:
            break
        if i in sample_indices:
            samples[i] = batch[task_config.column_name][:, :seq_len]
    assert len(samples) == len(sample_indices), (
        f"Could only load {len(samples)} of {len(sample_indices)} requested samples"
    )

    for sample_idx, query_pos in zip(sample_indices, query_positions, strict=True):
        input_ids = samples[sample_idx].to(device)
        token_strs = [tokenizer.decode(t) for t in input_ids[0]]  # pyright: ignore[reportAttributeAccessIssue]
        assert query_pos < input_ids.shape[1], (
            f"query_pos {query_pos} >= seq_len {input_ids.shape[1]} for sample {sample_idx}"
        )

        logger.info(f"Sample {sample_idx}: query pos {query_pos} ({token_strs[query_pos]!r})")

        contributions, ground_truth = _compute_datapoint_contributions(
            model, target_model, input_ids, query_pos, layer, mode, ci_threshold
        )

        result = DatapointContributions(
            contributions=contributions,
            ground_truth=ground_truth,
            tokens=token_strs,
            query_pos=query_pos,
        )

        _plot_combined_lines(result, run_id, layer, mode, out_dir, top_n_pairs, sample_idx)

    logger.info(f"All plots saved to {out_dir}")


if __name__ == "__main__":
    fire.Fire(plot_qk_c_datapoint)
