"""Analyze per-head weight norms and QK interactions of Layer 2 moderate components.

Produces:
1. Per-head U-vector norms for top Q and K components (like Fig 4 in the paper)
2. Weight-only QK interaction heatmap between moderate Q and K components
3. Comparison of attention patterns on samples where Q279 (social/human) is active vs inactive
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from spd.configs import LMTaskConfig
from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import LinearComponents
from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from spd.scripts.collect_attention_patterns import collect_attention_patterns_with_logits
from spd.scripts.rope_aware_qk import compute_qk_rope_coefficients, evaluate_qk_at_offsets

OUT_DIR = Path(__file__).parent / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAYER = 2
WANDB_PATH = "wandb:goodfire/spd/runs/s-55ea3f9b"

# Top moderate components in layer 2
Q_COMPONENTS = [335, 270, 279, 436, 66, 207, 261, 46]
K_COMPONENTS = [206, 224, 327, 167, 107, 320, 373, 1]


def plot_per_head_norms(
    model: ComponentModel,
    target_model: LlamaSimpleMLP,
) -> None:
    """Plot per-head U-vector norms for moderate Q and K components."""
    q_path = f"h.{LAYER}.attn.q_proj"
    k_path = f"h.{LAYER}.attn.k_proj"

    q_comp = model.components[q_path]
    k_comp = model.components[k_path]
    assert isinstance(q_comp, LinearComponents)
    assert isinstance(k_comp, LinearComponents)

    block = target_model._h[LAYER]
    n_q_heads = block.attn.n_head
    n_kv_heads = block.attn.n_key_value_heads
    head_dim = block.attn.head_dim

    # Q component per-head norms
    fig, (ax_q, ax_k) = plt.subplots(1, 2, figsize=(14, 5))

    U_q_all = q_comp.U.detach().float()  # (C, d_out)
    for c in Q_COMPONENTS:
        u = U_q_all[c].reshape(n_q_heads, head_dim)
        norms = u.norm(dim=1).cpu().numpy()
        ax_q.bar(
            [h + Q_COMPONENTS.index(c) * 0.1 for h in range(n_q_heads)],
            norms,
            width=0.1,
            label=f"Q{c}",
        )

    ax_q.set_xlabel("Head")
    ax_q.set_ylabel("||U|| per head")
    ax_q.set_title(f"Layer {LAYER} Q component per-head norms")
    ax_q.set_xticks(range(n_q_heads))
    ax_q.set_xticklabels([f"H{h}" for h in range(n_q_heads)])
    ax_q.legend(fontsize=7)

    # K component per-head norms
    U_k_all = k_comp.U.detach().float()  # (C, d_out)
    for c in K_COMPONENTS:
        u = U_k_all[c].reshape(n_kv_heads, head_dim)
        norms = u.norm(dim=1).cpu().numpy()
        # Expand for GQA display
        g = n_q_heads // n_kv_heads
        expanded_norms = np.repeat(norms, g)
        ax_k.bar(
            [h + K_COMPONENTS.index(c) * 0.1 for h in range(n_q_heads)],
            expanded_norms,
            width=0.1,
            label=f"K{c}",
        )

    ax_k.set_xlabel("Head")
    ax_k.set_ylabel("||U|| per head")
    ax_k.set_title(f"Layer {LAYER} K component per-head norms")
    ax_k.set_xticks(range(n_q_heads))
    ax_k.set_xticklabels([f"H{h}" for h in range(n_q_heads)])
    ax_k.legend(fontsize=7)

    fig.tight_layout()
    path = OUT_DIR / "layer2_per_head_norms.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_qk_interaction_heatmap(
    model: ComponentModel,
    target_model: LlamaSimpleMLP,
) -> None:
    """Plot weight-only QK interaction strength between moderate Q and K components at offset 1."""
    q_path = f"h.{LAYER}.attn.q_proj"
    k_path = f"h.{LAYER}.attn.k_proj"

    q_comp = model.components[q_path]
    k_comp = model.components[k_path]
    assert isinstance(q_comp, LinearComponents)
    assert isinstance(k_comp, LinearComponents)

    block = target_model._h[LAYER]
    n_q_heads = block.attn.n_head
    n_kv_heads = block.attn.n_key_value_heads
    head_dim = block.attn.head_dim
    g = n_q_heads // n_kv_heads

    rotary_cos = block.attn.rotary_cos
    rotary_sin = block.attn.rotary_sin
    assert isinstance(rotary_cos, torch.Tensor)
    assert isinstance(rotary_sin, torch.Tensor)

    U_q = q_comp.U[Q_COMPONENTS].detach().float().reshape(len(Q_COMPONENTS), n_q_heads, head_dim)
    U_k = k_comp.U[K_COMPONENTS].detach().float().reshape(len(K_COMPONENTS), n_kv_heads, head_dim)
    U_k_expanded = U_k.repeat_interleave(g, dim=1)

    offsets = (0, 1, 2, 3, 4)

    # Compute per-head interactions
    n_offsets = len(offsets)
    # (n_heads, n_offsets, n_q, n_k)
    W_per_head = np.zeros((n_q_heads, n_offsets, len(Q_COMPONENTS), len(K_COMPONENTS)))

    for h in range(n_q_heads):
        A, B = compute_qk_rope_coefficients(U_q[:, h, :], U_k_expanded[:, h, :])
        W_h = evaluate_qk_at_offsets(A, B, rotary_cos, rotary_sin, offsets)
        W_per_head[h] = W_h.cpu().numpy()

    # Plot heatmap at offset=1 (previous token) - mean across heads
    W_offset1_mean = W_per_head[:, 1, :, :].mean(axis=0)  # (n_q, n_k)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))

    # Mean + per-head heatmaps at offset 1
    all_data = [W_offset1_mean] + [W_per_head[h, 1] for h in range(n_q_heads)]
    titles = ["Mean"] + [f"H{h}" for h in range(n_q_heads)]
    vmax = max(np.abs(d).max() for d in all_data)

    for idx, (data, title) in enumerate(zip(all_data, titles)):
        row, col = divmod(idx, 4)
        ax = axes[row, col]
        im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title, fontweight="bold")
        ax.set_yticks(range(len(Q_COMPONENTS)))
        ax.set_yticklabels([f"Q{c}" for c in Q_COMPONENTS], fontsize=7)
        ax.set_xticks(range(len(K_COMPONENTS)))
        ax.set_xticklabels([f"K{c}" for c in K_COMPONENTS], fontsize=7, rotation=45)

    # Hide last cell
    axes[1, 3].set_visible(False)

    fig.suptitle(f"Layer {LAYER}: Weight-only QK interactions at offset=1 (moderate components)", fontweight="bold")
    fig.tight_layout()
    path = OUT_DIR / "layer2_qk_interaction_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def analyze_conditional_attention(
    model: ComponentModel,
    target_model: LlamaSimpleMLP,
    tokenizer: AutoTokenizer,  # pyright: ignore[reportMissingTypeArgument]
    config: object,
) -> None:
    """Compare attention patterns when Q279 (social/human content) is active vs inactive."""
    task_config = config.task_config  # pyright: ignore[reportAttributeAccessIssue]
    assert isinstance(task_config, LMTaskConfig)
    seq_len = target_model.config.n_ctx

    dataset_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=config.tokenizer_name,  # pyright: ignore[reportAttributeAccessIssue]
        split=task_config.eval_data_split,
        n_ctx=task_config.max_seq_len,
        is_tokenized=task_config.is_tokenized,
        streaming=task_config.streaming,
        column_name=task_config.column_name,
        shuffle_each_epoch=False,
    )
    loader, _ = create_data_loader(dataset_config=dataset_config, batch_size=1, buffer_size=1000)

    device = next(model.parameters()).device
    q_path = f"h.{LAYER}.attn.q_proj"

    for blk in target_model._h:
        blk.attn.flash_attention = False

    # Collect samples, categorize by Q279 activation
    active_attn = []  # attention patterns when Q279 is active at some position
    inactive_attn = []
    n_samples = 200

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_samples:
                break
            input_ids = batch[task_config.column_name][:, :seq_len].to(device)

            out = model(input_ids, cache_type="input")
            ci = model.calc_causal_importances(
                pre_weight_acts=out.cache, sampling="continuous", detach_inputs=True
            ).lower_leaky

            q279_ci = ci[q_path][0, :, 279]  # (seq_len,)
            is_active = (q279_ci > 0.1).any().item()

            results = collect_attention_patterns_with_logits(target_model, input_ids)
            _, logits = results[LAYER]
            attn_weights = torch.softmax(logits, dim=-1)  # (1, n_heads, T, T)

            if is_active:
                active_attn.append(attn_weights[0].cpu())
            else:
                inactive_attn.append(attn_weights[0].cpu())

            if (i + 1) % 50 == 0:
                logger.info(
                    f"Processed {i+1}/{n_samples}: {len(active_attn)} active, {len(inactive_attn)} inactive"
                )

    logger.info(f"Q279 active in {len(active_attn)}/{n_samples} samples")

    if not active_attn or not inactive_attn:
        logger.warning("Not enough samples in one category, skipping conditional analysis")
        return

    # Compute mean offset-1 attention for active vs inactive
    active_stack = torch.stack(active_attn)  # (N_active, n_heads, T, T)
    inactive_stack = torch.stack(inactive_attn)

    n_heads = active_stack.shape[1]
    T = active_stack.shape[2]

    # Mean attention to offset 1 (previous token) per head
    active_prev = []
    inactive_prev = []
    for h in range(n_heads):
        # diagonal offset=-1 gives attention to previous token
        active_diag = torch.diagonal(active_stack[:, h], offset=-1, dim1=-2, dim2=-1)
        inactive_diag = torch.diagonal(inactive_stack[:, h], offset=-1, dim1=-2, dim2=-1)
        active_prev.append(active_diag.mean().item())
        inactive_prev.append(inactive_diag.mean().item())

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(n_heads)
    width = 0.35
    ax.bar(x - width / 2, active_prev, width, label=f"Q279 active ({len(active_attn)} samples)")
    ax.bar(x + width / 2, inactive_prev, width, label=f"Q279 inactive ({len(inactive_attn)} samples)")
    ax.set_xlabel("Head")
    ax.set_ylabel("Mean attention to previous token")
    ax.set_title(f"Layer {LAYER}: Attention to t-1 when Q279 (social content) is active vs inactive")
    ax.set_xticks(x)
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.legend()

    path = OUT_DIR / "layer2_q279_conditional_attention.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")

    # Also look at mean attention across ALL offsets
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    max_offset = min(16, T)
    offsets_range = range(max_offset)

    for h in range(n_heads):
        row, col = divmod(h, 3)
        ax = axes[row, col]

        active_by_offset = []
        inactive_by_offset = []
        for d in offsets_range:
            active_diag = torch.diagonal(active_stack[:, h], offset=-d, dim1=-2, dim2=-1)
            inactive_diag = torch.diagonal(inactive_stack[:, h], offset=-d, dim1=-2, dim2=-1)
            active_by_offset.append(active_diag.mean().item())
            inactive_by_offset.append(inactive_diag.mean().item())

        ax.plot(list(offsets_range), active_by_offset, "b-", label="Q279 active", linewidth=2)
        ax.plot(list(offsets_range), inactive_by_offset, "r--", label="Q279 inactive", linewidth=2)
        ax.set_title(f"H{h}", fontweight="bold")
        ax.set_xlabel("Offset")
        ax.set_ylabel("Mean attention")
        if h == 0:
            ax.legend(fontsize=8)

    fig.suptitle(
        f"Layer {LAYER}: Attention offset profiles — Q279 active vs inactive", fontweight="bold"
    )
    fig.tight_layout()
    path = OUT_DIR / "layer2_q279_conditional_offset_profiles.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def main() -> None:
    run_info = SPDRunInfo.from_path(WANDB_PATH)
    model = ComponentModel.from_run_info(run_info)
    model.eval()

    target_model = model.target_model
    assert isinstance(target_model, LlamaSimpleMLP)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    assert run_info.config.tokenizer_name is not None
    tokenizer = AutoTokenizer.from_pretrained(run_info.config.tokenizer_name)

    logger.info("=== Plot 1: Per-head weight norms ===")
    plot_per_head_norms(model, target_model)

    logger.info("=== Plot 2: QK interaction heatmap ===")
    plot_qk_interaction_heatmap(model, target_model)

    logger.info("=== Plot 3: Conditional attention analysis (Q279) ===")
    analyze_conditional_attention(model, target_model, tokenizer, run_info.config)


if __name__ == "__main__":
    main()
