"""Analyze how Q335 and K206 modulate induction behavior in Layer 2.

Hypothesis: Q335 and K206 are anti-induction — they suppress induction in H4
and redistribute it across heads. When they're inactive (~30% of the time),
induction concentrates in H4.

This script:
1. Runs 500 dataset samples through the model
2. Categorizes by whether Q335/K206 are jointly active or inactive
3. Compares per-head induction scores between the two groups
4. Compares full attention offset profiles between the two groups
5. Shows specific example sequences from each group
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
from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from spd.scripts.collect_attention_patterns import collect_attention_patterns_with_logits

OUT_DIR = Path(__file__).parent / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WANDB_PATH = "wandb:goodfire/spd/runs/s-55ea3f9b"
LAYER = 2
N_SAMPLES = 500
CI_THRESHOLD = 0.1  # threshold for "active"


def main() -> None:
    run_info = SPDRunInfo.from_path(WANDB_PATH)
    model = ComponentModel.from_run_info(run_info)
    model.eval()
    config = run_info.config

    target_model = model.target_model
    assert isinstance(target_model, LlamaSimpleMLP)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for blk in target_model._h:
        blk.attn.flash_attention = False

    assert config.tokenizer_name is not None
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

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
    loader, _ = create_data_loader(dataset_config=dataset_config, batch_size=1, buffer_size=1000)

    q_path = f"h.{LAYER}.attn.q_proj"
    k_path = f"h.{LAYER}.attn.k_proj"

    n_heads = target_model._h[LAYER].attn.n_head

    # Collect per-head attention statistics for samples where Q335+K206 are
    # jointly active vs jointly inactive
    max_offset = 16

    # Store per-sample attention offset profiles
    both_active_offsets = []   # list of (n_heads, max_offset) arrays
    both_inactive_offsets = []
    q_only_offsets = []
    k_only_offsets = []

    # Also track induction-specific attention (attending to token after previous occurrence)
    both_active_induction = []
    both_inactive_induction = []

    # Example tokens for display
    both_active_examples = []
    both_inactive_examples = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= N_SAMPLES:
                break

            input_ids = batch[task_config.column_name][:, :seq_len].to(device)
            T = input_ids.shape[1]

            # Get CI values
            out = model(input_ids, cache_type="input")
            ci = model.calc_causal_importances(
                pre_weight_acts=out.cache, sampling="continuous", detach_inputs=True
            ).lower_leaky

            # Mean CI across sequence positions for Q335 and K206
            q335_ci_mean = ci[q_path][0, :, 335].mean().item()
            k206_ci_mean = ci[k_path][0, :, 206].mean().item()

            q335_active = q335_ci_mean > CI_THRESHOLD
            k206_active = k206_ci_mean > CI_THRESHOLD

            # Get actual attention patterns
            results = collect_attention_patterns_with_logits(target_model, input_ids)
            attn_weights = torch.softmax(results[LAYER][1], dim=-1)[0]  # (n_heads, T, T)

            # Compute offset profile: mean attention at each offset
            offset_profile = np.zeros((n_heads, max_offset))
            for d in range(max_offset):
                diag = torch.diagonal(attn_weights, offset=-d, dim1=-2, dim2=-1)
                offset_profile[:, d] = diag.float().mean(dim=1).cpu().numpy()

            # Compute induction score for this sample
            # Induction: for repeated tokens, attention to token after previous occurrence
            token_list = input_ids[0].cpu().tolist()
            induction_scores_per_head = np.zeros(n_heads)
            induction_count = 0
            for t in range(2, T):
                for prev_t in range(t - 1):
                    if token_list[prev_t] == token_list[t] and prev_t + 1 < t:
                        for h in range(n_heads):
                            induction_scores_per_head[h] += attn_weights[h, t, prev_t + 1].item()
                        induction_count += 1
            if induction_count > 0:
                induction_scores_per_head /= induction_count

            # Categorize
            if q335_active and k206_active:
                both_active_offsets.append(offset_profile)
                both_active_induction.append(induction_scores_per_head)
                if len(both_active_examples) < 5:
                    tokens = [tokenizer.decode(t) for t in input_ids[0][:20]]  # pyright: ignore[reportAttributeAccessIssue]
                    both_active_examples.append((i, q335_ci_mean, k206_ci_mean, tokens))
            elif not q335_active and not k206_active:
                both_inactive_offsets.append(offset_profile)
                both_inactive_induction.append(induction_scores_per_head)
                if len(both_inactive_examples) < 5:
                    tokens = [tokenizer.decode(t) for t in input_ids[0][:20]]  # pyright: ignore[reportAttributeAccessIssue]
                    both_inactive_examples.append((i, q335_ci_mean, k206_ci_mean, tokens))
            elif q335_active and not k206_active:
                q_only_offsets.append(offset_profile)
            else:
                k_only_offsets.append(offset_profile)

            if (i + 1) % 100 == 0:
                logger.info(
                    f"Processed {i+1}/{N_SAMPLES}: "
                    f"both_active={len(both_active_offsets)}, "
                    f"both_inactive={len(both_inactive_offsets)}, "
                    f"q_only={len(q_only_offsets)}, k_only={len(k_only_offsets)}"
                )

    logger.info(
        f"Final counts: both_active={len(both_active_offsets)}, "
        f"both_inactive={len(both_inactive_offsets)}, "
        f"q_only={len(q_only_offsets)}, k_only={len(k_only_offsets)}"
    )

    # --- Plot 1: Attention offset profiles ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    active_mean = np.mean(both_active_offsets, axis=0) if both_active_offsets else np.zeros((n_heads, max_offset))
    inactive_mean = np.mean(both_inactive_offsets, axis=0) if both_inactive_offsets else np.zeros((n_heads, max_offset))

    for h in range(n_heads):
        row, col = divmod(h, 3)
        ax = axes[row, col]
        x = list(range(max_offset))
        ax.plot(x, active_mean[h], "b-", linewidth=2, label=f"Both active ({len(both_active_offsets)})")
        ax.plot(x, inactive_mean[h], "r--", linewidth=2, label=f"Both inactive ({len(both_inactive_offsets)})")
        ax.set_title(f"H{h}", fontweight="bold")
        ax.set_xlabel("Offset")
        ax.set_ylabel("Mean attention")
        if h == 0:
            ax.legend(fontsize=7)

    fig.suptitle(
        f"Layer {LAYER}: Attention offset profiles\n"
        f"Q335+K206 both active vs both inactive (CI threshold={CI_THRESHOLD})",
        fontweight="bold",
    )
    fig.tight_layout()
    path = OUT_DIR / "induction_modulation_offset_profiles.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")

    # --- Plot 2: Per-head induction scores ---
    if both_active_induction and both_inactive_induction:
        active_ind = np.mean(both_active_induction, axis=0)
        inactive_ind = np.mean(both_inactive_induction, axis=0)

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(n_heads)
        width = 0.35
        ax.bar(x - width / 2, active_ind, width, label=f"Both active ({len(both_active_induction)})", color="tab:blue")
        ax.bar(x + width / 2, inactive_ind, width, label=f"Both inactive ({len(both_inactive_induction)})", color="tab:red")
        ax.set_xlabel("Head")
        ax.set_ylabel("Mean induction score")
        ax.set_title(f"Layer {LAYER}: Per-head induction scores\nQ335+K206 active vs inactive")
        ax.set_xticks(x)
        ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
        ax.legend()

        path = OUT_DIR / "induction_modulation_scores.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {path}")

    # --- Print example sequences ---
    logger.info("\n=== Example sequences where Q335+K206 are BOTH ACTIVE ===")
    for idx, q_ci, k_ci, toks in both_active_examples:
        logger.info(f"  Sample {idx}: Q335 CI={q_ci:.3f}, K206 CI={k_ci:.3f}")
        logger.info(f"    Tokens: {''.join(toks)}")

    logger.info("\n=== Example sequences where Q335+K206 are BOTH INACTIVE ===")
    for idx, q_ci, k_ci, toks in both_inactive_examples:
        logger.info(f"  Sample {idx}: Q335 CI={q_ci:.3f}, K206 CI={k_ci:.3f}")
        logger.info(f"    Tokens: {''.join(toks)}")


if __name__ == "__main__":
    main()
