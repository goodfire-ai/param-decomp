"""Analyze sparse semantic components in Layer 2 and their effect on attention.

Focuses on Q279 (ci=0.13, social/human content) and Q270 (ci=0.24, research/technical).
These fire rarely enough to get clean active/inactive splits.

For each component:
1. Find positions where the component is active vs inactive
2. Compare the attention pattern at those positions across heads
3. Show specific examples of what the component attends to when active
4. Examine whether the effect is distributed across heads
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
N_SAMPLES = 300


def analyze_component(
    component_idx: int,
    module: str,
    label: str,
    model: ComponentModel,
    target_model: LlamaSimpleMLP,
    tokenizer: AutoTokenizer,  # pyright: ignore[reportMissingTypeArgument]
    loader: object,
    task_config: LMTaskConfig,
    seq_len: int,
    device: torch.device,
) -> None:
    module_path = f"h.{LAYER}.attn.{module}"
    n_heads = target_model._h[LAYER].attn.n_head
    max_offset = 16

    # Per-position analysis: collect attention at positions where component is active vs inactive
    active_offsets_per_head = [[] for _ in range(n_heads)]  # list of (offset,) values
    inactive_offsets_per_head = [[] for _ in range(n_heads)]

    active_position_count = 0
    inactive_position_count = 0

    # Collect example contexts
    active_contexts: list[tuple[int, int, list[str], float]] = []  # (sample, pos, tokens, ci_val)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= N_SAMPLES:
                break
            input_ids = batch[task_config.column_name][:, :seq_len].to(device)
            T = input_ids.shape[1]

            out = model(input_ids, cache_type="input")
            ci = model.calc_causal_importances(
                pre_weight_acts=out.cache, sampling="continuous", detach_inputs=True
            ).lower_leaky

            comp_ci = ci[module_path][0, :, component_idx]  # (T,)

            results = collect_attention_patterns_with_logits(target_model, input_ids)
            attn_weights = torch.softmax(results[LAYER][1], dim=-1)[0]  # (n_heads, T, T)

            for t in range(max_offset, T):
                ci_val = comp_ci[t].item()
                is_active = ci_val > 0.5

                for h in range(n_heads):
                    offsets = []
                    for d in range(max_offset):
                        offsets.append(attn_weights[h, t, t - d].item())

                    if is_active:
                        active_offsets_per_head[h].append(offsets)
                    else:
                        inactive_offsets_per_head[h].append(offsets)

                if is_active:
                    active_position_count += 1
                    if len(active_contexts) < 10:
                        start = max(0, t - 5)
                        end = min(T, t + 3)
                        tok_strs = [tokenizer.decode(tid) for tid in input_ids[0, start:end]]  # pyright: ignore[reportAttributeAccessIssue]
                        active_contexts.append((i, t, tok_strs, ci_val))
                else:
                    inactive_position_count += 1

            if (i + 1) % 100 == 0:
                logger.info(
                    f"  [{label}] Processed {i+1}/{N_SAMPLES}: "
                    f"active_pos={active_position_count}, inactive_pos={inactive_position_count}"
                )

    logger.info(
        f"  [{label}] Final: active_pos={active_position_count}, "
        f"inactive_pos={inactive_position_count}"
    )

    if active_position_count == 0:
        logger.warning(f"  [{label}] No active positions found, skipping plots")
        return

    # --- Plot: Attention offset profiles ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for h in range(n_heads):
        row, col = divmod(h, 3)
        ax = axes[row, col]

        active_arr = np.array(active_offsets_per_head[h])
        inactive_arr = np.array(inactive_offsets_per_head[h])

        active_mean = active_arr.mean(axis=0) if len(active_arr) > 0 else np.zeros(max_offset)
        inactive_mean = inactive_arr.mean(axis=0) if len(inactive_arr) > 0 else np.zeros(max_offset)

        x = list(range(max_offset))
        ax.plot(x, active_mean, "b-", linewidth=2,
                label=f"Active ({active_position_count} pos)")
        ax.plot(x, inactive_mean, "r--", linewidth=2,
                label=f"Inactive ({inactive_position_count} pos)")

        ax.set_title(f"H{h}", fontweight="bold")
        ax.set_xlabel("Offset")
        ax.set_ylabel("Mean attention")
        if h == 0:
            ax.legend(fontsize=7)

    fig.suptitle(
        f"Layer {LAYER}: Attention offset profiles when {label} is active vs inactive\n"
        f"(CI > 0.5 at query position)",
        fontweight="bold",
    )
    fig.tight_layout()
    path = OUT_DIR / f"sparse_{label}_offset_profiles.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {path}")

    # --- Plot: Difference (active - inactive) ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for h in range(n_heads):
        row, col = divmod(h, 3)
        ax = axes[row, col]

        active_arr = np.array(active_offsets_per_head[h])
        inactive_arr = np.array(inactive_offsets_per_head[h])
        active_mean = active_arr.mean(axis=0) if len(active_arr) > 0 else np.zeros(max_offset)
        inactive_mean = inactive_arr.mean(axis=0) if len(inactive_arr) > 0 else np.zeros(max_offset)
        diff = active_mean - inactive_mean

        x = list(range(max_offset))
        ax.bar(x, diff, color=["tab:blue" if d > 0 else "tab:red" for d in diff])
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"H{h}", fontweight="bold")
        ax.set_xlabel("Offset")
        ax.set_ylabel("Active - Inactive")

    fig.suptitle(
        f"Layer {LAYER}: Attention difference when {label} is active\n"
        f"(positive = more attention when active)",
        fontweight="bold",
    )
    fig.tight_layout()
    path = OUT_DIR / f"sparse_{label}_attention_diff.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {path}")

    # --- Print example active contexts ---
    logger.info(f"\n  === Example contexts where {label} is active ===")
    for sample_idx, pos, toks, ci_val in active_contexts:
        logger.info(f"    Sample {sample_idx}, pos {pos}, CI={ci_val:.3f}: {'|'.join(toks)}")


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

    # Analyze Q279 (social/human content, ci=0.13)
    logger.info("=== Analyzing Q279 (social/human content) ===")
    # Need to recreate loader for each component since it's a generator
    loader1, _ = create_data_loader(dataset_config=dataset_config, batch_size=1, buffer_size=1000)
    analyze_component(279, "q_proj", "Q279", model, target_model, tokenizer, loader1, task_config, seq_len, device)

    # Analyze Q270 (research/technical, ci=0.24)
    logger.info("\n=== Analyzing Q270 (research/technical) ===")
    loader2, _ = create_data_loader(dataset_config=dataset_config, batch_size=1, buffer_size=1000)
    analyze_component(270, "q_proj", "Q270", model, target_model, tokenizer, loader2, task_config, seq_len, device)

    # Analyze Q436 (LaTeX/markup, ci=0.12)
    logger.info("\n=== Analyzing Q436 (LaTeX/markup) ===")
    loader3, _ = create_data_loader(dataset_config=dataset_config, batch_size=1, buffer_size=1000)
    analyze_component(436, "q_proj", "Q436", model, target_model, tokenizer, loader3, task_config, seq_len, device)

    # Also analyze K224 (complex technical nouns, ci=0.18)
    logger.info("\n=== Analyzing K224 (complex technical nouns) ===")
    loader4, _ = create_data_loader(dataset_config=dataset_config, batch_size=1, buffer_size=1000)
    analyze_component(224, "k_proj", "K224", model, target_model, tokenizer, loader4, task_config, seq_len, device)


if __name__ == "__main__":
    main()
