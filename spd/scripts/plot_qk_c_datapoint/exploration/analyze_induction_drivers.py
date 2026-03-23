"""Identify which QK component pairs drive H4's induction behavior.

At actual repeated-token positions (where token t matches a previous token at t'),
use the actual ground-truth attention logits from collect_attention_patterns to
measure induction strength, and correlate with per-position CI values to find
which components are most associated with strong H4 induction.

This avoids RoPE decomposition issues by working with attention patterns directly.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

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
INDUCTION_HEAD = 4


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

    task_config = config.task_config
    assert isinstance(task_config, LMTaskConfig)
    seq_len = target_model.config.n_ctx

    assert config.tokenizer_name is not None
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
    n_q_components = model.components[q_path].U.shape[0]
    n_k_components = model.components[k_path].U.shape[0]

    # For each induction event, record:
    # - The H4 attention weight to the induction target
    # - The per-component CI values at the query and key positions
    # Then correlate: which Q/K component CIs predict strong H4 induction attention?

    # Collect per-event data
    h4_induction_weights: list[float] = []
    q_ci_at_events: list[torch.Tensor] = []  # each: (n_q_components,)
    k_ci_at_events: list[torch.Tensor] = []  # each: (n_k_components,)

    # Also per-head induction weights for comparison
    all_heads_induction_weights: list[np.ndarray] = []

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

            q_ci = ci[q_path][0]  # (T, C_q)
            k_ci = ci[k_path][0]  # (T, C_k)

            results = collect_attention_patterns_with_logits(target_model, input_ids)
            attn_weights = torch.softmax(results[LAYER][1], dim=-1)[0]  # (n_heads, T, T)

            token_list = input_ids[0].cpu().tolist()

            for t in range(4, T):
                best_prev = -1
                for prev_t in range(t - 1):
                    if token_list[prev_t] == token_list[t] and prev_t + 1 < t:
                        best_prev = prev_t
                        break  # first occurrence

                if best_prev >= 0:
                    induction_key = best_prev + 1
                    h4_weight = attn_weights[INDUCTION_HEAD, t, induction_key].item()
                    all_head_weights = np.array([
                        attn_weights[h, t, induction_key].item() for h in range(n_heads)
                    ])

                    h4_induction_weights.append(h4_weight)
                    all_heads_induction_weights.append(all_head_weights)
                    q_ci_at_events.append(q_ci[t].cpu())
                    k_ci_at_events.append(k_ci[induction_key].cpu())

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i+1}/{N_SAMPLES}, events: {len(h4_induction_weights)}")

    logger.info(f"\nTotal induction events: {len(h4_induction_weights)}")
    logger.info(f"Mean H4 induction weight: {np.mean(h4_induction_weights):.4f}")

    if len(h4_induction_weights) < 100:
        logger.warning("Too few induction events, skipping")
        return

    h4_arr = np.array(h4_induction_weights)
    q_ci_arr = torch.stack(q_ci_at_events).numpy()  # (N_events, C_q)
    k_ci_arr = torch.stack(k_ci_at_events).numpy()  # (N_events, C_k)
    all_heads_arr = np.array(all_heads_induction_weights)  # (N_events, n_heads)

    # Correlate each Q component's CI with H4 induction weight
    q_corrs = []
    for c in range(n_q_components):
        ci_vals = q_ci_arr[:, c]
        if ci_vals.std() < 1e-6:
            q_corrs.append((c, 0.0, 1.0))
            continue
        r, p = stats.pearsonr(ci_vals, h4_arr)
        q_corrs.append((c, r, p))

    k_corrs = []
    for c in range(n_k_components):
        ci_vals = k_ci_arr[:, c]
        if ci_vals.std() < 1e-6:
            k_corrs.append((c, 0.0, 1.0))
            continue
        r, p = stats.pearsonr(ci_vals, h4_arr)
        k_corrs.append((c, r, p))

    # Sort by absolute correlation
    q_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    k_corrs.sort(key=lambda x: abs(x[1]), reverse=True)

    semantic_q = {335, 270, 279, 436, 66, 207, 261, 46}
    semantic_k = {206, 224, 327, 167, 107, 320, 373, 1}

    logger.info("\n=== Top Q components correlated with H4 induction ===")
    for c, r, p in q_corrs[:25]:
        marker = " *** SEMANTIC" if c in semantic_q else ""
        logger.info(f"  Q{c}: r={r:+.4f}, p={p:.2e}{marker}")

    logger.info("\n=== Top K components correlated with H4 induction ===")
    for c, r, p in k_corrs[:25]:
        marker = " *** SEMANTIC" if c in semantic_k else ""
        logger.info(f"  K{c}: r={r:+.4f}, p={p:.2e}{marker}")

    # Also correlate with other heads to see if semantic components affect them
    logger.info("\n=== Semantic component correlation with induction per head ===")
    for label, comp_idx, ci_arr_sel in [
        ("Q335", 335, q_ci_arr[:, 335]),
        ("Q270", 270, q_ci_arr[:, 270]),
        ("K206", 206, k_ci_arr[:, 206]),
        ("K224", 224, k_ci_arr[:, 224]),
    ]:
        logger.info(f"  {label}:")
        for h in range(n_heads):
            if ci_arr_sel.std() < 1e-6:
                continue
            r, p = stats.pearsonr(ci_arr_sel, all_heads_arr[:, h])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            logger.info(f"    H{h}: r={r:+.4f} {sig}")

    # --- Plot 1: Top Q and K component correlations ---
    fig, (ax_q, ax_k) = plt.subplots(1, 2, figsize=(16, 6))

    top_n = 20
    q_top = q_corrs[:top_n]
    colors_q = ["tab:red" if c in semantic_q else "tab:blue" for c, _, _ in q_top]
    ax_q.barh(range(top_n), [r for _, r, _ in q_top], color=colors_q, alpha=0.7)
    ax_q.set_yticks(range(top_n))
    ax_q.set_yticklabels([f"Q{c}" for c, _, _ in q_top])
    ax_q.set_xlabel("Correlation with H4 induction attention")
    ax_q.set_title("Q components (red = semantic modulator)")
    ax_q.invert_yaxis()
    ax_q.axvline(0, color="black", linewidth=0.5)

    k_top = k_corrs[:top_n]
    colors_k = ["tab:red" if c in semantic_k else "tab:blue" for c, _, _ in k_top]
    ax_k.barh(range(top_n), [r for _, r, _ in k_top], color=colors_k, alpha=0.7)
    ax_k.set_yticks(range(top_n))
    ax_k.set_yticklabels([f"K{c}" for c, _, _ in k_top])
    ax_k.set_xlabel("Correlation with H4 induction attention")
    ax_k.set_title("K components (red = semantic modulator)")
    ax_k.invert_yaxis()
    ax_k.axvline(0, color="black", linewidth=0.5)

    fig.suptitle(
        f"Layer {LAYER}: Which component CIs predict stronger H4 induction?\n"
        f"({len(h4_induction_weights)} induction events, {N_SAMPLES} samples)",
        fontweight="bold",
    )
    fig.tight_layout()
    path = OUT_DIR / "induction_drivers_correlation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"\nSaved {path}")

    # --- Plot 2: Semantic components vs induction per head ---
    sem_comps = [
        ("Q335", 335, q_ci_arr),
        ("Q270", 270, q_ci_arr),
        ("K206", 206, k_ci_arr),
        ("K224", 224, k_ci_arr),
    ]
    fig, axes = plt.subplots(1, len(sem_comps), figsize=(5 * len(sem_comps), 4))

    for ax_idx, (label, comp_idx, ci_full) in enumerate(sem_comps):
        ax = axes[ax_idx]
        ci_vals = ci_full[:, comp_idx]
        corrs_per_head = []
        for h in range(n_heads):
            if ci_vals.std() < 1e-6:
                corrs_per_head.append(0.0)
            else:
                r, _ = stats.pearsonr(ci_vals, all_heads_arr[:, h])
                corrs_per_head.append(r)

        colors = ["tab:orange" if h == INDUCTION_HEAD else "tab:blue" for h in range(n_heads)]
        ax.bar(range(n_heads), corrs_per_head, color=colors)
        ax.set_xticks(range(n_heads))
        ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
        ax.set_ylabel("Corr with induction attention")
        ax.set_title(label, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.5)

    fig.suptitle(
        f"Layer {LAYER}: Semantic component CI correlation with per-head induction\n"
        f"(orange = H{INDUCTION_HEAD}, the induction head)",
        fontweight="bold",
    )
    fig.tight_layout()
    path = OUT_DIR / "induction_drivers_semantic_per_head.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


if __name__ == "__main__":
    main()
