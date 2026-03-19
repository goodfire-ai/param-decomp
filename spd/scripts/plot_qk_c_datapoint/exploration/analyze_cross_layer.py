"""Analyze cross-layer interaction: L1 previous-token → L2 semantic routing.

Layer 1 has always-on Q316/K329 implementing previous-token attention.
Layer 2 has semantic modulators (Q335, Q270, K206) that route attention.

Questions:
1. Does L2's semantic component CI depend on what L1 wrote to the residual stream?
   (Correlate L1 head output norms with L2 component CIs across positions)
2. Do L1 and L2 attention patterns interact? (When L2 Q270 broadens attention,
   does L1 previous-token set up the key positions that L2 attends to?)
3. How do L1 and L2 offset profiles combine across the two layers?
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
N_SAMPLES = 300
MAX_OFFSET = 16

# Layer 1 always-on components
L1_Q_COMP = 316
L1_K_COMP = 329

# Layer 2 semantic components
L2_SEMANTIC_Q = [335, 270, 279, 436]
L2_SEMANTIC_K = [206, 224]


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

    n_heads = target_model._h[0].attn.n_head

    l1_q_path = "h.1.attn.q_proj"
    l1_k_path = "h.1.attn.k_proj"
    l2_q_path = "h.2.attn.q_proj"
    l2_k_path = "h.2.attn.k_proj"

    # Per-position data collection
    # For each position: L1 component CIs, L2 component CIs, L1/L2 attention offsets
    l1_q316_ci: list[float] = []
    l1_k329_ci: list[float] = []
    l2_q_cis: dict[int, list[float]] = {c: [] for c in L2_SEMANTIC_Q}
    l2_k_cis: dict[int, list[float]] = {c: [] for c in L2_SEMANTIC_K}

    # Per-head offset profiles conditioned on L2 Q270 activation
    # We'll split into tertiles of Q270 CI
    l2_q270_ci_all: list[float] = []
    l1_offset_per_pos: list[np.ndarray] = []  # (n_heads, MAX_OFFSET)
    l2_offset_per_pos: list[np.ndarray] = []  # (n_heads, MAX_OFFSET)

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

            results = collect_attention_patterns_with_logits(target_model, input_ids)
            l1_attn = torch.softmax(results[1][1], dim=-1)[0]  # (n_heads, T, T)
            l2_attn = torch.softmax(results[2][1], dim=-1)[0]  # (n_heads, T, T)

            l1_q316 = ci[l1_q_path][0, :, L1_Q_COMP].cpu()  # (T,)
            l1_k329 = ci[l1_k_path][0, :, L1_K_COMP].cpu()  # (T,)

            l2_q_ci_batch = {c: ci[l2_q_path][0, :, c].cpu() for c in L2_SEMANTIC_Q}
            l2_k_ci_batch = {c: ci[l2_k_path][0, :, c].cpu() for c in L2_SEMANTIC_K}

            for t in range(MAX_OFFSET, T):
                l1_q316_ci.append(l1_q316[t].item())
                l1_k329_ci.append(l1_k329[t].item())

                for c in L2_SEMANTIC_Q:
                    l2_q_cis[c].append(l2_q_ci_batch[c][t].item())
                for c in L2_SEMANTIC_K:
                    l2_k_cis[c].append(l2_k_ci_batch[c][t].item())

                l2_q270_ci_all.append(l2_q_ci_batch[270][t].item())

                l1_off = np.zeros((n_heads, MAX_OFFSET))
                l2_off = np.zeros((n_heads, MAX_OFFSET))
                for h in range(n_heads):
                    for d in range(MAX_OFFSET):
                        l1_off[h, d] = l1_attn[h, t, t - d].item()
                        l2_off[h, d] = l2_attn[h, t, t - d].item()
                l1_offset_per_pos.append(l1_off)
                l2_offset_per_pos.append(l2_off)

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i+1}/{N_SAMPLES}, positions: {len(l1_q316_ci)}")

    N = len(l1_q316_ci)
    logger.info(f"\nTotal positions: {N}")

    l1_q316_arr = np.array(l1_q316_ci)
    l1_k329_arr = np.array(l1_k329_ci)
    l2_q270_arr = np.array(l2_q270_ci_all)

    # --- Analysis 1: Correlation between L1 CIs and L2 CIs ---
    logger.info("\n=== L1 Q316 CI correlation with L2 semantic component CIs ===")
    for c in L2_SEMANTIC_Q:
        arr = np.array(l2_q_cis[c])
        r, p = stats.pearsonr(l1_q316_arr, arr)
        logger.info(f"  L1.Q316 vs L2.Q{c}: r={r:+.4f}, p={p:.2e}")
    for c in L2_SEMANTIC_K:
        arr = np.array(l2_k_cis[c])
        r, p = stats.pearsonr(l1_q316_arr, arr)
        logger.info(f"  L1.Q316 vs L2.K{c}: r={r:+.4f}, p={p:.2e}")

    logger.info("\n=== L1 K329 CI correlation with L2 semantic component CIs ===")
    for c in L2_SEMANTIC_Q:
        arr = np.array(l2_q_cis[c])
        r, p = stats.pearsonr(l1_k329_arr, arr)
        logger.info(f"  L1.K329 vs L2.Q{c}: r={r:+.4f}, p={p:.2e}")
    for c in L2_SEMANTIC_K:
        arr = np.array(l2_k_cis[c])
        r, p = stats.pearsonr(l1_k329_arr, arr)
        logger.info(f"  L1.K329 vs L2.K{c}: r={r:+.4f}, p={p:.2e}")

    # --- Plot 1: L1-L2 CI correlation matrix ---
    l1_labels = ["L1.Q316", "L1.K329"]
    l2_labels = [f"L2.Q{c}" for c in L2_SEMANTIC_Q] + [f"L2.K{c}" for c in L2_SEMANTIC_K]
    l1_arrs = [l1_q316_arr, l1_k329_arr]
    l2_arrs = [np.array(l2_q_cis[c]) for c in L2_SEMANTIC_Q] + [
        np.array(l2_k_cis[c]) for c in L2_SEMANTIC_K
    ]

    corr_mat = np.zeros((len(l1_labels), len(l2_labels)))
    for i_l1, a1 in enumerate(l1_arrs):
        for i_l2, a2 in enumerate(l2_arrs):
            r, _ = stats.pearsonr(a1, a2)
            corr_mat[i_l1, i_l2] = r

    fig, ax = plt.subplots(figsize=(10, 3))
    vmax = max(0.1, np.abs(corr_mat).max())
    im = ax.imshow(corr_mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_yticks(range(len(l1_labels)))
    ax.set_yticklabels(l1_labels)
    ax.set_xticks(range(len(l2_labels)))
    ax.set_xticklabels(l2_labels, rotation=30, ha="right")
    for i_l1 in range(len(l1_labels)):
        for i_l2 in range(len(l2_labels)):
            ax.text(i_l2, i_l1, f"{corr_mat[i_l1, i_l2]:.3f}", ha="center", va="center",
                    fontsize=9, fontweight="bold")
    ax.set_title("Cross-layer CI correlation: L1 always-on → L2 semantic components")
    fig.tight_layout()
    path = OUT_DIR / "cross_layer_ci_correlation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"\nSaved {path}")

    # --- Analysis 2: L1 and L2 offset profiles conditioned on L2 Q270 ---
    l1_offsets = np.array(l1_offset_per_pos)  # (N, n_heads, MAX_OFFSET)
    l2_offsets = np.array(l2_offset_per_pos)  # (N, n_heads, MAX_OFFSET)

    # CI values are bimodal (0 or 1 at position level), so use binary split
    low_mask = l2_q270_arr < 0.5  # Q270 OFF
    high_mask = l2_q270_arr > 0.5  # Q270 ON
    # No mid bucket needed for bimodal distribution

    logger.info(f"\nQ270 binary split: OFF(<0.5)={low_mask.sum()}, ON(>0.5)={high_mask.sum()}")

    # Plot 2: L1 and L2 offset profiles by Q270 tertile
    fig, axes = plt.subplots(2, n_heads, figsize=(4 * n_heads, 7))

    for h in range(n_heads):
        # L1
        ax = axes[0, h]
        for mask, label, color, ls in [
            (low_mask, f"Q270 OFF (n={low_mask.sum()})", "tab:blue", "-"),
            (high_mask, f"Q270 ON (n={high_mask.sum()})", "tab:red", "-"),
        ]:
            if mask.sum() > 0:
                profile = l1_offsets[mask, h, :].mean(axis=0)
                ax.plot(range(MAX_OFFSET), profile, color=color, linestyle=ls, linewidth=1.5,
                        label=label)
        ax.set_title(f"H{h}", fontweight="bold")
        if h == 0:
            ax.set_ylabel("L1 attention")
            ax.legend(fontsize=6)

        # L2
        ax = axes[1, h]
        for mask, label, color, ls in [
            (low_mask, "Q270 OFF", "tab:blue", "-"),
            (high_mask, "Q270 ON", "tab:red", "-"),
        ]:
            if mask.sum() > 0:
                profile = l2_offsets[mask, h, :].mean(axis=0)
                ax.plot(range(MAX_OFFSET), profile, color=color, linestyle=ls, linewidth=1.5,
                        label=label)
        if h == 0:
            ax.set_ylabel("L2 attention")
        ax.set_xlabel("Offset")

    fig.suptitle(
        "L1 vs L2 attention offset profiles, conditioned on L2 Q270 CI tertile\n"
        "(Does L2 semantic routing also affect L1 patterns? It shouldn't.)",
        fontweight="bold",
    )
    fig.tight_layout()
    path = OUT_DIR / "cross_layer_offset_by_q270.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")

    # --- Analysis 3: L1 attention at offset=1 correlation with L2 semantic CIs ---
    # Does L1's previous-token attention strength predict L2 component activation?
    logger.info("\n=== L1 H-specific offset=1 attention vs L2 semantic CIs ===")

    fig, axes = plt.subplots(len(L2_SEMANTIC_Q) + len(L2_SEMANTIC_K), n_heads,
                             figsize=(2.5 * n_heads, 2.5 * (len(L2_SEMANTIC_Q) + len(L2_SEMANTIC_K))))

    all_l2_comps = [(c, "Q", np.array(l2_q_cis[c])) for c in L2_SEMANTIC_Q] + [
        (c, "K", np.array(l2_k_cis[c])) for c in L2_SEMANTIC_K
    ]

    for row_idx, (c, proj, c_arr) in enumerate(all_l2_comps):
        label = f"L2.{proj}{c}"
        for h in range(n_heads):
            l1_off1 = l1_offsets[:, h, 1]  # L1 head h attention to previous token
            r, p = stats.pearsonr(l1_off1, c_arr)
            ax = axes[row_idx, h]
            ax.scatter(l1_off1[::50], c_arr[::50], alpha=0.15, s=3, c="tab:blue")
            ax.set_title(f"r={r:+.3f}", fontsize=8)
            if h == 0:
                ax.set_ylabel(label, fontsize=8)
            if row_idx == len(all_l2_comps) - 1:
                ax.set_xlabel(f"L1 H{h} off=1", fontsize=7)

        logger.info(f"  {label}:")
        for h in range(n_heads):
            l1_off1 = l1_offsets[:, h, 1]
            r, p = stats.pearsonr(l1_off1, c_arr)
            sig = "***" if p < 0.001 else ""
            logger.info(f"    L1.H{h} offset=1: r={r:+.4f} {sig}")

    fig.suptitle(
        "L1 per-head previous-token attention vs L2 semantic component CI\n"
        "(Does L1's attention pattern influence L2 component activation?)",
        fontweight="bold", fontsize=10,
    )
    fig.tight_layout()
    path = OUT_DIR / "cross_layer_l1_attn_vs_l2_ci.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")

    # --- Analysis 4: Combined L1+L2 effective attention ---
    # The residual stream means L2 output ADD to L1 output. But attention composition
    # (what L2 queries "see" via L1's writing) is the more interesting interaction.
    # Let's look at L1 offset profiles conditioned on L2 Q335 vs Q270 dominance.

    l2_q335_arr = np.array(l2_q_cis[335])

    # CI values are bimodal (0 or 1), so use binary thresholds
    q270_dominant = l2_q270_arr > 0.5  # Q270 is ON
    q335_only = (l2_q270_arr < 0.5) & (l2_q335_arr > 0.5)  # Q335 ON, Q270 OFF

    logger.info(f"\nRegime split: Q270 active(>0.5)={q270_dominant.sum()}, "
                f"Q335-only(Q270<0.5, Q335>0.5)={q335_only.sum()}")

    fig, axes = plt.subplots(2, n_heads, figsize=(4 * n_heads, 7))

    for h in range(n_heads):
        # L1
        ax = axes[0, h]
        p_270 = l1_offsets[q270_dominant, h, :].mean(axis=0)
        p_335 = l1_offsets[q335_only, h, :].mean(axis=0)
        ax.plot(range(MAX_OFFSET), p_270, "b-", linewidth=2, label="Q270 active (research)")
        ax.plot(range(MAX_OFFSET), p_335, "r--", linewidth=2, label="Q335-only (conversational)")
        ax.set_title(f"H{h}", fontweight="bold")
        if h == 0:
            ax.set_ylabel("L1 attention")
            ax.legend(fontsize=7)

        # L2
        ax = axes[1, h]
        p_270 = l2_offsets[q270_dominant, h, :].mean(axis=0)
        p_335 = l2_offsets[q335_only, h, :].mean(axis=0)
        ax.plot(range(MAX_OFFSET), p_270, "b-", linewidth=2, label="Q270 active (research)")
        ax.plot(range(MAX_OFFSET), p_335, "r--", linewidth=2, label="Q335-only (conversational)")
        if h == 0:
            ax.set_ylabel("L2 attention")
        ax.set_xlabel("Offset")

    fig.suptitle(
        "L1 vs L2 offset profiles: Q270 active (research) vs Q335-only (conversational)\n"
        "(L1 should be stable; L2 should show the semantic routing difference)",
        fontweight="bold",
    )
    fig.tight_layout()
    path = OUT_DIR / "cross_layer_regime_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


if __name__ == "__main__":
    main()
