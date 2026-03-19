"""Decompose attention at positions where induction and previous-token compete.

Find positions where:
1. There's a repeated token (induction opportunity) at offset > 3
2. Previous-token attention is also strong

Show which component pairs drive the induction target vs the previous token,
and how they differ. Use the existing plot_qk_c_datapoint decomposition.
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
from spd.scripts.plot_qk_c_datapoint.plot_qk_c_datapoint import _compute_datapoint_contributions

OUT_DIR = Path(__file__).parent / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WANDB_PATH = "wandb:goodfire/spd/runs/s-55ea3f9b"
LAYER = 2
N_SAMPLES = 200
INDUCTION_HEAD = 4

SEMANTIC_Q = {335, 270, 279, 436, 66, 207, 261, 46}
SEMANTIC_K = {206, 224, 327, 167, 107, 320, 373, 1}


def categorize_pair(qi: int, ki: int) -> str:
    """Categorize a component pair."""
    q_sem = qi in SEMANTIC_Q
    k_sem = ki in SEMANTIC_K
    if q_sem and k_sem:
        return "semantic-semantic"
    elif q_sem:
        return "semantic_q"
    elif k_sem:
        return "semantic_k"
    else:
        return "non-semantic"


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

    n_heads = target_model._h[LAYER].attn.n_head

    # Find good candidate positions
    candidates: list[dict] = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= N_SAMPLES:
                break

            input_ids = batch[task_config.column_name][:, :seq_len].to(device)
            T = input_ids.shape[1]

            results = collect_attention_patterns_with_logits(target_model, input_ids)
            attn = torch.softmax(results[LAYER][1], dim=-1)[0]

            token_list = input_ids[0].cpu().tolist()

            for t in range(10, min(T, 60)):  # restrict to early positions for readability
                # Find induction target
                for prev_t in range(t - 3):  # offset > 3
                    if token_list[prev_t] == token_list[t] and prev_t + 1 < t:
                        ik = prev_t + 1
                        h4_ind = attn[INDUCTION_HEAD, t, ik].item()
                        h4_prev = attn[INDUCTION_HEAD, t, t - 1].item()
                        h0_self = attn[0, t, t].item()

                        if h4_ind > 0.05:  # decent induction
                            tok = tokenizer.decode([token_list[t]])  # pyright: ignore
                            ctx = tokenizer.decode(token_list[max(0, t-3):t+1])  # pyright: ignore
                            candidates.append({
                                "sample": i,
                                "query_pos": t,
                                "induction_key": ik,
                                "offset": t - ik,
                                "h4_induction": h4_ind,
                                "h4_prevtok": h4_prev,
                                "h0_self": h0_self,
                                "token": tok,
                                "context": ctx,
                                "input_ids": input_ids,
                                "T": T,
                            })
                        break

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i+1}/{N_SAMPLES}, candidates: {len(candidates)}")

    logger.info(f"\nTotal candidates: {len(candidates)}")

    # Pick best: high H4 induction, induction target well-separated from previous token
    candidates.sort(key=lambda c: c["h4_induction"], reverse=True)

    if not candidates:
        logger.warning("No candidates found")
        return

    best = candidates[0]
    logger.info(f"\nBest candidate:")
    logger.info(f"  Sample {best['sample']}, query pos {best['query_pos']}")
    logger.info(f"  Token: '{best['token']}', context: '{best['context']}'")
    logger.info(f"  Induction key at pos {best['induction_key']} (offset {best['offset']})")
    logger.info(f"  H4 induction: {best['h4_induction']:.3f}")
    logger.info(f"  H4 prev-tok: {best['h4_prevtok']:.3f}")

    # Run decomposition
    contributions, ground_truth = _compute_datapoint_contributions(
        model, target_model, best["input_ids"],
        best["query_pos"], LAYER, mode="weighted", ci_threshold=0.5,
    )
    # contributions: (n_q_heads, C_q, C_k, n_key_positions)
    # ground_truth: (n_q_heads, n_key_positions)

    ik = best["induction_key"]
    prev_pos = best["query_pos"] - 1

    logger.info(f"\n=== Decomposition at induction key (pos {ik}) vs previous token (pos {prev_pos}) ===")

    for h in [INDUCTION_HEAD, 0, 5]:
        logger.info(f"\n  Head {h}:")
        logger.info(f"    Ground truth logit at induction key: {ground_truth[h, ik]:.4f}")
        logger.info(f"    Ground truth logit at prev token: {ground_truth[h, prev_pos]:.4f}")

        # Top pairs at induction key
        c_at_ik = contributions[h, :, :, ik]  # (C_q, C_k)
        flat = c_at_ik.ravel()
        top_idx = np.argsort(np.abs(flat))[::-1][:10]
        logger.info(f"    Top 10 pairs at induction key:")
        for rank, idx in enumerate(top_idx):
            qi, ki = divmod(idx, c_at_ik.shape[1])
            val = c_at_ik[qi, ki]
            cat = categorize_pair(qi, ki)
            logger.info(f"      #{rank+1}: Q{qi}×K{ki} = {val:+.4f} [{cat}]")

        # Top pairs at prev token
        c_at_prev = contributions[h, :, :, prev_pos]
        flat_prev = c_at_prev.ravel()
        top_idx_prev = np.argsort(np.abs(flat_prev))[::-1][:10]
        logger.info(f"    Top 10 pairs at previous token:")
        for rank, idx in enumerate(top_idx_prev):
            qi, ki = divmod(idx, c_at_prev.shape[1])
            val = c_at_prev[qi, ki]
            cat = categorize_pair(qi, ki)
            logger.info(f"      #{rank+1}: Q{qi}×K{ki} = {val:+.4f} [{cat}]")

    # --- Plot: contribution profiles along key positions for top pairs ---
    n_key = best["query_pos"] + 1
    tokens = [tokenizer.decode([best["input_ids"][0, t].item()]) for t in range(n_key)]  # pyright: ignore

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, h in enumerate([INDUCTION_HEAD, 0, 5]):
        # At induction key: top pairs
        c_at_ik = contributions[h, :, :, ik]
        flat = c_at_ik.ravel()
        top_pairs_ind = [divmod(idx, c_at_ik.shape[1]) for idx in np.argsort(np.abs(flat))[::-1][:5]]

        # At prev token: top pairs
        c_at_prev = contributions[h, :, :, prev_pos]
        flat_prev = c_at_prev.ravel()
        top_pairs_prev = [divmod(idx, c_at_prev.shape[1]) for idx in np.argsort(np.abs(flat_prev))[::-1][:5]]

        # Plot: top induction pairs
        ax = axes[0, col]
        for qi, ki in top_pairs_ind:
            profile = contributions[h, qi, ki, :n_key]
            cat = categorize_pair(qi, ki)
            ls = "--" if "semantic" in cat else "-"
            ax.plot(range(n_key), profile, ls, linewidth=1.5,
                    label=f"Q{qi}×K{ki}", alpha=0.8)
        # Mark induction key and prev token
        ax.axvline(ik, color="red", linestyle=":", linewidth=2, alpha=0.7, label="ind. target")
        ax.axvline(prev_pos, color="blue", linestyle=":", linewidth=2, alpha=0.7, label="prev tok")
        ax.set_title(f"H{h}: Top pairs at induction key", fontweight="bold")
        ax.set_ylabel("Contribution to logit")
        ax.legend(fontsize=6, ncol=2)
        if n_key <= 30:
            ax.set_xticks(range(n_key))
            ax.set_xticklabels(tokens, rotation=90, fontsize=5)

        # Plot: top prev-token pairs
        ax = axes[1, col]
        for qi, ki in top_pairs_prev:
            profile = contributions[h, qi, ki, :n_key]
            cat = categorize_pair(qi, ki)
            ls = "--" if "semantic" in cat else "-"
            ax.plot(range(n_key), profile, ls, linewidth=1.5,
                    label=f"Q{qi}×K{ki}", alpha=0.8)
        ax.axvline(ik, color="red", linestyle=":", linewidth=2, alpha=0.7)
        ax.axvline(prev_pos, color="blue", linestyle=":", linewidth=2, alpha=0.7)
        ax.set_title(f"H{h}: Top pairs at prev token", fontweight="bold")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Contribution to logit")
        ax.legend(fontsize=6, ncol=2)

    fig.suptitle(
        f"Layer {LAYER}: QK component decomposition — induction target vs previous token\n"
        f"Sample {best['sample']}, query '{best['token']}' at pos {best['query_pos']}, "
        f"induction target at pos {ik} (offset {best['offset']})",
        fontweight="bold",
    )
    fig.tight_layout()
    path = OUT_DIR / "decompose_induction_vs_prevtok.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"\nSaved {path}")

    # --- Plot 2: Stacked category contributions ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    categories = ["semantic-semantic", "semantic_q", "semantic_k", "non-semantic"]
    cat_colors = {"semantic-semantic": "tab:red", "semantic_q": "tab:orange",
                  "semantic_k": "tab:green", "non-semantic": "tab:blue"}

    for col, h in enumerate([INDUCTION_HEAD, 0, 5]):
        ax = axes[col]

        # Sum contributions by category at induction key vs prev token
        cat_vals_ind = {cat: 0.0 for cat in categories}
        cat_vals_prev = {cat: 0.0 for cat in categories}

        C_q, C_k = contributions.shape[1], contributions.shape[2]
        for qi in range(C_q):
            for ki in range(C_k):
                val_ind = contributions[h, qi, ki, ik]
                val_prev = contributions[h, qi, ki, prev_pos]
                if abs(val_ind) < 1e-6 and abs(val_prev) < 1e-6:
                    continue
                cat = categorize_pair(qi, ki)
                cat_vals_ind[cat] += val_ind
                cat_vals_prev[cat] += val_prev

        x = np.arange(2)
        bottom_pos = np.zeros(2)
        bottom_neg = np.zeros(2)

        for cat in categories:
            vals = np.array([cat_vals_ind[cat], cat_vals_prev[cat]])
            pos = np.maximum(vals, 0)
            neg = np.minimum(vals, 0)
            ax.bar(x, pos, bottom=bottom_pos, width=0.6, label=cat,
                   color=cat_colors[cat], alpha=0.7)
            ax.bar(x, neg, bottom=bottom_neg, width=0.6,
                   color=cat_colors[cat], alpha=0.7)
            bottom_pos += pos
            bottom_neg += neg

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["Induction\ntarget", "Previous\ntoken"])
        ax.set_ylabel("Sum of contributions")
        ax.set_title(f"H{h}", fontweight="bold")
        if col == 0:
            ax.legend(fontsize=7)

    fig.suptitle(
        "Component pair contributions by category: induction target vs previous token",
        fontweight="bold",
    )
    fig.tight_layout()
    path = OUT_DIR / "decompose_category_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


if __name__ == "__main__":
    main()
