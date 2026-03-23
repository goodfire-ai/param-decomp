"""Find concrete exemplar datapoints showing induction attention shift.

Find two samples with repeated tokens where:
- Sample A: K206 is OFF at the induction key position → strong H4 induction
- Sample B: K206 is ON at the induction key position → weaker H4, stronger H0/H5 local

Show side-by-side attention patterns at the specific induction positions.
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
    k_path = f"h.{LAYER}.attn.k_proj"
    q_path = f"h.{LAYER}.attn.q_proj"

    # Collect induction events with metadata
    events: list[dict] = []

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

            k206_ci = ci[k_path][0, :, 206].cpu()
            q270_ci = ci[q_path][0, :, 270].cpu()
            q335_ci = ci[q_path][0, :, 335].cpu()

            results = collect_attention_patterns_with_logits(target_model, input_ids)
            attn = torch.softmax(results[LAYER][1], dim=-1)[0]  # (n_heads, T, T)

            token_list = input_ids[0].cpu().tolist()

            for t in range(4, T):
                for prev_t in range(t - 1):
                    if token_list[prev_t] == token_list[t] and prev_t + 1 < t:
                        induction_key = prev_t + 1
                        h4_weight = attn[INDUCTION_HEAD, t, induction_key].item()

                        if h4_weight < 0.01:
                            continue  # skip very weak induction

                        per_head = [attn[h, t, induction_key].item() for h in range(n_heads)]
                        # Also get attention to offset 0 (self/BOS proxy)
                        per_head_self = [attn[h, t, t].item() for h in range(n_heads)]

                        # Decode the repeated token and context
                        repeated_tok = tokenizer.decode([token_list[t]])  # pyright: ignore[reportAttributeAccessIssue]
                        context_start = max(0, t - 5)
                        context = tokenizer.decode(token_list[context_start:t + 1])  # pyright: ignore[reportAttributeAccessIssue]

                        events.append({
                            "sample_idx": i,
                            "query_pos": t,
                            "prev_pos": prev_t,
                            "induction_key": induction_key,
                            "offset": t - induction_key,
                            "repeated_token": repeated_tok,
                            "context": context,
                            "h4_induction": h4_weight,
                            "per_head_induction": np.array(per_head),
                            "per_head_self": np.array(per_head_self),
                            "k206_at_key": k206_ci[induction_key].item(),
                            "q270_at_query": q270_ci[t].item(),
                            "q335_at_query": q335_ci[t].item(),
                            "attn_row": attn[:, t, :].cpu().numpy(),  # (n_heads, T)
                            "T": T,
                        })
                        break  # first occurrence only

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i+1}/{N_SAMPLES}, events: {len(events)}")

    logger.info(f"\nTotal induction events (H4>0.01): {len(events)}")

    # Split by K206 at key
    k206_on = [e for e in events if e["k206_at_key"] > 0.5]
    k206_off = [e for e in events if e["k206_at_key"] < 0.5]

    logger.info(f"K206 ON at key: {len(k206_on)}, OFF: {len(k206_off)}")

    # Find best exemplar pair:
    # K206 OFF with high H4 induction, and K206 ON with lower H4 induction
    # Both should have similar offsets for fair comparison
    k206_off_sorted = sorted(k206_off, key=lambda e: e["h4_induction"], reverse=True)
    k206_on_sorted = sorted(k206_on, key=lambda e: e["h4_induction"])

    # Find pair with similar offset
    best_off = None
    best_on = None
    for e_off in k206_off_sorted[:50]:
        for e_on in k206_on_sorted[:50]:
            if abs(e_off["offset"] - e_on["offset"]) <= 3:
                best_off = e_off
                best_on = e_on
                break
        if best_off is not None:
            break

    if best_off is None or best_on is None:
        logger.warning("Could not find matching pair, using top events regardless")
        best_off = k206_off_sorted[0] if k206_off_sorted else None
        best_on = k206_on_sorted[0] if k206_on_sorted else None

    # --- Log exemplar details ---
    for label, e in [("K206 OFF (strong induction)", best_off), ("K206 ON (weak induction)", best_on)]:
        if e is None:
            continue
        logger.info(f"\n=== {label} ===")
        logger.info(f"  Sample {e['sample_idx']}, query pos {e['query_pos']}, "
                     f"prev pos {e['prev_pos']}, offset {e['offset']}")
        logger.info(f"  Repeated token: '{e['repeated_token']}'")
        logger.info(f"  Context: '{e['context']}'")
        logger.info(f"  K206 at key: {e['k206_at_key']:.3f}, Q270 at query: {e['q270_at_query']:.3f}, "
                     f"Q335 at query: {e['q335_at_query']:.3f}")
        logger.info("  Per-head induction attention:")
        for h in range(n_heads):
            logger.info(f"    H{h}: induction={e['per_head_induction'][h]:.4f}, "
                         f"self={e['per_head_self'][h]:.4f}")

    if best_off is None or best_on is None:
        return

    # --- Plot 1: Side-by-side per-head induction + self attention ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x = np.arange(n_heads)
    width = 0.35

    # Induction attention
    ax = axes[0, 0]
    ax.bar(x - width/2, best_off["per_head_induction"], width,
           label=f"K206 OFF (key CI={best_off['k206_at_key']:.2f})", color="tab:blue")
    ax.bar(x + width/2, best_on["per_head_induction"], width,
           label=f"K206 ON (key CI={best_on['k206_at_key']:.2f})", color="tab:red")
    ax.set_ylabel("Attention to induction target")
    ax.set_title("Induction attention per head", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.legend()

    # Self attention
    ax = axes[0, 1]
    ax.bar(x - width/2, best_off["per_head_self"], width, label="K206 OFF", color="tab:blue")
    ax.bar(x + width/2, best_on["per_head_self"], width, label="K206 ON", color="tab:red")
    ax.set_ylabel("Self-attention")
    ax.set_title("Self-attention per head", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.legend()

    # Attention distribution for the query row (all key positions)
    T_plot = min(40, best_off["T"], best_on["T"])
    for ax_idx, (e, label) in enumerate([(best_off, "K206 OFF"), (best_on, "K206 ON")]):
        ax = axes[1, ax_idx]
        for h in range(n_heads):
            ax.plot(range(T_plot), e["attn_row"][h, :T_plot],
                    label=f"H{h}", linewidth=1.5, alpha=0.7)
        # Mark induction target
        if e["induction_key"] < T_plot:
            ax.axvline(e["induction_key"], color="black", linestyle="--", linewidth=1, alpha=0.5)
            ax.text(e["induction_key"], ax.get_ylim()[1] * 0.9, "ind.target",
                    fontsize=7, ha="center")
        # Mark query position
        if e["query_pos"] < T_plot:
            ax.axvline(e["query_pos"], color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("Key position")
        ax.set_ylabel("Attention weight")
        ax.set_title(f"{label}: query at pos {e['query_pos']} ('{e['repeated_token'].strip()}')",
                      fontweight="bold")
        ax.legend(fontsize=6, ncol=2)

    fig.suptitle(
        f"Layer {LAYER}: Induction exemplar — K206 OFF (strong H4) vs K206 ON (weaker H4)\n"
        f"K206 OFF: '{best_off['context'][:60]}' | K206 ON: '{best_on['context'][:60]}'",
        fontweight="bold", fontsize=10,
    )
    fig.tight_layout()
    path = OUT_DIR / "induction_exemplar_k206.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"\nSaved {path}")

    # --- Plot 2: Distribution of H4 induction by K206 state ---
    h4_k206_on = [e["h4_induction"] for e in k206_on]
    h4_k206_off = [e["h4_induction"] for e in k206_off]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(h4_k206_off, bins=50, alpha=0.7, label=f"K206 OFF (n={len(k206_off)})",
             color="tab:blue", density=True)
    ax1.hist(h4_k206_on, bins=50, alpha=0.7, label=f"K206 ON (n={len(k206_on)})",
             color="tab:red", density=True)
    ax1.set_xlabel("H4 attention to induction target")
    ax1.set_ylabel("Density")
    ax1.set_title("H4 induction weight distribution")
    ax1.legend()

    # Fraction of events where H4 induction > threshold
    thresholds = np.arange(0, 0.5, 0.02)
    frac_off = [np.mean(np.array(h4_k206_off) > t) for t in thresholds]
    frac_on = [np.mean(np.array(h4_k206_on) > t) for t in thresholds]
    ax2.plot(thresholds, frac_off, "b-", linewidth=2, label="K206 OFF")
    ax2.plot(thresholds, frac_on, "r-", linewidth=2, label="K206 ON")
    ax2.set_xlabel("H4 induction threshold")
    ax2.set_ylabel("Fraction of events above threshold")
    ax2.set_title("Survival curve: H4 induction strength")
    ax2.legend()

    fig.suptitle(
        f"Layer {LAYER}: K206 at key position suppresses H4 induction",
        fontweight="bold",
    )
    fig.tight_layout()
    path = OUT_DIR / "induction_exemplar_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


if __name__ == "__main__":
    main()
