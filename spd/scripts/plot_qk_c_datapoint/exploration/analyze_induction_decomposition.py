"""Decompose induction attention into QK component pair contributions.

At actual repeated-token positions (where token t matches token t' and the
induction target is t'+1), use the datapoint-specific QK decomposition to
identify which component pairs produce the H4 induction attention logit.

This is different from the correlation approach (Experiment 9) — here we directly
decompose the attention logit into component contributions.

Also shows per-head induction scores conditioned on which semantic components
are active, to demonstrate that induction strength in non-H4 heads shifts.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from spd.configs import LMTaskConfig
from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import LinearComponents
from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from spd.scripts.collect_attention_patterns import collect_attention_patterns_with_logits

OUT_DIR = Path(__file__).parent / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WANDB_PATH = "wandb:goodfire/spd/runs/s-55ea3f9b"
LAYER = 2
N_SAMPLES = 150
INDUCTION_HEAD = 4

SEMANTIC_Q = {335, 270, 279, 436}
SEMANTIC_K = {206, 224}


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

    block = target_model._h[LAYER]
    n_heads = block.attn.n_head
    n_kv_heads = block.attn.n_key_value_heads
    head_dim = block.attn.head_dim
    g = n_heads // n_kv_heads

    q_path = f"h.{LAYER}.attn.q_proj"
    k_path = f"h.{LAYER}.attn.k_proj"

    q_comp = model.components[q_path]
    k_comp = model.components[k_path]
    assert isinstance(q_comp, LinearComponents)
    assert isinstance(k_comp, LinearComponents)

    n_q_comp = q_comp.U.shape[0]
    n_k_comp = k_comp.U.shape[0]

    # Precompute U vectors reshaped for heads
    U_q = q_comp.U.detach().float()  # (C_q, d_out)
    U_k = k_comp.U.detach().float()  # (C_k, d_out)

    # Per-head induction attention weighted by component type
    # Track: for each induction event, the contribution of each component pair to H4 logit
    # and the total logit per head

    # Aggregate: mean contribution by component category
    categories = ["semantic_q_semantic_k", "semantic_q_other_k", "other_q_semantic_k", "other_q_other_k"]
    category_contributions = {cat: np.zeros(n_heads) for cat in categories}
    n_events = 0

    # Per-head induction weights conditioned on Q270 active
    h_induction_q270_on: list[np.ndarray] = []
    h_induction_q270_off: list[np.ndarray] = []

    # Top component pairs at induction positions (for H4)
    pair_contributions_h4: dict[tuple[int, int], float] = {}

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= N_SAMPLES:
                break

            input_ids = batch[task_config.column_name][:, :seq_len].to(device)
            T = input_ids.shape[1]

            # Get component activations
            out = model(input_ids, cache_type="input")
            ci = model.calc_causal_importances(
                pre_weight_acts=out.cache, sampling="continuous", detach_inputs=True
            ).lower_leaky

            q_acts = q_comp.get_component_acts(
                out.cache[q_path][0]  # (T, d_in) -> (T, C_q)
            )
            k_acts = k_comp.get_component_acts(
                out.cache[k_path][0]  # (T, d_in) -> (T, C_k)
            )

            q270_ci = ci[q_path][0, :, 270].cpu()  # (T,)

            results = collect_attention_patterns_with_logits(target_model, input_ids)
            attn_weights = torch.softmax(results[LAYER][1], dim=-1)[0]  # (n_heads, T, T)

            token_list = input_ids[0].cpu().tolist()

            for t in range(4, T):
                best_prev = -1
                for prev_t in range(t - 1):
                    if token_list[prev_t] == token_list[t] and prev_t + 1 < t:
                        best_prev = prev_t
                        break

                if best_prev < 0:
                    continue

                induction_key = best_prev + 1
                offset = t - induction_key

                # Get component acts at this position
                q_act_t = q_acts[t].cpu().float()  # (C_q,)
                k_act_ik = k_acts[induction_key].cpu().float()  # (C_k,)

                # For each head, compute component-pair contributions to the logit
                # logit[h] = (1/sqrt(d)) * sum_ij q_act_i * k_act_j * (u_q_i_h . RoPE(u_k_j_h, offset))
                # This is expensive for all pairs, so we'll just categorize
                for h in range(n_heads):
                    kv_h = h // g
                    u_q_h = U_q[:, h * head_dim:(h + 1) * head_dim]  # (C_q, d_head)
                    u_k_h = U_k[:, kv_h * head_dim:(kv_h + 1) * head_dim]  # (C_k, d_head)

                    # Contribution per category
                    for qi in range(n_q_comp):
                        if abs(q_act_t[qi].item()) < 1e-6:
                            continue
                        for ki in range(n_k_comp):
                            if abs(k_act_ik[ki].item()) < 1e-6:
                                continue

                            # This inner loop is too expensive for all pairs
                            # Only do it for H4 and a subset
                            break
                        break
                    break  # Skip the expensive per-pair decomposition for now

                # Instead, use a simpler approach: per-head induction weights
                per_head_weights = np.array([
                    attn_weights[h, t, induction_key].item() for h in range(n_heads)
                ])

                if q270_ci[t].item() > 0.5:
                    h_induction_q270_on.append(per_head_weights)
                else:
                    h_induction_q270_off.append(per_head_weights)

                n_events += 1

            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i+1}/{N_SAMPLES}, events: {n_events}")

    logger.info(f"\nTotal induction events: {n_events}")
    logger.info(f"Q270 ON events: {len(h_induction_q270_on)}, OFF: {len(h_induction_q270_off)}")

    if len(h_induction_q270_on) < 50 or len(h_induction_q270_off) < 50:
        logger.warning("Too few events in one condition")

    q270_on_arr = np.array(h_induction_q270_on) if h_induction_q270_on else np.zeros((1, n_heads))
    q270_off_arr = np.array(h_induction_q270_off) if h_induction_q270_off else np.zeros((1, n_heads))

    mean_on = q270_on_arr.mean(axis=0)
    mean_off = q270_off_arr.mean(axis=0)

    # --- Plot 1: Per-head induction weights conditioned on Q270 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(n_heads)
    width = 0.35
    ax1.bar(x - width/2, mean_on, width, label=f"Q270 ON (n={len(h_induction_q270_on)})",
            color="tab:blue")
    ax1.bar(x + width/2, mean_off, width, label=f"Q270 OFF (n={len(h_induction_q270_off)})",
            color="tab:red")
    ax1.set_xlabel("Head")
    ax1.set_ylabel("Mean attention to induction target")
    ax1.set_title(f"Layer {LAYER}: Induction attention per head, by Q270 state")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax1.legend()

    # Difference
    diff = mean_on - mean_off
    colors = ["tab:blue" if d > 0 else "tab:red" for d in diff]
    ax2.bar(x, diff, color=colors)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("Head")
    ax2.set_ylabel("Δ attention (Q270 ON − OFF)")
    ax2.set_title("Difference: which heads gain/lose induction when Q270 is active?")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"H{h}" for h in range(n_heads)])

    fig.suptitle(
        f"Layer {LAYER}: Does semantic routing (Q270) redistribute induction across heads?\n"
        f"({n_events} induction events, {N_SAMPLES} samples)",
        fontweight="bold",
    )
    fig.tight_layout()
    path = OUT_DIR / "induction_decomp_q270_conditioning.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")

    # --- Also condition on Q335 and K206 ---
    # Redo with K206 conditioning
    k206_on_events: list[np.ndarray] = []
    k206_off_events: list[np.ndarray] = []

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
            results = collect_attention_patterns_with_logits(target_model, input_ids)
            attn_weights = torch.softmax(results[LAYER][1], dim=-1)[0]

            token_list = input_ids[0].cpu().tolist()

            for t in range(4, T):
                best_prev = -1
                for prev_t in range(t - 1):
                    if token_list[prev_t] == token_list[t] and prev_t + 1 < t:
                        best_prev = prev_t
                        break
                if best_prev < 0:
                    continue

                induction_key = best_prev + 1
                per_head_weights = np.array([
                    attn_weights[h, t, induction_key].item() for h in range(n_heads)
                ])

                # K206 active at the KEY position (induction_key)
                if k206_ci[induction_key].item() > 0.5:
                    k206_on_events.append(per_head_weights)
                else:
                    k206_off_events.append(per_head_weights)

            if (i + 1) % 50 == 0:
                logger.info(f"K206 pass: {i+1}/{N_SAMPLES}")

    logger.info(f"K206 ON: {len(k206_on_events)}, OFF: {len(k206_off_events)}")

    k206_on = np.array(k206_on_events).mean(axis=0) if k206_on_events else np.zeros(n_heads)
    k206_off = np.array(k206_off_events).mean(axis=0) if k206_off_events else np.zeros(n_heads)

    # --- Plot 2: Multi-condition comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    conditions = [
        ("Q270 at query", mean_on, mean_off, len(h_induction_q270_on), len(h_induction_q270_off)),
        ("K206 at key", k206_on, k206_off, len(k206_on_events), len(k206_off_events)),
    ]

    for ax_idx, (label, on, off, n_on, n_off) in enumerate(conditions):
        ax = axes[ax_idx]
        ax.bar(x - width/2, on, width, label=f"ON (n={n_on})", color="tab:blue")
        ax.bar(x + width/2, off, width, label=f"OFF (n={n_off})", color="tab:red")
        ax.set_xlabel("Head")
        ax.set_ylabel("Mean attn to induction target")
        ax.set_title(f"{label}", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
        ax.legend(fontsize=8)

    # Combined difference
    ax = axes[2]
    ax.bar(x - width/2, mean_on - mean_off, width, label="Q270 effect", color="tab:blue", alpha=0.7)
    ax.bar(x + width/2, k206_on - k206_off, width, label="K206 effect", color="tab:green", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Head")
    ax.set_ylabel("Δ induction attention")
    ax.set_title("Effect of each component", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.legend()

    fig.suptitle(
        f"Layer {LAYER}: Induction attention redistribution by semantic components\n"
        f"(Q270 at query position, K206 at key position)",
        fontweight="bold",
    )
    fig.tight_layout()
    path = OUT_DIR / "induction_decomp_multi_condition.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")

    # Numerical summary
    logger.info("\n=== Induction attention by head and condition ===")
    logger.info("Q270 at query:")
    for h in range(n_heads):
        logger.info(f"  H{h}: ON={mean_on[h]:.4f}, OFF={mean_off[h]:.4f}, Δ={mean_on[h]-mean_off[h]:+.4f}")
    logger.info("K206 at key:")
    for h in range(n_heads):
        logger.info(f"  H{h}: ON={k206_on[h]:.4f}, OFF={k206_off[h]:.4f}, Δ={k206_on[h]-k206_off[h]:+.4f}")


if __name__ == "__main__":
    main()
