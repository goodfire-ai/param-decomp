"""Analyze V and O projection components in Layer 2.

The Q/K analysis told us *where* the model looks. The V/O analysis tells us
*what information* gets transported. Key questions:

1. What are the moderate V/O components? Do they split into semantic/induction?
2. Do V component activations correlate with Q/K semantic component CIs?
3. What do the V components write to the residual stream when H0/H5 attend locally
   vs when H4 does induction?
4. Per-head V component norms — which heads' value outputs are modulated?
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from transformers import AutoTokenizer

from spd.configs import LMTaskConfig
from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import LinearComponents
from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP

OUT_DIR = Path(__file__).parent / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WANDB_PATH = "wandb:goodfire/spd/runs/s-55ea3f9b"
LAYER = 2
N_SAMPLES = 200


def main() -> None:
    run_info = SPDRunInfo.from_path(WANDB_PATH)
    model = ComponentModel.from_run_info(run_info)
    model.eval()
    config = run_info.config

    target_model = model.target_model
    assert isinstance(target_model, LlamaSimpleMLP)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    task_config = config.task_config
    assert isinstance(task_config, LMTaskConfig)
    seq_len = target_model.config.n_ctx

    assert config.tokenizer_name is not None
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
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

    v_path = f"h.{LAYER}.attn.v_proj"
    o_path = f"h.{LAYER}.attn.o_proj"
    q_path = f"h.{LAYER}.attn.q_proj"
    k_path = f"h.{LAYER}.attn.k_proj"

    v_comp = model.components[v_path]
    o_comp = model.components[o_path]
    assert isinstance(v_comp, LinearComponents)
    assert isinstance(o_comp, LinearComponents)

    n_v_comp = v_comp.U.shape[0]
    n_o_comp = o_comp.U.shape[0]

    logger.info(f"V components: {n_v_comp}, O components: {n_o_comp}")

    # === Step 1: CI survey for V and O ===
    v_ci_means: list[float] = []
    o_ci_means: list[float] = []

    # Also collect Q/K CIs for cross-correlation
    q335_ci_list: list[np.ndarray] = []  # per sample, mean CI
    q270_ci_list: list[np.ndarray] = []
    k206_ci_list: list[np.ndarray] = []

    # V component CI per-position for top components
    v_ci_per_pos: dict[int, list[float]] = {}

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= N_SAMPLES:
                break

            input_ids = batch[task_config.column_name][:, :seq_len].to(device)

            out = model(input_ids, cache_type="input")
            ci = model.calc_causal_importances(
                pre_weight_acts=out.cache, sampling="continuous", detach_inputs=True
            ).lower_leaky

            v_ci = ci[v_path][0]  # (T, C_v)
            o_ci = ci[o_path][0]  # (T, C_o)

            if i == 0:
                # First sample: compute mean CI for all components
                v_ci_means = v_ci.mean(dim=0).cpu().tolist()
                o_ci_means = o_ci.mean(dim=0).cpu().tolist()
            else:
                v_mean = v_ci.mean(dim=0).cpu().tolist()
                o_mean = o_ci.mean(dim=0).cpu().tolist()
                v_ci_means = [a + b for a, b in zip(v_ci_means, v_mean)]
                o_ci_means = [a + b for a, b in zip(o_ci_means, o_mean)]

            # Collect Q/K CIs
            q335_ci_list.append(ci[q_path][0, :, 335].cpu().numpy())
            q270_ci_list.append(ci[q_path][0, :, 270].cpu().numpy())
            k206_ci_list.append(ci[k_path][0, :, 206].cpu().numpy())

            # Collect V CI for moderate components (will identify after first pass)
            if i < 50:
                for c in range(n_v_comp):
                    if c not in v_ci_per_pos:
                        v_ci_per_pos[c] = []
                    v_ci_per_pos[c].append(v_ci[:, c].mean().item())

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i+1}/{N_SAMPLES}")

    # Normalize means
    v_ci_means = [x / N_SAMPLES for x in v_ci_means]
    o_ci_means = [x / N_SAMPLES for x in o_ci_means]

    # Categorize V components
    v_sorted = sorted(enumerate(v_ci_means), key=lambda x: x[1], reverse=True)
    o_sorted = sorted(enumerate(o_ci_means), key=lambda x: x[1], reverse=True)

    logger.info("\n=== V component CI survey ===")
    n_always_on_v = sum(1 for _, ci in v_sorted if ci > 0.8)
    n_moderate_v = sum(1 for _, ci in v_sorted if 0.1 <= ci <= 0.8)
    n_low_v = sum(1 for _, ci in v_sorted if 0.01 <= ci < 0.1)
    n_dead_v = sum(1 for _, ci in v_sorted if ci < 0.01)
    logger.info(f"  Always-on (>0.8): {n_always_on_v}")
    logger.info(f"  Moderate (0.1-0.8): {n_moderate_v}")
    logger.info(f"  Low (0.01-0.1): {n_low_v}")
    logger.info(f"  Dead (<0.01): {n_dead_v}")
    logger.info("\n  Top 20 V components:")
    for idx, (c, ci) in enumerate(v_sorted[:20]):
        logger.info(f"    V{c}: CI={ci:.3f}")

    logger.info("\n=== O component CI survey ===")
    n_always_on_o = sum(1 for _, ci in o_sorted if ci > 0.8)
    n_moderate_o = sum(1 for _, ci in o_sorted if 0.1 <= ci <= 0.8)
    n_low_o = sum(1 for _, ci in o_sorted if 0.01 <= ci < 0.1)
    n_dead_o = sum(1 for _, ci in o_sorted if ci < 0.01)
    logger.info(f"  Always-on (>0.8): {n_always_on_o}")
    logger.info(f"  Moderate (0.1-0.8): {n_moderate_o}")
    logger.info(f"  Low (0.01-0.1): {n_low_o}")
    logger.info(f"  Dead (<0.01): {n_dead_o}")
    logger.info("\n  Top 20 O components:")
    for idx, (c, ci) in enumerate(o_sorted[:20]):
        logger.info(f"    O{c}: CI={ci:.3f}")

    # === Step 2: Per-head V norms for top V components ===
    top_v = [c for c, _ in v_sorted[:15]]
    U_v = v_comp.U.detach().float()  # (C_v, n_kv_heads * head_dim)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for c in top_v[:8]:
        u = U_v[c].reshape(n_kv_heads, head_dim)
        norms = u.norm(dim=1).cpu().numpy()
        if n_kv_heads < n_heads:
            norms = np.repeat(norms, g)
        ci_val = v_ci_means[c]
        ax.plot(range(n_heads), norms, "o-", label=f"V{c} (CI={ci_val:.2f})", markersize=4)
    ax.set_xlabel("Head")
    ax.set_ylabel("||U_v|| per head")
    ax.set_title("Top V components: per-head norms", fontweight="bold")
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.legend(fontsize=6)

    # O component per-head norms
    # O_proj maps from (n_heads * head_dim) -> d_model
    # U_o shape: (C_o, d_model). But O reads from head-concatenated space,
    # so V_o (input side) has shape (d_in=n_heads*head_dim, C_o)
    # The per-head contribution comes from V_o, not U_o
    V_o = o_comp.V.detach().float()  # (d_in=n_heads*head_dim, C_o)
    top_o = [c for c, _ in o_sorted[:8]]

    ax = axes[1]
    for c in top_o:
        v = V_o[:, c].reshape(n_heads, head_dim)
        norms = v.norm(dim=1).cpu().numpy()
        ci_val = o_ci_means[c]
        ax.plot(range(n_heads), norms, "o-", label=f"O{c} (CI={ci_val:.2f})", markersize=4)
    ax.set_xlabel("Head")
    ax.set_ylabel("||V_o|| per head (input side)")
    ax.set_title("Top O components: per-head input norms", fontweight="bold")
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.legend(fontsize=6)

    fig.suptitle(f"Layer {LAYER}: V and O component per-head weight norms", fontweight="bold")
    fig.tight_layout()
    path = OUT_DIR / "value_component_norms.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"\nSaved {path}")

    # === Step 3: Cross-correlate V CI with Q/K semantic CIs ===
    # Need per-position V CIs for top V components correlated with Q335/Q270/K206
    # Re-collect with targeted components
    top_v_to_check = [c for c, _ in v_sorted[:10]]
    v_ci_collected: dict[int, list[float]] = {c: [] for c in top_v_to_check}
    qk_ci_flat: dict[str, list[float]] = {"Q335": [], "Q270": [], "K206": []}

    loader2, _ = create_data_loader(dataset_config=dataset_config, batch_size=1, buffer_size=1000)

    with torch.no_grad():
        for i, batch in enumerate(loader2):
            if i >= 100:
                break
            input_ids = batch[task_config.column_name][:, :seq_len].to(device)
            T = input_ids.shape[1]

            out = model(input_ids, cache_type="input")
            ci = model.calc_causal_importances(
                pre_weight_acts=out.cache, sampling="continuous", detach_inputs=True
            ).lower_leaky

            v_ci = ci[v_path][0]  # (T, C_v)
            q335 = ci[q_path][0, :, 335].cpu()
            q270 = ci[q_path][0, :, 270].cpu()
            k206 = ci[k_path][0, :, 206].cpu()

            for t in range(10, T):
                for c in top_v_to_check:
                    v_ci_collected[c].append(v_ci[t, c].item())
                qk_ci_flat["Q335"].append(q335[t].item())
                qk_ci_flat["Q270"].append(q270[t].item())
                qk_ci_flat["K206"].append(k206[t].item())

    # Correlation matrix: V CI vs Q/K CI
    logger.info("\n=== V-component CI vs Q/K semantic CI correlations ===")
    qk_names = ["Q335", "Q270", "K206"]
    corr_vqk = np.zeros((len(top_v_to_check), len(qk_names)))

    for vi, vc in enumerate(top_v_to_check):
        v_arr = np.array(v_ci_collected[vc])
        for qi, qn in enumerate(qk_names):
            qk_arr = np.array(qk_ci_flat[qn])
            r, _ = stats.pearsonr(v_arr, qk_arr)
            corr_vqk[vi, qi] = r
        logger.info(f"  V{vc} (CI={v_ci_means[vc]:.3f}): " +
                     ", ".join(f"{qn}={corr_vqk[vi,qi]:+.3f}" for qi, qn in enumerate(qk_names)))

    # Plot correlation matrix
    fig, ax = plt.subplots(figsize=(6, 8))
    vmax = max(0.1, np.abs(corr_vqk).max())
    im = ax.imshow(corr_vqk, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=0.6, label="Pearson r")
    ax.set_yticks(range(len(top_v_to_check)))
    ax.set_yticklabels([f"V{c} (CI={v_ci_means[c]:.2f})" for c in top_v_to_check])
    ax.set_xticks(range(len(qk_names)))
    ax.set_xticklabels(qk_names)
    for vi in range(len(top_v_to_check)):
        for qi in range(len(qk_names)):
            ax.text(qi, vi, f"{corr_vqk[vi, qi]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title(f"Layer {LAYER}: V component CI vs Q/K semantic component CI\n"
                 "Do V components co-activate with the semantic modulators?", fontweight="bold")
    fig.tight_layout()
    path = OUT_DIR / "value_vs_qk_correlation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")

    # === Step 4: CI distribution for V (for article figure) ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    v_ci_arr = np.array(v_ci_means)
    ax.hist(v_ci_arr, bins=50, color="tab:purple", alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Mean CI")
    ax.set_ylabel("Count")
    ax.set_title(f"V component CI distribution (n={n_v_comp})", fontweight="bold")
    ax.axvline(0.1, color="red", linestyle="--", linewidth=1, label="Moderate threshold")
    ax.legend()

    ax = axes[1]
    o_ci_arr = np.array(o_ci_means)
    ax.hist(o_ci_arr, bins=50, color="tab:brown", alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Mean CI")
    ax.set_ylabel("Count")
    ax.set_title(f"O component CI distribution (n={n_o_comp})", fontweight="bold")
    ax.axvline(0.1, color="red", linestyle="--", linewidth=1, label="Moderate threshold")
    ax.legend()

    fig.suptitle(f"Layer {LAYER}: Value and Output projection component CI distributions",
                 fontweight="bold")
    fig.tight_layout()
    path = OUT_DIR / "value_ci_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")

    # === Step 5: What do the top V components write? ===
    # U_v maps component acts -> value space (which attention then selects from)
    # The per-head direction of V component output tells us what information is available
    # We can check: does the model's V components have different "information types" per head?

    # Cosine similarity between top V component U vectors, per head
    n_top = min(10, len(top_v))
    cos_sim_per_head = np.zeros((n_top, n_top, n_kv_heads))
    for h in range(n_kv_heads):
        for i in range(n_top):
            for j in range(n_top):
                u_i = U_v[top_v[i], h * head_dim:(h + 1) * head_dim]
                u_j = U_v[top_v[j], h * head_dim:(h + 1) * head_dim]
                cos = torch.nn.functional.cosine_similarity(u_i.unsqueeze(0), u_j.unsqueeze(0)).item()
                cos_sim_per_head[i, j, h] = cos

    # Plot cosine similarity for a couple of KV heads
    fig, axes = plt.subplots(1, min(n_kv_heads, 4), figsize=(5 * min(n_kv_heads, 4), 4))
    if n_kv_heads == 1:
        axes = [axes]
    for h in range(min(n_kv_heads, 4)):
        ax = axes[h]
        im = ax.imshow(cos_sim_per_head[:, :, h], cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(n_top))
        ax.set_xticklabels([f"V{top_v[i]}" for i in range(n_top)], rotation=45, fontsize=6)
        ax.set_yticks(range(n_top))
        ax.set_yticklabels([f"V{top_v[i]}" for i in range(n_top)], fontsize=6)
        ax.set_title(f"KV Head {h}", fontweight="bold")
    fig.colorbar(im, ax=axes[-1], shrink=0.8, label="Cosine similarity")
    fig.suptitle(f"Layer {LAYER}: Cosine similarity of top V component U-vectors per KV head",
                 fontweight="bold")
    fig.tight_layout()
    path = OUT_DIR / "value_cosine_similarity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


if __name__ == "__main__":
    main()
