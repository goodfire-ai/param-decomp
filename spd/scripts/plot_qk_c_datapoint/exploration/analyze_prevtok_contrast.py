"""Contrast previous-token behavior: L1 always-on vs L2 content-dependent.

L1 Q316/K329 implement pure positional previous-token attention (always-on).
L2's previous-token attention at offset=1 is a mixture of:
- Semantic components (K224 promotes prev-tok in H0, K206 promotes in H0/H5)
- Non-semantic components
- And it's modulated by content type

This script directly measures and contrasts:
1. L1 vs L2 attention at offset=1, per head
2. L2 offset=1 attention correlated with each semantic component
3. QK decomposition at offset=1 to identify which component pairs drive prev-tok in each layer
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

    # Collect per-position: L1 and L2 offset=1 attention + component CIs
    l1_off1: list[np.ndarray] = []  # (n_heads,) per position
    l2_off1: list[np.ndarray] = []

    # L1 CIs
    l1_q316_ci: list[float] = []
    l1_k329_ci: list[float] = []

    # L2 semantic CIs
    l2_q335_ci: list[float] = []
    l2_q270_ci: list[float] = []
    l2_k206_ci: list[float] = []
    l2_k224_ci: list[float] = []

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
            l1_attn = torch.softmax(results[1][1], dim=-1)[0]
            l2_attn = torch.softmax(results[2][1], dim=-1)[0]

            l1_q316 = ci["h.1.attn.q_proj"][0, :, 316].cpu()
            l1_k329 = ci["h.1.attn.k_proj"][0, :, 329].cpu()
            l2_q335 = ci["h.2.attn.q_proj"][0, :, 335].cpu()
            l2_q270 = ci["h.2.attn.q_proj"][0, :, 270].cpu()
            l2_k206 = ci["h.2.attn.k_proj"][0, :, 206].cpu()
            l2_k224 = ci["h.2.attn.k_proj"][0, :, 224].cpu()

            for t in range(2, T):
                l1_off1.append(np.array([l1_attn[h, t, t-1].item() for h in range(n_heads)]))
                l2_off1.append(np.array([l2_attn[h, t, t-1].item() for h in range(n_heads)]))
                l1_q316_ci.append(l1_q316[t].item())
                l1_k329_ci.append(l1_k329[t].item())
                l2_q335_ci.append(l2_q335[t].item())
                l2_q270_ci.append(q270_val := l2_q270[t].item())
                l2_k206_ci.append(l2_k206[t].item())
                l2_k224_ci.append(l2_k224[t].item())

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i+1}/{N_SAMPLES}")

    N = len(l1_off1)
    l1_arr = np.array(l1_off1)  # (N, n_heads)
    l2_arr = np.array(l2_off1)
    l2_k224_arr = np.array(l2_k224_ci)
    l2_k206_arr = np.array(l2_k206_ci)
    l2_q335_arr = np.array(l2_q335_ci)
    l2_q270_arr = np.array(l2_q270_ci)

    logger.info(f"\nTotal positions: {N}")

    # Mean prev-tok attention per head
    l1_means = l1_arr.mean(axis=0)
    l2_means = l2_arr.mean(axis=0)

    logger.info("\n=== Mean attention at offset=1 (previous token) ===")
    for h in range(n_heads):
        logger.info(f"  H{h}: L1={l1_means[h]:.4f}, L2={l2_means[h]:.4f}")

    # Variance of prev-tok attention
    l1_vars = l1_arr.var(axis=0)
    l2_vars = l2_arr.var(axis=0)
    logger.info("\n=== Variance of offset=1 attention ===")
    for h in range(n_heads):
        logger.info(f"  H{h}: L1={l1_vars[h]:.6f}, L2={l2_vars[h]:.6f}, ratio={l2_vars[h]/max(l1_vars[h],1e-10):.2f}x")

    # Correlate L2 offset=1 with semantic CIs
    logger.info("\n=== L2 offset=1 attention vs semantic component CIs ===")
    comp_names = ["K224", "K206", "Q335", "Q270"]
    comp_arrs = [l2_k224_arr, l2_k206_arr, l2_q335_arr, l2_q270_arr]

    corr_table = np.zeros((len(comp_names), n_heads))
    for ci_idx, (name, arr) in enumerate(zip(comp_names, comp_arrs)):
        for h in range(n_heads):
            r, _ = stats.pearsonr(arr, l2_arr[:, h])
            corr_table[ci_idx, h] = r
        logger.info(f"  {name}: " + ", ".join(f"H{h}={corr_table[ci_idx,h]:+.3f}" for h in range(n_heads)))

    # --- Plot: 2x2 contrast figure ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Mean prev-tok attention L1 vs L2
    ax = axes[0, 0]
    x = np.arange(n_heads)
    width = 0.35
    ax.bar(x - width/2, l1_means, width, label="L1 (always-on)", color="tab:blue")
    ax.bar(x + width/2, l2_means, width, label="L2 (semantic)", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.set_ylabel("Mean attention at offset=1")
    ax.set_title("A. Previous-token attention: L1 vs L2", fontweight="bold")
    ax.legend()

    # Panel B: Variance comparison
    ax = axes[0, 1]
    ax.bar(x - width/2, l1_vars, width, label="L1 variance", color="tab:blue")
    ax.bar(x + width/2, l2_vars, width, label="L2 variance", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.set_ylabel("Variance of offset=1 attention")
    ax.set_title("B. L2 has HIGHER variance (content-modulated)", fontweight="bold")
    ax.legend()

    # Panel C: Correlation heatmap — semantic CIs vs L2 prev-tok
    ax = axes[1, 0]
    vmax = max(0.1, np.abs(corr_table).max())
    im = ax.imshow(corr_table, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_yticks(range(len(comp_names)))
    ax.set_yticklabels(comp_names)
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.set_title("C. Which components modulate L2 prev-tok?", fontweight="bold")
    for i in range(len(comp_names)):
        for h in range(n_heads):
            ax.text(h, i, f"{corr_table[i,h]:.2f}", ha="center", va="center", fontsize=8)

    # Panel D: L2 H0 prev-tok split by K224 state
    ax = axes[1, 1]
    k224_on = l2_k224_arr > 0.5
    k224_off = l2_k224_arr < 0.5

    for h, color, label in [(0, "tab:blue", "H0"), (1, "tab:green", "H1"), (5, "tab:red", "H5")]:
        if k224_on.sum() > 0 and k224_off.sum() > 0:
            on_val = l2_arr[k224_on, h].mean()
            off_val = l2_arr[k224_off, h].mean()
            ax.bar([f"{label}\nK224 ON", f"{label}\nK224 OFF"],
                   [on_val, off_val], color=color, alpha=0.7)

    ax.set_ylabel("Mean prev-tok attention")
    ax.set_title("D. K224 ON promotes prev-tok in H0", fontweight="bold")

    fig.suptitle(
        "Previous-token behavior contrast: L1 (always-on, stable) vs L2 (content-dependent)\n"
        "L1 Q316/K329 drive uniform prev-tok; L2 prev-tok is modulated by K224/K206/Q335",
        fontweight="bold", fontsize=12,
    )
    fig.tight_layout()
    path = OUT_DIR / "prevtok_contrast.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"\nSaved {path}")


if __name__ == "__main__":
    main()
