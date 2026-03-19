"""Quantify how much attention logit variance Q270 vs Q335 explain.

For each head and key position offset, compute the fraction of attention logit
variance explained by Q270 CI, Q335 CI, and their interaction, using linear
regression (R² decomposition).

This answers: "When the model switches between local and broad attention,
how much of the total variation in attention is accounted for by the semantic
components?"
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression

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
MAX_OFFSET = 16


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

    n_heads = target_model._h[LAYER].attn.n_head
    q_path = f"h.{LAYER}.attn.q_proj"
    k_path = f"h.{LAYER}.attn.k_proj"

    # Collect per-position data
    q335_ci_list: list[float] = []
    q270_ci_list: list[float] = []
    k206_ci_list: list[float] = []
    k224_ci_list: list[float] = []
    attn_offsets_list: list[np.ndarray] = []  # (n_heads, MAX_OFFSET)
    logit_offsets_list: list[np.ndarray] = []  # (n_heads, MAX_OFFSET) pre-softmax

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
            logits = results[LAYER][1][0]  # (n_heads, T, T) pre-softmax
            attn = torch.softmax(results[LAYER][1], dim=-1)[0]  # (n_heads, T, T)

            q335 = ci[q_path][0, :, 335].cpu()
            q270 = ci[q_path][0, :, 270].cpu()
            k206 = ci[k_path][0, :, 206].cpu()
            k224 = ci[k_path][0, :, 224].cpu()

            for t in range(MAX_OFFSET, T):
                q335_ci_list.append(q335[t].item())
                q270_ci_list.append(q270[t].item())
                k206_ci_list.append(k206[t].item())
                k224_ci_list.append(k224[t].item())

                a_off = np.zeros((n_heads, MAX_OFFSET))
                l_off = np.zeros((n_heads, MAX_OFFSET))
                for h in range(n_heads):
                    for d in range(MAX_OFFSET):
                        a_off[h, d] = attn[h, t, t - d].item()
                        l_off[h, d] = logits[h, t, t - d].item()
                attn_offsets_list.append(a_off)
                logit_offsets_list.append(l_off)

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i+1}/{N_SAMPLES}, positions: {len(q335_ci_list)}")

    N = len(q335_ci_list)
    logger.info(f"\nTotal positions: {N}")

    q335 = np.array(q335_ci_list)
    q270 = np.array(q270_ci_list)
    k206 = np.array(k206_ci_list)
    k224 = np.array(k224_ci_list)
    attn_arr = np.array(attn_offsets_list)  # (N, n_heads, MAX_OFFSET)
    logit_arr = np.array(logit_offsets_list)

    # --- R² analysis: how much variance does each component CI explain? ---
    # For each (head, offset), fit: attn ~ Q335 + Q270 + K206 + K224
    # Then decompose into individual and joint R²

    component_names = ["Q335", "Q270", "K206", "K224"]
    component_arrs = [q335, q270, k206, k224]

    # Individual R² per component
    r2_individual = np.zeros((len(component_names), n_heads, MAX_OFFSET))
    # Joint R² (all 4 together)
    r2_joint = np.zeros((n_heads, MAX_OFFSET))
    # Unique R² (what each adds beyond the others)
    r2_unique = np.zeros((len(component_names), n_heads, MAX_OFFSET))

    for h in range(n_heads):
        for d in range(MAX_OFFSET):
            y = attn_arr[:, h, d]

            # Joint model
            X_all = np.column_stack(component_arrs)
            reg_all = LinearRegression().fit(X_all, y)
            r2_joint[h, d] = reg_all.score(X_all, y)

            for ci_idx, (name, x_arr) in enumerate(zip(component_names, component_arrs)):
                # Individual
                X_single = x_arr.reshape(-1, 1)
                reg_single = LinearRegression().fit(X_single, y)
                r2_individual[ci_idx, h, d] = reg_single.score(X_single, y)

                # Unique: fit without this component, compare to joint
                others = [a for j, a in enumerate(component_arrs) if j != ci_idx]
                X_others = np.column_stack(others)
                reg_others = LinearRegression().fit(X_others, y)
                r2_others = reg_others.score(X_others, y)
                r2_unique[ci_idx, h, d] = max(0, r2_joint[h, d] - r2_others)

    # --- Plot 1: Individual R² heatmaps ---
    fig, axes = plt.subplots(1, len(component_names) + 1, figsize=(5 * (len(component_names) + 1), 4))

    for ci_idx, name in enumerate(component_names):
        ax = axes[ci_idx]
        im = ax.imshow(r2_individual[ci_idx], aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.15)
        ax.set_yticks(range(n_heads))
        ax.set_yticklabels([f"H{h}" for h in range(n_heads)])
        ax.set_xticks(range(0, MAX_OFFSET, 2))
        ax.set_xlabel("Offset")
        ax.set_title(f"{name} alone", fontweight="bold")

        for hh in range(n_heads):
            for dd in range(MAX_OFFSET):
                val = r2_individual[ci_idx, hh, dd]
                if val > 0.02:
                    ax.text(dd, hh, f"{val:.2f}", ha="center", va="center", fontsize=5)

    # Joint
    ax = axes[-1]
    im = ax.imshow(r2_joint, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.25)
    fig.colorbar(im, ax=ax, shrink=0.8, label="R²")
    ax.set_yticks(range(n_heads))
    ax.set_yticklabels([f"H{h}" for h in range(n_heads)])
    ax.set_xticks(range(0, MAX_OFFSET, 2))
    ax.set_xlabel("Offset")
    ax.set_title("All 4 jointly", fontweight="bold")

    for hh in range(n_heads):
        for dd in range(MAX_OFFSET):
            val = r2_joint[hh, dd]
            if val > 0.02:
                ax.text(dd, hh, f"{val:.2f}", ha="center", va="center", fontsize=5)

    fig.suptitle(
        f"Layer {LAYER}: Fraction of attention variance (R²) explained by semantic component CIs\n"
        f"({N:,} positions across {N_SAMPLES} samples)",
        fontweight="bold",
    )
    fig.tight_layout()
    path = OUT_DIR / "variance_explained_r2_individual.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")

    # --- Plot 2: Unique R² (what each component adds beyond the others) ---
    fig, axes = plt.subplots(1, len(component_names), figsize=(5 * len(component_names), 4))

    for ci_idx, name in enumerate(component_names):
        ax = axes[ci_idx]
        im = ax.imshow(r2_unique[ci_idx], aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.05)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_yticks(range(n_heads))
        ax.set_yticklabels([f"H{h}" for h in range(n_heads)])
        ax.set_xticks(range(0, MAX_OFFSET, 2))
        ax.set_xlabel("Offset")
        ax.set_title(f"{name} unique", fontweight="bold")

        for hh in range(n_heads):
            for dd in range(MAX_OFFSET):
                val = r2_unique[ci_idx, hh, dd]
                if val > 0.005:
                    ax.text(dd, hh, f"{val:.3f}", ha="center", va="center", fontsize=5)

    fig.suptitle(
        f"Layer {LAYER}: Unique variance explained (R² added beyond other 3 components)\n"
        "How much does each component UNIQUELY contribute?",
        fontweight="bold",
    )
    fig.tight_layout()
    path = OUT_DIR / "variance_explained_r2_unique.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")

    # --- Summary table ---
    logger.info("\n=== Variance explained summary (R²) ===")
    logger.info("\nIndividual R² (top 5 per component):")
    for ci_idx, name in enumerate(component_names):
        entries = []
        for h in range(n_heads):
            for d in range(MAX_OFFSET):
                entries.append((h, d, r2_individual[ci_idx, h, d]))
        entries.sort(key=lambda x: x[2], reverse=True)
        logger.info(f"  {name}:")
        for h, d, r2 in entries[:5]:
            logger.info(f"    H{h} offset={d}: R²={r2:.4f}")

    logger.info(f"\nJoint R² (all 4, top 10):")
    entries = []
    for h in range(n_heads):
        for d in range(MAX_OFFSET):
            entries.append((h, d, r2_joint[h, d]))
    entries.sort(key=lambda x: x[2], reverse=True)
    for h, d, r2 in entries[:10]:
        logger.info(f"  H{h} offset={d}: R²={r2:.4f}")

    # --- Plot 3: Bar chart of total R² by head (summed over offsets 0-5) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    offsets_to_sum = range(6)
    bar_width = 0.15
    x = np.arange(n_heads)

    for ci_idx, name in enumerate(component_names):
        r2_sum = r2_individual[ci_idx, :, :6].sum(axis=1)
        ax.bar(x + ci_idx * bar_width, r2_sum, bar_width, label=name)

    r2_joint_sum = r2_joint[:, :6].sum(axis=1)
    ax.bar(x + len(component_names) * bar_width, r2_joint_sum, bar_width,
           label="Joint", color="black", alpha=0.5)

    ax.set_xlabel("Head")
    ax.set_ylabel("Sum R² (offsets 0-5)")
    ax.set_title(f"Layer {LAYER}: Total variance explained by semantic components (offsets 0-5)")
    ax.set_xticks(x + bar_width * 2)
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.legend()

    fig.tight_layout()
    path = OUT_DIR / "variance_explained_by_head.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


if __name__ == "__main__":
    main()
