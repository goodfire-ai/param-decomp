"""Analyze how CI values continuously modulate attention patterns.

The binary thresholding approach failed for Q335/K206 because they're active on
virtually all natural language (499/500 samples at CI>0.1). Instead, we treat CI
as a continuous variable and compute correlations with attention offset profiles.

For each component, at each head and offset:
  correlation(CI_value, attention_at_offset) across positions/samples

This reveals the continuous relationship between component activation strength
and attention distribution.
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
MAX_OFFSET = 16

COMPONENTS_TO_ANALYZE = [
    ("q_proj", 335, "Q335 (diverse/multilingual)"),
    ("k_proj", 206, "K206 (conversational)"),
    ("q_proj", 270, "Q270 (research/technical)"),
    ("q_proj", 279, "Q279 (social/human)"),
    ("k_proj", 224, "K224 (technical nouns)"),
]


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

    # Per-position data: ci_values and attention offsets for each component
    # component_key -> list of (ci_val, offsets_per_head) tuples
    data: dict[str, list[tuple[float, np.ndarray]]] = {
        f"{mod}.{idx}": [] for mod, idx, _ in COMPONENTS_TO_ANALYZE
    }

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
            attn_weights = torch.softmax(results[LAYER][1], dim=-1)[0]  # (n_heads, T, T)

            for mod, idx, _ in COMPONENTS_TO_ANALYZE:
                module_path = f"h.{LAYER}.attn.{mod}"
                comp_ci = ci[module_path][0, :, idx]  # (T,)

                for t in range(MAX_OFFSET, T):
                    ci_val = comp_ci[t].item()
                    offsets = np.zeros((n_heads, MAX_OFFSET))
                    for h in range(n_heads):
                        for d in range(MAX_OFFSET):
                            offsets[h, d] = attn_weights[h, t, t - d].item()

                    data[f"{mod}.{idx}"].append((ci_val, offsets))

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i+1}/{N_SAMPLES}")

    # Compute correlations: for each component, head, offset
    for mod, idx, label in COMPONENTS_TO_ANALYZE:
        key = f"{mod}.{idx}"
        entries = data[key]
        ci_vals = np.array([e[0] for e in entries])
        offsets_arr = np.array([e[1] for e in entries])  # (N, n_heads, max_offset)

        logger.info(f"\n=== {label} ===")
        logger.info(f"  N positions: {len(entries)}")
        logger.info(f"  CI range: [{ci_vals.min():.3f}, {ci_vals.max():.3f}], mean={ci_vals.mean():.3f}")

        # Correlation matrix: (n_heads, max_offset)
        corr_matrix = np.zeros((n_heads, MAX_OFFSET))
        pval_matrix = np.zeros((n_heads, MAX_OFFSET))

        for h in range(n_heads):
            for d in range(MAX_OFFSET):
                r, p = stats.pearsonr(ci_vals, offsets_arr[:, h, d])
                corr_matrix[h, d] = r
                pval_matrix[h, d] = p

        # Plot correlation heatmap
        fig, ax = plt.subplots(figsize=(12, 4))
        vmax = max(0.05, np.abs(corr_matrix).max())
        im = ax.imshow(
            corr_matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            interpolation="nearest",
        )
        fig.colorbar(im, ax=ax, label="Pearson r")

        ax.set_yticks(range(n_heads))
        ax.set_yticklabels([f"H{h}" for h in range(n_heads)])
        ax.set_xticks(range(MAX_OFFSET))
        ax.set_xticklabels([str(d) for d in range(MAX_OFFSET)])
        ax.set_xlabel("Offset (tokens back)")
        ax.set_ylabel("Head")
        ax.set_title(f"Layer {LAYER}: Correlation of {label} CI with attention at each offset")

        # Mark significant correlations
        for h in range(n_heads):
            for d in range(MAX_OFFSET):
                if pval_matrix[h, d] < 0.001:
                    ax.text(d, h, f"{corr_matrix[h, d]:.3f}", ha="center", va="center",
                            fontsize=6, fontweight="bold")

        fig.tight_layout()
        safe_label = label.split("(")[0].strip().replace(" ", "_")
        path = OUT_DIR / f"continuous_{safe_label}_correlation.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved {path}")

        # Log top correlations
        flat_idx = np.argsort(np.abs(corr_matrix).ravel())[::-1]
        logger.info("  Top 10 correlations:")
        for rank, fi in enumerate(flat_idx[:10]):
            h, d = divmod(fi, MAX_OFFSET)
            r = corr_matrix[h, d]
            p = pval_matrix[h, d]
            logger.info(f"    #{rank+1}: H{h} offset={d}, r={r:+.4f}, p={p:.2e}")

    # --- Summary plot: side-by-side correlation heatmaps for all components ---
    fig, axes = plt.subplots(1, len(COMPONENTS_TO_ANALYZE), figsize=(5 * len(COMPONENTS_TO_ANALYZE), 4))

    for ax_idx, (mod, idx, label) in enumerate(COMPONENTS_TO_ANALYZE):
        key = f"{mod}.{idx}"
        entries = data[key]
        ci_vals = np.array([e[0] for e in entries])
        offsets_arr = np.array([e[1] for e in entries])

        corr_matrix = np.zeros((n_heads, MAX_OFFSET))
        for h in range(n_heads):
            for d in range(MAX_OFFSET):
                r, _ = stats.pearsonr(ci_vals, offsets_arr[:, h, d])
                corr_matrix[h, d] = r

        ax = axes[ax_idx]
        vmax = 0.15
        im = ax.imshow(corr_matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_yticks(range(n_heads))
        ax.set_yticklabels([f"H{h}" for h in range(n_heads)])
        ax.set_xticks(range(0, MAX_OFFSET, 2))
        ax.set_xlabel("Offset")
        short_label = label.split("(")[0].strip()
        ax.set_title(short_label, fontweight="bold")

    fig.colorbar(im, ax=axes[-1], label="Pearson r", shrink=0.8)
    fig.suptitle(
        f"Layer {LAYER}: How component CI correlates with attention at each (head, offset)\n"
        f"Red = higher CI → more attention; Blue = higher CI → less attention",
        fontweight="bold",
    )
    fig.tight_layout()
    path = OUT_DIR / "continuous_all_components_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"\nSaved summary: {path}")


if __name__ == "__main__":
    main()
