"""Find contrasting exemplar samples and show side-by-side attention heatmaps.

Identifies:
- "Research" samples: Q270 CI is high (research/technical content)
- "Conversational" samples: Q270 CI is low, K206 CI is high

For each pair, plots full attention heatmaps (6 heads) side by side,
showing how the attention pattern shifts when Q270 activates.

Also verifies H4 independence: H4's induction behavior should be stable
across both sample types.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
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
N_SAMPLES = 500
MAX_OFFSET = 16


def compute_induction_score_per_head(
    attn_weights: torch.Tensor, token_list: list[int], n_heads: int, T: int
) -> np.ndarray:
    """Compute induction score per head: mean attention to token-after-previous-occurrence."""
    scores = np.zeros(n_heads)
    count = 0
    for t in range(2, T):
        for prev_t in range(t - 1):
            if token_list[prev_t] == token_list[t] and prev_t + 1 < t:
                for h in range(n_heads):
                    scores[h] += attn_weights[h, t, prev_t + 1].item()
                count += 1
    if count > 0:
        scores /= count
    return scores


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
    q_path = f"h.{LAYER}.attn.q_proj"
    k_path = f"h.{LAYER}.attn.k_proj"

    # Collect samples with their CI values and attention patterns
    samples: list[dict] = []

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

            q270_ci = ci[q_path][0, :, 270].mean().item()
            q335_ci = ci[q_path][0, :, 335].mean().item()
            k206_ci = ci[k_path][0, :, 206].mean().item()
            k224_ci = ci[k_path][0, :, 224].mean().item()

            results = collect_attention_patterns_with_logits(target_model, input_ids)
            attn_weights = torch.softmax(results[LAYER][1], dim=-1)[0]  # (n_heads, T, T)

            # Offset profile per head
            offset_profile = np.zeros((n_heads, MAX_OFFSET))
            for d in range(MAX_OFFSET):
                diag = torch.diagonal(attn_weights, offset=-d, dim1=-2, dim2=-1)
                offset_profile[:, d] = diag.float().mean(dim=1).cpu().numpy()

            # Induction score
            token_list = input_ids[0].cpu().tolist()
            induction_scores = compute_induction_score_per_head(attn_weights, token_list, n_heads, T)

            tokens_str = tokenizer.decode(input_ids[0][:40])  # pyright: ignore[reportAttributeAccessIssue]

            samples.append({
                "idx": i,
                "q270_ci": q270_ci,
                "q335_ci": q335_ci,
                "k206_ci": k206_ci,
                "k224_ci": k224_ci,
                "offset_profile": offset_profile,
                "induction_scores": induction_scores,
                "attn_weights": attn_weights.cpu().numpy()[:, :32, :32],  # keep first 32x32
                "tokens_str": tokens_str,
                "input_ids": input_ids[0].cpu(),
            })

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i+1}/{N_SAMPLES}")

    # Sort by Q270 CI to find high-research and low-research samples
    samples.sort(key=lambda s: s["q270_ci"])

    # Pick top 5 high-Q270 and bottom 5 low-Q270 (with high K206 to ensure "conversational")
    low_q270 = [s for s in samples[:50] if s["k206_ci"] > 0.5][:5]
    high_q270 = samples[-5:]

    logger.info("\n=== High Q270 (research/technical) samples ===")
    for s in high_q270:
        logger.info(
            f"  Sample {s['idx']}: Q270={s['q270_ci']:.3f}, K206={s['k206_ci']:.3f}, "
            f"Q335={s['q335_ci']:.3f}"
        )
        logger.info(f"    {s['tokens_str'][:100]}")

    logger.info("\n=== Low Q270 + High K206 (conversational) samples ===")
    for s in low_q270:
        logger.info(
            f"  Sample {s['idx']}: Q270={s['q270_ci']:.3f}, K206={s['k206_ci']:.3f}, "
            f"Q335={s['q335_ci']:.3f}"
        )
        logger.info(f"    {s['tokens_str'][:100]}")

    # --- Plot 1: Side-by-side offset profiles ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    high_profiles = np.mean([s["offset_profile"] for s in high_q270], axis=0)
    low_profiles = np.mean([s["offset_profile"] for s in low_q270], axis=0)

    for h in range(n_heads):
        row, col = divmod(h, 3)
        ax = axes[row, col]
        x = list(range(MAX_OFFSET))
        ax.plot(x, high_profiles[h], "b-", linewidth=2,
                label=f"High Q270 (research, n={len(high_q270)})")
        ax.plot(x, low_profiles[h], "r--", linewidth=2,
                label=f"Low Q270 + High K206 (conv, n={len(low_q270)})")
        ax.set_title(f"H{h}", fontweight="bold")
        ax.set_xlabel("Offset")
        ax.set_ylabel("Mean attention")
        if h == 0:
            ax.legend(fontsize=7)

    fig.suptitle(
        f"Layer {LAYER}: Attention offset profiles — research vs conversational content",
        fontweight="bold",
    )
    fig.tight_layout()
    path = OUT_DIR / "exemplar_offset_profiles.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"\nSaved {path}")

    # --- Plot 2: Per-head induction scores ---
    high_induction = np.mean([s["induction_scores"] for s in high_q270], axis=0)
    low_induction = np.mean([s["induction_scores"] for s in low_q270], axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(n_heads)
    width = 0.35
    ax.bar(x - width / 2, high_induction, width, label="Research (high Q270)", color="tab:blue")
    ax.bar(x + width / 2, low_induction, width, label="Conversational (low Q270)", color="tab:red")
    ax.set_xlabel("Head")
    ax.set_ylabel("Induction score")
    ax.set_title(f"Layer {LAYER}: Induction scores — research vs conversational")
    ax.set_xticks(x)
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.legend()

    path = OUT_DIR / "exemplar_induction_scores.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")

    # --- Plot 3: Side-by-side attention heatmaps for best exemplar pair ---
    if high_q270 and low_q270:
        best_high = high_q270[-1]  # highest Q270
        best_low = low_q270[0]    # lowest Q270 with high K206

        fig = plt.figure(figsize=(24, 14))
        gs = GridSpec(3, 6, figure=fig, height_ratios=[0.1, 1, 1])

        # Title row
        ax_title_high = fig.add_subplot(gs[0, :3])
        ax_title_high.text(
            0.5, 0.5,
            f"RESEARCH (Sample {best_high['idx']}): Q270={best_high['q270_ci']:.3f}\n"
            f"{best_high['tokens_str'][:80]}",
            ha="center", va="center", fontsize=10, fontweight="bold",
            transform=ax_title_high.transAxes,
        )
        ax_title_high.axis("off")

        ax_title_low = fig.add_subplot(gs[0, 3:])
        ax_title_low.text(
            0.5, 0.5,
            f"CONVERSATIONAL (Sample {best_low['idx']}): Q270={best_low['q270_ci']:.3f}\n"
            f"{best_low['tokens_str'][:80]}",
            ha="center", va="center", fontsize=10, fontweight="bold",
            transform=ax_title_low.transAxes,
        )
        ax_title_low.axis("off")

        sz = min(32, best_high["attn_weights"].shape[1])

        for h in range(n_heads):
            # Research
            row = 1 + h // 3
            col = h % 3
            ax_h = fig.add_subplot(gs[row, col]) if h < 3 else None
            if h >= 3:
                # Use second row for heads 3-5 in left half
                pass

        # Simpler approach: 2 rows of 6 (all heads), top=research, bottom=conversational
        plt.close(fig)

        fig, axes = plt.subplots(2, n_heads, figsize=(4 * n_heads, 8))
        sz = min(24, best_high["attn_weights"].shape[1])

        for h in range(n_heads):
            # Research
            ax = axes[0, h]
            im = ax.imshow(
                best_high["attn_weights"][h, :sz, :sz],
                aspect="auto", cmap="Blues", vmin=0, vmax=0.5,
            )
            ax.set_title(f"H{h}", fontweight="bold")
            if h == 0:
                ax.set_ylabel("Research\n(high Q270)", fontweight="bold")

            # Conversational
            ax = axes[1, h]
            im = ax.imshow(
                best_low["attn_weights"][h, :sz, :sz],
                aspect="auto", cmap="Blues", vmin=0, vmax=0.5,
            )
            if h == 0:
                ax.set_ylabel("Conversational\n(low Q270)", fontweight="bold")

        fig.colorbar(im, ax=axes, shrink=0.6, label="Attention weight")
        fig.suptitle(
            f"Layer {LAYER}: Attention heatmaps — research vs conversational exemplars\n"
            f"Research: Q270={best_high['q270_ci']:.3f}, Conv: Q270={best_low['q270_ci']:.3f}",
            fontweight="bold", fontsize=12,
        )
        fig.tight_layout()
        path = OUT_DIR / "exemplar_attention_heatmaps.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {path}")

    # --- Plot 4: H4 induction stability ---
    # Scatter: Q270 CI vs H4 induction score
    q270_vals = [s["q270_ci"] for s in samples]
    h4_induction = [s["induction_scores"][4] for s in samples]
    h0_induction = [s["induction_scores"][0] for s in samples]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(q270_vals, h4_induction, alpha=0.3, s=10, c="tab:blue")
    r_h4 = np.corrcoef(q270_vals, h4_induction)[0, 1]
    ax1.set_xlabel("Q270 mean CI")
    ax1.set_ylabel("H4 induction score")
    ax1.set_title(f"H4 (induction head): r={r_h4:.3f}")

    ax2.scatter(q270_vals, h0_induction, alpha=0.3, s=10, c="tab:red")
    r_h0 = np.corrcoef(q270_vals, h0_induction)[0, 1]
    ax2.set_xlabel("Q270 mean CI")
    ax2.set_ylabel("H0 induction score")
    ax2.set_title(f"H0 (modulated head): r={r_h0:.3f}")

    fig.suptitle(
        f"Layer {LAYER}: Q270 CI vs induction score — H4 is stable, H0 varies",
        fontweight="bold",
    )
    fig.tight_layout()
    path = OUT_DIR / "exemplar_h4_independence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")

    logger.info(f"\nH4 induction vs Q270 CI: r={r_h4:.4f}")
    logger.info(f"H0 induction vs Q270 CI: r={r_h0:.4f}")


if __name__ == "__main__":
    main()
