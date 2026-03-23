"""Investigate K347 — an unmapped semantic K component in Layer 2.

K347 was mentioned as conditionally active but never analyzed. This script:
1. Checks K347's CI distribution and activation frequency
2. Correlates K347 CI with attention patterns (like Experiment 6)
3. Compares K347 with K206 and K224 — is it redundant or distinct?
4. Causal ablation: what happens when K347 is removed?
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
from spd.models.components import LinearComponents
from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from spd.scripts.collect_attention_patterns import collect_attention_patterns_with_logits

OUT_DIR = Path(__file__).parent / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WANDB_PATH = "wandb:goodfire/spd/runs/s-55ea3f9b"
LAYER = 2
N_SAMPLES = 200
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
    k_path = f"h.{LAYER}.attn.k_proj"
    q_path = f"h.{LAYER}.attn.q_proj"

    # Collect per-position data
    k347_ci_list: list[float] = []
    k206_ci_list: list[float] = []
    k224_ci_list: list[float] = []
    q335_ci_list: list[float] = []
    q270_ci_list: list[float] = []
    attn_offsets: list[np.ndarray] = []

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

            k347 = ci[k_path][0, :, 347].cpu()
            k206 = ci[k_path][0, :, 206].cpu()
            k224 = ci[k_path][0, :, 224].cpu()
            q335 = ci[q_path][0, :, 335].cpu()
            q270 = ci[q_path][0, :, 270].cpu()

            results = collect_attention_patterns_with_logits(target_model, input_ids)
            attn = torch.softmax(results[LAYER][1], dim=-1)[0]

            for t in range(MAX_OFFSET, T):
                k347_ci_list.append(k347[t].item())
                k206_ci_list.append(k206[t].item())
                k224_ci_list.append(k224[t].item())
                q335_ci_list.append(q335[t].item())
                q270_ci_list.append(q270[t].item())

                off = np.zeros((n_heads, MAX_OFFSET))
                for h in range(n_heads):
                    for d in range(MAX_OFFSET):
                        off[h, d] = attn[h, t, t - d].item()
                attn_offsets.append(off)

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i+1}/{N_SAMPLES}")

    N = len(k347_ci_list)
    k347_arr = np.array(k347_ci_list)
    k206_arr = np.array(k206_ci_list)
    k224_arr = np.array(k224_ci_list)
    q335_arr = np.array(q335_ci_list)
    q270_arr = np.array(q270_ci_list)
    attn_arr = np.array(attn_offsets)

    logger.info(f"\nTotal positions: {N}")
    logger.info(f"K347 stats: mean={k347_arr.mean():.3f}, >0.5: {(k347_arr > 0.5).sum()}/{N} ({(k347_arr > 0.5).mean()*100:.1f}%)")
    logger.info(f"K206 stats: mean={k206_arr.mean():.3f}, >0.5: {(k206_arr > 0.5).sum()}/{N}")
    logger.info(f"K224 stats: mean={k224_arr.mean():.3f}, >0.5: {(k224_arr > 0.5).sum()}/{N}")

    # Cross-correlations between K components
    r_347_206, _ = stats.pearsonr(k347_arr, k206_arr)
    r_347_224, _ = stats.pearsonr(k347_arr, k224_arr)
    r_206_224, _ = stats.pearsonr(k206_arr, k224_arr)
    r_347_q335, _ = stats.pearsonr(k347_arr, q335_arr)
    r_347_q270, _ = stats.pearsonr(k347_arr, q270_arr)

    logger.info("\nCross-correlations:")
    logger.info(f"  K347-K206: r={r_347_206:+.3f}")
    logger.info(f"  K347-K224: r={r_347_224:+.3f}")
    logger.info(f"  K206-K224: r={r_206_224:+.3f}")
    logger.info(f"  K347-Q335: r={r_347_q335:+.3f}")
    logger.info(f"  K347-Q270: r={r_347_q270:+.3f}")

    # CI-attention correlations for K347
    corr_matrix = np.zeros((n_heads, MAX_OFFSET))
    for h in range(n_heads):
        for d in range(MAX_OFFSET):
            r, _ = stats.pearsonr(k347_arr, attn_arr[:, h, d])
            corr_matrix[h, d] = r

    logger.info("\nK347 CI-attention correlations (top 10):")
    flat_idx = np.argsort(np.abs(corr_matrix).ravel())[::-1]
    for rank, fi in enumerate(flat_idx[:10]):
        h, d = divmod(fi, MAX_OFFSET)
        logger.info(f"  H{h} offset={d}: r={corr_matrix[h, d]:+.4f}")

    # --- Plot 1: K347 CI distribution + comparison with K206/K224 ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    bins = np.linspace(0, 1.05, 40)
    ax.hist(k347_arr, bins=bins, alpha=0.6, label="K347", color="tab:purple", density=True)
    ax.hist(k206_arr, bins=bins, alpha=0.4, label="K206", color="tab:green", density=True)
    ax.hist(k224_arr, bins=bins, alpha=0.4, label="K224", color="tab:orange", density=True)
    ax.set_xlabel("CI value")
    ax.set_ylabel("Density")
    ax.set_title("CI distributions", fontweight="bold")
    ax.legend()

    # K347 vs K206 scatter
    ax = axes[0, 1]
    ax.scatter(k206_arr[::20], k347_arr[::20], alpha=0.15, s=5, c="tab:purple")
    ax.set_xlabel("K206 CI")
    ax.set_ylabel("K347 CI")
    ax.set_title(f"K347 vs K206 (r={r_347_206:+.3f})", fontweight="bold")

    # K347 attention correlation heatmap
    ax = axes[1, 0]
    vmax = max(0.05, np.abs(corr_matrix).max())
    im = ax.imshow(corr_matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_yticks(range(n_heads))
    ax.set_yticklabels([f"H{h}" for h in range(n_heads)])
    ax.set_xticks(range(0, MAX_OFFSET, 2))
    ax.set_xlabel("Offset")
    ax.set_title("K347 CI → attention correlation", fontweight="bold")
    for h in range(n_heads):
        for d in range(MAX_OFFSET):
            if abs(corr_matrix[h, d]) > 0.02:
                ax.text(d, h, f"{corr_matrix[h, d]:.2f}", ha="center", va="center", fontsize=5)

    # Comparison: K206 vs K347 correlation patterns (side-by-side)
    ax = axes[1, 1]
    corr_k206 = np.zeros((n_heads, MAX_OFFSET))
    for h in range(n_heads):
        for d in range(MAX_OFFSET):
            r, _ = stats.pearsonr(k206_arr, attn_arr[:, h, d])
            corr_k206[h, d] = r

    # Show H0 and H5 profiles for both
    for h, h_label in [(0, "H0"), (5, "H5")]:
        ax.plot(range(MAX_OFFSET), corr_matrix[h], "-", linewidth=2,
                label=f"K347→{h_label}", alpha=0.8)
        ax.plot(range(MAX_OFFSET), corr_k206[h], "--", linewidth=2,
                label=f"K206→{h_label}", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Offset")
    ax.set_ylabel("Pearson r")
    ax.set_title("K347 vs K206: attention modulation patterns", fontweight="bold")
    ax.legend(fontsize=7)

    fig.suptitle(
        f"Layer {LAYER}: K347 investigation — is it distinct from K206/K224?",
        fontweight="bold",
    )
    fig.tight_layout()
    path = OUT_DIR / "k347_investigation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"\nSaved {path}")

    # --- Causal ablation of K347 ---
    k_comp = model.components[k_path]
    assert isinstance(k_comp, LinearComponents)
    attn_module = target_model._h[LAYER].attn
    k_weight = attn_module.k_proj.weight

    V_sub = k_comp.V[:, [347]]
    U_sub = k_comp.U[[347]]
    k347_delta = torch.einsum("ic,co->oi", V_sub, U_sub)

    # Collect baseline and ablated offset profiles
    loader2, _ = create_data_loader(dataset_config=dataset_config, batch_size=1, buffer_size=1000)

    baseline_offsets = np.zeros((n_heads, MAX_OFFSET))
    ablated_offsets = np.zeros((n_heads, MAX_OFFSET))
    n_pos = 0

    with torch.no_grad():
        for i, batch in enumerate(loader2):
            if i >= 100:
                break
            input_ids = batch[task_config.column_name][:, :seq_len].to(device)
            T = input_ids.shape[1]

            # Baseline
            results = collect_attention_patterns_with_logits(target_model, input_ids)
            attn_b = torch.softmax(results[LAYER][1], dim=-1)[0]

            # Ablated
            k_weight.data -= k347_delta
            results_a = collect_attention_patterns_with_logits(target_model, input_ids)
            attn_a = torch.softmax(results_a[LAYER][1], dim=-1)[0]
            k_weight.data += k347_delta

            for h in range(n_heads):
                for d in range(MAX_OFFSET):
                    diag_b = torch.diagonal(attn_b[h], offset=-d)
                    diag_a = torch.diagonal(attn_a[h], offset=-d)
                    baseline_offsets[h, d] += diag_b[MAX_OFFSET:].sum().item()
                    ablated_offsets[h, d] += diag_a[MAX_OFFSET:].sum().item()
            n_pos += T - MAX_OFFSET

    baseline_offsets /= n_pos
    ablated_offsets /= n_pos
    diff = ablated_offsets - baseline_offsets

    logger.info("\nCausal ablation of K347:")
    for h in range(n_heads):
        top = sorted([(d, diff[h, d]) for d in range(MAX_OFFSET)], key=lambda x: abs(x[1]), reverse=True)[:3]
        changes = ", ".join(f"off={d}:{v:+.4f}" for d, v in top)
        logger.info(f"  H{h}: {changes}")

    # Total absolute change
    total_abs = np.abs(diff[:, :6]).sum(axis=1)
    logger.info(f"\n  Total |Δ| (offsets 0-5): {', '.join(f'H{h}:{v:.4f}' for h, v in enumerate(total_abs))}")


if __name__ == "__main__":
    main()
