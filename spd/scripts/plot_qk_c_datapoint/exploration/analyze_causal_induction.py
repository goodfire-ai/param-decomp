"""Causal ablation measured specifically at induction positions.

Ablate K206 (and Q335) from L2 weight matrices, then measure per-head attention
to induction targets at repeated-token positions. This confirms that the K206
suppression of H4 induction is causal, not just a confound of text type.

Also ablates the top induction components (Q195, K166) to show they causally
drive H4 induction — the complementary side of the story.
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


def compute_component_weight(comp: LinearComponents, indices: list[int]) -> torch.Tensor:
    V_sub = comp.V[:, indices]
    U_sub = comp.U[indices]
    return torch.einsum("ic,co->oi", V_sub, U_sub)


def collect_induction_weights(
    target_model: LlamaSimpleMLP,
    loader,
    task_config,
    seq_len: int,
    n_samples: int,
    n_heads: int,
    device: torch.device,
) -> tuple[np.ndarray, int]:
    """Collect per-head mean attention to induction targets. Returns (n_heads,) and event count."""
    totals = np.zeros(n_heads)
    count = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_samples:
                break
            input_ids = batch[task_config.column_name][:, :seq_len].to(device)
            T = input_ids.shape[1]

            results = collect_attention_patterns_with_logits(target_model, input_ids)
            attn = torch.softmax(results[LAYER][1], dim=-1)[0]

            token_list = input_ids[0].cpu().tolist()
            for t in range(4, T):
                for prev_t in range(t - 1):
                    if token_list[prev_t] == token_list[t] and prev_t + 1 < t:
                        ik = prev_t + 1
                        for h in range(n_heads):
                            totals[h] += attn[h, t, ik].item()
                        count += 1
                        break

            if (i + 1) % 50 == 0:
                logger.info(f"  {i+1}/{n_samples}, events: {count}")

    if count > 0:
        totals /= count
    return totals, count


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
    q_comp = model.components[f"h.{LAYER}.attn.q_proj"]
    k_comp = model.components[f"h.{LAYER}.attn.k_proj"]
    assert isinstance(q_comp, LinearComponents)
    assert isinstance(k_comp, LinearComponents)

    attn_module = target_model._h[LAYER].attn
    q_weight = attn_module.q_proj.weight
    k_weight = attn_module.k_proj.weight

    ablation_configs = [
        ("baseline", [], []),
        ("−K206", [], [206]),
        ("−Q335", [335], []),
        ("−Q335−K206", [335], [206]),
        ("−Q270", [270], []),
        # Induction components
        ("−Q195", [195], []),
        ("−K166", [], [166]),
        ("−Q195−K166", [195], [166]),
        # Top 5 induction Q + K
        ("−top5_induction_Q", [195, 499, 259, 110, 435], []),
        ("−top5_induction_K", [], [166, 5, 251, 55, 1]),
    ]

    results_dict: dict[str, tuple[np.ndarray, int]] = {}

    for name, q_indices, k_indices in ablation_configs:
        logger.info(f"\n=== {name} ===")

        q_delta = None
        k_delta = None

        if q_indices:
            q_delta = compute_component_weight(q_comp, q_indices)
            q_weight.data -= q_delta
        if k_indices:
            k_delta = compute_component_weight(k_comp, k_indices)
            k_weight.data -= k_delta

        # Need fresh loader each time (streaming)
        loader_fresh, _ = create_data_loader(
            dataset_config=dataset_config, batch_size=1, buffer_size=1000
        )

        try:
            means, count = collect_induction_weights(
                target_model, loader_fresh, task_config, seq_len, N_SAMPLES, n_heads, device
            )
            results_dict[name] = (means, count)
        finally:
            if q_delta is not None:
                q_weight.data += q_delta
            if k_delta is not None:
                k_weight.data += k_delta

    # --- Log results ---
    baseline_means, baseline_count = results_dict["baseline"]
    logger.info(f"\n\n=== RESULTS ({baseline_count} induction events per condition) ===")
    logger.info("\nBaseline per-head induction:")
    for h in range(n_heads):
        logger.info(f"  H{h}: {baseline_means[h]:.4f}")

    logger.info("\nCausal effects (Δ from baseline):")
    for name, (means, count) in results_dict.items():
        if name == "baseline":
            continue
        diff = means - baseline_means
        logger.info(f"\n  {name} ({count} events):")
        for h in range(n_heads):
            if abs(diff[h]) > 0.001:
                logger.info(f"    H{h}: {means[h]:.4f} (Δ={diff[h]:+.4f})")

    # --- Plot 1: Per-head induction for semantic ablations ---
    semantic_names = ["baseline", "−K206", "−Q335", "−Q335−K206", "−Q270"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(n_heads)
    width = 0.15
    for idx, name in enumerate(semantic_names):
        means, _ = results_dict[name]
        ax1.bar(x + idx * width, means, width, label=name)
    ax1.set_xlabel("Head")
    ax1.set_ylabel("Mean attention to induction target")
    ax1.set_title("Semantic component ablations → induction", fontweight="bold")
    ax1.set_xticks(x + width * 2)
    ax1.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax1.legend(fontsize=7)

    # Differences from baseline
    for idx, name in enumerate(semantic_names[1:]):
        means, _ = results_dict[name]
        diff = means - baseline_means
        ax2.bar(x + idx * width, diff, width, label=name)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("Head")
    ax2.set_ylabel("Δ induction attention")
    ax2.set_title("Change in induction when semantic components ablated", fontweight="bold")
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax2.legend(fontsize=7)

    fig.suptitle(
        f"Layer {LAYER}: Causal ablation of SEMANTIC components → effect on induction\n"
        f"(Does removing local-focus components free up attention for H4 induction?)",
        fontweight="bold",
    )
    fig.tight_layout()
    path = OUT_DIR / "causal_induction_semantic.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"\nSaved {path}")

    # --- Plot 2: Induction component ablations ---
    induction_names = ["baseline", "−Q195", "−K166", "−Q195−K166",
                       "−top5_induction_Q", "−top5_induction_K"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for idx, name in enumerate(induction_names):
        means, _ = results_dict[name]
        ax1.bar(x + idx * width, means, width, label=name)
    ax1.set_xlabel("Head")
    ax1.set_ylabel("Mean attention to induction target")
    ax1.set_title("Induction component ablations", fontweight="bold")
    ax1.set_xticks(x + width * 2.5)
    ax1.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax1.legend(fontsize=6)

    for idx, name in enumerate(induction_names[1:]):
        means, _ = results_dict[name]
        diff = means - baseline_means
        ax2.bar(x + idx * width, diff, width, label=name)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("Head")
    ax2.set_ylabel("Δ induction attention")
    ax2.set_title("Change in induction when induction components ablated", fontweight="bold")
    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax2.legend(fontsize=6)

    fig.suptitle(
        f"Layer {LAYER}: Causal ablation of INDUCTION components → effect on induction\n"
        f"(Do the top induction-correlated components actually drive induction?)",
        fontweight="bold",
    )
    fig.tight_layout()
    path = OUT_DIR / "causal_induction_components.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")

    # --- Plot 3: Combined summary ---
    fig, ax = plt.subplots(figsize=(12, 6))

    key_ablations = ["−K206", "−Q335", "−Q335−K206", "−Q270",
                     "−Q195", "−K166", "−Q195−K166", "−top5_induction_Q", "−top5_induction_K"]
    h4_diffs = []
    h0_diffs = []
    for name in key_ablations:
        means, _ = results_dict[name]
        diff = means - baseline_means
        h4_diffs.append(diff[INDUCTION_HEAD])
        h0_diffs.append(diff[0])

    x_pos = np.arange(len(key_ablations))
    width = 0.35
    ax.bar(x_pos - width/2, h4_diffs, width, label="H4 (induction)", color="tab:blue")
    ax.bar(x_pos + width/2, h0_diffs, width, label="H0 (local focus)", color="tab:red")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(key_ablations, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Δ induction attention")
    ax.set_title(
        f"Layer {LAYER}: Causal effects on H4 (induction) vs H0 (local focus)\n"
        f"Semantic ablations should help H4; Induction ablations should hurt H4",
        fontweight="bold",
    )
    ax.legend()
    fig.tight_layout()
    path = OUT_DIR / "causal_induction_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


if __name__ == "__main__":
    main()
