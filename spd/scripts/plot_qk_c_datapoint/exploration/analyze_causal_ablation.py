"""Causal ablation: zero out specific components and measure attention changes.

Instead of correlation, directly ablate Q335, Q270, or K206 by zeroing their
component activations in the Q/K projection, then recompute attention patterns.

This gives causal evidence: "removing component X causes attention to shift by Y."

Approach: Hook into LinearComponents.forward, inject a mask that zeros specific
components, and compare the resulting attention patterns to the unablated baseline.
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
N_SAMPLES = 200
MAX_OFFSET = 16

ABLATION_TARGETS = {
    "Q335": ("q_proj", [335]),
    "Q270": ("q_proj", [270]),
    "K206": ("k_proj", [206]),
    "K224": ("k_proj", [224]),
    "Q335+K206": None,  # special: ablate both
    "Q270+K224": None,  # special: ablate both
}


def compute_component_weight_contribution(
    comp: LinearComponents, indices: list[int]
) -> torch.Tensor:
    """Compute the weight matrix contribution of specific components.

    Returns: (d_out, d_in) tensor — the weight contribution to subtract for ablation.
    """
    V_sub = comp.V[:, indices]  # (d_in, len(indices))
    U_sub = comp.U[indices]  # (len(indices), d_out)
    return torch.einsum("ic,co->oi", V_sub, U_sub)  # (d_out, d_in)


def get_attention_with_ablation(
    target_model: LlamaSimpleMLP,
    component_model: ComponentModel,
    input_ids: torch.Tensor,
    ablate_q_indices: list[int] | None = None,
    ablate_k_indices: list[int] | None = None,
) -> torch.Tensor:
    """Get L2 attention weights, optionally ablating specific Q/K components.

    Ablation works by temporarily subtracting component weight contributions
    from the target model's Q/K projection weights.

    Returns: attention weights (n_heads, T, T)
    """
    q_path = f"h.{LAYER}.attn.q_proj"
    k_path = f"h.{LAYER}.attn.k_proj"

    q_comp = component_model.components[q_path]
    k_comp = component_model.components[k_path]
    assert isinstance(q_comp, LinearComponents)
    assert isinstance(k_comp, LinearComponents)

    attn_module = target_model._h[LAYER].attn
    q_weight = attn_module.q_proj.weight  # (d_out, d_in)
    k_weight = attn_module.k_proj.weight

    q_delta = None
    k_delta = None

    if ablate_q_indices:
        q_delta = compute_component_weight_contribution(q_comp, ablate_q_indices)
        q_weight.data -= q_delta

    if ablate_k_indices:
        k_delta = compute_component_weight_contribution(k_comp, ablate_k_indices)
        k_weight.data -= k_delta

    try:
        results = collect_attention_patterns_with_logits(target_model, input_ids)
        attn = torch.softmax(results[LAYER][1], dim=-1)[0]  # (n_heads, T, T)
    finally:
        if q_delta is not None:
            q_weight.data += q_delta
        if k_delta is not None:
            k_weight.data += k_delta

    return attn


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

    ablation_configs = [
        ("baseline", None, None),
        ("ablate_Q335", [335], None),
        ("ablate_Q270", [270], None),
        ("ablate_K206", None, [206]),
        ("ablate_K224", None, [224]),
        ("ablate_Q335_K206", [335], [206]),
        ("ablate_Q270_K224", [270], [224]),
    ]

    # Collect mean offset profiles per ablation condition
    offset_profiles: dict[str, np.ndarray] = {}
    for name, _, _ in ablation_configs:
        offset_profiles[name] = np.zeros((n_heads, MAX_OFFSET))

    n_positions = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= N_SAMPLES:
                break

            input_ids = batch[task_config.column_name][:, :seq_len].to(device)
            T = input_ids.shape[1]

            for name, ablate_q, ablate_k in ablation_configs:
                attn = get_attention_with_ablation(
                    target_model, model, input_ids,
                    ablate_q_indices=ablate_q,
                    ablate_k_indices=ablate_k,
                )

                for h in range(n_heads):
                    for d in range(MAX_OFFSET):
                        diag = torch.diagonal(attn[h], offset=-d)
                        offset_profiles[name][h, d] += diag[MAX_OFFSET:].sum().item()

            n_positions += T - MAX_OFFSET

            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i+1}/{N_SAMPLES}")

    # Normalize
    for name in offset_profiles:
        offset_profiles[name] /= n_positions

    # Compute differences from baseline
    baseline = offset_profiles["baseline"]
    diffs = {name: prof - baseline for name, prof in offset_profiles.items() if name != "baseline"}

    # --- Plot 1: Offset profiles baseline vs each ablation ---
    fig, axes = plt.subplots(len(diffs), n_heads, figsize=(3.5 * n_heads, 3 * len(diffs)))

    for row_idx, (name, diff) in enumerate(diffs.items()):
        for h in range(n_heads):
            ax = axes[row_idx, h]
            ax.bar(range(MAX_OFFSET), diff[h], color="tab:red", alpha=0.7)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_title(f"H{h}" if row_idx == 0 else "", fontweight="bold")
            if h == 0:
                ax.set_ylabel(name.replace("ablate_", "−"), fontsize=8)
            if row_idx == len(diffs) - 1:
                ax.set_xlabel("Offset")

    fig.suptitle(
        f"Layer {LAYER}: Causal effect of ablating components on attention offset profiles\n"
        f"(Change in mean attention vs baseline, {N_SAMPLES} samples)",
        fontweight="bold",
    )
    fig.tight_layout()
    path = OUT_DIR / "causal_ablation_offset_diffs.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"\nSaved {path}")

    # --- Plot 2: Summary bar chart — total absolute change per head per ablation ---
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n_heads)
    width = 0.12

    for idx, (name, diff) in enumerate(diffs.items()):
        total_abs = np.abs(diff[:, :6]).sum(axis=1)  # sum over offsets 0-5
        short = name.replace("ablate_", "−")
        ax.bar(x + idx * width, total_abs, width, label=short)

    ax.set_xlabel("Head")
    ax.set_ylabel("Total |Δattention| (offsets 0-5)")
    ax.set_title(f"Layer {LAYER}: Which heads are most affected by each ablation?")
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.legend(fontsize=7)

    fig.tight_layout()
    path = OUT_DIR / "causal_ablation_per_head_impact.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")

    # --- Log numerical summary ---
    logger.info("\n=== Causal ablation: offset profile changes ===")
    for name, diff in diffs.items():
        logger.info(f"\n  {name}:")
        for h in range(n_heads):
            top_changes = sorted(
                [(d, diff[h, d]) for d in range(MAX_OFFSET)],
                key=lambda x: abs(x[1]), reverse=True
            )[:3]
            changes_str = ", ".join(f"off={d}:{v:+.4f}" for d, v in top_changes)
            logger.info(f"    H{h}: {changes_str}")

    # --- Plot 3: Baseline vs ablated profiles for H0 and H5 ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for col, h in enumerate([0, 5]):
        ax_abs = axes[0, col]
        ax_abs.plot(range(MAX_OFFSET), baseline[h], "k-", linewidth=2, label="Baseline")
        for name, ablate_q, ablate_k in ablation_configs[1:5]:  # single ablations
            short = name.replace("ablate_", "−")
            ax_abs.plot(range(MAX_OFFSET), offset_profiles[name][h], "--", linewidth=1.2,
                        label=short, alpha=0.8)
        ax_abs.set_title(f"H{h}: Absolute attention profiles", fontweight="bold")
        ax_abs.set_ylabel("Mean attention")
        ax_abs.legend(fontsize=7)

        ax_diff = axes[1, col]
        for name, diff in list(diffs.items())[:4]:  # single ablations
            short = name.replace("ablate_", "−")
            ax_diff.plot(range(MAX_OFFSET), diff[h], "-", linewidth=1.5, label=short)
        ax_diff.axhline(0, color="black", linewidth=0.5)
        ax_diff.set_title(f"H{h}: Change from baseline", fontweight="bold")
        ax_diff.set_xlabel("Offset")
        ax_diff.set_ylabel("Δ attention")
        ax_diff.legend(fontsize=7)

    fig.suptitle(
        f"Layer {LAYER}: Causal ablation effect on H0 and H5 (the modulated heads)",
        fontweight="bold",
    )
    fig.tight_layout()
    path = OUT_DIR / "causal_ablation_h0_h5_detail.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


if __name__ == "__main__":
    main()
