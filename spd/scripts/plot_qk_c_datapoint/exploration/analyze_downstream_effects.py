"""Analyze downstream effects: how L2 routing affects L3 and predictions.

When L2's semantic routing shifts attention (local vs broad), this changes
what information is written to the residual stream, which L3 then reads.

Approach:
1. Run model with and without K206 (the strongest local-focus component)
2. Compare: per-token loss, L3 attention patterns, and final logit differences
3. Identify which token types are most affected by K206 ablation
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

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


def compute_component_weight(comp: LinearComponents, indices: list[int]) -> torch.Tensor:
    V_sub = comp.V[:, indices]
    U_sub = comp.U[indices]
    return torch.einsum("ic,co->oi", V_sub, U_sub)


def forward_and_get_loss(
    target_model: LlamaSimpleMLP, input_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass returning per-token loss and logits."""
    logits, _ = target_model(input_ids)  # returns (logits, loss)
    assert logits is not None
    # Per-token cross-entropy loss (shifted)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_labels.shape)
    return per_token_loss, logits


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
    ]

    # Collect per-sample metrics for each condition
    results: dict[str, dict] = {}

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

        loader, _ = create_data_loader(
            dataset_config=dataset_config, batch_size=1, buffer_size=1000
        )

        sample_losses: list[float] = []
        per_token_losses: list[np.ndarray] = []
        l3_offset_profiles: list[np.ndarray] = []

        try:
            with torch.no_grad():
                for i, batch in enumerate(loader):
                    if i >= N_SAMPLES:
                        break

                    input_ids = batch[task_config.column_name][:, :seq_len].to(device)
                    T = input_ids.shape[1]

                    loss, logits = forward_and_get_loss(target_model, input_ids)
                    sample_losses.append(loss.mean().item())
                    per_token_losses.append(loss[0].cpu().numpy())

                    # L3 attention patterns
                    results_attn = collect_attention_patterns_with_logits(target_model, input_ids)
                    if len(results_attn) > 3:  # L3 exists
                        l3_attn = torch.softmax(results_attn[3][1], dim=-1)[0]
                        l3_off = np.zeros((n_heads, MAX_OFFSET))
                        for h in range(n_heads):
                            for d in range(MAX_OFFSET):
                                diag = torch.diagonal(l3_attn[h], offset=-d)
                                l3_off[h, d] = diag[MAX_OFFSET:].float().mean().item()
                        l3_offset_profiles.append(l3_off)

                    if (i + 1) % 100 == 0:
                        logger.info(f"  {i+1}/{N_SAMPLES}")

        finally:
            if q_delta is not None:
                q_weight.data += q_delta
            if k_delta is not None:
                k_weight.data += k_delta

        results[name] = {
            "mean_loss": np.mean(sample_losses),
            "sample_losses": np.array(sample_losses),
            "per_token_losses": per_token_losses,
            "l3_offset_profiles": np.array(l3_offset_profiles) if l3_offset_profiles else None,
        }

        logger.info(f"  Mean loss: {results[name]['mean_loss']:.4f}")

    # --- Analysis ---
    baseline = results["baseline"]
    logger.info("\n\n=== Loss comparison ===")
    for name, res in results.items():
        delta = res["mean_loss"] - baseline["mean_loss"]
        logger.info(f"  {name}: loss={res['mean_loss']:.4f} (Δ={delta:+.4f})")

    # --- Plot 1: Loss comparison ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    names = list(results.keys())
    losses = [results[n]["mean_loss"] for n in names]
    colors = ["tab:green" if n == "baseline" else "tab:red" for n in names]
    ax1.bar(range(len(names)), losses, color=colors)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=20, ha="right")
    ax1.set_ylabel("Mean cross-entropy loss")
    ax1.set_title("Overall loss by ablation condition")

    # Per-sample loss scatter
    for name in names[1:]:
        ax2.scatter(
            baseline["sample_losses"],
            results[name]["sample_losses"],
            alpha=0.3, s=10, label=name,
        )
    ax2.plot([0, 8], [0, 8], "k--", linewidth=0.5)
    ax2.set_xlabel("Baseline loss")
    ax2.set_ylabel("Ablated loss")
    ax2.set_title("Per-sample loss: baseline vs ablated")
    ax2.legend(fontsize=7)

    fig.suptitle(
        f"Downstream effect: ablating L2 semantic components → prediction loss",
        fontweight="bold",
    )
    fig.tight_layout()
    path = OUT_DIR / "downstream_loss_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"\nSaved {path}")

    # --- Plot 2: L3 offset profiles ---
    if baseline["l3_offset_profiles"] is not None:
        fig, axes = plt.subplots(2, n_heads, figsize=(4 * n_heads, 7))

        baseline_l3 = baseline["l3_offset_profiles"].mean(axis=0)

        for h in range(n_heads):
            ax = axes[0, h]
            ax.plot(range(MAX_OFFSET), baseline_l3[h], "k-", linewidth=2, label="baseline")
            for name in names[1:]:
                l3 = results[name]["l3_offset_profiles"]
                if l3 is not None:
                    ax.plot(range(MAX_OFFSET), l3.mean(axis=0)[h], "--", linewidth=1.2,
                            label=name, alpha=0.8)
            ax.set_title(f"H{h}", fontweight="bold")
            if h == 0:
                ax.set_ylabel("L3 attention")
                ax.legend(fontsize=5)

            # Difference
            ax = axes[1, h]
            for name in names[1:]:
                l3 = results[name]["l3_offset_profiles"]
                if l3 is not None:
                    diff = l3.mean(axis=0)[h] - baseline_l3[h]
                    ax.plot(range(MAX_OFFSET), diff, "-", linewidth=1.5, label=name)
            ax.axhline(0, color="black", linewidth=0.5)
            if h == 0:
                ax.set_ylabel("Δ L3 attention")
            ax.set_xlabel("Offset")

        fig.suptitle(
            "Layer 3 attention: how L2 ablations propagate downstream",
            fontweight="bold",
        )
        fig.tight_layout()
        path = OUT_DIR / "downstream_l3_attention.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {path}")

    # --- Plot 3: Per-token loss difference distribution ---
    fig, axes = plt.subplots(1, len(names) - 1, figsize=(5 * (len(names) - 1), 4))
    if len(names) - 1 == 1:
        axes = [axes]

    for idx, name in enumerate(names[1:]):
        ax = axes[idx]
        # Compute per-token loss differences, averaged across samples
        baseline_losses = baseline["per_token_losses"]
        ablated_losses = results[name]["per_token_losses"]

        # Mean per-position loss difference
        min_len = min(len(bl) for bl in baseline_losses)
        bl_arr = np.array([bl[:min_len] for bl in baseline_losses])
        ab_arr = np.array([al[:min_len] for al in ablated_losses])
        diff_arr = ab_arr - bl_arr  # positive = ablation hurts

        mean_diff = diff_arr.mean(axis=0)
        ax.plot(range(len(mean_diff)), mean_diff, linewidth=1)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Token position")
        ax.set_ylabel("Δ loss (ablated − baseline)")
        ax.set_title(name, fontweight="bold")
        ax.set_xlim(0, min(100, len(mean_diff)))

    fig.suptitle("Per-position loss change from L2 ablation", fontweight="bold")
    fig.tight_layout()
    path = OUT_DIR / "downstream_per_position_loss.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


if __name__ == "__main__":
    main()
