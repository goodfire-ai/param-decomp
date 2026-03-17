"""Plot attention pattern changes from component ablation across heads.

Compares four conditions at layer 1:
  - Target model baseline
  - SPD model baseline (all-ones masks)
  - Full component ablation (q/k components zeroed at t/t-1)
  - Per-head component ablation (restricted to specific heads)

Produces plots:
  - Raw attention distributions at query position t, averaged over samples
  - Attention differences (ablated - SPD baseline)
  - Fractional attention change ((ablated - baseline) / baseline per offset)
  - Per-sample versions of all three (up to 10 individual samples)

Usage:
    python -m spd.scripts.attention_ablation_experiment.plot_attn_pattern_diffs \
        wandb:goodfire/spd/runs/s-275c8f21 \
        --components "h.1.attn.q_proj:279,h.1.attn.k_proj:177" \
        --restrict_to_heads L1H1 \
        --n_samples 1024
"""

import random
from pathlib import Path

import fire
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Int
from torch import Tensor

from spd.configs import LMTaskConfig
from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from spd.scripts.attention_ablation_experiment.attention_ablation_experiment import (
    _build_component_head_ablations,
    _build_deterministic_masks_multi_pos,
    _build_prev_token_component_positions,
    _infer_layer_from_components,
    parse_components,
    parse_heads,
    patched_attention_forward,
)
from spd.spd_types import ModelPath
from spd.utils.wandb_utils import parse_wandb_run_path

matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).parent


def plot_attn_pattern_diffs(
    wandb_path: ModelPath,
    components: str,
    restrict_to_heads: str,
    n_samples: int = 1024,
    max_offset_show: int = 20,
    max_offset_frac: int = 20,
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    parsed_components = parse_components(components)
    parsed_restrict_heads = parse_heads(restrict_to_heads)
    layer = _infer_layer_from_components(parsed_components)

    _entity, _project, run_id = parse_wandb_run_path(str(wandb_path))
    run_info = SPDRunInfo.from_path(wandb_path)
    config = run_info.config

    spd_model = ComponentModel.from_run_info(run_info)
    spd_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spd_model = spd_model.to(device)
    target_model = spd_model.target_model
    assert isinstance(target_model, LlamaSimpleMLP)
    for block in target_model._h:
        block.attn.flash_attention = False

    seq_len = target_model.config.n_ctx
    task_config = config.task_config
    assert isinstance(task_config, LMTaskConfig)
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
    loader, tokenizer = create_data_loader(
        dataset_config=dataset_config, batch_size=1, buffer_size=1000
    )

    out_dir = SCRIPT_DIR / "out" / run_id / "attn_pattern_diffs"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_heads = target_model.config.n_head
    conditions = ["target_baseline", "spd_baseline", "full_comp", "perhead_comp"]
    accum: dict[str, dict[int, dict[int, list[float]]]] = {
        c: {h: {o: [] for o in range(max_offset_show + 1)} for h in range(n_heads)}
        for c in conditions
    }

    restrict_label = "_".join(f"L{ly}H{hd}" for ly, hd in parsed_restrict_heads)
    logger.section(f"Attention pattern diffs (n={n_samples}, restrict={restrict_label})")

    sample_t_values: list[int] = []
    sample_token_labels: list[list[str]] = []
    decode = tokenizer.decode  # pyright: ignore[reportAttributeAccessIssue]

    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            if i >= n_samples:
                break
            input_ids: Int[Tensor, "batch pos"] = batch_data[task_config.column_name][
                :, :seq_len
            ].to(device)

            sample_seq_len = input_ids.shape[1]
            rng = random.Random(i)
            t = rng.randint(max_offset_show, min(sample_seq_len, 128) - 1)
            sample_t_values.append(t)

            if i < 10:
                labels = []
                for o in range(max_offset_show + 1):
                    pos = t - o
                    if pos >= 0:
                        tok_str = decode(input_ids[0, pos].item()).replace("\n", "\\n")
                    else:
                        tok_str = ""
                    labels.append(tok_str)
                sample_token_labels.append(labels)

            bs = (input_ids.shape[0], input_ids.shape[1])
            cp = _build_prev_token_component_positions(parsed_components, t)
            baseline_masks, full_ablated_masks = _build_deterministic_masks_multi_pos(
                spd_model, cp, bs, input_ids.device
            )
            comp_head_abls = _build_component_head_ablations(
                spd_model, parsed_components, parsed_restrict_heads, t
            )

            with patched_attention_forward(target_model) as d:
                target_model(input_ids)
            target_pat = d.patterns

            with patched_attention_forward(target_model) as d:
                spd_model(input_ids, mask_infos=baseline_masks)
            spd_pat = d.patterns

            with patched_attention_forward(target_model) as d:
                spd_model(input_ids, mask_infos=full_ablated_masks)
            full_pat = d.patterns

            with patched_attention_forward(
                target_model, component_head_ablations=comp_head_abls
            ) as d:
                spd_model(input_ids, mask_infos=baseline_masks)
            perhead_pat = d.patterns

            pats = {
                "target_baseline": target_pat,
                "spd_baseline": spd_pat,
                "full_comp": full_pat,
                "perhead_comp": perhead_pat,
            }
            for cond, pat in pats.items():
                for h in range(n_heads):
                    for o in range(max_offset_show + 1):
                        kp = t - o
                        if kp >= 0:
                            accum[cond][h][o].append(pat[layer][h, t, kp].item())

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{n_samples}")

    offsets = list(range(max_offset_show + 1))

    # --- Plot 1: Raw attention values ---
    styles = {
        "target_baseline": ("k", "-", 1.5, "Target baseline"),
        "spd_baseline": ("b", "-", 1.5, "SPD baseline"),
        "full_comp": ("r", "-", 1.5, "Full comp ablation"),
        "perhead_comp": ("g", "--", 1.5, f"Per-head comp ({restrict_label})"),
    }

    all_means = [
        np.mean(accum[c][h][o]) for c in conditions for h in range(n_heads) for o in offsets
    ]
    raw_ymax = max(all_means) * 1.1

    fig, axes = plt.subplots(n_heads, 1, figsize=(14, n_heads * 1.8), squeeze=False)
    for h in range(n_heads):
        ax = axes[h, 0]
        for cond, (color, ls, lw, label) in styles.items():
            means = [np.mean(accum[cond][h][o]) for o in offsets]
            stds = [np.std(accum[cond][h][o]) for o in offsets]
            ax.plot(offsets, means, color=color, linestyle=ls, linewidth=lw, label=label)
            ax.fill_between(
                offsets,
                [m - s for m, s in zip(means, stds, strict=True)],
                [m + s for m, s in zip(means, stds, strict=True)],
                alpha=0.1,
                color=color,
            )
        ax.set_ylim(-0.02, raw_ymax)
        ax.set_ylabel(f"H{h}", fontsize=10, fontweight="bold")
        ax.set_xlim(max_offset_show + 0.5, -0.5)
        ax.set_xticks(offsets)
        if h == 0:
            ax.legend(fontsize=10, loc="upper left")
        if h == n_heads - 1:
            ax.set_xlabel("Offset from query position", fontsize=9)

    fig.suptitle(
        f"Layer {layer} mean attention at query pos t (n={n_samples})",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    path = out_dir / f"attn_dist_mean_n{n_samples}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")

    # --- Plot 2: Differences from SPD baseline ---
    diff_styles = {
        "full_comp": ("r", "-", 1.5, "Full comp - SPD baseline"),
        "perhead_comp": ("g", "--", 1.5, "Per-head comp - SPD baseline"),
    }

    all_diff_means = []
    for cond in ["full_comp", "perhead_comp"]:
        for h in range(n_heads):
            for o in offsets:
                diffs = [
                    a - b
                    for a, b in zip(accum[cond][h][o], accum["spd_baseline"][h][o], strict=True)
                ]
                all_diff_means.append(np.mean(diffs))
    diff_ymin = min(all_diff_means) * 1.15
    diff_ymax = max(max(all_diff_means) * 1.15, 0.05)

    fig, axes = plt.subplots(n_heads, 1, figsize=(14, n_heads * 1.8), squeeze=False)
    for h in range(n_heads):
        ax = axes[h, 0]
        for cond, (color, ls, lw, label) in diff_styles.items():
            diffs_by_offset = []
            for o in offsets:
                sample_diffs = [
                    a - b
                    for a, b in zip(accum[cond][h][o], accum["spd_baseline"][h][o], strict=True)
                ]
                diffs_by_offset.append(sample_diffs)
            means = [np.mean(d) for d in diffs_by_offset]
            stds = [np.std(d) for d in diffs_by_offset]
            ax.plot(offsets, means, color=color, linestyle=ls, linewidth=lw, label=label)
            ax.fill_between(
                offsets,
                [m - s for m, s in zip(means, stds, strict=True)],
                [m + s for m, s in zip(means, stds, strict=True)],
                alpha=0.15,
                color=color,
            )
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_ylim(diff_ymin, diff_ymax)
        ax.set_ylabel(f"H{h}", fontsize=10, fontweight="bold")
        ax.set_xlim(max_offset_show + 0.5, -0.5)
        ax.set_xticks(offsets)
        if h == 0:
            ax.legend(fontsize=10, loc="upper left")
        if h == n_heads - 1:
            ax.set_xlabel("Offset from query position", fontsize=9)

    fig.suptitle(
        f"Layer {layer} attention change from ablation (n={n_samples})",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    path = out_dir / f"attn_diff_mean_n{n_samples}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")

    # --- Plot 3: Fractional change from SPD baseline ---
    # Normalize by mean baseline attention at each offset across all heads (more stable)
    frac_offsets = list(range(max_offset_frac + 1))
    frac_styles = {
        "full_comp": ("r", "-", 1.5, "Full comp fractional change"),
        "perhead_comp": ("g", "--", 1.5, f"Per-head comp fractional change ({restrict_label})"),
    }

    mean_baseline_by_offset = {
        o: np.mean([v for h in range(n_heads) for v in accum["spd_baseline"][h][o]])
        for o in frac_offsets
    }

    fig, axes = plt.subplots(n_heads, 1, figsize=(14, n_heads * 1.8), squeeze=False)
    for h in range(n_heads):
        ax = axes[h, 0]
        for cond, (color, ls, lw, label) in frac_styles.items():
            fracs_by_offset = []
            for o in frac_offsets:
                norm = mean_baseline_by_offset[o]
                sample_fracs = [
                    (a - b) / norm if norm > 1e-8 else 0.0
                    for a, b in zip(accum[cond][h][o], accum["spd_baseline"][h][o], strict=True)
                ]
                fracs_by_offset.append(sample_fracs)
            means = [np.mean(f) for f in fracs_by_offset]
            ax.plot(frac_offsets, means, color=color, linestyle=ls, linewidth=lw, label=label)
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_ylabel(f"H{h}", fontsize=10, fontweight="bold")
        ax.set_xlim(max_offset_frac + 0.5, -0.5)
        ax.set_xticks(frac_offsets)
        if h == 0:
            ax.legend(fontsize=10, loc="upper left")
        if h == n_heads - 1:
            ax.set_xlabel("Offset from query position", fontsize=9)

    fig.suptitle(
        f"Layer {layer} fractional attention change from ablation (n={n_samples})",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    path = out_dir / f"attn_frac_mean_n{n_samples}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")

    # --- Per-sample plots (up to 10) ---
    n_individual = min(n_samples, 10)
    for si in range(n_individual):
        t = sample_t_values[si]
        tok_labels = sample_token_labels[si]

        # Raw attention values
        sample_ymax = (
            max(
                accum[c][h][o][si]
                for c in conditions
                for h in range(n_heads)
                for o in offsets
                if si < len(accum[c][h][o])
            )
            * 1.1
        )

        fig, axes = plt.subplots(n_heads, 1, figsize=(14, n_heads * 1.8), squeeze=False)
        for h in range(n_heads):
            ax = axes[h, 0]
            for cond, (color, ls, lw, label) in styles.items():
                vals = [accum[cond][h][o][si] for o in offsets]
                ax.plot(offsets, vals, color=color, linestyle=ls, linewidth=lw, label=label)
            ax.set_ylim(-0.02, sample_ymax)
            ax.set_ylabel(f"H{h}", fontsize=10, fontweight="bold")
            ax.set_xlim(max_offset_show + 0.5, -0.5)
            ax.set_xticks(offsets)
            ax.set_xticklabels(tok_labels, fontsize=10, rotation=0, ha="center")
            if h == 0:
                ax.legend(fontsize=10, loc="upper left")
        fig.suptitle(
            f"Layer {layer} attention at query pos t={t} (sample {si})",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()
        path = out_dir / f"attn_dist_sample{si}_t{t}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Differences from SPD baseline
        fig, axes = plt.subplots(n_heads, 1, figsize=(14, n_heads * 1.8), squeeze=False)
        for h in range(n_heads):
            ax = axes[h, 0]
            for cond, (color, ls, lw, label) in diff_styles.items():
                vals = [accum[cond][h][o][si] - accum["spd_baseline"][h][o][si] for o in offsets]
                ax.plot(offsets, vals, color=color, linestyle=ls, linewidth=lw, label=label)
            ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
            ax.set_ylabel(f"H{h}", fontsize=10, fontweight="bold")
            ax.set_xlim(max_offset_show + 0.5, -0.5)
            ax.set_xticks(offsets)
            ax.set_xticklabels(tok_labels, fontsize=10, rotation=0, ha="center")
            if h == 0:
                ax.legend(fontsize=10, loc="upper left")
        fig.suptitle(
            f"Layer {layer} attention change at query pos t={t} (sample {si})",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()
        path = out_dir / f"attn_diff_sample{si}_t{t}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Fractional change from SPD baseline (limited to recent offsets)
        frac_tok_labels = tok_labels[: max_offset_frac + 1]
        fig, axes = plt.subplots(n_heads, 1, figsize=(14, n_heads * 1.8), squeeze=False)
        for h in range(n_heads):
            ax = axes[h, 0]
            for cond, (color, ls, lw, label) in frac_styles.items():
                vals = [
                    (accum[cond][h][o][si] - accum["spd_baseline"][h][o][si])
                    / mean_baseline_by_offset[o]
                    if mean_baseline_by_offset[o] > 1e-8
                    else 0.0
                    for o in frac_offsets
                ]
                ax.plot(frac_offsets, vals, color=color, linestyle=ls, linewidth=lw, label=label)
            ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
            ax.set_ylabel(f"H{h}", fontsize=10, fontweight="bold")
            ax.set_xlim(max_offset_frac + 0.5, -0.5)
            ax.set_xticks(frac_offsets)
            ax.set_xticklabels(frac_tok_labels, fontsize=10, rotation=0, ha="center")
            if h == 0:
                ax.legend(fontsize=10, loc="upper left")
        fig.suptitle(
            f"Layer {layer} fractional attention change at query pos t={t} (sample {si})",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()
        path = out_dir / f"attn_frac_sample{si}_t{t}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved sample {si} plots (t={t})")


if __name__ == "__main__":
    fire.Fire(plot_attn_pattern_diffs)
