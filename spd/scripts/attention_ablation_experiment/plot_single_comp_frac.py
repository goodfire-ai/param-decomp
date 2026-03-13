"""Plot fractional attention change for individual Q and K component ablations.

For each of the top N Q components and top N K components (ranked by marginal
attention contribution at a given offset), ablates that single component and
computes the fractional attention change.

Usage:
    python -m spd.scripts.attention_ablation_experiment.plot_single_comp_frac \
        wandb:goodfire/spd/runs/s-275c8f21 \
        --layer 1 --top_n 10 --n_samples 256
"""

import random
from collections.abc import Iterable
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
from spd.harvest.repo import HarvestRepo
from spd.harvest.schemas import ComponentSummary
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from spd.scripts.attention_ablation_experiment.attention_ablation_experiment import (
    _build_deterministic_masks_multi_pos,
    patched_attention_forward,
)
from spd.spd_types import ModelPath
from spd.utils.wandb_utils import parse_wandb_run_path

matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).parent


def _load_top_components(run_id: str, layer: int, top_n: int) -> tuple[list[int], list[int]]:
    """Return top N Q and K component indices ranked by mean CI."""
    repo = HarvestRepo.open_most_recent(run_id)
    assert repo is not None, f"No harvest data found for {run_id}"
    summary = repo.get_summary()

    q_path = f"h.{layer}.attn.q_proj"
    k_path = f"h.{layer}.attn.k_proj"

    def _top_by_ci(summary: dict[str, ComponentSummary], module_path: str) -> list[int]:
        components = [
            (s.component_idx, s.mean_activations["causal_importance"])
            for s in summary.values()
            if s.layer == module_path
        ]
        components.sort(key=lambda t: t[1], reverse=True)
        return [idx for idx, _ in components[:top_n]]

    return _top_by_ci(summary, q_path), _top_by_ci(summary, k_path)


def _run_ablation_loop(
    spd_model: ComponentModel,
    target_model: LlamaSimpleMLP,
    loader: Iterable[dict[str, Tensor]],
    task_config: LMTaskConfig,
    layer: int,
    component_indices: list[int],
    proj: str,
    n_samples: int,
    max_offset_show: int,
    device: torch.device,
    k_offset: int = 1,
) -> tuple[
    dict[int, dict[int, list[float]]],
    list[dict[int, dict[int, list[float]]]],
]:
    """Run ablation for a list of components from the same projection.

    Returns (baseline_accum, comp_accums) where each is head → offset → sample values.
    """
    seq_len = target_model.config.n_ctx
    n_heads = target_model.config.n_head
    offsets = list(range(max_offset_show + 1))
    module_path = f"h.{layer}.attn.{proj}"

    baseline_accum: dict[int, dict[int, list[float]]] = {
        h: {o: [] for o in offsets} for h in range(n_heads)
    }
    comp_accums: list[dict[int, dict[int, list[float]]]] = [
        {h: {o: [] for o in offsets} for h in range(n_heads)} for _ in component_indices
    ]

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
            bs = (input_ids.shape[0], input_ids.shape[1])

            # Position: q_proj at t, k_proj at t-k_offset
            comp_pos = t if "q_proj" in proj else t - k_offset

            # Baseline (shared)
            dummy_cp = [(module_path, 0, comp_pos)]
            baseline_masks, _ = _build_deterministic_masks_multi_pos(
                spd_model, dummy_cp, bs, input_ids.device
            )

            with patched_attention_forward(target_model) as d:
                spd_model(input_ids, mask_infos=baseline_masks)
            spd_pat = d.patterns

            for h in range(n_heads):
                for o in offsets:
                    kp = t - o
                    if kp >= 0:
                        baseline_accum[h][o].append(spd_pat[layer][h, t, kp].item())

            # Ablate each component
            for ci_idx, comp_idx in enumerate(component_indices):
                cp = [(module_path, comp_idx, comp_pos)]
                _, ablated_masks = _build_deterministic_masks_multi_pos(
                    spd_model, cp, bs, input_ids.device
                )

                with patched_attention_forward(target_model) as d:
                    spd_model(input_ids, mask_infos=ablated_masks)
                abl_pat = d.patterns

                for h in range(n_heads):
                    for o in offsets:
                        kp = t - o
                        if kp >= 0:
                            comp_accums[ci_idx][h][o].append(abl_pat[layer][h, t, kp].item())

            if (i + 1) % 50 == 0:
                logger.info(f"  {proj}: processed {i + 1}/{n_samples}")

    return baseline_accum, comp_accums


def _plot_frac(
    baseline_accum: dict[int, dict[int, list[float]]],
    comp_accums: list[dict[int, dict[int, list[float]]]],
    component_indices: list[int],
    proj: str,
    layer: int,
    n_heads: int,
    max_offset_show: int,
    n_samples: int,
    out_dir: Path,
) -> None:
    offsets = list(range(max_offset_show + 1))

    mean_baseline_by_offset = {
        o: np.mean([v for h in range(n_heads) for v in baseline_accum[h][o]]) for o in offsets
    }

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(n_heads, 1, figsize=(14, n_heads * 2.2), squeeze=False)
    for h in range(n_heads):
        ax = axes[h, 0]
        for ci_idx, comp_idx in enumerate(component_indices):
            frac_means = []
            for o in offsets:
                norm = mean_baseline_by_offset[o]
                if norm > 1e-8:
                    sample_fracs = [
                        (a - b) / norm
                        for a, b in zip(
                            comp_accums[ci_idx][h][o], baseline_accum[h][o], strict=True
                        )
                    ]
                else:
                    sample_fracs = [0.0] * len(baseline_accum[h][o])
                frac_means.append(np.mean(sample_fracs))
            ax.plot(
                offsets,
                frac_means,
                color=cmap(ci_idx % 10),
                linewidth=1.2,
                label=f"C{comp_idx}",
            )
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_ylabel(f"H{h}", fontsize=10, fontweight="bold")
        ax.set_xlim(max_offset_show + 0.5, -0.5)
        ax.set_xticks(offsets)
        if h == 0:
            ax.legend(fontsize=7, loc="upper left", ncol=2)
        if h == n_heads - 1:
            ax.set_xlabel("Offset from query position", fontsize=9)

    proj_label = "Q" if "q_proj" in proj else "K"
    fig.suptitle(
        f"Layer {layer} fractional attention change per {proj_label} component"
        f" (n={n_samples}, top {len(component_indices)})",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    path = out_dir / f"frac_{proj_label.lower()}_top{len(component_indices)}_n{n_samples}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def _plot_raw_attention(
    baseline_accum: dict[int, dict[int, list[float]]],
    comp_accums: list[dict[int, dict[int, list[float]]]],
    component_indices: list[int],
    proj: str,
    layer: int,
    n_heads: int,
    max_offset_show: int,
    n_samples: int,
    out_dir: Path,
) -> None:
    offsets = list(range(max_offset_show + 1))
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(n_heads, 1, figsize=(14, n_heads * 2.2), squeeze=False)
    for h in range(n_heads):
        ax = axes[h, 0]
        # Baseline
        baseline_means = [np.mean(baseline_accum[h][o]) for o in offsets]
        ax.plot(offsets, baseline_means, color="k", linewidth=1.5, label="SPD baseline")
        # Each ablated component
        for ci_idx, comp_idx in enumerate(component_indices):
            means = [np.mean(comp_accums[ci_idx][h][o]) for o in offsets]
            ax.plot(
                offsets,
                means,
                color=cmap(ci_idx % 10),
                linewidth=1.2,
                label=f"C{comp_idx}",
            )
        ax.set_ylabel(f"H{h}", fontsize=10, fontweight="bold")
        ax.set_xlim(max_offset_show + 0.5, -0.5)
        ax.set_xticks(offsets)
        if h == 0:
            ax.legend(fontsize=7, loc="upper left", ncol=2)
        if h == n_heads - 1:
            ax.set_xlabel("Offset from query position", fontsize=9)

    proj_label = "Q" if "q_proj" in proj else "K"
    fig.suptitle(
        f"Layer {layer} attention per {proj_label} component ablation"
        f" (n={n_samples}, top {len(component_indices)})",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    path = out_dir / f"attn_{proj_label.lower()}_top{len(component_indices)}_n{n_samples}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def _plot_diff(
    baseline_accum: dict[int, dict[int, list[float]]],
    comp_accums: list[dict[int, dict[int, list[float]]]],
    component_indices: list[int],
    proj: str,
    layer: int,
    n_heads: int,
    max_offset_show: int,
    n_samples: int,
    out_dir: Path,
) -> None:
    offsets = list(range(max_offset_show + 1))
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(n_heads, 1, figsize=(14, n_heads * 2.2), squeeze=False)
    for h in range(n_heads):
        ax = axes[h, 0]
        for ci_idx, comp_idx in enumerate(component_indices):
            diff_means = []
            for o in offsets:
                sample_diffs = [
                    a - b
                    for a, b in zip(comp_accums[ci_idx][h][o], baseline_accum[h][o], strict=True)
                ]
                diff_means.append(np.mean(sample_diffs))
            ax.plot(
                offsets,
                diff_means,
                color=cmap(ci_idx % 10),
                linewidth=1.2,
                label=f"C{comp_idx}",
            )
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_ylabel(f"H{h}", fontsize=10, fontweight="bold")
        ax.set_xlim(max_offset_show + 0.5, -0.5)
        ax.set_xticks(offsets)
        if h == 0:
            ax.legend(fontsize=7, loc="upper left", ncol=2)
        if h == n_heads - 1:
            ax.set_xlabel("Offset from query position", fontsize=9)

    proj_label = "Q" if "q_proj" in proj else "K"
    fig.suptitle(
        f"Layer {layer} attention change per {proj_label} component ablation"
        f" (n={n_samples}, top {len(component_indices)})",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    path = out_dir / f"diff_{proj_label.lower()}_top{len(component_indices)}_n{n_samples}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def _plot_grid(
    baseline_accum: dict[int, dict[int, list[float]]],
    comp_accums: list[dict[int, dict[int, list[float]]]],
    component_indices: list[int],
    proj: str,
    layer: int,
    n_heads: int,
    max_offset_show: int,
    n_samples: int,
    out_dir: Path,
    plot_type: str,
) -> None:
    """3x2 grid version of the single-component ablation plots.

    plot_type: "attn" (raw attention), "diff" (ablated - baseline), "frac" (fractional change).
    """
    offsets = list(range(max_offset_show + 1))
    cmap = plt.get_cmap("tab10")
    proj_label = "Q" if "q_proj" in proj else "K"

    # Precompute cross-head mean baseline for fractional normalization
    mean_baseline_by_offset: dict[int, float] = {}
    if plot_type == "frac":
        mean_baseline_by_offset = {
            o: float(np.mean([v for h in range(n_heads) for v in baseline_accum[h][o]]))
            for o in offsets
        }

    fig, axes = plt.subplots(3, 2, figsize=(9, 7.5), squeeze=False)
    all_axes_list = []

    for h in range(n_heads):
        row, col = divmod(h, 2)
        ax = axes[row, col]
        all_axes_list.append(ax)

        if plot_type == "attn":
            # Baseline
            bl_means = [float(np.mean(baseline_accum[h][o])) for o in offsets]
            bl_stds = [float(np.std(baseline_accum[h][o])) for o in offsets]
            bl_means_arr = np.array(bl_means)
            bl_stds_arr = np.array(bl_stds)
            ax.plot(offsets, bl_means_arr, color="black", linewidth=2, label="Baseline")
            ax.fill_between(
                offsets,
                bl_means_arr - bl_stds_arr,
                bl_means_arr + bl_stds_arr,
                alpha=0.15,
                color="gray",
            )
            # Ablated components
            for ci_idx, comp_idx in enumerate(component_indices):
                means = np.array([float(np.mean(comp_accums[ci_idx][h][o])) for o in offsets])
                stds = np.array([float(np.std(comp_accums[ci_idx][h][o])) for o in offsets])
                color = cmap(ci_idx % 10)
                ax.plot(offsets, means, color=color, linewidth=1, label=f"C{comp_idx}")
                ax.fill_between(offsets, means - stds, means + stds, alpha=0.1, color=color)

        elif plot_type == "diff":
            for ci_idx, comp_idx in enumerate(component_indices):
                sample_diffs_per_offset = []
                for o in offsets:
                    diffs = [
                        a - b
                        for a, b in zip(
                            comp_accums[ci_idx][h][o], baseline_accum[h][o], strict=True
                        )
                    ]
                    sample_diffs_per_offset.append(diffs)
                means = np.array([float(np.mean(d)) for d in sample_diffs_per_offset])
                stds = np.array([float(np.std(d)) for d in sample_diffs_per_offset])
                color = cmap(ci_idx % 10)
                ax.plot(offsets, means, color=color, linewidth=1, label=f"C{comp_idx}")
                ax.fill_between(offsets, means - stds, means + stds, alpha=0.1, color=color)
            ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")

        elif plot_type == "frac":
            for ci_idx, comp_idx in enumerate(component_indices):
                sample_fracs_per_offset = []
                for o in offsets:
                    norm = mean_baseline_by_offset[o]
                    if norm > 1e-8:
                        fracs = [
                            (a - b) / norm
                            for a, b in zip(
                                comp_accums[ci_idx][h][o], baseline_accum[h][o], strict=True
                            )
                        ]
                    else:
                        fracs = [0.0] * len(baseline_accum[h][o])
                    sample_fracs_per_offset.append(fracs)
                means = np.array([float(np.mean(f)) for f in sample_fracs_per_offset])
                stds = np.array([float(np.std(f)) for f in sample_fracs_per_offset])
                color = cmap(ci_idx % 10)
                ax.plot(offsets, means, color=color, linewidth=1, label=f"C{comp_idx}")
                ax.fill_between(offsets, means - stds, means + stds, alpha=0.1, color=color)
            ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")

        ax.set_title(f"H{h}", fontsize=11, fontweight="bold")
        ax.set_xticks(offsets)

        if h >= n_heads - 2:
            ax.set_xlabel("Offset")
        else:
            ax.set_xticklabels([])

        y_labels = {"attn": "Mean Attention", "diff": "Attention Diff", "frac": "Fractional Change"}
        if col == 0:
            ax.set_ylabel(y_labels[plot_type])

    # Shared y-limits
    all_ylims = [ax.get_ylim() for ax in all_axes_list]
    ymin = min(lo for lo, _ in all_ylims)
    ymax = max(hi for _, hi in all_ylims)
    for ax in all_axes_list:
        ax.set_ylim(ymin, ymax)

    # Legend in first subplot
    axes[0, 0].legend(fontsize=7, loc="upper right", ncol=2)

    fig.tight_layout(h_pad=2.0)
    path = (
        out_dir
        / f"{plot_type}_{proj_label.lower()}_L{layer}_top{len(component_indices)}_n{n_samples}_grid.png"
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_single_comp_frac(
    wandb_path: ModelPath,
    layer: int = 1,
    top_n: int = 10,
    k_offset: int = 1,
    n_samples: int = 256,
    max_offset_show: int = 20,
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    _entity, _project, run_id = parse_wandb_run_path(str(wandb_path))
    top_q, top_k = _load_top_components(run_id, layer, top_n)
    logger.info(f"Top {top_n} Q components: {top_q}")
    logger.info(f"Top {top_n} K components: {top_k}")

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

    suffix = f"_koff{k_offset}" if k_offset != 1 else ""
    out_dir = SCRIPT_DIR / "out" / run_id / f"single_comp_frac{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Q components
    logger.section(f"Q component ablations (top {top_n})")
    loader_q, _ = create_data_loader(dataset_config=dataset_config, batch_size=1, buffer_size=1000)
    q_baseline, q_accums = _run_ablation_loop(
        spd_model,
        target_model,
        loader_q,
        task_config,
        layer,
        top_q,
        "q_proj",
        n_samples,
        max_offset_show,
        device,
    )
    n_heads = target_model.config.n_head
    plot_args_q = (
        q_baseline,
        q_accums,
        top_q,
        "q_proj",
        layer,
        n_heads,
        max_offset_show,
        n_samples,
        out_dir,
    )
    _plot_raw_attention(*plot_args_q)
    _plot_diff(*plot_args_q)
    _plot_frac(*plot_args_q)
    for pt in ("attn", "diff", "frac"):
        _plot_grid(*plot_args_q, plot_type=pt)

    # K components
    logger.section(f"K component ablations (top {top_n}, k_offset={k_offset})")
    loader_k, _ = create_data_loader(dataset_config=dataset_config, batch_size=1, buffer_size=1000)
    k_baseline, k_accums = _run_ablation_loop(
        spd_model,
        target_model,
        loader_k,
        task_config,
        layer,
        top_k,
        "k_proj",
        n_samples,
        max_offset_show,
        device,
        k_offset=k_offset,
    )
    plot_args_k = (
        k_baseline,
        k_accums,
        top_k,
        "k_proj",
        layer,
        n_heads,
        max_offset_show,
        n_samples,
        out_dir,
    )
    _plot_raw_attention(*plot_args_k)
    _plot_diff(*plot_args_k)
    _plot_frac(*plot_args_k)
    for pt in ("attn", "diff", "frac"):
        _plot_grid(*plot_args_k, plot_type=pt)


if __name__ == "__main__":
    fire.Fire(plot_single_comp_frac)
