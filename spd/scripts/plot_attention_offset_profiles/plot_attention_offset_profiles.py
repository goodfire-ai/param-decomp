"""Plot mean attention weight and pre-softmax logit by relative offset for each head.

For each layer, produces a 3x2 grid (one subplot per head) with dual y-axes:
  - Left (blue): mean softmax attention weight at each offset τ
  - Right (red): mean pre-softmax QK logit at each offset τ

Offset τ = query_pos - key_pos, so τ=1 is the previous token.

Usage:
    python -m spd.scripts.plot_attention_offset_profiles.plot_attention_offset_profiles \
        wandb:goodfire/spd/runs/<run_id>
"""

from dataclasses import dataclass
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from spd.configs import LMTaskConfig
from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.models.component_model import SPDRunInfo
from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from spd.scripts.collect_attention_patterns import collect_attention_patterns_with_logits
from spd.spd_types import ModelPath
from spd.utils.wandb_utils import parse_wandb_run_path

SCRIPT_DIR = Path(__file__).parent
N_BATCHES = 100
BATCH_SIZE = 32
MAX_OFFSET = 16


@dataclass
class OffsetProfiles:
    """Mean and std of attention weights and logits per (layer, head, offset)."""

    attn_mean: NDArray[np.floating]  # (n_layers, n_heads, n_offsets)
    attn_std: NDArray[np.floating]  # (n_layers, n_heads, n_offsets)
    logit_mean: NDArray[np.floating]  # (n_layers, n_heads, n_offsets)
    logit_std: NDArray[np.floating]  # (n_layers, n_heads, n_offsets)


def _collect_offset_diagonals(
    model: LlamaSimpleMLP,
    input_ids: torch.Tensor,
    n_offsets: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Run one batch and return per-sequence mean at each offset.

    Returns:
        attn_vals: (n_layers, B, n_heads, n_offsets)
        logit_vals: (n_layers, B, n_heads, n_offsets)
    """
    results = collect_attention_patterns_with_logits(model, input_ids)
    n_layers = len(results)
    B = input_ids.shape[0]
    n_heads = results[0][0].shape[1]

    attn_vals = np.zeros((n_layers, B, n_heads, n_offsets))
    logit_vals = np.zeros((n_layers, B, n_heads, n_offsets))

    for layer_idx, (att, logits) in enumerate(results):
        for d in range(n_offsets):
            # att shape: (B, H, T, T). diagonal offset=-d gives (B, H, T-d)
            # mean over positions (dim 2) gives (B, H) — one value per sequence
            attn_diag = torch.diagonal(att, offset=-d, dim1=-2, dim2=-1)
            logit_diag = torch.diagonal(logits, offset=-d, dim1=-2, dim2=-1)
            attn_vals[layer_idx, :, :, d] = attn_diag.float().mean(dim=2).cpu().numpy()
            logit_vals[layer_idx, :, :, d] = logit_diag.float().mean(dim=2).cpu().numpy()

    return attn_vals, logit_vals


def _compute_offset_profiles_from_loader(
    model: LlamaSimpleMLP,
    loader: "DataLoader[dict[str, torch.Tensor]]",
    column_name: str,
    seq_len: int,
    n_batches: int,
    max_offset: int,
    device: torch.device,
) -> OffsetProfiles:
    """Collect per-sequence mean attention weight and logit at each offset from dataset.

    Std is computed across individual sequences for meaningful error bars.
    """
    n_offsets = max_offset + 1
    all_attn: list[NDArray[np.floating]] = []
    all_logit: list[NDArray[np.floating]] = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            input_ids = batch[column_name][:, :seq_len].to(device)
            attn_vals, logit_vals = _collect_offset_diagonals(model, input_ids, n_offsets)
            # (n_layers, B, n_heads, n_offsets)
            all_attn.append(attn_vals)
            all_logit.append(logit_vals)

            if (i + 1) % 25 == 0:
                logger.info(f"Processed {i + 1}/{n_batches} batches")

    assert all_attn
    # Concatenate along batch dim: (n_layers, total_seqs, n_heads, n_offsets)
    attn_cat = np.concatenate(all_attn, axis=1)
    logit_cat = np.concatenate(all_logit, axis=1)

    return OffsetProfiles(
        attn_mean=attn_cat.mean(axis=1),  # (n_layers, n_heads, n_offsets)
        attn_std=attn_cat.std(axis=1),
        logit_mean=logit_cat.mean(axis=1),
        logit_std=logit_cat.std(axis=1),
    )


def _compute_offset_profiles_random_tokens(
    model: LlamaSimpleMLP,
    n_batches: int,
    max_offset: int,
    device: torch.device,
) -> OffsetProfiles:
    """Same as dataset version but with random uniform token IDs."""
    seq_len = model.config.n_ctx
    vocab_size = model.config.vocab_size
    n_offsets = max_offset + 1
    all_attn: list[NDArray[np.floating]] = []
    all_logit: list[NDArray[np.floating]] = []

    with torch.no_grad():
        for i in range(n_batches):
            input_ids = torch.randint(0, vocab_size, (BATCH_SIZE, seq_len), device=device)
            attn_vals, logit_vals = _collect_offset_diagonals(model, input_ids, n_offsets)
            all_attn.append(attn_vals)
            all_logit.append(logit_vals)

            if (i + 1) % 25 == 0:
                logger.info(f"Processed {i + 1}/{n_batches} batches (random tokens)")

    assert all_attn
    attn_cat = np.concatenate(all_attn, axis=1)
    logit_cat = np.concatenate(all_logit, axis=1)

    return OffsetProfiles(
        attn_mean=attn_cat.mean(axis=1),
        attn_std=attn_cat.std(axis=1),
        logit_mean=logit_cat.mean(axis=1),
        logit_std=logit_cat.std(axis=1),
    )


def _plot_offset_profiles(
    attn_mean: NDArray[np.floating],
    attn_std: NDArray[np.floating],
    logit_mean: NDArray[np.floating],
    logit_std: NDArray[np.floating],
    n_heads: int,
    layer_idx: int,
    out_dir: Path,
) -> None:
    offsets = list(range(attn_mean.shape[1]))

    fig, axes = plt.subplots(3, 2, figsize=(9, 7.5), squeeze=False)
    all_left_axes = []
    all_right_axes = []

    for h in range(n_heads):
        row, col = divmod(h, 2)
        ax_left = axes[row, col]
        ax_right = ax_left.twinx()
        all_left_axes.append(ax_left)
        all_right_axes.append(ax_right)

        ax_left.plot(offsets, attn_mean[h], color="black", linewidth=1.5)
        ax_left.fill_between(
            offsets,
            attn_mean[h] - attn_std[h],
            attn_mean[h] + attn_std[h],
            alpha=0.15,
            color="gray",
        )
        ax_right.plot(offsets, logit_mean[h], color="tab:green", linewidth=1.5, linestyle="--")
        ax_right.fill_between(
            offsets,
            logit_mean[h] - logit_std[h],
            logit_mean[h] + logit_std[h],
            alpha=0.15,
            color="tab:green",
        )

        ax_left.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
        ax_left.set_title(f"H{h}", fontsize=11, fontweight="bold")
        ax_left.set_xticks(offsets)

        if h >= n_heads - 2:
            ax_left.set_xlabel("Offset")
        else:
            ax_left.set_xticklabels([])

        if col == 0:
            ax_left.set_ylabel("Mean Attention")
        if col == 1:
            ax_right.set_ylabel("Mean Pre-Softmax Logit")

    for ax in all_left_axes:
        ax.set_ylim(-0.01, 1.01)

    right_ylims = [ax.get_ylim() for ax in all_right_axes]
    right_ymin = min(lo for lo, _ in right_ylims)
    right_ymax = max(hi for _, hi in right_ylims)
    for ax in all_right_axes:
        ax.set_ylim(right_ymin, right_ymax)

    # Legend in first subplot
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color="black", linewidth=1.5, label="Attention"),
        Line2D([0], [0], color="tab:green", linewidth=1.5, linestyle="--", label="Logit"),
    ]
    axes[0, 0].legend(handles=legend_handles, fontsize=11, loc="upper right")

    fig.tight_layout(h_pad=2.0)
    path = out_dir / f"layer{layer_idx}_attention_offset_profiles.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_attention_offset_profiles(
    wandb_path: ModelPath,
    n_batches: int = N_BATCHES,
    max_offset: int = MAX_OFFSET,
) -> None:
    _entity, _project, run_id = parse_wandb_run_path(str(wandb_path))
    run_info = SPDRunInfo.from_path(wandb_path)

    out_dir = SCRIPT_DIR / "out" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    config = run_info.config
    assert config.pretrained_model_name is not None
    target_model = LlamaSimpleMLP.from_pretrained(config.pretrained_model_name)
    target_model.eval()

    for block in target_model._h:
        block.attn.flash_attention = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_model = target_model.to(device)

    n_layers = len(target_model._h)
    n_heads = target_model._h[0].attn.n_head
    seq_len = target_model.config.n_ctx
    logger.info(f"Model: {n_layers} layers, {n_heads} heads, seq_len={seq_len}")

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
    loader, _ = create_data_loader(
        dataset_config=dataset_config,
        batch_size=BATCH_SIZE,
        buffer_size=1000,
    )

    logger.info("Computing offset profiles from dataset...")
    dataset_profiles = _compute_offset_profiles_from_loader(
        target_model, loader, task_config.column_name, seq_len, n_batches, max_offset, device
    )

    logger.info("Computing offset profiles from random tokens...")
    random_profiles = _compute_offset_profiles_random_tokens(
        target_model, n_batches, max_offset, device
    )

    for label, profiles in [("dataset", dataset_profiles), ("random_tokens", random_profiles)]:
        subdir = out_dir / label
        subdir.mkdir(parents=True, exist_ok=True)
        np.save(subdir / "attn_mean.npy", profiles.attn_mean)
        np.save(subdir / "attn_std.npy", profiles.attn_std)
        np.save(subdir / "logit_mean.npy", profiles.logit_mean)
        np.save(subdir / "logit_std.npy", profiles.logit_std)

        for layer_idx in range(n_layers):
            _plot_offset_profiles(
                profiles.attn_mean[layer_idx],
                profiles.attn_std[layer_idx],
                profiles.logit_mean[layer_idx],
                profiles.logit_std[layer_idx],
                n_heads,
                layer_idx,
                subdir,
            )

    logger.info(f"All plots saved to {out_dir}")


if __name__ == "__main__":
    fire.Fire(plot_attention_offset_profiles)
