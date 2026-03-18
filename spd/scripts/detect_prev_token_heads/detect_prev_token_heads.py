"""Detect previous-token attention heads by measuring mean attention to position i-1.

For each layer and head, computes the average attention weight from position i to
position i-1 across many data batches. Heads with a high score consistently attend
to the previous token, a key building block of induction circuits.

Usage:
    python -m spd.scripts.detect_prev_token_heads.detect_prev_token_heads \
        wandb:goodfire/spd/runs/<run_id>
"""

from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray

from spd.configs import LMTaskConfig
from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.models.component_model import SPDRunInfo
from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from spd.scripts.collect_attention_patterns import collect_attention_patterns
from spd.spd_types import ModelPath
from spd.utils.wandb_utils import parse_wandb_run_path

SCRIPT_DIR = Path(__file__).parent
N_BATCHES = 100
BATCH_SIZE = 32


def _plot_score_heatmap(
    scores: NDArray[np.floating],
    out_path: Path,
) -> None:
    n_layers, n_heads = scores.shape
    fig, ax = plt.subplots(figsize=(max(6, n_heads * 1.2), max(4, n_layers * 1.0)))

    im = ax.imshow(scores, aspect="auto", cmap="Blues", vmin=0, vmax=0.95)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="Mean attention to t-1")

    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=10)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{li}" for li in range(n_layers)], fontsize=10)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")

    for layer_idx in range(n_layers):
        for h in range(n_heads):
            val = scores[layer_idx, h]
            text_color = "white" if val > 0.65 else "black"
            ax.text(
                h, layer_idx, f"{val:.3f}", ha="center", va="center", fontsize=9, color=text_color
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def _plot_score_heatmap_combined(
    scores_left: NDArray[np.floating],
    scores_right: NDArray[np.floating],
    title_left: str,
    title_right: str,
    out_path: Path,
) -> None:
    n_layers, n_heads = scores_left.shape
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(max(6, n_heads * 1.2) * 2 + 1, max(4, n_layers * 1.0))
    )

    im = None
    for ax, scores, title in [(ax_l, scores_left, title_left), (ax_r, scores_right, title_right)]:
        im = ax.imshow(scores, aspect="auto", cmap="Blues", vmin=0, vmax=0.95)
        ax.set_xticks(range(n_heads))
        ax.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=10)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{li}" for li in range(n_layers)], fontsize=10)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(title, fontsize=12)

        for layer_idx in range(n_layers):
            for h in range(n_heads):
                val = scores[layer_idx, h]
                text_color = "white" if val > 0.65 else "black"
                ax.text(
                    h,
                    layer_idx,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                )

    fig.tight_layout()
    assert im is not None
    fig.colorbar(im, ax=[ax_l, ax_r], shrink=0.8, pad=0.01, label="Mean attention to t-1")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def _plot_attention_patterns(
    patterns: list[torch.Tensor],
    out_path: Path,
    max_pos: int = 128,
) -> None:
    """Plot grid of attention patterns (one per head, truncated to max_pos)."""
    n_layers = len(patterns)
    n_heads = patterns[0].shape[0]

    fig, axes = plt.subplots(
        n_layers,
        n_heads,
        figsize=(n_heads * 3, n_layers * 3),
        squeeze=False,
    )

    for layer_idx in range(n_layers):
        for h in range(n_heads):
            ax = axes[layer_idx, h]
            pattern = patterns[layer_idx][h, :max_pos, :max_pos].numpy()
            ax.imshow(pattern, aspect="auto", cmap="viridis", vmin=0)
            ax.set_title(f"L{layer_idx}H{h}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if h == 0:
                ax.set_ylabel(f"Layer {layer_idx}", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def detect_prev_token_heads(wandb_path: ModelPath, n_batches: int = N_BATCHES) -> None:
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

    accum_scores = np.zeros((n_layers, n_heads))
    accum_patterns = [torch.zeros(n_heads, seq_len, seq_len) for _ in range(n_layers)]
    single_patterns: list[torch.Tensor] | None = None
    n_processed = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            input_ids = batch[task_config.column_name][:, :seq_len].to(device)
            patterns = collect_attention_patterns(target_model, input_ids)

            if i == 0:
                single_patterns = [att[0].float().cpu() for att in patterns]

            for layer_idx, att in enumerate(patterns):
                diag = torch.diagonal(att, offset=-1, dim1=-2, dim2=-1)  # (batch, heads, T-1)
                accum_scores[layer_idx] += diag.float().mean(dim=(0, 2)).cpu().numpy()
                accum_patterns[layer_idx] += att.float().mean(dim=0).cpu()

            n_processed += 1
            if (i + 1) % 25 == 0:
                logger.info(f"Processed {i + 1}/{n_batches} batches")

    assert n_processed > 0
    accum_scores /= n_processed
    for layer_idx in range(n_layers):
        accum_patterns[layer_idx] /= n_processed

    logger.info(f"Previous-token scores (n={n_processed} batches):")
    for layer_idx in range(n_layers):
        for h in range(n_heads):
            score = accum_scores[layer_idx, h]
            marker = " <-- prev-token head" if score > 0.3 else ""
            logger.info(f"  L{layer_idx}H{h}: {score:.4f}{marker}")

    np.save(out_dir / "prev_token_scores.npy", accum_scores)
    _plot_score_heatmap(accum_scores, out_dir / "prev_token_scores.png")
    _plot_attention_patterns(accum_patterns, out_dir / "mean_attention_patterns.png")
    assert single_patterns is not None
    _plot_attention_patterns(single_patterns, out_dir / "single_attention_patterns.png")

    # Generate combined plot if random token scores exist
    random_scores_path = out_dir / "prev_token_scores_random_tokens.npy"
    if random_scores_path.exists():
        _plot_score_heatmap_combined(
            scores_left=np.load(random_scores_path),
            scores_right=accum_scores,
            title_left="Random token sequences",
            title_right="Dataset samples",
            out_path=out_dir / "prev_token_scores_combined.png",
        )

    logger.info(f"All plots saved to {out_dir}")


def plot_combined(run_id: str) -> None:
    """Generate combined side-by-side plot from saved .npy scores."""
    out_dir = SCRIPT_DIR / "out" / run_id
    random_scores = np.load(out_dir / "prev_token_scores_random_tokens.npy")
    dataset_scores = np.load(out_dir / "prev_token_scores.npy")
    _plot_score_heatmap_combined(
        scores_left=random_scores,
        scores_right=dataset_scores,
        title_left="Random token sequences",
        title_right="Dataset samples",
        out_path=out_dir / "prev_token_scores_combined.png",
    )


if __name__ == "__main__":
    fire.Fire(detect_prev_token_heads)
