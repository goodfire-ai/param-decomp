"""Sweep fixed source values (alpha) and measure CE loss on the validation set.

For each alpha in [0, 1], computes mask = CI + (1 - CI) * alpha for all components,
runs the model over the validation set with those masks, and records CE loss.

At alpha=0: mask = CI (CI used directly as masks)
At alpha=1: mask = 1 (all components unmasked)

Usage:
    python spd/scripts/alpha_sweep/alpha_sweep.py wandb:goodfire/spd/runs/s-55ea3f9b
    python spd/scripts/alpha_sweep/alpha_sweep.py s-55ea3f9b s-e8bde534 --n_alphas 20
"""

import argparse
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from spd.configs import LMTaskConfig, SamplingType
from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import make_mask_infos
from spd.spd_types import ModelPath


def compute_ce_at_alpha(
    model: ComponentModel,
    batches: list[Int[Tensor, "batch seq"]],
    alpha: float,
    sampling: SamplingType,
    device: str,
) -> float:
    """Compute mean CE loss over batches with mask = CI + (1 - CI) * alpha."""
    total_loss = 0.0
    total_tokens = 0

    for batch in batches:
        batch = batch.to(device)

        # Get CI values
        pre_weight_output = model(batch, cache_type="input")
        ci = model.calc_causal_importances(
            pre_weight_acts=pre_weight_output.cache, sampling=sampling
        )

        # mask = CI + (1 - CI) * alpha
        component_masks: dict[str, Float[Tensor, "... C"]] = {}
        for name, ci_vals in ci.lower_leaky.items():
            component_masks[name] = ci_vals + (1 - ci_vals) * alpha

        mask_infos = make_mask_infos(component_masks)
        logits = model(batch, mask_infos=mask_infos)

        # CE loss: next-token prediction
        flat_logits = einops.rearrange(logits, "b s v -> (b s) v")
        flat_labels = einops.rearrange(batch, "b s -> (b s)")
        # Shift: predict next token from current
        loss = F.cross_entropy(flat_logits[:-1], flat_labels[1:], reduction="sum")
        n_tokens = flat_labels[1:].numel()
        total_loss += loss.item()
        total_tokens += n_tokens

    return total_loss / total_tokens


def run_alpha_sweep(
    wandb_path: ModelPath,
    alphas: list[float],
    n_batches: int,
    device: str,
) -> tuple[str, list[float]]:
    """Run alpha sweep for a single model. Returns (run_id, ce_losses)."""
    run_info = SPDRunInfo.from_path(wandb_path)
    config = run_info.config
    run_id = str(wandb_path).split("/")[-1]

    logger.info(f"Loading model {run_id}...")
    model = ComponentModel.from_run_info(run_info).to(device)
    model.eval()

    logger.info("Creating validation data loader...")
    task_config = config.task_config
    assert isinstance(task_config, LMTaskConfig)
    eval_dataset_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=config.tokenizer_name,
        split=task_config.eval_data_split,
        n_ctx=task_config.max_seq_len,
        is_tokenized=task_config.is_tokenized,
        streaming=task_config.streaming,
        column_name=task_config.column_name,
        shuffle_each_epoch=False,
        seed=task_config.dataset_seed,
    )
    eval_loader, _tokenizer = create_data_loader(
        dataset_config=eval_dataset_config,
        batch_size=config.eval_batch_size,
        buffer_size=task_config.buffer_size,
    )

    logger.info(f"Collecting {n_batches} validation batches...")
    batches: list[Int[Tensor, "batch seq"]] = []
    for i, batch in enumerate(eval_loader):
        if i >= n_batches:
            break
        if isinstance(batch, dict):
            batch = batch["input_ids"]
        batches.append(batch)

    ce_losses: list[float] = []
    for alpha in alphas:
        with torch.no_grad():
            ce = compute_ce_at_alpha(model, batches, alpha, config.sampling, device)
        ce_losses.append(ce)
        logger.info(f"  alpha={alpha:.3f}  CE={ce:.4f}")

    return run_id, ce_losses


def plot_alpha_sweep(
    results: dict[str, list[float]],
    alphas: list[float],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for run_id, ce_losses in results.items():
        ax.plot(alphas, ce_losses, "o-", markersize=4, label=run_id)

    ax.set_xlabel(r"Source value $\alpha$")
    ax.set_ylabel("CE loss (val)")
    ax.set_title(r"Validation CE vs fixed source $\alpha$")
    if len(results) > 1:
        ax.legend()

    # Annotate endpoints
    ax.annotate(
        r"$\alpha=0$: CI masks",
        xy=(0, 0),
        xycoords="axes fraction",
        fontsize=8,
        color="grey",
    )
    ax.annotate(
        r"$\alpha=1$: unmasked",
        xy=(1, 0),
        xycoords="axes fraction",
        fontsize=8,
        color="grey",
        ha="right",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plot saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep fixed source alpha and measure CE loss")
    parser.add_argument(
        "run_ids",
        nargs="+",
        help="WandB run IDs (with or without wandb: prefix)",
    )
    parser.add_argument(
        "--n_alphas", type=int, default=11, help="Number of alpha values (default: 11)"
    )
    parser.add_argument(
        "--n_batches", type=int, default=10, help="Number of val batches (default: 10)"
    )
    parser.add_argument("--output", default="alpha_sweep.png", help="Output plot path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    alphas = list(np.linspace(0, 1, args.n_alphas))

    results: dict[str, list[float]] = {}
    for run_id in args.run_ids:
        wandb_path: ModelPath = run_id if ":" in run_id else f"wandb:goodfire/spd/runs/{run_id}"
        rid, ce_losses = run_alpha_sweep(wandb_path, alphas, args.n_batches, args.device)
        results[rid] = ce_losses

    plot_alpha_sweep(results, alphas, Path(args.output))

    # Print results
    print(f"\n{'alpha':>8}", end="")
    for rid in results:
        print(f"  {rid:>16}", end="")
    print()
    for i, alpha in enumerate(alphas):
        print(f"{alpha:>8.3f}", end="")
        for rid in results:
            print(f"  {results[rid][i]:>16.4f}", end="")
        print()


if __name__ == "__main__":
    main()
