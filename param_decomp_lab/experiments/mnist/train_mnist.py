"""Pretrain an MNIST MLP target for the memorization PD experiment.

YAML-driven (`pd-mnist-pretrain path/to/train_config.yaml`). Trains the same architecture
and recipe across conditions; only `label_noise_p` (and, for the size ladder,
`n_train_examples`) change. Saves the checkpoint, the train config, and the exact
memorized-set spec (subsample indices + possibly-corrupted labels) so the decomposition
loader can reproduce the training distribution.
"""

from pathlib import Path

import fire
import torch
import wandb
from jaxtyping import Float, Int
from torch import Tensor

from param_decomp.log import logger
from param_decomp.schedule import get_scheduled_value
from param_decomp_lab.distributed import get_device
from param_decomp_lab.experiments.mnist.data import (
    MnistMemorizedDataset,
    build_memorized_set,
    load_raw_mnist,
)
from param_decomp_lab.experiments.mnist.models import (
    MNIST_CHECKPOINT_FILENAME,
    MNIST_MEMSET_FILENAME,
    MNIST_TRAIN_CONFIG_FILENAME,
    MnistMLP,
    MnistTrainConfig,
)
from param_decomp_lab.infra.run_files import ExecutionStamp, save_file
from param_decomp_lab.infra.wandb import init_wandb
from param_decomp_lab.seed import set_seed

# Optional silico live-telemetry integration (no-op when silico is absent).
try:
    from silico.slurm_telemetry import register_wandb_url, report_progress
except Exception:  # pragma: no cover - silico not installed

    def register_wandb_url() -> None:  # type: ignore[misc]
        return None

    def report_progress(**_: object) -> None:  # type: ignore[misc]
        return None


@torch.no_grad()
def accuracy(
    model: MnistMLP,
    images: Float[Tensor, "n 784"],
    labels: Int[Tensor, " n"],
    chunk: int = 10000,
) -> float:
    model.eval()
    correct = 0
    for start in range(0, images.shape[0], chunk):
        logits = model(images[start : start + chunk])
        correct += (logits.argmax(-1) == labels[start : start + chunk]).sum().item()
    model.train()
    return correct / images.shape[0]


def run_train(config: MnistTrainConfig, device: str) -> dict[str, float]:
    stamp = ExecutionStamp.create(run_type="train", create_snapshot=False)
    out_dir = stamp.out_dir
    logger.info(f"Run ID: {stamp.run_id}  out_dir: {out_dir}")

    model_cfg = config.mnist_model_config
    run_name = (
        f"mnist_mlp_p{config.label_noise_p}_N{config.n_train_examples or 60000}"
        f"_w{model_cfg.width}_L{model_cfg.n_hidden_layers}_seed{config.seed}"
    )

    if config.wandb_project:
        init_wandb(
            project=config.wandb_project,
            run_id=stamp.run_id,
            config=config,
            name=run_name,
            tags=[
                "mnist-target",
                f"p{config.label_noise_p}",
                f"N{config.n_train_examples or 60000}",
            ],
        )
        register_wandb_url()
        print(f"WANDB_URL {wandb.run.get_url()}", flush=True)  # type: ignore[union-attr]

    # --- Data: build the (possibly corrupted, possibly subsampled) memorized set. ---
    train_x_full, train_y_full, test_x, test_y = load_raw_mnist(
        config.data_dir, normalize=config.normalize
    )
    indices, mem_labels = build_memorized_set(
        train_y_full,
        label_noise_p=config.label_noise_p,
        label_noise_seed=config.label_noise_seed,
        n_train_examples=config.n_train_examples,
        subsample_seed=config.subsample_seed,
        n_classes=model_cfg.n_classes,
    )
    mem_x = train_x_full[indices].to(device)
    mem_y = mem_labels.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)
    # Clean (true) labels of the memorized examples, for a "memorization gap" readout.
    clean_mem_y = train_y_full[indices].to(device)
    frac_corrupted = (mem_y != clean_mem_y).float().mean().item()
    logger.info(
        f"Memorized set: N={mem_x.shape[0]}, fraction labels != true = {frac_corrupted:.3f}"
    )

    # --- Save config + memorized-set spec up front (so resume/inspection works). ---
    save_file(config.model_dump(mode="json"), out_dir / MNIST_TRAIN_CONFIG_FILENAME)
    save_file(
        {"indices": indices.cpu(), "labels": mem_labels.cpu()},
        out_dir / MNIST_MEMSET_FILENAME,
    )

    model = MnistMLP(model_cfg).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr_schedule.start_val,
        weight_decay=config.weight_decay,
    )

    loader = MnistMemorizedDataset(
        mem_x, mem_y, batch_size=config.batch_size, device=device, shuffle=True, seed=config.seed
    )
    data_iter = iter(loader)

    for step in range(config.steps):
        lr = get_scheduled_value(step, config.steps, config.lr_schedule)
        for g in optimizer.param_groups:
            g["lr"] = lr

        batch_x, batch_y = next(data_iter)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = torch.nn.functional.cross_entropy(logits, batch_y)
        loss.backward()
        optimizer.step()

        if step % config.print_freq == 0:
            logger.info(f"step {step}: loss={loss.item():.4e} lr={lr:.2e}")
            if config.wandb_project:
                wandb.log({"train/loss": loss.item(), "train/lr": lr}, step=step)
        if step % config.eval_every == 0 or step == config.steps - 1:
            train_acc = accuracy(model, mem_x, mem_y)
            test_acc = accuracy(model, test_x, test_y)
            logger.info(f"step {step}: train_acc={train_acc:.4f} test_acc={test_acc:.4f}")
            if config.wandb_project:
                wandb.log({"eval/train_acc": train_acc, "eval/test_acc": test_acc}, step=step)
        report_progress(step=step + 1, total_steps=config.steps, phase="train")

    train_acc = accuracy(model, mem_x, mem_y)
    test_acc = accuracy(model, test_x, test_y)
    save_file(model.state_dict(), out_dir / MNIST_CHECKPOINT_FILENAME)
    logger.info(f"FINAL: train_acc={train_acc:.4f} test_acc={test_acc:.4f}  saved to {out_dir}")

    metrics = {
        "final_train_acc": train_acc,
        "final_test_acc": test_acc,
        "frac_corrupted": frac_corrupted,
        "n_train": float(mem_x.shape[0]),
    }
    if config.wandb_project:
        wandb.log({f"final/{k}": v for k, v in metrics.items()}, step=config.steps - 1)
        wandb.summary.update(
            {"run_dir": str(out_dir), **{f"final/{k}": v for k, v in metrics.items()}}
        )
        wandb.finish()
    return metrics


def main(config_path: str | Path) -> None:
    config = MnistTrainConfig.from_file(config_path)
    set_seed(config.seed)
    device = get_device()
    logger.info(f"Using device: {device}")
    run_train(config, device)


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
