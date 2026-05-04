from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from torch.nn import functional as F
from tqdm import tqdm, trange

from param_decomp.experiments.ih.configs import InductionHeadsTrainConfig, InductionModelConfig
from param_decomp.experiments.ih.model import InductionTransformer
from param_decomp.log import logger
from param_decomp.utils.data_utils import DatasetGeneratedDataLoader, InductionDataset
from param_decomp.utils.distributed_utils import get_device
from param_decomp.utils.general_utils import set_seed
from param_decomp.utils.run_utils import ExecutionStamp, save_file


def linear_lr(step: int, steps: int) -> float:
    return 1 - (step / steps)


def constant_lr(*_: int) -> float:
    return 1.0


def cosine_decay_lr(step: int, steps: int) -> float:
    return np.cos(0.5 * np.pi * step / (steps - 1))


def warmup_lr(
    step: int, steps: int, lr_fn: Callable[[int, int], float], warmup_steps: int
) -> float:
    if step < warmup_steps:
        return step / warmup_steps
    else:
        return lr_fn(step - warmup_steps, steps - warmup_steps)


def train(
    model: InductionTransformer,
    dataloader: DatasetGeneratedDataLoader[tuple[torch.Tensor, torch.Tensor]],
    log_wandb: bool,
    steps: int,
    print_freq: int,
    lr: float,
    weight_decay: float,
    lr_schedule: Literal["linear", "cosine", "constant"],
    lr_warmup: int | float,
) -> tuple[list[float], list[int]]:
    hooks = []

    if lr_schedule == "linear":
        lr_schedule_fn = linear_lr
    elif lr_schedule == "cosine":
        lr_schedule_fn = cosine_decay_lr
    elif lr_schedule == "constant":
        lr_schedule_fn = constant_lr

    if lr_warmup > 0:
        warmup_steps = lr_warmup if isinstance(lr_warmup, int) else int(lr_warmup * steps)
        lr_schedule_fn = partial(warmup_lr, warmup_steps=warmup_steps, lr_fn=lr_schedule_fn)

    opt = torch.optim.AdamW(list(model.parameters()), lr=lr, weight_decay=weight_decay)
    losses = []
    loss_steps = []

    data_iter = iter(dataloader)
    with trange(steps, ncols=0) as t:
        for step in t:
            step_lr = lr * lr_schedule_fn(step, steps)
            for group in opt.param_groups:
                group["lr"] = step_lr
            opt.zero_grad(set_to_none=True)
            # Labels is the token to predict with in-context learning
            batch, labels = next(data_iter)
            out = model(batch)
            loss = F.cross_entropy(
                out[:, -1, :],
                labels,
            )
            loss.backward()
            opt.step()

            if hooks:
                hook_data = dict(model=model, step=step, opt=opt, error=loss, loss=loss, lr=step_lr)
                for h in hooks:
                    h(hook_data)
            if step % print_freq == 0 or (step + 1 == steps):
                tqdm.write(f"Step {step} Loss: {loss.item()}")
                loss_steps.append(step)
                losses.append(loss.item())
                t.set_postfix(
                    loss=loss.item(),
                    lr=step_lr,
                )
                if log_wandb:
                    wandb.log({"loss": loss.item(), "lr": step_lr}, step=step)

    return losses, loss_steps


def get_model_and_dataloader(
    config: InductionHeadsTrainConfig, device: str
) -> tuple[InductionTransformer, DatasetGeneratedDataLoader[tuple[torch.Tensor, torch.Tensor]]]:
    """
    Create the model and dataloader based on the config.
    """
    model = InductionTransformer(config.ih_model_config).to(device)

    # Create the dataset and dataloader
    dataset = InductionDataset(
        seq_len=config.ih_model_config.seq_len,
        vocab_size=config.ih_model_config.vocab_size,
        device=device,
        prefix_window=config.prefix_window,
    )

    dataloader = DatasetGeneratedDataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    return model, dataloader


def run_train(config: InductionHeadsTrainConfig, device: str) -> None:
    model, dataloader = get_model_and_dataloader(config, device)

    run_name = ""
    run_name += (
        f"induction_heads_v{config.ih_model_config.vocab_size}_seq{config.ih_model_config.seq_len}"
    )
    run_name += f"_heads{config.ih_model_config.n_heads}_layers{config.ih_model_config.n_layers}"
    run_name += f"_steps{config.steps}_batch{config.batch_size}_lr{config.lr}"
    if config.ih_model_config.use_ff:
        run_name += f"_dmodel{config.ih_model_config.d_model}"
        run_name += f"_ff_fanout{config.ih_model_config.ff_fanout}"
    if config.lr_schedule:
        run_name += f"_lr_schedule_{config.lr_schedule}"
    run_name += f"use_ff_{config.ih_model_config.use_ff}"
    run_name += f"use_pos_encoding_{config.ih_model_config.use_pos_encoding}"
    run_name += f"use_layer_norm_{config.ih_model_config.use_layer_norm}"

    execution_stamp = ExecutionStamp.create(run_type="train", create_snapshot=False)
    out_dir = execution_stamp.out_dir
    logger.info(f"Run ID: {execution_stamp.run_id}")
    logger.info(f"Output directory: {out_dir}")

    if config.wandb_project:
        tags = [f"ih_{config.ih_model_config.vocab_size}vocab_{config.ih_model_config.seq_len}seq"]
        wandb.init(
            id=execution_stamp.run_id, project=config.wandb_project, name=run_name, tags=tags
        )

    # Save config
    config_path = out_dir / "ih_train_config.yaml"
    save_file(config.model_dump(mode="json"), config_path)
    if config.wandb_project:
        wandb.save(str(config_path), base_path=out_dir, policy="now")
    logger.info(f"Saved config to {config_path}")

    losses, loss_steps = train(
        model=model,
        dataloader=dataloader,
        log_wandb=False,
        steps=config.steps,
        print_freq=100,
        lr=config.lr,
        weight_decay=config.weight_decay,
        lr_schedule=config.lr_schedule,
        lr_warmup=config.lr_warmup,
    )

    plot_loss_curve(
        losses=losses,
        steps=loss_steps,
        out_dir=out_dir,
    )

    plot_attention_maps_post_training(
        model=model,
        dataloader=dataloader,
        steps=config.attention_maps_n_steps,
        out_dir=out_dir,
    )

    model_path = out_dir / "ih.pth"
    save_file(model.state_dict(), model_path)
    if config.wandb_project:
        wandb.save(str(model_path), base_path=out_dir, policy="now")
    logger.info(f"Saved model to {model_path}")


def plot_loss_curve(
    losses: list[float],
    steps: list[int],
    out_dir: Path,
) -> None:
    # Plot the loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, label="Training Loss", color="blue")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid()
    plt.savefig(out_dir / "train_loss_curve.png")
    plt.close()


def plot_attention_maps_post_training(
    model: InductionTransformer,
    dataloader: DatasetGeneratedDataLoader[tuple[torch.Tensor, torch.Tensor]],
    steps: int,
    out_dir: Path,
):
    model.eval()
    eval_attention_weights = []
    with torch.no_grad():
        for _ in range(0, steps):
            tokens, _ = next(iter(dataloader))
            attn_weights = model.get_attention_weights(tokens)
            eval_attention_weights.append(attn_weights)

        eval_attention_weights = torch.cat(eval_attention_weights, dim=0)

        # For each layer, for each head, plot the average and max attention weights
        avg_attn_weights = eval_attention_weights.mean(dim=0)
        max_attn_weights = eval_attention_weights.max(dim=0).values

        for layer_index in range(model.config.n_layers):
            for head_index in range(model.config.n_heads):
                avg_attn = avg_attn_weights[layer_index, head_index, :, :].cpu().numpy()
                max_attn = max_attn_weights[layer_index, head_index, :, :].cpu().numpy()

                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                assert isinstance(ax, np.ndarray), "Expected ax to be a numpy array of axes"
                ax[0].imshow(avg_attn, cmap="viridis", aspect="auto")
                ax[0].set_title(f"Layer {layer_index + 1}, Head {head_index + 1} - Avg Attention")
                ax[1].imshow(max_attn, cmap="viridis", aspect="auto")
                ax[1].set_title(f"Layer {layer_index + 1}, Head {head_index + 1} - Max Attention")
                plt.colorbar(ax[0].images[0], ax=ax[0])
                plt.colorbar(ax[1].images[0], ax=ax[1])
                plt.tight_layout()

                fig.savefig(out_dir / f"attention_layer{layer_index + 1}_head{head_index + 1}.png")
                plt.close(fig)


if __name__ == "__main__":
    seq_length = 64
    # The prefix window is the segment of the string the first induction
    # pair is guaranteed to land within.
    # We need 4 "spots" for the induction pair,
    # and 1 spot for the BOS token.
    prefix_window = seq_length - 3
    device = get_device()
    config = InductionHeadsTrainConfig(
        ih_model_config=InductionModelConfig(
            vocab_size=128,
            seq_len=seq_length,
            d_model=16,
            n_heads=1,
            n_layers=2,
            ff_fanout=4,
            use_ff=False,
            use_layer_norm=False,
            use_pos_encoding=True,
        ),
        wandb_project="induction_heads",
        steps=100000,
        batch_size=1024,
        lr=1e-3,
        weight_decay=0.01,
        lr_schedule="constant",
        seed=42,
        attention_maps_n_steps=100,
        prefix_window=prefix_window,
        lr_warmup=1000,
    )

    set_seed(config.seed)

    run_train(
        config=config,
        device=device,
    )
