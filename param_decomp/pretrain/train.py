"""
Unified training script for multiple model families (Llama, GPT-2).

This script is adapted from https://github.com/goodfire-ai/simple_stories_train

Usage:
```bash
python -m param_decomp.pretrain.train [CONFIG.yaml]
```
- CONFIG.yaml contains the training config. If not provided, a default config is used.

To run on multiple GPUs:
```bash
torchrun --standalone --nproc_per_node=N -m param_decomp.pretrain.train ...
```
where N is the number of GPUs.
"""

import math
import os
import time
import warnings
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import fire
import numpy as np
import torch
import torch._inductor.config as torch_inductor_config
import torch.distributed as dist
import torch.nn as nn
import wandb
import yaml
from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from param_decomp.base_config import BaseConfig
from param_decomp.data import DatasetConfig, create_data_loader
from param_decomp.log import logger
from param_decomp.pretrain.models import MODEL_CLASSES, ModelConfig
from param_decomp.settings import PARAM_DECOMP_OUT_DIR
from param_decomp.utils.distributed_utils import DistributedState, log0
from param_decomp.utils.run_utils import ExecutionStamp


def is_checkpoint_step(step: int) -> bool:
    """Return True if step is a checkpoint step (powers of 2 up to 1000, then multiples of 1000)."""
    return (0 < step < 1000 and (step & (step - 1)) == 0) or step % 1000 == 0


def save_configs(
    save_dir: Path,
    config_dict: dict[str, Any],
    model_config_dict: dict[str, Any],
) -> None:
    """Save training and model configs to YAML files."""
    config_file = save_dir / "final_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_dict, f)
    log0(f"Saved config to {config_file}")
    model_config_file = save_dir / "model_config.yaml"
    with open(model_config_file, "w") as f:
        yaml.dump(model_config_dict, f)
    log0(f"Saved model config to {model_config_file}")

    if config_dict.get("wandb_project"):
        wandb.save(str(config_file), policy="now", base_path=save_dir)
        log0(f"Saved config to wandb from {str(config_file)}")
        wandb.save(str(model_config_file), policy="now", base_path=save_dir)
        log0(f"Saved model config to wandb from {str(model_config_file)}")


def save_model(
    save_dir: Path, model: nn.Module, step: int, wandb_project: str | None = None
) -> None:
    """Save model checkpoint and optionally upload to W&B."""
    state_dict = model.state_dict()
    # Remove DDP prefixes if present
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model_file = save_dir / f"model_step_{step}.pt"
    torch.save(state_dict, model_file)
    log0(f"Saved model to {model_file}")

    if wandb_project is not None:
        wandb.save(str(model_file), policy="now", base_path=save_dir)
        log0(f"Saved model to wandb: {str(model_file)}")


def log_metrics(step: int, metrics: dict[str, Any]) -> None:
    """Log metrics to W&B."""
    wandb.log(metrics, step=step)


def log_generations(step: int, generations: list[list[str]]) -> None:
    """Log generation samples to W&B."""
    wandb.log(
        {
            "generation_tables": wandb.Table(
                data=generations,
                columns=["step", "generated text"],
            )
        },
        step=step,
    )


def load_config[T: BaseModel](
    config_path_or_obj: Path | str | T | None,
    config_model: type[T],
) -> T:
    """Load the config of class `config_model`, either from YAML file, existing config object, or None."""
    if isinstance(config_path_or_obj, config_model):
        return config_path_or_obj

    config_dict = {}
    if isinstance(config_path_or_obj, str):
        config_path_or_obj = Path(config_path_or_obj)

    if config_path_or_obj is not None:
        assert isinstance(config_path_or_obj, Path), (
            f"invalid config type {type(config_path_or_obj)}"
        )
        assert config_path_or_obj.suffix == ".yaml", f"Config file {config_path_or_obj} not .yaml."
        assert Path(config_path_or_obj).exists(), f"Config file {config_path_or_obj} doesn't exist."
        with open(config_path_or_obj) as f:
            config_dict = yaml.safe_load(f)

    return config_model(**config_dict)


class Config(BaseConfig):
    wandb_project: str | None = Field(
        None, description="WandB project name. If None, will not use WandB."
    )
    train_dataset_config: DatasetConfig = Field(..., description="Dataset config for training")
    val_dataset_config: DatasetConfig = Field(..., description="Dataset config for validation")
    output_dir: Path = Field(
        PARAM_DECOMP_OUT_DIR / "target_models",
        description="Directory to write logs and checkpoints",
    )
    model: ModelConfig = Field(..., description="Model configuration")
    batch_size: PositiveInt = Field(
        ..., description="Total batch size (divided across DDP processes)"
    )
    num_iterations: PositiveInt = Field(..., description="Number of training steps")
    inference_only: bool = Field(False, description="If True, don't update gradients")
    learning_rate: PositiveFloat = Field(..., description="Learning rate")
    warmup_iters: NonNegativeInt = Field(
        ..., description="Number of iterations to warmup the learning rate"
    )
    learning_rate_decay_frac: PositiveFloat = Field(
        ..., ge=0, le=1, description="Fraction of lr to decay to. 0 decays to 0, 1 doesn't decay"
    )
    weight_decay: NonNegativeFloat = Field(..., description="Weight decay")
    grad_clip: NonNegativeFloat | None = Field(..., description="Maximum gradient magnitude")
    val_loss_every: NonNegativeInt = Field(
        ..., description="Every how many steps to evaluate val loss?"
    )
    val_max_steps: NonNegativeInt = Field(
        ..., description="Max number of batches to use for validation"
    )
    train_log_every: NonNegativeInt = Field(100, description="How often to log train loss?")
    sample_every: NonNegativeInt = Field(..., description="How often to sample from the model?")
    tensorcores: bool = Field(True, description="Use TensorCores?")
    device: str | None = Field(None, description="Device to use. If None, will autodetect.")
    compile: bool = Field(True, description="Compile the model?")
    dtype: Literal["float32", "float16", "bfloat16"] = Field(..., description="Data type")
    zero_stage: Literal[0, 1, 2, 3] = Field(
        0, description="Zero redundancy optimizer stage (0/1/2/3)"
    )
    intermediate_checkpoints: bool = Field(
        ..., description="Save intermediate checkpoints (done at steps 0, 1, 2, 4, 8, ...)?"
    )
    from_pretrained: str | Path | None = Field(
        None, description="Path to a wandb string or a local path to a checkpoint to finetune from"
    )

    DEPRECATED_CONFIG_KEYS: ClassVar[list[str]] = ["flash_attention"]

    @model_validator(mode="before")
    @classmethod
    def handle_deprecated_config_keys(cls, config_dict: dict[str, Any]) -> dict[str, Any]:
        for key in list(config_dict.keys()):
            if key in cls.DEPRECATED_CONFIG_KEYS:
                log0(
                    f"{key} is deprecated in the top-level config (use model config instead). Removing."
                )
                del config_dict[key]
        return config_dict


def main(config_path_or_obj: Path | str | Config | None = None) -> None:
    log0(f"Running pytorch {torch.__version__}")
    load_dotenv(override=True)
    config = load_config(config_path_or_obj, config_model=Config)

    T = config.train_dataset_config.n_ctx - 1  # Training sequence length (positions to train on)

    # set up DDP (distributed data parallel). torchrun sets this env variable
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        zero_stage = config.zero_stage
        dist_state = DistributedState(
            rank=ddp_rank, world_size=ddp_world_size, local_rank=ddp_local_rank, backend="nccl"
        )
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        zero_stage = 0
        ddp_world_size = 1
        master_process = True
        dist_state = None
        if config.device:
            device = config.device
        else:
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
    logger.info(f"using device: {device}")

    # Calculate per-process batch size from total batch size
    assert config.batch_size % ddp_world_size == 0, (
        f"batch_size ({config.batch_size}) must be divisible by ddp_world_size ({ddp_world_size})"
    )
    B = config.batch_size // ddp_world_size

    device_type = "cuda" if "cuda" in device else "cpu"

    # dtype context
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[config.dtype]
    ctx = (
        torch.amp.autocast(device_type=device_type, dtype=ptdtype)  # type: ignore
        if device_type == "cuda"
        else nullcontext()
    )

    # rng / reproducibility
    torch.manual_seed(45)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(45)

    # TF32
    if config.tensorcores:
        torch.set_float32_matmul_precision("high")

    # Instantiate model using discriminated union config
    model_cls = MODEL_CLASSES[config.model.model_type]
    model: nn.Module = model_cls(config.model)

    # Load pretrained weights
    if config.from_pretrained is not None:
        assert hasattr(model_cls, "from_pretrained"), (
            f"Model {config.model.model_type} does not support from_pretrained"
        )
        pretrained_model = model_cls.from_pretrained(config.from_pretrained)  # type: ignore[attr-defined]
        model.load_state_dict(pretrained_model.state_dict())
    model.to(device)

    model.train()
    model.to(device)
    if config.compile:
        if device_type == "cpu":
            warnings.warn(
                "compile may not be compatible with cpu, use `--compile=False` if issues",
                stacklevel=1,
            )
        if hasattr(torch_inductor_config, "coordinate_descent_tuning"):
            torch_inductor_config.coordinate_descent_tuning = True
        log0("compiling the model...")
        model = cast(nn.Module, torch.compile(model))  # type: ignore[reportArgumentType]

    train_loader, train_tokenizer = create_data_loader(
        dataset_config=config.train_dataset_config,
        batch_size=B,
        buffer_size=1000,
        global_seed=0,
        dist_state=dist_state,
    )
    train_iter = iter(train_loader)

    val_loader, _ = create_data_loader(
        dataset_config=config.val_dataset_config,
        batch_size=B,
        buffer_size=1000,
        global_seed=0,
        dist_state=None,  # Don't split validation data - all ranks evaluate same data
    )

    # logging
    run_id: str | None = None
    if config.wandb_project is not None and master_process:
        execution_stamp = ExecutionStamp.create(run_type="train", create_snapshot=False)
        run_id = execution_stamp.run_id
        wandb.init(
            id=run_id,
            project=config.wandb_project,
            config=config.model_dump(mode="json"),
        )

    # DDP wrap
    raw_model: nn.Module = model
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
        raw_model = model.module

    # optimizer
    optimizer = raw_model.configure_optimizers(  # pyright: ignore[reportCallIssue]
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(0.9, 0.95),
        device_type=device,
        zero_stage=zero_stage,
    )

    # lr schedule
    def get_lr(it: int) -> float:
        min_lr = config.learning_rate * config.learning_rate_decay_frac
        if it < config.warmup_iters:
            return config.learning_rate * (it + 1) / config.warmup_iters
        if it > config.num_iterations:
            return min_lr
        decay_ratio = (it - config.warmup_iters) / (config.num_iterations - config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (config.learning_rate - min_lr)

    # IO dirs
    logfile = None
    checkpoints_dir = None
    output_dir = None
    if config.output_dir and master_process:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path(config.output_dir) / f"{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        logfile = output_dir / "main.log"
        with open(logfile, "w") as f:
            pass
        save_configs(
            output_dir,
            config_dict=config.model_dump(mode="json"),
            model_config_dict=config.model.model_dump(mode="json"),
        )
        # Save tokenizer to output_dir alongside configs and upload to W&B if enabled
        tokenizer_file = output_dir / "tokenizer.json"
        train_tokenizer.save_pretrained(str(output_dir))  # pyright: ignore[reportAttributeAccessIssue]
        log0(f"Saved tokenizer to {output_dir}")
        if config.wandb_project is not None and master_process:
            wandb.save(str(tokenizer_file), policy="now", base_path=output_dir)
            log0(f"Saved tokenizer to wandb from {str(tokenizer_file)}")
        checkpoints_dir = output_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        if config.intermediate_checkpoints:
            save_model(checkpoints_dir, raw_model, step=0, wandb_project=config.wandb_project)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    timings: list[float] = []
    generations: list[list[Any]] = []
    # For ETA calculation
    training_start_time = time.time()

    for step in range(1, config.num_iterations + 1):
        last_step = step == config.num_iterations

        # validation
        if config.val_loss_every > 0 and (step % config.val_loss_every == 0 or last_step):
            model.eval()
            val_loader_iter = iter(val_loader)
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(config.val_max_steps):
                    try:
                        bat = next(val_loader_iter)["input_ids"].to(torch.long)
                    except StopIteration:
                        break
                    x = bat.view(B, T + 1)[:, :-1]
                    y = bat.view(B, T + 1)[:, 1:]
                    x, y = x.to(device), y.to(device)
                    _, loss = model(x, y, return_logits=False)
                    val_loss += float(loss.item()) if loss is not None else 0.0
                val_loss /= config.val_max_steps
            if config.wandb_project is not None and master_process:
                log_metrics(step, {"val_loss": val_loss})
            log0(f"val loss {val_loss}")
            if master_process and logfile is not None:
                with open(logfile, "a") as f:
                    f.write(f"s:{step} tel:{val_loss}\n")

        # sample generations
        if config.sample_every > 0 and (step % config.sample_every == 0 or last_step):
            model.eval()
            # Get EOS token ID - HuggingFace tokenizers have eos_token_id attribute
            eos_id = train_tokenizer.eos_token_id  # pyright: ignore[reportAttributeAccessIssue]
            start_ids = [eos_id] if eos_id is not None else [0]
            xg = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
            max_new_tokens = 32
            temperature = 1.0
            top_k = 40
            yg = cast(Any, raw_model).generate(
                xg, max_new_tokens, temperature=temperature, top_k=top_k
            )
            if master_process:
                log0("---------------")
                log0(train_tokenizer.decode(yg[0].tolist()))  # pyright: ignore[reportAttributeAccessIssue]
                log0("---------------")
                if config.wandb_project is not None and master_process:
                    decoded = train_tokenizer.decode(yg[0].tolist())  # pyright: ignore[reportAttributeAccessIssue]
                    generations.append([step, decoded])
                    log_generations(step, generations)

        if last_step:
            break

        # training
        model.train()

        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        try:
            bat = next(train_iter)["input_ids"].to(torch.long)
        except StopIteration:
            log0("\n\n\nDepleted train_loader, resetting for next epoch\n\n\n")
            train_iter = iter(train_loader)
            bat = next(train_iter)["input_ids"].to(torch.long)

        x = bat.view(B, T + 1)[:, :-1]
        y = bat.view(B, T + 1)[:, 1:]
        x, y = x.to(device), y.to(device)
        with ctx:
            _, loss = model(x, y, return_logits=False)
        if not config.inference_only:
            loss.backward()  # type: ignore[arg-type]
        if ddp:
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)  # type: ignore[arg-type]
        lossf_value = float(loss.detach().item())  # type: ignore[union-attr]
        norm = None
        if config.grad_clip is not None:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()

        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

        t1 = time.time()
        if step % config.train_log_every == 0:
            tokens_per_second = ddp_world_size * B * T / (t1 - t0)
            norm_str = f"norm {norm:.4f}" if norm is not None else ""
            # Calculate ETA
            elapsed = t1 - training_start_time
            steps_done = step
            steps_remaining = config.num_iterations - step
            eta_seconds = (elapsed / steps_done) * steps_remaining if steps_done > 0 else 0
            eta_h, eta_rem = divmod(int(eta_seconds), 3600)
            eta_m, eta_s = divmod(eta_rem, 60)
            eta_str = f"{eta_h}h {eta_m:02d}m" if eta_h > 0 else f"{eta_m}m {eta_s:02d}s"
            log0(
                f"step {step:4d}/{config.num_iterations} | loss {lossf_value:.6f} | {norm_str} | "
                f"lr {lr:.2e} | {(t1 - t0) * 1000:.2f}ms | {tokens_per_second:.0f} tok/s | ETA {eta_str}"
            )
        if config.wandb_project is not None and master_process:
            log_metrics(step, {"train_loss": lossf_value, "lr": lr})
        if master_process and logfile is not None:
            with open(logfile, "a") as f:
                f.write(f"step:{step} loss:{lossf_value}\n")

        if (
            checkpoints_dir is not None
            and master_process
            and (
                (config.intermediate_checkpoints and is_checkpoint_step(step))
                or step == config.num_iterations - 1
            )
        ):
            save_model(checkpoints_dir, raw_model, step=step, wandb_project=config.wandb_project)

        if step > 1 and (step > config.num_iterations - 20):
            timings.append(t1 - t0)

    timings = timings[-20:]
    log0(f"final {len(timings)} iters avg: {np.mean(timings) * 1000:.3f}ms")
    log0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    if ddp:
        destroy_process_group()

    if config.wandb_project is not None and master_process:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
