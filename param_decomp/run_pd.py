"""Run PD on a model."""

import gc
import os
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.parallel
import wandb
from jaxtyping import Float
from PIL import Image
from torch import Tensor, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from param_decomp.configs import (
    LoggingConfig,
    PDConfig,
    PersistentPGDReconLossConfig,
    PersistentPGDReconSubsetLossConfig,
    RepeatAcrossBatchScope,
    RuntimeConfig,
)
from param_decomp.eval import evaluate
from param_decomp.identity_insertion import insert_identity_operations_
from param_decomp.log import logger
from param_decomp.metrics import METRIC_REGISTRY
from param_decomp.metrics.base import LossMetricConfig, Metric, MetricConfig
from param_decomp.metrics.builtin.faithfulness_loss import faithfulness_loss
from param_decomp.metrics.context import MetricContext
from param_decomp.models.batch_and_loss_fns import (
    PDTarget,
    ReconstructionLoss,
    RunBatch,
    move_batch_to_device,
)
from param_decomp.models.component_model import ComponentModel, OutputWithCache
from param_decomp.run import Run
from param_decomp.settings import PARAM_DECOMP_OUT_DIR
from param_decomp.utils.data_utils import loop_dataloader
from param_decomp.utils.distributed_utils import (
    DistributedState,
    avg_metrics_across_ranks,
    get_distributed_state,
    is_main_process,
    seed_per_rank,
    sync_across_processes,
)
from param_decomp.utils.general_utils import (
    bf16_autocast,
    combine_nonoverlapping_dicts,
    get_scheduled_value,
    save_pre_run_info,
)
from param_decomp.utils.logging_utils import get_grad_norms_dict, local_log
from param_decomp.utils.module_utils import expand_module_patterns
from param_decomp.utils.run_utils import save_file
from param_decomp.utils.wandb_utils import init_wandb, try_wandb


def run_faithfulness_warmup(
    component_model: ComponentModel,
    component_params: list[torch.nn.Parameter],
    config: PDConfig,
) -> None:
    """Run faithfulness warmup phase to improve initialization."""
    logger.info("Starting faithfulness warmup phase...")
    assert component_params, "component_params is empty"

    faithfulness_warmup_optimizer = optim.AdamW(
        component_params,
        lr=config.faithfulness_warmup_lr,
        weight_decay=config.faithfulness_warmup_weight_decay,
    )

    for warmup_step in range(config.faithfulness_warmup_steps):
        faithfulness_warmup_optimizer.zero_grad()
        loss = faithfulness_loss(component_model.calc_weight_deltas())
        loss.backward()
        faithfulness_warmup_optimizer.step()

        if warmup_step % 100 == 0 or warmup_step == config.faithfulness_warmup_steps - 1:
            logger.info(
                f"Faithfulness warmup step {warmup_step + 1} / {config.faithfulness_warmup_steps}; "
                f"Faithfulness loss: {loss.item():.9f}"
            )
    del faithfulness_warmup_optimizer
    gc.collect()
    torch.cuda.empty_cache()


def forward_and_build_ctx(
    batch: Any,
    *,
    step: int,
    is_eval: bool,
    device: str,
    wrapped_model: nn.Module,
    component_model: ComponentModel,
    config: PDConfig,
    reconstruction_loss: ReconstructionLoss,
) -> MetricContext:
    """Run the target forward (registering DDP grad hooks for this step) and build a MetricContext.

    The `wrapped_model(...)` call is load-bearing: it registers DDP gradient hooks for this step
    even when no metric reads through the DDP wrapper directly.
    """
    batch = move_batch_to_device(batch, device)
    target_model_output: OutputWithCache = wrapped_model(batch, cache_type="input")
    ci = component_model.calc_causal_importances(
        pre_weight_acts=target_model_output.cache,
        detach_inputs=False,
        sampling=config.sampling,
    )
    weight_deltas = component_model.calc_weight_deltas()
    return MetricContext(
        model=component_model,
        config=config,
        batch=batch,
        target_out=target_model_output.output,
        pre_weight_acts=target_model_output.cache,
        ci=ci,
        weight_deltas=weight_deltas,
        step=step,
        reconstruction_loss=reconstruction_loss,
        is_eval=is_eval,
    )


def _build_metric_instances(
    config: PDConfig,
    logging_config: LoggingConfig,
    component_model: ComponentModel,
    device: str,
) -> tuple[dict[str, Metric[MetricConfig]], dict[str, Metric[MetricConfig]]]:
    """Instantiate one metric instance per class-name key. Same model+device for both buckets."""
    loss_instances: dict[str, Metric[MetricConfig]] = {}
    for metric_name, cfg in config.loss_metrics.items():
        cls = METRIC_REGISTRY[metric_name]
        loss_instances[metric_name] = cls(cfg, model=component_model, device=device)

    eval_instances: dict[str, Metric[MetricConfig]] = {}
    for metric_name, cfg in logging_config.eval_metrics.items():
        cls = METRIC_REGISTRY[metric_name]
        eval_instances[metric_name] = cls(cfg, model=component_model, device=device)

    return loss_instances, eval_instances


def compute_losses(
    loss_instances: dict[str, Metric[MetricConfig]],
    ctx: MetricContext,
) -> dict[str, Float[Tensor, ""] | None]:
    """Compute per-metric live loss tensors for the current training step.

    Each metric's `update(ctx)` returns the per-batch scalar (a graph-attached tensor that the
    caller will backprop through), or None if the metric is gated off (e.g. PPGD before its
    `start_frac`).
    """
    return {metric_name: m.update(ctx) for metric_name, m in loss_instances.items()}


def optimize(
    target_model: nn.Module,
    config: PDConfig,
    logging_config: LoggingConfig,
    runtime_config: RuntimeConfig,
    device: str,
    train_loader: DataLoader[Any],
    eval_loader: DataLoader[Any],
    run_batch: RunBatch,
    reconstruction_loss: ReconstructionLoss,
    out_dir: Path | None,
    tied_weights: list[tuple[str, str]] | None = None,
) -> None:
    """Run the optimization loop."""
    train_iterator = loop_dataloader(train_loader)
    eval_iterator = loop_dataloader(eval_loader)

    if is_main_process():
        logger.info(f"Train+eval logs saved to directory: {out_dir}")

    if config.identity_module_info is not None:
        insert_identity_operations_(
            target_model,
            identity_module_info=config.identity_module_info,
        )

    target_model.requires_grad_(False)
    module_path_info = expand_module_patterns(target_model, config.all_module_info)

    model = ComponentModel(
        target_model=target_model,
        run_batch=run_batch,
        module_path_info=module_path_info,
        ci_config=config.ci_config,
        sigmoid_type=config.sigmoid_type,
    )
    model.to(device)

    # Diverge global RNG per rank so stochastic masks/sources differ across DP workers.
    seed_per_rank(config.seed)

    dist_state = get_distributed_state()
    wrapped_model: nn.Module = model
    component_model: ComponentModel
    if dist_state is not None:
        if dist_state.backend == "nccl":
            device_id = dist_state.local_rank
            wrapped_model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[device_id], output_device=device_id
            )
        else:
            wrapped_model = torch.nn.parallel.DistributedDataParallel(model)
        component_model = cast(ComponentModel, wrapped_model.module)
    else:
        component_model = model
    assert isinstance(component_model, ComponentModel), "component_model is not a ComponentModel"

    if tied_weights is not None:
        for src_name, tgt_name in tied_weights:
            tgt = component_model.components[tgt_name]
            src = component_model.components[src_name]
            assert tgt is not None and src is not None, (
                f"Cannot tie weights between {src_name} and {tgt_name} - one or both are None"
            )
            tgt.U.data = src.V.data.T
            tgt.V.data = src.U.data.T

    component_params: list[torch.nn.Parameter] = []
    for name in component_model.target_module_paths:
        component_params.extend(component_model.components[name].parameters())
    ci_fn_params = list(component_model.ci_fn.parameters())
    assert len(component_params) > 0, "No parameters found in components to optimize"

    components_optimizer = optim.AdamW(
        component_params,
        lr=config.components_optimizer.lr_schedule.start_val,
        betas=config.components_optimizer.betas,
        weight_decay=config.components_optimizer.weight_decay,
    )
    ci_fn_optimizer = optim.AdamW(
        ci_fn_params,
        lr=config.ci_fn_optimizer.lr_schedule.start_val,
        betas=config.ci_fn_optimizer.betas,
        weight_decay=config.ci_fn_optimizer.weight_decay,
    )

    if config.faithfulness_warmup_steps > 0:
        run_faithfulness_warmup(component_model, component_params, config)

    loss_instances, eval_only_instances = _build_metric_instances(
        config, logging_config, component_model, device
    )
    all_instances = {**loss_instances, **eval_only_instances}

    for step in tqdm(range(config.steps + 1), ncols=0, disable=not is_main_process()):
        components_optimizer.zero_grad()
        ci_fn_optimizer.zero_grad()

        components_lr = get_scheduled_value(
            step=step, total_steps=config.steps, config=config.components_optimizer.lr_schedule
        )
        ci_fn_lr = get_scheduled_value(
            step=step, total_steps=config.steps, config=config.ci_fn_optimizer.lr_schedule
        )
        for group in components_optimizer.param_groups:
            group["lr"] = components_lr
        for group in ci_fn_optimizer.param_groups:
            group["lr"] = ci_fn_lr

        batch_log_data: defaultdict[str, float] = defaultdict(float)

        build_ctx = partial(
            forward_and_build_ctx,
            step=step,
            device=device,
            wrapped_model=wrapped_model,
            component_model=component_model,
            config=config,
            reconstruction_loss=reconstruction_loss,
        )

        with bf16_autocast(enabled=runtime_config.autocast_bf16):
            ctx = build_ctx(next(train_iterator), is_eval=False)
            losses = compute_losses(loss_instances, ctx)

        total_loss = torch.zeros((), device=device)
        for metric_name, loss_val in losses.items():
            if loss_val is None:
                continue
            cfg = cast(LossMetricConfig, loss_instances[metric_name].cfg)
            assert cfg.coeff is not None
            total_loss = total_loss + cfg.coeff * loss_val
            batch_log_data[f"train/loss/{type(loss_instances[metric_name]).__name__}"] = (
                loss_val.item()
            )
        batch_log_data["train/loss/total"] = total_loss.item()

        for metric_name, m in loss_instances.items():
            m.before_backward(losses[metric_name])

        total_loss.backward()

        for m in loss_instances.values():
            m.after_backward()

        # --- Train Logging --- #
        if step % logging_config.train_log_freq == 0:
            avg_metrics = avg_metrics_across_ranks(batch_log_data, device=device)
            batch_log_data = cast(defaultdict[str, float], avg_metrics)

            grad_norms = get_grad_norms_dict(component_model, device)
            combine_nonoverlapping_dicts(
                batch_log_data, {f"train/grad_norms/{k}": v for k, v in grad_norms.items()}
            )
            batch_log_data["train/schedules/lr/components"] = components_lr
            batch_log_data["train/schedules/lr/ci_fn"] = ci_fn_lr

            if is_main_process():
                assert out_dir is not None
                tqdm.write(f"--- Step {step} ---")
                tqdm.write(f"LR[components]: {components_lr:.6f}")
                tqdm.write(f"LR[ci_fn]: {ci_fn_lr:.6f}")
                for name, value in batch_log_data.items():
                    tqdm.write(f"{name}: {value:.15f}")
                local_log(batch_log_data, step, out_dir)
                if wandb.run is not None:
                    try_wandb(wandb.log, batch_log_data, step=step)

        # --- Evaluation --- #
        if step % logging_config.eval_freq == 0:
            with torch.no_grad(), bf16_autocast(enabled=runtime_config.autocast_bf16):
                slow_step: bool = (
                    logging_config.slow_eval_on_first_step
                    if step == 0
                    else step % logging_config.slow_eval_freq == 0
                )

                metrics = evaluate(
                    instances=all_instances,
                    eval_iterator=eval_iterator,
                    ctx_builder=partial(build_ctx, is_eval=True),
                    n_eval_steps=logging_config.n_eval_steps,
                    slow_step=slow_step,
                )

                if is_main_process():
                    assert out_dir is not None
                    for k, v in metrics.items():
                        tqdm.write(f"eval/{k}: {v}")
                    local_log(metrics, step, out_dir)
                    if wandb.run is not None:
                        wandb_logs = {
                            f"eval/{k}": wandb.Image(v) if isinstance(v, Image.Image) else v
                            for k, v in metrics.items()
                        }
                        try_wandb(wandb.log, wandb_logs, step=step)

                del metrics
                gc.collect()
                torch.cuda.empty_cache()

        # --- Saving Checkpoint --- #
        if (
            (
                logging_config.save_freq is not None
                and step % logging_config.save_freq == 0
                and step > 0
            )
            or step == config.steps
        ) and is_main_process():
            assert out_dir is not None
            save_file(component_model.state_dict(), out_dir / f"model_{step}.pth")
            logger.info(f"Saved model, optimizer, and out_dir to {out_dir}")
            if wandb.run is not None:
                try_wandb(
                    wandb.save,
                    str(out_dir / f"model_{step}.pth"),
                    base_path=str(out_dir),
                    policy="now",
                )

        # Skip gradient step at the very last step (last step is just for plotting/logging).
        if step != config.steps:
            sync_across_processes()
            if config.components_optimizer.grad_clip_norm is not None:
                clip_grad_norm_(component_params, config.components_optimizer.grad_clip_norm)
            if config.ci_fn_optimizer.grad_clip_norm is not None:
                clip_grad_norm_(ci_fn_params, config.ci_fn_optimizer.grad_clip_norm)
            components_optimizer.step()
            ci_fn_optimizer.step()

    if is_main_process():
        logger.info("Finished training loop.")


def _validate_pgd_scope(config: PDConfig, dist_state: DistributedState | None) -> None:
    """Assert that persistent PGD `repeat_across_batch` divides the per-rank training batch size."""
    world_size = dist_state.world_size if dist_state is not None else 1
    assert config.batch_size % world_size == 0, (
        f"batch_size {config.batch_size} not divisible by world size {world_size}"
    )
    per_rank = config.batch_size // world_size
    for metric_name, cfg in config.loss_metrics.items():
        if isinstance(
            cfg, PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig
        ) and isinstance(cfg.scope, RepeatAcrossBatchScope):
            n = cfg.scope.n_sources
            assert per_rank % n == 0, (
                f"{metric_name}: repeat_across_batch n_sources={n} must divide "
                f"per-rank batch_size={per_rank}"
            )


def run_pd(
    config: PDConfig,
    logging_config: LoggingConfig,
    runtime_config: RuntimeConfig,
    target: PDTarget,
    train_loader: DataLoader[Any],
    eval_loader: DataLoader[Any],
    device: str,
    *,
    run: Run | None = None,
    artifacts: dict[str, Any] | None = None,
    wandb_project: str | None = None,
    wandb_tags: list[str] | None = None,
) -> Path | None:
    """Run a full PD decomposition: setup, optimize, cleanup.

    `run` is written to ``run_metadata.yaml``.  Driver-mediated callers
    (via ``experiments/runner.py``) pass a fully populated ``Run``;
    notebook callers can omit it and a minimal one is synthesized.

    ``wandb_project`` is a deploy-time parameter (which W&B account/project to log
    to), not part of the reproducible ``Run`` config. ``None`` disables W&B.

    All ranks call this function. Only the main process does wandb/logging setup.
    Returns the output directory on the main process and None on other ranks.
    """
    _validate_pgd_scope(config, get_distributed_state())

    out_dir: Path | None
    if is_main_process():
        artifacts = artifacts or {}
        if run is None:
            run = Run(
                driver_path=None,
                pd=config,
                logging=logging_config,
                runtime=runtime_config,
            )
        run_id = run.run_id
        out_dir = PARAM_DECOMP_OUT_DIR / "decompositions" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Run ID: {run_id}")
        logger.info(f"Output directory: {out_dir}")

        tags = list(wandb_tags or [])
        slurm_array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")
        if slurm_array_job_id is not None:
            tags.append(f"slurm-array-job-id_{slurm_array_job_id}")

        if wandb_project:
            init_wandb(
                wandb_project,
                run_id,
                configs={
                    "pd": config,
                    "logging": logging_config,
                    "runtime": runtime_config,
                },
                name=run.logging.wandb_run_name,
                tags=tags,
                view_meta=run.logging.view_meta,
            )

        logger.info(config)

        save_pre_run_info(
            save_to_wandb=wandb_project is not None,
            out_dir=out_dir,
            run=run,
            artifacts=artifacts,
        )
    else:
        out_dir = None

    optimize(
        target_model=target.model,
        config=config,
        logging_config=logging_config,
        runtime_config=runtime_config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        run_batch=target.run_batch,
        reconstruction_loss=target.reconstruction_loss,
        out_dir=out_dir,
        tied_weights=target.tied_weights,
    )

    if is_main_process() and wandb.run is not None:
        wandb.finish()

    return out_dir
