"""Run PD on a model.

Two entry points:

- ``optimize(target, train_loader, eval_loader, *, pd_config, logging_config,
  runtime_config, device, run)`` — **the notebook entry point.** Pure
  trainer: takes everything explicitly, doesn't know about RunConfig /
  drivers / YAML. ``run`` is a ``PDRun`` (use ``PDRun.silent()`` /
  ``PDRun.local(out_dir)`` / ``PDRun.with_wandb(out_dir, ...)``).
- ``run_pd(run_cfg, *, device, ...)`` — **the driver-mediated wrapper.**
  Materializes runtime inputs via ``materialize_run``, builds a
  ``PDRun.for_run`` (writes ``run_config.yaml``, inits wandb from ``run_cfg``
  fields), then hands off to ``optimize``. Used by ``pd-run`` / ``_worker.py``.

``materialize_run`` is the composition root: a standalone function that
turns a ``RunConfig`` into the ``(target, train_loader, eval_loader)`` tuple
``optimize`` expects. Driver-mediated callers go through ``materialize_run``;
notebook callers construct those three objects themselves and skip the
indirection.
"""

import gc
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.parallel
from jaxtyping import Float
from torch import Tensor, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from param_decomp.configs import LoggingConfig, PDConfig, RuntimeConfig
from param_decomp.driver_path import load_driver
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
    move_batch_to_device,
)
from param_decomp.models.component_model import ComponentModel, OutputWithCache
from param_decomp.pd_run import PDRun
from param_decomp.run import RunConfig
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
)
from param_decomp.utils.logging_utils import get_grad_norms_dict
from param_decomp.utils.module_utils import expand_module_patterns


def materialize_run(
    run_cfg: RunConfig,
    *,
    device: str,
    dist_state: DistributedState | None = None,
) -> tuple[PDTarget, DataLoader[Any], DataLoader[Any]]:
    """Compose the ``(target, train_loader, eval_loader)`` tuple ``optimize`` needs.

    Resolves the driver from ``run_cfg.driver_path``, validates that ``run_cfg``
    is the driver's expected subtype, then calls ``build_target`` /
    ``build_train_loader`` / ``build_eval_loader``. Driver-mediated callers use
    this; notebook callers construct those three objects themselves and skip
    the indirection.
    """
    driver = load_driver(run_cfg.driver_path)
    assert isinstance(run_cfg, driver.config_type), (
        f"RunConfig has type {type(run_cfg).__name__}, "
        f"expected {driver.config_type.__name__} from driver {run_cfg.driver_path}"
    )
    return (
        driver.build_target(run_cfg),
        driver.build_train_loader(run_cfg, device=device, dist_state=dist_state),
        driver.build_eval_loader(run_cfg, device=device, dist_state=dist_state),
    )


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
    torch.cuda.empty_cache()
    gc.collect()


def _build_ctx(
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
    # The wrapped_model(...) call here is what registers DDP gradient hooks for this step.
    # Required even if no metric uses the DDP wrapper directly.
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
    target: PDTarget,
    train_loader: DataLoader[Any],
    eval_loader: DataLoader[Any],
    *,
    pd_config: PDConfig,
    logging_config: LoggingConfig,
    runtime_config: RuntimeConfig,
    device: str,
    run: PDRun,
) -> None:
    """Run the optimization loop. The notebook / script entry point.

    Pure trainer: takes a ``PDTarget`` plus dataloaders plus the three configs.
    No ``RunConfig``, no driver, no YAML, no wandb-init responsibility.

    ``run`` is a ``PDRun`` carrying the output channels (local files +
    optional wandb + checkpoints). Use ``PDRun.silent()`` for no-persistence
    runs, ``PDRun.local(out_dir)`` for local files, or
    ``PDRun.with_wandb(out_dir, project=...)`` for wandb.

    All ranks call this function; ``run`` is automatically a no-op on
    non-main ranks.
    """
    _dist_state = get_distributed_state()
    pd_config.validate_pgd_scope(
        world_size=_dist_state.world_size if _dist_state is not None else 1
    )

    train_iterator = loop_dataloader(train_loader)
    eval_iterator = loop_dataloader(eval_loader)

    if run.out_dir is not None:
        logger.info(f"Train+eval logs saved to directory: {run.out_dir}")

    target_model = target.model
    run_batch = target.run_batch
    reconstruction_loss = target.reconstruction_loss
    tied_weights = target.tied_weights

    if pd_config.identity_module_info is not None:
        insert_identity_operations_(
            target_model,
            identity_module_info=pd_config.identity_module_info,
        )

    target_model.requires_grad_(False)
    module_path_info = expand_module_patterns(target_model, pd_config.all_module_info)

    model = ComponentModel(
        target_model=target_model,
        run_batch=run_batch,
        module_path_info=module_path_info,
        ci_config=pd_config.ci_config,
        sigmoid_type=pd_config.sigmoid_type,
    )
    model.to(device)

    # Diverge global RNG per rank so stochastic masks/sources differ across DP workers.
    seed_per_rank(pd_config.seed)

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
        lr=pd_config.components_optimizer.lr_schedule.start_val,
        betas=pd_config.components_optimizer.betas,
        weight_decay=pd_config.components_optimizer.weight_decay,
    )
    ci_fn_optimizer = optim.AdamW(
        ci_fn_params,
        lr=pd_config.ci_fn_optimizer.lr_schedule.start_val,
        betas=pd_config.ci_fn_optimizer.betas,
        weight_decay=pd_config.ci_fn_optimizer.weight_decay,
    )

    if pd_config.faithfulness_warmup_steps > 0:
        run_faithfulness_warmup(component_model, component_params, pd_config)

    loss_instances, eval_only_instances = _build_metric_instances(
        pd_config, logging_config, component_model, device
    )
    all_instances = {**loss_instances, **eval_only_instances}

    for step in tqdm(range(pd_config.steps + 1), ncols=0, disable=not is_main_process()):
        components_optimizer.zero_grad()
        ci_fn_optimizer.zero_grad()

        components_lr = get_scheduled_value(
            step=step,
            total_steps=pd_config.steps,
            config=pd_config.components_optimizer.lr_schedule,
        )
        ci_fn_lr = get_scheduled_value(
            step=step, total_steps=pd_config.steps, config=pd_config.ci_fn_optimizer.lr_schedule
        )
        for group in components_optimizer.param_groups:
            group["lr"] = components_lr
        for group in ci_fn_optimizer.param_groups:
            group["lr"] = ci_fn_lr

        batch_log_data: defaultdict[str, float] = defaultdict(float)

        build_ctx = partial(
            _build_ctx,
            step=step,
            device=device,
            wrapped_model=wrapped_model,
            component_model=component_model,
            config=pd_config,
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
            batch_log_data[f"loss/{type(loss_instances[metric_name]).__name__}"] = loss_val.item()
        batch_log_data["loss/total"] = total_loss.item()

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
                batch_log_data, {f"grad_norms/{k}": v for k, v in grad_norms.items()}
            )
            batch_log_data["schedules/lr/components"] = components_lr
            batch_log_data["schedules/lr/ci_fn"] = ci_fn_lr

            run.console(
                f"--- Step {step} ---",
                f"LR[components]: {components_lr:.6f}",
                f"LR[ci_fn]: {ci_fn_lr:.6f}",
                *(f"train/{name}: {value:.15f}" for name, value in batch_log_data.items()),
            )
            run.log(batch_log_data, step=step, section="train")

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

                run.console(*(f"eval/{k}: {v}" for k, v in metrics.items()))
                run.log(metrics, step=step, section="eval")

                del metrics
                torch.cuda.empty_cache()
                gc.collect()

        # --- Saving Checkpoint --- #
        if (
            logging_config.save_freq is not None
            and step % logging_config.save_freq == 0
            and step > 0
        ) or step == pd_config.steps:
            run.checkpoint(component_model.state_dict(), step=step)

        # Skip gradient step at the very last step (last step is just for plotting/logging).
        if step != pd_config.steps:
            sync_across_processes()
            if pd_config.components_optimizer.grad_clip_norm is not None:
                clip_grad_norm_(component_params, pd_config.components_optimizer.grad_clip_norm)
            if pd_config.ci_fn_optimizer.grad_clip_norm is not None:
                clip_grad_norm_(ci_fn_params, pd_config.ci_fn_optimizer.grad_clip_norm)
            components_optimizer.step()
            ci_fn_optimizer.step()

    if is_main_process():
        logger.info("Finished training loop.")


def run_pd(
    run_cfg: RunConfig,
    *,
    device: str,
    dist_state: DistributedState | None = None,
    wandb_project: str | None = None,
    launch_id: str | None = None,
) -> Path | None:
    """Driver-mediated PD run. Composition root for ``pd-run`` / ``_worker.py``.

    Steps:
    1. Materialize ``(target, train_loader, eval_loader)`` from the driver
       (``materialize_run``).
    2. Build a ``PDRun.for_run`` — creates ``PARAM_DECOMP_OUT_DIR/
       decompositions/<run_id>/``, writes ``run_config.yaml``, inits wandb if
       ``wandb_project`` is set.
    3. Hand off to ``optimize`` with the run handle.
    4. ``run.finish()`` for wandb cleanup.

    For notebook / script use, call ``optimize(...)`` directly with a
    ``PDRun`` of your choosing (``PDRun.local`` / ``PDRun.with_wandb`` /
    ``PDRun.silent``).

    ``wandb_project`` is a deploy-time parameter (which W&B account/project to
    log to), not part of the reproducible ``RunConfig``. ``None`` disables W&B.
    ``launch_id`` is the SLURM launch identifier shared by every run in a
    sweep — used as a W&B tag so the sweep can be queried as a group.

    All ranks call this function. Returns the output directory on the main
    process and ``None`` on other ranks.
    """
    target, train_loader, eval_loader = materialize_run(
        run_cfg, device=device, dist_state=dist_state
    )
    # target.model lands on `device` via ComponentModel.to(device) inside optimize().

    pd_run = PDRun.for_run(run_cfg, wandb_project=wandb_project, launch_id=launch_id)
    try:
        optimize(
            target=target,
            train_loader=train_loader,
            eval_loader=eval_loader,
            pd_config=run_cfg.pd,
            logging_config=run_cfg.logging,
            runtime_config=run_cfg.runtime,
            device=device,
            run=pd_run,
        )
    finally:
        pd_run.finish()

    return pd_run.out_dir
