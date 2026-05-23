"""The PD optimization loop.

This module exposes one entrypoint, :func:`optimize`. It is the sole way to run PD from the
core library — there is no driver-mediated wrapper, no `RunConfig`, no registry. Callers
build their target model, dataloaders, loss objective (via `PDConfig.loss_metrics`), and
list of eval `Metric` instances themselves, then hand them in.
"""

import gc
from collections import defaultdict
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.parallel
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from param_decomp.batch_and_loss_fns import (
    ReconstructionLoss,
    RunBatch,
    move_batch_to_device,
)
from param_decomp.component_model import ComponentModel, OutputWithCache, component_grad_norms
from param_decomp.configs import Cadence, PDConfig, RuntimeConfig
from param_decomp.decomposition_targets import (
    insert_identity_operations_,
    resolve_decomposition_targets,
)
from param_decomp.distributed import (
    avg_metrics_across_ranks,
    get_distributed_state,
    is_main_process,
    seed_all_ranks,
    seed_per_rank,
    sync_across_processes,
)
from param_decomp.faithfulness_warmup import run_faithfulness_warmup
from param_decomp.log import logger
from param_decomp.metrics.base import LossMetricConfig, Metric
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.dispatch import instantiate_loss_metrics
from param_decomp.metrics.output import collect_metric_outputs
from param_decomp.metrics.persistent_pgd_recon import validate_pgd_scope
from param_decomp.run_sink import RunSink
from param_decomp.schedule import get_scheduled_value
from param_decomp.torch_helpers import bf16_autocast, loop_dataloader


def _build_metric_context(
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
        batch=batch,
        target_out=target_model_output.output,
        pre_weight_acts=target_model_output.cache,
        ci=ci,
        weight_deltas=weight_deltas,
        step=step,
        total_steps=config.steps,
        use_delta_component=config.use_delta_component,
        sampling=config.sampling,
        n_mask_samples=config.n_mask_samples,
        reconstruction_loss=reconstruction_loss,
        is_eval=is_eval,
    )


def optimize(
    target_model: nn.Module,
    train_loader: DataLoader[Any],
    run_batch: RunBatch,
    reconstruction_loss: ReconstructionLoss,
    pd_config: PDConfig,
    runtime_config: RuntimeConfig,
    cadence: Cadence,
    sink: RunSink,
    eval_loader: DataLoader[Any],
    eval_metrics: list[Metric[Any]],
    n_eval_steps: int,
) -> None:
    """Run the PD optimization loop.

    Pure trainer: takes the target model, the train loader, the run-batch / reconstruction
    callables, the two configs, the cadence (when to emit), the sink (where output goes), and
    the eval-pass triple (`eval_loader`, `eval_metrics`, `n_eval_steps`). No `RunConfig`, no
    driver, no YAML, no wandb-init responsibility.

    `eval_metrics` is a list of caller-instantiated `Metric` objects. They are bound to the
    `ComponentModel` and device inside this function via `Metric.bind(...)`; the caller does
    not have to construct them with model/device. `n_eval_steps` is the number of eval-loader
    batches consumed on each eval tick.

    All ranks call this function; `sink` is automatically a no-op on non-main ranks.
    """
    dist_state = get_distributed_state()
    device = runtime_config.device
    validate_pgd_scope(
        pd_config.loss_metrics,
        batch_size=pd_config.batch_size,
        world_size=dist_state.world_size if dist_state is not None else 1,
    )

    train_iterator = loop_dataloader(train_loader)
    eval_iterator = loop_dataloader(eval_loader)

    if pd_config.identity_decomposition_targets is not None:
        insert_identity_operations_(
            target_model,
            identity_decomposition_targets=pd_config.identity_decomposition_targets,
        )

    target_model.requires_grad_(False)
    decomposition_targets = resolve_decomposition_targets(
        target_model, pd_config.all_decomposition_target_configs
    )

    seed_all_ranks(pd_config.seed)
    model = ComponentModel(
        target_model=target_model,
        run_batch=run_batch,
        decomposition_targets=decomposition_targets,
        ci_config=pd_config.ci_config,
        sigmoid_type=pd_config.sigmoid_type,
    )
    model.to(device)

    # Diverge global RNG per rank so stochastic masks/sources differ across DP workers.
    seed_per_rank(pd_config.seed)

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

    if pd_config.tied_weights is not None:
        for src_name, tgt_name in pd_config.tied_weights:
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

    loss_instances = instantiate_loss_metrics(pd_config, component_model, device)
    for m in eval_metrics:
        m.bind(model=component_model, device=device)

    # Loss metrics are auto-evaluated alongside dedicated eval metrics. We disallow duplicate
    # registry names across the two pools because `evaluate()` keys metrics by class name.
    eval_only_instances: dict[str, Metric[Any]] = {}
    for m in eval_metrics:
        metric_name = type(m).__name__
        assert metric_name not in eval_only_instances, f"duplicate eval metric {metric_name!r}"
        eval_only_instances[metric_name] = m
    overlap = sorted(set(loss_instances) & set(eval_only_instances))
    assert not overlap, (
        f"eval_metrics overlap with pd_config.loss_metrics: {overlap}. Loss metrics are "
        "automatically evaluated; remove the duplicates from eval_metrics."
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

        with bf16_autocast(enabled=runtime_config.autocast_bf16):
            ctx = _build_metric_context(
                next(train_iterator),
                step=step,
                is_eval=False,
                device=device,
                wrapped_model=wrapped_model,
                component_model=component_model,
                config=pd_config,
                reconstruction_loss=reconstruction_loss,
            )
            losses = {name: m.update(ctx) for name, m in loss_instances.items()}

        total_loss = torch.zeros((), device=device)
        active_loss_names: list[str] = []
        for metric_name, loss_val in losses.items():
            if loss_val is None:
                continue
            active_loss_names.append(metric_name)
            cfg = cast(LossMetricConfig, loss_instances[metric_name].cfg)
            assert cfg.coeff is not None
            total_loss = total_loss + cfg.coeff * loss_val
            batch_log_data[f"loss/{type(loss_instances[metric_name]).__name__}"] = loss_val.item()
        assert active_loss_names, (
            f"No active loss metrics returned a loss at step {step}. "
            f"Configured loss metrics: {list(loss_instances)}"
        )
        batch_log_data["loss/total"] = total_loss.item()

        for metric_name, m in loss_instances.items():
            m.before_backward(losses[metric_name])

        total_loss.backward()

        for m in loss_instances.values():
            m.after_backward()

        # --- Train Logging --- #
        if cadence.should_log_train(step):
            avg_metrics = avg_metrics_across_ranks(batch_log_data, device=device)
            batch_log_data = cast(defaultdict[str, float], avg_metrics)

            grad_norms = component_grad_norms(component_model, device)
            grad_norm_log_data = {f"grad_norms/{k}": v for k, v in grad_norms.items()}
            assert not set(batch_log_data) & set(grad_norm_log_data)
            batch_log_data.update(grad_norm_log_data)
            batch_log_data["schedules/lr/components"] = components_lr
            batch_log_data["schedules/lr/ci_fn"] = ci_fn_lr

            sink.console(
                f"--- Step {step} ---",
                f"LR[components]: {components_lr:.6f}",
                f"LR[ci_fn]: {ci_fn_lr:.6f}",
                *(f"train/{name}: {value:.15f}" for name, value in batch_log_data.items()),
            )
            sink.log({f"train/{k}": v for k, v in batch_log_data.items()}, step=step)

        # --- Evaluation --- #
        if cadence.should_eval(step):
            with torch.no_grad(), bf16_autocast(enabled=runtime_config.autocast_bf16):
                slow_step = cadence.should_run_slow_eval(step)
                active = [m for m in all_instances.values() if not (m.slow and not slow_step)]
                for m in active:
                    m.reset()
                for _ in range(n_eval_steps):
                    ctx = _build_metric_context(
                        next(eval_iterator),
                        step=step,
                        is_eval=True,
                        device=device,
                        wrapped_model=wrapped_model,
                        component_model=component_model,
                        config=pd_config,
                        reconstruction_loss=reconstruction_loss,
                    )
                    for m in active:
                        m.update(ctx)
                metrics = collect_metric_outputs(active)

                sink.console(*(f"eval/{k}: {v}" for k, v in metrics.items()))
                sink.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)

                del metrics
                torch.cuda.empty_cache()
                gc.collect()

        # --- Saving Checkpoint --- #
        if step == pd_config.steps or cadence.should_save(step):
            sink.checkpoint(component_model.state_dict(), step=step)

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
