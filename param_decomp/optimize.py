"""The PD optimization loop.

This module exposes one entrypoint, :func:`optimize`. It is the sole way to run PD from the
core library — there is no driver-mediated wrapper, no `RunConfig`, no registry. Callers
build their target model, dataloaders, loss objective (via `PDConfig.loss_metrics`), and
list of eval `Metric` instances themselves, then hand them in.
"""

import gc
from collections import defaultdict
from collections.abc import Generator
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.parallel
from datasets import IterableDataset
from jaxtyping import Float
from torch import Tensor, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from param_decomp.configs import PDConfig, RuntimeConfig
from param_decomp.distributed import (
    avg_metrics_across_ranks,
    get_distributed_state,
    is_main_process,
    seed_per_rank,
    sync_across_processes,
)
from param_decomp.eval import collect_metric_outputs
from param_decomp.identity_insertion import insert_identity_operations_
from param_decomp.log import logger
from param_decomp.metrics.base import LossMetricConfig, Metric
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.faithfulness_loss import faithfulness_loss
from param_decomp.metrics.loss_metrics import LOSS_METRIC_CLASSES
from param_decomp.metrics.persistent_pgd import validate_pgd_scope
from param_decomp.models.batch_and_loss_fns import (
    ReconstructionLoss,
    RunBatch,
    move_batch_to_device,
)
from param_decomp.models.component_model import ComponentModel, OutputWithCache
from param_decomp.module_info import expand_module_patterns
from param_decomp.run_sink import RunSink
from param_decomp.schedule import get_scheduled_value
from param_decomp.torch_helpers import (
    bf16_autocast,
    combine_nonoverlapping_dicts,
    runtime_cast,
)


def loop_dataloader[T](dl: DataLoader[T]) -> Generator[T]:
    """Loop over a dataloader, resetting the iterator when it is exhausted.

    Ensures that each epoch gets different data, even when using a distributed sampler.
    """
    epoch = 0
    dl_iter = iter(dl)
    while True:
        try:
            yield next(dl_iter)
        except StopIteration:
            logger.warning("Dataloader exhausted, resetting iterator.")
            epoch += 1
            if isinstance(dl.sampler, DistributedSampler):
                dl.sampler.set_epoch(epoch)
            if isinstance(dl.dataset, IterableDataset):
                dl.dataset.set_epoch(epoch)
            dl_iter = iter(dl)
            yield next(dl_iter)


def _grad_norms_dict(
    component_model: ComponentModel, device: torch.device | str
) -> dict[str, float]:
    """Per-parameter gradient norms for component params and CI fn params."""
    out: dict[str, float] = {}

    comp_grad_norm_sq_sum: Float[Tensor, ""] = torch.zeros((), device=device)
    for target_module_path, component in component_model.components.items():
        for local_param_name, local_param in component.named_parameters():
            param_grad = runtime_cast(Tensor, local_param.grad)
            param_grad_sum_sq = param_grad.pow(2).sum()
            key = f"components/{target_module_path}.{local_param_name}"
            out[key] = param_grad_sum_sq.sqrt().item()
            comp_grad_norm_sq_sum += param_grad_sum_sq

    ci_fn_grad_norm_sq_sum: Float[Tensor, ""] = torch.zeros((), device=device)
    for local_param_name, local_param in component_model.ci_fn.named_parameters():
        ci_fn_grad = runtime_cast(Tensor, local_param.grad)
        ci_fn_grad_sum_sq = ci_fn_grad.pow(2).sum()
        key = f"ci_fns/{local_param_name}"
        assert key not in out, f"Key {key} already exists in grad norms log"
        out[key] = ci_fn_grad_sum_sq.sqrt().item()
        ci_fn_grad_norm_sq_sum += ci_fn_grad_sum_sq

    out["summary/components"] = comp_grad_norm_sq_sum.sqrt().item()
    out["summary/ci_fns"] = ci_fn_grad_norm_sq_sum.sqrt().item()
    out["summary/total"] = (comp_grad_norm_sq_sum + ci_fn_grad_norm_sq_sum).sqrt().item()
    return out


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


def _build_metric_args(
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


def _build_loss_instances(
    config: PDConfig,
    component_model: ComponentModel,
    device: str,
) -> dict[str, Metric[Any]]:
    """Instantiate one loss-metric instance per `pd_config.loss_metrics` entry."""
    instances: dict[str, Metric[Any]] = {}
    for cfg in config.loss_metrics:
        assert cfg.type not in instances, f"duplicate loss metric {cfg.type!r}"
        cls = LOSS_METRIC_CLASSES[cfg.type]
        m = cls(cfg)
        m.bind(model=component_model, device=device)
        instances[cfg.type] = m
    return instances


def compute_losses(
    loss_instances: dict[str, Metric[Any]],
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
    train_loader: DataLoader[Any],
    eval_loader: DataLoader[Any],
    *,
    run_batch: RunBatch,
    reconstruction_loss: ReconstructionLoss,
    pd_config: PDConfig,
    runtime_config: RuntimeConfig,
    sink: RunSink,
    eval_metrics: list[Metric[Any]],
    device: str,
) -> None:
    """Run the PD optimization loop.

    Pure trainer: takes the target model, the two dataloaders, the run-batch / reconstruction
    callables, the two configs, the sink, and the eval metrics. No `RunConfig`, no driver, no
    YAML, no wandb-init responsibility — `sink` owns all of that.

    `eval_metrics` is a list of caller-instantiated `Metric` objects. They are bound to the
    `ComponentModel` and device inside this function via `Metric.bind(...)`; the caller does
    not have to construct them with model/device.

    All ranks call this function; `sink` is automatically a no-op on non-main ranks.
    """
    dist_state = get_distributed_state()
    validate_pgd_scope(
        pd_config.loss_metrics,
        batch_size=pd_config.batch_size,
        world_size=dist_state.world_size if dist_state is not None else 1,
    )

    train_iterator = loop_dataloader(train_loader)
    eval_iterator = loop_dataloader(eval_loader)

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

    loss_instances = _build_loss_instances(pd_config, component_model, device)
    for m in eval_metrics:
        m.bind(model=component_model, device=device)

    # Loss metrics are auto-evaluated alongside dedicated eval metrics. We disallow duplicate
    # registry names across the two pools because `evaluate()` keys metrics by class name.
    eval_only_instances: dict[str, Metric[Any]] = {type(m).__name__: m for m in eval_metrics}
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
            ctx = _build_metric_args(
                next(train_iterator),
                step=step,
                is_eval=False,
                device=device,
                wrapped_model=wrapped_model,
                component_model=component_model,
                config=pd_config,
                reconstruction_loss=reconstruction_loss,
            )
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
        if sink.should_log_train(step):
            avg_metrics = avg_metrics_across_ranks(batch_log_data, device=device)
            batch_log_data = cast(defaultdict[str, float], avg_metrics)

            grad_norms = _grad_norms_dict(component_model, device)
            combine_nonoverlapping_dicts(
                batch_log_data, {f"grad_norms/{k}": v for k, v in grad_norms.items()}
            )
            batch_log_data["schedules/lr/components"] = components_lr
            batch_log_data["schedules/lr/ci_fn"] = ci_fn_lr

            sink.console(
                f"--- Step {step} ---",
                f"LR[components]: {components_lr:.6f}",
                f"LR[ci_fn]: {ci_fn_lr:.6f}",
                *(f"train/{name}: {value:.15f}" for name, value in batch_log_data.items()),
            )
            sink.log(batch_log_data, step=step, section="train")

        # --- Evaluation --- #
        if sink.should_eval(step):
            with torch.no_grad(), bf16_autocast(enabled=runtime_config.autocast_bf16):
                slow_step = sink.should_run_slow_eval(step)
                active = [m for m in all_instances.values() if not (m.slow and not slow_step)]
                for m in active:
                    m.reset()
                for _ in range(sink.n_eval_steps):
                    ctx = _build_metric_args(
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
                sink.log(metrics, step=step, section="eval")

                del metrics
                torch.cuda.empty_cache()
                gc.collect()

        # --- Saving Checkpoint --- #
        if sink.should_save(step, total_steps=pd_config.steps):
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
