"""PD optimization loop.

`optimize()` is the sole core entrypoint; `EvalLoop` bundles the eval runtime objects
with their timing.
"""

import gc
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.parallel
from pydantic import PositiveInt
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
from param_decomp.metrics.dispatch import instantiate_metrics
from param_decomp.metrics.output import collect_metric_outputs
from param_decomp.metrics.persistent_pgd_recon import validate_pgd_scope
from param_decomp.run_sink import RunSink
from param_decomp.schedule import get_scheduled_value
from param_decomp.torch_helpers import bf16_autocast, loop_dataloader


@dataclass(frozen=True)
class EvalLoop:
    """Eval-loop runtime objects bundled with their timing.

    Pass `eval_loop=None` to `optimize` to skip eval entirely. When set, the trainer
    evaluates every `every` steps; on steps that are also multiples of `slow_every`,
    slow metrics fire too. `slow_every` must be a multiple of `every` — `should_run_slow_eval`
    is only checked on steps where `should_eval` already fired.

    `optimize` calls `Metric.bind(model, device)` on every entry of `metrics` before the
    loop starts.
    """

    loader: DataLoader[Any]
    metrics: list[Metric[Any]]
    n_steps: PositiveInt
    every: PositiveInt
    slow_every: PositiveInt
    slow_on_first_step: bool = True

    def __post_init__(self) -> None:
        assert self.slow_every % self.every == 0, (
            f"slow_every ({self.slow_every}) must be a multiple of every ({self.every})"
        )

    def should_eval(self, step: int) -> bool:
        return step % self.every == 0

    def should_run_slow_eval(self, step: int) -> bool:
        """Whether slow eval should fire at `step`.

        Slow eval is gated on top of `should_eval`; only call this on steps where
        `should_eval` is already true.
        """
        if step == 0:
            return self.slow_on_first_step
        return step % self.slow_every == 0


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


def tie_component_weights(
    component_model: ComponentModel, tied_weights: list[tuple[str, str]]
) -> None:
    for src_name, tgt_name in tied_weights:
        tgt = component_model.components[tgt_name]
        src = component_model.components[src_name]
        assert tgt is not None and src is not None, (
            f"Cannot tie weights between {src_name} and {tgt_name} - one or both are None"
        )
        tgt.U.data = src.V.data.T
        tgt.V.data = src.U.data.T


def optimize(
    target_model: nn.Module,
    train_loader: DataLoader[Any],
    run_batch: RunBatch,
    reconstruction_loss: ReconstructionLoss,
    pd_config: PDConfig,
    runtime_config: RuntimeConfig,
    sink: RunSink,
    cadence: Cadence,
    eval_loop: EvalLoop | None = None,
) -> None:
    """Run the PD optimization loop.

    Builds the `ComponentModel` internally, instantiates loss metrics from
    `pd_config.loss_metrics`, optionally runs a faithfulness warmup, then loops for
    `pd_config.steps + 1` training steps. Every step computes losses from
    `loss_metrics`, accumulates them weighted by their `coeff`, and backprops. Train
    logging, checkpointing, and eval each fire on their own schedule.

    Under DDP, the trainer wraps `ComponentModel` in `DistributedDataParallel` for
    gradient sync. Every rank executes this function and every rank calls every `sink`
    method — the `RunSink` implementation owns any rank-aware filtering it wants (e.g.
    only writing files / wandb on `is_main_process()`).

    Args:
        target_model: The frozen model whose parameters are being decomposed. Mutated
            in place: forced to `requires_grad=False` and put in `eval()` mode.
        train_loader: Train data loader. Looped indefinitely until `pd_config.steps`
            is reached.
        run_batch: Callable that runs one batch through the wrapped target model and
            returns its output tensor.
        reconstruction_loss: Callable returning `(loss_sum, n_elements)` used by
            recon-style loss metrics.
        pd_config: Algorithm specification — CI fn, loss metrics, optimizers,
            decomposition targets, seed, tied weights, warmup, etc.
        runtime_config: Compute substrate — device, autocast, data-parallelism.
        sink: Output destination. Metric keys handed to `sink.log` are pre-namespaced
            (`train/...`, `eval/...`). See note above on rank semantics.
        cadence: Train-log + checkpoint period. The final step always checkpoints
            regardless of `cadence.save_every`.
        eval_loop: Optional eval bundle. Pass `None` to skip eval entirely.
    """
    dist_state = get_distributed_state()
    device = runtime_config.device
    validate_pgd_scope(
        pd_config.loss_metrics,
        batch_size=pd_config.batch_size,
        world_size=dist_state.world_size if dist_state is not None else 1,
    )

    train_iterator = loop_dataloader(train_loader)
    eval_iterator = loop_dataloader(eval_loop.loader) if eval_loop is not None else None

    if pd_config.identity_decomposition_targets is not None:
        insert_identity_operations_(
            target_model,
            identity_decomposition_targets=pd_config.identity_decomposition_targets,
        )

    target_model.requires_grad_(False)
    target_model.eval()
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
        tie_component_weights(component_model, pd_config.tied_weights)

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

    loss_instances, eval_instances = instantiate_metrics(
        pd_config,
        component_model,
        device,
        eval_metrics=eval_loop.metrics if eval_loop is not None else None,
    )

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
        if eval_loop is not None and eval_loop.should_eval(step):
            assert eval_iterator is not None
            with torch.no_grad(), bf16_autocast(enabled=runtime_config.autocast_bf16):
                slow_step = eval_loop.should_run_slow_eval(step)
                active = [m for m in eval_instances.values() if not (m.slow and not slow_step)]
                for m in active:
                    m.reset()
                for _ in range(eval_loop.n_steps):
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
