"""Model-agnostic pieces of the PD step: metric context, loss step, and the eval pass.

These are the parts of the step that do not depend on how the model is wrapped (DDP vs
FSDP) or how checkpoints are written: building the per-batch `MetricContext`,
accumulating the weighted loss and running backward (plus the metrics'
`before_backward` / `after_backward` hooks), the eval pass, and LR scheduling.

With the torch trainer retired (oracle at git tag `torch-oracle`), the live consumer is
`experiments.lm.offline_eval`, which runs `run_eval_pass` over an `EvalLoop` to score a
JAX-exported checkpoint with the reference yaml's eval metrics.
"""

import gc
import signal
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
from pydantic import PositiveInt
from torch import Tensor
from torch.utils.data import DataLoader

from param_decomp.batch_and_loss_fns import ReconstructionLoss, move_batch_to_device
from param_decomp.component_model import ComponentModelProtocol, OutputWithCache
from param_decomp.metrics.base import Metric
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.output import collect_metric_outputs
from param_decomp.torch_helpers import bf16_autocast
from param_decomp_config.losses import LossMetricConfig
from param_decomp_config.pd import PDConfig
from param_decomp_config.schedule import get_scheduled_value

__all__ = [
    "EvalLoop",
    "_SigtermFlag",
    "_assert_ctx_invariants",
    "_build_metric_context",
    "_install_sigterm_flag",
    "empty_cuda_cache_and_collect",
    "run_eval_pass",
    "run_loss_step",
    "scheduled_lrs",
]


@dataclass
class _SigtermFlag:
    """Mutable flag flipped by a SIGTERM handler so the train loop can react."""

    received: bool = False


def _install_sigterm_flag() -> _SigtermFlag:
    """Install a SIGTERM handler that flips a flag, and return the flag.

    SLURM sends SIGTERM to all ranks at job-kill / preemption time. The handler
    is intentionally minimal (set a flag, return) — Python's signal handlers
    aren't strictly async-signal-safe, and we want the actual checkpoint save
    to happen at a known-safe point in the train loop. No teardown: ``Trainer.run``
    owns the process for its lifetime and the next SIGTERM after ``run`` returns
    can take the default action.
    """
    flag = _SigtermFlag()

    def _handler(signum: int, frame: Any) -> None:
        del signum, frame
        flag.received = True

    signal.signal(signal.SIGTERM, _handler)
    return flag


@dataclass(frozen=True)
class EvalLoop:
    """Eval-loop runtime objects bundled with their timing.

    Pass ``eval_loop=None`` to :meth:`Trainer.run` (or :func:`optimize`) to skip
    eval entirely. When set, the trainer evaluates every ``every`` steps; on steps
    that are also multiples of ``slow_every``, slow metrics fire too. ``slow_every``
    must be a multiple of ``every`` — the trainer only checks :meth:`should_run_slow_eval`
    on steps where :meth:`should_eval` already fired.

    Attributes:
        loader: Eval data loader. Looped for the lifetime of training.
        metrics: Caller-instantiated eval ``Metric``s. ``optimize`` calls
            ``Metric.bind(model, device)`` on each before the loop.
        n_steps: Number of eval batches per eval pass.
        every: Period (in train steps) between eval passes.
        slow_every: Period (in train steps) between *slow* eval passes. Must
            be a multiple of ``every``.
        slow_on_first_step: Whether slow eval fires at step 0.
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
        """Whether a (regular) eval pass should fire at ``step``."""
        return step % self.every == 0

    def should_run_slow_eval(self, step: int) -> bool:
        """Whether slow eval metrics should fire at ``step``.

        Slow eval is gated on top of ``should_eval``; callers are expected to
        only call this on steps where ``should_eval`` is already true.
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
    component_model: ComponentModelProtocol,
    config: PDConfig,
    reconstruction_loss: ReconstructionLoss,
    weight_deltas: dict[str, Tensor],
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


def _assert_ctx_invariants(ctx: MetricContext, device: str, step: int) -> None:
    """Fail loudly if anything is off about the metric context handed to the
    loss metrics — wrong device, non-finite target output, empty ci dict, etc.
    These would otherwise propagate silently through the loss + backward path.
    """
    assert isinstance(ctx.target_out, torch.Tensor)
    device_prefix = str(device).split(":")[0]
    assert str(ctx.target_out.device).startswith(device_prefix), (
        f"ctx.target_out device mismatch at step {step}: target_out on "
        f"{ctx.target_out.device}, trainer on {device}"
    )
    assert torch.isfinite(ctx.target_out).all(), f"non-finite values in target_out at step {step}"
    assert ctx.ci.lower_leaky, f"empty ci.lower_leaky dict at step {step}"
    assert ctx.ci.upper_leaky.keys() == ctx.ci.lower_leaky.keys(), (
        f"ci upper/lower leaky key mismatch at step {step}"
    )
    for name, t in ctx.ci.lower_leaky.items():
        assert torch.isfinite(t).all(), f"non-finite ci.lower_leaky[{name!r}] at step {step}"
        assert str(t.device).startswith(device_prefix), (
            f"ci.lower_leaky[{name!r}] device mismatch at step {step}: {t.device} vs {device}"
        )


def scheduled_lrs(step: int, *, total_steps: int, config: PDConfig) -> tuple[float, float]:
    """The ``(components_lr, ci_fn_lr)`` for ``step`` under the two LR schedules."""
    components_lr = get_scheduled_value(
        step=step, total_steps=total_steps, config=config.components_optimizer.lr_schedule
    )
    ci_fn_lr = get_scheduled_value(
        step=step, total_steps=total_steps, config=config.ci_fn_optimizer.lr_schedule
    )
    return components_lr, ci_fn_lr


def run_loss_step(
    *,
    batch: Any,
    step: int,
    device: str,
    wrapped_model: nn.Module,
    component_model: ComponentModelProtocol,
    loss_metrics: dict[str, Metric[Any]],
    config: PDConfig,
    reconstruction_loss: ReconstructionLoss,
    autocast_bf16: bool,
) -> tuple[Tensor, defaultdict[str, float]]:
    """One forward + weighted-loss + backward for the train step.

    Computes weight deltas (outside autocast so faithfulness residuals stay fp32), builds
    the metric context, runs every loss metric's ``update`` under autocast, sums
    ``coeff * loss`` into ``total_loss``, then runs ``before_backward`` /
    ``total_loss.backward()`` / ``after_backward``. Does NOT zero grads, clip, step, or
    log — the caller owns those (and any model-wrap-specific concerns like residual-start
    around this call). Returns ``(total_loss, batch_log_data)`` where ``batch_log_data``
    carries the ``loss/<MetricClass>`` + ``loss/total`` scalars.
    """
    batch_log_data: defaultdict[str, float] = defaultdict(float)

    # Compute weight_deltas OUTSIDE bf16_autocast so FaithfulnessLoss residuals are fp32
    weight_deltas = component_model.calc_weight_deltas()

    with bf16_autocast(enabled=autocast_bf16):
        ctx = _build_metric_context(
            batch,
            step=step,
            is_eval=False,
            device=device,
            wrapped_model=wrapped_model,
            component_model=component_model,
            config=config,
            reconstruction_loss=reconstruction_loss,
            weight_deltas=weight_deltas,
        )
        _assert_ctx_invariants(ctx, device, step)
        losses = {name: m.update(ctx) for name, m in loss_metrics.items()}

    total_loss = torch.zeros((), device=device)
    active_loss_names: list[str] = []
    for metric_name, loss_val in losses.items():
        if loss_val is None:
            continue
        active_loss_names.append(metric_name)
        assert torch.isfinite(loss_val).all(), (
            f"non-finite loss from metric {metric_name!r} at step {step}: {loss_val}"
        )
        cfg = cast(LossMetricConfig, loss_metrics[metric_name].cfg)
        assert cfg.coeff is not None
        total_loss = total_loss + cfg.coeff * loss_val
        batch_log_data[f"loss/{metric_name}"] = loss_val.item()
    assert active_loss_names, (
        f"No active loss metrics returned a loss at step {step}. "
        f"Configured loss metrics: {list(loss_metrics)}"
    )
    assert torch.isfinite(total_loss).all(), (
        f"total_loss is non-finite at step {step}: {total_loss}"
    )
    batch_log_data["loss/total"] = total_loss.item()

    for metric_name, m in loss_metrics.items():
        m.before_backward(losses[metric_name])

    total_loss.backward()

    for m in loss_metrics.values():
        m.after_backward()

    return total_loss, batch_log_data


def run_eval_pass(
    *,
    eval_iterator: Iterator[Any],
    n_steps: int,
    slow_step: bool,
    all_instances: dict[str, Metric[Any]],
    step: int,
    device: str,
    wrapped_model: nn.Module,
    component_model: ComponentModelProtocol,
    config: PDConfig,
    reconstruction_loss: ReconstructionLoss,
    autocast_bf16: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one eval pass and return ``(fast_metrics, slow_metrics)`` outputs.

    Resets and updates the active metrics over ``n_steps`` eval batches under
    ``no_grad`` + autocast, then collects outputs. Slow metrics are included only when
    ``slow_step`` is true. The caller logs the returned dicts (so it can choose the
    ``eval/`` vs ``slow_eval/`` namespaces) and is responsible for any post-eval cache
    cleanup. ``slow_metrics`` is empty when no slow metric is active.
    """
    eval_weight_deltas = component_model.calc_weight_deltas()
    with torch.no_grad(), bf16_autocast(enabled=autocast_bf16):
        active = [m for m in all_instances.values() if not (m.slow and not slow_step)]
        for m in active:
            m.reset()
        for _ in range(n_steps):
            ctx = _build_metric_context(
                next(eval_iterator),
                step=step,
                is_eval=True,
                device=device,
                wrapped_model=wrapped_model,
                component_model=component_model,
                config=config,
                reconstruction_loss=reconstruction_loss,
                weight_deltas=eval_weight_deltas,
            )
            for m in active:
                m.update(ctx)
        fast_metrics = collect_metric_outputs([m for m in active if not m.slow])
        slow_active = [m for m in active if m.slow]
        slow_metrics = collect_metric_outputs(slow_active) if slow_active else {}
    return fast_metrics, slow_metrics


def empty_cuda_cache_and_collect() -> None:
    """Free cached eval allocations + run a GC pass (called after an eval pass)."""
    torch.cuda.empty_cache()
    gc.collect()
