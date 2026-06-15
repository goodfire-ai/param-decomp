"""2-pool eval pass. Pool A holds BOTH the CI fn and the full V/U replica, so it
builds the entire ``MetricContext`` locally — no cross-pool CI ship (the 3-pool
CI→PPGD eval edge is gone). Pool B (chunkwise) barriers through.

Reductions inside eval metrics are scoped to the Pool A subgroup via
``use_reduction_group(world.ci_pool_group)`` (the Pool A all-reduce group) so the
chunkwise pool doesn't block on them.
"""

import gc
import time
from collections.abc import Iterator
from typing import Any

import torch
import torch.distributed as dist

from param_decomp.batch_and_loss_fns import ReconstructionLoss, move_batch_to_device
from param_decomp.distributed import sync_across_processes, use_reduction_group
from param_decomp.log import logger
from param_decomp.metrics.base import Metric
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.output import collect_metric_outputs
from param_decomp.run_sink import RunSink
from param_decomp.torch_helpers import bf16_autocast
from param_decomp_config.pd import PDConfig, RuntimeConfig
from param_decomp_lab.experiments.lm.vendored.component_model import LMComponentModel
from param_decomp_lab.three_pool.eval_step import _slice_batch_dim0
from param_decomp_lab.three_pool.two_pool_context import PoolAContext, TwoPoolContext


def _build_metric_context_two_pool(
    batch: Any,
    *,
    ctx: TwoPoolContext,
    step: int,
    device: str,
    component_model: LMComponentModel,
    config: PDConfig,
    reconstruction_loss: ReconstructionLoss,
) -> MetricContext | None:
    """Build a ``MetricContext`` under 2-pool. Returns the context on Pool A ranks
    (which hold both the CI fn and the V/U replica); ``None`` on chunkwise."""
    batch = move_batch_to_device(batch, device)
    if isinstance(batch, torch.Tensor):
        assert batch.shape[0] == ctx.world.batch_global, (
            f"eval batch has {batch.shape[0]} rows but the 2-pool slices it as a global "
            f"batch of {ctx.world.batch_global}; set eval.batch_size == pd.batch_size "
            f"({ctx.world.batch_global})"
        )
    if not isinstance(ctx, PoolAContext):
        return None
    batch_local, _ = _slice_batch_dim0(batch, ctx.role.batch_slice(ctx.world.batch_local_ci))
    target_out, pre_weight_acts = component_model.forward_with_pre_weight_acts(batch_local)
    weight_deltas = component_model.calc_weight_deltas()
    ci = component_model.calc_causal_importances(
        pre_weight_acts=pre_weight_acts, detach_inputs=False, sampling=config.sampling
    )
    return MetricContext(
        model=component_model,
        batch=batch_local,
        target_out=target_out,
        pre_weight_acts=pre_weight_acts,
        ci=ci,
        weight_deltas=weight_deltas,
        step=step,
        total_steps=config.steps,
        use_delta_component=config.use_delta_component,
        sampling=config.sampling,
        n_mask_samples=config.n_mask_samples,
        reconstruction_loss=reconstruction_loss,
        is_eval=True,
    )


def run_two_pool_eval_step(
    eval_iterator: Iterator[Any],
    *,
    n_steps: int,
    slow_step: bool,
    metrics: list[Metric[Any]],
    ctx: TwoPoolContext,
    step: int,
    device: str,
    component_model: LMComponentModel,
    config: PDConfig,
    runtime_config: RuntimeConfig,
    reconstruction_loss: ReconstructionLoss,
    sink: RunSink,
) -> None:
    """One 2-pool eval pass over ``n_steps`` batches. Only Pool A runs metrics;
    chunkwise barriers through. Metric all-reductions are confined to the Pool A
    subgroup via ``use_reduction_group``."""
    sync_across_processes()
    is_pool_a = isinstance(ctx, PoolAContext)
    active = [m for m in metrics if not (m.slow and not slow_step)] if is_pool_a else []
    pool_a_group = ctx.world.ci_pool_group if is_pool_a else None
    with (
        torch.no_grad(),
        bf16_autocast(runtime_config.autocast_bf16),
        use_reduction_group(pool_a_group),
    ):
        for m in active:
            m.reset()
        for i in range(n_steps):
            batch = next(eval_iterator)
            metric_ctx = _build_metric_context_two_pool(
                batch,
                ctx=ctx,
                step=step,
                device=device,
                component_model=component_model,
                config=config,
                reconstruction_loss=reconstruction_loss,
            )
            if metric_ctx is not None:
                for m in active:
                    t0 = time.time()
                    m.update(metric_ctx)
                    logger.info(
                        f"eval/update({type(m).__name__}) step={i} took {time.time() - t0:.2f}s"
                    )
        results: dict[str, Any] | None
        if active:
            results = {}
            for m in active:
                results.update(collect_metric_outputs([m]))
        else:
            results = None
    sync_across_processes()
    pool_a_leader_rank = ctx.world.ci_ranks[0]
    payload: list[dict[str, Any] | None] = [
        results if ctx.role.rank == pool_a_leader_rank else None
    ]
    dist.broadcast_object_list(
        payload, src=pool_a_leader_rank, group=ctx.world.cross_pool_p2p_group
    )
    if ctx.role.rank == 0:
        rank0_results = payload[0]
        assert rank0_results is not None
        sink.console(*(f"eval/{k}: {v}" for k, v in rank0_results.items()))
        sink.log({f"eval/{k}": v for k, v in rank0_results.items()}, step=step)
    torch.cuda.empty_cache()
    gc.collect()
