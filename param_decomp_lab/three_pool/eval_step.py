"""3-pool eval pass — builds ``MetricContext`` cross-pool, runs metrics on PPGD.

Same algebra as ``param_decomp.optimize._build_metric_context`` (the 1-pool eval
builder), but each pool only runs the work backed by state it actually holds:

  CI   pool: target_fwd → CI fn fwd → ship full CIOutputs to PPGD.
  PPGD pool: target_fwd → calc_weight_deltas → recv CI from CI → assemble MetricContext.
  chunkwise pool: barrier through.

Reductions inside eval metrics are scoped to the PPGD pool subgroup via
``use_reduction_group(world.ppgd_pool_group)`` so CI and chunkwise don't block on them.

The ``CIOutputs`` ship covers ``lower_leaky``, ``upper_leaky``, ``pre_sigmoid``
in one packed buffer — any metric reading ``ctx.ci.*`` works without a
per-metric audit.
"""

import gc
import time
from collections.abc import Iterator
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

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
from param_decomp_lab.three_pool.context import ChunkContext, CIContext, PoolContext, PPGDContext


def _slice_batch_dim0(batch: Any, sl: slice) -> tuple[Any, int]:
    """Slice along the leading (batch) dim and return ``(slice, seq_len)``.

    Matches the convention used by ``_slice_batch_for_ppgd`` /
    ``_slice_batch_for_chunkwise``: Tensor batches are sliced; dict batches are
    returned unchanged (callers feeding dicts are responsible for handling that
    upstream).
    """
    batch_local = batch[sl] if isinstance(batch, Tensor) else batch
    if isinstance(batch_local, Tensor):
        seq_len = batch_local.shape[1] if batch_local.ndim >= 2 else 1
    else:
        assert isinstance(batch_local, dict) and "input_ids" in batch_local
        seq_len = batch_local["input_ids"].shape[1]
    return batch_local, seq_len


def _build_metric_context_three_pool(
    batch: Any,
    *,
    ctx: PoolContext,
    step: int,
    device: str,
    component_model: LMComponentModel,
    config: PDConfig,
    reconstruction_loss: ReconstructionLoss,
    c_per_site: dict[str, int],
) -> MetricContext | None:
    """Build a ``MetricContext`` under 3-pool. Returns the context on PPGD ranks;
    returns ``None`` on CI (after shipping CI to PPGD) and chunkwise (no-op).
    """
    batch = move_batch_to_device(batch, device)
    # The eval batch is global: every pool rank receives all `batch_global` rows and
    # carves out its own `[slice_idx*bl : (slice_idx+1)*bl]` slice (bl = batch_global //
    # n_pool). If the loader hands back fewer than batch_global rows, the high-index
    # slices fall off the end and yield an empty (0-row) local batch — SDPA then returns
    # None and the model forward dies on `y.transpose` with a cryptic NoneType error.
    # Require the eval loader's batch_size to equal batch_global so every slice is full.
    if isinstance(batch, Tensor):
        assert batch.shape[0] == ctx.world.batch_global, (
            f"eval batch has {batch.shape[0]} rows but the 3-pool slices it as a global "
            f"batch of {ctx.world.batch_global}; set eval.batch_size == pd.batch_size "
            f"({ctx.world.batch_global})"
        )
    match ctx:
        case CIContext():
            batch_local, _ = _slice_batch_dim0(
                batch, ctx.role.batch_slice(ctx.world.batch_local_ci)
            )
            _out, pre_weight_acts = component_model.forward_with_pre_weight_acts(batch_local)
            ci = component_model.calc_causal_importances(
                pre_weight_acts=pre_weight_acts,
                detach_inputs=False,
                sampling=config.sampling,
            )
            ctx.portals.ci_eval_to_ppgd.send(ctx.role, ci)
            return None
        case PPGDContext():
            batch_local, seq_len = _slice_batch_dim0(
                batch, ctx.role.batch_slice(ctx.world.batch_local_ppgd)
            )
            target_out, pre_weight_acts = component_model.forward_with_pre_weight_acts(batch_local)
            weight_deltas = component_model.calc_weight_deltas()
            ci = ctx.portals.ci_eval_from_ci_pool.recv(
                ctx.role, c_per_site, seq_len=seq_len, device=torch.device(device)
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
        case ChunkContext():
            return None


def run_eval_step(
    eval_iterator: Iterator[Any],
    *,
    n_steps: int,
    slow_step: bool,
    metrics: list[Metric[Any]],
    ctx: PoolContext,
    step: int,
    device: str,
    component_model: LMComponentModel,
    config: PDConfig,
    runtime_config: RuntimeConfig,
    reconstruction_loss: ReconstructionLoss,
    c_per_site: dict[str, int],
    sink: RunSink,
) -> None:
    """One 3-pool eval pass over ``n_steps`` batches.

    All pools call this; only PPGD ranks run ``metric.update`` / ``compute``.
    CI ships full CIOutputs to PPGD per batch; chunkwise barriers through.

    Metric all-reductions are confined to the PPGD subgroup via
    ``use_reduction_group``. CI + chunkwise must NOT call ``all_reduce`` inside this
    scope (they don't, by construction — they execute none of the metric code).

    ``slow_step`` is a pass-through filter: any metric whose ``slow`` class-attr
    is True only runs when ``slow_step`` is True.
    """
    # NOTE: `torch.cuda.synchronize()` is unsafe here — it drains ALL
    # CUDA streams including any pending async NCCL collectives on side
    # streams. The structural fix (cross_pool_p2p_group) made this barrier
    # safe without needing a drain.
    sync_across_processes()
    is_ppgd = isinstance(ctx, PPGDContext)
    active = [m for m in metrics if not (m.slow and not slow_step)] if is_ppgd else []
    ppgd_group = ctx.world.ppgd_pool_group if is_ppgd else None
    with (
        torch.no_grad(),
        bf16_autocast(runtime_config.autocast_bf16),
        use_reduction_group(ppgd_group),
    ):
        for m in active:
            m.reset()
        for i in range(n_steps):
            batch = next(eval_iterator)
            metric_ctx = _build_metric_context_three_pool(
                batch,
                ctx=ctx,
                step=step,
                device=device,
                component_model=component_model,
                config=config,
                reconstruction_loss=reconstruction_loss,
                c_per_site=c_per_site,
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
                t0 = time.time()
                single = collect_metric_outputs([m])
                logger.info(
                    f"eval/compute({type(m).__name__}) took "
                    f"{time.time() - t0:.2f}s ({len(single)} outputs)"
                )
                results.update(single)
        else:
            results = None
    # Barrier on default_pg (30-min timeout) before the broadcast — non-PPGD
    # ranks otherwise reach the broadcast immediately and wait beyond the
    # 10-min default timeout of cross_pool_p2p_group while PPGD finishes
    # slow metric computation.
    sync_across_processes()
    # Only PPGD computes metrics. Ship `results` to rank 0 via the
    # all-rank cross_pool_p2p_group so rank 0 (the only real sink) can log.
    ppgd_leader_rank = ctx.world.ppgd_ranks[0]
    payload: list[dict[str, Any] | None] = [results if ctx.role.rank == ppgd_leader_rank else None]
    dist.broadcast_object_list(payload, src=ppgd_leader_rank, group=ctx.world.cross_pool_p2p_group)
    if ctx.role.rank == 0:
        rank0_results = payload[0]
        assert rank0_results is not None
        sink.console(*(f"eval/{k}: {v}" for k, v in rank0_results.items()))
        sink.log({f"eval/{k}": v for k, v in rank0_results.items()}, step=step)
    torch.cuda.empty_cache()
    gc.collect()
