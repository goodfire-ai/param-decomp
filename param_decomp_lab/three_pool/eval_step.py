"""3-pool eval pass — builds ``MetricContext`` cross-pool, runs metrics on PPGD.

Same algebra as ``param_decomp.optimize._build_metric_context`` (the 1-pool eval
builder), but each pool only runs the work backed by state it actually holds:

  CI   pool: target_fwd → CI fn fwd → ship full CIOutputs to PPGD.
  PPGD pool: target_fwd → calc_weight_deltas → recv CI from CI → assemble MetricContext.
  LW   pool: barrier through.

Reductions inside eval metrics are scoped to the PPGD pool subgroup via
``use_reduction_group(world.ppgd_pool_group)`` so CI and LW don't block on them.

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
from param_decomp.component_model import ComponentModel
from param_decomp.configs import PDConfig, RuntimeConfig
from param_decomp.distributed import sync_across_processes, use_reduction_group
from param_decomp.log import logger
from param_decomp.metrics.base import Metric
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.output import collect_metric_outputs
from param_decomp.run_sink import RunSink
from param_decomp.torch_helpers import bf16_autocast
from param_decomp_lab.three_pool.layout import ThreePoolLayout
from param_decomp_lab.three_pool.portals import Portals


def _slice_batch_dim0(batch: Any, sl: slice) -> tuple[Any, int]:
    """Slice along the leading (batch) dim and return ``(slice, seq_len)``.

    Matches the convention used by ``_slice_batch_for_ppgd`` /
    ``_slice_batch_for_layerwise``: Tensor batches are sliced; dict batches are
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
    layout: ThreePoolLayout,
    portals: Portals,
    step: int,
    device: str,
    component_model: ComponentModel,
    config: PDConfig,
    reconstruction_loss: ReconstructionLoss,
    c_per_site: dict[str, int],
) -> MetricContext | None:
    """Build a ``MetricContext`` under 3-pool. Returns the context on PPGD ranks;
    returns ``None`` on CI (after shipping CI to PPGD) and LW (no-op).
    """
    batch = move_batch_to_device(batch, device)
    match layout.my_pool:
        case "ci":
            assert layout.my_ci_slice_idx is not None
            batch_local, _ = _slice_batch_dim0(batch, layout.my_batch_slice_ci())
            target_output = component_model(batch_local, cache_type="input")
            ci = component_model.calc_causal_importances(
                pre_weight_acts=target_output.cache,
                detach_inputs=False,
                sampling=config.sampling,
            )
            portals.ci_outputs_to_ppgd.send(ci, my_ci_slice_idx=layout.my_ci_slice_idx)
            return None
        case "ppgd":
            assert layout.my_ppgd_slice_idx is not None
            batch_local, seq_len = _slice_batch_dim0(batch, layout.my_batch_slice_ppgd())
            target_output = component_model(batch_local, cache_type="input")
            weight_deltas = component_model.calc_weight_deltas()
            ci = portals.ci_outputs_to_ppgd.recv(
                my_ppgd_slice_idx=layout.my_ppgd_slice_idx,
                site_to_c=c_per_site,
                seq_len=seq_len,
                device=torch.device(device),
            )
            return MetricContext(
                model=component_model,
                batch=batch_local,
                target_out=target_output.output,
                pre_weight_acts=target_output.cache,
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
        case "layerwise":
            return None


def run_eval_step(
    eval_iterator: Iterator[Any],
    *,
    n_steps: int,
    slow_step: bool,
    metrics: list[Metric[Any]],
    layout: ThreePoolLayout,
    portals: Portals,
    step: int,
    device: str,
    component_model: ComponentModel,
    config: PDConfig,
    runtime_config: RuntimeConfig,
    reconstruction_loss: ReconstructionLoss,
    c_per_site: dict[str, int],
    sink: RunSink,
) -> None:
    """One 3-pool eval pass over ``n_steps`` batches.

    All pools call this; only PPGD ranks run ``metric.update`` / ``compute``.
    CI ships full CIOutputs to PPGD per batch; LW barriers through.

    Metric all-reductions are confined to the PPGD subgroup via
    ``use_reduction_group``. CI + LW must NOT call ``all_reduce`` inside this
    scope (they don't, by construction — they execute none of the metric code).

    ``slow_step`` is a pass-through filter: any metric whose ``slow`` class-attr
    is True only runs when ``slow_step`` is True.
    """
    # NOTE: `torch.cuda.synchronize()` is unsafe here — it drains ALL
    # CUDA streams including any pending async NCCL collectives on side
    # streams. The structural fix (cross_pool_p2p_group) made this barrier
    # safe without needing a drain.
    sync_across_processes()
    active = (
        [m for m in metrics if not (m.slow and not slow_step)] if layout.my_pool == "ppgd" else []
    )
    ppgd_group = layout.world.ppgd_pool_group if layout.my_pool == "ppgd" else None
    with (
        torch.no_grad(),
        bf16_autocast(runtime_config.autocast_bf16),
        use_reduction_group(ppgd_group),
    ):
        for m in active:
            m.reset()
        for i in range(n_steps):
            batch = next(eval_iterator)
            ctx = _build_metric_context_three_pool(
                batch,
                layout=layout,
                portals=portals,
                step=step,
                device=device,
                component_model=component_model,
                config=config,
                reconstruction_loss=reconstruction_loss,
                c_per_site=c_per_site,
            )
            if ctx is not None:
                for m in active:
                    t0 = time.time()
                    m.update(ctx)
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
    ppgd_leader_rank = layout.world.ppgd_ranks[0]
    payload: list[dict[str, Any] | None] = [results if layout.my_rank == ppgd_leader_rank else None]
    dist.broadcast_object_list(
        payload, src=ppgd_leader_rank, group=layout.world.cross_pool_p2p_group
    )
    if layout.my_rank == 0:
        rank0_results = payload[0]
        assert rank0_results is not None
        sink.console(*(f"eval/{k}: {v}" for k, v in rank0_results.items()))
        sink.log({f"eval/{k}": v for k, v in rank0_results.items()}, step=step)
    torch.cuda.empty_cache()
    gc.collect()
