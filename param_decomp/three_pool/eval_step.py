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
from collections.abc import Iterator
from typing import Any

import torch
from torch import Tensor

from param_decomp._trace import trace
from param_decomp.batch_and_loss_fns import ReconstructionLoss, move_batch_to_device
from param_decomp.component_model import ComponentModel
from param_decomp.configs import PDConfig, RuntimeConfig
from param_decomp.distributed import sync_across_processes, use_reduction_group
from param_decomp.metrics.base import Metric
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.output import collect_metric_outputs
from param_decomp.run_sink import RunSink
from param_decomp.three_pool.layout import ThreePoolLayout
from param_decomp.two_pool.runtime import autocast_bf16


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
            trace("eval/ci/0_slice_batch")
            batch_local, _ = _slice_batch_dim0(batch, layout.my_batch_slice_ci())
            trace("eval/ci/1_target_fwd")
            target_output = component_model(batch_local, cache_type="input")
            trace("eval/ci/2_ci_fn_fwd")
            ci = component_model.calc_causal_importances(
                pre_weight_acts=target_output.cache,
                detach_inputs=False,
                sampling=config.sampling,
            )
            trace("eval/ci/3_send_ci_to_ppgd: enter")
            layout.send_ci_eval_to_ppgd(ci)
            trace("eval/ci/3_send_ci_to_ppgd: done")
            return None
        case "ppgd":
            trace("eval/ppgd/0_slice_batch")
            batch_local, seq_len = _slice_batch_dim0(batch, layout.my_batch_slice_ppgd())
            trace("eval/ppgd/1_target_fwd")
            target_output = component_model(batch_local, cache_type="input")
            trace("eval/ppgd/2_calc_weight_deltas")
            weight_deltas = component_model.calc_weight_deltas()
            trace("eval/ppgd/3_recv_ci_from_ci_pool: enter")
            ci = layout.recv_ci_eval_from_ci_pool(
                c_per_site, seq_len=seq_len, device=torch.device(device)
            )
            trace("eval/ppgd/3_recv_ci_from_ci_pool: done")
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

    Stream fences before/after the global ``sync_across_processes()`` calls so any
    in-flight cross-pool p2p (D5b/D7 sends from training, eval CI ship from this
    pass) drains before the default-PG ``dist.barrier()`` collective. Without
    these, NCCL's default communicator can be left "dirty" with un-progressed
    p2p work, and the subsequent barrier hangs (see the May 27 deadlock).
    """
    trace(f"eval/A_enter step={step} slow={slow_step}")
    # NOTE: `torch.cuda.synchronize()` is unsafe here: it drains ALL CUDA streams
    # including pending async NCCL recvs (e.g. the V/U bcast under defer_vu_opt).
    # We instead rely on traces to identify any default-PG p2p that didn't drain
    # cleanly before this barrier — that's the actual issue if eval hangs here.
    sync_across_processes()  # align all pools before eval
    trace("eval/C_pre_eval_barrier: done")
    active = (
        [m for m in metrics if not (m.slow and not slow_step)] if layout.my_pool == "ppgd" else []
    )
    ppgd_group = layout.world.ppgd_pool_group if layout.my_pool == "ppgd" else None

    with (
        torch.no_grad(),
        autocast_bf16(runtime_config.autocast_bf16),
        use_reduction_group(ppgd_group),
    ):
        for m in active:
            m.reset()
        for i in range(n_steps):
            trace(f"eval/D_step_{i}_next_batch")
            batch = next(eval_iterator)
            trace(f"eval/D_step_{i}_build_ctx: enter")
            ctx = _build_metric_context_three_pool(
                batch,
                layout=layout,
                step=step,
                device=device,
                component_model=component_model,
                config=config,
                reconstruction_loss=reconstruction_loss,
                c_per_site=c_per_site,
            )
            trace(f"eval/D_step_{i}_build_ctx: done")
            if ctx is not None:
                for j, m in enumerate(active):
                    trace(f"eval/D_step_{i}_metric_{j}_{type(m).__name__}_update")
                    m.update(ctx)
        if active:
            trace("eval/E_collect_metric_outputs: enter")
            results = collect_metric_outputs(active)
            trace("eval/E_collect_metric_outputs: done")
            if layout.my_is_pool_leader:
                sink.console(*(f"eval/{k}: {v}" for k, v in results.items()))
                sink.log({f"eval/{k}": v for k, v in results.items()}, step=step)
    sync_across_processes()  # align all pools after eval
    trace("eval/G_post_eval_barrier: done")
    torch.cuda.empty_cache()
    gc.collect()
    trace("eval/Z_done")
