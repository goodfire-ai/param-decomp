"""CI pool training step: target_fwd → CI fn fwd → bcast → imp_min + dead-time prefetch → fused bwd → opt step.

CI pool is new under 3-pool. Each CI rank holds the full CI fn (replicated)
and processes one DP shard of the batch. Per-step flow (numbered to match
``DESIGN.md`` `ci/N_phase` labels):

  1. CI fn fwd on H_T (pre-cached by previous step's dead-time prefetch)
     → CI_T = {s: [B_local_ci, S, C_s]}. CI-fn graph retained for step 8.
  2. async send CI_T per-site → Layerwise + full-model → PPGD (kicks off NIC).
  3. imp_min loss on CI_T.upper_leaky → leaf grad (kept inside the CI fn
     backward graph; combined with downstream grads at step 8).
  4. Dead-time prefetch: target_fwd(batch T+1) → H_{T+1} (cache_type=input).
     Inside autocast + no_grad; runs concurrently with the cross-pool sends
     because the NIC and GPU compute streams are independent.
  5. recv g_CI_LW from Layerwise pool (per-site, stitched across K_lw_per_ci slices).
  6. recv g_CI_PPGD from PPGD pool (per-site, stitched across K_ppgd_per_ci slices).
  7. Assemble g_CI_total per site = g_CI_LW + g_CI_PPGD on CI rank's slice.
  8. Fused backward: torch.autograd.backward(
         tensors=[imp_min_loss, *(ci.lower_leaky[s] for s in all_sites)],
         grad_tensors=[None, *(g_CI_total[s] for s in all_sites)],
     )
     — single backward through the CI fn graph; imp_min's gradient enters
     via ci.upper_leaky; downstream gradients enter via ci.lower_leaky.
  9. in-pool AVG-reduce on CI fn grads (each CI rank computed on its slice;
     averaging gives the global per-example gradient).
  10. AdamW step on CI fn.

Returns the prefetched H_{T+1} cache so the runner can thread it into the
next step's call. For the very first step (or after the last batch), pass
``h_cache_T=None`` / ``batch_T_plus_1=None`` respectively.
"""

# pyright: reportArgumentType=false

from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_fn
import torch.nn as nn
from torch import Tensor

from param_decomp.component_model import ComponentModel
from param_decomp.metrics.importance_minimality import (
    _finalize as _finalize_imp_min,
)
from param_decomp.metrics.importance_minimality import (
    _get_linear_annealed_p,
    _per_component_sums,
)
from param_decomp.three_pool.layout import ThreePoolLayout
from param_decomp.three_pool.profiler import PhaseProfiler
from param_decomp.three_pool.runtime import _ThreePoolRuntime
from param_decomp.two_pool.runtime import autocast_bf16


def _importance_minimality_loss(
    ci_upper: dict[str, Tensor],
    current_frac_of_training: float,
    cfg: _ThreePoolRuntime,
    ci_pool_group: dist.ProcessGroup,
    n_ci_pool: int,
) -> Tensor:
    """Exact (across CI pool) importance-minimality loss.

    Each CI rank computes ``per_component_sums`` on its own batch slice, then
    we autograd-aware-SUM-reduce both ``per_component_sums`` and ``n_examples``
    across the CI pool. The finalized loss is the same on every CI rank;
    backward gives each rank's local CI fn gradient, which the in-pool
    AVG-reduce at step 9 combines (standard DDP convention).

    Autograd note: ``torch.distributed.nn.functional.all_reduce`` is the
    autograd-aware variant — forward sums, backward broadcasts the upstream
    gradient unchanged to every rank's input (correct for SUM since
    ``∂global/∂local_i = 1`` for all i).
    """
    annealed_p = _get_linear_annealed_p(
        current_frac_of_training=current_frac_of_training,
        initial_p=cfg.imp_min_pnorm,
        p_anneal_start_frac=cfg.imp_min_p_anneal_start_frac,
        p_anneal_final_p=cfg.imp_min_p_anneal_final_p,
        p_anneal_end_frac=cfg.imp_min_p_anneal_end_frac,
    )
    per_component_sums, n_examples = _per_component_sums(
        ci_upper_leaky=ci_upper, pnorm=annealed_p, eps=cfg.imp_min_eps
    )
    if n_ci_pool > 1:
        per_component_sums = {
            k: dist_fn.all_reduce(v, op=dist.ReduceOp.SUM, group=ci_pool_group)
            for k, v in per_component_sums.items()
        }
        # n_examples is a Python int derived purely from local CI shape (which
        # is uniform across CI pool by construction — batch_local_ci is the
        # same on every CI rank). So we can multiply rather than reduce.
        n_examples = n_examples * n_ci_pool
    return _finalize_imp_min(
        per_component_sums=per_component_sums,
        n_examples=n_examples,
        beta=cfg.imp_min_beta,
        # world_size=1 because per_component_sums + n_examples are now global,
        # so the log term inside _finalize computes log2(1 + global_sum) directly.
        world_size=1,
    )


def _target_fwd_and_cache(
    component_model: ComponentModel,
    batch: Any,
    bf16_autocast: bool,
) -> dict[str, Tensor]:
    """Run target forward (no grad) and return the per-site pre-weight act cache.

    Used both for the on-demand H_T (step 0) and the dead-time prefetch of
    H_{T+1}. The model output (logits / hidden state) is dropped — CI pool
    only needs the cached inputs to the decomposition sites.
    """
    with torch.no_grad(), autocast_bf16(bf16_autocast):
        out = component_model(batch, cache_type="input")
    # Upcast cache to fp32 so the CI fn forward gets fp32 inputs (CI fn fwd
    # runs in fp32 — its loss landscape is small-number-sensitive).
    return {k: v.to(torch.float32) for k, v in out.cache.items()}


def step_ci(
    layout: ThreePoolLayout,
    component_model: ComponentModel,
    optimizer: torch.optim.Optimizer,
    ci_fn_params: list[nn.Parameter],
    batch_T: Any,
    batch_T_plus_1: Any | None,
    h_cache_T: dict[str, Tensor] | None,
    cfg: _ThreePoolRuntime,
    current_frac_of_training: float,
    profiler: PhaseProfiler | None = None,
) -> tuple[dict[str, float], dict[str, Tensor] | None]:
    """One CI-pool training step.

    Returns ``(metrics, h_cache_T_plus_1)``. The runner threads
    ``h_cache_T_plus_1`` into the next call as ``h_cache_T``.

    On step 0, ``h_cache_T`` is ``None`` and we compute it inline.
    On the last step, ``batch_T_plus_1`` is ``None`` and we skip the prefetch.
    """
    p = profiler if profiler is not None else PhaseProfiler(enabled=False)
    assert layout.my_pool == "ci"
    device = next(component_model.parameters()).device

    # CI pool is multi-rank DP across batch; slice global batches to this rank's shard.
    sl = layout.my_batch_slice_ci()
    batch_T_local = batch_T[sl] if isinstance(batch_T, Tensor) else batch_T
    batch_T_plus_1_local = (
        batch_T_plus_1[sl]
        if (batch_T_plus_1 is not None and isinstance(batch_T_plus_1, Tensor))
        else batch_T_plus_1
    )

    # 0. If we don't have H_T from a previous prefetch (step 0), compute it now.
    if h_cache_T is None:
        with p.phase("ci/0_target_fwd_T_sync"):
            h_cache_T = _target_fwd_and_cache(component_model, batch_T_local, cfg.bf16_autocast)

    # 1. CI fn forward — produces CI values for ALL sites on this rank's batch slice.
    # CI fn weights are fp32 (replicated across CI pool); inputs are fp32 too.
    # Graph retained for the fused backward at step 8.
    with p.phase("ci/1_ci_fn_fwd"):
        ci = component_model.calc_causal_importances(
            pre_weight_acts=h_cache_T,
            sampling="continuous",
            detach_inputs=False,
        )

    # 2. Async send CI_T → Layerwise (per-site, sub-sliced) + PPGD (full-model, sub-sliced).
    with p.phase("ci/2_async_send_ci"):
        send_works_lw, send_bufs_lw = layout.async_send_ci_to_layerwise(ci.lower_leaky)
        send_works_pgd, send_bufs_pgd = layout.async_send_ci_to_ppgd(ci.lower_leaky)

    # 3. imp_min loss on ci.upper_leaky. Graph retained — gradient enters the
    # CI-fn backward at step 8 via this leaf, in parallel with the downstream
    # g_CI seeds on ci.lower_leaky.
    with p.phase("ci/3_imp_min"):
        loss_imp = _importance_minimality_loss(
            ci.upper_leaky,
            current_frac_of_training,
            cfg,
            ci_pool_group=layout.world.ci_pool_group,
            n_ci_pool=layout.world.n_ci,
        )

    # 4. Dead-time prefetch: target_fwd(batch T+1) → H_{T+1}. Runs concurrently
    # with the cross-pool sends + the downstream pools' recon work.
    h_cache_T_plus_1: dict[str, Tensor] | None = None
    if batch_T_plus_1_local is not None:
        with p.phase("ci/4_prefetch_target_fwd"):
            h_cache_T_plus_1 = _target_fwd_and_cache(
                component_model, batch_T_plus_1_local, cfg.bf16_autocast
            )

    # 5/6. Recv per-site CI grads from Layerwise + PPGD. recv_* posts irecvs
    # upfront so they pipeline; stitches sub-slices into [B_local_ci, S, C_s] fp32.
    # seq_len inferred from CI value shape (which we already have on this rank).
    sample_ci = next(iter(ci.lower_leaky.values()))
    assert sample_ci.ndim == 3, f"expected CI shape [B, S, C]; got {sample_ci.shape}"
    seq_len = sample_ci.shape[1]
    with p.phase("ci/5_recv_g_ci_from_lw"):
        g_ci_lw = layout.recv_g_ci_from_layerwise(cfg.c_per_site, seq_len, device)
    with p.phase("ci/6_recv_g_ci_from_ppgd"):
        g_ci_pgd = layout.recv_g_ci_from_ppgd(cfg.c_per_site, seq_len, device)

    # 7. Assemble g_CI_total per site (already on this CI rank's batch slice).
    # Both summands are [B_local_ci, S, C_s]; loss coefficients were baked into
    # the gradients on the LW / PPGD side (they scaled their losses before bwd).
    with p.phase("ci/7_assemble"):
        g_ci_total = {s: g_ci_lw[s] + g_ci_pgd[s] for s in layout.world.all_sites}

    # 8. Fused backward through the CI fn graph. imp_min provides its gradient
    # implicitly via ci.upper_leaky's autograd link to the CI fn params;
    # downstream grads are injected as grad_tensors on ci.lower_leaky.
    optimizer.zero_grad(set_to_none=True)
    with p.phase("ci/8_fused_bwd"):
        scaled_imp = cfg.coeff_imp * loss_imp
        torch.autograd.backward(
            tensors=[scaled_imp, *(ci.lower_leaky[s] for s in layout.world.all_sites)],
            grad_tensors=[None, *(g_ci_total[s] for s in layout.world.all_sites)],
        )

    # 9. In-pool AVG-reduce on CI fn grads.
    with p.phase("ci/9_in_pool_allreduce"):
        layout.all_reduce_ci_fn_grads(ci_fn_params)

    # 10. AdamW step.
    with p.phase("ci/10_opt_step"):
        optimizer.step()

    # Make sure the async CI sends from step 2 have flushed before we touch
    # ci.lower_leaky's underlying storage next step.
    with p.phase("ci/11_wait_sends"):
        for w in send_works_lw:
            w.wait()
        for w in send_works_pgd:
            w.wait()
        del send_bufs_lw, send_bufs_pgd

    return {"loss/imp": loss_imp.item()}, h_cache_T_plus_1
