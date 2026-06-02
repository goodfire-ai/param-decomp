"""CI pool training step.

CI pool is new under 3-pool. Each CI rank holds the full CI fn (replicated)
and processes one DP shard of the batch.

The step is a sequence of typed phases. Each phase consumes the previous
phase's typed bundle, so the dependency order is a type constraint: you cannot
send CI values before computing them (``_send_ci_values`` takes ``CiForward``),
cannot assemble the total CI grad before receiving both halves (``GciTotal``
takes ``GciReceived``), cannot run the fused backward before both the imp-min
loss and the assembled grads exist (it takes ``CiForward`` + ``ImpMinLoss`` +
``GciTotal``), and cannot wait the sends before posting them (``wait`` takes
``CiSends``).

The cross-pool exchanges all route through this CI rank's own portal bundle
(``ctx.portals.ci_to_lw`` etc.), so a CI step physically cannot reach for a
different pool's edges.

Phases (numbered to match ``DESIGN.md`` ``ci/N``):

  0.  Step 0 only: target_fwd to build H_T (subsequent steps reuse the prev
      iter's prefetch).
  1.  CI fn fwd on H_T → CI_T per site. Graph retained for the fused bwd at 8.
  2.  Async send CI_T → LW (per-site, sub-sliced) + PPGD (full-model,
      sub-sliced). Kicks the NIC so 4 + 5 + 6 can pipeline.
  3.  imp_min loss on ``CI.upper_leaky`` (still on graph — gradient enters
      the fused bwd at 8).
  4.  Dead-time prefetch: target_fwd(batch T+1) → H_{T+1}. GPU work overlaps
      with the NIC sends in 2 and the downstream pool work it triggered.
  5.  Recv g_CI from LW (per-site, stitched across K_lw_per_ci slices).
  6.  Recv g_CI from PPGD (per-site, stitched across K_ppgd_per_ci slices).
  7.  Assemble g_CI_total per site on this CI rank's batch slice.
  8.  Fused autograd backward: imp_min via ``upper_leaky``, downstream grads
      injected on ``lower_leaky``. One pass through the CI fn graph.
  9.  In-pool AVG-reduce on CI fn grads (standard DDP).
  9b. Cross-pool grad clip on CI fn (n_replicas=n_ci dedup).
  10. AdamW step.
  11. Wait the step-2 async sends to flush before storage gets reused next iter.

Returns ``(metrics, h_cache_T_plus_1)``. The runner threads the prefetched
cache into the next call as ``h_cache_T``. Phases 2 + 4 give the headline
overlap (NIC sends concurrent with the prefetch target fwd).
"""

# pyright: reportArgumentType=false

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

from param_decomp._trace import trace
from param_decomp.component_model import CIOutputs
from param_decomp.grad_clip import cross_pool_clip_grad_norm
from param_decomp.metrics.importance_minimality import (
    annealed_pnorm,
    finalize_imp_min,
    per_component_lp_sums,
)
from param_decomp.torch_helpers import bf16_autocast
from param_decomp_lab.experiments.lm.vendored.component_model import LMComponentModel
from param_decomp_lab.three_pool.context import CIContext
from param_decomp_lab.three_pool.portals import CIPortals, InFlightSends, all_reduce_ci_fn_grads
from param_decomp_lab.three_pool.role import CIRole
from param_decomp_lab.three_pool.runtime import _ThreePoolRuntime


@dataclass(frozen=True)
class CiForward:
    """Phase ci/1 output: the CI fn forward graph + its derived seq_len.

    Holds the live autograd graph (``ci``) that phases 2 (send), 3 (imp_min),
    and 8 (fused backward) all attach to. Constructing this is the only way to
    obtain a ``CIOutputs`` the rest of the step can act on.
    """

    ci: CIOutputs
    seq_len: int


@dataclass(frozen=True)
class CiSends:
    """Phase ci/2 output: the two in-flight CI-value sends (LW + PPGD)."""

    to_lw: InFlightSends
    to_ppgd: InFlightSends


@dataclass(frozen=True)
class ImpMinLoss:
    """Phase ci/3 output: the (CI-pool-global) importance-minimality scalar,
    still attached to the CI fn graph via ``ci.upper_leaky``."""

    loss: Tensor


@dataclass(frozen=True)
class GciReceived:
    """Phases ci/5 + ci/6 output: per-site CI grads from LW and from PPGD,
    each stitched onto this CI rank's batch slice."""

    from_lw: dict[str, Tensor]
    from_ppgd: dict[str, Tensor]


@dataclass(frozen=True)
class GciTotal:
    """Phase ci/7 output: ``g_CI_LW + g_CI_PPGD`` per site, ready to seed the
    fused backward on ``ci.lower_leaky``."""

    per_site: dict[str, Tensor]


def step_ci(
    ctx: CIContext,
    component_model: LMComponentModel,
    optimizer: torch.optim.Optimizer,
    ci_fn_params: list[nn.Parameter],
    batch_T: Any,
    batch_T_plus_1: Any | None,
    h_cache_T: dict[str, Tensor] | None,
    cfg: _ThreePoolRuntime,
    current_frac_of_training: float,
    should_log: bool,
) -> tuple[dict[str, float], dict[str, Tensor] | None]:
    """One CI-pool training step.

    Returns ``(metrics, h_cache_T_plus_1)``. On step 0, ``h_cache_T`` is
    ``None`` and we compute it inline (phase 0). On the last step,
    ``batch_T_plus_1`` is ``None`` and the prefetch is skipped.
    """
    device = next(component_model.parameters()).device

    batch_T_local, batch_T_plus_1_local = _slice_batches_for_ci(batch_T, batch_T_plus_1, ctx)

    if h_cache_T is None:
        h_cache_T = _target_fwd_and_cache(component_model, batch_T_local, cfg.bf16_autocast)

    fwd = _ci_fn_forward(component_model, h_cache_T, ctx, cfg)
    sends = _send_ci_values(ctx.portals, ctx.role, fwd)
    imp = _imp_min_phase(fwd, current_frac_of_training, cfg, ctx)
    h_cache_T_plus_1 = _prefetch_next_h(component_model, batch_T_plus_1_local, cfg)
    received = _recv_g_ci(ctx.portals, ctx.role, cfg, fwd.seq_len, device)
    total = _assemble_g_ci_total(received, ctx, cfg, fwd.seq_len)

    optimizer.zero_grad(set_to_none=True)
    # Diagnostic: sync before the bwd so phase("ci/8a") measures only the bwd
    # itself, not waiting for prior default-stream work (ci/1 fwd kernels, ci/4
    # prefetch, etc.) to drain. If ci/8a drops sharply from ~600 ms, the original
    # wall was dominated by pending stream work, not by the bwd. Remove after diagnosis.
    if os.environ.get("PD_SYNC_BEFORE_8A", "").strip() in ("1", "true", "yes"):
        torch.cuda.synchronize()
    _fused_backward_through_ci_fn(imp, fwd, total, ctx, cfg)
    _maybe_emit_ci_fn_bwd_breakdown(component_model)

    all_reduce_ci_fn_grads(ctx.world, ci_fn_params)
    if cfg.grad_clip_norm_ci_fn is not None:
        cross_pool_clip_grad_norm(
            ci_fn_params,
            cfg.grad_clip_norm_ci_fn,
            group=ctx.world.ci_pool_group,
            n_replicas=ctx.world.n_ci,
        )
    optimizer.step()

    sends.to_lw.wait()
    sends.to_ppgd.wait()

    # imp is already globally aggregated inside ``_imp_min_phase`` (per_component_sums
    # + n_examples SUM-reduced across CI pool), so every CI rank holds the same
    # scalar. Divide by ``n_ci`` so the logger's cross-pool SUM all-reduce gives back
    # the global value exactly once.
    #
    # ``.item()`` is a CPU↔GPU sync — only pay it on steps we actually log to.
    if should_log:
        imp_value = imp.loss.item()
        metrics = {"loss/imp": imp_value, "_raw/imp_num": imp_value / ctx.world.n_ci}
    else:
        metrics = {}
    return metrics, h_cache_T_plus_1


def _ci_fn_forward(
    component_model: LMComponentModel,
    h_cache_T: dict[str, Tensor],
    ctx: CIContext,
    cfg: _ThreePoolRuntime,
) -> CiForward:
    """Phase ci/1. CI fn forward on H_T → ``CIOutputs`` (graph retained for ci/8)."""
    with bf16_autocast(cfg.bf16_autocast):
        ci = component_model.calc_causal_importances(
            pre_weight_acts=h_cache_T, sampling="continuous", detach_inputs=False
        )
    seq_len = _seq_len_from_ci(ci.lower_leaky)
    _assert_ci_shapes(ci.lower_leaky, ctx, seq_len, cfg)
    return CiForward(ci=ci, seq_len=seq_len)


def _send_ci_values(portals: CIPortals, role: CIRole, fwd: CiForward) -> CiSends:
    """Phase ci/2. Async-ship CI_T to LW (per-site) and PPGD (full-model)."""
    to_lw = portals.ci_to_lw.send(role, fwd.ci.lower_leaky)
    to_ppgd = portals.ci_to_ppgd.send(role, fwd.ci.lower_leaky)
    return CiSends(to_lw=to_lw, to_ppgd=to_ppgd)


def _imp_min_phase(
    fwd: CiForward,
    current_frac_of_training: float,
    cfg: _ThreePoolRuntime,
    ctx: CIContext,
) -> ImpMinLoss:
    """Phase ci/3. Importance-minimality loss on ``ci.upper_leaky``."""
    loss = _importance_minimality_loss(
        fwd.ci.upper_leaky,
        current_frac_of_training,
        cfg,
        ci_pool_group=ctx.world.ci_pool_group,
        n_ci_pool=ctx.world.n_ci,
    )
    return ImpMinLoss(loss=loss)


def _prefetch_next_h(
    component_model: LMComponentModel,
    batch_T_plus_1_local: Any | None,
    cfg: _ThreePoolRuntime,
) -> dict[str, Tensor] | None:
    """Phase ci/4. Dead-time prefetch of H_{T+1} (skipped on the last step)."""
    if batch_T_plus_1_local is None:
        return None
    return _target_fwd_and_cache(component_model, batch_T_plus_1_local, cfg.bf16_autocast)


def _recv_g_ci(
    portals: CIPortals,
    role: CIRole,
    cfg: _ThreePoolRuntime,
    seq_len: int,
    device: torch.device,
) -> GciReceived:
    """Phases ci/5 + ci/6. Recv CI grads from LW and PPGD."""
    from_lw = portals.g_ci_from_lw.recv(role, cfg.c_per_site, seq_len, device)
    from_ppgd = portals.g_ci_from_ppgd.recv(role, cfg.c_per_site, seq_len, device)
    return GciReceived(from_lw=from_lw, from_ppgd=from_ppgd)


def _slice_batches_for_ci(
    batch_T: Any, batch_T_plus_1: Any | None, ctx: CIContext
) -> tuple[Any, Any | None]:
    """Pull this CI rank's DP shard of batch T and T+1.

    CI pool is multi-rank DP across the global batch; every CI rank gets a
    disjoint slice.
    """
    sl = ctx.role.batch_slice(ctx.world.batch_local_ci)
    batch_T_local = batch_T[sl] if isinstance(batch_T, Tensor) else batch_T
    batch_T_plus_1_local = (
        batch_T_plus_1[sl]
        if (batch_T_plus_1 is not None and isinstance(batch_T_plus_1, Tensor))
        else batch_T_plus_1
    )
    return batch_T_local, batch_T_plus_1_local


def _seq_len_from_ci(ci_lower: dict[str, Tensor]) -> int:
    """Extract seq_len from CI value shape. Asserts [B, S, C] layout."""
    sample_ci = next(iter(ci_lower.values()))
    assert sample_ci.ndim == 3, f"expected CI shape [B, S, C]; got {sample_ci.shape}"
    return sample_ci.shape[1]


def _assert_ci_shapes(
    ci_lower: dict[str, Tensor],
    ctx: CIContext,
    seq_len: int,
    cfg: _ThreePoolRuntime,
) -> None:
    """Sanity-check CI fn outputs match [B_local_ci, seq_len, C_s] per site.

    Catches misconfigured ``c_per_site`` or a wrong per-rank batch slice fast.
    """
    batch_local_ci = ctx.world.batch_local_ci
    for s, c in cfg.c_per_site.items():
        t = ci_lower[s]
        assert t.shape == (batch_local_ci, seq_len, c), (
            f"ci.lower_leaky[{s!r}] shape {tuple(t.shape)} != "
            f"expected ({batch_local_ci}, {seq_len}, {c})"
        )


def _assemble_g_ci_total(
    received: GciReceived,
    ctx: CIContext,
    cfg: _ThreePoolRuntime,
    seq_len: int,
) -> GciTotal:
    """Phase ci/7. ``g_CI_total[s] = g_CI_LW[s] + g_CI_PPGD[s]``.

    Both summands live on this CI rank's batch slice [B_local_ci, S, C_s].
    Loss coefficients were already baked in by LW/PPGD before they bwd'd.
    """
    batch_local_ci = ctx.world.batch_local_ci
    per_site: dict[str, Tensor] = {}
    for s in ctx.world.all_sites:
        c = cfg.c_per_site[s]
        lw, pgd = received.from_lw[s], received.from_ppgd[s]
        assert lw.shape == (batch_local_ci, seq_len, c), (
            f"g_ci_lw[{s!r}] shape {tuple(lw.shape)} != expected ({batch_local_ci}, {seq_len}, {c})"
        )
        assert pgd.shape == (batch_local_ci, seq_len, c), (
            f"g_ci_pgd[{s!r}] shape {tuple(pgd.shape)} != "
            f"expected ({batch_local_ci}, {seq_len}, {c})"
        )
        per_site[s] = lw + pgd
    return GciTotal(per_site=per_site)


def _maybe_emit_ci_fn_bwd_breakdown(component_model: LMComponentModel) -> None:
    """Emit per-stage CI fn bwd times as ``trace()`` lines when the bwd profile is on.

    Records the ``post_bwd`` event immediately (anchoring the input projector's bwd
    end time on the stream), synchronizes to flush events, then walks the CI fn
    modules to find the ``GlobalSharedTransformerCiFn`` instance and emits one
    ``phase: ci/8a_stage_<label>`` line per stage. No-op when profiling is off.

    Must be called right after ``_fused_backward_through_ci_fn`` so ``post_bwd``
    lands on the stream before any subsequent kernels (optimizer.step etc).
    """
    if component_model.ci_fn is None:
        return
    from param_decomp.ci_fns import GlobalSharedTransformerCiFn

    for m in component_model.ci_fn.modules():
        if isinstance(m, GlobalSharedTransformerCiFn) and m._bwd_events:
            m.record_post_bwd_event()
            torch.cuda.synchronize()
            for label, t_ms in m.compute_bwd_breakdown().items():
                trace(f"phase: ci/8a_stage_{label}: {t_ms:.1f}ms")
            return


def _fused_backward_through_ci_fn(
    imp: ImpMinLoss,
    fwd: CiForward,
    total: GciTotal,
    ctx: CIContext,
    cfg: _ThreePoolRuntime,
) -> None:
    """Phase ci/8. Backward through the CI fn graph.

    Two gradient seeds enter the graph:
      * ``coeff_imp * loss_imp`` — flows via ``ci.upper_leaky``. Under the
        SUM-grad convention the imp-min loss uses the detached-global-residual
        trick, so its backward flows ONLY through this rank's local CI values (a
        partial sum) — no cross-rank NCCL in the backward.
      * ``g_CI_total[s]`` per site — injected directly on ``ci.lower_leaky[s]``.
        96 separate gradient seeds rejoining at the shared CI fn output.

    Diagnostic split: each seed runs its own ``torch.autograd.backward`` call
    with ``retain_graph=True`` on the first so the second still sees the
    graph. Gradient accumulation onto the CI fn params is the same as one
    fused call. This is purely so the per-phase profiler can attribute time
    between the two backward paths — to find out which one dominates and
    where to optimize next.
    """
    loss_imp = imp.loss
    assert loss_imp.dim() == 0, f"loss_imp must be scalar; got {loss_imp.shape}"
    scaled_imp = cfg.coeff_imp * loss_imp
    lower_leaky_tensors = [fwd.ci.lower_leaky[s] for s in ctx.world.all_sites]
    g_ci_total_seeds = [total.per_site[s] for s in ctx.world.all_sites]
    torch.autograd.backward(
        tensors=lower_leaky_tensors,
        grad_tensors=g_ci_total_seeds,
        retain_graph=True,
    )
    torch.autograd.backward(tensors=[scaled_imp], grad_tensors=[None])


def _target_fwd_and_cache(
    component_model: LMComponentModel,
    batch: Any,
    enabled: bool,
) -> dict[str, Tensor]:
    """target_fwd (no grad) returning the per-site pre-weight act cache.

    Used by phase 0 (on-demand H_T) and phase 4 (dead-time H_{T+1} prefetch).
    Cache is upcast to fp32 so the downstream CI fn fwd gets fp32 inputs.
    """
    with torch.no_grad(), bf16_autocast(enabled):
        _out, cache = component_model.forward_with_pre_weight_acts(batch)
    return {k: v.to(torch.float32) for k, v in cache.items()}


def _importance_minimality_loss(
    ci_upper: dict[str, Tensor],
    current_frac_of_training: float,
    cfg: _ThreePoolRuntime,
    ci_pool_group: dist.ProcessGroup,
    n_ci_pool: int,
) -> Tensor:
    """Exact (across CI pool) importance-minimality loss, SUM-convention variant.

    Each CI rank computes ``per_component_sums`` on its disjoint batch slice; we
    materialize the GLOBAL sums (needed inside ``finalize_imp_min``'s ``log2`` and
    mean) via the detached-global-residual trick:

      ``global = local + (all_reduce_sum(local.detach()) - local.detach())``

    Forward value is the global sum; backward ``∂global/∂local = 1`` and the
    cross-rank term is detached, so the gradient flows ONLY through THIS rank's
    local CI values — a true partial sum. The CI-pool SUM all-reduce on the CI-fn
    grads (``all_reduce_ci_fn_grads``) then reassembles the single-pool total.

    This replaces the old autograd-aware ``dist_fn.all_reduce``, whose backward
    SUM-reduced the REPLICATED upstream gradient across the pool, leaving each
    rank with ``n_ci ×`` its partial — correct only because the (old) CI-pool AVG
    divided it back out. Under the SUM convention that ``n_ci`` would no longer
    cancel; the residual trick removes the pool-size dependence at the source (see
    ``SUM_GRAD_CONVENTION.md``).

    ``n_examples`` is uniform across CI ranks (same batch_local_ci), so multiply
    rather than reduce.
    """
    annealed_p = annealed_pnorm(
        current_frac_of_training=current_frac_of_training,
        initial_p=cfg.imp_min_pnorm,
        p_anneal_start_frac=cfg.imp_min_p_anneal_start_frac,
        p_anneal_final_p=cfg.imp_min_p_anneal_final_p,
        p_anneal_end_frac=cfg.imp_min_p_anneal_end_frac,
    )
    per_component_sums, n_examples = per_component_lp_sums(
        ci_upper_leaky=ci_upper, pnorm=annealed_p, eps=cfg.imp_min_eps
    )
    if n_ci_pool > 1:
        per_component_sums = {
            k: _global_sum_local_grad(v, ci_pool_group) for k, v in per_component_sums.items()
        }
        n_examples = n_examples * n_ci_pool
    return finalize_imp_min(
        per_component_sums=per_component_sums,
        n_examples=n_examples,
        beta=cfg.imp_min_beta,
    )


def _global_sum_local_grad(local: Tensor, group: dist.ProcessGroup) -> Tensor:
    """Global sum in the forward, local-only gradient in the backward.

    ``local + (all_reduce_sum(local.detach()) - local.detach())`` — the residual
    is detached, so forward equals the global sum while backward flows only
    through this rank's ``local`` (``∂/∂local = 1``). The result is a true partial
    sum that SUM-composes across the CI pool (see ``SUM_GRAD_CONVENTION.md``).
    """
    global_detached = local.detach().clone()
    dist.all_reduce(global_detached, op=dist.ReduceOp.SUM, group=group)
    return local + (global_detached - local.detach())
