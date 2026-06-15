"""Pool A training step (2-pool): merged CI fn + adversary on one rank.

Pool A holds the replicated CI fn (DDP across Pool A, grads SUM-reduced in-pool)
AND a full V/U replica + persistent PPGD sources. Because the CI forward and the
adversary run on the SAME rank and SAME batch slice, the 3-pool CI↔PPGD edge (mask
send + g_CI return) is gone: the adversary's g_CI is the local ``.grad`` of the
re-leafed CI mask, summed with the chunkwise pool's g_CI before the fused CI
backward. The only surviving cross-pool edges are Pool A ↔ chunk (masks out, g_CI
back, V/U grads out, updated V/U in).

The gradient assembly is identical to the 3-pool (see ``SUM_GRAD_CONVENTION.md``),
with the adversary half of g_CI computed locally instead of received over the wire:

  * CI-fn grad seed = ``g_CI_chunk + g_CI_adversary + imp_min`` (the imp-min via the
    detached-global-residual trick over the Pool A group; g_CI_adversary is the
    scaled ``.grad`` on the re-leafed mask; g_CI_chunk arrives from Pool B). Fused
    in one backward through the CI-fn graph, then SUM-reduced over the Pool A group.
  * V/U grad = chunkwise(owner) + adversary(replica). Pool A's adversary V/U grad is
    a per-rank partial (scaled by ``1/n_a``), SUM-reduced over the Pool A group, then
    shipped (leader-only) to the chunk leaders, where the contribute-once leader trick
    folds it into the chunk SUM exactly once.

Phases:

  A0. Shared target forward (no-grad, under the recon-loss strategy's LM-head
      bypass): yields BOTH the recon target hidden state AND the per-site
      pre-weight-act cache the CI fn reads.
  A1. CI fn forward on the cache → CI_T (graph retained for the fused backward).
  A2. Async send CI_T masks → chunkwise.
  A3. imp_min loss on ``ci.upper_leaky`` (graph retained).
  A4. Adversary: re-leaf ``ci.lower_leaky`` fp32, warmup-refine sources, final recon
      forward, one ``autograd.grad`` → raw g_VU + g_CI(local leaves) + g_sources.
      Scale each consumer's normalization in place.
  A5. V/U grads: send to chunk leaders (after the Pool A SUM-reduce); final source step.
  A6. Recv g_CI from chunkwise; assemble g_CI_total per site (chunk + local adversary).
  A7. Fused backward through the CI fn graph seeded by g_CI_total + imp_min.
  A8. Async in-pool SUM-reduce of CI-fn grads; clip; AdamW.
  A9. Blocking recv of updated V/U from chunkwise → copy into the model replica.
"""

# pyright: reportArgumentType=false

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from param_decomp._trace import trace
from param_decomp.component_model import CIOutputs
from param_decomp.grad_clip import cross_pool_clip_grad_norm
from param_decomp.metrics.persistent_pgd_state import PersistentPGDState
from param_decomp.torch_helpers import bf16_autocast
from param_decomp_lab.experiments.lm.vendored.component_model import LMComponentModel
from param_decomp_lab.three_pool.portals import (
    all_reduce_ci_fn_grads_async,
    sum_reduce_ppgd_grads,
)
from param_decomp_lab.three_pool.recon_loss_strategy import ReconLossStrategy
from param_decomp_lab.three_pool.reductions import per_param_grad_norms
from param_decomp_lab.three_pool.runtime import _ThreePoolRuntime
from param_decomp_lab.three_pool.step_ci import (
    _importance_minimality_loss,
    _mean_l0,
)
from param_decomp_lab.three_pool.step_ppgd import (
    RawGrads,
    _autograd_grads_wrt_vu_ci_and_sources,
    _copy_vu_into_model_in_place,
    _releaf_ci_fp32_for_grads,
    _scale_grads_in_place,
    _vu_templates,
    _warmup_and_recon,
)
from param_decomp_lab.three_pool.two_pool_context import PoolAContext


@dataclass(frozen=True)
class CiForward:
    """Phase A1 output: the live CI fn forward graph + derived seq_len."""

    ci: CIOutputs
    seq_len: int


def step_pool_a(
    ctx: PoolAContext,
    component_model: LMComponentModel,
    optimizer: torch.optim.Optimizer,
    ci_fn_params: list[nn.Parameter],
    ppgd_state: PersistentPGDState,
    batch_T: Any,
    cfg: _ThreePoolRuntime,
    strategy: ReconLossStrategy,
    step: int,
    n_steps: int,
    current_frac_of_training: float,
    *,
    should_log: bool,
) -> dict[str, float]:
    """One Pool A step. Trains the CI fn; updates the adversary sources; ships
    masks + V/U grads to chunkwise and receives the fresh V/U replica back."""
    device = next(component_model.parameters()).device
    all_sites = list(ctx.world.all_sites)

    ppgd_state.update_lr(step=step, total_steps=n_steps)
    batch_local, seq_len = _slice_batch_for_pool_a(batch_T, ctx)

    # A0 + A1 + A3 + A4 (warmup/recon) all share the residual-start + bypass context.
    trace(f"step_pool_a {step}: use_cached_residual enter (prefix fwd)")
    with component_model.use_cached_residual(batch_local), strategy.context():
        trace(f"step_pool_a {step}: shared target_forward")
        target_out, h_cache = _shared_target_forward(component_model, batch_local, cfg)
        trace(f"step_pool_a {step}: ci_fn forward")
        fwd = _ci_fn_forward(component_model, h_cache, ctx, cfg)
        mean_l0 = _mean_l0(fwd.ci.lower_leaky, ctx.world.n_ci) if should_log else 0.0
        trace(f"step_pool_a {step}: send CI masks to chunk")
        sends_to_chunk = ctx.portals.ci_to_chunk.send(ctx.role.as_ci(), fwd.ci.lower_leaky)
        trace(f"step_pool_a {step}: CI masks sent; adversary/imp")
        imp_loss = _importance_minimality_loss(
            fwd.ci.upper_leaky,
            current_frac_of_training,
            cfg,
            ci_pool_group=ctx.world.ci_pool_group,
            n_ci_pool=ctx.world.n_ci,
        )
        weight_deltas = component_model.calc_weight_deltas()
        ci_scratch = _releaf_ci_fp32_for_grads(fwd.ci.lower_leaky)
        _assert_ci_scratch_shapes(ci_scratch, ctx, seq_len, cfg)
        recon = _warmup_and_recon(
            ppgd_state, component_model, batch_local, target_out, ci_scratch, weight_deltas, cfg
        )

    raw = _autograd_grads_wrt_vu_ci_and_sources(
        recon.sum_loss, component_model, ci_scratch, all_sites, ppgd_state.sources
    )
    _scale_adversary_grads(raw, recon.n_examples, ctx, cfg)

    # Cross-pool exchange order is load-bearing: CI and the adversary are co-located on
    # ONE Pool A rank, so the chunkwise pool's send-g_CI / recv-g_VU pair (which the
    # 3-pool serviced on two DIFFERENT ranks concurrently) must be serviced here in the
    # SAME order the chunkwise step issues them — recv g_CI FIRST, then send g_VU — or
    # the two pools deadlock (each blocked on a send the other hasn't posted a recv for).
    # g_CI: chunkwise's contribution arrives over the wire; the adversary's is local.
    trace(f"step_pool_a {step}: recv g_CI from chunk (blocking cross-pool)")
    g_ci_chunk = ctx.portals.g_ci_from_chunk.recv(ctx.role.as_ci(), cfg.c_per_site, seq_len, device)

    # V/U grads: SUM-reduce across Pool A, then ship (leader-only) to chunk leaders.
    trace(f"step_pool_a {step}: g_CI recv done; sum-reduce g_VU over Pool A")
    sum_reduce_ppgd_grads(ctx.world, [*raw.v.values(), *raw.u.values()])
    trace(f"step_pool_a {step}: send g_VU to chunk")
    ctx.portals.g_vu_to_chunk.send(ctx.role.as_ppgd(), raw.v, raw.u)
    # Final (N+1)'th source step. For per-batch-per-position sources this is a no-op
    # reduce (per-rank-independent); for a broadcast (shared) source the AVG over Pool A
    # reassembles the full-batch grad — matching the warmup-step reduce in the state.
    ppgd_state.step(ppgd_state.reduce_source_grads(raw.sources))

    g_ci_total = _assemble_g_ci_total(g_ci_chunk, raw.ci, ctx, cfg, seq_len)

    optimizer.zero_grad(set_to_none=True)
    _fused_backward_through_ci_fn(imp_loss, fwd, g_ci_total, ctx, cfg)

    trace(f"step_pool_a {step}: all-reduce ci_fn grads over Pool A")
    in_flight_ci_grad_reduce = all_reduce_ci_fn_grads_async(ctx.world, ci_fn_params)
    in_flight_ci_grad_reduce.wait()
    trace(f"step_pool_a {step}: ci_fn grad all-reduce done")

    assert component_model.ci_fn is not None, "Pool A must hold its CI fn"
    grad_norms = (
        per_param_grad_norms(
            (f"ci_fns/{name}", p) for name, p in component_model.ci_fn.named_parameters()
        )
        if should_log
        else {}
    )

    if cfg.grad_clip_norm_ci_fn is not None:
        cross_pool_clip_grad_norm(
            ci_fn_params,
            cfg.grad_clip_norm_ci_fn,
            group=ctx.world.ci_pool_group,
            n_replicas=ctx.world.n_ci,
        )
    optimizer.step()

    trace(f"step_pool_a {step}: wait masks-send to chunk")
    sends_to_chunk.wait()

    trace(f"step_pool_a {step}: recv updated V/U from chunk (blocking cross-pool)")
    v_templates, u_templates = _vu_templates(component_model, all_sites)
    v_new, u_new = ctx.portals.updated_vu_from_chunk.post_recv(v_templates, u_templates).wait()
    trace(f"step_pool_a {step}: updated V/U recv done")
    _copy_vu_into_model_in_place(component_model, v_new, u_new, all_sites)

    if should_log:
        imp_value = imp_loss.item()
        metrics = {
            "loss/imp": imp_value,
            "_raw/imp_num": imp_value / ctx.world.n_ci,
            "_raw/l0": mean_l0,
            "loss/ppgd": (recon.sum_loss / recon.n_examples).item(),
            "_raw/ppgd_num": recon.sum_loss.item(),
            "_raw/ppgd_den": float(recon.n_examples),
            **grad_norms,
        }
    else:
        metrics = {}
    return metrics


def _shared_target_forward(
    component_model: LMComponentModel, batch_local: Any, cfg: _ThreePoolRuntime
) -> tuple[Tensor, dict[str, Tensor]]:
    """Phase A0. One no-grad target forward under the strategy's LM-head bypass,
    returning BOTH the recon target (the per-rank clean output the adversary
    reconstructs) and the per-site pre-weight-act cache the CI fn reads.

    The cache (captured before the decomposed weights) is independent of the LM-head
    bypass, so running under ``strategy.context()`` yields the bypassed hidden-state
    target for recon AND the correct CI input cache in a single forward. The cache is
    upcast to fp32 so the CI fn forward gets fp32 inputs (matching ``step_ci``).
    """
    with torch.no_grad(), bf16_autocast(cfg.bf16_autocast):
        target_out, cache = component_model.forward_with_pre_weight_acts(batch_local)
    h_cache = {k: v.to(torch.float32) for k, v in cache.items()}
    return target_out.detach(), h_cache


def _ci_fn_forward(
    component_model: LMComponentModel,
    h_cache: dict[str, Tensor],
    ctx: PoolAContext,
    cfg: _ThreePoolRuntime,
) -> CiForward:
    """Phase A1. CI fn forward → ``CIOutputs`` (graph retained for the fused backward)."""
    with bf16_autocast(cfg.bf16_autocast):
        ci = component_model.calc_causal_importances(
            pre_weight_acts=h_cache, sampling="continuous", detach_inputs=False
        )
    sample = next(iter(ci.lower_leaky.values()))
    assert sample.ndim == 3, f"expected CI shape [B, S, C]; got {sample.shape}"
    seq_len = sample.shape[1]
    batch_local_a = ctx.world.batch_local_ci
    for s, c in cfg.c_per_site.items():
        t = ci.lower_leaky[s]
        assert t.shape == (batch_local_a, seq_len, c), (
            f"ci.lower_leaky[{s!r}] shape {tuple(t.shape)} != ({batch_local_a}, {seq_len}, {c})"
        )
    return CiForward(ci=ci, seq_len=seq_len)


def _slice_batch_for_pool_a(batch: Any, ctx: PoolAContext) -> tuple[Any, int]:
    """Pull this Pool A rank's DP shard of the batch + extract its seq_len."""
    sl = ctx.role.batch_slice(ctx.world.batch_local_ci)
    batch_local = batch[sl] if isinstance(batch, Tensor) else batch
    if isinstance(batch_local, Tensor):
        seq_len = batch_local.shape[1] if batch_local.ndim >= 2 else 1
    else:
        assert isinstance(batch_local, dict) and "input_ids" in batch_local
        seq_len = batch_local["input_ids"].shape[1]
    return batch_local, seq_len


def _assert_ci_scratch_shapes(
    ci_scratch: dict[str, Tensor], ctx: PoolAContext, seq_len: int, cfg: _ThreePoolRuntime
) -> None:
    batch_local_a = ctx.world.batch_local_ppgd
    for s, c in cfg.c_per_site.items():
        t = ci_scratch[s]
        assert t.shape == (batch_local_a, seq_len, c), (
            f"ci_scratch[{s!r}] shape {tuple(t.shape)} != ({batch_local_a}, {seq_len}, {c})"
        )


def _scale_adversary_grads(
    raw: RawGrads, n_examples_local: int, ctx: PoolAContext, cfg: _ThreePoolRuntime
) -> None:
    """Apply the PPGD normalization in place. Identical to the 3-pool PPGD scale
    (``step_ppgd._scale_grads``), with ``n_ppgd_ranks`` → Pool A's batch arity ``n_a``.

    V/U and CI share one partial-sum scale ``coeff_ppgd / (n_examples_local * n_a)``
    (both are partial sums of the canonical PPGD loss; the Pool A V/U SUM-reduce and
    the Pool A CI-fn SUM-reduce each reassemble the full-batch grad). Sources are
    per-rank-local adversary state optimized against this rank's own recon mean —
    ``1 / n_examples_local``, no coeff, no ``1/n_a``.
    """
    n_a = ctx.world.n_ci
    vu_and_ci_grad_scale = cfg.coeff_ppgd / (n_examples_local * n_a)
    source_grad_scale = 1.0 / n_examples_local
    _scale_grads_in_place(raw.v, vu_and_ci_grad_scale)
    _scale_grads_in_place(raw.u, vu_and_ci_grad_scale)
    _scale_grads_in_place(raw.ci, vu_and_ci_grad_scale)
    _scale_grads_in_place(raw.sources, source_grad_scale)


def _assemble_g_ci_total(
    g_ci_chunk: dict[str, Tensor],
    g_ci_adversary: dict[str, Tensor],
    ctx: PoolAContext,
    cfg: _ThreePoolRuntime,
    seq_len: int,
) -> dict[str, Tensor]:
    """``g_CI_total[s] = g_CI_chunk[s] + g_CI_adversary[s]`` per site.

    ``g_ci_chunk`` arrives from Pool B on this rank's batch slice; ``g_ci_adversary``
    is the scaled local ``.grad`` on the re-leafed mask (same slice). Both are
    ``[B_local_a, S, C_s]``; loss coefficients are already baked in.
    """
    batch_local_a = ctx.world.batch_local_ci
    per_site: dict[str, Tensor] = {}
    for s in ctx.world.all_sites:
        c = cfg.c_per_site[s]
        chunk, adv = g_ci_chunk[s], g_ci_adversary[s]
        assert chunk.shape == (batch_local_a, seq_len, c), (
            f"g_ci_chunk[{s!r}] shape {tuple(chunk.shape)} != ({batch_local_a}, {seq_len}, {c})"
        )
        assert adv.shape == (batch_local_a, seq_len, c), (
            f"g_ci_adversary[{s!r}] shape {tuple(adv.shape)} != ({batch_local_a}, {seq_len}, {c})"
        )
        per_site[s] = chunk + adv
    return per_site


def _fused_backward_through_ci_fn(
    imp_loss: Tensor,
    fwd: CiForward,
    g_ci_total: dict[str, Tensor],
    ctx: PoolAContext,
    cfg: _ThreePoolRuntime,
) -> None:
    """Phase A7. Single fused backward through the CI fn graph (matches ``step_ci``).

    ``coeff_imp * imp_loss`` flows via ``ci.upper_leaky`` (its detached-global-residual
    backward is a local partial); ``g_ci_total[s]`` seeds ``ci.lower_leaky[s]``.
    """
    assert imp_loss.dim() == 0, f"imp_loss must be scalar; got {imp_loss.shape}"
    scaled_imp = cfg.coeff_imp * imp_loss
    lower_leaky_tensors = [fwd.ci.lower_leaky[s] for s in ctx.world.all_sites]
    g_ci_seeds = [g_ci_total[s] for s in ctx.world.all_sites]
    torch.autograd.backward(
        tensors=[*lower_leaky_tensors, scaled_imp],
        grad_tensors=[*g_ci_seeds, None],
    )
