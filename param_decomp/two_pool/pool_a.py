"""Pool-A training step: target+CI forward, layerwise streaming loss, home losses, opt step.

Pool A trains V/U + CI fn. The numbered phases are:

  0. Wait for any pending async weight ship from previous step.
  1. Target + CI fn forward — CI graph retained for the combined backward.
  2. Async-send CI values to pool B (overlaps with home-loss compute).
  3. Faithfulness loss (forward only; backward in phase 7).
  4. Importance-minimality loss (forward only; backward in phase 7).
  5. Streaming layerwise stoch recon — at the step level so the per-site loop
     is visible. One site per iter: build mask, forward through model, recon
     loss, immediate ``.backward()``. Peak memory bounded to ~1× iter
     activations.
  6. Recv per-site V/U grads + per-slice CI grads from pool B.
  7. Combined backward: home (faith+imp) + CI-fn graph seeded by combined
     (pool-B + layerwise) CI grads. Adds pool B's V/U grads to layerwise's.
  8. In-block DDP all-reduce on grads.
  8b. Cross-pool grad clip on the global norm (matches single-pool's
     ``clip_grad_norm_``).
  9. AdamW step.
  10. Async-ship updated V/U to pool B for next step.
  11. Wait on the CI send from phase 2.

``step_pool_a`` reads as ~40 lines of orchestration. The per-site layerwise
loop stays at the top level — that loop IS the structure of the step. The
named helpers below correspond to the other phases. Profiler tags
(``a/<n>_<name>``) are preserved.
"""

# pyright: reportArgumentType=false, reportOperatorIssue=false, reportAttributeAccessIssue=false

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from param_decomp.component_model import ComponentModel
from param_decomp.grad_clip import cross_pool_clip_grad_norm
from param_decomp.masks import make_mask_infos
from param_decomp.metrics.importance_minimality import (
    _finalize as _finalize_imp_min,
)
from param_decomp.metrics.importance_minimality import (
    _get_linear_annealed_p,
    _per_component_sums,
)
from param_decomp.two_pool.layout import BlockDDPLayout
from param_decomp.two_pool.loss_strategy import LayerwiseLossStrategy
from param_decomp.two_pool.profiler import PhaseProfiler
from param_decomp.two_pool.runtime import _TwoPoolRuntime, autocast_bf16


def _faithfulness_loss(
    component_model: ComponentModel, device: torch.device, numel_global: int
) -> tuple[Tensor, Tensor, int]:
    """Standard faithfulness loss: ``‖W_target − VU.T‖²_F / numel_global`` over this
    rank's owned sites.

    Single-pool computes ``sum_sq_global / numel_global``; the per-element
    gradient on each ``V_i`` / ``U_i`` is then ``∝ 1 / numel_global``. To match
    that gradient scale in multi-pool we still divide by ``numel_global`` (not
    rank-local ``numel_owned``) — otherwise per-element gradients scale up by
    ``numel_global / numel_owned`` and the trajectory diverges from single-pool
    (most visibly during the unclipped faithfulness warmup).

    Returns ``(scalar_loss, sum_sq, numel_owned)``. The raw ``numel_owned`` is
    what the logger needs as the denominator for the
    ``SUM(num) / SUM(den)`` global-ratio reconstruction across blocks.
    """
    weight_deltas = component_model.calc_weight_deltas()
    sum_sq = torch.zeros((), device=device)
    numel_owned = 0
    for d in weight_deltas.values():
        sum_sq = sum_sq + (d**2).sum()
        numel_owned += d.numel()
    assert sum_sq.dim() == 0, f"sum_sq should be scalar; got shape {sum_sq.shape}"
    assert numel_owned > 0, "rank must own at least one site's params"
    return sum_sq / numel_global, sum_sq, numel_owned


def _importance_minimality_loss(
    ci_upper: dict[str, Tensor],
    current_frac_of_training: float,
    cfg: _TwoPoolRuntime,
) -> Tensor:
    """Importance-minimality loss matching single-pool semantics.

    ``world_size=1`` because the pool-A target+CI forward (phase a/1) runs on
    the FULL global batch (only the layerwise stoch loss slices the batch —
    NOT the CI fn forward). Each rank's local ``sum`` over its owned sites
    already equals the global sum for those sites; cross-block aggregation is
    SUM-across-disjoint-site-sets and lives entirely in the logger.
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
    return _finalize_imp_min(
        per_component_sums=per_component_sums,
        n_examples=n_examples,
        beta=cfg.imp_min_beta,
        world_size=1,
    )


def step_pool_a(
    layout: BlockDDPLayout,
    component_model: ComponentModel,
    optimizer: torch.optim.Optimizer,
    all_params: list[nn.Parameter],
    component_params: list[nn.Parameter],
    ci_fn_params: list[nn.Parameter],
    batch: Any,
    cfg: _TwoPoolRuntime,
    strategy: LayerwiseLossStrategy,
    current_frac_of_training: float,
    profiler: PhaseProfiler | None = None,
) -> dict[str, float]:
    """One training step on a pool-A rank.

    Reads as orchestration: each phase delegates to a helper, except the
    semantically central per-site layerwise loop which stays inline so the
    step's structure is immediately visible.
    """
    p = profiler if profiler is not None else PhaseProfiler(enabled=False)
    n_sites_total = len(cfg.c_per_site)

    _wait_pending_weight_send(component_model, p)

    # ``strategy.context`` chooses logits-vs-hidden output of the model
    # forwards for this step; ``recon_loss`` matches the choice.
    with strategy.context(component_model.target_model):
        fwd = _target_and_ci_forward(component_model, batch, cfg, layout, p)
        ci_send = _async_send_owned_ci_to_pool_b(layout, fwd.ci, p)
        home = _home_losses_forward_only(component_model, fwd, cfg, current_frac_of_training, p)
        optimizer.zero_grad(set_to_none=True)

        # ── Phase a/5: per-site streaming layerwise loop ──
        # The CI-fn graph is retained from phase a/1; we re-leaf the sliced
        # CI values so each per-site backward stops at the leaf rather than
        # traversing the CI-fn graph (the combined backward in phase a/7
        # does that traversal once with the merged seed).
        with p.phase("a/5_layerwise"), autocast_bf16(cfg.bf16_autocast):
            sl = layout.my_batch_slice_a()
            batch_local = batch[sl] if isinstance(batch, Tensor) else batch
            target_local = fwd.target_out[sl].detach()
            ci_lower_leaves = _make_sliced_ci_leaves(fwd.ci, sl, layout.my_owned_sites)
            stoch_total = 0.0
            for site in layout.my_owned_sites:
                pred = _masked_forward_one_site(component_model, batch_local, ci_lower_leaves, site)
                loss, n_positions = strategy.recon_loss(pred=pred, target=target_local)
                assert loss.dim() == 0, f"recon_loss should return scalar; got {loss.shape}"
                # Per-site contribution: ``coeff_stoch * sum_kl_s / (n_pos *
                # n_sites_total)``. Backward per iter so this site's autograd
                # graph is freed before the next forward.
                (cfg.coeff_stoch * loss / (n_positions * n_sites_total)).backward()
                stoch_total += (loss / n_positions).item()
            stoch_n_owned = len(layout.my_owned_sites)
            loss_stoch_scalar = stoch_total / stoch_n_owned

    pool_b_grads = _recv_grads_from_pool_b(layout, component_model, fwd.ci, p)
    _combined_backward(
        component_model,
        fwd.ci,
        ci_lower_leaves,
        sl,
        pool_b_grads,
        home,
        layout,
        cfg,
        p,
    )
    _in_block_all_reduce_grads(layout, all_params, p)
    clip_norms = _cross_pool_grad_clip(component_params, ci_fn_params, layout, cfg, p)
    _optimizer_step(optimizer, p)
    weight_send = _async_send_updated_vu_to_pool_b(component_model, layout, p)
    _wait_async_ci_send(ci_send, p)
    _stash_pending_weight_send(component_model, weight_send)

    return _step_metrics(home, loss_stoch_scalar, stoch_total, stoch_n_owned, clip_norms)


# =============================================================================
# Local data bundles for threading state through phases without dropping the
# per-step orchestration function into a wall of positional args.
# =============================================================================


@dataclass
class _ForwardOutputs:
    """Output of phase a/1's combined target + CI fn forward."""

    target_out: Tensor  # logits or pre-LM-head hidden, per strategy
    ci: Any  # CIOutputs — contains upper_leaky + lower_leaky dicts


@dataclass
class _HomeLossOutputs:
    """Outputs of phases a/3 + a/4 (forward only — backward is in a/7)."""

    loss_faith: Tensor
    faith_sum_sq: Tensor
    faith_numel: int
    loss_imp: Tensor


@dataclass
class _PoolBGrads:
    v_grads: dict[str, Tensor]
    u_grads: dict[str, Tensor]
    ci_grads: dict[str, Tensor]


@dataclass
class _AsyncWorkHandle:
    """Generic (works, buffers) bundle for an async cross-pool send."""

    works: list[Any]
    buffers: list[Any]


@dataclass
class _ClipNorms:
    components: float
    ci_fn: float


# =============================================================================
# Phase helpers (other than the inline layerwise loop above).
# =============================================================================


def _wait_pending_weight_send(component_model: ComponentModel, p: PhaseProfiler) -> None:
    """Phase a/0. Wait for any pending async V/U send from the previous step.

    The previous step kicks off an async send in phase a/10 and stashes the
    handle on the model. We must wait for it before mutating V/U via the
    optimizer step — otherwise the wire reads inconsistent data.
    """
    with p.phase("a/0_wait_prev_weight_send"):
        pending = getattr(component_model, "_pending_weight_sends", None)
        if pending is not None:
            for w in pending[0]:
                w.wait()
            component_model._pending_weight_sends = None  # type: ignore[attr-defined]


def _target_and_ci_forward(
    component_model: ComponentModel,
    batch: Any,
    cfg: _TwoPoolRuntime,
    layout: BlockDDPLayout,
    p: PhaseProfiler,
) -> _ForwardOutputs:
    """Phase a/1. Target forward (cached) + CI fn forward.

    Inside ``bf16_autocast``: target_fwd's SDPA only has a fast kernel for
    bf16/fp16 on H200; fp32 falls back to O(N²) math backend. Faithfulness
    loss is computed OUTSIDE autocast (see ``_faithfulness_loss``).

    CI fn graph is retained — phase a/7 will backward through it.
    """
    with p.phase("a/1_target_and_ci_fwd"), autocast_bf16(cfg.bf16_autocast):
        out = component_model(batch, cache_type="input")
        target_out = out.output
        ci = component_model.calc_causal_importances(
            pre_weight_acts=out.cache,
            sampling="continuous",
            detach_inputs=False,
        )
        _assert_ci_shapes(ci, layout, cfg)
        _maybe_log_dtype_sanity(component_model, layout, target_out, out.cache, ci, cfg)
    return _ForwardOutputs(target_out=target_out, ci=ci)


def _assert_ci_shapes(ci: Any, layout: BlockDDPLayout, cfg: _TwoPoolRuntime) -> None:
    """CI lower_leaky/upper_leaky are dict[site → [batch, seq, C_s]] full-batch."""
    batch_global = cfg.batch_global
    for s in layout.my_owned_sites:
        assert s in ci.lower_leaky, f"CI missing site {s!r}"
        t = ci.lower_leaky[s]
        assert t.ndim == 3, f"ci.lower_leaky[{s!r}] expected 3D [B,S,C], got {t.shape}"
        assert t.shape[0] == batch_global, (
            f"ci.lower_leaky[{s!r}] batch dim {t.shape[0]} != batch_global {batch_global}"
        )
        assert t.shape[-1] == cfg.c_per_site[s], (
            f"ci.lower_leaky[{s!r}] C dim {t.shape[-1]} != c_per_site {cfg.c_per_site[s]}"
        )


def _maybe_log_dtype_sanity(
    component_model: ComponentModel,
    layout: BlockDDPLayout,
    target_out: Tensor,
    cache: dict[str, Tensor],
    ci: Any,
    cfg: _TwoPoolRuntime,
) -> None:
    """One-time dtype sanity print on step 0 from rank 0 only."""
    if getattr(component_model, "_dtype_logged", False) or layout.my_rank != 0:
        return
    sample_act = next(iter(cache.values()))
    print(
        f"[two_pool sanity rank0] bf16_autocast={cfg.bf16_autocast}  "
        f"use_fused_kl={cfg.use_fused_kl}  "
        f"target_out.dtype={target_out.dtype}  "
        f"target_out.shape={tuple(target_out.shape)}  "
        f"cached_act.dtype={sample_act.dtype}  "
        f"ci.lower_leaky.dtype={next(iter(ci.lower_leaky.values())).dtype}",
        flush=True,
    )
    component_model._dtype_logged = True  # type: ignore[attr-defined]


def _async_send_owned_ci_to_pool_b(
    layout: BlockDDPLayout,
    ci: Any,
    p: PhaseProfiler,
) -> _AsyncWorkHandle:
    """Phase a/2. Async-send owned CI values to pool B.

    Don't block here — pool B is doing its own target_fwd in parallel and
    will pull these values when it needs them. We wait at the end of the
    step (phase a/11) before the next iter mutates the underlying tensors.
    """
    with p.phase("a/2_async_send_ci"):
        works, buffers = layout.async_send_owned_ci_to_pool_b(
            {s: ci.lower_leaky[s] for s in layout.my_owned_sites}
        )
    return _AsyncWorkHandle(works=works, buffers=buffers)


def _home_losses_forward_only(
    component_model: ComponentModel,
    fwd: _ForwardOutputs,
    cfg: _TwoPoolRuntime,
    current_frac_of_training: float,
    p: PhaseProfiler,
) -> _HomeLossOutputs:
    """Phases a/3 + a/4. Compute faith and imp losses (forward only).

    Backward folds into the combined backward (a/7) — we want a single fused
    backward through faith + imp + CI-fn-seeded-from-layerwise-and-pool-B.
    """
    device = fwd.target_out.device
    with p.phase("a/3_faith"):
        loss_faith, faith_sum_sq, faith_numel = _faithfulness_loss(
            component_model, device, cfg.numel_global
        )
    with p.phase("a/4_imp"):
        loss_imp = _importance_minimality_loss(fwd.ci.upper_leaky, current_frac_of_training, cfg)
    assert loss_faith.dim() == 0 and loss_imp.dim() == 0, (
        f"home losses should be scalars; got faith={loss_faith.shape}, imp={loss_imp.shape}"
    )
    return _HomeLossOutputs(
        loss_faith=loss_faith,
        faith_sum_sq=faith_sum_sq,
        faith_numel=faith_numel,
        loss_imp=loss_imp,
    )


def _make_sliced_ci_leaves(
    ci: Any,
    sl: slice,
    owned_sites: tuple[str, ...],
) -> dict[str, Tensor]:
    """Build detached, requires-grad leaves of CI at this rank's batch slice.

    Layerwise's per-iter backward stops at these leaves rather than traversing
    the (retained) CI-fn graph; the combined backward in phase a/7 picks up
    the leaf .grads and merges them with pool B's CI grads to seed a single
    fused backward through the CI fn.
    """
    return {s: ci.lower_leaky[s][sl].detach().requires_grad_(True) for s in owned_sites}


def _masked_forward_one_site(
    component_model: ComponentModel,
    batch_local: Any,
    ci_lower_leaves: dict[str, Tensor],
    site: str,
) -> Tensor:
    """One layerwise iter's forward: build mask from CI leaf, run component
    model with that single-site mask active, return pred.

    The mask is ``ci + (1 - ci) * uniform`` — single-pool layerwise's standard
    stochastic mask. ``delta_mask`` is a per-site random gate on the
    ``target - V@U.T`` delta component.
    """
    ci_s = ci_lower_leaves[site]
    u = torch.rand_like(ci_s)
    mask = ci_s + (1 - ci_s) * u
    delta = component_model.target_weight(site) - component_model.components[site].weight
    delta_mask = torch.rand(ci_s.shape[:-1], device=ci_s.device, dtype=ci_s.dtype)
    mask_infos = make_mask_infos(
        {site: mask},
        weight_deltas_and_masks={site: (delta, delta_mask)},
        routing_masks="all",
    )
    return component_model(batch_local, mask_infos=mask_infos)


def _recv_grads_from_pool_b(
    layout: BlockDDPLayout,
    component_model: ComponentModel,
    ci: Any,
    p: PhaseProfiler,
) -> _PoolBGrads:
    """Phase a/6. Recv per-site V/U grads + per-slice CI grads from pool B.

    Pool B already SUM-reduced V/U grads internally and scaled by
    ``1/n_pool_b``, so the values arriving are the full-batch V/U
    contribution. ``ci_grads`` are full-batch CI tensors (pool B sees all
    positions for the rank's owned sites).
    """
    with p.phase("a/6_recv_grads_from_b"):
        v_templates = {s: component_model.components[s].V for s in layout.my_owned_sites}
        u_templates = {s: component_model.components[s].U for s in layout.my_owned_sites}
        ci_lower_owned_full = {s: ci.lower_leaky[s] for s in layout.my_owned_sites}
        v_grads, u_grads, ci_grads = layout.recv_grads_from_pool_b(
            v_templates,
            u_templates,
            ci_lower_owned_full,
        )
    for s in layout.my_owned_sites:
        assert v_grads[s].shape == component_model.components[s].V.shape, (
            f"pool-B v_grad shape mismatch for {s!r}: "
            f"{v_grads[s].shape} vs {component_model.components[s].V.shape}"
        )
        assert u_grads[s].shape == component_model.components[s].U.shape, (
            f"pool-B u_grad shape mismatch for {s!r}: "
            f"{u_grads[s].shape} vs {component_model.components[s].U.shape}"
        )
        assert ci_grads[s].shape == ci.lower_leaky[s].shape, (
            f"pool-B ci_grad shape mismatch for {s!r}: "
            f"{ci_grads[s].shape} vs {ci.lower_leaky[s].shape}"
        )
    return _PoolBGrads(v_grads=v_grads, u_grads=u_grads, ci_grads=ci_grads)


def _combined_backward(
    component_model: ComponentModel,
    ci: Any,
    ci_lower_leaves: dict[str, Tensor],
    sl: slice,
    pool_b: _PoolBGrads,
    home: _HomeLossOutputs,
    layout: BlockDDPLayout,
    cfg: _TwoPoolRuntime,
    p: PhaseProfiler,
) -> None:
    """Phase a/7. Fused backward through home losses + CI-fn graph.

    Two contributions need to merge before the CI-fn backward:

      * Pool B's ``ci_grads[s]`` are FULL-batch shape (pool B saw all
        positions for the rank's owned sites).
      * Layerwise's ``ci_lower_leaves[s].grad`` lives at this rank's batch
        slice ``sl`` only.

    We splat layerwise's slice grad into the full-batch pool-B tensor so the
    backward through ``ci.lower_leaky`` sees the combined seed. Pool B's V/U
    grads are simply added to the V/U .grad accumulator that layerwise
    already populated.
    """
    with p.phase("a/7_seed_and_backward"):
        combined_ci_grads: dict[str, Tensor] = {}
        for s in layout.my_owned_sites:
            grad = pool_b.ci_grads[s].clone()
            layerwise_slice_grad = ci_lower_leaves[s].grad
            assert layerwise_slice_grad is not None, (
                f"layerwise should have populated ci_lower_leaves[{s!r}].grad"
            )
            assert layerwise_slice_grad.shape == grad[sl].shape, (
                f"layerwise slice grad shape {layerwise_slice_grad.shape} != "
                f"target slice shape {grad[sl].shape} for site {s!r}"
            )
            grad[sl] += layerwise_slice_grad
            combined_ci_grads[s] = grad
        for s in layout.my_owned_sites:
            comp = component_model.components[s]
            assert comp.V.grad is not None and comp.U.grad is not None, (
                "layerwise should have populated V/U .grad"
            )
            comp.V.grad.add_(pool_b.v_grads[s])
            comp.U.grad.add_(pool_b.u_grads[s])
        total_home = cfg.coeff_faith * home.loss_faith + cfg.coeff_imp * home.loss_imp
        assert total_home.dim() == 0, f"total_home should be scalar; got {total_home.shape}"
        torch.autograd.backward(
            tensors=[total_home, *(ci.lower_leaky[s] for s in layout.my_owned_sites)],
            grad_tensors=[None, *(combined_ci_grads[s] for s in layout.my_owned_sites)],
        )


def _in_block_all_reduce_grads(
    layout: BlockDDPLayout,
    all_params: list[nn.Parameter],
    p: PhaseProfiler,
) -> None:
    """Phase a/8. In-block DDP all-reduce on all params' grads.

    DDP partners within a block share the same site set; they batch-shard
    layerwise (different slices) but compute identical faith/imp
    (data-independent). AVG-within-block reconciles to the correct
    full-batch grad.
    """
    with p.phase("a/8_in_block_allreduce"):
        layout.all_reduce_grads_in_block(all_params)


def _cross_pool_grad_clip(
    component_params: list[nn.Parameter],
    ci_fn_params: list[nn.Parameter],
    layout: BlockDDPLayout,
    cfg: _TwoPoolRuntime,
    p: PhaseProfiler,
) -> _ClipNorms:
    """Phase a/8b. Clip global gradient norm across all pool-A blocks.

    Single-pool's ``clip_grad_norm_`` works on a single rank's identical-via-
    DDP grads; in multi-pool, parameters are sharded across blocks, so we
    must compute the cross-block norm. ``cross_pool_clip_grad_norm`` does the
    all-reduce-SUM with ``/n_per_block`` dedup (DDP partners within a block
    hold identical grads post-phase-a/8).
    """
    components_norm = 0.0
    ci_fn_norm = 0.0
    with p.phase("a/8b_grad_clip"):
        n_per_block = layout.world.n_per_block
        if cfg.grad_clip_norm_components is not None:
            norm_t = cross_pool_clip_grad_norm(
                component_params,
                cfg.grad_clip_norm_components,
                group=layout.world.pool_a_group,
                n_replicas=n_per_block,
            )
            assert norm_t.dim() == 0, f"clip norm should be scalar; got {norm_t.shape}"
            components_norm = norm_t.item()
        if cfg.grad_clip_norm_ci_fn is not None:
            norm_t = cross_pool_clip_grad_norm(
                ci_fn_params,
                cfg.grad_clip_norm_ci_fn,
                group=layout.world.pool_a_group,
                n_replicas=n_per_block,
            )
            ci_fn_norm = norm_t.item()
    return _ClipNorms(components=components_norm, ci_fn=ci_fn_norm)


def _optimizer_step(optimizer: torch.optim.Optimizer, p: PhaseProfiler) -> None:
    """Phase a/9. AdamW step on V/U + CI fn params."""
    with p.phase("a/9_opt_step"):
        optimizer.step()


def _async_send_updated_vu_to_pool_b(
    component_model: ComponentModel,
    layout: BlockDDPLayout,
    p: PhaseProfiler,
) -> _AsyncWorkHandle:
    """Phase a/10. Async-ship updated V/U back to pool B for next step.

    The wait happens at the START of the next step (phase a/0), so this send
    runs concurrently with pool B's next iter target_fwd + PPGD warmup.
    """
    with p.phase("a/10_async_send_weights"):
        v_owned = {s: component_model.components[s].V for s in layout.my_owned_sites}
        u_owned = {s: component_model.components[s].U for s in layout.my_owned_sites}
        works, buffers = layout.async_send_updated_weights_to_pool_b(v_owned, u_owned)
    return _AsyncWorkHandle(works=works, buffers=buffers)


def _wait_async_ci_send(ci_send: _AsyncWorkHandle, p: PhaseProfiler) -> None:
    """Phase a/11. Wait on the CI send from phase a/2.

    Must complete before the next step's CI forward mutates the underlying
    ci.lower_leaky storage that the send buffers reference.
    """
    with p.phase("a/11_wait_async_ci_send"):
        for w in ci_send.works:
            w.wait()
        ci_send.buffers.clear()


def _stash_pending_weight_send(
    component_model: ComponentModel,
    weight_send: _AsyncWorkHandle,
) -> None:
    """Stash the weight-send handle on the model for next step's phase a/0 wait."""
    component_model._pending_weight_sends = (  # type: ignore[attr-defined]
        weight_send.works,
        weight_send.buffers,
    )


def _step_metrics(
    home: _HomeLossOutputs,
    loss_stoch_scalar: float,
    stoch_raw_num: float,
    stoch_raw_den: int,
    clip_norms: _ClipNorms,
) -> dict[str, float]:
    """Per-step metrics dict: per-rank display scalars + raw ingredients for
    the cross-block logger reduction + pre-clip grad norms.

    Raw ``_raw/<loss>_{num,den}`` are designed so the global value is
    ``SUM(num)/SUM(den)`` (for ratio losses faith and stoch) or ``SUM(num)``
    (for additive losses imp). See ``two_pool/reductions.py``.
    """
    return {
        "loss/faith": home.loss_faith.item(),
        "loss/imp": home.loss_imp.item(),
        "loss/stoch": loss_stoch_scalar,
        "_raw/faith_num": home.faith_sum_sq.item(),
        "_raw/faith_den": float(home.faith_numel),
        "_raw/imp_num": home.loss_imp.item(),
        "_raw/stoch_num": stoch_raw_num,
        "_raw/stoch_den": float(stoch_raw_den),
        "grad_norm/components": clip_norms.components,
        "grad_norm/ci_fn": clip_norms.ci_fn,
    }


def run_faithfulness_warmup_pool_a(
    *,
    component_model: ComponentModel,
    component_params: list[nn.Parameter],
    n_steps: int,
    lr: float,
    weight_decay: float,
    numel_global: int,
) -> None:
    """Single-pool-equivalent faithfulness warmup on pool A only.

    Pool B has no V/U params to optimize, so the warmup is a no-op there.
    Done once at startup so 2-pool runs initialize the same way as single-pool.
    """
    warmup_opt = torch.optim.AdamW(component_params, lr=lr, weight_decay=weight_decay)
    for _ in range(n_steps):
        warmup_opt.zero_grad()
        device = component_params[0].device
        loss, _, _ = _faithfulness_loss(component_model, device, numel_global)
        loss.backward()
        warmup_opt.step()
    del warmup_opt
    torch.cuda.empty_cache()
