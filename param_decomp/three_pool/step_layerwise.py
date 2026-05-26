"""Layerwise pool training step.

Recast of ``two_pool.pool_a.step_pool_a`` with the CI-fn-side bits moved out
to the CI pool, plus an optional async pipeline (``defer_vu_opt=True``) that
hides the V/U opt step + V/U ship-back behind T+1's CI fn forward window on
the CI pool.

Phases (numbered to match ``DESIGN.md`` ``lw/N``):

  A1. Post async irecv for CI_T from the owning CI rank (overlaps with A2).
  A2. target_fwd(batch_T) → L_T on this rank's batch slice.
  B.  Async mode only: finalize prev iter's deferred all_reduce → grad clip →
      AdamW step → async ship V/U. Blocking waits here overlap with A2's
      kernels (default CUDA stream) on the GPU.
  C.  Zero ``param.grad`` for fresh accumulation.
  D1. Faithfulness loss + backward (V/U-only, doesn't need CI).
  D2. Wait CI recv; re-leaf as fp32 ``requires_grad=True`` so the layerwise
      backward populates ``leaf.grad`` for D4.
  D3. Layerwise stoch recon, streaming per owned site — the
      semantically meaningful for-loop lives at this step level (one
      forward+backward per site bounds peak activation memory).
  D4. Send g_CI back to CI pool (per-rank, on owned sites).
  D5. Recv g_VU from PPGD pool (block leader recvs, then in-block bcast).
  D6. Combine V/U grads: faith + layerwise already in ``.grad``; add PPGD's.
  E.  Tail. Sync mode: in-block all_reduce → grad clip → AdamW → async send
      V/U. Async mode: kickoff async all_reduce, return its pending state.

Phases 1 + 3 give the headline overlap (CI recv on the NIC while
faithfulness runs on the GPU). The phase-B + phase-E async kickoff give the
deferred-mode overlap (they hide behind T+1's CI fn forward window on CI pool).
"""

# pyright: reportArgumentType=false, reportOperatorIssue=false, reportAttributeAccessIssue=false

from typing import Any

import torch
import torch.distributed as dist  # noqa: F401  (used in type hints)
import torch.nn as nn
from torch import Tensor

from param_decomp._trace import phase_trace_enabled, trace
from param_decomp.component_model import ComponentModel
from param_decomp.grad_clip import cross_pool_clip_grad_norm
from param_decomp.masks import make_mask_infos
from param_decomp.three_pool.layout import ThreePoolLayout
from param_decomp.three_pool.profiler import PhaseProfiler
from param_decomp.three_pool.runtime import _ThreePoolRuntime
from param_decomp.two_pool.loss_strategy import LayerwiseLossStrategy
from param_decomp.two_pool.runtime import autocast_bf16

PendingAllReduce = list[tuple[list[Tensor], Tensor, "dist.Work"]]


def step_layerwise(
    layout: ThreePoolLayout,
    component_model: ComponentModel,
    optimizer: torch.optim.Optimizer,
    all_params: list[nn.Parameter],
    batch: Any,
    cfg: _ThreePoolRuntime,
    strategy: LayerwiseLossStrategy,
    *,
    defer_vu_opt: bool,
    prev_pending_all_reduce: PendingAllReduce | None,
    should_log: bool,
    profiler: PhaseProfiler | None = None,
) -> tuple[dict[str, float], PendingAllReduce | None]:
    """One LW step. Branches on ``defer_vu_opt`` for sync vs async pipeline.

    Sync (``defer_vu_opt=False``): A → D → sync tail (all_reduce, clip, opt,
    async send V/U). Returns ``(metrics, None)``.

    Async (``defer_vu_opt=True``): A → B-finalize-prev (concurrent w/ A2) →
    D → kickoff async all_reduce, return its pending state. Caller must
    thread it across iters and drain via ``finalize_layerwise_async_drain``.
    """
    p = profiler if profiler is not None else PhaseProfiler(enabled=False)
    assert layout.my_pool == "layerwise"
    device = next(component_model.parameters()).device
    n_sites_total = len(cfg.c_per_site)

    batch_local, seq_len = _slice_batch_for_layerwise(batch, layout)

    with strategy.context(component_model.target_model):
        with p.phase("lw/A1_post_async_recv_ci"):
            ci_recv, ci_recv_works = layout.async_recv_ci_from_ci_pool(
                {s: cfg.c_per_site[s] for s in layout.my_owned_sites},
                seq_len=seq_len,
                device=device,
            )
        with p.phase("lw/A2_target_fwd"), torch.no_grad(), autocast_bf16(cfg.bf16_autocast):
            target_local = component_model(batch_local).detach()

    if defer_vu_opt and prev_pending_all_reduce is not None:
        _finalize_prev_iter_async(
            layout, component_model, optimizer, all_params, prev_pending_all_reduce, cfg, p
        )

    for param in all_params:
        param.grad = None

    with strategy.context(component_model.target_model):
        with p.phase("lw/D1_faith"):
            loss_faith, faith_sum_sq_t, faith_numel = _faithfulness_loss(
                component_model, device, cfg.numel_global
            )
            (cfg.coeff_faith * loss_faith).backward()

        with p.phase("lw/D2_wait_ci_recv"):
            for w in ci_recv_works:
                w.wait()
        ci_recv_leaves = _releaf_ci_fp32_for_grads(ci_recv, layout.my_owned_sites)
        _assert_ci_recv_shapes(ci_recv_leaves, layout, seq_len, cfg)

        with p.phase("lw/D3_layerwise"), autocast_bf16(cfg.bf16_autocast):
            # Accumulate the display value as a GPU tensor (not a Python float) so
            # the per-site ``.item()`` doesn't force a CPU↔GPU sync that serializes
            # each site's bwd against the next. ``loss_s.detach()`` so accumulator
            # doesn't retain autograd graph.
            stoch_total_t = torch.zeros((), device=device)
            for i, s in enumerate(layout.my_owned_sites):
                if phase_trace_enabled():
                    trace(f"lw/D3 site {i + 1}/{len(layout.my_owned_sites)}: {s} fwd+bwd")
                loss_s, n_positions = _layerwise_one_site(
                    component_model, batch_local, target_local, ci_recv_leaves, s, strategy
                )
                assert loss_s.dim() == 0, f"layerwise loss for site {s!r} must be scalar"
                (cfg.coeff_stoch * loss_s / (n_positions * n_sites_total)).backward()
                stoch_total_t = stoch_total_t + (loss_s.detach() / n_positions)
            stoch_n_owned = len(layout.my_owned_sites)

        with p.phase("lw/D4_send_g_ci"):
            g_ci_owned = {s: ci_recv_leaves[s].grad for s in layout.my_owned_sites}
            assert all(g is not None for g in g_ci_owned.values()), (
                "layerwise backward should have populated ci_recv_leaves[s].grad"
            )
            layout.send_g_ci_to_ci_pool(g_ci_owned)

        v_grads_pgd, u_grads_pgd = _recv_g_vu_from_ppgd(layout, component_model, p)
        _combine_vu_grads_in_place(component_model, layout, v_grads_pgd, u_grads_pgd, p)

    if should_log:
        stoch_total_value = stoch_total_t.item()
        metrics = {
            "loss/faith": loss_faith.item(),
            "loss/stoch": stoch_total_value / stoch_n_owned,
            "_raw/faith_num": faith_sum_sq_t.item(),
            "_raw/faith_den": float(faith_numel),
            "_raw/stoch_num": stoch_total_value,
            "_raw/stoch_den": float(stoch_n_owned),
        }
    else:
        metrics = {}

    if defer_vu_opt:
        with p.phase("lw/E_kickoff_async_allreduce"):
            new_pending = layout.async_all_reduce_grads_in_block_kickoff(all_params)
        return metrics, new_pending

    _sync_tail(layout, component_model, optimizer, all_params, cfg, p)
    return metrics, None


def finalize_layerwise_async_drain(
    layout: ThreePoolLayout,
    component_model: ComponentModel,
    optimizer: torch.optim.Optimizer,
    all_params: list[nn.Parameter],
    pending_all_reduce: PendingAllReduce,
    grad_clip_norm: float | None,
) -> None:
    """End-of-training drain in async mode: finish the final iter's deferred
    opt step. Skips the V/U async send since training is over.
    """
    assert layout.my_pool == "layerwise"
    _wait_pending_weight_send(component_model)
    layout.wait_and_unflatten_all_reduce(pending_all_reduce)
    if grad_clip_norm is not None:
        cross_pool_clip_grad_norm(
            all_params,
            grad_clip_norm,
            group=layout.world.layerwise_pool_group,
            n_replicas=layout.world.n_per_block,
        )
    optimizer.step()


def run_faithfulness_warmup_layerwise(
    *,
    component_model: ComponentModel,
    component_params: list[nn.Parameter],
    n_steps: int,
    lr: float,
    weight_decay: float,
    numel_global: int,
) -> None:
    """Single-pool-equivalent faithfulness warmup on the LW pool only.

    CI pool has no V/U; PPGD pool's V/U is a transient replica that gets
    overwritten each step. So warmup only makes sense on LW. Mirrors
    ``two_pool.pool_a.run_faithfulness_warmup_pool_a``.
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


def _slice_batch_for_layerwise(batch: Any, layout: ThreePoolLayout) -> tuple[Any, int]:
    """Pull this LW rank's batch slice + extract its seq_len."""
    sl = layout.my_batch_slice_lw()
    batch_local = batch[sl] if isinstance(batch, Tensor) else batch
    if isinstance(batch_local, Tensor):
        seq_len = batch_local.shape[1] if batch_local.ndim >= 2 else 1
    else:
        assert isinstance(batch_local, dict) and "input_ids" in batch_local
        seq_len = batch_local["input_ids"].shape[1]
    return batch_local, seq_len


def _finalize_prev_iter_async(
    layout: ThreePoolLayout,
    component_model: ComponentModel,
    optimizer: torch.optim.Optimizer,
    all_params: list[nn.Parameter],
    prev_pending_all_reduce: PendingAllReduce,
    cfg: _ThreePoolRuntime,
    p: PhaseProfiler,
) -> None:
    """Phase lw/B (async mode only). Finish the previous iter's deferred opt.

    Sequence: wait pending V/U send → wait + unflatten the prev all_reduce →
    cross-pool grad clip → AdamW step → async ship updated V/U to PPGD.
    Blocking waits in here overlap with phase A2's target_fwd kernels (which
    run on the default CUDA stream while NCCL waits hit their own stream).
    """
    with p.phase("lw/B1_wait_prev_weight_send"):
        _wait_pending_weight_send(component_model)
    with p.phase("lw/B2_wait_and_unflatten_allreduce"):
        layout.wait_and_unflatten_all_reduce(prev_pending_all_reduce)
    if cfg.grad_clip_norm_components is not None:
        with p.phase("lw/B2b_grad_clip"):
            cross_pool_clip_grad_norm(
                all_params,
                cfg.grad_clip_norm_components,
                group=layout.world.layerwise_pool_group,
                n_replicas=layout.world.n_per_block,
            )
    with p.phase("lw/B3_opt_step"):
        optimizer.step()
    with p.phase("lw/B4_async_send_vu"):
        _async_send_owned_vu_to_ppgd(component_model, layout)


def _releaf_ci_fp32_for_grads(
    ci_recv: dict[str, Tensor], owned_sites: tuple[str, ...]
) -> dict[str, Tensor]:
    """Upcast CI (bf16 on the wire) to fp32 and re-leaf with ``requires_grad=True``
    so the layerwise backward populates ``leaf.grad`` that the CI pool merges
    into its CI-fn fp32 grads.
    """
    return {
        s: ci_recv[s].detach().to(torch.float32).clone().requires_grad_(True) for s in owned_sites
    }


def _assert_ci_recv_shapes(
    ci_recv_leaves: dict[str, Tensor],
    layout: ThreePoolLayout,
    seq_len: int,
    cfg: _ThreePoolRuntime,
) -> None:
    """Sanity-check the CI leaves match what the CI pool said it'd send.

    Catches a wrong ``c_per_site`` config or a per-rank batch mismatch fast.
    """
    batch_local_lw = layout.world.batch_local_lw
    for s in layout.my_owned_sites:
        c = cfg.c_per_site[s]
        t = ci_recv_leaves[s]
        assert t.shape == (batch_local_lw, seq_len, c), (
            f"ci_recv_leaves[{s!r}] shape {tuple(t.shape)} != "
            f"expected ({batch_local_lw}, {seq_len}, {c})"
        )


def _layerwise_one_site(
    component_model: ComponentModel,
    batch_local: Any,
    target_local: Tensor,
    ci_recv_leaves: dict[str, Tensor],
    site: str,
    strategy: LayerwiseLossStrategy,
) -> tuple[Tensor, int]:
    """Phase lw/D3 (per-site body). One stochastic masked forward + recon.

    Returns ``(sum_loss, n_positions)`` raw — caller scales by
    ``coeff_stoch / (n_positions * n_sites_total)`` and calls ``backward()``
    so the per-site graph is freed between iterations (bounds peak memory).
    """
    ci_s = ci_recv_leaves[site]
    u = torch.rand_like(ci_s)
    mask = ci_s + (1 - ci_s) * u
    delta = component_model.target_weight(site) - component_model.components[site].weight
    delta_mask = torch.rand(ci_s.shape[:-1], device=ci_s.device, dtype=ci_s.dtype)
    mask_infos = make_mask_infos(
        {site: mask},
        weight_deltas_and_masks={site: (delta, delta_mask)},
        routing_masks="all",
    )
    pred = component_model(batch_local, mask_infos=mask_infos)
    loss, n_positions = strategy.recon_loss(pred=pred, target=target_local)
    return loss, n_positions


def _recv_g_vu_from_ppgd(
    layout: ThreePoolLayout,
    component_model: ComponentModel,
    p: PhaseProfiler,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Phase lw/D5. Recv V/U grads from PPGD pool (leader recvs, in-block bcast)."""
    with p.phase("lw/D5_recv_g_vu_from_ppgd"):
        v_templates = {s: component_model.components[s].V for s in layout.my_owned_sites}
        u_templates = {s: component_model.components[s].U for s in layout.my_owned_sites}
        v_grads_pgd, u_grads_pgd = layout.recv_g_vu_from_ppgd(v_templates, u_templates)
    for s in layout.my_owned_sites:
        assert v_grads_pgd[s].shape == component_model.components[s].V.shape, (
            f"v_grads_pgd[{s!r}] shape mismatch from PPGD send"
        )
        assert u_grads_pgd[s].shape == component_model.components[s].U.shape, (
            f"u_grads_pgd[{s!r}] shape mismatch from PPGD send"
        )
    return v_grads_pgd, u_grads_pgd


def _combine_vu_grads_in_place(
    component_model: ComponentModel,
    layout: ThreePoolLayout,
    v_grads_pgd: dict[str, Tensor],
    u_grads_pgd: dict[str, Tensor],
    p: PhaseProfiler,
) -> None:
    """Phase lw/D6. Add PPGD's V/U grads to .grad (which already has faith+lw)."""
    with p.phase("lw/D6_combine_vu_grads"):
        for s in layout.my_owned_sites:
            comp = component_model.components[s]
            assert comp.V.grad is not None and comp.U.grad is not None, (
                "faith + layerwise should have populated V/U .grad"
            )
            comp.V.grad.add_(v_grads_pgd[s])
            comp.U.grad.add_(u_grads_pgd[s])


def _sync_tail(
    layout: ThreePoolLayout,
    component_model: ComponentModel,
    optimizer: torch.optim.Optimizer,
    all_params: list[nn.Parameter],
    cfg: _ThreePoolRuntime,
    p: PhaseProfiler,
) -> None:
    """Phase lw/E (sync mode). Blocking all_reduce → clip → AdamW → async send V/U.

    Functionally equivalent to the pre-deferral 2-pool tail; safe to coexist
    with PPGD's sync recv at end of step T.
    """
    with p.phase("lw/E1_wait_prev_weight_send"):
        _wait_pending_weight_send(component_model)
    with p.phase("lw/E2_in_block_allreduce"):
        layout.all_reduce_grads_in_block(all_params)
    if cfg.grad_clip_norm_components is not None:
        with p.phase("lw/E2b_grad_clip"):
            cross_pool_clip_grad_norm(
                all_params,
                cfg.grad_clip_norm_components,
                group=layout.world.layerwise_pool_group,
                n_replicas=layout.world.n_per_block,
            )
    with p.phase("lw/E3_opt_step"):
        optimizer.step()
    with p.phase("lw/E4_async_send_vu"):
        _async_send_owned_vu_to_ppgd(component_model, layout)


def _async_send_owned_vu_to_ppgd(component_model: ComponentModel, layout: ThreePoolLayout) -> None:
    """Kickoff async ship of updated V/U → PPGD. Stash handles on the model."""
    v_owned = {s: component_model.components[s].V for s in layout.my_owned_sites}
    u_owned = {s: component_model.components[s].U for s in layout.my_owned_sites}
    weight_send_works, weight_send_buffers = layout.async_send_updated_vu_to_ppgd(v_owned, u_owned)
    component_model._pending_weight_sends = (  # type: ignore[attr-defined]
        weight_send_works,
        weight_send_buffers,
    )


def _wait_pending_weight_send(component_model: ComponentModel) -> None:
    """Wait + clear any pending async V/U send from a previous iter.

    Defense against the opt step mutating V/U while the previous async send
    still reads it.
    """
    pending = getattr(component_model, "_pending_weight_sends", None)
    if pending is not None:
        for w in pending[0]:
            w.wait()
        component_model._pending_weight_sends = None  # type: ignore[attr-defined]


def _faithfulness_loss(
    component_model: ComponentModel, device: torch.device, numel_global: int
) -> tuple[Tensor, Tensor, int]:
    """‖W_target − VU.T‖²_F / numel_global, summed across this rank's owned sites.

    See ``two_pool.pool_a._faithfulness_loss`` for why we divide by
    ``numel_global`` not ``numel_owned`` — keeps per-element grad scale aligned
    with single-pool's, so the unclipped faithfulness warmup converges to the
    same V/U as single-pool.

    Returns ``(scalar_loss, sum_sq, numel_owned)`` — the ``numel_owned`` is the
    denominator the logger uses for ``SUM(num) / SUM(den)`` global-ratio
    reconstruction across blocks.
    """
    weight_deltas = component_model.calc_weight_deltas()
    sum_sq = torch.zeros((), device=device)
    numel_owned = 0
    for d in weight_deltas.values():
        sum_sq = sum_sq + (d**2).sum()
        numel_owned += d.numel()
    return sum_sq / numel_global, sum_sq, numel_owned
