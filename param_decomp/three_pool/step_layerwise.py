"""Layerwise pool training step — split into ``_main`` and ``_tail``.

Recast of ``two_pool.pool_a.step_pool_a`` with the CI-fn-side bits moved out
to the CI pool. Split into two functions so the V/U opt step + V/U ship-back
can either run at end of step T (sync mode, default) OR be deferred to start
of step T+1 (when ``ThreePoolConfig.defer_vu_opt=True``). The runner in
``optimize.py`` decides; the two functions don't know.

``step_layerwise_main`` phases (numbered to match ``DESIGN.md`` `lw/N`):

  1. Post async irecv for CI_T from the owning CI rank. Runs concurrently
     with target_fwd below.
  2. target_fwd(batch_T) → L_T (this rank's batch slice). Strategy controls
     whether L_T is logits or pre-LM-head hidden state.
  3. Faithfulness loss + backward (V/U-only; doesn't need CI). Runs
     concurrently with the CI recv on the NIC.
  4. Wait for CI recv. Upcast bf16 → fp32 and re-leaf with requires_grad=True
     so the layerwise backward populates leaf.grad → shipped to CI pool at
     step 6.
  5. Layerwise stoch recon (per owned site, streaming) — populates
     ``ci_recv_leaves[s].grad`` and V/U .grad (additive on top of faith's).
  6. Synchronous send of g_CI_LW back to CI pool.
  7. Recv g_VU_PPGD from PPGD pool (block leader recvs, then in-block bcast).
  8. Combine V/U grads: layerwise + faith already in .grad; add PPGD's.

After ``_main`` returns, V/U ``.grad`` is fully populated for the AdamW step
in ``_tail``. ``_main`` clears ``.grad`` to None at the top (replacing what
the optimizer would do — keeps the optimizer out of ``_main``'s signature).

``step_layerwise_tail`` phases:

  0. Wait for pending async weight-ship from the previous tail (so V/U isn't
     mutated by the opt step while the previous send still reads it).
  9. In-block all-reduce on V/U + faith grads (DDP within block group).
  10. AdamW step on V/U.
  11. Async ship updated V/U → PPGD pool. The work handle is stashed on
      ``component_model._pending_weight_sends`` and waited on by the *next*
      tail (phase 0).

Phases 1 + 3 give the headline overlap: CI recv on the NIC while faithfulness
runs on the GPU. Phases 9-11 give the deferred-mode overlap: they hide behind
T+1's CI fn forward window on CI pool.
"""

# pyright: reportArgumentType=false, reportOperatorIssue=false, reportAttributeAccessIssue=false

from typing import Any

import torch
import torch.distributed as dist  # noqa: F401  (used in type hints)
import torch.nn as nn
from torch import Tensor

from param_decomp.component_model import ComponentModel
from param_decomp.grad_clip import cross_pool_clip_grad_norm
from param_decomp.masks import make_mask_infos
from param_decomp.three_pool.layout import ThreePoolLayout
from param_decomp.three_pool.profiler import PhaseProfiler
from param_decomp.three_pool.runtime import _ThreePoolRuntime
from param_decomp.two_pool.loss_strategy import LayerwiseLossStrategy
from param_decomp.two_pool.runtime import autocast_bf16


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


def _layerwise_loss_streaming(
    component_model: ComponentModel,
    batch_local: Any,
    target_local: Tensor,
    ci_recv_leaves: dict[str, Tensor],
    owned_sites: tuple[str, ...],
    recon_loss: Any,
    coeff_stoch: float,
    n_sites_total: int,
) -> tuple[float, float, int]:
    """Per-site streaming layerwise loss. Identical contract to
    ``two_pool.pool_a._layerwise_loss_streaming`` — backward per iter so peak
    memory stays bounded to ~1×iter.

    Per-site contribution scales as
    ``coeff_stoch * sum_kl_s / (n_positions * n_sites_total)`` so the same
    YAML coefficient transfers from 2-pool / single-pool trainers.

    Returns ``(scalar_value, raw_num, raw_den)``: scalar is per-rank
    ``total_value / n_owned``; raw num/den let the logger compute global mean
    via ``SUM(num) / SUM(den)`` across the LW pool.
    """
    n_owned = len(owned_sites)
    total_value = 0.0
    for s in owned_sites:
        ci_s = ci_recv_leaves[s]
        u = torch.rand_like(ci_s)
        mask = ci_s + (1 - ci_s) * u
        delta = component_model.target_weight(s) - component_model.components[s].weight
        delta_mask = torch.rand(ci_s.shape[:-1], device=ci_s.device, dtype=ci_s.dtype)
        mask_infos = make_mask_infos(
            {s: mask},
            weight_deltas_and_masks={s: (delta, delta_mask)},
            routing_masks="all",
        )
        pred = component_model(batch_local, mask_infos=mask_infos)
        loss, n_positions = recon_loss(pred=pred, target=target_local)
        scaled = coeff_stoch * loss / (n_positions * n_sites_total)
        scaled.backward()
        total_value += (loss / n_positions).item()
    return total_value / n_owned, total_value, n_owned


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
    prev_pending_all_reduce: list[tuple[list[Tensor], Tensor, "dist.Work"]] | None,
    profiler: PhaseProfiler | None = None,
) -> tuple[dict[str, float], list[tuple[list[Tensor], Tensor, "dist.Work"]] | None]:
    """One LW step. Branches on ``defer_vu_opt`` for sync vs async pipeline.

    Sync (``defer_vu_opt=False``):
      Phase A (V/U-independent) → Phase D (V/U-dependent) → Phase E sync tail.
      Equivalent to vanilla SPD step: post recv_ci, target_fwd, faith, layerwise,
      combine grads, sync all_reduce, AdamW, async ship V/U. Returns
      ``(metrics, None)`` — no pending all_reduce state across iterations.

    Async (``defer_vu_opt=True``):
      Phase A → finalize prev iter's grads (concurrent w/ target_fwd's kernels)
      → Phase D → kickoff async all_reduce on this iter's grads → return state.

      The finalize block runs ``wait_pending_send → wait_and_unflatten →
      AdamW → async send_vu``. While Python blocks on the wait, target_fwd's
      kernels run on the default CUDA stream and the all_reduce runs on its
      NCCL stream — real concurrency. The wait+opt+send completes before
      Phase D needs V/U.

      Caller must thread ``prev_pending_all_reduce`` across iterations
      (``None`` on iter 0; the returned state on subsequent iters). After the
      training loop, caller must drain the final pending state via
      ``finalize_layerwise_async_drain``.
    """
    p = profiler if profiler is not None else PhaseProfiler(enabled=False)
    assert layout.my_pool == "layerwise"
    device = next(component_model.parameters()).device

    sl = layout.my_batch_slice_lw()
    batch_local = batch[sl] if isinstance(batch, Tensor) else batch
    if isinstance(batch_local, Tensor):
        seq_len = batch_local.shape[1] if batch_local.ndim >= 2 else 1
    else:
        assert isinstance(batch_local, dict) and "input_ids" in batch_local
        seq_len = batch_local["input_ids"].shape[1]

    # ── Phase A: V/U-independent. Kicks GPU + NCCL work that runs in
    # background while the finalize block (below) blocks Python on NCCL waits.
    with strategy.context(component_model.target_model):
        with p.phase("lw/A1_post_async_recv_ci"):
            ci_recv, ci_recv_works = layout.async_recv_ci_from_ci_pool(
                {s: cfg.c_per_site[s] for s in layout.my_owned_sites},
                seq_len=seq_len,
                device=device,
            )
        with p.phase("lw/A2_target_fwd"), torch.no_grad(), autocast_bf16(cfg.bf16_autocast):
            target_local = component_model(batch_local).detach()

    # ── Finalize prev iter (async mode only). The blocking waits overlap with
    # the target_fwd kernels enqueued above on the default CUDA stream.
    if defer_vu_opt and prev_pending_all_reduce is not None:
        with p.phase("lw/B1_wait_prev_weight_send"):
            pending = getattr(component_model, "_pending_weight_sends", None)
            if pending is not None:
                for w in pending[0]:
                    w.wait()
                component_model._pending_weight_sends = None  # type: ignore[attr-defined]
        with p.phase("lw/B2_wait_and_unflatten_allreduce"):
            layout.wait_and_unflatten_all_reduce(prev_pending_all_reduce)
        # Cross-pool grad clip on V/U (LW pool's only param group). Matches
        # single-pool's clip on the global norm across all decomposition sites.
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
            v_owned = {s: component_model.components[s].V for s in layout.my_owned_sites}
            u_owned = {s: component_model.components[s].U for s in layout.my_owned_sites}
            weight_send_works, weight_send_buffers = layout.async_send_updated_vu_to_ppgd(
                v_owned, u_owned
            )
            component_model._pending_weight_sends = (  # type: ignore[attr-defined]
                weight_send_works,
                weight_send_buffers,
            )

    # ── Phase C: zero V/U .grad for new accumulation.
    for param in all_params:
        param.grad = None

    # ── Phase D: V/U-dependent work.
    with strategy.context(component_model.target_model):
        with p.phase("lw/D1_faith"):
            loss_faith, faith_sum_sq_t, faith_numel = _faithfulness_loss(
                component_model, device, cfg.numel_global
            )
            (cfg.coeff_faith * loss_faith).backward()

        with p.phase("lw/D2_wait_ci_recv"):
            for w in ci_recv_works:
                w.wait()
        ci_recv_leaves: dict[str, Tensor] = {
            s: ci_recv[s].detach().to(torch.float32).clone().requires_grad_(True)
            for s in layout.my_owned_sites
        }

        with p.phase("lw/D3_layerwise"), autocast_bf16(cfg.bf16_autocast):
            loss_stoch_value, stoch_total_value, stoch_n_owned = _layerwise_loss_streaming(
                component_model,
                batch_local,
                target_local,
                ci_recv_leaves,
                layout.my_owned_sites,
                strategy.recon_loss,
                cfg.coeff_stoch,
                n_sites_total=len(cfg.c_per_site),
            )

        with p.phase("lw/D4_send_g_ci"):
            g_ci_owned = {s: ci_recv_leaves[s].grad for s in layout.my_owned_sites}
            assert all(g is not None for g in g_ci_owned.values()), (
                "layerwise backward should have populated ci_recv_leaves[s].grad"
            )
            layout.send_g_ci_to_ci_pool(g_ci_owned)

        with p.phase("lw/D5_recv_g_vu_from_ppgd"):
            v_templates = {s: component_model.components[s].V for s in layout.my_owned_sites}
            u_templates = {s: component_model.components[s].U for s in layout.my_owned_sites}
            v_grads_pgd, u_grads_pgd = layout.recv_g_vu_from_ppgd(v_templates, u_templates)

        with p.phase("lw/D6_combine_vu_grads"):
            for s in layout.my_owned_sites:
                comp = component_model.components[s]
                assert comp.V.grad is not None and comp.U.grad is not None, (
                    "layerwise + faith should have populated V/U .grad"
                )
                comp.V.grad.add_(v_grads_pgd[s])
                comp.U.grad.add_(u_grads_pgd[s])

    # ── Phase E: tail. Async kickoff (deferred) or sync all_reduce + opt + send.
    # See ``three_pool.reductions`` for how the raw (num, den) is combined into
    # global faith and stoch on rank 0.
    metrics = {
        "loss/faith": loss_faith.item(),
        "loss/stoch": loss_stoch_value,
        "_raw/faith_num": faith_sum_sq_t.item(),
        "_raw/faith_den": float(faith_numel),
        "_raw/stoch_num": stoch_total_value,
        "_raw/stoch_den": float(stoch_n_owned),
    }

    if defer_vu_opt:
        with p.phase("lw/E_kickoff_async_allreduce"):
            new_pending = layout.async_all_reduce_grads_in_block_kickoff(all_params)
        return metrics, new_pending

    # Sync tail: matches the pre-deferral behavior exactly.
    with p.phase("lw/E1_wait_prev_weight_send"):
        pending = getattr(component_model, "_pending_weight_sends", None)
        if pending is not None:
            for w in pending[0]:
                w.wait()
            component_model._pending_weight_sends = None  # type: ignore[attr-defined]
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
        v_owned = {s: component_model.components[s].V for s in layout.my_owned_sites}
        u_owned = {s: component_model.components[s].U for s in layout.my_owned_sites}
        weight_send_works, weight_send_buffers = layout.async_send_updated_vu_to_ppgd(
            v_owned, u_owned
        )
        component_model._pending_weight_sends = (  # type: ignore[attr-defined]
            weight_send_works,
            weight_send_buffers,
        )
    return metrics, None


def finalize_layerwise_async_drain(
    layout: ThreePoolLayout,
    component_model: ComponentModel,
    optimizer: torch.optim.Optimizer,
    all_params: list[nn.Parameter],
    pending_all_reduce: list[tuple[list[Tensor], Tensor, "dist.Work"]],
    grad_clip_norm: float | None,
) -> None:
    """End-of-training drain in async mode: finish the final iter's deferred
    opt step. Skips the V/U async send since training is over.
    """
    assert layout.my_pool == "layerwise"
    pending = getattr(component_model, "_pending_weight_sends", None)
    if pending is not None:
        for w in pending[0]:
            w.wait()
        component_model._pending_weight_sends = None  # type: ignore[attr-defined]
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
    """Single-pool-equivalent faithfulness warmup on the Layerwise pool only.

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
