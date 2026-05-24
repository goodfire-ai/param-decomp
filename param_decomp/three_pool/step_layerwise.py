"""Layerwise pool training step: target_fwd → recv CI → layerwise stoch + faith → send g_CI → recv g_VU → opt step.

Recast of ``two_pool.pool_a.step_pool_a`` with the CI-fn-side bits moved out
to the CI pool. Per-step flow (numbered to match ``DESIGN.md`` `lw/N_phase`):

  0. Wait for pending async weight-ship from the previous step before any
     optimizer step touches V/U.
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
  9. In-block all-reduce on V/U grads (DDP within block group).
  10. AdamW step on V/U.
  11. Async ship updated V/U → PPGD pool. The work flushes during the next
      step's CI-fn-fwd window; we wait on it at step 0 of the next step.

Step 1 + 3 give us the headline overlap: the CI recv runs on the NIC while
faithfulness runs on the GPU, hiding the CI recv latency behind a useful
local compute.
"""

# pyright: reportArgumentType=false, reportOperatorIssue=false, reportAttributeAccessIssue=false

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from param_decomp.component_model import ComponentModel
from param_decomp.masks import make_mask_infos
from param_decomp.three_pool.layout import ThreePoolLayout
from param_decomp.three_pool.profiler import PhaseProfiler
from param_decomp.three_pool.runtime import _ThreePoolRuntime
from param_decomp.two_pool.loss_strategy import LayerwiseLossStrategy
from param_decomp.two_pool.runtime import autocast_bf16


def _faithfulness_loss(component_model: ComponentModel, device: torch.device) -> Tensor:
    """‖W_target − VU.T‖²_F / numel, summed across this rank's owned sites.

    Same formula as two_pool's ``_faithfulness_loss``. Pool sharding means each
    LW rank only iterates over its owned sites' weight_deltas, but the
    coefficient is left as-is (same per-site magnitude as in single-pool).
    """
    weight_deltas = component_model.calc_weight_deltas()
    sum_sq = torch.zeros((), device=device)
    numel = 0
    for d in weight_deltas.values():
        sum_sq = sum_sq + (d**2).sum()
        numel += d.numel()
    return sum_sq / numel


def _layerwise_loss_streaming(
    component_model: ComponentModel,
    batch_local: Any,
    target_local: Tensor,
    ci_recv_leaves: dict[str, Tensor],
    owned_sites: tuple[str, ...],
    recon_loss: Any,
    coeff_stoch: float,
    n_sites_total: int,
) -> float:
    """Per-site streaming layerwise loss. Identical contract to
    ``two_pool.pool_a._layerwise_loss_streaming`` — backward per iter so peak
    memory stays bounded to ~1×iter.

    Per-site contribution scales as
    ``coeff_stoch * sum_kl_s / (n_positions * n_sites_total)`` so the same
    YAML coefficient transfers from 2-pool / single-pool trainers.
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
    return total_value / n_owned


def step_layerwise(
    layout: ThreePoolLayout,
    component_model: ComponentModel,
    optimizer: torch.optim.Optimizer,
    all_params: list[nn.Parameter],
    batch: Any,
    cfg: _ThreePoolRuntime,
    strategy: LayerwiseLossStrategy,
    profiler: PhaseProfiler | None = None,
) -> dict[str, float]:
    """One training step on a Layerwise-pool rank."""
    p = profiler if profiler is not None else PhaseProfiler(enabled=False)
    assert layout.my_pool == "layerwise"
    device = next(component_model.parameters()).device

    # 0. Wait for pending async weight-ship from previous step.
    with p.phase("lw/0_wait_prev_weight_send"):
        pending = getattr(component_model, "_pending_weight_sends", None)
        if pending is not None:
            for w in pending[0]:
                w.wait()
            component_model._pending_weight_sends = None  # type: ignore[attr-defined]

    # Per-rank batch slice
    sl = layout.my_batch_slice_lw()
    batch_local = batch[sl] if isinstance(batch, Tensor) else batch

    # Infer seq_len for the CI recv buffer alloc.
    # Assumes batch is [B, S, ...] tensor or dict with input_ids of that shape.
    if isinstance(batch_local, Tensor):
        seq_len = batch_local.shape[1] if batch_local.ndim >= 2 else 1
    else:
        # Fall back: take any tensor in the dict.
        assert isinstance(batch_local, dict) and "input_ids" in batch_local
        seq_len = batch_local["input_ids"].shape[1]

    with strategy.context(component_model.target_model):
        # 1. Post async CI recv. Doesn't block target_fwd / faith below.
        with p.phase("lw/1_post_async_recv_ci"):
            ci_recv, ci_recv_works = layout.async_recv_ci_from_ci_pool(
                {s: cfg.c_per_site[s] for s in layout.my_owned_sites},
                seq_len=seq_len,
                device=device,
            )

        # 2. target forward. bf16 autocast (frozen model, fast SDPA kernels).
        with p.phase("lw/2_target_fwd"), torch.no_grad(), autocast_bf16(cfg.bf16_autocast):
            target_local = component_model(batch_local).detach()

    optimizer.zero_grad(set_to_none=True)

    # 3. Faithfulness loss + backward. Outside autocast — fp32-sensitive.
    # Concurrent (CPU-wise) with the CI recv on the NIC.
    with p.phase("lw/3_faith"):
        loss_faith = _faithfulness_loss(component_model, device)
        (cfg.coeff_faith * loss_faith).backward()

    # 4. Wait for CI recv; upcast to fp32 + re-leaf so layerwise backward
    # populates leaf.grad (which we'll ship back to CI pool).
    with p.phase("lw/4_wait_ci_recv"):
        for w in ci_recv_works:
            w.wait()
    ci_recv_leaves: dict[str, Tensor] = {
        s: ci_recv[s].detach().to(torch.float32).clone().requires_grad_(True)
        for s in layout.my_owned_sites
    }

    # 5. Streaming layerwise — per-site backward, iter-local graph freed.
    with p.phase("lw/5_layerwise"), autocast_bf16(cfg.bf16_autocast):
        loss_stoch_value = _layerwise_loss_streaming(
            component_model,
            batch_local,
            target_local,
            ci_recv_leaves,
            layout.my_owned_sites,
            strategy.recon_loss,
            cfg.coeff_stoch,
            n_sites_total=len(cfg.c_per_site),
        )

    # 6. Ship g_CI_LW back to CI pool (sync — grads are ready now).
    with p.phase("lw/6_send_g_ci"):
        g_ci_owned = {s: ci_recv_leaves[s].grad for s in layout.my_owned_sites}
        assert all(g is not None for g in g_ci_owned.values()), (
            "layerwise backward should have populated ci_recv_leaves[s].grad"
        )
        layout.send_g_ci_to_ci_pool(g_ci_owned)

    # 7. Recv g_VU from PPGD pool (leader recv + in-block bcast).
    with p.phase("lw/7_recv_g_vu_from_ppgd"):
        v_templates = {s: component_model.components[s].V for s in layout.my_owned_sites}
        u_templates = {s: component_model.components[s].U for s in layout.my_owned_sites}
        v_grads_pgd, u_grads_pgd = layout.recv_g_vu_from_ppgd(v_templates, u_templates)

    # 8. Combine PPGD's V/U grads into the existing .grad accumulator.
    # Layerwise stoch + faith already populated V/U .grad; just add PPGD's.
    with p.phase("lw/8_combine_vu_grads"):
        for s in layout.my_owned_sites:
            comp = component_model.components[s]
            assert comp.V.grad is not None and comp.U.grad is not None, (
                "layerwise + faith should have populated V/U .grad"
            )
            comp.V.grad.add_(v_grads_pgd[s])
            comp.U.grad.add_(u_grads_pgd[s])

    # 9. In-block all-reduce on V/U + faith grads (DDP within block group).
    with p.phase("lw/9_in_block_allreduce"):
        layout.all_reduce_grads_in_block(all_params)

    # 10. AdamW step on V/U.
    with p.phase("lw/10_opt_step"):
        optimizer.step()

    # 11. Async ship updated V/U back to PPGD pool. Work flushes during next
    # step's CI-fn-fwd window; we wait on it at step 0 of the next step.
    with p.phase("lw/11_async_send_vu"):
        v_owned = {s: component_model.components[s].V for s in layout.my_owned_sites}
        u_owned = {s: component_model.components[s].U for s in layout.my_owned_sites}
        weight_send_works, weight_send_buffers = layout.async_send_updated_vu_to_ppgd(
            v_owned, u_owned
        )
    component_model._pending_weight_sends = (  # type: ignore[attr-defined]
        weight_send_works,
        weight_send_buffers,
    )

    return {
        "loss/faith": loss_faith.item(),
        "loss/stoch": loss_stoch_value,
    }


def run_faithfulness_warmup_layerwise(
    *,
    component_model: ComponentModel,
    component_params: list[nn.Parameter],
    n_steps: int,
    lr: float,
    weight_decay: float,
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
        loss = _faithfulness_loss(component_model, device)
        loss.backward()
        warmup_opt.step()
    del warmup_opt
    torch.cuda.empty_cache()
