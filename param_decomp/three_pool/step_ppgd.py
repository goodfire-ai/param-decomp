"""PPGD pool training step — split into ``_main`` and ``_recv_vu_tail``.

Recast of ``two_pool.pool_b.step_pool_b`` for the 3-pool topology, split into
two functions so the V/U recv from LW can either run at end of step T
(sync mode) OR be deferred to start of step T+1 (when
``ThreePoolConfig.defer_vu_opt=True``, mirroring LW's symmetric deferral —
otherwise the LW's deferred send would deadlock with PPGD's blocking recv).

Other 3-pool differences from 2-pool:

  * CI values come from the CI pool (instead of LW leaders).
  * g_CI grads go back to the CI pool (instead of LW leaders).
  * **No final outer source-step**: the warmup inner loop owns source updates;
    the final recon backward only seeds V/U + CI grads.

``step_ppgd_main`` phases (numbered to match ``DESIGN.md`` `ppgd/N`):

  1. Post async irecv for CI_T from the owning CI rank (concurrent with
     target_fwd).
  2. target_fwd(batch_T) → L_T (per-PPGD-rank slice).
  3. Wait for CI recv; re-leaf as fp32 for CI grad extraction.
  4. PPGD warmup — refines persistent adversarial sources in-place.
  5. Final PPGD recon loss with refined sources.
  6. Backward: extract g_VU + g_CI only (no source backward).
  7. In-pool sum-reduce on g_VU (so each PPGD rank holds the full-batch grad).
  8. Send g_VU to LW block leaders (PPGD-leader-only — values are identical
     after the sum-reduce).
  9. Send g_CI to CI pool (every rank, on its batch slice).

``step_ppgd_recv_vu_tail``:

  10. Blocking recv of updated V/U from LW leaders + copy into components.
      In sync mode runs at end of step T; in deferred mode runs at top of
      step T+1 (where it overlaps with CI fn fwd on the CI pool).
"""

# pyright: reportArgumentType=false

from typing import Any

import torch
import torch.distributed as dist  # noqa: F401  (used in type hints)
from torch import Tensor

from param_decomp.component_model import ComponentModel
from param_decomp.metrics.persistent_pgd_state import PersistentPGDState
from param_decomp.three_pool.layout import LayerwiseBlockGroup, ThreePoolLayout
from param_decomp.three_pool.profiler import PhaseProfiler
from param_decomp.three_pool.runtime import _ThreePoolRuntime
from param_decomp.two_pool.loss_strategy import LayerwiseLossStrategy
from param_decomp.two_pool.runtime import autocast_bf16


def step_ppgd(
    layout: ThreePoolLayout,
    component_model: ComponentModel,
    ppgd_state: PersistentPGDState,
    batch: Any,
    cfg: _ThreePoolRuntime,
    strategy: LayerwiseLossStrategy,
    step: int,
    n_steps: int,
    *,
    defer_vu_opt: bool,
    prev_pending_recv_vu: list[tuple["LayerwiseBlockGroup", Tensor, "dist.Work"]] | None,
    profiler: PhaseProfiler | None = None,
) -> tuple[dict[str, float], list[tuple["LayerwiseBlockGroup", Tensor, "dist.Work"]] | None]:
    """One PPGD step. Branches on ``defer_vu_opt`` for sync vs async pipeline.

    Sync (``defer_vu_opt=False``):
      Phase A (V/U-independent: recv_ci + target_fwd) → Phase D (V/U-dep:
      warmup + recon + backward + sends) → sync recv_vu at end. Returns
      ``(metrics, None)``.

    Async (``defer_vu_opt=True``):
      Phase A → finalize prev iter's V/U recv (concurrent w/ target_fwd
      kernels) → Phase D → kickoff async V/U recv → return state.

      Symmetric to ``step_layerwise``'s deferral. Required when LW defers:
      otherwise PPGD's blocking sync recv at end of step T would deadlock
      against LW's deferred async send (which doesn't fire until top of T+1).
    """
    p = profiler if profiler is not None else PhaseProfiler(enabled=False)
    assert layout.my_pool == "ppgd"
    device = next(component_model.parameters()).device
    all_sites = list(layout.world.all_sites)

    ppgd_state.update_lr(step=step, total_steps=n_steps)

    sl = layout.my_batch_slice_ppgd()
    batch_local = batch[sl] if isinstance(batch, Tensor) else batch
    if isinstance(batch_local, Tensor):
        seq_len = batch_local.shape[1] if batch_local.ndim >= 2 else 1
    else:
        assert isinstance(batch_local, dict) and "input_ids" in batch_local
        seq_len = batch_local["input_ids"].shape[1]

    v_templates = {s: component_model.components[s].V for s in all_sites}
    u_templates = {s: component_model.components[s].U for s in all_sites}

    with strategy.context(component_model.target_model):
        # ── Phase A: V/U-independent. Kicks NCCL recv_ci + GPU target_fwd.
        with p.phase("pgd/A1_post_async_recv_ci"):
            ci_recv, ci_recv_works = layout.async_recv_ci_from_ci_pool_ppgd(
                cfg.c_per_site,
                seq_len=seq_len,
                device=device,
            )
        with p.phase("pgd/A2_target_fwd"), torch.no_grad(), autocast_bf16(cfg.bf16_autocast):
            target_out = component_model(batch_local).detach()

        # ── Finalize prev iter (async mode only). The wait for prev iter's V/U
        # broadcast overlaps with the target_fwd kernels above.
        if defer_vu_opt and prev_pending_recv_vu is not None:
            with p.phase("pgd/B_wait_and_copy_prev_vu"):
                v_new, u_new = layout.wait_and_unpack_updated_vu(
                    prev_pending_recv_vu, v_templates, u_templates
                )
                with torch.no_grad():
                    for s in all_sites:
                        component_model.components[s].V.copy_(v_new[s])
                        component_model.components[s].U.copy_(u_new[s])

        # ── Phase D: V/U-dependent. weight_deltas + warmup + recon use V/U,
        # which is fresh now (either from sync recv last iter or just-copied).
        with p.phase("pgd/D1_calc_weight_deltas"):
            weight_deltas = component_model.calc_weight_deltas()

        with p.phase("pgd/D2_wait_ci_recv"):
            for w in ci_recv_works:
                w.wait()
        ci_scratch: dict[str, Tensor] = {
            s: v.detach().to(torch.float32).clone().requires_grad_(True) for s, v in ci_recv.items()
        }

        with p.phase("pgd/D3_warmup"), autocast_bf16(cfg.bf16_autocast):
            ppgd_state.warmup(
                model=component_model,
                batch=batch_local,
                target_out=target_out,
                ci=ci_scratch,
                weight_deltas=weight_deltas,
            )

        with p.phase("pgd/D4_recon"), autocast_bf16(cfg.bf16_autocast):
            sum_loss, n_examples = ppgd_state.compute_recon_sum_and_n(
                model=component_model,
                batch=batch_local,
                target_out=target_out,
                ci=ci_scratch,
                weight_deltas=weight_deltas,
            )
            loss_ppgd = sum_loss / n_examples

    total_ppgd = cfg.coeff_ppgd * loss_ppgd / layout.world.n_ppgd

    with p.phase("pgd/D5_backward"):
        params: list[Tensor] = []
        for s in all_sites:
            params.append(component_model.components[s].V)
            params.append(component_model.components[s].U)
        ci_list = [ci_scratch[s] for s in all_sites]
        grads = torch.autograd.grad(total_ppgd, params + ci_list, retain_graph=False)

    n_sites = len(all_sites)
    v_grads = {s: grads[2 * i] for i, s in enumerate(all_sites)}
    u_grads = {s: grads[2 * i + 1] for i, s in enumerate(all_sites)}
    ci_grads = {s: grads[2 * n_sites + i] for i, s in enumerate(all_sites)}

    with p.phase("pgd/D6_in_pool_sum_reduce"):
        layout.sum_reduce_ppgd_grads([*v_grads.values(), *u_grads.values()])

    with p.phase("pgd/D7_send_g_vu_to_lw"):
        layout.send_g_vu_to_layerwise(v_grads, u_grads)

    with p.phase("pgd/D8_send_g_ci_to_ci_pool"):
        layout.send_g_ci_to_ci_pool_ppgd(ci_grads)

    # ── Phase E: recv updated V/U. Async kickoff (deferred) or sync recv.
    # Raw num/den are the pool-B-style additive ingredients: SUM across PPGD
    # pool gives global ``sum_loss / global_n_examples`` = global mean per
    # position. (n_examples is uniform across PPGD ranks so AVG would
    # numerically coincide, but using raw keeps the contract identical to the
    # LW pool's faith/stoch path.)
    metrics = {
        "loss/ppgd": loss_ppgd.item(),
        "_raw/ppgd_num": sum_loss.item(),
        "_raw/ppgd_den": float(n_examples),
    }

    if defer_vu_opt:
        with p.phase("pgd/E_kickoff_async_recv_vu"):
            new_pending = layout.async_recv_updated_vu_from_layerwise_kickoff(
                v_templates, u_templates
            )
        return metrics, new_pending

    # Sync recv at end of step.
    with p.phase("pgd/E_sync_recv_vu"):
        v_new, u_new = layout.recv_updated_vu_from_layerwise(v_templates, u_templates)
        with torch.no_grad():
            for s in all_sites:
                component_model.components[s].V.copy_(v_new[s])
                component_model.components[s].U.copy_(u_new[s])
    return metrics, None


def finalize_ppgd_async_drain(
    layout: ThreePoolLayout,
    component_model: ComponentModel,
    pending_recv_vu: list[tuple["LayerwiseBlockGroup", Tensor, "dist.Work"]],
) -> None:
    """End-of-training drain in async mode: complete the final iter's V/U recv
    so the saved checkpoint (gathered from LW pool's V/U separately) is
    consistent with what PPGD pool used last."""
    assert layout.my_pool == "ppgd"
    all_sites = list(layout.world.all_sites)
    v_templates = {s: component_model.components[s].V for s in all_sites}
    u_templates = {s: component_model.components[s].U for s in all_sites}
    v_new, u_new = layout.wait_and_unpack_updated_vu(pending_recv_vu, v_templates, u_templates)
    with torch.no_grad():
        for s in all_sites:
            component_model.components[s].V.copy_(v_new[s])
            component_model.components[s].U.copy_(u_new[s])
