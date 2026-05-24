"""PPGD pool training step: target_fwd → recv CI → PPGD warmup + recon → bwd → send grads → recv V/U.

Recast of ``two_pool.pool_b.step_pool_b`` for the 3-pool topology:

  * CI values come from the CI pool (instead of LW leaders).
  * g_CI grads go back to the CI pool (instead of LW leaders).
  * g_VU grads still go to LW block leaders.
  * Updated V/U still come from LW block leaders.
  * **No final outer source-step**: the warmup inner loop owns source updates;
    the final recon backward only seeds V/U + CI grads. Matches the design
    decision locked in during the planning discussion.

Per-step flow (numbered to match ``DESIGN.md`` `ppgd/N_phase`):

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
  10. Recv updated V/U from LW leaders (blocking — V/U must be fresh before
      the next step's PPGD warmup).
"""

# pyright: reportArgumentType=false

from typing import Any

import torch
from torch import Tensor

from param_decomp.component_model import ComponentModel
from param_decomp.metrics.persistent_pgd_state import PersistentPGDState
from param_decomp.three_pool.layout import ThreePoolLayout
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
    profiler: PhaseProfiler | None = None,
) -> dict[str, float]:
    """One training step on a PPGD-pool rank."""
    p = profiler if profiler is not None else PhaseProfiler(enabled=False)
    assert layout.my_pool == "ppgd"
    device = next(component_model.parameters()).device

    # Top-of-step PPGD LR schedule bump (mirrors the single-pool metric).
    ppgd_state.update_lr(step=step, total_steps=n_steps)

    weight_deltas = component_model.calc_weight_deltas()

    sl = layout.my_batch_slice_ppgd()
    batch_local = batch[sl] if isinstance(batch, Tensor) else batch

    # Infer seq_len for the CI recv buffer alloc.
    if isinstance(batch_local, Tensor):
        seq_len = batch_local.shape[1] if batch_local.ndim >= 2 else 1
    else:
        assert isinstance(batch_local, dict) and "input_ids" in batch_local
        seq_len = batch_local["input_ids"].shape[1]

    with strategy.context(component_model.target_model):
        # 1. Post async CI recv. Runs concurrently with target_fwd.
        with p.phase("ppgd/1_post_async_recv_ci"):
            ci_recv, ci_recv_works = layout.async_recv_ci_from_ci_pool_ppgd(
                cfg.c_per_site,
                seq_len=seq_len,
                device=device,
            )

        # 2. target forward (frozen, no grad) — runs in parallel with CI recv.
        with p.phase("ppgd/2_target_fwd"), torch.no_grad(), autocast_bf16(cfg.bf16_autocast):
            target_out = component_model(batch_local).detach()

        # 3. Wait for CI recv. Re-leaf as fp32 so the final backward populates
        # ci_scratch[s].grad → shipped back to CI pool at step 9.
        with p.phase("ppgd/3_wait_ci_recv"):
            for w in ci_recv_works:
                w.wait()
        ci_scratch: dict[str, Tensor] = {
            s: v.detach().to(torch.float32).clone().requires_grad_(True) for s, v in ci_recv.items()
        }

        # 4. PPGD warmup. Refines persistent adversarial sources in-place — the
        # inner loop owns source updates (no outer source step).
        with p.phase("ppgd/4_warmup"), autocast_bf16(cfg.bf16_autocast):
            ppgd_state.warmup(
                model=component_model,
                batch=batch_local,
                target_out=target_out,
                ci=ci_scratch,
                weight_deltas=weight_deltas,
            )

        # 5. Final PPGD recon loss with refined sources.
        with p.phase("ppgd/5_recon"), autocast_bf16(cfg.bf16_autocast):
            sum_loss, n_examples = ppgd_state.compute_recon_sum_and_n(
                model=component_model,
                batch=batch_local,
                target_out=target_out,
                ci=ci_scratch,
                weight_deltas=weight_deltas,
            )
            loss_ppgd = sum_loss / n_examples

    # Scale by 1/N_ppgd so a SUM-reduce of V/U grads across PPGD pool equals
    # the full-batch grad. CI grads aren't scaled here — the CI pool's stitch
    # of K_ppgd_per_ci slices into a single [B_local_ci, ...] tensor implicitly
    # gives the right per-example contribution.
    total_ppgd = cfg.coeff_ppgd * loss_ppgd / layout.world.n_ppgd

    # 6. Extract V/U + ci_scratch grads via autograd.grad (no .grad pollution).
    with p.phase("ppgd/6_backward"):
        all_sites = list(layout.world.all_sites)
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

    # 7. In-pool sum-reduce on V/U grads. Coalesced bucketing in the layout helper.
    with p.phase("ppgd/7_in_pool_sum_reduce"):
        layout.sum_reduce_ppgd_grads([*v_grads.values(), *u_grads.values()])

    # 8. Leader sends g_VU to LW block leaders (everyone else no-ops).
    with p.phase("ppgd/8_send_g_vu_to_lw"):
        layout.send_g_vu_to_layerwise(v_grads, u_grads)

    # 9. All PPGD ranks send their slice of g_CI to the owning CI rank.
    with p.phase("ppgd/9_send_g_ci_to_ci_pool"):
        layout.send_g_ci_to_ci_pool_ppgd(ci_grads)

    # 10. Blocking recv of updated V/U from LW. Must complete before next step's
    # PPGD warmup. (Could be deferred to start-of-next-step for overlap, but
    # the LW leaders' async send was kicked off at end of LW's step, so this
    # recv typically returns quickly.)
    with p.phase("ppgd/10_recv_updated_vu"):
        v_templates = {s: component_model.components[s].V for s in all_sites}
        u_templates = {s: component_model.components[s].U for s in all_sites}
        v_new, u_new = layout.recv_updated_vu_from_layerwise(v_templates, u_templates)
        with torch.no_grad():
            for s in all_sites:
                component_model.components[s].V.copy_(v_new[s])
                component_model.components[s].U.copy_(u_new[s])

    return {"loss/ppgd": loss_ppgd.item()}
