"""PPGD pool training step — split into ``step_ppgd`` and ``finalize_ppgd_async_drain``.

Recast of ``two_pool.pool_b.step_pool_b`` for 3-pool. Same PPGD math; the
differences are (a) CI comes from the CI pool not LW leaders, (b) g_CI goes
back to CI pool not LW, and (c) no final outer source-step (the warmup inner
loop owns source updates).

Phases (numbered to match ``DESIGN.md`` ``ppgd/N``):

  A1. Post async irecv for CI_T from the owning CI rank (concurrent with A2).
  A2. target_fwd(batch_T) on the rank's PPGD slice → L_T.
  B.  Async-mode only: wait + copy prev iter's V/U recv (overlaps with A2's
      kernels on the default CUDA stream while NCCL waits run on theirs).
  D1. Compute weight_deltas (V/U-dependent — fresh now).
  D2. Wait CI recv; re-leaf as fp32 for downstream CI grad extraction.
  D3. PPGD warmup refines persistent adversarial sources in place.
  D4. Recon loss with refined sources.
  D5. Backward: ``torch.autograd.grad`` for g_VU + g_CI (no source backward).
  D6. Sum-reduce g_VU within PPGD pool → each rank holds the full-batch grad.
  D7. Send g_VU to LW block leaders (PPGD-leader-only).
  D8. Send g_CI to CI pool (every rank, on its batch slice).
  E.  Recv updated V/U from LW:
        * sync mode → blocking recv at end of step (returns
          ``(metrics, None)``);
        * async mode → kickoff async recv, return handles for next step's
          phase B.

``step_ppgd`` reads as ~25 lines of orchestration; the named helpers below
correspond one-to-one with the phases above. Async-mode + sync-mode differ
only at the tail.
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

PendingRecvVU = list[tuple["LayerwiseBlockGroup", Tensor, "dist.Work"]]


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
    prev_pending_recv_vu: PendingRecvVU | None,
    profiler: PhaseProfiler | None = None,
) -> tuple[dict[str, float], PendingRecvVU | None]:
    """One PPGD step. Branches on ``defer_vu_opt`` for sync vs async pipeline.

    Async mode is required when LW defers — otherwise PPGD's blocking sync recv
    at end of step T would deadlock against LW's deferred async send (which
    doesn't fire until top of T+1).
    """
    p = profiler if profiler is not None else PhaseProfiler(enabled=False)
    assert layout.my_pool == "ppgd"
    device = next(component_model.parameters()).device
    all_sites = list(layout.world.all_sites)

    ppgd_state.update_lr(step=step, total_steps=n_steps)
    batch_local, seq_len = _slice_batch_for_ppgd(batch, layout)
    v_templates, u_templates = _vu_templates(component_model, all_sites)

    with strategy.context(component_model.target_model):
        ci_recv, ci_recv_works = _async_recv_ci_from_ci_pool(layout, cfg, seq_len, device, p)
        target_out = _target_forward_no_grad(component_model, batch_local, cfg, p)

        # Async-mode bookkeeping: this iter's V/U arrived in the prev iter's
        # async kickoff. Wait + copy here, overlapping the NCCL wait with the
        # A2 target_fwd kernels enqueued above.
        if defer_vu_opt and prev_pending_recv_vu is not None:
            _wait_and_copy_prev_vu_into_model(
                layout, component_model, prev_pending_recv_vu, v_templates, u_templates, p
            )

        weight_deltas = _compute_weight_deltas(component_model, p)
        _wait_ci_recv(ci_recv_works, p)
        ci_scratch = _releaf_ci_fp32_for_grads(ci_recv)
        _assert_ci_scratch_shapes(ci_scratch, layout, seq_len, cfg)

        _ppgd_inner_warmup(
            ppgd_state, component_model, batch_local, target_out, ci_scratch, weight_deltas, cfg, p
        )
        sum_loss, n_examples = _ppgd_recon_forward(
            ppgd_state, component_model, batch_local, target_out, ci_scratch, weight_deltas, cfg, p
        )

    # Scale by 1/n_ppgd so D6's SUM-reduce produces the full-batch gradient.
    total_ppgd = cfg.coeff_ppgd * (sum_loss / n_examples) / layout.world.n_ppgd

    v_grads, u_grads, ci_grads = _autograd_grad_for_vu_and_ci(
        total_ppgd, component_model, ci_scratch, all_sites, p
    )
    _sum_reduce_vu_grads_within_ppgd_pool(layout, v_grads, u_grads, p)
    _send_g_vu_to_layerwise(layout, v_grads, u_grads, p)
    _send_g_ci_to_ci_pool(layout, ci_grads, p)

    metrics = _step_metrics(sum_loss, n_examples)

    # Recv updated V/U from LW — branch on defer_vu_opt.
    if defer_vu_opt:
        new_pending = _async_kickoff_recv_vu_from_lw(layout, v_templates, u_templates, p)
        return metrics, new_pending
    _sync_recv_vu_from_lw_and_copy(layout, component_model, v_templates, u_templates, all_sites, p)
    return metrics, None


def finalize_ppgd_async_drain(
    layout: ThreePoolLayout,
    component_model: ComponentModel,
    pending_recv_vu: PendingRecvVU,
) -> None:
    """End-of-training drain in async mode: complete the final iter's V/U recv
    so the saved checkpoint (gathered from LW pool's V/U separately) is
    consistent with what PPGD pool used last.
    """
    assert layout.my_pool == "ppgd"
    all_sites = list(layout.world.all_sites)
    v_templates, u_templates = _vu_templates(component_model, all_sites)
    v_new, u_new = layout.wait_and_unpack_updated_vu(pending_recv_vu, v_templates, u_templates)
    _copy_vu_into_model_in_place(component_model, v_new, u_new, all_sites)


def _slice_batch_for_ppgd(batch: Any, layout: ThreePoolLayout) -> tuple[Any, int]:
    """Pull this PPGD rank's batch slice + extract its seq_len.

    Returns ``(batch_local, seq_len)``. The seq_len is used by the CI recv to
    pre-allocate buffers of the right shape.
    """
    sl = layout.my_batch_slice_ppgd()
    batch_local = batch[sl] if isinstance(batch, Tensor) else batch
    if isinstance(batch_local, Tensor):
        seq_len = batch_local.shape[1] if batch_local.ndim >= 2 else 1
    else:
        assert isinstance(batch_local, dict) and "input_ids" in batch_local
        seq_len = batch_local["input_ids"].shape[1]
    return batch_local, seq_len


def _vu_templates(
    component_model: ComponentModel, all_sites: list[str]
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """V/U tensors per site — used as recv buffers in async kickoff/drain."""
    v_templates: dict[str, Tensor] = {s: component_model.components[s].V for s in all_sites}
    u_templates: dict[str, Tensor] = {s: component_model.components[s].U for s in all_sites}
    return v_templates, u_templates


def _async_recv_ci_from_ci_pool(
    layout: ThreePoolLayout,
    cfg: _ThreePoolRuntime,
    seq_len: int,
    device: torch.device,
    p: PhaseProfiler,
) -> tuple[dict[str, Tensor], list[Any]]:
    """Phase ppgd/A1. Post irecvs for CI values from the CI pool.

    Works run on NIC concurrently with the target_fwd kernels enqueued in A2
    — pool doesn't need CI for target_fwd, so we save the recv latency by
    overlapping.
    """
    with p.phase("pgd/A1_post_async_recv_ci"):
        return layout.async_recv_ci_from_ci_pool_ppgd(
            cfg.c_per_site,
            seq_len=seq_len,
            device=device,
        )


def _target_forward_no_grad(
    component_model: ComponentModel,
    batch_local: Any,
    cfg: _ThreePoolRuntime,
    p: PhaseProfiler,
) -> Tensor:
    """Phase ppgd/A2. Frozen target forward on the PPGD rank's batch slice.

    No grad — we only need the output as the recon target. Autocast under
    bf16 so SDPA picks the flash kernel on H200.
    """
    with p.phase("pgd/A2_target_fwd"), torch.no_grad(), autocast_bf16(cfg.bf16_autocast):
        return component_model(batch_local).detach()


def _wait_and_copy_prev_vu_into_model(
    layout: ThreePoolLayout,
    component_model: ComponentModel,
    prev_pending_recv_vu: PendingRecvVU,
    v_templates: dict[str, Tensor],
    u_templates: dict[str, Tensor],
    p: PhaseProfiler,
) -> None:
    """Phase ppgd/B (async mode only). Finalize the prev iter's V/U recv.

    Waits + copies into ``components[s].V/U`` in place. The blocking wait
    overlaps with the A2 target_fwd kernels enqueued just above — that's the
    headline win of async mode.
    """
    with p.phase("pgd/B_wait_and_copy_prev_vu"):
        v_new, u_new = layout.wait_and_unpack_updated_vu(
            prev_pending_recv_vu, v_templates, u_templates
        )
        _copy_vu_into_model_in_place(component_model, v_new, u_new, list(layout.world.all_sites))


def _compute_weight_deltas(component_model: ComponentModel, p: PhaseProfiler) -> dict[str, Tensor]:
    """Phase ppgd/D1. Compute ``target_weight - V @ U.T`` per site.

    V/U is fresh now (either from sync recv last iter or just-copied in B).
    Used by the PPGD warmup + recon as the optional delta-component path.
    """
    with p.phase("pgd/D1_calc_weight_deltas"):
        return component_model.calc_weight_deltas()


def _wait_ci_recv(ci_recv_works: list[Any], p: PhaseProfiler) -> None:
    """Phase ppgd/D2 (first half). Block on the irecvs from phase A1."""
    with p.phase("pgd/D2_wait_ci_recv"):
        for w in ci_recv_works:
            w.wait()


def _releaf_ci_fp32_for_grads(ci_recv: dict[str, Tensor]) -> dict[str, Tensor]:
    """Phase ppgd/D2 (second half). Re-leaf CI as fp32 ``requires_grad=True``.

    CI is shipped in bf16; we upcast + re-leaf so the leaf has fp32 ``.grad``
    that CI pool can merge into its fp32 CI-fn grads.
    """
    return {
        s: v.detach().to(torch.float32).clone().requires_grad_(True) for s, v in ci_recv.items()
    }


def _assert_ci_scratch_shapes(
    ci_scratch: dict[str, Tensor],
    layout: ThreePoolLayout,
    seq_len: int,
    cfg: _ThreePoolRuntime,
) -> None:
    """Sanity-check the CI scratch tensors match what the CI pool said it'd send.

    Catches a wrong ``c_per_site`` config or a per-rank batch mismatch fast.
    """
    batch_local_ppgd = layout.world.batch_local_ppgd
    for s, c in cfg.c_per_site.items():
        t = ci_scratch[s]
        assert t.shape == (batch_local_ppgd, seq_len, c), (
            f"ci_scratch[{s!r}] shape {tuple(t.shape)} != "
            f"expected ({batch_local_ppgd}, {seq_len}, {c})"
        )


def _ppgd_inner_warmup(
    ppgd_state: PersistentPGDState,
    component_model: ComponentModel,
    batch_local: Any,
    target_out: Tensor,
    ci_scratch: dict[str, Tensor],
    weight_deltas: dict[str, Tensor],
    cfg: _ThreePoolRuntime,
    p: PhaseProfiler,
) -> None:
    """Phase ppgd/D3. Refine the persistent adversarial sources in place."""
    with p.phase("pgd/D3_warmup"), autocast_bf16(cfg.bf16_autocast):
        ppgd_state.warmup(
            model=component_model,
            batch=batch_local,
            target_out=target_out,
            ci=ci_scratch,
            weight_deltas=weight_deltas,
        )


def _ppgd_recon_forward(
    ppgd_state: PersistentPGDState,
    component_model: ComponentModel,
    batch_local: Any,
    target_out: Tensor,
    ci_scratch: dict[str, Tensor],
    weight_deltas: dict[str, Tensor],
    cfg: _ThreePoolRuntime,
    p: PhaseProfiler,
) -> tuple[Tensor, int]:
    """Phase ppgd/D4. Final recon loss with the refined sources.

    Returns ``(sum_loss, n_examples)`` raw so the logger can SUM-reduce across
    the PPGD pool to recover ``global_mean = SUM(sum_loss) / SUM(n_examples)``.
    """
    with p.phase("pgd/D4_recon"), autocast_bf16(cfg.bf16_autocast):
        sum_loss, n_examples = ppgd_state.compute_recon_sum_and_n(
            model=component_model,
            batch=batch_local,
            target_out=target_out,
            ci=ci_scratch,
            weight_deltas=weight_deltas,
        )
    assert sum_loss.dim() == 0, f"sum_loss should be scalar; got {sum_loss.shape}"
    assert n_examples > 0, f"n_examples must be positive; got {n_examples}"
    return sum_loss, n_examples


def _autograd_grad_for_vu_and_ci(
    total_ppgd: Tensor,
    component_model: ComponentModel,
    ci_scratch: dict[str, Tensor],
    all_sites: list[str],
    p: PhaseProfiler,
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
    """Phase ppgd/D5. Backward via ``torch.autograd.grad`` for V/U + CI only.

    No source backward here — the warmup inner loop already updated the
    sources. ``retain_graph=False`` since this is the last graph use.
    """
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
    for s in all_sites:
        assert v_grads[s].shape == component_model.components[s].V.shape, (
            f"v_grad[{s!r}] shape mismatch"
        )
        assert u_grads[s].shape == component_model.components[s].U.shape, (
            f"u_grad[{s!r}] shape mismatch"
        )
        assert ci_grads[s].shape == ci_scratch[s].shape, f"ci_grad[{s!r}] shape mismatch"
    return v_grads, u_grads, ci_grads


def _sum_reduce_vu_grads_within_ppgd_pool(
    layout: ThreePoolLayout,
    v_grads: dict[str, Tensor],
    u_grads: dict[str, Tensor],
    p: PhaseProfiler,
) -> None:
    """Phase ppgd/D6. SUM-reduce V/U grads within the PPGD pool.

    Each PPGD rank computed g_VU on its batch slice, scaled by 1/n_ppgd. The
    SUM-reduce here combines them so every PPGD rank ends up holding the
    full-batch V/U gradient.
    """
    with p.phase("pgd/D6_in_pool_sum_reduce"):
        layout.sum_reduce_ppgd_grads([*v_grads.values(), *u_grads.values()])


def _send_g_vu_to_layerwise(
    layout: ThreePoolLayout,
    v_grads: dict[str, Tensor],
    u_grads: dict[str, Tensor],
    p: PhaseProfiler,
) -> None:
    """Phase ppgd/D7. Send V/U grads to LW block leaders.

    PPGD-leader-only send: after the D6 sum-reduce, all PPGD ranks hold
    identical g_VU, so only the leader needs to ship it.
    """
    with p.phase("pgd/D7_send_g_vu_to_lw"):
        layout.send_g_vu_to_layerwise(v_grads, u_grads)


def _send_g_ci_to_ci_pool(
    layout: ThreePoolLayout,
    ci_grads: dict[str, Tensor],
    p: PhaseProfiler,
) -> None:
    """Phase ppgd/D8. Send per-site CI grads to the CI pool.

    Every PPGD rank sends its own batch slice — CI pool stitches the slices
    back into a full-batch tensor before backwarding through the CI-fn graph.
    """
    with p.phase("pgd/D8_send_g_ci_to_ci_pool"):
        layout.send_g_ci_to_ci_pool_ppgd(ci_grads)


def _async_kickoff_recv_vu_from_lw(
    layout: ThreePoolLayout,
    v_templates: dict[str, Tensor],
    u_templates: dict[str, Tensor],
    p: PhaseProfiler,
) -> PendingRecvVU:
    """Phase ppgd/E (async mode). Kickoff async irecv of updated V/U from LW.

    The wait + copy happens at the start of next iter's phase B, so the
    recv overlaps with next iter's CI fn forward on the CI pool.
    """
    with p.phase("pgd/E_kickoff_async_recv_vu"):
        return layout.async_recv_updated_vu_from_layerwise_kickoff(v_templates, u_templates)


def _sync_recv_vu_from_lw_and_copy(
    layout: ThreePoolLayout,
    component_model: ComponentModel,
    v_templates: dict[str, Tensor],
    u_templates: dict[str, Tensor],
    all_sites: list[str],
    p: PhaseProfiler,
) -> None:
    """Phase ppgd/E (sync mode). Blocking recv of updated V/U + copy into model.

    Used when LW is also in sync mode. Cannot coexist with LW's deferred send
    (which doesn't fire until top of next step — would deadlock here).
    """
    with p.phase("pgd/E_sync_recv_vu"):
        v_new, u_new = layout.recv_updated_vu_from_layerwise(v_templates, u_templates)
        _copy_vu_into_model_in_place(component_model, v_new, u_new, all_sites)


def _copy_vu_into_model_in_place(
    component_model: ComponentModel,
    v_new: dict[str, Tensor],
    u_new: dict[str, Tensor],
    all_sites: list[str],
) -> None:
    """In-place copy of received V/U into the model's component params.

    In-place rather than reassign — defense in depth against anyone keeping a
    Python reference to the original tensors (e.g. an optimizer state).
    """
    with torch.no_grad():
        for s in all_sites:
            component_model.components[s].V.copy_(v_new[s])
            component_model.components[s].U.copy_(u_new[s])


def _step_metrics(sum_loss: Tensor, n_examples: int) -> dict[str, float]:
    """Per-step metrics dict: per-rank display scalar + raw (num, den) for the
    cross-PPGD-pool logger reduction.

    Each PPGD rank handles a batch slice; the global mean per position is
    ``SUM(sum_loss) / SUM(n_examples)`` across the pool. ``n_examples`` is
    uniform across PPGD ranks (same batch_local_ppgd) so AVG would numerically
    coincide, but using raw keeps the contract identical to LW's faith/stoch
    cross-block reduction.
    """
    return {
        "loss/ppgd": (sum_loss / n_examples).item(),
        "_raw/ppgd_num": sum_loss.item(),
        "_raw/ppgd_den": float(n_examples),
    }
