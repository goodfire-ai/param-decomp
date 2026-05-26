"""PPGD pool training step — split into ``step_ppgd`` and ``finalize_ppgd_async_drain``.

Recast of ``two_pool.pool_b.step_pool_b`` for 3-pool. Same PPGD math; the
differences are (a) CI comes from the CI pool not LW leaders, (b) g_CI goes
back to CI pool not LW, and (c) the final source step is fused with the
V/U+CI backward (a single multi-target ``autograd.grad`` produces all
gradients from one forward), so D3 runs ``n_warmup_steps - 1`` iterations
instead of N.

Phases (numbered to match ``DESIGN.md`` ``ppgd/N``):

  A1. Post async irecv for CI_T from the owning CI rank (concurrent with A2).
  A2. target_fwd(batch_T) on the rank's PPGD slice → L_T.
  B.  Async-mode only: wait + copy prev iter's V/U recv (overlaps with A2's
      kernels on the default CUDA stream while NCCL waits run on theirs).
  D1. Compute weight_deltas (V/U-dependent — fresh now).
  D2. Wait CI recv; re-leaf as fp32 for downstream CI grad extraction.
  D3. PPGD warmup refines persistent adversarial sources in place
      (``max(n_warmup_steps - 1, 0)`` iterations; the final source step is
      fused into D5+D6b below).
  D4. Recon loss with refined sources.
  D5. Backward: ``torch.autograd.grad`` for g_VU + g_CI + g_sources in one
      pass.
  D5b. Send g_CI to CI pool (every rank, on its batch slice). Peer-to-peer
       point-to-point — no PPGD-internal reduce needed, so fires immediately
       after backward to unblock CI's recv-wait sooner.
  D6. Sum-reduce g_VU + g_sources within PPGD pool → each rank holds the
      full-batch grad.
  D6b. Final fused PGD source step using the reduced source grads.
  D7. Send g_VU to LW block leaders (PPGD-leader-only).
  E.  Recv updated V/U from LW:
        * sync mode → blocking recv at end of step (returns
          ``(metrics, None)``);
        * async mode → kickoff async recv, return handles for next step's
          phase B.

Architectural note on the absence of a per-site for-loop at the step level:
PPGD really does *one* fused forward + *one* fused backward across all sites
in D3-D5 — the per-site iteration is just gathering tensors into flat
lists for ``torch.autograd.grad`` (an implementation detail of that API
producing multiple grads in one call). The semantically meaningful unit at
the step level is the whole pool, not a single site.
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
    should_log: bool,
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
        with p.phase("pgd/A1_post_async_recv_ci"):
            ci_recv_pending = layout.async_recv_ci_from_ci_pool_ppgd(
                cfg.c_per_site, seq_len=seq_len, device=device
            )
        with p.phase("pgd/A2_target_fwd"), torch.no_grad(), autocast_bf16(cfg.bf16_autocast):
            target_out = component_model(batch_local).detach()

        # Async-mode bookkeeping: this iter's V/U arrived in the prev iter's
        # async kickoff. Wait + copy here, overlapping the NCCL wait with the
        # A2 target_fwd kernels enqueued above.
        if defer_vu_opt and prev_pending_recv_vu is not None:
            _wait_and_copy_prev_vu_into_model(
                layout, component_model, prev_pending_recv_vu, v_templates, u_templates, p
            )

        with p.phase("pgd/D1_calc_weight_deltas"):
            weight_deltas = component_model.calc_weight_deltas()
        with p.phase("pgd/D2_wait_ci_recv"):
            ci_recv = ci_recv_pending.wait_and_unpack()
        ci_scratch = _releaf_ci_fp32_for_grads(ci_recv)
        _assert_ci_scratch_shapes(ci_scratch, layout, seq_len, cfg)

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
        assert sum_loss.dim() == 0, f"sum_loss should be scalar; got {sum_loss.shape}"
        assert n_examples > 0, f"n_examples must be positive; got {n_examples}"

    # Scale by 1/n_ppgd so D6's SUM-reduce produces the full-batch gradient.
    total_ppgd = cfg.coeff_ppgd * (sum_loss / n_examples) / layout.world.n_ppgd

    # Fuse the final PGD source step into this backward: one autograd.grad
    # call producing V/U + CI + source grads. The warmup() loop above ran
    # ``max(n_warmup_steps - 1, 0)`` iterations; the final source update
    # happens here using grads from this backward (after the in-pool reduce).
    # Cuts one full PPGD forward+backward off D3_warmup per step at the
    # cost of computing V/U+CI grads at source iterate S_{N-1} instead of S_N
    # (benign for persistent PGD whose adversarial signal accumulates across
    # batches).
    v_grads, u_grads, ci_grads, source_grads = _autograd_grad_for_vu_ci_and_sources(
        total_ppgd,
        component_model,
        ci_scratch,
        all_sites,
        ppgd_state.sources,
        p,
    )
    assert source_grads is not None, "fused source step always runs"
    # CI grad send first: it's peer-to-peer (each PPGD rank → its paired CI
    # rank) so it doesn't need the in-pool reduce, and CI's recv is on the
    # critical path. Sequencing it behind D6 wasted ~110 ms of CI wait time.
    with p.phase("pgd/D5b_send_g_ci_to_ci_pool"):
        layout.send_g_ci_to_ci_pool_ppgd(ci_grads)
    with p.phase("pgd/D6_in_pool_sum_reduce"):
        # Reduce sources alongside V/U. Sign-PGD makes SUM-vs-AVG equivalent
        # for the source step (sign(SUM) == sign(AVG)); reducing here keeps
        # source state consistent across PPGD ranks for the step below.
        layout.sum_reduce_ppgd_grads(
            [*v_grads.values(), *u_grads.values(), *source_grads.values()]
        )
    with p.phase("pgd/D6b_apply_source_step"):
        ppgd_state.apply_source_step_from_grads(source_grads)
    with p.phase("pgd/D7_send_g_vu_to_lw"):
        layout.send_g_vu_to_layerwise(v_grads, u_grads)

    # ``.item()`` calls force CPU↔GPU sync. With async NCCL ops in D5b/D6/D7
    # still in flight on side streams, syncing here pulls forward the wait
    # for them — making PPGD's critical path appear ~200 ms longer than it
    # actually is. Defer these to log steps only.
    if should_log:
        with p.phase("pgd/Dx_metrics_sync"):
            metrics = {
                "loss/ppgd": (sum_loss / n_examples).item(),
                "_raw/ppgd_num": sum_loss.item(),
                "_raw/ppgd_den": float(n_examples),
            }
    else:
        metrics = {}

    if defer_vu_opt:
        with p.phase("pgd/E_kickoff_async_recv_vu"):
            new_pending = layout.async_recv_updated_vu_from_layerwise_kickoff(
                v_templates, u_templates
            )
        return metrics, new_pending

    with p.phase("pgd/E_sync_recv_vu"):
        v_new, u_new = layout.recv_updated_vu_from_layerwise(v_templates, u_templates)
        _copy_vu_into_model_in_place(component_model, v_new, u_new, all_sites)
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
    """Pull this PPGD rank's batch slice + extract its seq_len."""
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


def _wait_and_copy_prev_vu_into_model(
    layout: ThreePoolLayout,
    component_model: ComponentModel,
    prev_pending_recv_vu: PendingRecvVU,
    v_templates: dict[str, Tensor],
    u_templates: dict[str, Tensor],
    p: PhaseProfiler,
) -> None:
    """Phase ppgd/B (async mode only). Wait + copy prev iter's V/U recv into model.

    Blocking wait overlaps with the A2 target_fwd kernels enqueued just above —
    that's the headline win of async mode.
    """
    with p.phase("pgd/B_wait_and_copy_prev_vu"):
        v_new, u_new = layout.wait_and_unpack_updated_vu(
            prev_pending_recv_vu, v_templates, u_templates
        )
        _copy_vu_into_model_in_place(component_model, v_new, u_new, list(layout.world.all_sites))


def _releaf_ci_fp32_for_grads(ci_recv: dict[str, Tensor]) -> dict[str, Tensor]:
    """Upcast CI (bf16 on the wire) to fp32 and re-leaf with ``requires_grad=True``
    so the autograd.grad call below populates fp32 ``.grad`` that the CI pool
    can merge into its fp32 CI-fn grads.
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


def _autograd_grad_for_vu_ci_and_sources(
    total_ppgd: Tensor,
    component_model: ComponentModel,
    ci_scratch: dict[str, Tensor],
    all_sites: list[str],
    sources: dict[str, Tensor],
    p: PhaseProfiler,
) -> tuple[
    dict[str, Tensor], dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]
]:
    """Phase ppgd/D5. One ``torch.autograd.grad`` for V/U + CI + PPGD sources.

    Returning all four gradient sets from a single backward lets the caller
    fuse the final PGD source step with the V/U+CI backward (saving a separate
    forward + source-only backward; see ``persistent_pgd_state.warmup``
    docstring). The flat target list is an artifact of ``autograd.grad``
    returning grads in input order; we split back into dicts keyed by site.
    ``retain_graph=False`` since this is the last graph use.
    """
    source_keys = list(sources.keys())
    with p.phase("pgd/D5_backward"):
        params: list[Tensor] = []
        for s in all_sites:
            params.append(component_model.components[s].V)
            params.append(component_model.components[s].U)
        ci_list = [ci_scratch[s] for s in all_sites]
        source_list = [sources[k] for k in source_keys]
        grads = torch.autograd.grad(
            total_ppgd, params + ci_list + source_list, retain_graph=False
        )

    n_sites = len(all_sites)
    v_grads = {s: grads[2 * i] for i, s in enumerate(all_sites)}
    u_grads = {s: grads[2 * i + 1] for i, s in enumerate(all_sites)}
    ci_grads = {s: grads[2 * n_sites + i] for i, s in enumerate(all_sites)}
    source_grads = {k: grads[3 * n_sites + i] for i, k in enumerate(source_keys)}
    for s in all_sites:
        assert v_grads[s].shape == component_model.components[s].V.shape, (
            f"v_grad[{s!r}] shape mismatch"
        )
        assert u_grads[s].shape == component_model.components[s].U.shape, (
            f"u_grad[{s!r}] shape mismatch"
        )
        assert ci_grads[s].shape == ci_scratch[s].shape, f"ci_grad[{s!r}] shape mismatch"
    for k in source_keys:
        assert source_grads[k].shape == sources[k].shape, (
            f"source_grad[{k!r}] shape mismatch"
        )
    return v_grads, u_grads, ci_grads, source_grads


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
