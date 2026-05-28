"""PPGD pool training step.

PPGD pool: CI comes from the CI pool, g_CI goes back to the CI pool, and the
final fwd+bwd's source gradient is extracted alongside V/U + CI in a single
multi-target ``autograd.grad`` and used to apply one more PGD source step.
Total source updates per training step = ``n_warmup_steps + 1``.

The step is a sequence of typed phases. The handoff types make the order a type
constraint: the recon sum (``ReconSum``) requires the refined sources from
warmup, the grads require the recon sum, the scaled grads require the raw grads,
and the cross-pool sends consume the scaled grads — so e.g. "send g_VU before
the in-pool reduce" or "step sources before differentiating" become
unrepresentable.

Every cross-pool exchange routes through this PPGD rank's own portal bundle
(``ctx.portals.ci_from_ci_pool`` etc.), so a PPGD step cannot reach for another
pool's edges.

Phases (numbered to match ``DESIGN.md`` ``ppgd/N``):

  A1. Post async irecv for CI_T from the owning CI rank (concurrent with A2).
  A2. target_fwd(batch_T) on the rank's PPGD slice → L_T.
  D1. Compute weight_deltas (V/U-dependent — fresh now).
  D2. Wait CI recv; re-leaf as fp32 for downstream CI grad extraction.
  D3. ``n_warmup_steps`` supplemental source-only PGD iterations on the
      persistent adversarial sources.
  D4. Recon loss with the refined sources (the (N+1)'th forward).
  D5. Backward: one ``torch.autograd.grad`` over the UNSCALED recon sum yields
      raw g_VU + g_CI + g_sources; each consumer's own normalization is applied
      explicitly afterward (V/U + CI: coeff_ppgd / n_examples_global; sources:
      1 / n_examples_local — no coeff, no 1/n_ppgd).
  D5b. Send g_CI to CI pool (every rank, on its batch slice). Peer-to-peer
       point-to-point — no PPGD-internal reduce needed, so fires immediately
       after backward to unblock CI's recv-wait sooner.
  D6. Sum-reduce g_VU within PPGD pool → each rank holds the full-batch grad.
      Source grads are NOT reduced: per_batch_per_position sources are per-rank
      independent (asserted at state construction), so each rank steps its own.
  D6b. Final PGD source step (the (N+1)'th source update): step on this rank's
      own source grads, exactly as warmup does.
  D7. Send g_VU to LW block leaders (PPGD-leader-only).
  E.  Blocking recv of updated V/U from LW at end of step → copy into model.

Architectural note on the absence of a per-site for-loop at the step level:
PPGD really does *one* fused forward + *one* fused backward across all sites
in D3-D5 — the per-site iteration is just gathering tensors into flat
lists for ``torch.autograd.grad`` (an implementation detail of that API
producing multiple grads in one call). The semantically meaningful unit at
the step level is the whole pool, not a single site.
"""

# pyright: reportArgumentType=false

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from param_decomp.component_model import ComponentModel
from param_decomp.metrics.persistent_pgd_state import PersistentPGDState
from param_decomp.torch_helpers import bf16_autocast
from param_decomp_lab.three_pool.context import PPGDContext
from param_decomp_lab.three_pool.loss_strategy import LayerwiseLossStrategy
from param_decomp_lab.three_pool.portals import PendingCiValues, sum_reduce_ppgd_grads
from param_decomp_lab.three_pool.runtime import _ThreePoolRuntime


@dataclass(frozen=True)
class ReconSum:
    """Phases ppgd/D4 (after D3 warmup) output: the UNSCALED Σ-over-this-rank's-
    examples recon loss + the example count. The grads phase requires this."""

    sum_loss: Tensor
    n_examples: int


@dataclass(frozen=True)
class RawGrads:
    """Phase ppgd/D5 output: raw ∂sum_loss/∂· grads, UNSCALED, keyed by site
    (V/U/CI) or source key (sources). Each consumer normalizes in place next."""

    v: dict[str, Tensor]
    u: dict[str, Tensor]
    ci: dict[str, Tensor]
    sources: dict[str, Tensor]


def step_ppgd(
    ctx: PPGDContext,
    component_model: ComponentModel,
    ppgd_state: PersistentPGDState,
    batch: Any,
    cfg: _ThreePoolRuntime,
    strategy: LayerwiseLossStrategy,
    step: int,
    n_steps: int,
    *,
    should_log: bool,
) -> dict[str, float]:
    """One PPGD step: A → D → blocking recv of updated V/U from LW → copy in."""
    device = next(component_model.parameters()).device
    all_sites = list(ctx.world.all_sites)

    ppgd_state.update_lr(step=step, total_steps=n_steps)
    batch_local, seq_len = _slice_batch_for_ppgd(batch, ctx)
    v_templates, u_templates = _vu_templates(component_model, all_sites)

    with strategy.context(component_model.target_model):
        ci_recv_pending = _post_ci_recv(ctx, cfg, seq_len, device)
        target_out = _target_fwd(component_model, batch_local, cfg)
        weight_deltas = component_model.calc_weight_deltas()
        ci_scratch = _wait_ci_and_releaf(ci_recv_pending, ctx, seq_len, cfg)
        recon = _warmup_and_recon(
            ppgd_state, component_model, batch_local, target_out, ci_scratch, weight_deltas, cfg
        )

    raw = _autograd_grads_wrt_vu_ci_and_sources(
        recon.sum_loss, component_model, ci_scratch, all_sites, ppgd_state.sources
    )
    _scale_grads(raw, recon.n_examples, ctx, cfg)

    # CI grad send first: peer-to-peer (each PPGD rank → its paired CI rank), no
    # in-pool reduce needed, and CI's recv is on the critical path. Sequencing it
    # behind the V/U reduce wasted ~110 ms of CI wait time.
    ctx.portals.g_ci_to_ci_pool.send(ctx.role, raw.ci)
    # V/U grads: SUM-reduce across the pool to reassemble the full-batch gradient
    # (the per-rank scale already carries 1/n_ppgd_ranks). Sources are NOT bundled
    # here: their cross-rank reduction is scope-dependent (per_batch_per_position
    # is per-rank-independent and must not be reduced; a blind SUM would mix
    # unrelated per-position sources).
    sum_reduce_ppgd_grads(ctx.world, [*raw.v.values(), *raw.u.values()])
    ctx.portals.g_vu_to_lw.send(ctx.role, raw.v, raw.u)
    # Final (N+1)'th source step. per_batch_per_position sources are per-rank
    # independent, so no cross-rank reduce — step on this rank's own grads,
    # exactly as warmup does.
    ppgd_state.step(raw.sources)

    # ``.item()`` calls force CPU↔GPU sync. With async NCCL ops in D5b/D6/D7
    # still in flight on side streams, syncing here pulls forward the wait
    # for them — making PPGD's critical path appear ~200 ms longer than it
    # actually is. Defer these to log steps only.
    if should_log:
        metrics = {
            "loss/ppgd": (recon.sum_loss / recon.n_examples).item(),
            "_raw/ppgd_num": recon.sum_loss.item(),
            "_raw/ppgd_den": float(recon.n_examples),
        }
    else:
        metrics = {}

    v_new, u_new = ctx.portals.updated_vu_from_lw.post_recv(v_templates, u_templates).wait()
    _copy_vu_into_model_in_place(component_model, v_new, u_new, all_sites)
    return metrics


def _post_ci_recv(
    ctx: PPGDContext,
    cfg: _ThreePoolRuntime,
    seq_len: int,
    device: torch.device,
) -> PendingCiValues:
    """Phase ppgd/A1. Post the async full-model CI-values irecv (waited at D2)."""
    return ctx.portals.ci_from_ci_pool.post_recv(
        ctx.role, cfg.c_per_site, seq_len=seq_len, device=device
    )


def _target_fwd(
    component_model: ComponentModel, batch_local: Any, cfg: _ThreePoolRuntime
) -> Tensor:
    """Phase ppgd/A2. Detached target forward on this rank's batch slice."""
    with torch.no_grad(), bf16_autocast(cfg.bf16_autocast):
        return component_model(batch_local).detach()


def _wait_ci_and_releaf(
    pending: PendingCiValues,
    ctx: PPGDContext,
    seq_len: int,
    cfg: _ThreePoolRuntime,
) -> dict[str, Tensor]:
    """Phase ppgd/D2. Wait the CI recv, re-leaf fp32 with grad for D5."""
    ci_recv = pending.wait()
    ci_scratch = _releaf_ci_fp32_for_grads(ci_recv)
    _assert_ci_scratch_shapes(ci_scratch, ctx, seq_len, cfg)
    return ci_scratch


def _warmup_and_recon(
    ppgd_state: PersistentPGDState,
    component_model: ComponentModel,
    batch_local: Any,
    target_out: Tensor,
    ci_scratch: dict[str, Tensor],
    weight_deltas: dict[str, Tensor],
    cfg: _ThreePoolRuntime,
) -> ReconSum:
    """Phases ppgd/D3 + ppgd/D4. Warmup refines sources, then the (N+1)'th
    recon forward over the refined sources yields the UNSCALED recon sum."""
    # No reduce hook: per-batch-per-position sources are independent per batch
    # element, so each PPGD rank's slice is self-contained (asserted at state
    # construction in optimize.py). Warmup steps on this rank's own grads.
    with bf16_autocast(cfg.bf16_autocast):
        ppgd_state.warmup(
            model=component_model,
            batch=batch_local,
            target_out=target_out,
            ci=ci_scratch,
            weight_deltas=weight_deltas,
        )
    with bf16_autocast(cfg.bf16_autocast):
        sum_loss, n_examples = ppgd_state.compute_recon_sum_and_n(
            model=component_model,
            batch=batch_local,
            target_out=target_out,
            ci=ci_scratch,
            weight_deltas=weight_deltas,
        )
    assert sum_loss.dim() == 0, f"sum_loss should be scalar; got {sum_loss.shape}"
    assert n_examples > 0, f"n_examples must be positive; got {n_examples}"
    return ReconSum(sum_loss=sum_loss, n_examples=n_examples)


def _scale_grads(
    raw: RawGrads, n_examples_local: int, ctx: PPGDContext, cfg: _ThreePoolRuntime
) -> None:
    """Phase ppgd/D5 (post). Apply each consumer's own normalization in place.

    V/U and CI reproduce the serial gradient of the canonical PPGD loss
      coeff_ppgd * recon_sum_loss / n_examples_global,
    where n_examples_global = n_examples_local * n_ppgd_ranks. Each rank holds
    only its batch-slice's contribution, so the per-rank scale carries the
    1/n_ppgd_ranks and the in-pool SUM-reduce reassembles the full-batch grad.

    Sources are per-rank-local adversary state optimized against THIS rank's own
    recon mean (recon_sum_loss / n_examples_local) — no coeff, no 1/n_ppgd
    (those are V/U-reduction artifacts). Identical to the warmup source grad.

    Folding any one consumer's scaling into the differentiated scalar — as an
    earlier version did, dividing by n_ppgd for the V/U reduce — silently
    mis-scales the others (it gave the source step a spurious 1/n_ppgd).
    """
    n_ppgd_ranks = ctx.world.n_ppgd
    vu_and_ci_grad_scale = cfg.coeff_ppgd / (n_examples_local * n_ppgd_ranks)
    source_grad_scale = 1.0 / n_examples_local
    _scale_grads_in_place(raw.v, vu_and_ci_grad_scale)
    _scale_grads_in_place(raw.u, vu_and_ci_grad_scale)
    _scale_grads_in_place(raw.ci, vu_and_ci_grad_scale)
    _scale_grads_in_place(raw.sources, source_grad_scale)


def _slice_batch_for_ppgd(batch: Any, ctx: PPGDContext) -> tuple[Any, int]:
    """Pull this PPGD rank's batch slice + extract its seq_len."""
    sl = ctx.role.batch_slice(ctx.world.batch_local_ppgd)
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
    """V/U tensors per site — used as recv buffers for the V/U recv from LW."""
    v_templates: dict[str, Tensor] = {s: component_model.components[s].V for s in all_sites}
    u_templates: dict[str, Tensor] = {s: component_model.components[s].U for s in all_sites}
    return v_templates, u_templates


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
    ctx: PPGDContext,
    seq_len: int,
    cfg: _ThreePoolRuntime,
) -> None:
    """Sanity-check the CI scratch tensors match what the CI pool said it'd send.

    Catches a wrong ``c_per_site`` config or a per-rank batch mismatch fast.
    """
    batch_local_ppgd = ctx.world.batch_local_ppgd
    for s, c in cfg.c_per_site.items():
        t = ci_scratch[s]
        assert t.shape == (batch_local_ppgd, seq_len, c), (
            f"ci_scratch[{s!r}] shape {tuple(t.shape)} != "
            f"expected ({batch_local_ppgd}, {seq_len}, {c})"
        )


def _scale_grads_in_place(grads: dict[str, Tensor], scale: float) -> None:
    """Multiply each gradient by ``scale`` in place (mutates the autograd outputs)."""
    for grad in grads.values():
        grad.mul_(scale)


def _autograd_grads_wrt_vu_ci_and_sources(
    recon_sum_loss: Tensor,
    component_model: ComponentModel,
    ci_scratch: dict[str, Tensor],
    all_sites: list[str],
    sources: dict[str, Tensor],
) -> RawGrads:
    """Phase ppgd/D5. One ``torch.autograd.grad`` over the UNSCALED recon sum.

    Differentiates ``recon_sum_loss`` — the raw Σ-over-this-rank's-examples recon
    loss, NOT yet divided by any example count or multiplied by any coefficient —
    once, w.r.t. V/U, the received-CI scratch leaves, and the PPGD sources. The
    caller (``step_ppgd``) applies each consumer's own normalization afterward,
    because the three consumers need different scalings and folding any of them
    into this scalar would mis-scale the others.

    Fusing all four gradient sets into one backward avoids a separate source-only
    forward+backward. ``retain_graph=False`` — last use of the graph.

    Returns a ``RawGrads`` bundle, each dict keyed by site (sources keyed by
    source key).
    """
    source_keys = list(sources.keys())
    v_params = [component_model.components[s].V for s in all_sites]
    u_params = [component_model.components[s].U for s in all_sites]
    ci_leaves = [ci_scratch[s] for s in all_sites]
    source_tensors = [sources[k] for k in source_keys]

    # autograd.grad returns grads in the order targets are passed. Keep the four
    # groups contiguous and split the result back out by group length.
    grad_targets = v_params + u_params + ci_leaves + source_tensors
    flat_grads = torch.autograd.grad(recon_sum_loss, grad_targets, retain_graph=False)

    n_sites = len(all_sites)
    n_sources = len(source_keys)
    v_grads = dict(zip(all_sites, flat_grads[0:n_sites], strict=True))
    u_grads = dict(zip(all_sites, flat_grads[n_sites : 2 * n_sites], strict=True))
    ci_grads = dict(zip(all_sites, flat_grads[2 * n_sites : 3 * n_sites], strict=True))
    source_grads = dict(
        zip(source_keys, flat_grads[3 * n_sites : 3 * n_sites + n_sources], strict=True)
    )
    for s in all_sites:
        assert v_grads[s].shape == component_model.components[s].V.shape, (
            f"v_grad[{s!r}] shape mismatch"
        )
        assert u_grads[s].shape == component_model.components[s].U.shape, (
            f"u_grad[{s!r}] shape mismatch"
        )
        assert ci_grads[s].shape == ci_scratch[s].shape, f"ci_grad[{s!r}] shape mismatch"
    for k in source_keys:
        assert source_grads[k].shape == sources[k].shape, f"source_grad[{k!r}] shape mismatch"
    return RawGrads(v=v_grads, u=u_grads, ci=ci_grads, sources=source_grads)


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
