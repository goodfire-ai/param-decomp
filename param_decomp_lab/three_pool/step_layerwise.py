"""Layerwise pool training step.

Trains V/U on the LW pool with the CI fn living on the CI pool.

The step is a sequence of typed phases threaded through ``strategy.context``.
The handoff types make the dependency order a type constraint: the CI grads can
only be sent once the per-site streaming backward has populated the re-leafed CI
tensors' ``.grad`` (``_send_g_ci`` consumes the ``CiLeaves`` those grads live
on), and the CI values can only be consumed after the posted recv is waited
(``CiLeaves`` is built from a ``PendingCiValues.wait()``).

Every cross-pool exchange routes through this LW rank's own portal bundle
(``ctx.portals.ci_from_ci_pool`` etc.), so an LW step cannot reach for another
pool's edges.

Phases (numbered to match ``DESIGN.md`` ``lw/N``):

  A1. Post async irecv for CI_T from the owning CI rank (overlaps with A2).
  A2. target_fwd(batch_T) → L_T on this rank's batch slice.
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
  E.  Tail. In-block all_reduce → grad clip → AdamW → async send V/U.

Phases A1 + D1 give the headline overlap (CI recv on the NIC while
faithfulness runs on the GPU).
"""

# pyright: reportArgumentType=false, reportOperatorIssue=false, reportAttributeAccessIssue=false

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from param_decomp._trace import phase_trace_enabled, trace
from param_decomp.component_model import ComponentModel
from param_decomp.grad_clip import cross_pool_clip_grad_norm
from param_decomp.masks import make_mask_infos
from param_decomp.torch_helpers import bf16_autocast
from param_decomp_lab.three_pool.context import LWContext
from param_decomp_lab.three_pool.loss_strategy import LayerwiseLossStrategy
from param_decomp_lab.three_pool.portals import (
    LWPortals,
    PendingCiValues,
    all_reduce_grads_in_block,
)
from param_decomp_lab.three_pool.role import LWRole
from param_decomp_lab.three_pool.runtime import _ThreePoolRuntime


@dataclass(frozen=True)
class Faith:
    """Phase lw/D1 output: faithfulness loss (already backward'd into V/U .grad)
    plus the raw sum-sq + numel the logger needs for the global ratio."""

    loss: Tensor
    sum_sq: Tensor
    numel: int


@dataclass(frozen=True)
class CiLeaves:
    """Phase lw/D2 output: re-leafed fp32 CI values (requires_grad=True) per
    owned site. The layerwise backward populates ``leaf.grad``; phase D4 reads
    that grad off these exact leaves to ship back to the CI pool."""

    per_site: dict[str, Tensor]


@dataclass(frozen=True)
class Stoch:
    """Phase lw/D3 output: accumulated stochastic-recon display value (GPU
    tensor) + the count of owned sites it averages over."""

    total: Tensor
    n_owned: int


def step_layerwise(
    ctx: LWContext,
    component_model: ComponentModel,
    optimizer: torch.optim.Optimizer,
    all_params: list[nn.Parameter],
    batch: Any,
    cfg: _ThreePoolRuntime,
    strategy: LayerwiseLossStrategy,
    *,
    should_log: bool,
) -> dict[str, float]:
    """One LW step: A → D → tail (all_reduce, clip, opt, async send V/U)."""
    device = next(component_model.parameters()).device

    batch_local, seq_len = _slice_batch_for_layerwise(batch, ctx)

    with strategy.context(component_model.target_model):
        ci_recv_pending = _post_ci_recv(ctx, cfg, seq_len, device)
        target_local = _target_fwd(component_model, batch_local, cfg)

    for param in all_params:
        param.grad = None

    with strategy.context(component_model.target_model):
        faith = _faithfulness_phase(component_model, device, cfg)

        ci_leaves = _wait_ci_and_releaf(ci_recv_pending, ctx, seq_len, cfg)
        stoch = _layerwise_streaming_phase(
            component_model, batch_local, target_local, ci_leaves, ctx, cfg, strategy
        )

        _send_g_ci(ctx.portals, ctx.role, ci_leaves)
        _recv_and_combine_g_vu(ctx, component_model)

    if should_log:
        stoch_total_value = stoch.total.item()
        metrics = {
            "loss/faith": faith.loss.item(),
            "loss/stoch": stoch_total_value / stoch.n_owned,
            "_raw/faith_num": faith.sum_sq.item(),
            "_raw/faith_den": float(faith.numel),
            "_raw/stoch_num": stoch_total_value,
            "_raw/stoch_den": float(stoch.n_owned),
        }
    else:
        metrics = {}

    _sync_tail(ctx, component_model, optimizer, all_params, cfg)
    return metrics


def _post_ci_recv(
    ctx: LWContext,
    cfg: _ThreePoolRuntime,
    seq_len: int,
    device: torch.device,
) -> PendingCiValues:
    """Phase lw/A1. Post the async CI-values irecv (waited at D2)."""
    owned_sites = ctx.role.owned_sites
    return ctx.portals.ci_from_ci_pool.post_recv(
        ctx.role,
        {s: cfg.c_per_site[s] for s in owned_sites},
        seq_len=seq_len,
        device=device,
    )


def _target_fwd(
    component_model: ComponentModel, batch_local: Any, cfg: _ThreePoolRuntime
) -> Tensor:
    """Phase lw/A2. Detached target forward on this rank's batch slice."""
    with torch.no_grad(), bf16_autocast(cfg.bf16_autocast):
        return component_model(batch_local).detach()


def _faithfulness_phase(
    component_model: ComponentModel, device: torch.device, cfg: _ThreePoolRuntime
) -> Faith:
    """Phase lw/D1. Faithfulness loss + backward into V/U .grad."""
    loss, sum_sq, numel = _faithfulness_loss(component_model, device, cfg.numel_global)
    (cfg.coeff_faith * loss).backward()
    return Faith(loss=loss, sum_sq=sum_sq, numel=numel)


def _wait_ci_and_releaf(
    pending: PendingCiValues,
    ctx: LWContext,
    seq_len: int,
    cfg: _ThreePoolRuntime,
) -> CiLeaves:
    """Phase lw/D2. Wait the CI recv, re-leaf fp32 with grad for the bwd."""
    ci_recv = pending.wait()
    per_site = _releaf_ci_fp32_for_grads(ci_recv, ctx.role.owned_sites)
    _assert_ci_recv_shapes(per_site, ctx, seq_len, cfg)
    return CiLeaves(per_site=per_site)


def _layerwise_streaming_phase(
    component_model: ComponentModel,
    batch_local: Any,
    target_local: Tensor,
    ci_leaves: CiLeaves,
    ctx: LWContext,
    cfg: _ThreePoolRuntime,
    strategy: LayerwiseLossStrategy,
) -> Stoch:
    """Phase lw/D3. Per-owned-site stochastic recon, streaming fwd+bwd."""
    owned_sites = ctx.role.owned_sites
    n_sites_total = len(cfg.c_per_site)
    device = target_local.device
    with bf16_autocast(cfg.bf16_autocast):
        # Accumulate the display value as a GPU tensor (not a Python float) so
        # the per-site ``.item()`` doesn't force a CPU↔GPU sync that serializes
        # each site's bwd against the next. ``loss_s.detach()`` so accumulator
        # doesn't retain autograd graph.
        stoch_total_t = torch.zeros((), device=device)
        for i, s in enumerate(owned_sites):
            if phase_trace_enabled():
                trace(f"lw/D3 site {i + 1}/{len(owned_sites)}: {s} fwd+bwd")
            loss_s, n_positions = _layerwise_one_site(
                component_model, batch_local, target_local, ci_leaves.per_site, s, strategy
            )
            assert loss_s.dim() == 0, f"layerwise loss for site {s!r} must be scalar"
            (cfg.coeff_stoch * loss_s / (n_positions * n_sites_total)).backward()
            stoch_total_t = stoch_total_t + (loss_s.detach() / n_positions)
    return Stoch(total=stoch_total_t, n_owned=len(owned_sites))


def _send_g_ci(portals: LWPortals, role: LWRole, ci_leaves: CiLeaves) -> None:
    """Phase lw/D4. Ship per-owned-site CI grads back to the CI pool."""
    g_ci_owned = {s: ci_leaves.per_site[s].grad for s in role.owned_sites}
    assert all(g is not None for g in g_ci_owned.values()), (
        "layerwise backward should have populated ci_leaves[s].grad"
    )
    portals.g_ci_to_ci_pool.send(role, g_ci_owned)


def _recv_and_combine_g_vu(ctx: LWContext, component_model: ComponentModel) -> None:
    """Phases lw/D5 + lw/D6. Recv PPGD's V/U grads, add to existing .grad."""
    v_grads_pgd, u_grads_pgd = _recv_g_vu_from_ppgd(ctx, component_model)
    _combine_vu_grads_in_place(component_model, ctx.role.owned_sites, v_grads_pgd, u_grads_pgd)


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
    overwritten each step. So warmup only makes sense on LW.
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


def _slice_batch_for_layerwise(batch: Any, ctx: LWContext) -> tuple[Any, int]:
    """Pull this LW rank's batch slice + extract its seq_len."""
    sl = ctx.role.batch_slice(ctx.world.batch_local_lw)
    batch_local = batch[sl] if isinstance(batch, Tensor) else batch
    if isinstance(batch_local, Tensor):
        seq_len = batch_local.shape[1] if batch_local.ndim >= 2 else 1
    else:
        assert isinstance(batch_local, dict) and "input_ids" in batch_local
        seq_len = batch_local["input_ids"].shape[1]
    return batch_local, seq_len


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
    ctx: LWContext,
    seq_len: int,
    cfg: _ThreePoolRuntime,
) -> None:
    """Sanity-check the CI leaves match what the CI pool said it'd send.

    Catches a wrong ``c_per_site`` config or a per-rank batch mismatch fast.
    """
    batch_local_lw = ctx.world.batch_local_lw
    for s in ctx.role.owned_sites:
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
    ctx: LWContext,
    component_model: ComponentModel,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Phase lw/D5. Recv V/U grads from PPGD pool (leader recvs, in-block bcast)."""
    owned_sites = ctx.role.owned_sites
    v_templates = {s: component_model.components[s].V for s in owned_sites}
    u_templates = {s: component_model.components[s].U for s in owned_sites}
    v_grads_pgd, u_grads_pgd = ctx.portals.g_vu_from_ppgd.recv(ctx.role, v_templates, u_templates)
    for s in owned_sites:
        assert v_grads_pgd[s].shape == component_model.components[s].V.shape, (
            f"v_grads_pgd[{s!r}] shape mismatch from PPGD send"
        )
        assert u_grads_pgd[s].shape == component_model.components[s].U.shape, (
            f"u_grads_pgd[{s!r}] shape mismatch from PPGD send"
        )
    return v_grads_pgd, u_grads_pgd


def _combine_vu_grads_in_place(
    component_model: ComponentModel,
    owned_sites: tuple[str, ...],
    v_grads_pgd: dict[str, Tensor],
    u_grads_pgd: dict[str, Tensor],
) -> None:
    """Phase lw/D6. Add PPGD's V/U grads to .grad (which already has faith+lw)."""
    for s in owned_sites:
        comp = component_model.components[s]
        assert comp.V.grad is not None and comp.U.grad is not None, (
            "faith + layerwise should have populated V/U .grad"
        )
        comp.V.grad.add_(v_grads_pgd[s])
        comp.U.grad.add_(u_grads_pgd[s])


def _sync_tail(
    ctx: LWContext,
    component_model: ComponentModel,
    optimizer: torch.optim.Optimizer,
    all_params: list[nn.Parameter],
    cfg: _ThreePoolRuntime,
) -> None:
    """Phase lw/E (sync mode). Blocking all_reduce → clip → AdamW → async send V/U.

    Safe to coexist with PPGD's sync recv at end of step T.
    """
    _wait_pending_weight_send(component_model)
    all_reduce_grads_in_block(ctx.world, ctx.role, all_params)
    if cfg.grad_clip_norm_components is not None:
        cross_pool_clip_grad_norm(
            all_params,
            cfg.grad_clip_norm_components,
            group=ctx.world.layerwise_pool_group,
            n_replicas=ctx.world.n_per_block,
        )
    optimizer.step()
    _async_send_owned_vu_to_ppgd(component_model, ctx)


def _async_send_owned_vu_to_ppgd(component_model: ComponentModel, ctx: LWContext) -> None:
    """Kickoff async ship of updated V/U → PPGD. Stash the in-flight send on the model."""
    owned_sites = ctx.role.owned_sites
    v_owned = {s: component_model.components[s].V for s in owned_sites}
    u_owned = {s: component_model.components[s].U for s in owned_sites}
    in_flight = ctx.portals.updated_vu_to_ppgd.send(ctx.role, v_owned, u_owned)
    component_model._pending_weight_send = in_flight  # type: ignore[attr-defined]


def _wait_pending_weight_send(component_model: ComponentModel) -> None:
    """Wait + clear any pending async V/U send from a previous iter.

    Defense against the opt step mutating V/U while the previous async send
    still reads it.
    """
    pending = getattr(component_model, "_pending_weight_send", None)
    if pending is not None:
        pending.wait()
        component_model._pending_weight_send = None  # type: ignore[attr-defined]


def _faithfulness_loss(
    component_model: ComponentModel, device: torch.device, numel_global: int
) -> tuple[Tensor, Tensor, int]:
    """‖W_target − VU.T‖²_F / numel_global, summed across this rank's owned sites.

    We divide by ``numel_global`` not ``numel_owned`` to keep the per-element
    grad scale aligned with single-pool's, so the unclipped faithfulness warmup
    converges to the same V/U as single-pool.

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
