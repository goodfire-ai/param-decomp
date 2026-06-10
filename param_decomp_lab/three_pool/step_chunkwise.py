"""Chunkwise pool training step.

Trains V/U on the chunkwise pool with the CI fn living on the CI pool.

The step is a sequence of typed phases threaded through ``strategy.context``.
The handoff types make the dependency order a type constraint: the CI grads can
only be sent once the per-site streaming backward has populated the re-leafed CI
tensors' ``.grad`` (``_send_g_ci`` consumes the ``CiLeaves`` those grads live
on), and the CI values can only be consumed after the posted recv is waited
(``CiLeaves`` is built from a ``PendingCiValues.wait()``).

Every cross-pool exchange routes through this chunk rank's own portal bundle
(``ctx.portals.ci_from_ci_pool`` etc.), so a chunkwise step cannot reach for another
pool's edges.

Phases (numbered to match ``DESIGN.md`` ``chunk/N``):

  A1. Post async irecv for CI_T from the owning CI rank (overlaps with A2).
  A2. target_fwd(batch_T) → L_T on this rank's batch slice.
  C.  Zero ``param.grad`` for fresh accumulation.
  D1. Faithfulness loss + backward (V/U-only, doesn't need CI).
  D2. Wait CI recv; re-leaf as fp32 ``requires_grad=True`` so the recon
      backward populates ``leaf.grad`` for D4.
  D3. Stochastic recon, streaming one forward+backward per entry of this chunk's
      recon plan (``cfg.recon_plan`` — see ``recon_plan.py``). The default
      per-site plan does one forward per site (the original layerwise loop);
      a subset plan does joint/per-position-routed forwards over all the chunk's
      sites. The for-loop lives at this step level (one forward+backward per entry
      bounds peak activation memory). The stoch grad is normalized by ``N_est`` (the
      global total of recon forwards), generalizing the old ``n_sites_total``.
  D4. Send g_CI back to CI pool (per-rank, on this chunk's sites).
  D5. Recv g_VU from PPGD pool (chunk leader recvs, then in-chunk bcast).
  D6. Combine V/U grads: faith + stoch already in ``.grad``; add PPGD's.
  E.  Tail. In-chunk all_reduce → grad clip → AdamW → async send V/U.

Phases A1 + D1 give the headline overlap (CI recv on the NIC while
faithfulness runs on the GPU).
"""

# pyright: reportArgumentType=false, reportOperatorIssue=false, reportAttributeAccessIssue=false

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

from param_decomp.component_model import ComponentModelProtocol
from param_decomp.grad_clip import cross_pool_clip_grad_norm
from param_decomp.masks import RoutingMasks, make_mask_infos
from param_decomp.torch_helpers import bf16_autocast
from param_decomp_lab.experiments.lm.vendored.component_model import LMComponentModel
from param_decomp_lab.three_pool.context import ChunkContext
from param_decomp_lab.three_pool.portals import (
    ChunkPortals,
    PendingCiValues,
    all_reduce_grads_in_chunk,
)
from param_decomp_lab.three_pool.recon_loss_strategy import ReconLossStrategy
from param_decomp_lab.three_pool.recon_plan import ForwardRouting
from param_decomp_lab.three_pool.reductions import per_param_grad_norms
from param_decomp_lab.three_pool.role import ChunkRole
from param_decomp_lab.three_pool.runtime import _ThreePoolRuntime

WeightDeltasFn = Callable[[tuple[str, ...]], dict[str, Tensor]]
"""Produces FRESH grad-carrying ``{site: target_weight - VU}`` deltas for the given
sites. Called once per recon forward so each forward owns an independent delta subgraph
(see ``recon_one_forward``); placement is the fn's concern (plain for the 3-pool, the
DTensor-aware ``calc_weight_deltas`` for the flat FSDP path)."""


@dataclass(frozen=True)
class Faith:
    """Phase chunk/D1 output: faithfulness loss (already backward'd into V/U .grad)
    plus the raw sum-sq + numel the logger needs for the global ratio."""

    loss: Tensor
    sum_sq: Tensor
    numel: int


@dataclass(frozen=True)
class CiLeaves:
    """Phase chunk/D2 output: re-leafed fp32 CI values (requires_grad=True) per
    site. The recon backward populates ``leaf.grad``; phase D4 reads that grad
    off these exact leaves to ship back to the CI pool."""

    per_site: dict[str, Tensor]


@dataclass(frozen=True)
class Stoch:
    """Phase chunk/D3 output: accumulated stochastic-recon display value (GPU
    tensor) + the count of recon forwards it averages over."""

    total: Tensor
    n_forwards: int


def step_chunkwise(
    ctx: ChunkContext,
    component_model: LMComponentModel,
    optimizer: torch.optim.Optimizer,
    all_params: list[nn.Parameter],
    batch: Any,
    cfg: _ThreePoolRuntime,
    strategy: ReconLossStrategy,
    *,
    should_log: bool,
) -> dict[str, float]:
    """One chunkwise step: A → D → tail (all_reduce, clip, opt, async send V/U)."""
    device = next(component_model.parameters()).device

    batch_local, seq_len = _slice_batch_for_chunkwise(batch, ctx)

    # Residual-start: cache the clean residual entering the decomposed layer once; the clean
    # target_fwd and every masked recon forward below run only the suffix (no-op for GPT-2 q/k,
    # where the decomposed layer is 0).
    with component_model.use_cached_residual(batch_local):
        with strategy.context():
            ci_recv_pending = _post_ci_recv(ctx, cfg, seq_len, device)
            target_local = _target_fwd(component_model, batch_local, cfg)

        for param in all_params:
            param.grad = None

        with strategy.context():
            faith = _faithfulness_phase(component_model, device, cfg, ctx)
            # Snapshot the faith-only V/U grad (chunk leader; non-leaders skip faith)
            # before stoch accumulates on top, for the per-loss grad-norm breakdown.
            faith_vu = (
                _snapshot_owned_vu_grads(component_model, ctx.role.sites) if should_log else None
            )

            ci_leaves = _wait_ci_and_releaf(ci_recv_pending, ctx, seq_len, cfg)
            stoch = _chunkwise_streaming_phase(
                component_model, batch_local, target_local, ci_leaves, ctx, cfg, strategy
            )

            _send_g_ci(ctx.portals, ctx.role, ci_leaves)
            ppgd_vu = _recv_and_combine_g_vu(ctx, component_model, return_ppgd=should_log)

    if should_log:
        stoch_total_value = stoch.total.item()
        metrics = {
            "loss/faith": faith.loss.item(),
            "loss/stoch": stoch_total_value / stoch.n_forwards,
            "_raw/faith_num": faith.sum_sq.item(),
            "_raw/faith_den": float(faith.numel),
            "_raw/stoch_num": stoch_total_value,
            "_raw/stoch_den": float(stoch.n_forwards),
            **_component_grad_sumsq_by_loss(component_model, ctx, faith_vu, ppgd_vu),
        }
    else:
        metrics = {}

    grad_norms = _sync_tail(ctx, component_model, optimizer, all_params, cfg, should_log=should_log)
    metrics.update(grad_norms)
    return metrics


def _post_ci_recv(
    ctx: ChunkContext,
    cfg: _ThreePoolRuntime,
    seq_len: int,
    device: torch.device,
) -> PendingCiValues:
    """Phase chunk/A1. Post the async CI-values irecv (waited at D2)."""
    sites = ctx.role.sites
    return ctx.portals.ci_from_ci_pool.post_recv(
        ctx.role,
        {s: cfg.c_per_site[s] for s in sites},
        seq_len=seq_len,
        device=device,
    )


def _target_fwd(
    component_model: LMComponentModel, batch_local: Any, cfg: _ThreePoolRuntime
) -> Tensor:
    """Phase chunk/A2. Detached target forward on this rank's batch slice."""
    with torch.no_grad(), bf16_autocast(cfg.bf16_autocast):
        return component_model(batch_local).detach()


def _faithfulness_phase(
    component_model: LMComponentModel,
    device: torch.device,
    cfg: _ThreePoolRuntime,
    ctx: ChunkContext,
) -> Faith:
    """Phase chunk/D1. Faithfulness loss + backward into V/U .grad (chunk leader only).

    Contribute-once (see ``SUM_GRAD_CONVENTION.md``): faith is computed from the
    replicated V/U weights, so it is identical across a chunk's DP partners. Under
    the chunk SUM-reduce it must land on exactly ONE rank — the chunk leader runs
    the backward; non-leaders skip it (their faith grad would be a duplicate). The
    chunk SUM then spreads the leader's faith grad to every replica exactly once.

    The loss / sum_sq / numel are still computed on every rank (they feed the
    logger's global ratio, not the gradient), so logging is unchanged.
    """
    loss, sum_sq, numel = _faithfulness_loss(component_model, device, cfg.numel_global)
    if ctx.role.is_chunk_leader:
        (cfg.coeff_faith * loss).backward()
    return Faith(loss=loss, sum_sq=sum_sq, numel=numel)


def _wait_ci_and_releaf(
    pending: PendingCiValues,
    ctx: ChunkContext,
    seq_len: int,
    cfg: _ThreePoolRuntime,
) -> CiLeaves:
    """Phase chunk/D2. Wait the CI recv, re-leaf fp32 with grad for the bwd."""
    ci_recv = pending.wait()
    per_site = _releaf_ci_fp32_for_grads(ci_recv, ctx.role.sites)
    _assert_ci_recv_shapes(per_site, ctx, seq_len, cfg)
    return CiLeaves(per_site=per_site)


def _chunkwise_streaming_phase(
    component_model: LMComponentModel,
    batch_local: Any,
    target_local: Tensor,
    ci_leaves: CiLeaves,
    ctx: ChunkContext,
    cfg: _ThreePoolRuntime,
    strategy: ReconLossStrategy,
) -> Stoch:
    """Phase chunk/D3. Stochastic recon over the recon plan, streaming fwd+bwd.

    Generate this chunk's list of recon forwards from ``cfg.recon_plan`` (one
    forward per entry — see ``recon_plan.py``) and run each as a streaming
    masked forward+backward. The default ``PerSitePlan`` produces one forward per
    site (the original layerwise loop); a ``SubsetReconPlan`` produces
    ``n_samples`` joint/subset forwards over all the chunk's sites.

    Each forward's backward seeds the stoch gradient onto BOTH the re-leafed CI
    values (→ CI pool) and the V/U weights (→ chunk). Under the SUM-grad
    convention (see ``SUM_GRAD_CONVENTION.md``) both reductions are SUM, so the
    seed is a single partial sum normalized only by the honest GLOBAL count —
    and the SAME scale serves both destinations (no per-destination split, no
    ``/ n_ci`` to survive a CI-pool AVG).
    """
    sites = ctx.role.sites
    mask_shape = next(iter(ci_leaves.per_site.values())).shape[:-1]
    routings = cfg.recon_plan.generate(sites, mask_shape, target_local.device)
    return _run_routing_forwards(
        component_model=component_model,
        batch_local=batch_local,
        target_local=target_local,
        ci_leaves=ci_leaves.per_site,
        routings=routings,
        weight_deltas_fn=make_weight_deltas_fn(component_model),
        coeff_stoch=cfg.coeff_stoch,
        n_est=cfg.n_est,
        chunk_dp=ctx.world.chunk_dp,
        strategy=strategy,
        bf16_autocast_enabled=cfg.bf16_autocast,
    )


def make_weight_deltas_fn(
    component_model: "LMComponentModel | ComponentModelProtocol",
) -> WeightDeltasFn:
    """A ``WeightDeltasFn`` that recomputes fresh grad-carrying deltas per forward.

    Each call runs ``component_model.calc_weight_deltas()`` and filters to the
    requested sites, so every recon forward gets an independent ``target - VU``
    subgraph (the streaming per-forward backward then frees only its own graph).
    The placement is the model's: the plain ``LMComponentModel`` returns plain
    tensors, the flat FSDP ``FsdpComponentAdapter`` returns DTensor-aware deltas
    (it ``.to_local()``s the ``target - VU`` to a plain tensor while keeping the
    grad on V/U's native ``Shard(0)``). Pass the adapter — not the raw inner
    ``LMComponentModel`` — on the flat path so deltas are plain.
    """

    def fn(sites: tuple[str, ...]) -> dict[str, Tensor]:
        all_deltas = component_model.calc_weight_deltas()
        return {s: all_deltas[s] for s in sites}

    return fn


def _run_routing_forwards(
    component_model: LMComponentModel,
    batch_local: Any,
    target_local: Tensor,
    ci_leaves: dict[str, Tensor],
    routings: list[ForwardRouting],
    weight_deltas_fn: WeightDeltasFn | None,
    coeff_stoch: float,
    n_est: int,
    chunk_dp: int,
    strategy: ReconLossStrategy,
    bf16_autocast_enabled: bool,
) -> Stoch:
    """Run one masked forward+backward per routing; seed CI/V/U grads.

    Context-free (no ``World``/portals) so the gradient-scaling grad check can
    drive it directly. Each forward's backward seeds the stoch gradient onto the
    re-leafed CI values (→ CI pool) and onto V/U (→ chunk). Under the SUM-grad
    convention (see ``SUM_GRAD_CONVENTION.md``) both cross-rank reductions are
    SUM, so the seed is a single partial sum normalized only by the honest GLOBAL
    count — the SAME ``stoch_grad_denom`` serves both destinations.

    Derivation. Single-pool's stoch estimator averages over ``N_est`` recon
    forwards (one per site for the layerwise estimator → ``N_est = n_sites_total``;
    ``n_samples`` for subset recon), backprop'ing ``coeff_stoch * sum_loss /
    n_examples`` with ``n_examples = N_est * P_global`` (``P_global`` = global
    positions), so each ``(forward, position)`` contributes a seed of
    ``coeff_stoch / (N_est * P_global)``.

    In 3-pool each chunk rank covers ``P_local = P_global / chunk_dp`` distinct
    positions. Its seed is the SAME per-(forward, position) value; the CI-pool
    SUM and the chunk SUM then reassemble the full single-pool sum (each global
    position contributes exactly once). The denom is the honest global count
    ``N_est * P_global`` with ``P_global = n_positions * chunk_dp`` — the only
    pool factor that survives is the ``local → global`` position conversion
    ``chunk_dp``, which is part of the global count, not a transport factor.
    The old ``/ n_ci`` (to survive the CI-pool AVG) is gone.

    ``N_est`` is the global total of recon forwards across the whole chunkwise pool
    (``cfg.n_est``); it generalises the old ``n_sites_total`` factor and equals it
    for the default per-site plan (so this is bit-exact with the old path there).
    """
    device = target_local.device
    with bf16_autocast(bf16_autocast_enabled):
        # Accumulate the display value as a GPU tensor (not a Python float) so the
        # per-forward ``.item()`` doesn't force a CPU↔GPU sync that serializes
        # each forward's bwd against the next. ``loss_f.detach()`` so the
        # accumulator doesn't retain the autograd graph.
        stoch_total_t = torch.zeros((), device=device)
        for sites, routing in routings:
            loss_f, n_positions = recon_one_forward(
                component_model,
                batch_local,
                target_local,
                ci_leaves,
                sites,
                routing,
                strategy,
                weight_deltas_fn,
            )
            assert loss_f.dim() == 0, f"recon loss for sites {sites!r} must be scalar"
            n_positions_global = n_positions * chunk_dp
            stoch_grad_denom = n_positions_global * n_est
            (coeff_stoch * loss_f / stoch_grad_denom).backward()
            stoch_total_t = stoch_total_t + (loss_f.detach() / n_positions)
    return Stoch(total=stoch_total_t, n_forwards=len(routings))


def _send_g_ci(portals: ChunkPortals, role: ChunkRole, ci_leaves: CiLeaves) -> None:
    """Phase chunk/D4. Ship per-site CI grads back to the CI pool."""
    g_ci_owned = {s: ci_leaves.per_site[s].grad for s in role.sites}
    assert all(g is not None for g in g_ci_owned.values()), (
        "recon backward should have populated ci_leaves[s].grad"
    )
    portals.g_ci_to_ci_pool.send(role, g_ci_owned)


def _recv_and_combine_g_vu(
    ctx: ChunkContext, component_model: LMComponentModel, *, return_ppgd: bool
) -> dict[str, dict[str, Tensor]] | None:
    """Phases chunk/D5 + chunk/D6. Recv PPGD's V/U grads (chunk leader only), add to
    existing .grad.

    Contribute-once: PPGD's grad is replicated across chunk ranks, so only the
    chunk leader recvs and adds it. The chunk SUM-reduce (chunk/E) then spreads it
    to every replica exactly once (see ``SUM_GRAD_CONVENTION.md``).

    When ``return_ppgd``, returns the leader's PPGD V/U grads as
    ``{site: {"V": ..., "U": ...}}`` (for the per-loss grad-norm breakdown);
    ``None`` on non-leaders or when not requested.
    """
    v_grads_pgd, u_grads_pgd = _recv_g_vu_from_ppgd(ctx, component_model)
    if not ctx.role.is_chunk_leader:
        return None
    _combine_vu_grads_in_place(component_model, ctx.role.sites, v_grads_pgd, u_grads_pgd)
    if not return_ppgd:
        return None
    return {s: {"V": v_grads_pgd[s], "U": u_grads_pgd[s]} for s in ctx.role.sites}


def _snapshot_owned_vu_grads(
    component_model: LMComponentModel, sites: tuple[str, ...]
) -> dict[str, dict[str, Tensor]]:
    """Clone the current owned V/U grads per site (skipping params whose grad is
    still ``None`` — e.g. faith is chunk-leader-only)."""
    out: dict[str, dict[str, Tensor]] = {}
    for s in sites:
        per_param = {
            name: p.grad.detach().clone()
            for name, p in component_model.components[s].named_parameters()
            if p.grad is not None
        }
        if per_param:
            out[s] = per_param
    return out


def _component_grad_sumsq_by_loss(
    component_model: LMComponentModel,
    ctx: ChunkContext,
    faith_vu: dict[str, dict[str, Tensor]] | None,
    ppgd_vu: dict[str, dict[str, Tensor]] | None,
) -> dict[str, float]:
    """Pre-clip GLOBAL grad sum-sq on owned V/U, split by loss term (chunk leader).

    faith + ppgd are contribute-once (chunk-leader-only); stoch is each rank's
    partial. One chunk SUM-all-reduce of ``[faith, ppgd, total]`` recovers each
    term's global grad (``stoch = total - faith - ppgd``); the leader takes sum-sq.
    Returns ``_raw/comp_gradsq/{faith,stoch,ppgd}`` on the chunk leader, ``{}``
    elsewhere. All chunk ranks must call this (it runs a collective).
    """
    params = [
        (s, name, p)
        for s in ctx.role.sites
        for name, p in component_model.components[s].named_parameters()
    ]
    device = params[0][2].device

    def flatten(src: dict[str, dict[str, Tensor]] | None) -> Tensor:
        return torch.cat(
            [
                src[s][name].detach().reshape(-1).float()
                if src is not None and s in src and name in src[s]
                else torch.zeros(p.numel(), device=device)
                for s, name, p in params
            ]
        )

    total = torch.cat(
        [
            p.grad.detach().reshape(-1).float()
            if p.grad is not None
            else torch.zeros(p.numel(), device=device)
            for _, _, p in params
        ]
    )
    stacked = torch.stack([flatten(faith_vu), flatten(ppgd_vu), total])
    dist.all_reduce(stacked, op=dist.ReduceOp.SUM, group=ctx.world.chunk_groups[ctx.role.chunk_idx])
    if not ctx.role.is_chunk_leader:
        return {}
    faith_g, ppgd_g, total_g = stacked[0], stacked[1], stacked[2]
    stoch_g = total_g - faith_g - ppgd_g
    return {
        "_raw/comp_gradsq/faith": faith_g.pow(2).sum().item(),
        "_raw/comp_gradsq/stoch": stoch_g.pow(2).sum().item(),
        "_raw/comp_gradsq/ppgd": ppgd_g.pow(2).sum().item(),
    }


def run_faithfulness_warmup_chunkwise(
    *,
    component_model: LMComponentModel,
    component_params: list[nn.Parameter],
    n_steps: int,
    lr: float,
    weight_decay: float,
    numel_global: int,
) -> None:
    """Single-pool-equivalent faithfulness warmup on the chunkwise pool only.

    CI pool has no V/U; PPGD pool's V/U is a transient replica that gets
    overwritten each step. So warmup only makes sense on the chunkwise pool.
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


def _slice_batch_for_chunkwise(batch: Any, ctx: ChunkContext) -> tuple[Any, int]:
    """Pull this chunk rank's batch slice + extract its seq_len."""
    sl = ctx.role.batch_slice(ctx.world.batch_local_chunk)
    batch_local = batch[sl] if isinstance(batch, Tensor) else batch
    if isinstance(batch_local, Tensor):
        seq_len = batch_local.shape[1] if batch_local.ndim >= 2 else 1
    else:
        assert isinstance(batch_local, dict) and "input_ids" in batch_local
        seq_len = batch_local["input_ids"].shape[1]
    return batch_local, seq_len


def _releaf_ci_fp32_for_grads(
    ci_recv: dict[str, Tensor], sites: tuple[str, ...]
) -> dict[str, Tensor]:
    """Upcast CI (fp16 on the wire — bounded masks) to fp32 and re-leaf with
    ``requires_grad=True`` so the recon backward populates ``leaf.grad`` that the
    CI pool merges into its CI-fn fp32 grads.
    """
    return {s: ci_recv[s].detach().to(torch.float32).clone().requires_grad_(True) for s in sites}


def _assert_ci_recv_shapes(
    ci_recv_leaves: dict[str, Tensor],
    ctx: ChunkContext,
    seq_len: int,
    cfg: _ThreePoolRuntime,
) -> None:
    """Sanity-check the CI leaves match what the CI pool said it'd send.

    Catches a wrong ``c_per_site`` config or a per-rank batch mismatch fast.
    """
    batch_local_chunk = ctx.world.batch_local_chunk
    for s in ctx.role.sites:
        c = cfg.c_per_site[s]
        t = ci_recv_leaves[s]
        assert t.shape == (batch_local_chunk, seq_len, c), (
            f"ci_recv_leaves[{s!r}] shape {tuple(t.shape)} != "
            f"expected ({batch_local_chunk}, {seq_len}, {c})"
        )


def recon_masked_forward(
    component_model: LMComponentModel,
    batch_local: Any,
    ci_recv_leaves: dict[str, Tensor],
    sites: tuple[str, ...],
    routing: RoutingMasks,
    weight_deltas_fn: "WeightDeltasFn | None",
) -> Tensor:
    """The masked suffix forward for one recon forward: draw the per-site masks, run the
    component model, return its prediction (the pre-LM-head hidden state under
    ``strategy.context()``'s bypass, else logits).

    Split out from :func:`recon_one_forward` so the flat FSDP path can checkpoint JUST the
    forward (the activation-heavy part worth recomputing) and run the recon loss OUTSIDE
    the checkpoint — the fused-linear-KL recon is a custom ``autograd.Function`` that saves
    a precomputed gradient, which is incompatible with non-reentrant checkpoint recompute
    (the saved-tensor accounting misaligns). The 2-pool path keeps the loss inside
    :func:`recon_one_forward` (it streams a per-forward backward, no checkpoint).

    ``weight_deltas_fn`` produces a FRESH grad-carrying ``target_weight - VU`` per site for
    THIS forward (``None`` disables the delta component); see :func:`recon_one_forward` for
    why it's a fn (fresh per-forward subgraph) and not a shared dict. The deltas draw no
    RNG, so the per-site ``u``/``delta_mask`` draws stay identical to a single-shared-dict
    path."""
    weight_deltas = weight_deltas_fn(sites) if weight_deltas_fn is not None else None
    component_masks: dict[str, Tensor] = {}
    weight_deltas_and_masks: dict[str, tuple[Tensor, Tensor]] | None = (
        {} if weight_deltas is not None else None
    )
    for site in sites:
        ci_s = ci_recv_leaves[site]
        u = torch.rand_like(ci_s)
        component_masks[site] = ci_s + (1 - ci_s) * u
        if weight_deltas_and_masks is not None:
            assert weight_deltas is not None
            delta_mask = torch.rand(ci_s.shape[:-1], device=ci_s.device, dtype=ci_s.dtype)
            weight_deltas_and_masks[site] = (weight_deltas[site], delta_mask)
    mask_infos = make_mask_infos(
        component_masks,
        weight_deltas_and_masks=weight_deltas_and_masks,
        routing_masks=routing,
    )
    return component_model(batch_local, mask_infos=mask_infos)


def recon_one_forward(
    component_model: LMComponentModel,
    batch_local: Any,
    target_local: Tensor,
    ci_recv_leaves: dict[str, Tensor],
    sites: tuple[str, ...],
    routing: RoutingMasks,
    strategy: ReconLossStrategy,
    weight_deltas_fn: "WeightDeltasFn | None",
) -> tuple[Tensor, int]:
    """Phase chunk/D3 (per-forward body). One stochastic masked forward + recon.

    ``sites`` are the sites swapped in for this forward (the keys of ``mask_infos``);
    ``routing`` gates which positions route to them. The masked forward is
    :func:`recon_masked_forward`; this wraps it with the strategy's recon loss. Returns
    ``(sum_loss, n_positions)`` raw — the caller scales and calls ``backward()`` so the
    per-forward graph is freed between iterations (bounds peak memory).
    """
    pred = recon_masked_forward(
        component_model, batch_local, ci_recv_leaves, sites, routing, weight_deltas_fn
    )
    loss, n_positions = strategy.recon_loss(pred=pred, target=target_local)
    return loss, n_positions


def _recv_g_vu_from_ppgd(
    ctx: ChunkContext,
    component_model: LMComponentModel,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Phase chunk/D5. Recv V/U grads from PPGD pool (chunk leader only).

    Non-leaders get empty dicts — PPGD's grad is replicated, so only the leader
    contributes it (see ``_recv_and_combine_g_vu``).
    """
    sites = ctx.role.sites
    v_templates = {s: component_model.components[s].V for s in sites}
    u_templates = {s: component_model.components[s].U for s in sites}
    v_grads_pgd, u_grads_pgd = ctx.portals.g_vu_from_ppgd.recv(ctx.role, v_templates, u_templates)
    if ctx.role.is_chunk_leader:
        for s in sites:
            assert v_grads_pgd[s].shape == component_model.components[s].V.shape, (
                f"v_grads_pgd[{s!r}] shape mismatch from PPGD send"
            )
            assert u_grads_pgd[s].shape == component_model.components[s].U.shape, (
                f"u_grads_pgd[{s!r}] shape mismatch from PPGD send"
            )
    return v_grads_pgd, u_grads_pgd


def _combine_vu_grads_in_place(
    component_model: LMComponentModel,
    sites: tuple[str, ...],
    v_grads_pgd: dict[str, Tensor],
    u_grads_pgd: dict[str, Tensor],
) -> None:
    """Phase chunk/D6. Add PPGD's V/U grads to .grad (which already has faith+stoch)."""
    for s in sites:
        comp = component_model.components[s]
        assert comp.V.grad is not None and comp.U.grad is not None, (
            "faith + stoch should have populated V/U .grad"
        )
        comp.V.grad.add_(v_grads_pgd[s])
        comp.U.grad.add_(u_grads_pgd[s])


def _sync_tail(
    ctx: ChunkContext,
    component_model: LMComponentModel,
    optimizer: torch.optim.Optimizer,
    all_params: list[nn.Parameter],
    cfg: _ThreePoolRuntime,
    *,
    should_log: bool,
) -> dict[str, float]:
    """Phase chunk/E (sync mode). Blocking all_reduce → clip → AdamW → async send V/U.

    Safe to coexist with PPGD's sync recv at end of step T. Returns this chunk's
    pre-clip component grad norms (``grad_norms/components/<site>.<param>``) on
    log steps, ``{}`` otherwise — captured after the in-chunk SUM-reduce (so each
    norm is the true global per-site grad) and before the clip.
    """
    _wait_pending_weight_send(component_model)
    all_reduce_grads_in_chunk(ctx.world, ctx.role, all_params)
    grad_norms = (
        per_param_grad_norms(
            (f"components/{s}.{name}", p)
            for s in ctx.role.sites
            for name, p in component_model.components[s].named_parameters()
        )
        if should_log
        else {}
    )
    if cfg.grad_clip_norm_components is not None:
        cross_pool_clip_grad_norm(
            all_params,
            cfg.grad_clip_norm_components,
            group=ctx.world.chunkwise_pool_group,
            n_replicas=ctx.world.chunk_dp,
        )
    optimizer.step()
    _async_send_owned_vu_to_ppgd(component_model, ctx)
    return grad_norms


def _async_send_owned_vu_to_ppgd(component_model: LMComponentModel, ctx: ChunkContext) -> None:
    """Kickoff async ship of updated V/U → PPGD. Stash the in-flight send on the model."""
    sites = ctx.role.sites
    v_owned = {s: component_model.components[s].V for s in sites}
    u_owned = {s: component_model.components[s].U for s in sites}
    in_flight = ctx.portals.updated_vu_to_ppgd.send(ctx.role, v_owned, u_owned)
    component_model._pending_weight_send = in_flight  # type: ignore[attr-defined]


def _wait_pending_weight_send(component_model: LMComponentModel) -> None:
    """Wait + clear any pending async V/U send from a previous iter.

    Defense against the opt step mutating V/U while the previous async send
    still reads it.
    """
    pending = getattr(component_model, "_pending_weight_send", None)
    if pending is not None:
        pending.wait()
        component_model._pending_weight_send = None  # type: ignore[attr-defined]


def _faithfulness_loss(
    component_model: LMComponentModel, device: torch.device, numel_global: int
) -> tuple[Tensor, Tensor, int]:
    """‖W_target − VU.T‖²_F / numel_global, summed across this rank's owned sites.

    We divide by ``numel_global`` not ``numel_owned`` to keep the per-element
    grad scale aligned with single-pool's, so the unclipped faithfulness warmup
    converges to the same V/U as single-pool.

    Returns ``(scalar_loss, sum_sq, numel_owned)`` — the ``numel_owned`` is the
    denominator the logger uses for ``SUM(num) / SUM(den)`` global-ratio
    reconstruction across chunks.
    """
    weight_deltas = component_model.calc_weight_deltas()
    sum_sq = torch.zeros((), device=device)
    numel_owned = 0
    for d in weight_deltas.values():
        sum_sq = sum_sq + (d**2).sum()
        numel_owned += d.numel()
    return sum_sq / numel_global, sum_sq, numel_owned
