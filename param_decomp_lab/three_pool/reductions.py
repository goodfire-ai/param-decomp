"""Cross-pool reductions for logging.

Each step function emits per-rank display scalars (``loss/*``) plus raw
additive ingredients (``_raw/*``) that the logger SUM-reduces across each
pool, then finalizes into global scalars on rank 0.

Global scalars:
  ``faith_global = SUM(faith_num) / SUM(faith_den)``    (chunkwise pool)
  ``stoch_global = SUM(stoch_num) / SUM(stoch_den)``    (chunkwise pool)
  ``imp_global   = SUM(imp_num)``                        (CI pool — see note)
  ``ppgd_global  = SUM(ppgd_num)  / SUM(ppgd_den)``     (PPGD pool)
  ``mean_l0      = SUM(l0)``                             (CI pool — see note)

Note on l0: each CI rank divides its per-slice mean L0 by ``n_ci`` before
exposing ``_raw/l0`` so the CI-pool SUM all-reduce yields the batch-mean L0
(total active components per token across all sites, threshold 0).

Note on imp: the CI pool already all-reduces ``per_component_sums`` +
``n_examples`` SUM-wise across the CI pool inside its loss computation, so
every CI rank's ``loss_imp`` scalar IS already the global value. The step
function divides by ``n_ci`` before exposing as ``_raw/imp_num`` so that the
pool-wide all-reduce SUM gives back the global value exactly once.

Chunkwise pool's all-reduce SUM scales every raw value by ``1 / chunk_dp`` *before*
the SUM. That single division collapses two reductions into one and is
mathematically equivalent to AVG-within-chunk then SUM-across-chunks:

  * For values identical across DDP partners in a chunk (faith — the forward
    runs on the FULL batch, so partners produce the same scalar):
    ``sum_over_partners(value / chunk_dp) = value``, and the cross-chunk SUM
    recovers ``sum_over_chunks(value)``.
  * For values that differ across DDP partners (stoch — partners process
    disjoint batch slices): ``sum_over_partners(value / chunk_dp) =
    mean_over_partners(value)``, i.e. the cross-slice mean per site, which is
    what we want before summing across chunks.

Memory uses MAX (the bottleneck rank is what matters).
"""

import math
from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch import nn

from param_decomp_lab.three_pool.context import ChunkContext, CIContext, PoolContext, PPGDContext
from param_decomp_lab.three_pool.portals import _batch_p2p

CHUNK_RAW_KEYS: tuple[str, ...] = (
    "_raw/faith_num",
    "_raw/faith_den",
    "_raw/stoch_num",
    "_raw/stoch_den",
)
CI_RAW_KEYS: tuple[str, ...] = ("_raw/imp_num", "_raw/l0")
PPGD_RAW_KEYS: tuple[str, ...] = ("_raw/ppgd_num", "_raw/ppgd_den")


def aggregate_max_memory_to_rank0(
    ctx: PoolContext,
    device: torch.device,
) -> dict[str, float] | None:
    """MAX-reduce CUDA peak memory within each pool; non-rank-0 pool leaders
    send their value to rank 0.

    Returns ``{mem/<pool>_peak_gb for pool in (chunk, ci, ppgd)}`` on rank 0,
    ``None`` everywhere else.
    """
    world = ctx.world
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    val = torch.tensor([peak_gb], device=device)
    match ctx:
        case ChunkContext():
            dist.all_reduce(val, op=dist.ReduceOp.MAX, group=world.chunkwise_pool_group)
            if ctx.role.rank == 0:
                chunk_peak = val.item()
                ci_val = torch.empty(1, device=device)
                pgd_val = torch.empty(1, device=device)
                for w in _batch_p2p(
                    world.cross_pool_p2p_group,
                    [
                        (dist.irecv, ci_val, world.ci_ranks[0]),
                        (dist.irecv, pgd_val, world.ppgd_ranks[0]),
                    ],
                ):
                    w.wait()
                return {
                    "mem/chunk_peak_gb": chunk_peak,
                    "mem/ci_peak_gb": ci_val.item(),
                    "mem/ppgd_peak_gb": pgd_val.item(),
                }
            return None
        case CIContext():
            dist.all_reduce(val, op=dist.ReduceOp.MAX, group=world.ci_pool_group)
            if ctx.role.is_pool_leader:
                for w in _batch_p2p(world.cross_pool_p2p_group, [(dist.isend, val, 0)]):
                    w.wait()
            return None
        case PPGDContext():
            dist.all_reduce(val, op=dist.ReduceOp.MAX, group=world.ppgd_pool_group)
            if ctx.role.is_pool_leader:
                for w in _batch_p2p(world.cross_pool_p2p_group, [(dist.isend, val, 0)]):
                    w.wait()
            return None


def aggregate_losses_to_rank0(
    loss_dict: dict[str, float],
    ctx: PoolContext,
    device: torch.device,
) -> dict[str, float] | None:
    """SUM raw (num, den) ingredients within each pool, ship CI's and PPGD's
    sums to rank 0, and finalize the global ``loss/*`` scalars there.
    """
    world = ctx.world
    match ctx:
        case ChunkContext():
            keys = list(CHUNK_RAW_KEYS)
            chunk_dp = world.chunk_dp
            vals = torch.tensor(
                [loss_dict[k] / chunk_dp for k in keys], device=device, dtype=torch.float64
            )
            dist.all_reduce(vals, op=dist.ReduceOp.SUM, group=world.chunkwise_pool_group)
            if ctx.role.rank == 0:
                chunk = {k: vals[i].item() for i, k in enumerate(keys)}
                ci_keys = list(CI_RAW_KEYS)
                ci_vals = torch.empty(len(ci_keys), device=device, dtype=torch.float64)
                pgd_keys = list(PPGD_RAW_KEYS)
                pgd_vals = torch.empty(len(pgd_keys), device=device, dtype=torch.float64)
                for w in _batch_p2p(
                    world.cross_pool_p2p_group,
                    [
                        (dist.irecv, ci_vals, world.ci_ranks[0]),
                        (dist.irecv, pgd_vals, world.ppgd_ranks[0]),
                    ],
                ):
                    w.wait()
                ci = {k: ci_vals[i].item() for i, k in enumerate(ci_keys)}
                pgd = {k: pgd_vals[i].item() for i, k in enumerate(pgd_keys)}
                return {
                    "loss/faith": chunk["_raw/faith_num"] / chunk["_raw/faith_den"],
                    "loss/stoch": chunk["_raw/stoch_num"] / chunk["_raw/stoch_den"],
                    "loss/imp": ci["_raw/imp_num"],
                    "loss/ppgd": pgd["_raw/ppgd_num"] / pgd["_raw/ppgd_den"],
                    "metrics/mean_l0": ci["_raw/l0"],
                }
            return None
        case CIContext():
            keys = list(CI_RAW_KEYS)
            vals = torch.tensor([loss_dict[k] for k in keys], device=device, dtype=torch.float64)
            dist.all_reduce(vals, op=dist.ReduceOp.SUM, group=world.ci_pool_group)
            if ctx.role.is_pool_leader:
                for w in _batch_p2p(world.cross_pool_p2p_group, [(dist.isend, vals, 0)]):
                    w.wait()
            return None
        case PPGDContext():
            keys = list(PPGD_RAW_KEYS)
            vals = torch.tensor([loss_dict[k] for k in keys], device=device, dtype=torch.float64)
            dist.all_reduce(vals, op=dist.ReduceOp.SUM, group=world.ppgd_pool_group)
            if ctx.role.is_pool_leader:
                for w in _batch_p2p(world.cross_pool_p2p_group, [(dist.isend, vals, 0)]):
                    w.wait()
            return None


COMPONENT_GRAD_BY_LOSS_KEYS: tuple[str, ...] = (
    "_raw/comp_gradsq/faith",
    "_raw/comp_gradsq/stoch",
    "_raw/comp_gradsq/ppgd",
)


def aggregate_component_grad_by_loss_to_rank0(
    metrics: dict[str, float],
    ctx: PoolContext,
    device: torch.device,
) -> dict[str, float] | None:
    """SUM the per-chunk component grad sum-sq-by-loss (chunk leaders only, stashed
    in ``metrics`` under ``_raw/comp_gradsq/*``) across the chunkwise pool → per-loss grad
    norms on rank 0.

    Chunkwise-pool-internal (CI/PPGD own no component grads and don't participate). Each
    chunk leader contributes its chunk's sum-sq; non-leaders contributed 0. Returns
    ``grad_norms/by_loss/{faith,stoch,ppgd}/components`` (short keys; the trainer
    renames them to metric class names), ``None`` off rank 0 / off the chunkwise pool.
    """
    match ctx:
        case ChunkContext():
            vals = torch.tensor(
                [metrics.get(k, 0.0) for k in COMPONENT_GRAD_BY_LOSS_KEYS],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(vals, op=dist.ReduceOp.SUM, group=ctx.world.chunkwise_pool_group)
            if ctx.role.rank != 0:
                return None
            return {
                "grad_norms/by_loss/faith/components": math.sqrt(vals[0].item()),
                "grad_norms/by_loss/stoch/components": math.sqrt(vals[1].item()),
                "grad_norms/by_loss/ppgd/components": math.sqrt(vals[2].item()),
            }
        case CIContext() | PPGDContext():
            return None


def per_param_grad_norms(named: Iterable[tuple[str, nn.Parameter]]) -> dict[str, float]:
    """Pre-clip L2 grad norm of each param, keyed ``grad_norms/<name>``.

    ``NaN`` for any param whose grad was never populated (mirrors single-pool
    ``component_grad_norms``). Each norm is a CPU↔GPU sync, so call only on log
    steps. ``name`` should already carry the pool's sub-namespace, e.g.
    ``components/<site>.V`` or ``ci_fns/<param>``.
    """
    out: dict[str, float] = {}
    for name, p in named:
        out[f"grad_norms/{name}"] = (
            float("nan") if p.grad is None else p.grad.detach().float().norm().item()
        )
    return out


def aggregate_grad_norms_to_rank0(
    metrics: dict[str, float],
    ctx: PoolContext,
    device: torch.device,
) -> dict[str, float] | None:
    """Gather each pool's pre-clip per-parameter grad norms (stashed in ``metrics``
    under ``grad_norms/...`` by the step fns) to rank 0, then add summary norms.

    Matches single-pool ``component_grad_norms``' key layout:
    ``grad_norms/components/<site>.<param>``, ``grad_norms/ci_fns/<param>``, and
    ``grad_norms/summary/{components,ci_fns,total}``. Params are sharded across
    pools, so the chunkwise pool all-gathers its component norms within-group and the CI
    leader ships its ci-fn norms to rank 0 over the cross-pool p2p group (after
    the loss + memory shipments, matching send/recv order). PPGD owns no trained
    params and contributes nothing. Returns the rank-0 dict, ``None`` elsewhere.

    Each owning rank holds the SUM-reduced (i.e. true global) grad for its params
    by the time the step fn computes these norms, so the per-param values already
    match single-pool — this just collects them. A no-op data path is fine: on
    non-log steps the trainer doesn't call this (it logs only on log steps).
    """
    del device
    local = {k: v for k, v in metrics.items() if k.startswith("grad_norms/")}
    world = ctx.world
    match ctx:
        case ChunkContext():
            n_chunkwise = dist.get_world_size(group=world.chunkwise_pool_group)
            gathered: list[dict[str, float] | None] = [None] * n_chunkwise
            dist.all_gather_object(gathered, local, group=world.chunkwise_pool_group)
            if ctx.role.rank != 0:
                return None
            components: dict[str, float] = {}
            for d in gathered:
                assert d is not None
                # Chunk DP partners hold identical (reduced) grads → identical
                # norms; deduping by key collapses the replicas.
                components.update(d)
            ci_buf: list[dict[str, float] | None] = [None]
            dist.recv_object_list(ci_buf, src=world.ci_ranks[0], group=world.cross_pool_p2p_group)
            ci_fns = ci_buf[0]
            assert ci_fns is not None
            return _finalize_grad_norms(components, ci_fns)
        case CIContext():
            if ctx.role.is_pool_leader:
                dist.send_object_list([local], dst=0, group=world.cross_pool_p2p_group)
            return None
        case PPGDContext():
            return None


def _finalize_grad_norms(
    components: dict[str, float], ci_fns: dict[str, float]
) -> dict[str, float]:
    """Merge the gathered per-param norms and add the three summary norms.

    Summaries are the L2 over each pool's per-param norms (and over both), so a
    missing grad (logged as NaN by the step fn) propagates NaN into the summary —
    matching single-pool ``component_grad_norms``.
    """
    out = {**components, **ci_fns}
    comp_sq = sum(v * v for v in components.values())
    ci_sq = sum(v * v for v in ci_fns.values())
    out["grad_norms/summary/components"] = math.sqrt(comp_sq)
    out["grad_norms/summary/ci_fns"] = math.sqrt(ci_sq)
    out["grad_norms/summary/total"] = math.sqrt(comp_sq + ci_sq)
    return out
