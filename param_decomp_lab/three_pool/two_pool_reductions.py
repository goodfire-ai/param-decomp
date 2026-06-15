"""Cross-pool reductions for 2-pool logging.

Mirrors ``three_pool/reductions.py`` but over two pools. Pool A emits the imp /
ppgd / l0 raw ingredients; the chunkwise pool emits faith / stoch. Rank 0 (the
chunk-0 leader) finalizes the global ``loss/*`` scalars.

Pool A's raw keys are SUM-reduced over the Pool A group (``ci_pool_group``) — see
``three_pool/reductions.py`` for the per-key SUM/AVG rationale (imp and l0 are
pre-divided by ``n_a`` so the SUM yields the global value once; ppgd num/den SUM
straightforwardly). The chunkwise pool's raw keys are scaled by ``1/chunk_dp``
before the SUM (collapsing AVG-within-chunk then SUM-across-chunks into one op).
"""

import math

import torch
import torch.distributed as dist

from param_decomp_lab.three_pool.context import ChunkContext
from param_decomp_lab.three_pool.portals import _batch_p2p
from param_decomp_lab.three_pool.reductions import per_param_grad_norms
from param_decomp_lab.three_pool.two_pool_context import PoolAContext, TwoPoolContext

CHUNK_RAW_KEYS: tuple[str, ...] = (
    "_raw/faith_num",
    "_raw/faith_den",
    "_raw/stoch_num",
    "_raw/stoch_den",
)
POOL_A_RAW_KEYS: tuple[str, ...] = (
    "_raw/imp_num",
    "_raw/l0",
    "_raw/ppgd_num",
    "_raw/ppgd_den",
)

__all__ = [
    "aggregate_losses_to_rank0",
    "aggregate_max_memory_to_rank0",
    "aggregate_grad_norms_to_rank0",
    "aggregate_component_grad_by_loss_to_rank0",
    "per_param_grad_norms",
]


def aggregate_max_memory_to_rank0(
    ctx: TwoPoolContext, device: torch.device
) -> dict[str, float] | None:
    """MAX-reduce CUDA peak memory within each pool; the Pool A leader ships its value
    to rank 0. Returns ``{mem/chunk_peak_gb, mem/pool_a_peak_gb}`` on rank 0."""
    world = ctx.world
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    val = torch.tensor([peak_gb], device=device)
    match ctx:
        case ChunkContext():
            dist.all_reduce(val, op=dist.ReduceOp.MAX, group=world.chunkwise_pool_group)
            if ctx.role.rank == 0:
                chunk_peak = val.item()
                a_val = torch.empty(1, device=device)
                for w in _batch_p2p(
                    world.cross_pool_p2p_group, [(dist.irecv, a_val, world.ci_ranks[0])]
                ):
                    w.wait()
                return {"mem/chunk_peak_gb": chunk_peak, "mem/pool_a_peak_gb": a_val.item()}
            return None
        case PoolAContext():
            dist.all_reduce(val, op=dist.ReduceOp.MAX, group=world.ci_pool_group)
            if ctx.role.is_pool_leader:
                for w in _batch_p2p(world.cross_pool_p2p_group, [(dist.isend, val, 0)]):
                    w.wait()
            return None


def aggregate_losses_to_rank0(
    loss_dict: dict[str, float], ctx: TwoPoolContext, device: torch.device
) -> dict[str, float] | None:
    """SUM raw (num, den) ingredients within each pool, ship Pool A's sums to rank 0,
    and finalize the global ``loss/*`` scalars there."""
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
                a_keys = list(POOL_A_RAW_KEYS)
                a_vals = torch.empty(len(a_keys), device=device, dtype=torch.float64)
                for w in _batch_p2p(
                    world.cross_pool_p2p_group, [(dist.irecv, a_vals, world.ci_ranks[0])]
                ):
                    w.wait()
                a = {k: a_vals[i].item() for i, k in enumerate(a_keys)}
                return {
                    "loss/faith": chunk["_raw/faith_num"] / chunk["_raw/faith_den"],
                    "loss/stoch": chunk["_raw/stoch_num"] / chunk["_raw/stoch_den"],
                    "loss/imp": a["_raw/imp_num"],
                    "loss/ppgd": a["_raw/ppgd_num"] / a["_raw/ppgd_den"],
                    "metrics/mean_l0": a["_raw/l0"],
                }
            return None
        case PoolAContext():
            keys = list(POOL_A_RAW_KEYS)
            vals = torch.tensor([loss_dict[k] for k in keys], device=device, dtype=torch.float64)
            dist.all_reduce(vals, op=dist.ReduceOp.SUM, group=world.ci_pool_group)
            if ctx.role.is_pool_leader:
                for w in _batch_p2p(world.cross_pool_p2p_group, [(dist.isend, vals, 0)]):
                    w.wait()
            return None


def aggregate_grad_norms_to_rank0(
    metrics: dict[str, float], ctx: TwoPoolContext, device: torch.device
) -> dict[str, float] | None:
    """Gather each pool's pre-clip per-parameter grad norms (stashed in ``metrics``
    under ``grad_norms/...``) to rank 0, then add summary norms. Chunkwise all-gathers
    its component norms within-group; the Pool A leader ships its ci-fn norms to rank 0."""
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
                components.update(d)
            ci_buf: list[dict[str, float] | None] = [None]
            dist.recv_object_list(ci_buf, src=world.ci_ranks[0], group=world.cross_pool_p2p_group)
            ci_fns = ci_buf[0]
            assert ci_fns is not None
            return _finalize_grad_norms(components, ci_fns)
        case PoolAContext():
            if ctx.role.is_pool_leader:
                dist.send_object_list([local], dst=0, group=world.cross_pool_p2p_group)
            return None


COMPONENT_GRAD_BY_LOSS_KEYS: tuple[str, ...] = (
    "_raw/comp_gradsq/faith",
    "_raw/comp_gradsq/stoch",
    "_raw/comp_gradsq/ppgd",
)


def aggregate_component_grad_by_loss_to_rank0(
    metrics: dict[str, float], ctx: TwoPoolContext, device: torch.device
) -> dict[str, float] | None:
    """SUM the per-chunk component grad sum-sq-by-loss (chunk leaders, stashed under
    ``_raw/comp_gradsq/*``) across the chunkwise pool → per-loss grad norms on rank 0.
    Chunkwise-internal; Pool A owns no component grads of record and returns ``None``."""
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
        case PoolAContext():
            return None


def _finalize_grad_norms(
    components: dict[str, float], ci_fns: dict[str, float]
) -> dict[str, float]:
    out = {**components, **ci_fns}
    comp_sq = sum(v * v for v in components.values())
    ci_sq = sum(v * v for v in ci_fns.values())
    out["grad_norms/summary/components"] = math.sqrt(comp_sq)
    out["grad_norms/summary/ci_fns"] = math.sqrt(ci_sq)
    out["grad_norms/summary/total"] = math.sqrt(comp_sq + ci_sq)
    return out
