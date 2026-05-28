"""Cross-pool reductions for logging.

Each step function emits per-rank display scalars (``loss/*``) plus raw
additive ingredients (``_raw/*``) that the logger SUM-reduces across each
pool, then finalizes into global scalars on rank 0.

Global scalars:
  ``faith_global = SUM(faith_num) / SUM(faith_den)``    (LW pool)
  ``stoch_global = SUM(stoch_num) / SUM(stoch_den)``    (LW pool)
  ``imp_global   = SUM(imp_num)``                        (CI pool — see note)
  ``ppgd_global  = SUM(ppgd_num)  / SUM(ppgd_den)``     (PPGD pool)

Note on imp: the CI pool already all-reduces ``per_component_sums`` +
``n_examples`` SUM-wise across the CI pool inside its loss computation, so
every CI rank's ``loss_imp`` scalar IS already the global value. The step
function divides by ``n_ci`` before exposing as ``_raw/imp_num`` so that the
pool-wide all-reduce SUM gives back the global value exactly once.

LW pool's all-reduce SUM scales every raw value by ``1 / n_per_block`` *before*
the SUM. That single division collapses two reductions into one and is
mathematically equivalent to AVG-within-block then SUM-across-blocks:

  * For values identical across DDP partners in a block (faith — the forward
    runs on the FULL batch, so partners produce the same scalar):
    ``sum_over_partners(value / n_per_block) = value``, and the cross-block SUM
    recovers ``sum_over_blocks(value)``.
  * For values that differ across DDP partners (stoch — partners process
    disjoint batch slices): ``sum_over_partners(value / n_per_block) =
    mean_over_partners(value)``, i.e. the cross-slice mean per site, which is
    what we want before summing across blocks.

Memory uses MAX (the bottleneck rank is what matters).
"""

import torch
import torch.distributed as dist

from param_decomp_lab.three_pool.context import CIContext, LWContext, PoolContext, PPGDContext

LW_RAW_KEYS: tuple[str, ...] = (
    "_raw/faith_num",
    "_raw/faith_den",
    "_raw/stoch_num",
    "_raw/stoch_den",
)
CI_RAW_KEYS: tuple[str, ...] = ("_raw/imp_num",)
PPGD_RAW_KEYS: tuple[str, ...] = ("_raw/ppgd_num", "_raw/ppgd_den")


def aggregate_max_memory_to_rank0(
    ctx: PoolContext,
    device: torch.device,
) -> dict[str, float] | None:
    """MAX-reduce CUDA peak memory within each pool; non-rank-0 pool leaders
    send their value to rank 0.

    Returns ``{mem/<pool>_peak_gb for pool in (lw, ci, ppgd)}`` on rank 0,
    ``None`` everywhere else.
    """
    world = ctx.world
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    val = torch.tensor([peak_gb], device=device)
    match ctx:
        case LWContext():
            dist.all_reduce(val, op=dist.ReduceOp.MAX, group=world.layerwise_pool_group)
            if ctx.role.rank == 0:
                lw_peak = val.item()
                ci_val = torch.empty(1, device=device)
                pgd_val = torch.empty(1, device=device)
                dist.recv(ci_val, src=world.ci_ranks[0], group=world.cross_pool_p2p_group)
                dist.recv(pgd_val, src=world.ppgd_ranks[0], group=world.cross_pool_p2p_group)
                return {
                    "mem/lw_peak_gb": lw_peak,
                    "mem/ci_peak_gb": ci_val.item(),
                    "mem/ppgd_peak_gb": pgd_val.item(),
                }
            return None
        case CIContext():
            dist.all_reduce(val, op=dist.ReduceOp.MAX, group=world.ci_pool_group)
            if ctx.role.is_pool_leader:
                dist.send(val, dst=0, group=world.cross_pool_p2p_group)
            return None
        case PPGDContext():
            dist.all_reduce(val, op=dist.ReduceOp.MAX, group=world.ppgd_pool_group)
            if ctx.role.is_pool_leader:
                dist.send(val, dst=0, group=world.cross_pool_p2p_group)
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
        case LWContext():
            keys = list(LW_RAW_KEYS)
            n_per_block = world.n_per_block
            vals = torch.tensor(
                [loss_dict[k] / n_per_block for k in keys], device=device, dtype=torch.float64
            )
            dist.all_reduce(vals, op=dist.ReduceOp.SUM, group=world.layerwise_pool_group)
            if ctx.role.rank == 0:
                lw = {k: vals[i].item() for i, k in enumerate(keys)}
                ci_keys = list(CI_RAW_KEYS)
                ci_vals = torch.empty(len(ci_keys), device=device, dtype=torch.float64)
                dist.recv(ci_vals, src=world.ci_ranks[0], group=world.cross_pool_p2p_group)
                ci = {k: ci_vals[i].item() for i, k in enumerate(ci_keys)}
                pgd_keys = list(PPGD_RAW_KEYS)
                pgd_vals = torch.empty(len(pgd_keys), device=device, dtype=torch.float64)
                dist.recv(pgd_vals, src=world.ppgd_ranks[0], group=world.cross_pool_p2p_group)
                pgd = {k: pgd_vals[i].item() for i, k in enumerate(pgd_keys)}
                return {
                    "loss/faith": lw["_raw/faith_num"] / lw["_raw/faith_den"],
                    "loss/stoch": lw["_raw/stoch_num"] / lw["_raw/stoch_den"],
                    "loss/imp": ci["_raw/imp_num"],
                    "loss/ppgd": pgd["_raw/ppgd_num"] / pgd["_raw/ppgd_den"],
                }
            return None
        case CIContext():
            keys = list(CI_RAW_KEYS)
            vals = torch.tensor([loss_dict[k] for k in keys], device=device, dtype=torch.float64)
            dist.all_reduce(vals, op=dist.ReduceOp.SUM, group=world.ci_pool_group)
            if ctx.role.is_pool_leader:
                dist.send(vals, dst=0, group=world.cross_pool_p2p_group)
            return None
        case PPGDContext():
            keys = list(PPGD_RAW_KEYS)
            vals = torch.tensor([loss_dict[k] for k in keys], device=device, dtype=torch.float64)
            dist.all_reduce(vals, op=dist.ReduceOp.SUM, group=world.ppgd_pool_group)
            if ctx.role.is_pool_leader:
                dist.send(vals, dst=0, group=world.cross_pool_p2p_group)
            return None
