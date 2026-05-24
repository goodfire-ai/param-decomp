"""Cross-pool reductions for logging.

Each pool ``all_reduce`` s within-pool, then the two non-rank-0 pools' leaders
send their averaged tensors to rank 0 via P2P. Convention: rank 0 is always
the Layerwise pool's block 0 leader (validated in
``optimize_three_pool._validate_pd_config_for_three_pool``).

Memory uses MAX (the bottleneck rank is what matters); loss values use AVG.
"""

import torch
import torch.distributed as dist

from param_decomp.three_pool.layout import ThreePoolLayout

# Fixed schemas — every rank in a pool produces the same keys. Hardcoding lets
# us pre-allocate tensors of the right shape for the cross-pool send/recv.
LW_LOSS_KEYS: tuple[str, ...] = ("loss/faith", "loss/stoch")
CI_LOSS_KEYS: tuple[str, ...] = ("loss/imp",)
PPGD_LOSS_KEYS: tuple[str, ...] = ("loss/ppgd",)


def aggregate_max_memory_to_rank0(
    layout: ThreePoolLayout,
    device: torch.device,
) -> dict[str, float] | None:
    """MAX-reduce CUDA peak memory within each pool; non-rank-0 pool leaders
    send their value to rank 0.

    Returns ``{mem/<pool>_peak_gb for pool in (lw, ci, ppgd)}`` on rank 0,
    ``None`` everywhere else.
    """
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    val = torch.tensor([peak_gb], device=device)
    match layout.my_pool:
        case "layerwise":
            dist.all_reduce(val, op=dist.ReduceOp.MAX, group=layout.world.layerwise_pool_group)
            if layout.my_rank == 0:
                lw_peak = val.item()
                ci_val = torch.empty(1, device=device)
                pgd_val = torch.empty(1, device=device)
                dist.recv(ci_val, src=layout.world.ci_ranks[0])
                dist.recv(pgd_val, src=layout.world.ppgd_ranks[0])
                return {
                    "mem/lw_peak_gb": lw_peak,
                    "mem/ci_peak_gb": ci_val.item(),
                    "mem/ppgd_peak_gb": pgd_val.item(),
                }
            return None
        case "ci":
            dist.all_reduce(val, op=dist.ReduceOp.MAX, group=layout.world.ci_pool_group)
            if layout.my_is_pool_leader:
                dist.send(val, dst=0)
            return None
        case "ppgd":
            dist.all_reduce(val, op=dist.ReduceOp.MAX, group=layout.world.ppgd_pool_group)
            if layout.my_is_pool_leader:
                dist.send(val, dst=0)
            return None


def aggregate_losses_to_rank0(
    loss_dict: dict[str, float],
    layout: ThreePoolLayout,
    device: torch.device,
) -> dict[str, float] | None:
    """Average loss values within each pool; non-rank-0 pool leaders send
    their averaged tensors to rank 0.

    Returns the combined dict on rank 0, ``None`` everywhere else.
    """
    match layout.my_pool:
        case "layerwise":
            keys = list(LW_LOSS_KEYS)
            vals = torch.tensor([loss_dict.get(k, 0.0) for k in keys], device=device)
            dist.all_reduce(vals, op=dist.ReduceOp.AVG, group=layout.world.layerwise_pool_group)
            if layout.my_rank == 0:
                lw_losses = {k: vals[i].item() for i, k in enumerate(keys)}
                ci_keys = list(CI_LOSS_KEYS)
                ci_vals = torch.empty(len(ci_keys), device=device)
                dist.recv(ci_vals, src=layout.world.ci_ranks[0])
                ci_losses = {k: ci_vals[i].item() for i, k in enumerate(ci_keys)}
                pgd_keys = list(PPGD_LOSS_KEYS)
                pgd_vals = torch.empty(len(pgd_keys), device=device)
                dist.recv(pgd_vals, src=layout.world.ppgd_ranks[0])
                pgd_losses = {k: pgd_vals[i].item() for i, k in enumerate(pgd_keys)}
                return {**lw_losses, **ci_losses, **pgd_losses}
            return None
        case "ci":
            keys = list(CI_LOSS_KEYS)
            vals = torch.tensor([loss_dict.get(k, 0.0) for k in keys], device=device)
            dist.all_reduce(vals, op=dist.ReduceOp.AVG, group=layout.world.ci_pool_group)
            if layout.my_is_pool_leader:
                dist.send(vals, dst=0)
            return None
        case "ppgd":
            keys = list(PPGD_LOSS_KEYS)
            vals = torch.tensor([loss_dict.get(k, 0.0) for k in keys], device=device)
            dist.all_reduce(vals, op=dist.ReduceOp.AVG, group=layout.world.ppgd_pool_group)
            if layout.my_is_pool_leader:
                dist.send(vals, dst=0)
            return None
