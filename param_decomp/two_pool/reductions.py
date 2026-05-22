"""Cross-pool reductions for logging.

Pool A and pool B each ``all_reduce`` within-pool, then pool B's leader sends
to rank 0 (which is always in pool A). Mirrors how ``run_pd.optimize`` uses
``avg_metrics_across_ranks`` — every rank contributes, the global rank 0 ends
up with the full picture.

Memory uses MAX (the bottleneck rank is what matters); loss values use AVG.
"""

import torch
import torch.distributed as dist

from param_decomp.two_pool.layout import BlockDDPLayout

# Fixed schemas — every rank in a pool produces the same keys. Hardcoding lets
# us pre-allocate tensors of the right shape for the cross-pool send/recv.
POOL_A_LOSS_KEYS: tuple[str, ...] = ("loss/faith", "loss/imp", "loss/stoch")
POOL_B_LOSS_KEYS: tuple[str, ...] = ("loss/ppgd",)


def aggregate_max_memory_to_rank0(
    layout: BlockDDPLayout,
    device: torch.device,
) -> dict[str, float] | None:
    """MAX-reduce CUDA peak memory within each pool; ship pool B's to rank 0.

    Returns ``{mem/pool_a_peak_gb, mem/pool_b_peak_gb}`` on rank 0,
    ``None`` everywhere else.
    """
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    val = torch.tensor([peak_gb], device=device)
    if layout.my_pool == "a":
        dist.all_reduce(val, op=dist.ReduceOp.MAX, group=layout.world.pool_a_group)
        if layout.my_rank == 0:
            pool_a_peak = val.item()
            b_val = torch.empty(1, device=device)
            dist.recv(b_val, src=layout.world.pool_b_ranks[0])
            return {
                "mem/pool_a_peak_gb": pool_a_peak,
                "mem/pool_b_peak_gb": b_val.item(),
            }
        return None
    else:
        dist.all_reduce(val, op=dist.ReduceOp.MAX, group=layout.world.pool_b_group)
        if layout.my_is_pool_leader:
            dist.send(val, dst=0)
        return None


def aggregate_losses_to_rank0(
    loss_dict: dict[str, float],
    layout: BlockDDPLayout,
    device: torch.device,
) -> dict[str, float] | None:
    """Average loss values within each pool, then ship pool B's averages to rank 0.

    Layout: pool A = ranks 0..N_pool_a-1 (with rank 0 = global rank 0). Pool B
    leader sends its averaged tensor to global rank 0.
    """
    if layout.my_pool == "a":
        keys = list(POOL_A_LOSS_KEYS)
        vals = torch.tensor([loss_dict.get(k, 0.0) for k in keys], device=device)
        dist.all_reduce(vals, op=dist.ReduceOp.AVG, group=layout.world.pool_a_group)
        if layout.my_rank == 0:
            pool_a_losses = {k: vals[i].item() for i, k in enumerate(keys)}
            # Receive pool B's averaged losses
            b_keys = list(POOL_B_LOSS_KEYS)
            b_vals = torch.empty(len(b_keys), device=device)
            dist.recv(b_vals, src=layout.world.pool_b_ranks[0])
            pool_b_losses = {k: b_vals[i].item() for i, k in enumerate(b_keys)}
            return {**pool_a_losses, **pool_b_losses}
        return None
    else:
        keys = list(POOL_B_LOSS_KEYS)
        vals = torch.tensor([loss_dict.get(k, 0.0) for k in keys], device=device)
        dist.all_reduce(vals, op=dist.ReduceOp.AVG, group=layout.world.pool_b_group)
        if layout.my_is_pool_leader:
            dist.send(vals, dst=0)
        return None
