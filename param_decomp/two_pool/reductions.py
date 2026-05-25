"""Cross-pool reductions for logging.

The per-step metrics dict carries two flavors of values:

  * Per-rank display scalars (``loss/faith``, ``loss/imp``, ``loss/stoch``,
    ``loss/ppgd``) — what the rank computed locally, in whatever per-rank
    units the step function chose. Useful for sanity but **not directly
    comparable across topologies** (different ranks own different sites and
    batch slices).

  * Raw additive ingredients prefixed ``_raw/`` (``faith_num``, ``faith_den``,
    ``imp_num``, ``stoch_num``, ``stoch_den``, ``ppgd_num``, ``ppgd_den``).
    These are designed so the global scalar is recoverable from a cross-rank
    SUM:

      - ``faith_global  = SUM(faith_num)  / SUM(faith_den)``
      - ``imp_global    = SUM(imp_num)``                       (no denominator)
      - ``stoch_global  = SUM(stoch_num)  / SUM(stoch_den)``
      - ``ppgd_global   = SUM(ppgd_num)   / SUM(ppgd_den)``

Pool A's aggregator scales every raw value by ``1 / n_per_block`` *before*
the pool-A all-reduce SUM. That single division collapses two reductions
into one and is mathematically equivalent to AVG-within-block then
SUM-across-blocks:

  * For values that are identical across DDP partners in a block (faith and
    imp — the CI fn forward runs on the FULL batch, so partners produce the
    same scalar): ``sum_over_partners(value / n_per_block) = value``, and the
    cross-block SUM recovers ``sum_over_blocks(value)``.
  * For values that differ across DDP partners (stoch — partners process
    disjoint batch slices): ``sum_over_partners(value / n_per_block) =
    mean_over_partners(value)``, i.e. the cross-slice mean per site, which is
    what we want before summing across blocks.

Memory uses MAX (the bottleneck rank is what matters).
"""

import torch
import torch.distributed as dist

from param_decomp.two_pool.layout import BlockDDPLayout

POOL_A_RAW_KEYS: tuple[str, ...] = (
    "_raw/faith_num",
    "_raw/faith_den",
    "_raw/imp_num",
    "_raw/stoch_num",
    "_raw/stoch_den",
)
POOL_B_RAW_KEYS: tuple[str, ...] = (
    "_raw/ppgd_num",
    "_raw/ppgd_den",
)


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
    """SUM raw (num, den) ingredients within each pool, ship pool B's sums to
    rank 0, and finalize to ``loss/<name>`` global scalars there.

    See the module docstring for the math behind the ``1 / n_per_block``
    scale-then-SUM trick used on pool A.
    """
    if layout.my_pool == "a":
        keys = list(POOL_A_RAW_KEYS)
        n_per_block = layout.world.n_per_block
        # Scale before all-reduce SUM: see module docstring.
        vals = torch.tensor(
            [loss_dict[k] / n_per_block for k in keys], device=device, dtype=torch.float64
        )
        dist.all_reduce(vals, op=dist.ReduceOp.SUM, group=layout.world.pool_a_group)
        if layout.my_rank == 0:
            a = {k: vals[i].item() for i, k in enumerate(keys)}
            # Receive pool B's raw sums (already pool-B-wide SUMs).
            b_keys = list(POOL_B_RAW_KEYS)
            b_vals = torch.empty(len(b_keys), device=device, dtype=torch.float64)
            dist.recv(b_vals, src=layout.world.pool_b_ranks[0])
            b = {k: b_vals[i].item() for i, k in enumerate(b_keys)}
            return {
                "loss/faith": a["_raw/faith_num"] / a["_raw/faith_den"],
                "loss/imp": a["_raw/imp_num"],
                "loss/stoch": a["_raw/stoch_num"] / a["_raw/stoch_den"],
                "loss/ppgd": b["_raw/ppgd_num"] / b["_raw/ppgd_den"],
            }
        return None
    else:
        keys = list(POOL_B_RAW_KEYS)
        vals = torch.tensor([loss_dict[k] for k in keys], device=device, dtype=torch.float64)
        dist.all_reduce(vals, op=dist.ReduceOp.SUM, group=layout.world.pool_b_group)
        if layout.my_is_pool_leader:
            dist.send(vals, dst=0)
        return None
