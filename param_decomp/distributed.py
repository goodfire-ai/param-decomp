"""Utilities for distributed data parallel training (torchrun or MPI).

Process-group bring-up and teardown live in `param_decomp_lab.utils.distributed`
because only the lab experiment runners initialize distributed; core only reads
the cached state and runs collectives.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ReduceOp
from torch.types import Number


@dataclass(frozen=True, slots=True)
class DistributedState:
    """Immutable snapshot of the distributed runtime state for this process."""

    rank: int
    world_size: int
    local_rank: int
    backend: Literal["nccl", "gloo"]


# Module-level cached state used as a single source of truth.
# Written by `param_decomp_lab.utils.distributed.init_distributed/cleanup_distributed`.
_state: DistributedState | None = None

_SHOULD_GET_INITIALIZED: bool = os.environ.get("WORLD_SIZE") is not None


def get_distributed_state() -> DistributedState | None:
    """If in a distributed setting, assert that the distributed state is initialized and return the
    cached distributed state. If not initialized, assert that the distributed state is not
    initialized and returns None.

    Returns:
        DistributedState | None: The current process's distributed state snapshot, or None if not in a
        distributed setting.
    """
    if _SHOULD_GET_INITIALIZED:
        assert _state is not None
        return _state
    else:
        assert _state is None
        return None


def is_distributed() -> bool:
    """Check if running in distributed mode using cached state."""
    state = get_distributed_state()
    return state is not None


def is_main_process() -> bool:
    """Check if current process is rank 0."""
    state = get_distributed_state()
    if state is None:
        return True
    return state.rank == 0


def is_local_main_process() -> bool:
    """Check if current process is local_rank 0 (one per node in multi-node setups)."""
    state = get_distributed_state()
    if state is None:
        return True
    return state.local_rank == 0


def sync_across_processes() -> None:
    """Synchronize all processes."""
    if is_distributed():
        dist.barrier()


def all_reduce(
    tensor: torch.Tensor, op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM
) -> torch.Tensor:
    """All-reduce a tensor across all processes.

    Args:
        tensor: Tensor to reduce
        op: Reduction operation (default: SUM)

    Returns:
        Reduced tensor
    """
    if is_distributed():
        dist.all_reduce(tensor, op=op)
    return tensor


def broadcast_tensor(tensor: Tensor) -> Tensor:
    """Broadcast tensor data from rank 0 to all ranks, in-place."""
    if is_distributed():
        dist.broadcast(tensor, src=0)
    return tensor


def sum_metrics_across_ranks(
    metrics: Mapping[str, Number], device: str | torch.device
) -> Mapping[str, float]:
    assert is_distributed(), "Can only sum metrics across ranks if running in distributed mode"
    metric_values = torch.tensor([metrics[k] for k in metrics], device=device)
    metric_values = all_reduce(metric_values, op=ReduceOp.SUM)
    return {k: metric_values[i].item() for i, k in enumerate(metrics)}


def avg_metrics_across_ranks(
    metrics: Mapping[str, Number], device: str | torch.device
) -> Mapping[str, float]:
    state = get_distributed_state()
    if state is None:
        return metrics
    world_size = state.world_size
    assert world_size > 0, "World size must be greater than 0"
    sum_metrics = sum_metrics_across_ranks(metrics, device)
    return {k: v / world_size for k, v in sum_metrics.items()}


def gather_all_tensors(tensor: Tensor) -> list[Tensor]:
    """Gather tensors from all distributed processes.

    Requires all tensors to have identical shapes across all ranks.

    Args:
        tensor: The tensor to gather from all ranks
        group: The process group (defaults to WORLD)

    Returns:
        List of tensors from all ranks (including local rank)
    """
    state = get_distributed_state()
    if state is None:
        return [tensor]

    tensor = tensor.contiguous()

    gathered = [torch.zeros_like(tensor) for _ in range(state.world_size)]
    torch.distributed.all_gather(gathered, tensor)

    # Replace our rank's entry with the original to preserve autograd
    gathered[state.rank] = tensor

    return gathered


def seed_per_rank(base_seed: int) -> None:
    """Set global RNG to a unique seed per rank, so stochastic operations diverge across DDP ranks.

    Uses base_seed * world_size + rank to guarantee no collisions across any (base_seed, rank) pair.
    In non-distributed mode, just sets seed to base_seed.
    """
    dist_state = get_distributed_state()
    world_size = dist_state.world_size if dist_state is not None else 1
    rank = dist_state.rank if dist_state is not None else 0
    seed = base_seed * world_size + rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
