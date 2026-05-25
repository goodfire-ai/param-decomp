"""Utilities for distributed data parallel training (torchrun or MPI).

Process-group bring-up and teardown live in `param_decomp_lab.distributed`
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
    """Immutable snapshot of the distributed runtime state for this process.

    Attributes:
        rank: Global rank of this process within the world.
        world_size: Total number of processes participating in the job.
        local_rank: Rank of this process within its node (one per GPU on multi-GPU nodes).
        backend: The torch.distributed backend in use.
    """

    rank: int
    world_size: int
    local_rank: int
    backend: Literal["nccl", "gloo"]


# Module-level cached state used as a single source of truth.
# Written by `param_decomp_lab.distributed.init_distributed/cleanup_distributed`.
_state: DistributedState | None = None

_SHOULD_GET_INITIALIZED: bool = os.environ.get("WORLD_SIZE") is not None


def get_distributed_state() -> DistributedState | None:
    """Return the cached distributed state for this process, or None when not distributed.

    Whether the process is distributed is determined once at import time from the
    ``WORLD_SIZE`` environment variable. In a distributed setting the state must have been
    initialized by ``param_decomp_lab.distributed`` before this is called; otherwise the
    state must remain unset. Both invariants are asserted.
    """
    if _SHOULD_GET_INITIALIZED:
        assert _state is not None
        return _state
    else:
        assert _state is None
        return None


def is_distributed() -> bool:
    """Return True if this process is participating in a distributed job."""
    state = get_distributed_state()
    return state is not None


def is_main_process() -> bool:
    """Return True on global rank 0 (or always in non-distributed runs)."""
    state = get_distributed_state()
    if state is None:
        return True
    return state.rank == 0


def is_local_main_process() -> bool:
    """Return True on local rank 0 (one process per node in multi-node setups)."""
    state = get_distributed_state()
    if state is None:
        return True
    return state.local_rank == 0


def sync_across_processes() -> None:
    """Block until every rank reaches this point; no-op outside distributed mode."""
    if is_distributed():
        dist.barrier()


def all_reduce(
    tensor: torch.Tensor, op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM
) -> torch.Tensor:
    """All-reduce ``tensor`` across all ranks in place; returns the same tensor unchanged
    in non-distributed mode.

    Args:
        op: Reduction operation applied across ranks.
    """
    if is_distributed():
        dist.all_reduce(tensor, op=op)
    return tensor


def broadcast_tensor(tensor: Tensor) -> Tensor:
    """Broadcast ``tensor`` from rank 0 to every other rank in place."""
    if is_distributed():
        dist.broadcast(tensor, src=0)
    return tensor


def sum_metrics_across_ranks(
    metrics: Mapping[str, Number], device: str | torch.device
) -> Mapping[str, float]:
    """Sum each metric value across all ranks.

    Args:
        metrics: Per-key scalar metrics to reduce. All ranks must pass the same keys.
        device: Device used to stage the reduction tensor.
    """
    assert is_distributed(), "Can only sum metrics across ranks if running in distributed mode"
    metric_values = torch.tensor([metrics[k] for k in metrics], device=device)
    metric_values = all_reduce(metric_values, op=ReduceOp.SUM)
    return {k: metric_values[i].item() for i, k in enumerate(metrics)}


def avg_metrics_across_ranks(
    metrics: Mapping[str, Number], device: str | torch.device
) -> Mapping[str, float]:
    """Average each metric value across all ranks; returns ``metrics`` unchanged when
    running non-distributed.

    Args:
        metrics: Per-key scalar metrics to reduce. All ranks must pass the same keys.
        device: Device used to stage the reduction tensor.
    """
    state = get_distributed_state()
    if state is None:
        return metrics
    world_size = state.world_size
    assert world_size > 0, "World size must be greater than 0"
    sum_metrics = sum_metrics_across_ranks(metrics, device)
    return {k: v / world_size for k, v in sum_metrics.items()}


def gather_all_tensors(tensor: Tensor) -> list[Tensor]:
    """Gather ``tensor`` from every rank into a list indexed by rank.

    Requires identical shapes across ranks. The local rank's entry is replaced with the
    original tensor to preserve autograd through this rank's contribution. In
    non-distributed mode returns ``[tensor]``.
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


def seed_all_ranks(seed: int) -> None:
    """Set the same torch CPU and (if available) CUDA RNG seed on every rank."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def seed_per_rank(base_seed: int) -> None:
    """Seed the global RNG with a per-rank value so stochastic ops diverge across ranks.

    The effective seed is ``base_seed * world_size + rank``, which is unique for every
    ``(base_seed, rank)`` pair. In non-distributed mode this is just ``base_seed``.
    """
    dist_state = get_distributed_state()
    world_size = dist_state.world_size if dist_state is not None else 1
    rank = dist_state.rank if dist_state is not None else 0
    seed = base_seed * world_size + rank
    seed_all_ranks(seed)
