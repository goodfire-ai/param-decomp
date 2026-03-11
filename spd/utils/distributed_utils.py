"""Utilities for distributed data parallel training (torchrun or MPI)."""

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from typing import Any, Literal

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ReduceOp
from torch.types import Number

from spd.configs import Config
from spd.log import logger
from spd.utils.general_utils import runtime_cast


@dataclass(frozen=True, slots=True)
class DistributedState:
    """Immutable snapshot of the distributed runtime state for this process."""

    rank: int
    world_size: int
    local_rank: int
    backend: Literal["nccl", "gloo"]


# Module-level cached state used as a single source of truth
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


def init_distributed() -> DistributedState | None:
    global _state
    assert _state is None, "Distributed state already initialized"
    assert not dist.is_initialized()

    if not _SHOULD_GET_INITIALIZED:
        return None

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    logger.info(f"init_distributed: using {backend=}")

    world_size = int(runtime_cast(str, os.environ.get("WORLD_SIZE")))
    rank = int(runtime_cast(str, os.environ.get("RANK")))
    local_rank = int(runtime_cast(str, os.environ.get("LOCAL_RANK")))
    device = torch.device(f"cuda:{local_rank}")
    logger.info(f"init_distributed: {world_size=}, {rank=}, {local_rank=}, {device=}")

    if backend == "nccl":
        torch.cuda.set_device(device)

    assert (master_addr := os.environ.get("MASTER_ADDR")) is not None
    assert (master_port := os.environ.get("MASTER_PORT")) is not None
    logger.info(f"init_distributed: MASTER_ADDR: {master_addr}, MASTER_PORT: {master_port}")

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=world_size,
        rank=rank,
        device_id=None if backend == "gloo" else device,
    )

    _state = DistributedState(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        backend=backend,
    )

    return _state


def cleanup_distributed() -> None:
    """Clean up distributed process group and reset cached state."""
    global _state
    if is_distributed():
        dist.destroy_process_group()
    _state = None


def with_distributed_cleanup[**P, T](fn: Callable[P, T]) -> Callable[P, T]:
    """Decorator to clean up distributed state after function execution."""

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return fn(*args, **kwargs)
        finally:
            cleanup_distributed()

    return wrapper


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


def print0(*args: Any, **kwargs: Any) -> None:
    """Print only on rank 0 process.

    Works with both torchrun (RANK env var) and init_distributed() setups.
    """
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


def log0(msg: str) -> None:
    """Log only on rank 0 process.

    Works with both torchrun (RANK env var) and init_distributed() setups.
    """
    if int(os.environ.get("RANK", 0)) == 0:
        logger.info(msg)


def get_device() -> str:
    """Get device for current process."""
    state = get_distributed_state()
    if state is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if state.backend == "gloo":
        return "cpu"
    return f"cuda:{state.local_rank}"


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


def ensure_cached_and_call[**P, T](fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Call `fn` on local_rank 0 per node to cache downloads, barrier, then call on all ranks.

    In multi-node setups where /tmp is node-local, this ensures each node downloads once
    (via local_rank 0) rather than having rank 0 download to a path inaccessible to other nodes.
    """
    if is_distributed():
        if is_local_main_process():
            _ = fn(*args, **kwargs)
        sync_across_processes()
        return fn(*args, **kwargs)
    return fn(*args, **kwargs)


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


def get_config_json(config: Config) -> str:
    return f"json:{json.dumps(config.model_dump(mode='json'))}"
