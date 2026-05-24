"""Lab-side helpers for setting up distributed training with torchrun.

Core `param_decomp.distributed` exposes the read-only state and reduce/gather
primitives that `optimize()` uses. Process-group bring-up and teardown — plus the
rank-0 logger, per-process device pick, and download-once helper — live here because
only the lab experiment runners and pretrain scripts call them.
"""

import os
from collections.abc import Callable
from functools import wraps

import torch
import torch.distributed as dist

from param_decomp.base_config import runtime_cast
from param_decomp.distributed import (
    _SHOULD_GET_INITIALIZED,
    DistributedState,
    is_distributed,
    is_local_main_process,
    sync_across_processes,
)
from param_decomp.log import logger


def init_distributed() -> DistributedState | None:
    """Bring up the torch process group and populate the cached `DistributedState`.

    Reads ``WORLD_SIZE``, ``RANK``, ``LOCAL_RANK``, ``MASTER_ADDR``, and ``MASTER_PORT``
    from the environment (as torchrun sets them) and picks the ``nccl`` backend when
    CUDA is available, else ``gloo``. As a side effect, writes the constructed state
    into ``param_decomp.distributed._state`` so the core read-only accessors return it.

    Returns:
        The `DistributedState` for this process, or ``None`` when distributed should
        not be initialized (see ``_SHOULD_GET_INITIALIZED``).
    """
    # Import inside the function so we can mutate the cached module-level state.
    import param_decomp.distributed as core_dist

    assert core_dist._state is None, "Distributed state already initialized"
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

    core_dist._state = DistributedState(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        backend=backend,
    )

    return core_dist._state


def cleanup_distributed() -> None:
    """Destroy the torch process group and clear the cached `DistributedState`.

    Safe to call when distributed was never initialized.
    """
    import param_decomp.distributed as core_dist

    if is_distributed():
        dist.destroy_process_group()
    core_dist._state = None


def with_distributed_cleanup[**P, T](fn: Callable[P, T]) -> Callable[P, T]:
    """Wrap `fn` so `cleanup_distributed` runs in a ``finally`` block on return/raise."""

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return fn(*args, **kwargs)
        finally:
            cleanup_distributed()

    return wrapper


def log0(msg: str) -> None:
    """Log `msg` at info level on rank 0 only.

    Reads ``RANK`` directly from the environment, so this works before
    `init_distributed` has been called.
    """
    if int(os.environ.get("RANK", 0)) == 0:
        logger.info(msg)


def get_device() -> str:
    """Return the device string for the current process.

    Falls back to ``"cuda"`` or ``"cpu"`` outside distributed. With ``gloo`` returns
    ``"cpu"``; with ``nccl`` returns ``"cuda:{local_rank}"``.
    """
    from param_decomp.distributed import get_distributed_state

    state = get_distributed_state()
    if state is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if state.backend == "gloo":
        return "cpu"
    return f"cuda:{state.local_rank}"


def ensure_cached_and_call[**P, T](fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Run `fn` once per node to populate caches, barrier, then run on every rank.

    In multi-node setups where ``/tmp`` is node-local, this ensures each node downloads
    once (via local-rank 0) rather than having global rank 0 download to a path
    inaccessible to other nodes. Outside distributed, `fn` runs once.

    Args:
        fn: Callable whose side effect is to populate a node-local cache.
        *args: Positional args forwarded to `fn`.
        **kwargs: Keyword args forwarded to `fn`.

    Returns:
        The result of `fn(*args, **kwargs)` on the current rank.
    """
    if is_distributed():
        if is_local_main_process():
            _ = fn(*args, **kwargs)
        sync_across_processes()
        return fn(*args, **kwargs)
    return fn(*args, **kwargs)
