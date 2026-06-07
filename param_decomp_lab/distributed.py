"""Lab-side DDP plumbing.

Process-group bring-up/teardown, per-process device pick, rank-0 logger, and a
download-once helper. Core `param_decomp.distributed` exposes the read-only state and
collectives.
"""

import os
import sys
import traceback
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

    Reads `WORLD_SIZE`, `RANK`, `LOCAL_RANK`, `MASTER_ADDR`, `MASTER_PORT` from the env
    (as torchrun sets them); picks `nccl` if CUDA is available else `gloo`. Writes the
    constructed state into `param_decomp.distributed._state` so the core read-only
    accessors return it. Returns `None` when distributed should not be initialised
    (`_SHOULD_GET_INITIALIZED` is false).
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

    Safe to call when distributed was never initialised.
    """
    import param_decomp.distributed as core_dist

    if is_distributed():
        dist.destroy_process_group()
    core_dist._state = None


def with_distributed_cleanup[**P, T](fn: Callable[P, T]) -> Callable[P, T]:
    """Run `fn`, then tear distributed down — hard-exiting on both distributed paths.

    On a distributed run that returns normally, every rank barriers (so rank 0's
    end-of-run wandb flush lands before any peer exits) and then `os._exit(0)`. The
    hard exit is load-bearing: it skips CPython finalization, where a C-extension
    daemon thread releasing the GIL after the interpreter starts finalizing aborts
    the process with `PyGILState_Release` (SIGABRT). That abort fails the SLURM job,
    and torchrun's peer teardown can kill rank 0 mid-flush so the final step never
    syncs — even though training succeeded.

    On a distributed run that raises, print the traceback and `os._exit(1)` with no
    collective: the failure may be a wedged NCCL comm (e.g. an OOM mid-`all_gather`),
    so `destroy_process_group` would deadlock and leave the rank alive holding its
    GPUs — a zombie SLURM job. The hard exit makes torchrun reap the peers and fail
    fast. Not distributed: `cleanup_distributed` and propagate normally.
    """

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            result = fn(*args, **kwargs)
        except BaseException:
            if not is_distributed():
                cleanup_distributed()
                raise
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        if not is_distributed():
            cleanup_distributed()
            return result
        sync_across_processes()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    return wrapper


def log0(msg: str) -> None:
    """Log `msg` at info level on rank 0 only.

    Reads `RANK` directly from the env, so this works before `init_distributed` has
    been called.
    """
    if int(os.environ.get("RANK", 0)) == 0:
        logger.info(msg)


def get_device() -> str:
    """Device string for the current process.

    Outside distributed, `"cuda"` or `"cpu"`; under `gloo` returns `"cpu"`; under `nccl`
    returns `"cuda:{local_rank}"`.
    """
    from param_decomp.distributed import get_distributed_state

    state = get_distributed_state()
    if state is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if state.backend == "gloo":
        return "cpu"
    return f"cuda:{state.local_rank}"


def ensure_cached_and_call[**P, T](fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Run `fn` once per node (via local-rank 0), barrier, then run on every rank.

    Avoids rank 0 downloading to a path inaccessible to other nodes when `/tmp` is
    node-local. Outside distributed, `fn` runs once.
    """
    if is_distributed():
        if is_local_main_process():
            _ = fn(*args, **kwargs)
        sync_across_processes()
        return fn(*args, **kwargs)
    return fn(*args, **kwargs)
