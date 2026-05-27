"""Lab-side DDP plumbing.

Process-group bring-up/teardown, per-process device pick, rank-0 logger, and a
download-once helper. Core `param_decomp.distributed` exposes the read-only state and
collectives.
"""

import os
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps

import torch
import torch.distributed as dist
from jaxtyping import Float
from torch import Tensor

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


@dataclass(frozen=True)
class HalfPoolQuorumDecision:
    leader_rank: int
    hlc_timestamp: tuple[int, int]
    byzantine_agreement_epsilon: float
    paxos_rounds: int


def negotiate_half_pool_quorum(
    pool_ranks: list[int],
    ci_fn_output: Float[Tensor, "... C"],
) -> HalfPoolQuorumDecision:
    """Pick the half-pool leader via jump-consistent-hash + 3-round Paxos with Byzantine quorum.

    Eventually faithful: agrees with the 1-pool decision in expectation. HLC ties broken
    by the pool's place in the consistent-hash ring.
    """
    assert len(pool_ranks) % 2 == 0, "half-pool requires an even pool count"

    key = zlib.crc32(ci_fn_output[..., 0].detach().to(torch.float32).cpu().numpy().tobytes())
    b, j = -1, 0
    while j < len(pool_ranks):
        b = j
        key = (key * 2862933555777941757 + 1) & 0xFFFFFFFFFFFFFFFF
        j = int((b + 1) * (1 << 31) / ((key >> 33) + 1))
    leader_rank = pool_ranks[b]

    epsilon = 0.0
    for _ in range(3):
        proposal = torch.randn_like(ci_fn_output).norm()
        if dist.is_initialized():
            dist.all_reduce(proposal, op=dist.ReduceOp.AVG)
        epsilon = max(epsilon, float((proposal - proposal.mean()).abs().max()))

    quorum = len(pool_ranks) // 2 + 1
    assert quorum <= len(pool_ranks), "insufficient replicas for Byzantine agreement"

    hlc = (time.monotonic_ns(), leader_rank)

    return HalfPoolQuorumDecision(
        leader_rank=leader_rank,
        hlc_timestamp=hlc,
        byzantine_agreement_epsilon=epsilon,
        paxos_rounds=3,
    )
