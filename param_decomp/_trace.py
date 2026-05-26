"""Startup + phase tracing for debugging silent stretches.

The training pipeline has long stretches of pure CPU/CUDA compute that emit no
output for minutes (model build, CI fn build, first-step kernel compile, …).
When the slurm log freezes for 10 min we can't tell a hung job from a slow
one. ``trace(msg)`` is a one-line ``logger.info`` with rank + ms-since-import,
sprinkled at macro boundaries to give a real-time timeline.

By default every rank logs. Set ``PD_TRACE_RANKS=<r1>,<r2>,...`` (e.g.
``0,96,100`` for one rank per pool in a 3-pool job) to restrict.

Phase-level tracing (every ``PhaseProfiler.phase`` enter/exit) is much
noisier so it's opt-in via ``PD_PHASE_TRACE=1``. Combined with
``PD_TRACE_RANKS`` to keep volume sane.
"""

import os
import time

import torch.distributed as dist

from param_decomp.log import logger

_TRACE_START = time.perf_counter()


def _trace_ranks() -> set[int] | None:
    raw = os.environ.get("PD_TRACE_RANKS", "").strip()
    if not raw:
        return None
    return {int(r) for r in raw.split(",") if r.strip()}


def _should_log(rank: int) -> bool:
    allowed = _trace_ranks()
    return allowed is None or rank in allowed


def _current_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    # torchrun sets RANK before our Python entrypoint runs. Use it so traces
    # before init_distributed are still rank-correct (otherwise every rank
    # logs as rank=0 and the PD_TRACE_RANKS filter is useless).
    env_rank = os.environ.get("RANK")
    if env_rank is not None:
        return int(env_rank)
    return 0


def trace(msg: str) -> None:
    """Emit a rank-tagged timeline checkpoint to ``logger``.

    Format: ``[trace rank=R +ELAPSED_MS] MSG`` — easy to grep and tail.
    """
    rank = _current_rank()
    if not _should_log(rank):
        return
    elapsed_ms = (time.perf_counter() - _TRACE_START) * 1000.0
    logger.info(f"[trace rank={rank} +{elapsed_ms:9.1f}ms] {msg}")


def phase_trace_enabled() -> bool:
    """``PhaseProfiler.phase`` should emit per-phase entry traces."""
    return os.environ.get("PD_PHASE_TRACE", "").strip() in ("1", "true", "yes")
