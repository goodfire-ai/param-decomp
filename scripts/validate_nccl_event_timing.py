"""Validate PD_NCCL_EVENT_TIMING (`_time_nccl_op`) measures what it claims.

The docstring claims it separates "peer not ready, CPU spinning" (large cpu_ms,
small gpu_ms) from "real wire transfer" (large gpu_ms). This script imposes a
*known* 3s peer-wait and a *known* fast transfer and reports what the actual
mechanism records, so we can see whether the discriminator holds.

Run on 2 GPUs:
    torchrun --nproc_per_node=2 scripts/validate_nccl_event_timing.py
"""

import os
import time

os.environ["PD_NCCL_EVENT_TIMING"] = "1"  # must be set before import

import torch
import torch.distributed as dist
from param_decomp.three_pool import layout as L

PAYLOAD_NUMEL = 64 * 1024 * 1024  # 256 MB fp32 — big enough to time transfer


def drain() -> list[tuple[str, float, float]]:
    """Same event math as flush_nccl_event_timings, but return the numbers."""
    out = []
    for name, pre, post, cpu_ms in L._NCCL_EVENT_BUFFER:
        post.synchronize()
        out.append((name, cpu_ms, pre.elapsed_time(post)))
    L._NCCL_EVENT_BUFFER.clear()
    return out


def report(rank: int, scenario: str, rows: list[tuple[str, float, float]]) -> None:
    for name, cpu_ms, gpu_ms in rows:
        print(
            f"[rank {rank}] {scenario:28s} {name:24s} cpu={cpu_ms:8.1f}ms  gpu={gpu_ms:8.1f}ms",
            flush=True,
        )


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    assert dist.get_world_size() == 2, "needs exactly 2 ranks"

    big = torch.ones(PAYLOAD_NUMEL, device=dev)

    # Warm up NCCL so first-call init doesn't pollute timings.
    w = torch.ones(8, device=dev)
    dist.all_reduce(w)
    torch.cuda.synchronize()
    drain()
    if rank == 0:
        print(
            f"\nBLOCKING_WAIT={os.environ.get('TORCH_NCCL_BLOCKING_WAIT', 'unset')} "
            f"ASYNC_ERR={os.environ.get('TORCH_NCCL_ASYNC_ERROR_HANDLING', 'unset')}\n",
            flush=True,
        )

    # ── Scenario A: known 3s peer-wait. Sender sleeps; receiver's recv must wait. ──
    dist.barrier()
    torch.cuda.synchronize()
    if rank == 0:
        time.sleep(3.0)
        with L._time_nccl_op("blocking_send"):
            dist.send(big, dst=1)
    else:
        with L._time_nccl_op("blocking_recv_3s_wait"):
            dist.recv(big, src=0)
    report(rank, "A: 3s peer-wait", drain())

    # ── Scenario B: simultaneous — transfer only, no wait. ──
    dist.barrier()
    torch.cuda.synchronize()
    if rank == 0:
        with L._time_nccl_op("blocking_send"):
            dist.send(big, dst=1)
    else:
        with L._time_nccl_op("blocking_recv_no_wait"):
            dist.recv(big, src=0)
    report(rank, "B: transfer only", drain())

    # ── Scenario C: async enqueue only (mimics async_send_ci_to_layerwise). ──
    # The wrapper encloses only the isend/irecv, NOT the .wait(). Does gpu_ms
    # capture anything for a 256MB payload?
    dist.barrier()
    torch.cuda.synchronize()
    if rank == 0:
        time.sleep(3.0)
        with L._time_nccl_op("isend_enqueue_only"):
            work = dist.isend(big, dst=1)
        work.wait()  # outside the timed block, like the real code
    else:
        with L._time_nccl_op("irecv_enqueue_only_3s_wait"):
            work = dist.irecv(big, src=0)
        work.wait()  # outside the timed block
    torch.cuda.synchronize()
    report(rank, "C: async enqueue-only", drain())

    dist.barrier()
    if rank == 0:
        print(
            "\nINTERPRETATION:\n"
            "  A vs B on the receiver: if the 3s wait shows up in gpu_ms (not cpu_ms),\n"
            "    the docstring's discriminator is BACKWARDS for this env.\n"
            "  C: if irecv gpu_ms is ~0 despite a 256MB payload + 3s wait, the\n"
            "    enqueue-only wrappers measure nothing.\n",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
