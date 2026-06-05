"""Minimal repro: does batch_isend_irecv hang when the COALESCED group's total
element count crosses 2^31? rank0 fans `fanout` sends of N fp16 elts each to ranks 1..fanout."""
import os

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)
dev = torch.device("cuda", rank)
N = int(float(os.environ["N"]))          # elements per transfer
fanout = world - 1
total = N * fanout
if rank == 0:
    print(f"N={N:,} fanout={fanout} total={total:,}  (2^31={2**31:,})  total>2^31: {total > 2**31}", flush=True)

for it in range(3):
    if rank == 0:
        bufs = [torch.ones(N, dtype=torch.float16, device=dev) for _ in range(fanout)]
        ops = [dist.P2POp(dist.isend, bufs[i], i + 1) for i in range(fanout)]
        for w in dist.batch_isend_irecv(ops):
            w.wait()
    else:
        buf = torch.empty(N, dtype=torch.float16, device=dev)
        for w in dist.batch_isend_irecv([dist.P2POp(dist.irecv, buf, 0)]):
            w.wait()
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print(f"  iter {it} OK", flush=True)
if rank == 0:
    print("ALL ITERS COMPLETED — no hang", flush=True)
dist.destroy_process_group()
