"""Faithful repro of the 3-pool step's cross-pool pattern, stripped of all model code.
4 ranks: 0=CI, 1=chunk, 2/3=PPGD. Mirrors step_ci ordering: post value-sends ASYNC,
then BLOCK on grad-recv, wait sends only at the very end. Large (rendezvous) transfers.
If this deadlocks, the mechanism is isolated (no model / compile / 3-pool plumbing)."""
import os

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
assert world == 4, "ranks: 0=CI 1=chunk 2,3=PPGD"
torch.cuda.set_device(rank)
dev = torch.device("cuda", rank)
BIG = int(float(os.environ.get("BIG", 1207959552)))    # CI<->chunk elts (2.4GB fp16)
SMALL = int(float(os.environ.get("SMALL", 603979776)))  # CI<->ppgd per-target (1.2GB fp16)
CI, CHUNK, PA, PB = 0, 1, 2, 3
fp = torch.float16

def newbuf(n): return torch.empty(n, dtype=fp, device=dev)
def onesbuf(n): return torch.ones(n, dtype=fp, device=dev)
def spin():  # mimic the compute gap between send and recv phases
    x = torch.randn(4096, 4096, device=dev)
    for _ in range(20): x = (x @ x) * 1e-3
    torch.cuda.synchronize()

for it in range(4):
    if rank == CI:
        # phase: async value-sends (chunk + both ppgd), NOT waited yet
        s_chunk = dist.batch_isend_irecv([dist.P2POp(dist.isend, onesbuf(BIG), CHUNK)])
        s_ppgd = dist.batch_isend_irecv([dist.P2POp(dist.isend, onesbuf(SMALL), PA),
                                         dist.P2POp(dist.isend, onesbuf(SMALL), PB)])
        spin()
        # phase: BLOCKING grad-recv (like step_ci line 156, before waiting the sends)
        rg = dist.batch_isend_irecv([dist.P2POp(dist.irecv, newbuf(BIG), CHUNK),
                                     dist.P2POp(dist.irecv, newbuf(SMALL), PA),
                                     dist.P2POp(dist.irecv, newbuf(SMALL), PB)])
        for w in rg: w.wait()
        for w in (*s_chunk, *s_ppgd): w.wait()  # sends waited last
    elif rank == CHUNK:
        rv = dist.batch_isend_irecv([dist.P2POp(dist.irecv, newbuf(BIG), CI)]); [w.wait() for w in rv]
        spin()
        sg = dist.batch_isend_irecv([dist.P2POp(dist.isend, onesbuf(BIG), CI)]); [w.wait() for w in sg]
    else:  # PA / PB
        rv = dist.batch_isend_irecv([dist.P2POp(dist.irecv, newbuf(SMALL), CI)]); [w.wait() for w in rv]
        spin()
        sg = dist.batch_isend_irecv([dist.P2POp(dist.isend, onesbuf(SMALL), CI)]); [w.wait() for w in sg]
    torch.cuda.synchronize(); dist.barrier()
    if rank == CI: print(f"iter {it} OK", flush=True)
if rank == CI: print("ALL ITERS COMPLETED — no deadlock", flush=True)
dist.destroy_process_group()
