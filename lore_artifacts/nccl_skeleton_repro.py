"""Faithful comm-skeleton of the 3-pool step (NO model). 8 ranks, real topology
(n_ci=2, chunk_dp=2, n_ppgd=4 → fanout=2 on CI<->PPGD), real transfer sizes.
Adds what the minimal bidir repro lacked: concurrent in-pool collectives + the V/U leg.
Mirrors step_ci / step_chunkwise / step_ppgd ordering. If THIS deadlocks, the tipping
ingredient is isolated (then bisect by removing pieces)."""
import torch
import torch.distributed as dist

dist.init_process_group("nccl")
r, world = dist.get_rank(), dist.get_world_size()
assert world == 8
torch.cuda.set_device(r); dev = torch.device("cuda", r)
fp = torch.float16
CI=[0,1]; CHUNK=[2,3]; PPGD=[4,5,6,7]
ci_g   = dist.new_group(CI); chunk_g = dist.new_group(CHUNK); ppgd_g = dist.new_group(PPGD)
xpool  = dist.new_group(list(range(8)))
# V/U broadcast group: {chunk leader=2} ∪ PPGD
bcast_g = dist.new_group([2,4,5,6,7])
SEQ=2048; C=73728
VAL_CHUNK = 16*SEQ*C    # bl_chunk=16  -> 2.4B elts
VAL_PPGD  = 8*SEQ*C     # bl_ppgd=8    -> 1.2B
VU        = 1_360_000_000  # V/U grads ~1.36B
CIGRAD    = 1_200_000_000  # CI-fn grad all-reduce ~1.2B
def buf(n): return torch.empty(n, dtype=fp, device=dev)
def ones(n): return torch.ones(n, dtype=fp, device=dev)
def spin():
    # heavy, SM-saturating compute (~backward-scale) to contend with in-flight comm
    x = torch.randn(8192,8192,device=dev)
    for _ in range(150): x=(x@x)*1e-3
    torch.cuda.synchronize()
def isend(t,dst): return dist.P2POp(dist.isend,t,dst)
def irecv(t,src): return dist.P2POp(dist.irecv,t,src)

for it in range(4):
    if r in CI:
        ci_slot = CI.index(r)                  # 0 or 1
        my_ppgd = PPGD[ci_slot*2:ci_slot*2+2]  # fanout 2
        my_chunk = CHUNK[ci_slot]              # 1:1
        # phase: async value-sends (chunk + 2 ppgd)
        sv = dist.batch_isend_irecv([isend(ones(VAL_CHUNK),my_chunk),
                                     isend(ones(VAL_PPGD),my_ppgd[0]), isend(ones(VAL_PPGD),my_ppgd[1])])
        # imp_min in-pool collective (CI group)
        dist.all_reduce(torch.ones(16,device=dev), group=ci_g)
        spin()
        # blocking grad-recv (chunk + 2 ppgd)
        rg = dist.batch_isend_irecv([irecv(buf(VAL_CHUNK),my_chunk),
                                     irecv(buf(VAL_PPGD),my_ppgd[0]), irecv(buf(VAL_PPGD),my_ppgd[1])])
        for w in rg: w.wait()
        # big CI-grad all-reduce async (CI group), overlap, then wait sends
        wci = dist.all_reduce(ones(CIGRAD), group=ci_g, async_op=True)
        spin()
        wci.wait()
        for w in sv: w.wait()
    elif r in CHUNK:
        cslot = CHUNK.index(r); my_ci = CI[cslot]; leader = (r==2)
        rv = dist.batch_isend_irecv([irecv(buf(VAL_CHUNK),my_ci)]); [w.wait() for w in rv]
        spin()
        sg = dist.batch_isend_irecv([isend(ones(VAL_CHUNK),my_ci)]); [w.wait() for w in sg]
        if leader:  # recv V/U grads from ppgd leader (4)
            rvu = dist.batch_isend_irecv([irecv(buf(VU),4)]); [w.wait() for w in rvu]
        dist.all_reduce(ones(VU), group=chunk_g)      # in-chunk all-reduce
        b = ones(VU) if leader else buf(VU)
        dist.broadcast(b, src=2, group=bcast_g)        # send updated V/U -> ppgd
    else:  # PPGD
        pslot = PPGD.index(r); my_ci = CI[pslot//2]; leader = (r==4)
        rv = dist.batch_isend_irecv([irecv(buf(VAL_PPGD),my_ci)]); [w.wait() for w in rv]
        spin()
        dist.all_reduce(ones(VU), group=ppgd_g)        # sum-reduce V/U
        sg = dist.batch_isend_irecv([isend(ones(VAL_PPGD),my_ci)]); [w.wait() for w in sg]
        if leader:  # send V/U grads -> chunk leader (2)
            svu = dist.batch_isend_irecv([isend(ones(VU),2)]); [w.wait() for w in svu]
        dist.broadcast(buf(VU), src=2, group=bcast_g)  # recv updated V/U
    torch.cuda.synchronize(); dist.barrier()
    if r==0: print(f"iter {it} OK", flush=True)
if r==0: print("ALL ITERS COMPLETED — no deadlock", flush=True)
dist.destroy_process_group()
