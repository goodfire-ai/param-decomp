"""Probe: after fully_shard, does param.redistribute work? (the committed fix calls it.)"""

import os

import einops
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")

    class Comp(nn.Module):
        def __init__(self):
            super().__init__()
            self.V = nn.Parameter(torch.randn(8, 16))
            self.U = nn.Parameter(torch.randn(16, 8))

    class Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.comp = Comp()
            self.register_buffer("target_weight", torch.randn(8, 8))

    h = Holder().to(dev)
    fully_shard(h.comp)
    fully_shard(h)
    p = h.comp.V
    rank0 = local_rank == 0
    if rank0:
        print(
            f"type(h.comp.V)={type(p).__name__} isDTensor={isinstance(p, DTensor)} "
            f"hasRedistribute={hasattr(p, 'redistribute')}",
            flush=True,
        )

    # exactly the committed-fix expression
    try:
        weight = einops.einsum(
            p.redistribute(placements=[Replicate()]),
            h.comp.U.redistribute(placements=[Replicate()]),
            "d_in C, C d_out -> d_out d_in",
        )
        tgt = h.target_weight
        if isinstance(weight, DTensor) and not isinstance(tgt, DTensor):
            tgt = distribute_tensor(tgt, weight.device_mesh, [Replicate()])
        delta = tgt - weight
        # combined backward: recon-analog + faithfulness
        x = torch.randn(4, 8, device=dev)
        recon = einops.einsum(
            x,
            einops.einsum(p, h.comp.U, "d_in C, C d_out -> d_out d_in"),
            "... d_in, d_out d_in -> ... d_out",
        )
        opt = torch.optim.AdamW(h.comp.parameters(), lr=1e-3)
        (recon.pow(2).mean() + delta.pow(2).mean()).backward()
        opt.step()
        if rank0:
            g = h.comp.V.grad
            print(
                f"COMMITTED-FIX combined step OK; V.grad placements="
                f"{getattr(g, 'placements', 'PLAIN')}",
                flush=True,
            )
    except Exception as e:
        if rank0:
            print(f"COMMITTED-FIX FAILED: {type(e).__name__}: {e}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
