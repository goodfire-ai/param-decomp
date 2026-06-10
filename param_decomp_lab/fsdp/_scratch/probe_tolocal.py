"""Probe: does .to_local() on the Replicate delta keep V.grad at Shard(0)?

Run: torchrun --standalone --nproc_per_node=2 probe_tolocal.py
"""
import os

import einops
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor


def main():
    lr = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(lr)
    dev = torch.device(f"cuda:{lr}")

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.V = nn.Parameter(torch.randn(8, 16))
            self.U = nn.Parameter(torch.randn(16, 8))
            self.register_buffer("tw", torch.randn(8, 8))

    m = Block().to(dev)
    fully_shard(m)

    def replicate(t):
        return t.redistribute(placements=[Replicate()]) if isinstance(t, DTensor) else t

    weight = einops.einsum(replicate(m.V), replicate(m.U), "d_in C, C d_out -> d_out d_in")
    tw = distribute_tensor(m.tw, weight.device_mesh, [Replicate()])
    delta = tw - weight  # Replicate DTensor

    delta_local = delta.to_local()  # plain tensor?
    loss = (delta_local**2).sum() / delta_local.numel()

    if lr == 0:
        print(f"delta isDTensor={isinstance(delta, DTensor)} "
              f"delta_local isDTensor={isinstance(delta_local, DTensor)} "
              f"loss isDTensor={isinstance(loss, DTensor)}", flush=True)
        # mimic faithfulness accumulator: plain += plain
        acc = torch.zeros((), device=dev)
        try:
            acc += loss.detach()
            print("ACC plain+=loss.detach() OK", flush=True)
        except Exception as e:
            print(f"ACC FAILED: {type(e).__name__}: {e}", flush=True)

    loss.backward()
    if lr == 0:
        g = m.V.grad
        print(f"V.grad placements={getattr(g, 'placements', 'PLAIN')} "
              f"local={tuple(g.to_local().shape) if isinstance(g, DTensor) else g.shape}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
