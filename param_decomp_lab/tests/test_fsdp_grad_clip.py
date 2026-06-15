"""DTensor-aware grad clip: torch 2.11 `clip_grad_norm_` reduces across the mesh."""

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from param_decomp_lab.fsdp.grad_clip import clip_grad_norm_no_sync, clip_grad_norm_with_norm


def _maybe_init_pg() -> bool:
    if dist.is_initialized():
        return False
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29591",
        rank=0,
        world_size=1,
    )
    return True


def test_clip_grad_norm_global_over_sharded_dtensor():
    started = _maybe_init_pg()
    try:
        mesh = init_device_mesh("cpu", (1,))
        full_grad = torch.arange(1, 9, dtype=torch.float32)
        expected_norm = full_grad.norm().item()

        param = torch.nn.Parameter(distribute_tensor(torch.zeros(8), mesh, [Shard(0)]))
        param.grad = distribute_tensor(full_grad.clone(), mesh, [Shard(0)])

        max_norm = 3.0
        returned = clip_grad_norm_with_norm([param], max_norm)
        assert isinstance(returned, DTensor)
        assert abs(float(returned.full_tensor()) - expected_norm) < 1e-4

        assert isinstance(param.grad, DTensor)
        clipped = param.grad.full_tensor()
        assert abs(clipped.norm().item() - max_norm) < 1e-3
    finally:
        if started:
            dist.destroy_process_group()


def test_clip_grad_norm_no_sync_runs():
    started = _maybe_init_pg()
    try:
        mesh = init_device_mesh("cpu", (1,))
        param = torch.nn.Parameter(distribute_tensor(torch.zeros(4), mesh, [Shard(0)]))
        param.grad = distribute_tensor(torch.full((4,), 10.0), mesh, [Shard(0)])
        clip_grad_norm_no_sync([param], max_norm=1.0)
        assert isinstance(param.grad, DTensor)
        assert abs(param.grad.full_tensor().norm().item() - 1.0) < 1e-3
    finally:
        if started:
            dist.destroy_process_group()
