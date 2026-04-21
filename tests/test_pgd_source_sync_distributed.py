"""Distributed tests for PGD source synchronization behavior.

Verifies that:
- shared_across_batch: sources are synced across ranks (broadcast init + all-reduced gradients)
- unique_per_datapoint: sources are independent per rank (local init + local gradients)

This file can be run in two ways:

1. Directly with torchrun (fastest):
   torchrun --standalone --nproc_per_node=2 --master_port=29504 tests/test_pgd_source_sync_distributed.py

2. Via pytest (runs torchrun in subprocess):
   pytest tests/test_pgd_source_sync_distributed.py
"""

import os
import subprocess
from pathlib import Path
from typing import override

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from spd.configs import LayerwiseCiConfig, PGDReconLossConfig
from spd.metrics.pgd_utils import pgd_masked_recon_loss_update
from spd.models.component_model import ComponentModel
from spd.routing import AllLayersRouter
from spd.utils.distributed_utils import (
    cleanup_distributed,
    gather_all_tensors,
    get_distributed_state,
    init_distributed,
    sync_across_processes,
)
from spd.utils.module_utils import ModulePathInfo


class _OneLayerLinearModel(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_in, d_out, bias=False)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


def _make_component_model(fc_weight: Tensor) -> ComponentModel:
    d_out, d_in = fc_weight.shape
    target = _OneLayerLinearModel(d_in=d_in, d_out=d_out)
    with torch.no_grad():
        target.fc.weight.copy_(fc_weight)
    target.requires_grad_(False)
    return ComponentModel(
        target_model=target,
        module_path_info=[ModulePathInfo(module_path="fc", C=1)],
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[2]),
        pretrained_model_output_attr=None,
        sigmoid_type="leaky_hard",
    )


def _test_shared_across_batch_sources_synced():
    """With shared_across_batch, PGD sources are broadcast from rank 0 and gradients are
    all-reduced, so with identical data + model the losses must be identical across ranks."""
    state = get_distributed_state()
    assert state is not None
    rank = state.rank

    device = "cpu"

    # Use identical model, data, and CI on all ranks
    torch.manual_seed(42)
    model = _make_component_model(torch.randn(4, 3, dtype=torch.float32)).to(device)

    torch.manual_seed(555)
    batch = torch.randn(2, 3, dtype=torch.float32, device=device)
    with torch.no_grad():
        target_out = model(batch).detach()
    ci = {"fc": torch.full((2, 1), 0.5, dtype=torch.float32, device=device)}

    pgd_config = PGDReconLossConfig(
        init="random", step_size=0.1, n_steps=3, mask_scope="shared_across_batch"
    )
    router = AllLayersRouter()

    torch.manual_seed(7)
    with torch.no_grad():
        sum_loss, _ = pgd_masked_recon_loss_update(
            model=model,
            batch=batch,
            ci=ci,
            weight_deltas=None,
            target_out=target_out,
            output_loss_type="mse",
            router=router,
            pgd_config=pgd_config,
        )

    gathered_losses = gather_all_tensors(sum_loss.unsqueeze(0))

    if rank == 0:
        torch.testing.assert_close(gathered_losses[0], gathered_losses[1])
        print("✓ shared_across_batch source sync test passed")


def _test_unique_per_datapoint_sources_independent():
    """With unique_per_datapoint, PGD sources are initialized independently per rank and gradients
    are not all-reduced, so with different data the full PGD trajectories diverge."""
    state = get_distributed_state()
    assert state is not None
    rank = state.rank

    device = "cpu"

    # Shared model weights across ranks
    torch.manual_seed(42)
    model = _make_component_model(torch.randn(4, 3, dtype=torch.float32)).to(device)

    # Rank-dependent batch data
    torch.manual_seed(1000 + rank)
    batch = torch.randn(2, 3, dtype=torch.float32, device=device)
    with torch.no_grad():
        target_out = model(batch).detach()
    ci = {"fc": torch.rand(2, 1, dtype=torch.float32, device=device)}

    pgd_config = PGDReconLossConfig(
        init="random", step_size=0.1, n_steps=3, mask_scope="unique_per_datapoint"
    )
    router = AllLayersRouter()

    # Rank-dependent seed for PGD random init
    torch.manual_seed(rank * 100)
    with torch.no_grad():
        sum_loss, _ = pgd_masked_recon_loss_update(
            model=model,
            batch=batch,
            ci=ci,
            weight_deltas=None,
            target_out=target_out,
            output_loss_type="mse",
            router=router,
            pgd_config=pgd_config,
        )

    gathered_losses = gather_all_tensors(sum_loss.unsqueeze(0))

    if rank == 0:
        assert not torch.allclose(gathered_losses[0], gathered_losses[1]), (
            f"unique_per_datapoint losses should differ across ranks, "
            f"got {gathered_losses[0].item()} and {gathered_losses[1].item()}"
        )
        print("✓ unique_per_datapoint source independence test passed")


def run_all_tests():
    """Run all distributed tests when called directly with torchrun."""
    init_distributed()
    try:
        state = get_distributed_state()
        assert state is not None
        rank = state.rank
        world_size = state.world_size

        assert world_size == 2, f"Tests require exactly 2 ranks, got {world_size}"

        tests = [
            ("shared_across_batch sources synced", _test_shared_across_batch_sources_synced),
            (
                "unique_per_datapoint sources independent",
                _test_unique_per_datapoint_sources_independent,
            ),
        ]

        if rank == 0:
            print(f"\nRunning {len(tests)} PGD source sync tests...\n")

        for test_name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                if rank == 0:
                    print(f"✗ {test_name} failed: {e}")
                raise
            sync_across_processes()

        if rank == 0:
            print(f"\n✓ All {len(tests)} PGD source sync distributed tests passed!\n")
    finally:
        cleanup_distributed()


# ===== Pytest wrapper =====
@pytest.mark.slow
class TestPGDSourceSync:
    """Pytest wrapper for PGD source sync distributed tests."""

    def test_pgd_source_sync_distributed(self):
        """Run distributed tests via torchrun in subprocess."""
        script_path = Path(__file__).resolve()

        cmd = [
            "torchrun",
            "--standalone",
            "--nproc_per_node=2",
            "--master_port",
            "29504",
            str(script_path),
        ]

        new_env = os.environ.copy()
        new_env["CUDA_VISIBLE_DEVICES"] = ""

        result = subprocess.run(cmd, env=new_env, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            raise RuntimeError(f"Distributed test failed with code {result.returncode}")

        print(result.stderr)


if __name__ == "__main__":
    run_all_tests()
