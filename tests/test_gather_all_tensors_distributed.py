"""Distributed tests for gather_all_tensors function.

This file can be run in two ways:

1. Directly with torchrun (fastest):
   torchrun --standalone --nproc_per_node=2 --master_port=29503 tests/test_gather_all_tensors_distributed.py

2. Via pytest (runs torchrun in subprocess):
   pytest tests/test_gather_all_tensors_distributed.py
"""

import os
import subprocess
from pathlib import Path

import pytest
import torch

from param_decomp.utils.distributed_utils import (
    cleanup_distributed,
    gather_all_tensors,
    get_distributed_state,
    init_distributed,
    sync_across_processes,
)


def _test_gather_identical_shapes():
    """Test gathering tensors with identical shapes across ranks."""
    state = get_distributed_state()
    assert state is not None
    rank = state.rank
    world_size = state.world_size

    # Each rank has a different tensor with same shape
    tensor = torch.tensor([rank * 1.0, rank * 2.0])

    # Gather from all ranks
    gathered = gather_all_tensors(tensor)

    # Should have one tensor per rank
    assert len(gathered) == world_size

    # Check shapes are all identical
    for t in gathered:
        assert t.shape == tensor.shape

    # Check values
    for i, t in enumerate(gathered):
        expected = torch.tensor([i * 1.0, i * 2.0])
        torch.testing.assert_close(t, expected)

    # Verify that our rank's entry is the original tensor (preserves autograd)
    assert gathered[rank] is tensor

    if rank == 0:
        print("✓ Gather identical shapes test passed")


def _test_gather_scalar_tensors():
    """Test gathering scalar tensors."""
    state = get_distributed_state()
    assert state is not None
    rank = state.rank
    world_size = state.world_size

    # Scalar tensor with rank-specific value
    tensor = torch.tensor(rank * 10.0)

    # Gather from all ranks
    gathered = gather_all_tensors(tensor)

    # Should have one tensor per rank
    assert len(gathered) == world_size

    # Check values
    for i, t in enumerate(gathered):
        expected = torch.tensor(i * 10.0)
        torch.testing.assert_close(t, expected)

    if rank == 0:
        print("✓ Gather scalar tensors test passed")


def _test_gather_multidimensional_tensors():
    """Test gathering multidimensional tensors."""
    state = get_distributed_state()
    assert state is not None
    rank = state.rank
    world_size = state.world_size

    # 2D tensor with rank-specific values
    tensor = torch.full((3, 4), fill_value=float(rank))

    # Gather from all ranks
    gathered = gather_all_tensors(tensor)

    # Should have one tensor per rank
    assert len(gathered) == world_size

    # Check shapes
    for t in gathered:
        assert t.shape == (3, 4)

    # Check values
    for i, t in enumerate(gathered):
        expected = torch.full((3, 4), fill_value=float(i))
        torch.testing.assert_close(t, expected)

    if rank == 0:
        print("✓ Gather multidimensional tensors test passed")


def _test_gather_empty_tensor():
    """Test gathering empty tensors."""
    state = get_distributed_state()
    assert state is not None
    rank = state.rank

    # Empty tensor with consistent shape
    tensor = torch.tensor([])

    # Gather from all ranks
    gathered = gather_all_tensors(tensor)

    # All gathered tensors should be empty
    for t in gathered:
        assert t.numel() == 0
        assert t.shape == tensor.shape

    if rank == 0:
        print("✓ Gather empty tensor test passed")


def _test_gather_float_tensor():
    """Test gathering tensors."""
    state = get_distributed_state()
    assert state is not None
    rank = state.rank
    world_size = state.world_size

    # Tensor with rank-specific pattern
    tensor = torch.arange(10, dtype=torch.float32) + rank * 10

    gathered = gather_all_tensors(tensor)

    # Should have one tensor per rank
    assert len(gathered) == world_size

    for i, t in enumerate(gathered):
        expected = torch.arange(10, dtype=torch.float32) + i * 10
        torch.testing.assert_close(t, expected)

    if rank == 0:
        print("✓ Gather float tensor test passed")


def _test_gather_preserves_autograd():
    """Test that gathered tensor for current rank preserves autograd."""
    state = get_distributed_state()
    assert state is not None
    rank = state.rank

    # Tensor with gradient tracking
    tensor = torch.tensor([rank * 1.0, rank * 2.0], requires_grad=True)

    # Gather from all ranks
    gathered = gather_all_tensors(tensor)

    # Our rank's tensor should be the original (preserving autograd)
    assert gathered[rank] is tensor
    assert gathered[rank].requires_grad

    if rank == 0:
        print("✓ Gather preserves autograd test passed")


def run_all_tests():
    """Run all distributed tests when called directly with torchrun."""
    # Initialize distributed once for all tests
    init_distributed()
    try:
        state = get_distributed_state()
        assert state is not None
        rank = state.rank
        world_size = state.world_size

        assert world_size == 2, f"Tests require exactly 2 ranks, got {world_size}"

        tests = [
            ("Gather identical shapes", _test_gather_identical_shapes),
            ("Gather scalar tensors", _test_gather_scalar_tensors),
            ("Gather multidimensional tensors", _test_gather_multidimensional_tensors),
            ("Gather empty tensor", _test_gather_empty_tensor),
            ("Gather float tensor", _test_gather_float_tensor),
            ("Gather preserves autograd", _test_gather_preserves_autograd),
        ]

        if rank == 0:
            print(f"\nRunning {len(tests)} gather_all_tensors tests...\n")

        for test_name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                if rank == 0:
                    print(f"✗ {test_name} failed: {e}")
                raise
            # Small barrier to ensure clean test separation
            sync_across_processes()

        if rank == 0:
            print(f"\n✓ All {len(tests)} distributed tests passed!\n")
    finally:
        cleanup_distributed()


# ===== Pytest wrapper =====
# This allows running via pytest, which will spawn torchrun in a subprocess
@pytest.mark.slow
class TestGatherAllTensors:
    """Pytest wrapper for gather_all_tensors tests."""

    def test_gather_all_tensors_distributed(self):
        """Run distributed tests via torchrun in subprocess."""
        script_path = Path(__file__).resolve()

        # ports should be globally unique in tests to allow test parallelization
        # see discussion at: https://github.com/goodfire-ai/param-decomp/pull/186
        cmd = [
            "torchrun",
            "--standalone",
            "--nproc_per_node=2",
            "--master_port",
            "29503",
            str(script_path),
        ]

        # disable cuda so we run on cpu:
        new_env = os.environ.copy()
        new_env["CUDA_VISIBLE_DEVICES"] = ""

        result = subprocess.run(cmd, env=new_env, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            raise RuntimeError(f"Distributed test failed with code {result.returncode}")

        # Print output for visibility (torchrun outputs to stderr)
        print(result.stderr)


if __name__ == "__main__":
    # When run directly with torchrun, execute all tests
    run_all_tests()
