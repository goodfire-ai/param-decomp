"""Distributed test verifying that seed_per_rank makes torch RNG diverge across ranks.

This file can be run in two ways:

1. Directly with torchrun (fastest):
   torchrun --standalone --nproc_per_node=2 --master_port=29505 tests/test_seed_per_rank_distributed.py

2. Via pytest (runs torchrun in subprocess):
   pytest tests/test_seed_per_rank_distributed.py
"""

import os
import subprocess
from pathlib import Path

import pytest
import torch

from spd.utils.distributed_utils import (
    cleanup_distributed,
    gather_all_tensors,
    get_distributed_state,
    init_distributed,
    seed_per_rank,
)


def _run_test():
    """After seed_per_rank, torch.randn produces different values on each rank."""
    init_distributed()
    try:
        state = get_distributed_state()
        assert state is not None
        assert state.world_size == 2, f"Test requires exactly 2 ranks, got {state.world_size}"

        seed_per_rank(42)
        samples = torch.randn(10)
        gathered = gather_all_tensors(samples)

        if state.rank == 0:
            assert not torch.allclose(gathered[0], gathered[1]), (
                "Random samples should differ across ranks after seed_per_rank"
            )
    finally:
        cleanup_distributed()


@pytest.mark.slow
class TestSeedPerRank:
    def test_seed_per_rank_distributed(self):
        cmd = [
            "torchrun",
            "--standalone",
            "--nproc_per_node=2",
            "--master_port",
            "29505",
            str(Path(__file__).resolve()),
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
    _run_test()
