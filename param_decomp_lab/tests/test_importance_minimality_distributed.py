"""Distributed test verifying imp/freq-minimality live update matches eval compute.

The live training loss returned by `<Loss>.update(ctx)` must equal the eval-side scalar
from `compute()` even when per-rank CI distributions differ. Both must use exact global
sums (not local-sums-times-world-size), otherwise the convex log term in
`finalize_freq_min` produces a Jensen upward bias on `update` relative to `compute`
(the imp term is linear, so its update/compute match trivially; freq is the convex one).

This file can be run in two ways:

1. Directly with torchrun (fastest):
   torchrun --standalone --nproc_per_node=2 --master_port=29507 param_decomp_lab/tests/test_importance_minimality_distributed.py

2. Via pytest (runs torchrun in subprocess):
   pytest param_decomp_lab/tests/test_importance_minimality_distributed.py
"""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from param_decomp.distributed import get_distributed_state
from param_decomp.metrics.importance_minimality import (
    FrequencyMinimalityLoss,
    FrequencyMinimalityLossConfig,
    ImportanceMinimalityLoss,
    ImportanceMinimalityLossConfig,
)
from param_decomp_lab.distributed import cleanup_distributed, init_distributed


@dataclass
class _FakeCI:
    upper_leaky: dict[str, torch.Tensor]
    lower_leaky: dict[str, torch.Tensor]
    pre_sigmoid: dict[str, torch.Tensor]


@dataclass
class _FakeCtx:
    ci: _FakeCI
    current_frac_of_training: float


def _make_metric(pnorm: float, eps: float, device: str) -> ImportanceMinimalityLoss:
    cfg = ImportanceMinimalityLossConfig(coeff=1.0, pnorm=pnorm, eps=eps)
    m = ImportanceMinimalityLoss(cfg)
    # Bypass Metric.bind (which wants a real ComponentModel). We only need device set
    # and reset() called to initialise the accumulators.
    m.model = None  # pyright: ignore[reportAttributeAccessIssue]
    m.device = device
    m._bound = True
    m.reset()
    return m


def _make_freq_metric(
    pnorm: float, eps: float, reference_token_count: int, device: str
) -> FrequencyMinimalityLoss:
    cfg = FrequencyMinimalityLossConfig(
        coeff=1.0, pnorm=pnorm, eps=eps, reference_token_count=reference_token_count
    )
    m = FrequencyMinimalityLoss(cfg)
    m.model = None  # pyright: ignore[reportAttributeAccessIssue]
    m.device = device
    m._bound = True
    m.reset()
    return m


def _per_rank_ci(rank: int) -> dict[str, torch.Tensor]:
    """Deliberately non-uniform CI inputs across ranks."""
    # Rank-0 has small values; rank-1+ has progressively larger values. This makes the
    # local sums diverge so the old `* world_size` approximation under Jensen would
    # over-estimate the loss.
    torch.manual_seed(123 + rank)
    base = torch.rand(4, 6, dtype=torch.float32)  # [B, C]
    scale = float(rank + 1) * 0.5
    return {
        "layer1": (base * scale).clone(),
        "layer2": (base * scale * 2.0).clone(),
    }


def _run_test() -> None:
    init_distributed()
    try:
        state = get_distributed_state()
        assert state is not None
        rank = state.rank
        device = "cpu"

        # ---- non-uniform per-rank CI inputs ----
        upper_leaky = _per_rank_ci(rank)
        # update wants `ctx.ci.upper_leaky`; lower_leaky and pre_sigmoid go unused
        # by ImportanceMinimalityLoss.update so we provide empty dicts.
        ci = _FakeCI(upper_leaky=upper_leaky, lower_leaky={}, pre_sigmoid={})
        ctx: Any = _FakeCtx(ci=ci, current_frac_of_training=0.0)

        # ---- run both branches (imp: linear; freq: convex log — the real test) ----
        metric = _make_metric(pnorm=1.0, eps=0.0, device=device)
        live_loss = metric.update(ctx)
        eval_loss = metric.compute()
        assert isinstance(eval_loss, torch.Tensor)

        # Both must be identical scalars: update() now does an autograd-aware all_reduce
        # so its local return value is computed from global sums, exactly like compute().
        if rank == 0:
            print(f"live_loss={live_loss.item():.10f}  eval_loss={eval_loss.item():.10f}")
        assert torch.allclose(live_loss, eval_loss, atol=1e-6), (
            f"rank={rank}: live update() loss {live_loss.item()} != "
            f"eval compute() loss {eval_loss.item()}"
        )

        # Freq carries the convex log2(1 + a'*f) term: a per-rank local-sum*world_size
        # approximation would Jensen-overestimate. update() must match compute() exactly.
        freq_metric = _make_freq_metric(
            pnorm=1.0, eps=0.0, reference_token_count=256, device=device
        )
        freq_live = freq_metric.update(ctx)
        freq_eval = freq_metric.compute()
        assert isinstance(freq_eval, torch.Tensor)
        assert torch.allclose(freq_live, freq_eval, atol=1e-6), (
            f"rank={rank}: live freq update() {freq_live.item()} != compute() {freq_eval.item()}"
        )

        # ---- second update with different per-rank CI; live must still match the
        #      _per-batch_ eval (compute over only this batch) ----
        metric2 = _make_freq_metric(pnorm=1.0, eps=0.0, reference_token_count=256, device=device)
        upper_leaky_2 = {k: v * (rank + 2.0) for k, v in upper_leaky.items()}
        ctx2: Any = _FakeCtx(
            ci=_FakeCI(upper_leaky=upper_leaky_2, lower_leaky={}, pre_sigmoid={}),
            current_frac_of_training=0.0,
        )
        live2 = metric2.update(ctx2)
        eval2 = metric2.compute()
        assert isinstance(eval2, torch.Tensor)
        assert torch.allclose(live2, eval2, atol=1e-6), (
            f"rank={rank}: live2 {live2.item()} != eval2 {eval2.item()}"
        )

        # ---- gradient sanity: backward through live loss reaches each rank's CI ----
        # The autograd-aware dist_fn.all_reduce(SUM) is designed for DDP: its backward
        # all_reduce-SUMs grad_output across ranks. Under DDP that gradient is then
        # gradient-averaged across ranks by the optimizer, so the effective gradient
        # seen by the optimizer matches the analytical d(global_loss)/d(local_ci_r).
        # Here we just verify a non-zero gradient flows to every rank's input.
        metric3 = _make_metric(pnorm=2.0, eps=0.0, device=device)
        upper_leaky_grad = {
            k: v.detach().clone().requires_grad_(True) for k, v in upper_leaky.items()
        }
        ctx3: Any = _FakeCtx(
            ci=_FakeCI(upper_leaky=upper_leaky_grad, lower_leaky={}, pre_sigmoid={}),
            current_frac_of_training=0.0,
        )
        live3 = metric3.update(ctx3)
        live3.backward()
        for name, t in upper_leaky_grad.items():
            assert t.grad is not None, f"rank={rank} layer={name}: no grad flowed"
            assert torch.isfinite(t.grad).all(), f"rank={rank} layer={name}: non-finite grad"
            assert (t.grad != 0).any(), f"rank={rank} layer={name}: all-zero grad"

        if rank == 0:
            print("OK: live update() matches eval compute() under non-uniform per-rank CI")
    finally:
        cleanup_distributed()


@pytest.mark.slow
class TestImportanceMinimalityDistributed:
    def test_update_matches_compute_distributed(self) -> None:
        cmd = [
            "torchrun",
            "--standalone",
            "--nproc_per_node=2",
            "--master_port",
            "29507",
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
