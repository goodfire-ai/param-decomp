"""Distributed grad check for the 2-pool trainer (SUM-grad convention).

The 2-pool variant merges the 3-pool CI and PPGD pools into a single Pool A
(adversary + CI fn co-located, same batch slice). This test validates that the
fully-reduced gradient on the REPLICATED params — the CI-fn weights (trained on
Pool A) and each chunk's V/U weights (trained on chunkwise) — equals the
single-process full-batch gradient for the identical total loss (faith + stoch +
imp-min + ppgd recon). It reuses the EXACT single-process reference from the
3-pool grad check (the decomposition objective is unchanged), so a pass means the
2-pool gradient assembly is mathematically identical to the 3-pool's.

Topology: non-square ``n_a=4`` Pool A ranks, 2 chunks of ``chunk_dp=2`` — exercises
the CI-fine routing regime on the Pool A ↔ chunk edge and cross-chunk summation.

Run directly:
   CUDA_VISIBLE_DEVICES="" torchrun --standalone --nproc_per_node=8 \
     --master_port=29532 param_decomp_lab/tests/test_two_pool_grad_check_distributed.py
or via pytest (spawns torchrun in a subprocess).
"""

# pyright: reportArgumentType=false, reportOptionalMemberAccess=false
# pyright: reportAttributeAccessIssue=false, reportPrivateUsage=false

import os
import subprocess
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
from torch import Tensor

from param_decomp.metrics.persistent_pgd_state import (
    PersistentPGDState,
    scope_needs_replica_sync,
)
from param_decomp_config.losses import BSCScope, PersistentPGDSourceScope, SCScope
from param_decomp_lab.batch_and_loss_fns import recon_loss_mse
from param_decomp_lab.distributed import cleanup_distributed, init_distributed

# Reuse the 3-pool grad check's model / config / reference verbatim — the
# decomposition objective is identical, so the single-process reference applies.
from param_decomp_lab.tests.test_three_pool_grad_check_distributed import (
    _ALL_SITES,
    _BATCH_GLOBAL,
    _C,
    _PPGD_INIT_SEED,
    _SEQ_LEN,
    _SITES_BLOCK0,
    _SITES_BLOCK1,
    _build_component_model,
    _build_runtime,
    _global_batch,
    _make_ppgd_state,
    _pin_stoch_rng,
    _reference_grads,
)
from param_decomp_lab.three_pool.layout import Chunk
from param_decomp_lab.three_pool.recon_loss_strategy import ReconLossStrategy
from param_decomp_lab.three_pool.runtime import _ThreePoolRuntime
from param_decomp_lab.three_pool.step_chunkwise import step_chunkwise
from param_decomp_lab.three_pool.step_pool_a import step_pool_a
from param_decomp_lab.three_pool.two_pool_context import (
    PoolAContext,
    build_two_pool_context,
)
from param_decomp_lab.three_pool.two_pool_layout import build_two_world

# 2-pool topology: chunks first (rank 0 = chunk-0 leader), then Pool A.
_BLOCK0_RANKS = [0, 1]  # chunk_dp = 2
_BLOCK1_RANKS = [2, 3]
_POOL_A_RANKS = [4, 5, 6, 7]  # n_a = 4 (non-square vs chunk_dp=2 → CI-fine routing)
_WORLD_SIZE = 8


def _build_two_pool_runtime(
    numel_global: int, scope: PersistentPGDSourceScope
) -> _ThreePoolRuntime:
    """The 3-pool runtime with ci_ranks == ppgd_ranks == Pool A ranks and the
    2-pool chunk layout (rank 0 = chunk-0 leader)."""
    base = _build_runtime(numel_global, scope)
    chunks = (
        Chunk(ranks=tuple(_BLOCK0_RANKS), sites=tuple(_SITES_BLOCK0)),
        Chunk(ranks=tuple(_BLOCK1_RANKS), sites=tuple(_SITES_BLOCK1)),
    )
    return _ThreePoolRuntime(
        **{
            **base.__dict__,
            "ci_ranks": tuple(_POOL_A_RANKS),
            "ppgd_ranks": tuple(_POOL_A_RANKS),
            "chunks": chunks,
        }
    )


def _gather_initial_ppgd_sources(
    ppgd_state: PersistentPGDState | None,
    ctx: Any,
    world: Any,
    scope: PersistentPGDSourceScope,
) -> dict[str, Tensor] | None:
    """Reassemble the reference's initial PPGD sources from the distributed ranks.

    - bsc: each Pool A rank owns an independent batch slice; stitch
      the slices into one ``(B, S, source_c)`` source so the reference replays the exact
      per-position trajectory.
    - broadcast: all Pool A ranks broadcast-inited to the SAME ``(1, S, source_c)`` source,
      so the reference's single shared source is just rank 0's copy.

    Collective on every rank (``all_gather_object`` on the world group); returns the
    reassembled sources on rank 0, ``None`` elsewhere.
    """
    payload: dict[str, Any] | None = None
    if isinstance(ctx, PoolAContext):
        assert ppgd_state is not None
        sl = ctx.role.batch_slice(world.batch_local_ppgd)
        start = sl.start if sl.start is not None else 0
        payload = {
            "start": start,
            "sources": {s: ppgd_state.sources[s].detach().clone() for s in ppgd_state.sources},
        }
    gathered: list[Any] = [None] * _WORLD_SIZE
    dist.all_gather_object(gathered, payload)
    if dist.get_rank() != 0:
        return None
    pool_a_items = [item for item in gathered if item is not None]
    match scope:
        case SCScope():
            shared = pool_a_items[0]["sources"]
            for item in pool_a_items[1:]:
                for s, t in item["sources"].items():
                    torch.testing.assert_close(
                        t, shared[s], msg=f"broadcast source diverged across Pool A on site {s!r}"
                    )
            return {s: t.clone() for s, t in shared.items()}
        case BSCScope():
            source_c = _C + 1  # use_delta_component
            full = {s: torch.zeros(_BATCH_GLOBAL, _SEQ_LEN, source_c) for s in _ALL_SITES}
            for item in pool_a_items:
                start = item["start"]
                for s, t in item["sources"].items():
                    full[s][start : start + t.shape[0]] = t
            return full
        case _:
            raise AssertionError(f"unsupported scope in 2-pool grad check: {type(scope).__name__}")


def _neutralize_optimizer(opt: torch.optim.Optimizer) -> None:
    opt.step = lambda *a, **k: None  # type: ignore[method-assign]


def _run_distributed_step(
    scope: PersistentPGDSourceScope,
) -> tuple[dict[str, Tensor], dict[str, Tensor] | None]:
    world = build_two_world(
        pool_a_ranks=list(_POOL_A_RANKS),
        chunks=[
            Chunk(ranks=tuple(_BLOCK0_RANKS), sites=tuple(_SITES_BLOCK0)),
            Chunk(ranks=tuple(_BLOCK1_RANKS), sites=tuple(_SITES_BLOCK1)),
        ],
        batch_global=_BATCH_GLOBAL,
        pg_timeout=timedelta(seconds=120),
        device=None,
    )
    rank = dist.get_rank()
    ctx = build_two_pool_context(world, rank)
    cm = _build_component_model(rank)
    numel_global = sum(cm.target_weight(s).numel() for s in _ALL_SITES)
    runtime = _build_two_pool_runtime(numel_global, scope)
    strategy = ReconLossStrategy.unfused(recon_loss_mse)
    batch = _global_batch()

    ppgd_state: PersistentPGDState | None = None
    if isinstance(ctx, PoolAContext):
        # Distinct per-rank seed so broadcast-init is doing real work (each rank's own
        # randn differs; only the broadcast makes them agree). Per-batch-per-position
        # genuinely wants distinct per-rank sources.
        torch.manual_seed(_PPGD_INIT_SEED + rank)
        replica_sync_group = world.ci_pool_group if scope_needs_replica_sync(scope) else None
        ppgd_state = _make_ppgd_state(world, runtime, strategy, replica_sync_group)
    ref_sources = _gather_initial_ppgd_sources(ppgd_state, ctx, world, scope)

    captured: dict[str, Tensor] = {}
    match ctx:
        case PoolAContext():
            assert ppgd_state is not None
            ci_fn_params = list(cm.ci_fn.parameters())
            opt = torch.optim.AdamW(ci_fn_params, lr=0.0)
            _neutralize_optimizer(opt)
            _pin_stoch_rng()
            step_pool_a(
                ctx,
                cm,
                opt,
                ci_fn_params,
                ppgd_state,
                batch_T=batch,
                cfg=runtime,
                strategy=strategy,
                step=0,
                n_steps=1,
                current_frac_of_training=0.0,
                should_log=False,
            )
            if ctx.role.is_pool_leader:
                for n, p in cm.ci_fn.named_parameters():
                    assert p.grad is not None, f"ci_fn.{n} grad is None"
                    captured[f"ci_fn.{n}"] = p.grad.detach().clone()
        case _:  # ChunkContext
            comp_params = [p for s in ctx.role.sites for p in cm.components[s].parameters()]
            opt = torch.optim.AdamW(comp_params, lr=0.0)
            _neutralize_optimizer(opt)
            _pin_stoch_rng()
            step_chunkwise(ctx, cm, opt, comp_params, batch, runtime, strategy, should_log=False)
            if ctx.role.is_chunk_leader:
                for s in ctx.role.sites:
                    for n, p in cm.components[s].named_parameters():
                        assert p.grad is not None, f"components.{s}.{n} grad is None"
                        captured[f"components.{s}.{n}"] = p.grad.detach().clone()
    return captured, ref_sources


_SCOPES: dict[str, PersistentPGDSourceScope] = {
    "bsc": BSCScope(),
    "sc": SCScope(),
}


def _run_test(scope_name: str) -> None:
    init_distributed()
    try:
        rank = dist.get_rank()
        assert dist.get_world_size() == _WORLD_SIZE
        scope = _SCOPES[scope_name]

        captured, ref_sources = _run_distributed_step(scope)
        gathered: list[Any] = [None] * _WORLD_SIZE
        dist.all_gather_object(gathered, captured)

        if rank == 0:
            dist_grads: dict[str, Tensor] = {}
            for d in gathered:
                assert d is not None
                dist_grads.update(d)
            assert ref_sources is not None
            _report_and_assert(dist_grads, ref_sources, scope, scope_name)
    finally:
        cleanup_distributed()


def _report_and_assert(
    dist_grads: dict[str, Tensor],
    ref_sources: dict[str, Tensor],
    scope: PersistentPGDSourceScope,
    scope_name: str,
) -> None:
    ref_grads = _reference_grads(ref_sources, scope)
    keys = sorted(dist_grads.keys())
    assert set(keys) == set(ref_grads.keys()), (
        f"param-key mismatch:\n dist={sorted(dist_grads)}\n ref ={sorted(ref_grads)}"
    )
    print(f"\n=== 2-pool SUM-grad convention [{scope_name}]: distributed vs reference ===")
    print("topology: n_a=4 (Pool A), chunk_dp=2 (2 chunks) (NON-square)")
    worst = 0.0
    for k in keys:
        dg, rg = dist_grads[k], ref_grads[k]
        assert dg.shape == rg.shape, f"{k}: shape {tuple(dg.shape)} vs {tuple(rg.shape)}"
        denom = rg.abs().max().clamp_min(1e-12)
        rel = (dg - rg).abs().max().item() / denom.item()
        mask = rg.abs() > 1e-8
        ratio = (dg[mask] / rg[mask]).mean().item() if mask.any() else float("nan")
        worst = max(worst, rel)
        print(f"  {k:34s} max|Δ|/max|ref|={rel:.2e}  mean(dist/ref)={ratio:.6f}")
    print(f"worst relative error across all params [{scope_name}]: {worst:.2e}")
    for k in keys:
        torch.testing.assert_close(
            dist_grads[k],
            ref_grads[k],
            rtol=2e-4,
            atol=2e-5,
            msg=lambda m, k=k: f"grad mismatch on {k}:\n{m}",
        )
    print(f"PASS [{scope_name}]: all reduced grads match the single-process reference.")


if __name__ == "__main__":
    import sys

    _run_test(sys.argv[1] if len(sys.argv) > 1 else "bsc")


@pytest.mark.slow
class TestTwoPoolGradCheckDistributed:
    @pytest.mark.parametrize(
        ("scope_name", "master_port"),
        [("bsc", "29532"), ("sc", "29533")],
    )
    def test_grad_check(self, scope_name: str, master_port: str) -> None:
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={_WORLD_SIZE}",
            "--master_port",
            master_port,
            str(Path(__file__).resolve()),
            scope_name,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            raise RuntimeError(f"grad check failed (code {result.returncode})")
        print(result.stdout)
