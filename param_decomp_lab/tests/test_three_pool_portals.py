"""Cross-pool portal round-trip identity — the guard that was missing.

The cross-pool exchanges are hand-built point-to-point: each side independently computes
peers/counts/order, and NCCL p2p has no tags, so a mismatch is *expressible* and fails as a
silent hang (this is what let the seq-2048 fanout deadlock ship). This test exercises each
portal's send→recv on a real (gloo, CPU) multi-rank world and asserts the received tensor equals
a **ground-truth global pattern** — so a wrong pairing/fanout produces wrong values (or a hang
caught by the launcher timeout), not a green CI.

Topology under test is CI-coarse: ``n_ci=2 < n_ppgd=4`` → fanout=2 on the CI↔PPGD edge (the
exact regime that deadlocked), and square on CI↔chunk. 8 ranks: chunk 0-1, CI 2-3, PPGD 4-7.

Run directly: ``CUDA_VISIBLE_DEVICES="" torchrun --standalone --nproc_per_node=8 <thisfile>``.
"""

import os
import subprocess
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from param_decomp_lab.distributed import cleanup_distributed, init_distributed
from param_decomp_lab.three_pool.context import (
    ChunkContext,
    CIContext,
    PPGDContext,
    build_pool_context,
)
from param_decomp_lab.three_pool.layout import Chunk, World, build_world

_WORLD_SIZE = 8
_BATCH_GLOBAL = 8  # bl_ci=4 (n_ci=2), bl_ppgd=2 (n_ppgd=4), bl_chunk=4 (chunk_dp=2)
_SEQ = 3
_SITES = ("layers.0.mlp.gate_proj", "layers.0.mlp.up_proj")
_C = {"layers.0.mlp.gate_proj": 5, "layers.0.mlp.up_proj": 7}


def _world() -> World:
    return build_world(
        ci_ranks=[2, 3],
        chunks=[Chunk(ranks=(0, 1), sites=_SITES)],
        ppgd_ranks=[4, 5, 6, 7],
        batch_global=_BATCH_GLOBAL,
        pg_timeout=timedelta(seconds=60),
        device=None,
    )


def _pattern(global_rows: range, site: str) -> torch.Tensor:
    """Ground-truth value for a contiguous block of global batch rows: every entry of global
    batch row ``b`` is the integer ``b`` (constant over seq + component). Small integers are
    exact in the fp16/bf16 wire dtypes, and a wrong fanout/pairing delivers the wrong global
    rows — i.e. a wrong integer — to a pool, caught at atol < 1. (Batch-slice routing is exactly
    the bug class; per-site pack offsets are covered by the shape/stitch asserts in the portals.)"""
    c = _C[site]
    rows = torch.tensor([float(b) for b in global_rows]).view(-1, 1, 1)
    return rows.expand(len(global_rows), _SEQ, c).contiguous()


def _slice_rows(slice_idx: int, b_local: int) -> range:
    return range(slice_idx * b_local, (slice_idx + 1) * b_local)


def _run() -> None:
    init_distributed()
    try:
        world = _world()
        ctx = build_pool_context(world, dist.get_rank())

        # CI builds its batch-slice of the global pattern (per site, full C).
        ci_full = (
            {s: _pattern(_slice_rows(ctx.role.slice_idx, world.batch_local_ci), s) for s in _SITES}
            if isinstance(ctx, CIContext)
            else {}
        )

        cpu = torch.device("cpu")

        # ── CI → PPGD values (fanout=2) ── (send.wait() blocks for the concurrent recv)
        if isinstance(ctx, CIContext):
            ctx.portals.ci_to_ppgd.send(ctx.role, ci_full).wait()
        if isinstance(ctx, PPGDContext):
            got = ctx.portals.ci_from_ci_pool.post_recv(ctx.role, _C, _SEQ, cpu).wait()
            want = {s: _pattern(_slice_rows(ctx.role.slice_idx, world.batch_local_ppgd), s) for s in _SITES}
            for s in _SITES:
                torch.testing.assert_close(got[s].float(), want[s], rtol=0, atol=0.4,
                                           msg=lambda m, s=s: f"CI→PPGD value mismatch {s}:\n{m}")

        # ── CI → chunk values (square here) ──
        if isinstance(ctx, CIContext):
            ctx.portals.ci_to_chunk.send(ctx.role, ci_full).wait()
        if isinstance(ctx, ChunkContext):
            got = ctx.portals.ci_from_ci_pool.post_recv(ctx.role, _C, _SEQ, cpu).wait()
            want = {s: _pattern(_slice_rows(ctx.role.within_chunk_idx, world.batch_local_chunk), s) for s in _SITES}
            for s in _SITES:
                torch.testing.assert_close(got[s].float(), want[s], rtol=0, atol=0.4,
                                           msg=lambda m, s=s: f"CI→chunk value mismatch {s}:\n{m}")

        # ── PPGD → CI grads (fanout=2, stitched on CI side) ──
        if isinstance(ctx, PPGDContext):
            g = {s: _pattern(_slice_rows(ctx.role.slice_idx, world.batch_local_ppgd), s) for s in _SITES}
            ctx.portals.g_ci_to_ci_pool.send(ctx.role, g)
        if isinstance(ctx, CIContext):
            got = ctx.portals.g_ci_from_ppgd.recv(ctx.role, _C, _SEQ, cpu)
            want = {s: _pattern(_slice_rows(ctx.role.slice_idx, world.batch_local_ci), s) for s in _SITES}
            for s in _SITES:
                torch.testing.assert_close(got[s], want[s], rtol=0, atol=0.4,
                                           msg=lambda m, s=s: f"PPGD→CI grad mismatch {s}:\n{m}")

        if dist.get_rank() == 0:
            print("PASS: cross-pool portal round-trips identity-correct at fanout=2.")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    _run()


@pytest.mark.slow
class TestThreePoolPortals:
    def test_portal_roundtrip_fanout(self) -> None:
        cmd = ["torchrun", "--standalone", f"--nproc_per_node={_WORLD_SIZE}",
               "--master_port", "29537", str(Path(__file__).resolve())]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
            raise RuntimeError(f"portal round-trip failed (code {result.returncode})")
        print(result.stdout)
