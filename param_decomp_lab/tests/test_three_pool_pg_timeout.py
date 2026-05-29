"""Regression tests for the 3-pool checkpoint-save NCCL-watchdog-timeout fix.

Two independent failure surfaces are covered:

  * ``_resolve_pg_timeout`` reads the ``PD_3POOL_PG_TIMEOUT_S`` override (the
    knob the save-watchdog repro uses to force the bug at small scale) and
    otherwise returns the generous default.

  * ``build_world`` threads its ``pg_timeout`` into *every* ``dist.new_group``
    call. This is the actual production fix: ``new_group`` does NOT inherit the
    timeout passed to ``init_process_group`` — with ``timeout=None`` it falls
    back to the 10-min NCCL default. The 3-pool runs all real collectives on
    these subgroups, so a slow checkpoint save trips that 10-min watchdog
    unless the timeout is set explicitly here.
"""

import datetime

import pytest
import torch.distributed as dist

from param_decomp_lab.three_pool.layout import LayerwiseBlockGroup, build_world
from param_decomp_lab.three_pool.optimize import (
    _DEFAULT_PG_TIMEOUT,
    _rank_invariant_fingerprint_core,
    _resolve_pg_timeout,
)


def test_resolve_pg_timeout_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PD_3POOL_PG_TIMEOUT_S", raising=False)
    assert _resolve_pg_timeout() == _DEFAULT_PG_TIMEOUT
    assert datetime.timedelta(minutes=30) == _DEFAULT_PG_TIMEOUT


def test_resolve_pg_timeout_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PD_3POOL_PG_TIMEOUT_S", "30")
    assert _resolve_pg_timeout() == datetime.timedelta(seconds=30)


def test_build_world_threads_timeout_into_every_new_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every subgroup created by build_world must carry the passed timeout —
    not new_group's 10-min NCCL default. Monkeypatches the world's collective
    primitives so this runs without a real process group or any GPU."""
    captured_timeouts: list[datetime.timedelta | None] = []

    def fake_new_group(ranks: list[int], timeout: datetime.timedelta | None = None) -> object:
        del ranks
        captured_timeouts.append(timeout)
        return object()

    monkeypatch.setattr(dist, "new_group", fake_new_group)
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)

    pg_timeout = datetime.timedelta(minutes=30)
    build_world(
        ci_ranks=[2],
        layerwise_block_groups=[
            LayerwiseBlockGroup(ranks=(0,), owned_sites=("h.0.attn.q_proj",)),
            LayerwiseBlockGroup(ranks=(1,), owned_sites=("h.1.attn.q_proj",)),
        ],
        ppgd_ranks=[3],
        batch_global=4,
        pg_timeout=pg_timeout,
        device=None,
    )

    assert captured_timeouts, "build_world created no process groups"
    assert all(t == pg_timeout for t in captured_timeouts), (
        f"every new_group must get pg_timeout={pg_timeout}; got {captured_timeouts}"
    )


def test_fingerprint_core_old_and_new_formats_agree() -> None:
    """The resume topology check must be rank-invariant AND tolerate the
    pre-fix rank-local fingerprint format that production checkpoints (e.g.
    p-a5b667e9) were written with."""
    old_format_rank0 = {
        "world_size": 144,
        "ci_ranks": list(range(96, 120)),
        "ppgd_ranks": list(range(120, 144)),
        "n_layerwise_blocks": 4,
        "my_rank": 0,
        "my_pool": "layerwise",
        "owned_sites": ["h.0.attn.q_proj", "h.0.attn.k_proj"],
    }
    new_format = {
        "world_size": 144,
        "ci_ranks": list(range(96, 120)),
        "ppgd_ranks": list(range(120, 144)),
        "layerwise_blocks": [
            {"ranks": list(range(24)), "owned_sites": ["h.0.attn.q_proj"]},
            {"ranks": list(range(24, 48)), "owned_sites": ["h.12.attn.q_proj"]},
            {"ranks": list(range(48, 72)), "owned_sites": ["h.24.attn.q_proj"]},
            {"ranks": list(range(72, 96)), "owned_sites": ["h.36.attn.q_proj"]},
        ],
    }
    assert _rank_invariant_fingerprint_core(old_format_rank0) == _rank_invariant_fingerprint_core(
        new_format
    )


def test_fingerprint_core_catches_topology_mismatch() -> None:
    base = {
        "world_size": 8,
        "ci_ranks": [6],
        "ppgd_ranks": [7],
        "layerwise_blocks": [{"ranks": [0, 1], "owned_sites": ["h.0.attn.q_proj"]}],
    }
    changed_ppgd = {**base, "ppgd_ranks": [5]}
    assert _rank_invariant_fingerprint_core(base) != _rank_invariant_fingerprint_core(changed_ppgd)
