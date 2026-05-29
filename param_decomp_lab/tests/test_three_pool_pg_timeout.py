"""Regression tests for the 3-pool checkpoint-save NCCL-watchdog-timeout fix.

Three failure surfaces are covered:

  * ``_resolve_pg_timeout`` reads the ``PD_3POOL_PG_TIMEOUT_S`` override (the
    knob the watchdog-safe-at-low-timeout test uses to force a tight bound) and
    otherwise returns the default.

  * ``build_world`` threads its ``pg_timeout`` into *every* ``dist.new_group``
    call. This is the actual production fix: ``new_group`` does NOT inherit the
    timeout passed to ``init_process_group`` — with ``timeout=None`` it falls
    back to the 10-min NCCL default. The 3-pool runs all real collectives on
    these subgroups, so a slow collective trips that watchdog unless the timeout
    is set explicitly here.

  * The async-consolidation invariant: the union of every rank's
    self-contained partial must exactly cover the full model's V/U + CI-fn
    state-dict keys, so ``assemble_model_state_dict_from_partials`` can rebuild
    the checkpoint off the train loop with no live ranks.

The default PG timeout came down from 30 min to 10 min once consolidation moved
off the train loop: the longest on-loop collective gap is now the fast-eval
pass + a partial-write barrier (minutes), not the old ~10-min rank-0 read of
~100 GB of partials.
"""

import datetime

import pytest
import torch.distributed as dist

from param_decomp_lab.three_pool.checkpoint import (
    ci_fn_state_keys,
    owned_model_state_keys,
)
from param_decomp_lab.three_pool.layout import LayerwiseBlockGroup, build_world
from param_decomp_lab.three_pool.optimize import (
    _DEFAULT_PG_TIMEOUT,
    _rank_invariant_fingerprint_core,
    _resolve_pg_timeout,
)


def test_resolve_pg_timeout_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PD_3POOL_PG_TIMEOUT_S", raising=False)
    assert _resolve_pg_timeout() == _DEFAULT_PG_TIMEOUT
    assert datetime.timedelta(minutes=10) == _DEFAULT_PG_TIMEOUT


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


def test_leader_partials_exactly_cover_model_state() -> None:
    """The per-leader partial slices (LW owned-site V/U + CI fn) must partition
    the full model's fillable keys with no gaps or overlaps — the invariant the
    async consolidation asserts before assembling the checkpoint."""
    sites = ("h.0.attn.q_proj", "h.1.attn.q_proj", "h.2.mlp.c_fc")
    model_keys = {
        "_components.h-0-attn-q_proj.V",
        "_components.h-0-attn-q_proj.U",
        "_components.h-1-attn-q_proj.V",
        "_components.h-1-attn-q_proj.U",
        "_components.h-2-mlp-c_fc.V",
        "_components.h-2-mlp-c_fc.U",
        "ci_fn._global_ci_fn.embed.weight",
        "ci_fn._global_ci_fn.proj.weight",
        # target-model keys (not owned by any leader; come from the fresh buffer)
        "target_model.transformer.wte.weight",
        "target_model.transformer.h.0.attn.c_attn.weight",
    }
    target_keys = {k for k in model_keys if k.startswith("target_model.")}
    fillable = model_keys - target_keys

    # Two LW blocks: block 0 owns sites 0+1, block 1 owns site 2. CI leader owns ci_fn.
    block0 = owned_model_state_keys(model_keys, owned_sites=(sites[0], sites[1]))
    block1 = owned_model_state_keys(model_keys, owned_sites=(sites[2],))
    ci = ci_fn_state_keys(model_keys)

    assert block0.isdisjoint(block1)
    assert block0.isdisjoint(ci)
    assert block1.isdisjoint(ci)
    assert block0 | block1 | ci == fillable
    assert target_keys.isdisjoint(block0 | block1 | ci)
