"""Fine-tune init from a parent checkpoint (SPEC S33).

`init_from_parent` loads the parent's trained V/U + ci_fn onto a fresh reference state
and keeps the fresh optimizer / sources and `step = 0`; the config-level
`assert_finetune_structural_compat` guard fires before the orbax restore when the parent's
decomposition structure (sites / C / ci-fn arch) doesn't match the new config's.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
import yaml

from jax_single_pool.checkpoint import init_from_parent, make_checkpoint_manager, save_state
from jax_single_pool.config import build_from_schema
from jax_single_pool.run import assert_finetune_structural_compat
from jax_single_pool.tests.test_checkpoint import _build
from param_decomp_config.experiment import ResumeProvenance

CONFIGS = Path(__file__).parent.parent / "configs"


def test_init_from_parent_loads_components_resets_schedule(tmp_path: Path):
    # A parent run: train a couple of steps so its V/U + ci_fn + sources + step are
    # non-trivial, then checkpoint.
    tgt, parent_state, step, resid = _build(seed=1)
    for i in range(2):
        parent_state, _ = step(parent_state, tgt, resid, jax.random.PRNGKey(i))
    assert int(parent_state.step) == 2

    parent_ckpt_dir = tmp_path / "parent" / "ckpts"
    mgr = make_checkpoint_manager(parent_ckpt_dir, keep_last=2)
    save_state(mgr, 2, parent_state)

    # A fresh fine-tune reference (DIFFERENT seed): every leaf is independently initialized,
    # so a carried-over leaf must have come from the parent, an un-carried one must not.
    _, fresh, _, _ = _build(seed=7)
    finetuned = init_from_parent(parent_ckpt_dir, parent_step=2, reference=fresh)

    # components + ci_fn come from the parent.
    for a, b in zip(
        jax.tree.leaves(finetuned.components), jax.tree.leaves(parent_state.components), strict=True
    ):
        assert jnp.array_equal(a, b)
    for a, b in zip(
        jax.tree.leaves(finetuned.ci_fn), jax.tree.leaves(parent_state.ci_fn), strict=True
    ):
        assert jnp.array_equal(a, b)

    # step resets to 0 for the fresh schedule.
    assert int(finetuned.step) == 0
    assert finetuned.step.dtype == jnp.int32

    # optimizer states + sources are the FRESH reference's (not the parent's). The fresh
    # sources are RNG-drawn from seed 7; the parent's from seed 1 — they must differ.
    for state_key, fresh_src in fresh.sources.items():
        for site, arr in fresh_src.items():
            assert jnp.array_equal(finetuned.sources[state_key][site], arr)
            assert not jnp.array_equal(
                finetuned.sources[state_key][site], parent_state.sources[state_key][site]
            )
    for a, b in zip(
        jax.tree.leaves(finetuned.components_opt_state),
        jax.tree.leaves(fresh.components_opt_state),
        strict=True,
    ):
        assert jnp.array_equal(a, b)


def test_init_from_parent_rejects_missing_step(tmp_path: Path):
    tgt, parent_state, step, resid = _build(seed=1)
    parent_ckpt_dir = tmp_path / "parent" / "ckpts"
    mgr = make_checkpoint_manager(parent_ckpt_dir, keep_last=2)
    save_state(mgr, 2, parent_state)

    _, fresh, _, _ = _build(seed=7)
    with pytest.raises(AssertionError, match="parent step 99 not in"):
        init_from_parent(parent_ckpt_dir, parent_step=99, reference=fresh)


def _stamp(raw: dict[str, object], run_dir: Path) -> Path:
    raw = dict(raw, run_id="p-0123abcd", out_dir=str(run_dir.parent))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(raw))
    return run_dir


def test_structural_compat_passes_on_matching_changes_only(tmp_path: Path):
    raw = yaml.safe_load((CONFIGS / "llama8b_l18_b128_cmp32.yaml").read_text())
    parent_dir = _stamp(raw, tmp_path / "p-0123abcd")

    # A fine-tune that changes ONLY the LR + steps (same sites/C, same ci-fn arch) is OK.
    new_raw = dict(raw)
    new_raw["pd"] = dict(raw["pd"], steps=raw["pd"]["steps"] // 2)
    new_raw["pd"]["components_optimizer"] = dict(
        raw["pd"]["components_optimizer"],
        lr_schedule=dict(raw["pd"]["components_optimizer"]["lr_schedule"], start_val=1e-4),
    )
    new_cfg = build_from_schema(dict(new_raw, run_id="p-aaaaaaaa", out_dir=str(tmp_path)))
    prov = ResumeProvenance(parent_run_dir=parent_dir, parent_step=10)
    assert_finetune_structural_compat(new_cfg, prov)


def test_structural_compat_fires_on_changed_C(tmp_path: Path):
    raw = yaml.safe_load((CONFIGS / "llama8b_l18_b128_cmp32.yaml").read_text())
    parent_dir = _stamp(raw, tmp_path / "p-0123abcd")

    new_raw = dict(raw)
    new_targets = [dict(t, C=t["C"] // 2) for t in raw["pd"]["decomposition_targets"]]
    new_raw["pd"] = dict(raw["pd"], decomposition_targets=new_targets)
    new_cfg = build_from_schema(dict(new_raw, run_id="p-aaaaaaaa", out_dir=str(tmp_path)))
    prov = ResumeProvenance(parent_run_dir=parent_dir, parent_step=10)
    with pytest.raises(AssertionError, match="fine-tune sites mismatch"):
        assert_finetune_structural_compat(new_cfg, prov)
