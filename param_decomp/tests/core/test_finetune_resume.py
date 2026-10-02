"""Fine-tuning restores the parent decomposition before initializing fresh training state."""

from dataclasses import replace
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import pytest
import yaml
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.core.built_run import RunInstance
from param_decomp.core.checkpoint import make_checkpoint_manager, save_state
from param_decomp.core.configs import KeepLastNCheckpoints, ResumeProvenance
from param_decomp.core.run import _resolve_training_start, _start_training
from param_decomp.core.run_files import LAUNCH_CONFIG_FILENAME
from param_decomp.core.sharding import hsdp_mesh
from param_decomp.core.train import Decomposition, PDState, TrainState
from param_decomp.experiments.lm.config import build_from_schema
from param_decomp.experiments.lm.training import assert_finetune_structural_compat
from param_decomp.lm.batch import LMBatchWithDocuments
from param_decomp.tests.core.test_checkpoint import _build, _optimizer

CONFIGS = Path(__file__).parents[2] / "experiments" / "lm" / "configs"
DATA_ROOT = Path("out")


def _initialize_finetune(
    fresh: PDState[LMBatchWithDocuments], parent_dir: Path, parent_step: int
) -> PDState[LMBatchWithDocuments]:
    mesh = hsdp_mesh(1, 1, 1)
    destination = jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(
            leaf.shape, leaf.dtype, sharding=NamedSharding(mesh, P())
        ),
        fresh,
    )

    def initialize_decomposition() -> Decomposition[LMBatchWithDocuments]:
        raise AssertionError("fine-tuning must restore without initializing a decomposition")

    def initialize(
        decomposition: Decomposition[LMBatchWithDocuments],
    ) -> PDState[LMBatchWithDocuments]:
        components_optimizer = _optimizer(muon=False, grad_clip_norm=0.01, dimension_numbers=None)
        ci_fn_optimizer = _optimizer(muon=False, grad_clip_norm=None, dimension_numbers=None)
        return TrainState(
            decomposition=decomposition,
            training=replace(
                fresh.training,
                components_opt_state=components_optimizer.init(
                    eqx.filter(decomposition.components, eqx.is_array)
                ),
                ci_fn_opt_state=ci_fn_optimizer.init(eqx.filter(decomposition.ci_fn, eqx.is_array)),
            ),
        )

    run = RunInstance(
        run_name="fine-tune",
        run_id="p-aaaaaaaa",
        out_dir=parent_dir.parent,
        wandb=None,
        resume_provenance=ResumeProvenance(parent_run_dir=parent_dir, parent_step=parent_step),
    )
    return _start_training(
        _resolve_training_start(run, None, 100),
        initialize_decomposition=initialize_decomposition,
        initialize=initialize,
        destination=destination,
        mesh=mesh,
        is_main=False,
    )


def test_finetune_loads_components_resets_schedule(tmp_path: Path):
    # A parent run: train a couple of steps so its V/U + ci_fn + sources + step are
    # non-trivial, then checkpoint.
    model, parent_state, step, resid = _build(seed=1)
    for i in range(2):
        parent_state, _ = step(model, parent_state, resid, jax.random.PRNGKey(i))
    assert int(parent_state.training.step) == 2

    parent_ckpt_dir = tmp_path / "parent" / "ckpts"
    mgr = make_checkpoint_manager(parent_ckpt_dir, KeepLastNCheckpoints(n=2))
    save_state(mgr, 2, parent_state)

    # Different seeds distinguish inherited parameters from fresh adversaries.
    _, fresh, _, _ = _build(seed=7)
    finetuned = _initialize_finetune(fresh, parent_ckpt_dir.parent, parent_step=2)

    # components + ci_fn come from the parent.
    for a, b in zip(
        jax.tree.leaves(finetuned.decomposition.components),
        jax.tree.leaves(parent_state.decomposition.components),
        strict=True,
    ):
        assert jnp.array_equal(a, b)
    for a, b in zip(
        jax.tree.leaves(finetuned.decomposition.ci_fn),
        jax.tree.leaves(parent_state.decomposition.ci_fn),
        strict=True,
    ):
        assert jnp.array_equal(a, b)

    # step resets to 0 for the fresh schedule.
    assert int(finetuned.training.step) == 0
    assert finetuned.training.step.dtype == jnp.int32

    # Sources and optimizer state start fresh under the new run's schedule.
    for state_key, fresh_adv in fresh.training.adversaries.items():
        got = jax.tree.leaves(finetuned.training.adversaries[state_key].sources)
        parent = jax.tree.leaves(parent_state.training.adversaries[state_key].sources)
        assert all(
            jnp.array_equal(a, b)
            for a, b in zip(got, jax.tree.leaves(fresh_adv.sources), strict=True)
        )
        assert any(not jnp.array_equal(a, b) for a, b in zip(got, parent, strict=True))
    for a, b in zip(
        jax.tree.leaves(finetuned.training.components_opt_state),
        jax.tree.leaves(fresh.training.components_opt_state),
        strict=True,
    ):
        assert jnp.array_equal(a, b)

    def forbid_decomposition_init() -> Decomposition[LMBatchWithDocuments]:
        raise AssertionError("resume must not initialize parameters")

    def forbid_training_init(
        _decomposition: Decomposition[LMBatchWithDocuments],
    ) -> PDState[LMBatchWithDocuments]:
        raise AssertionError("resume must not initialize training history")

    run = RunInstance(
        run_name="fine-tune",
        run_id="p-aaaaaaaa",
        out_dir=tmp_path,
        wandb=None,
        resume_provenance=ResumeProvenance(parent_run_dir=parent_ckpt_dir.parent, parent_step=2),
    )
    with make_checkpoint_manager(run.run_dir / "ckpts", KeepLastNCheckpoints(n=2)) as own:
        save_state(own, 0, finetuned)
        start = _resolve_training_start(run, own, 100)
        save_state(own, 2, parent_state)
        resumed = _start_training(
            start,
            initialize_decomposition=forbid_decomposition_init,
            initialize=forbid_training_init,
            destination=jax.tree.map(ocp.utils.to_shape_dtype_struct, finetuned),
            mesh=hsdp_mesh(1, 1, 1),
            is_main=False,
        )
    for expected, actual in zip(jax.tree.leaves(finetuned), jax.tree.leaves(resumed), strict=True):
        assert jnp.array_equal(expected, actual)


def test_finetune_rejects_missing_parent_step(tmp_path: Path):
    _, parent_state, _, _ = _build(seed=1)
    parent_ckpt_dir = tmp_path / "parent" / "ckpts"
    mgr = make_checkpoint_manager(parent_ckpt_dir, KeepLastNCheckpoints(n=2))
    save_state(mgr, 2, parent_state)

    _, fresh, _, _ = _build(seed=7)
    with pytest.raises(FileNotFoundError, match="99"):
        _initialize_finetune(fresh, parent_ckpt_dir.parent, parent_step=99)


def _stamp(raw: dict[str, object], run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / LAUNCH_CONFIG_FILENAME).write_text(yaml.safe_dump(raw))
    return run_dir


def test_structural_compat_passes_on_matching_changes_only(tmp_path: Path):
    raw = yaml.safe_load((CONFIGS / "llama3_1_8b.yaml").read_text())
    parent_dir = _stamp(raw, tmp_path / "p-0123abcd")

    # A fine-tune that changes ONLY the LR + steps (same sites/C, same ci-fn arch) is OK.
    new_raw = dict(raw)
    new_raw["pd"] = dict(raw["pd"], steps=raw["pd"]["steps"] // 2)
    new_raw["pd"]["components_optimizer"] = dict(
        raw["pd"]["components_optimizer"],
        lr_schedule=dict(raw["pd"]["components_optimizer"]["lr_schedule"], max_val=1e-4),
    )
    new_cfg, _ = build_from_schema(new_raw, "p-aaaaaaaa", DATA_ROOT)
    prov = ResumeProvenance(parent_run_dir=parent_dir, parent_step=10)
    assert_finetune_structural_compat(new_cfg, prov, DATA_ROOT)


def test_structural_compat_fires_on_changed_C(tmp_path: Path):
    raw = yaml.safe_load((CONFIGS / "llama3_1_8b.yaml").read_text())
    parent_dir = _stamp(raw, tmp_path / "p-0123abcd")

    new_raw = dict(raw)
    old_sites = raw["decomposition"]["sites"]
    halved_cs = {matrix: c // 2 for matrix, c in old_sites["cs"].items()}
    new_raw["decomposition"] = dict(raw["decomposition"], sites=dict(old_sites, cs=halved_cs))
    new_cfg, _ = build_from_schema(new_raw, "p-aaaaaaaa", DATA_ROOT)
    prov = ResumeProvenance(parent_run_dir=parent_dir, parent_step=10)
    with pytest.raises(AssertionError, match="fine-tune sites mismatch"):
        assert_finetune_structural_compat(new_cfg, prov, DATA_ROOT)
