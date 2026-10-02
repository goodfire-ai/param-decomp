"""Per-batch sources preserve checkpoint trajectories and dense Adam semantics."""

from dataclasses import replace
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

from param_decomp.core.adversary import (
    PersistentAdversary,
    SourcesAdamState,
    SourceStacks,
    full_source_components,
    init_sources_adam_state,
)
from param_decomp.core.checkpoint import (
    make_checkpoint_manager,
    restore_step,
    save_state,
)
from param_decomp.core.ci_fn.implementations.global_mlp import GlobalMLPCIFnArch
from param_decomp.core.ci_fn.interface import TapSpec
from param_decomp.core.components import DenseFactorization, SiteSpec, init_component_stacks
from param_decomp.core.configs import (
    AdamPGDConfig,
    AdamWOptimizerConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    KeepAllCheckpoints,
    MergedStochasticSubsetPooledPPGDReconLossConfig,
    SourcePoolConfig,
)
from param_decomp.core.init_placed import init_source_pool_sharded
from param_decomp.core.losses import BatchFrequency
from param_decomp.core.masking import sample_source_pool
from param_decomp.core.objective import build_objective
from param_decomp.core.run_state import _adamw_optimizer
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.core.train import Decomposition, PDState, PDTrainingState, TrainState

SITES = (SiteSpec("site", DenseFactorization(d_in=4, d_out=4, C=4), "dense"),)
POOL = SourcePoolConfig(size_per_batch_element=8)
ADAM = AdamPGDConfig(beta1=0.8, beta2=0.9, lr_schedule=ScheduleConfig.constant(1e-3))
BATCH = 8


def _mesh(replicate: int, fsdp: int, tp: int) -> Mesh:
    n = replicate * fsdp * tp
    return Mesh(
        np.asarray(jax.devices()[:n]).reshape(replicate, fsdp, tp),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )


def _state(pool: SourcePoolConfig, batch: int, mesh: Mesh) -> PDState[object]:
    decomposition = Decomposition(
        components=init_component_stacks(SITES, jax.random.key(0)),
        ci_fn=GlobalMLPCIFnArch(
            hidden_dims=(4,), has_position_axis=False, input_taps=(TapSpec("input", 4),)
        ).initialize(SITES, None, jax.random.key(1)),
    )
    decomposition = jax.device_put(decomposition, NamedSharding(mesh, P()))
    sources = init_source_pool_sharded(SITES, pool, batch, jnp.float32, jax.random.key(2), mesh)
    with jax.set_mesh(mesh):
        adversary = PersistentAdversary(
            sources=sources,
            opt_state=init_sources_adam_state(sources),
            state_key="pool",
            optimizer=ADAM,
            n_warmup=2,
        )
    objective = build_objective(
        (
            FaithfulnessLossConfig(coeff=1.0),
            ImportanceMinimalityLossConfig(coeff=1e-4, gamma=ScheduleConfig.constant(1.0)),
            MergedStochasticSubsetPooledPPGDReconLossConfig(
                name="pool",
                coeff=1.0,
                pool=pool,
                optimizer=ADAM,
                n_warmup_steps=2,
                adv_fraction=ScheduleConfig.constant(1.0),
            ),
        ),
        SITES,
    )
    optimizer = _adamw_optimizer(AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(1e-3)), 1)
    return TrainState(
        decomposition=decomposition,
        training=PDTrainingState(
            frequency=BatchFrequency(),
            objective=jax.device_put(objective, NamedSharding(mesh, P())),
            components_opt_state=optimizer.init(eqx.filter(decomposition.components, eqx.is_array)),
            ci_fn_opt_state=optimizer.init(eqx.filter(decomposition.ci_fn, eqx.is_array)),
            adversaries={"pool": adversary},
            step=jax.device_put(jnp.zeros((), jnp.int32), NamedSharding(mesh, P())),
        ),
    )


def _ascent(state: PDState[object], key: Array, mesh: Mesh) -> tuple[PDState[object], SourceStacks]:
    def step(adversary: PersistentAdversary) -> tuple[PersistentAdversary, SourceStacks]:
        def objective(sources: SourceStacks) -> Array:
            sampled = sample_source_pool(key, sources, (BATCH,))["site"]
            return jnp.mean(full_source_components(sampled.components)) + jnp.mean(sampled.delta)

        grad = jax.grad(objective)(adversary.sources)
        return adversary.final_ascend(grad, jnp.array(0.0), key), grad

    with jax.set_mesh(mesh):
        adversary, grad = jax.jit(step)(state.training.adversaries["pool"])
        training = replace(
            state.training, adversaries={"pool": adversary}, step=state.training.step + 1
        )
    return replace(state, training=training), grad


@pytest.mark.parametrize(
    "topology",
    [
        (1, 1, 1),
        pytest.param((1, 2, 2), marks=pytest.mark.multidevice),
        pytest.param((2, 2, 1), marks=pytest.mark.multidevice),
    ],
)
def test_batch_checkpoint_continues_on_the_restoring_mesh(
    tmp_path: Path, topology: tuple[int, int, int]
):
    if np.prod(topology) > jax.device_count():
        pytest.skip("requires four CPU devices")
    original_mesh = _mesh(1, 1, 1)
    state, _ = _ascent(_state(POOL, BATCH, original_mesh), jax.random.key(3), original_mesh)
    state, _ = _ascent(state, jax.random.key(4), original_mesh)
    expected, _ = _ascent(state, jax.random.key(5), original_mesh)
    destination = _mesh(*topology)
    reference = _state(POOL, BATCH, destination)
    with make_checkpoint_manager(tmp_path / "ckpts", KeepAllCheckpoints()) as manager:
        save_state(manager, 2, state)
        restored = restore_step(
            manager, jax.tree.map(ocp.utils.to_shape_dtype_struct, reference), 2
        )
    for actual, saved, target in zip(
        jax.tree.leaves(restored), jax.tree.leaves(state), jax.tree.leaves(reference), strict=True
    ):
        np.testing.assert_array_equal(actual, saved)
        assert actual.sharding == target.sharding
    continued, _ = _ascent(restored, jax.random.key(5), destination)
    for actual, uninterrupted in zip(
        jax.tree.leaves(continued), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(actual, uninterrupted)
    opt = continued.training.adversaries["pool"].opt_state
    assert isinstance(opt, SourcesAdamState)
    assert float(opt.step_count) == 3


@pytest.mark.parametrize("topology", [(1, 1, 1), (1, 2, 2), (2, 2, 1), (1, 1, 4), (4, 1, 1)])
@pytest.mark.multidevice
def test_sources_and_adam_moments_retain_only_device_local_minipools(
    topology: tuple[int, int, int],
):
    if np.prod(topology) > jax.device_count():
        pytest.skip("requires four CPU devices")
    replicate, fsdp, tp = topology
    state = _state(POOL, BATCH, _mesh(*topology))
    adversary = state.training.adversaries["pool"]
    assert isinstance(adversary.opt_state, SourcesAdamState)
    local_batch = BATCH // (replicate * fsdp)
    for storage in (adversary.sources, adversary.opt_state.m, adversary.opt_state.v):
        stack = storage.stacks["dense"]
        components = full_source_components(stack.components)
        assert components.shape == (1, BATCH, POOL.size_per_batch_element, 4)
        assert stack.delta.shape == (1, BATCH, POOL.size_per_batch_element)
        for shard in components.addressable_shards:
            assert shard.data.shape == (1, local_batch, POOL.size_per_batch_element, 4 // tp)
        for shard in stack.delta.addressable_shards:
            assert shard.data.shape == (1, local_batch, POOL.size_per_batch_element)


@pytest.mark.parametrize(
    "batch,particles",
    [(BATCH, 4), (BATCH // 2, 16)],
)
def test_batch_checkpoint_rejects_changed_pool_dimensions(
    tmp_path: Path, batch: int, particles: int
):
    mesh = _mesh(1, 1, 1)
    with make_checkpoint_manager(tmp_path / "ckpts", KeepAllCheckpoints()) as manager:
        save_state(manager, 0, _state(POOL, BATCH, mesh))
        with pytest.raises(ValueError, match="[Ss]hape"):
            restore_step(
                manager,
                jax.tree.map(
                    ocp.utils.to_shape_dtype_struct,
                    _state(SourcePoolConfig(size_per_batch_element=particles), batch, mesh),
                ),
                0,
            )


def test_unsampled_batch_particles_keep_adam_decay_and_movement():
    mesh = _mesh(1, 1, 1)
    before, _ = _ascent(_state(POOL, BATCH, mesh), jax.random.key(3), mesh)
    after, gradient = _ascent(before, jax.random.key(4), mesh)
    previous = before.training.adversaries["pool"]
    current = after.training.adversaries["pool"]
    assert isinstance(previous.opt_state, SourcesAdamState)
    assert isinstance(current.opt_state, SourcesAdamState)
    for source, updated, first, second, new_first, new_second, grad in zip(
        jax.tree.leaves(previous.sources),
        jax.tree.leaves(current.sources),
        jax.tree.leaves(previous.opt_state.m),
        jax.tree.leaves(previous.opt_state.v),
        jax.tree.leaves(current.opt_state.m),
        jax.tree.leaves(current.opt_state.v),
        jax.tree.leaves(gradient),
        strict=True,
    ):
        unsampled = (np.asarray(grad) == 0) & (np.asarray(first) > 0)
        assert unsampled.any()
        np.testing.assert_allclose(
            np.asarray(new_first)[unsampled], np.asarray(first)[unsampled] * ADAM.beta1
        )
        np.testing.assert_allclose(
            np.asarray(new_second)[unsampled], np.asarray(second)[unsampled] * ADAM.beta2
        )
        assert (np.asarray(updated)[unsampled] > np.asarray(source)[unsampled]).all()
