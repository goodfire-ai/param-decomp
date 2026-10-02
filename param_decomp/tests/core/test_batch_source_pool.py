"""Per-batch pool sampling preserves the global objective and layout invariance."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

from param_decomp.core.adversary import (
    BlockedSourceComponents,
    Sources,
    SourceStack,
    SourceStacks,
    full_source_components,
    init_persistent_sources,
)
from param_decomp.core.components import BlockedFactorization, DenseFactorization, SiteSpec
from param_decomp.core.configs import (
    AdamPGDConfig,
    MergedStochasticSubsetPooledPPGDReconLossConfig,
    SourcePoolConfig,
)
from param_decomp.core.init_placed import (
    init_persistent_sources_from_config,
    persistent_sources_shardings_from_config,
)
from param_decomp.core.masking import sample_source_pool
from param_decomp.core.model import Positioned
from param_decomp.core.schedule import ScheduleConfig

SITES = (
    SiteSpec("dense_a", DenseFactorization(d_in=8, d_out=8, C=8), "dense"),
    SiteSpec("dense_b", DenseFactorization(d_in=8, d_out=8, C=8), "dense"),
    SiteSpec(
        "blocked",
        BlockedFactorization(n_blocks=4, d_in=8, d_out=8, c_per_block=2),
        "blocked",
    ),
)
POOL = SourcePoolConfig(size_per_batch_element=3)
BATCH = 32


def _pool(particles: int) -> SourceStacks:
    return init_persistent_sources(SITES, (BATCH, particles), jnp.float32, jax.random.key(1))


@pytest.mark.parametrize("particles", [1, 3])
@pytest.mark.parametrize("positions", [(), (5,)])
def test_batch_particles_and_global_gradient_normalization(
    particles: int, positions: tuple[int, ...]
):
    sources = _pool(particles)
    key = jax.random.key(51)
    sampled = sample_source_pool(key, sources, (BATCH, *positions))
    indices = np.asarray(
        jax.vmap(lambda batch_key: jax.random.choice(batch_key, particles))(
            jax.random.split(key, BATCH)
        )
    )
    batch_indices = np.arange(BATCH)
    for site in SITES:
        source = sampled[site.name]
        values = full_source_components(source.components)
        assert values.shape == (BATCH, *(1 for _ in positions), site.C)
        assert source.delta.shape == (BATCH, *(1 for _ in positions))
        original = sources.site(site.name)
        np.testing.assert_array_equal(
            np.asarray(values).reshape(BATCH, site.C),
            np.asarray(full_source_components(original.components))[batch_indices, indices],
        )
        np.testing.assert_array_equal(
            np.asarray(source.delta).reshape(BATCH),
            np.asarray(original.delta)[batch_indices, indices],
        )

    example_weights = jnp.arange(1, BATCH + 1, dtype=jnp.float32) / BATCH

    def loss(candidate: SourceStacks) -> Array:
        draw = sample_source_pool(key, candidate, (BATCH, *positions))
        total = jnp.array(0.0)
        for site in SITES:
            values = full_source_components(draw[site.name].components)
            full = jnp.broadcast_to(values, (BATCH, *positions, site.C))
            weights = example_weights.reshape(BATCH, *(1 for _ in full.shape[1:]))
            total += jnp.mean(full * weights)
            delta = jnp.broadcast_to(draw[site.name].delta, (BATCH, *positions))
            total += jnp.mean(delta * example_weights.reshape(BATCH, *(1 for _ in positions)))
        return total

    grads = jax.jit(jax.grad(loss))(sources)
    expected = np.zeros((BATCH, particles), dtype=np.float32)
    expected[batch_indices, indices] = np.asarray(example_weights) / BATCH
    for site in SITES:
        grad = grads.site(site.name)
        np.testing.assert_allclose(grad.delta, expected, atol=1e-7)
        np.testing.assert_allclose(
            full_source_components(grad.components),
            np.broadcast_to(expected[..., None] / site.C, (BATCH, particles, site.C)),
            atol=1e-7,
        )


def _placed_pool(sources: SourceStacks, mesh: Mesh) -> SourceStacks:
    data_axes = mesh.axis_names[:-1]
    stacks = {}
    for name, stack in sources.stacks.items():
        match stack.components:
            case BlockedSourceComponents(values=values):
                components = BlockedSourceComponents(
                    jax.device_put(
                        values, NamedSharding(mesh, P(None, data_axes, None, "tp", None))
                    )
                )
            case jax.Array():
                components = jax.device_put(
                    stack.components, NamedSharding(mesh, P(None, data_axes, None, "tp"))
                )
        stacks[name] = SourceStack(
            components=components,
            delta=jax.device_put(stack.delta, NamedSharding(mesh, P(None, data_axes, None))),
        )
    return SourceStacks(stacks=stacks, site_stack_indices=sources.site_stack_indices)


@pytest.mark.parametrize(
    "shape,axes",
    [
        ((1, 1), ("data", "tp")),
        ((2, 1), ("data", "tp")),
        ((4, 1), ("data", "tp")),
        ((2, 2), ("data", "tp")),
        ((2, 2, 1), ("replicate", "fsdp", "tp")),
        ((1, 2, 2), ("replicate", "fsdp", "tp")),
    ],
)
@pytest.mark.parametrize("positions", [(), (5,)])
@pytest.mark.multidevice
def test_batch_gather_and_transpose_are_local_and_layout_invariant(
    shape: tuple[int, ...], axes: tuple[str, ...], positions: tuple[int, ...]
):
    devices = int(np.prod(shape))
    if devices > jax.device_count():
        pytest.skip("requires more CPU devices; use --xla_force_host_platform_device_count=4")
    sources = _pool(POOL.size_per_batch_element)
    key = jax.random.key(51)
    expected = sample_source_pool(key, sources, (BATCH, *positions))
    mesh = Mesh(
        np.asarray(jax.devices()[:devices]).reshape(shape),
        axes,
        axis_types=(AxisType.Explicit,) * len(axes),
    )
    placed = _placed_pool(sources, mesh)
    with jax.set_mesh(mesh):

        def forward_and_transpose(pool: SourceStacks) -> tuple[Sources, SourceStacks]:
            sampled, transpose = jax.vjp(
                lambda p: sample_source_pool(key, p, (BATCH, *positions)), pool
            )
            cotangent = jax.tree.map(lambda value: jnp.ones_like(value) / BATCH, sampled)
            return sampled, transpose(cotangent)[0]

        compiled = jax.jit(forward_and_transpose).lower(placed).compile()
        actual, grads = compiled(placed)
        hlo = compiled.as_text()
    for result, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(result, reference)
    indices = np.asarray(
        jax.vmap(lambda batch_key: jax.random.choice(batch_key, POOL.size_per_batch_element))(
            jax.random.split(key, BATCH)
        )
    )
    expected_grad = np.zeros((BATCH, POOL.size_per_batch_element), dtype=np.float32)
    expected_grad[np.arange(BATCH), indices] = 1 / BATCH
    for site in SITES:
        grad = grads.site(site.name)
        np.testing.assert_array_equal(grad.delta, expected_grad)
        np.testing.assert_array_equal(
            full_source_components(grad.components),
            np.broadcast_to(expected_grad[..., None], (BATCH, POOL.size_per_batch_element, site.C)),
        )
    for collective in (
        "all-reduce(",
        "all-gather(",
        "reduce-scatter(",
        "all-to-all(",
        "collective-permute(",
    ):
        assert collective not in hlo, collective


def test_sampling_refuses_a_different_batch_size():
    with pytest.raises(AssertionError):
        sample_source_pool(jax.random.key(1), _pool(POOL.size_per_batch_element), (BATCH - 1,))


def _loss_config(pool: SourcePoolConfig) -> MergedStochasticSubsetPooledPPGDReconLossConfig:
    return MergedStochasticSubsetPooledPPGDReconLossConfig(
        coeff=1.0,
        pool=pool,
        adv_fraction=ScheduleConfig.constant(0.5),
        optimizer=AdamPGDConfig(lr_schedule=ScheduleConfig.constant(0.01)),
    )


@pytest.mark.multidevice
def test_batch_initialization_matches_declared_state_and_unplaced_values():
    if jax.device_count() < 4:
        pytest.skip("requires four CPU devices")
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(1, 2, 2),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    cfg = _loss_config(POOL)
    key = jax.random.key(16)
    expected = init_persistent_sources(
        SITES, (BATCH, POOL.size_per_batch_element), jnp.float32, key
    )
    declared = persistent_sources_shardings_from_config(SITES, Positioned(5), cfg, BATCH, mesh)
    with jax.set_mesh(mesh):
        actual = init_persistent_sources_from_config(SITES, Positioned(5), cfg, BATCH, key, mesh)
    for result, reference, sharding in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), jax.tree.leaves(declared), strict=True
    ):
        np.testing.assert_array_equal(result, reference)
        assert result.sharding == sharding
    dense = actual.stacks["dense"]
    assert isinstance(dense.components, jax.Array)
    assert dense.components.shape == (2, BATCH, POOL.size_per_batch_element, 8)
    assert isinstance(dense.components.sharding, NamedSharding)
    assert dense.components.sharding.spec == P(None, ("replicate", "fsdp"), None, "tp")
    blocked = actual.stacks["blocked"]
    assert isinstance(blocked.components, BlockedSourceComponents)
    assert blocked.components.values.shape == (1, BATCH, POOL.size_per_batch_element, 4, 2)
    assert isinstance(blocked.components.values.sharding, NamedSharding)
    assert blocked.components.values.sharding.spec == P(
        None, ("replicate", "fsdp"), None, "tp", None
    )
    assert dense.delta.shape == (2, BATCH, POOL.size_per_batch_element)
    assert isinstance(dense.delta.sharding, NamedSharding)
    assert dense.delta.sharding.spec == P(None, ("replicate", "fsdp"), None)


@pytest.mark.parametrize("particles", [1, 3, 8])
@pytest.mark.multidevice
def test_pool_size_is_independent_of_data_mesh(particles: int):
    if jax.device_count() < 2:
        pytest.skip("requires two CPU devices")
    mesh = Mesh(
        np.asarray(jax.devices()[:2]).reshape(2, 1),
        ("data", "tp"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    cfg = _loss_config(SourcePoolConfig(size_per_batch_element=particles))
    with jax.set_mesh(mesh):
        sources = init_persistent_sources_from_config(
            SITES, Positioned(5), cfg, 6, jax.random.key(16), mesh
        )
    assert sources.stacks["dense"].delta.shape == (2, 6, particles)
    with pytest.raises(ValueError, match="should evenly divide"):
        init_persistent_sources_from_config(SITES, Positioned(5), cfg, 3, jax.random.key(16), mesh)
