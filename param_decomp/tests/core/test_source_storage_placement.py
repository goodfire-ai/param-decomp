"""Source storage can differ from the expert layout that consumes sampled values."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.core.adversary import BlockedSourceComponents, SourceStack, SourceStacks
from param_decomp.core.components import (
    BlockedFactorization,
    SelectedCI,
    SiteSpec,
    site_stack_indices_for,
)
from param_decomp.core.masking import read_source_mask, sample_source_pool


@pytest.mark.multidevice
@pytest.mark.parametrize(
    "storage",
    [
        P(None, "data", None, "tp", None),
        P(None, "data", None, None, None),
        P(None, "data", None, None, "tp"),
    ],
)
def test_pool_storage_partition_is_independent_of_expert_reads(storage: P):
    if jax.device_count() < 4:
        pytest.skip("requires four devices")
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(2, 2),
        ("data", "tp"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    batch, positions, experts, components, selected = 4, 3, 4, 4, 2
    particles = 4
    sites = (
        SiteSpec(
            "site",
            BlockedFactorization(n_blocks=experts, d_in=4, d_out=4, c_per_block=components),
            "site",
        ),
    )
    table_np = np.linspace(
        0.01, 0.99, batch * particles * experts * components, dtype=np.float32
    ).reshape(1, batch, particles, experts, components)
    ids_np = (
        np.arange(batch * positions * selected).reshape(batch, positions, selected)
        + np.arange(batch)[:, None, None]
    ) % experts
    ids_np = ids_np.astype(np.int32)
    weights_np = np.linspace(
        -0.5, 1.0, batch * positions * selected * components, dtype=np.float32
    ).reshape(batch, positions, selected, components)
    key = jax.random.key(31)
    indices = np.asarray(
        jax.vmap(lambda batch_key: jax.random.choice(batch_key, particles))(
            jax.random.split(key, batch)
        )
    )
    expected = np.empty_like(weights_np)
    expected_grad = np.zeros_like(table_np)
    for b in range(batch):
        particle = indices[b]
        for t in range(positions):
            for k in range(selected):
                expert = ids_np[b, t, k]
                expected[b, t, k] = 0.2 + 0.8 * table_np[0, b, particle, expert]
                expected_grad[0, b, particle, expert] += 0.8 * weights_np[b, t, k]

    with jax.set_mesh(mesh):
        batch_layout = NamedSharding(mesh, P("data", None, None))
        ids = jax.device_put(ids_np, batch_layout)
        weights = jax.device_put(weights_np, NamedSharding(mesh, P("data", None, None, None)))
        table = jax.device_put(table_np, NamedSharding(mesh, storage))
        delta = jax.device_put(
            np.zeros((1, batch, particles), np.float32),
            NamedSharding(mesh, P(None, "data", None)),
        )

        def objective(table: jax.Array):
            token_ci = jnp.full(
                (batch, positions, selected, components),
                0.2,
                jnp.float32,
                out_sharding=NamedSharding(mesh, P("data", None, None, None)),
            )
            ci = SelectedCI(token_ci.reshape(batch, positions, selected * components), ids, experts)
            sources = SourceStacks(
                stacks={
                    "site": SourceStack(components=BlockedSourceComponents(table), delta=delta)
                },
                site_stack_indices=site_stack_indices_for(sites),
            )
            sampled = sample_source_pool(key, sources, (batch, positions))
            sampled_components = sampled["site"].components
            assert isinstance(sampled_components, BlockedSourceComponents)
            assert jax.typeof(sampled_components.values).sharding.spec[-2:] == storage[-2:]
            mask = read_source_mask(ci, sampled["site"]).compose()
            assert isinstance(mask, SelectedCI)
            tokens = mask.values.reshape(batch, positions, selected, components)
            return jnp.sum(tokens * weights), tokens

        (_, result), gradient = jax.jit(jax.value_and_grad(objective, has_aux=True))(table)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(gradient, expected_grad, rtol=1e-6, atol=1e-7)
        assert gradient.sharding.is_equivalent_to(table.sharding, table.ndim)
