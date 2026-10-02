"""Persistent sources as target-declared semantic stacks (`SourceStacks`).

Storage mirrors `ComponentStacks`: one slot-major stack per `SiteSpec.group`, consumers
read the `per_site()` view. Pins: (1) the view is bit-identical to the standalone per-site
draw under the same per-site keys (the RNG-chain pin — stacking moved storage, not the
stream); (2) the declared shardings mirror the container leaf-for-leaf with the slot axis
replicated and `{batch: data, C/expert: tp}` preserved; (3) the stochastic-rounding store
folds its key per container leaf, codified as a golden."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike

from param_decomp.core.adversary import (
    BlockedSourceComponents,
    SourceStacks,
    _draw_site_source,
    init_persistent_sources,
    source_values_to_float,
    store_sources,
    store_unit_float,
)
from param_decomp.core.components import (
    BlockedFactorization,
    DenseFactorization,
    SiteSpec,
    site_stack_indices_for,
)
from param_decomp.core.configs import BatchSourceShape
from param_decomp.core.init_placed import init_sources_sharded, persistent_sources_shardings
from param_decomp.core.model import PositionAxis, Positioned, Positionless

N_EXPERTS, C_PER_EXPERT, DENSE_C = 4, 3, 6


def _sites() -> tuple[SiteSpec, ...]:
    """Two semantic groups per factorization, interleaved in site order, so slot ≠ site
    index and two same-shape expert groups must NOT share a stack."""
    expert = BlockedFactorization(n_blocks=N_EXPERTS, d_in=8, d_out=4, c_per_block=C_PER_EXPERT)
    dense = DenseFactorization(d_in=8, d_out=8, C=DENSE_C)
    return (
        SiteSpec(name="l0.gate", factorization=expert, group="gate"),
        SiteSpec(name="l0.up", factorization=expert, group="up"),
        SiteSpec(name="l0.shared", factorization=dense, group="shared"),
        SiteSpec(name="l1.gate", factorization=expert, group="gate"),
        SiteSpec(name="l1.up", factorization=expert, group="up"),
        SiteSpec(name="l1.shared", factorization=dense, group="shared"),
    )


def _mesh() -> Mesh:
    return Mesh(
        np.asarray(jax.devices()).reshape(-1, 1),
        ("data", "tp"),
        axis_types=(AxisType.Explicit,) * 2,
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16, jnp.uint16])
def test_stacks_follow_the_declared_grouping_and_views_pin_the_per_site_draws(dtype: DTypeLike):
    sites = _sites()
    leading = (2, 5)
    key = jax.random.PRNGKey(3)
    stacks = init_persistent_sources(sites, leading, dtype, key)

    assert stacks.site_stack_indices == site_stack_indices_for(sites)
    assert set(stacks.stacks) == {"gate", "up", "shared"}
    gate = stacks.stacks["gate"].components
    assert isinstance(gate, BlockedSourceComponents)
    assert gate.values.shape == (2, *leading, N_EXPERTS, C_PER_EXPERT)
    shared = stacks.stacks["shared"].components
    assert isinstance(shared, jax.Array) and shared.shape == (2, *leading, DENSE_C)
    assert stacks.stacks["up"].delta.shape == (2, *leading)
    assert all(leaf.dtype == dtype for leaf in jax.tree.leaves(stacks))

    # the RNG-chain pin: keys split per SITE in site order; slot j of a group's stack is
    # the standalone draw under that site's key, whatever the interleaving
    view = stacks.per_site()
    assert list(view) == [spec.name for spec in sites]
    keys = jax.random.split(key, len(sites))
    for spec, site_key in zip(sites, keys, strict=True):
        want = jax.tree.map(
            lambda a: store_unit_float(a, dtype),
            _draw_site_source(site_key, spec.factorization, leading),
        )
        for got, ref in zip(jax.tree.leaves(view[spec.name]), jax.tree.leaves(want), strict=True):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))

    # slicing commutes with the elementwise dequant: site views of the float view are
    # the float views of the site slices
    float_view = source_values_to_float(stacks).per_site()
    for name, site in view.items():
        for got, ref in zip(
            jax.tree.leaves(float_view[name]),
            jax.tree.leaves(source_values_to_float(site)),
            strict=True,
        ):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


def test_grouping_refuses_mixed_factorizations_within_a_group():
    expert = BlockedFactorization(n_blocks=N_EXPERTS, d_in=8, d_out=4, c_per_block=C_PER_EXPERT)
    sites = (
        SiteSpec(name="a", factorization=expert, group="g"),
        SiteSpec(
            name="b", factorization=DenseFactorization(d_in=8, d_out=8, C=expert.C), group="g"
        ),
    )
    with pytest.raises(AssertionError, match="mixes factorizations"):
        init_persistent_sources(sites, (1, 2), jnp.float32, jax.random.PRNGKey(0))


def test_declared_shardings_mirror_the_container_with_the_slot_axis_replicated():
    """Every (positions x source_shape) arm: the shardings tree has the container's exact
    structure (the fit check zips it leaf-for-leaf), the slot axis is never sharded, the
    batch-B leading axis rides `data`, dense C and the expert block axis ride `tp`."""
    sites = _sites()
    mesh = _mesh()
    n_data = mesh.shape["data"]
    batch = 4 * n_data
    cases: list[tuple[PositionAxis, BatchSourceShape, P]] = [
        (Positioned(7), "bc", P(None, "data", None)),
        (Positioned(7), "bsc", P(None, "data", None)),
        (Positionless(), "bc", P(None, "data")),
    ]
    for positions, source_shape, delta_spec in cases:
        shardings = persistent_sources_shardings(sites, positions, source_shape, batch, mesh)
        placed = init_sources_sharded(
            sites,
            positions,
            source_shape,
            batch,
            jnp.uint16,
            jax.random.PRNGKey(1),
            mesh,
        )
        assert jax.tree.structure(shardings) == jax.tree.structure(placed)
        for group, stack in shardings.stacks.items():
            assert stack.delta.spec == delta_spec, (source_shape, group)
            match stack.components:
                case BlockedSourceComponents(values=values):
                    assert isinstance(values, NamedSharding)
                    assert values.spec == P(*delta_spec, "tp", None), (source_shape, group)
                case components:
                    assert components.spec == P(*delta_spec, "tp"), (source_shape, group)
        for leaf, sharding in zip(jax.tree.leaves(placed), jax.tree.leaves(shardings), strict=True):
            assert leaf.sharding.is_equivalent_to(sharding, leaf.ndim), (source_shape, leaf.shape)
    with pytest.raises(ValueError, match="positionless"):
        init_sources_sharded(
            sites,
            Positionless(),
            "bsc",
            batch,
            jnp.float32,
            jax.random.PRNGKey(1),
            mesh,
        )


def test_stochastic_store_stream_is_a_function_of_the_container_leaf_order():
    """`store_sources` folds `key` per leaf INDEX of the container. The golden codifies
    the stream for this layout: a leaf-enumeration change (a regrouping, a reordering of
    `SourceStack`'s fields) changes the trajectory and must re-pin deliberately."""
    sites = _sites()
    stored = init_persistent_sources(sites, (2, 3), jnp.uint16, jax.random.PRNGKey(5))
    values = jax.tree.map(
        lambda a: jnp.full_like(a, 0.5 + 0.2 / 65535.0), source_values_to_float(stored)
    )
    landed = store_sources(stored, values, jax.random.PRNGKey(6))
    assert jax.tree.structure(landed) == jax.tree.structure(stored)
    assert all(leaf.dtype == jnp.uint16 for leaf in jax.tree.leaves(landed))
    # every coordinate lands on one of the two neighbours of 0.5·65535 + 0.2
    moved = [np.asarray(leaf, np.int64) - 32767 for leaf in jax.tree.leaves(landed)]
    assert all(set(np.unique(m)) <= {0, 1} for m in moved)
    checksums = tuple(int(m.sum()) for m in moved)
    assert checksums == GOLDEN_STORE_CHECKSUMS, checksums


GOLDEN_STORE_CHECKSUMS = (107, 11, 51, 8, 90, 7)


def test_source_stacks_is_a_pytree_with_static_slots():
    stacks = init_persistent_sources(_sites(), (1, 2), jnp.float32, jax.random.PRNGKey(0))
    leaves, treedef = jax.tree.flatten(stacks)
    rebuilt = jax.tree.unflatten(treedef, leaves)
    assert isinstance(rebuilt, SourceStacks)
    assert rebuilt.site_stack_indices == stacks.site_stack_indices
    assert len(leaves) == 2 * 3
