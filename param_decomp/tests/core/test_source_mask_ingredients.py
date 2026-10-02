"""Source reads admit one routing frame, retained through stacking and rematerialization."""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

from param_decomp.core.adversary import BlockedSourceComponents, SiteSource
from param_decomp.core.components import SelectedCI, SiteCI
from param_decomp.core.masking import read_source_mask, source_masking
from param_decomp.core.source_mask import SourceMaskIngredients


def _stack_layers(
    values: Mapping[str, SourceMaskIngredients],
) -> dict[str, SourceMaskIngredients]:
    return {"expert": jax.tree.map(lambda *xs: jnp.stack(xs), values["0"], values["1"])}


@pytest.mark.multidevice
@pytest.mark.skipif(len(jax.devices()) < 4, reason="requires four devices")
@pytest.mark.parametrize("source_leading", [(1, 1), (4, 1), (1, 3), (4, 3)])
def test_paired_expert_masks_keep_one_frame_through_scan_and_source_gradients(
    source_leading: tuple[int, int],
) -> None:
    layers, batch, positions, k, experts, c = 2, 4, 3, 2, 4, 2
    ids = jnp.array(
        [
            [[0, 1], [0, 1], [0, 1]],
            [[0, 2], [2, 0], [0, 2]],
            [[1, 3], [0, 3], [3, 2]],
            [[2, 3], [2, 3], [2, 3]],
        ],
        jnp.int32,
    )
    layer_ids = jnp.stack([ids, (ids + 1) % experts])
    values = jnp.linspace(0.1, 0.9, layers * batch * positions * k * c).reshape(
        layers, batch, positions, k, c
    )
    table = jnp.linspace(0.15, 0.85, layers * int(np.prod(source_leading)) * experts * c).reshape(
        layers, *source_leading, experts, c
    )

    def oracle(vals: Array, sources: Array) -> Array:
        total = jnp.float32(0)
        for layer in range(layers):
            ci = SelectedCI(vals[layer].reshape(batch, positions, k * c), layer_ids[layer], experts)
            pair = read_source_mask(
                ci,
                SiteSource(BlockedSourceComponents(sources[layer]), jnp.zeros(source_leading)),
            )
            selected = pair.compose()
            assert isinstance(selected, SelectedCI)
            total += jnp.sin(selected.values).sum()
        return total

    expected, expected_grads = jax.jit(jax.value_and_grad(oracle, argnums=(0, 1)))(values, table)
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(2, 2),
        ("data", "tp"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    with jax.set_mesh(mesh):
        placed_ids = jax.device_put(layer_ids, NamedSharding(mesh, P(None, "data", None, None)))
        placed_values = jax.device_put(
            values, NamedSharding(mesh, P(None, "data", None, None, None))
        )
        source_batch = "data" if source_leading[0] > 1 else None
        placed_table = jax.device_put(
            table, NamedSharding(mesh, P(None, source_batch, None, "tp", None))
        )

        def placed_loss(vals: Array, sources: Array) -> Array:
            cis: dict[str, SiteCI] = {}
            source_sites: dict[str, SiteSource] = {}
            for layer in range(layers):
                ci = SelectedCI(
                    vals[layer].reshape(batch, positions, k * c), placed_ids[layer], experts
                )
                cis[str(layer)] = ci
                source_sites[str(layer)] = SiteSource(
                    BlockedSourceComponents(sources[layer]), jnp.zeros(source_leading)
                )
            masking = source_masking(cis, source_sites)
            stacked = _stack_layers(masking.ingredients)["expert"]
            # Routing occurs once; the remaining leaves are the source and delta payloads.
            assert len(jax.tree.leaves(stacked)) == len(jax.tree.leaves(stacked.ci)) + 2

            @jax.checkpoint
            def layer_loss(carry: Array, pair: SourceMaskIngredients) -> tuple[Array, None]:
                mask = pair.compose()
                assert isinstance(mask, SelectedCI)
                return carry + jnp.sin(mask.values).sum(), None

            total, _ = jax.lax.scan(layer_loss, jnp.float32(0), stacked)
            return total

        actual, actual_grads = jax.jit(jax.value_and_grad(placed_loss, argnums=(0, 1)))(
            placed_values, placed_table
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
        for got, want in zip(actual_grads, expected_grads, strict=True):
            np.testing.assert_allclose(got, want, rtol=2e-6, atol=1e-6)
        assert jax.typeof(actual_grads[1]).sharding == jax.typeof(placed_table).sharding


def test_independent_selected_orders_align_at_source_read() -> None:
    ids = jnp.array([[[2, 0], [1, 2]]], dtype=jnp.int32)
    values = jnp.array([[[0.1, 0.2, 0.3, 0.4], [0.2, 0.4, 0.6, 0.8]]])
    source = SiteSource(
        BlockedSourceComponents(jnp.arange(6, dtype=jnp.float32).reshape(1, 1, 3, 2) / 6),
        jnp.ones((1, 1)),
    )
    normal = read_source_mask(SelectedCI(values, ids, 3), source)
    reversed_slots = read_source_mask(
        SelectedCI(values.reshape(1, 2, 2, 2)[:, :, ::-1].reshape(1, 2, 4), ids[..., ::-1], 3),
        source,
    )
    normal_mask, reversed_mask = normal.compose(), reversed_slots.compose()
    assert isinstance(normal_mask, SelectedCI) and isinstance(reversed_mask, SelectedCI)
    np.testing.assert_array_equal(
        normal_mask.values.reshape(1, 2, 2, 2),
        reversed_mask.values.reshape(1, 2, 2, 2)[:, :, ::-1],
    )
