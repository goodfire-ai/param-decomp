"""Selected slots have token placement; component statistics have an explicit C layout."""

import jax
import jax.numpy as jnp
from jax.sharding import AbstractMesh, AxisType, NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.core.components import SelectedCI
from param_decomp.core.decomposed_linear import constrain_component_activation
from param_decomp.core.dict_utils import FrozenMapping
from param_decomp.core.placement import PlacedRule


def test_selected_slots_do_not_inherit_full_component_partition():
    mesh = AbstractMesh((2, 2), ("data", "tp"), axis_types=(AxisType.Explicit,) * 2)
    component_row = PlacedRule(
        mesh, "activations/component", FrozenMapping({"batch": ("data",), "C": ("tp",)})
    )

    def admit(values: jax.Array, indices: jax.Array):
        return constrain_component_activation(SelectedCI(values, indices, 4), component_row)

    selected = jax.eval_shape(
        admit,
        jax.ShapeDtypeStruct((4, 3, 8), jnp.float32, sharding=NamedSharding(mesh, P())),
        jax.ShapeDtypeStruct((4, 3, 2), jnp.int32, sharding=NamedSharding(mesh, P())),
    )
    assert isinstance(selected, SelectedCI)
    assert isinstance(selected.values.sharding, NamedSharding)
    assert isinstance(selected.block_indices.sharding, NamedSharding)
    assert selected.values.sharding.spec == P("data", None, None)
    assert selected.block_indices.sharding.spec == P("data", None, None)
