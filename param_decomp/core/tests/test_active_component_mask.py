import jax.numpy as jnp

from param_decomp.core.components import component_stacks_from_sites, mask_component_stacks


def test_mask_component_stacks_zeroes_inactive_factor_entries_only() -> None:
    V = jnp.arange(15, dtype=jnp.float32).reshape(3, 5) + 1
    U = jnp.arange(10, dtype=jnp.float32).reshape(5, 2) + 1
    stacks = component_stacks_from_sites({"s": (V, U)})
    masked = mask_component_stacks(stacks, {"s": jnp.array([True, False, True, False, True])})
    got_v, got_u = masked.site("s")
    assert jnp.array_equal(got_v[:, [0, 2, 4]], V[:, [0, 2, 4]])
    assert jnp.array_equal(got_u[[0, 2, 4], :], U[[0, 2, 4], :])
    assert jnp.all(got_v[:, [1, 3]] == 0.0)
    assert jnp.all(got_u[[1, 3], :] == 0.0)
    assert mask_component_stacks(stacks, None) is stacks
