"""svd_null_tail init (PDConfig.component_init): exact SVD start, exact null tail,
prefix-stable across C, CI head pinned 1/0 by slot."""

import jax
import jax.numpy as jnp

from param_decomp.core.ci_fn import (
    LayerwiseMLPCIArch,
    build_ci_fn,
    pin_ci_head_null_tail,
)
from param_decomp.core.components import (
    ComponentStacks,
    SiteSpec,
    init_stack_arrays_svd_null_tail,
    site_slots_for,
)

SITES = (SiteSpec("linear1", d_in=40, d_out=10, C=100),)


def _weights(key: jax.Array) -> dict[str, jax.Array]:
    return {"linear1": jax.random.normal(key, (10, 40))}


def _stacks(sites: tuple[SiteSpec, ...], weights: dict[str, jax.Array]) -> ComponentStacks:
    return ComponentStacks(
        stacks=init_stack_arrays_svd_null_tail(sites, weights, jax.random.key(1)),
        site_slots=site_slots_for(sites),
    )


def test_svd_slots_reproduce_the_weight_and_tail_is_null() -> None:
    weights = _weights(jax.random.key(0))
    V, U = _stacks(SITES, weights).site("linear1")
    assert V.shape == (40, 100) and U.shape == (100, 10)
    r = 10
    assert jnp.allclose(V[:, :r] @ U[:r, :], weights["linear1"].T, atol=1e-4)
    assert jnp.allclose(V @ U, weights["linear1"].T, atol=1e-4)  # tail adds nothing
    assert jnp.all(U[r:, :] == 0.0)
    assert jnp.all(jnp.linalg.norm(V[:, r:], axis=0) > 0.0)  # tail V columns are live draws


def test_tail_v_columns_are_prefix_stable_across_c() -> None:
    weights = _weights(jax.random.key(0))
    small = _stacks((SiteSpec("linear1", 40, 10, C=50),), weights).site("linear1")
    large = _stacks((SiteSpec("linear1", 40, 10, C=100),), weights).site("linear1")
    assert jnp.array_equal(small[0], large[0][:, :50])


def test_ci_head_pinned_one_for_rank_slots_zero_for_tail() -> None:
    ci_fn = build_ci_fn(
        LayerwiseMLPCIArch(hidden_dims=(50,), has_position_axis=False), SITES, jax.random.key(2)
    )
    pinned = pin_ci_head_null_tail(ci_fn, SITES)
    ci = pinned({"linear1": jax.random.normal(jax.random.key(3), (8, 40))}, remat=False)
    assert jnp.all(ci.lower["linear1"][:, :10] == 1.0)
    assert jnp.all(ci.lower["linear1"][:, 10:] == 0.0)
