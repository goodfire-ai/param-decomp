"""Slot surgery primitives (#820): function preservation at birth, GradMax first-gradient
alignment, edit locality, bitwise rollback, fail-closed arms."""

from typing import Any, cast

import jax
import jax.numpy as jnp
import optax
import pytest

from param_decomp.core.ci_fn import LayerwiseMLPCIArch, build_ci_fn
from param_decomp.core.components import SiteSpec, component_stacks_from_sites
from param_decomp.core.slot_surgery import (
    birth_direction_from_grad,
    birth_slot,
    find_inactive_slot,
    protected_mask,
    rollback_trial,
    select_birth_site,
    snapshot_trial,
)
from param_decomp.core.train import Decomposition, TrainingItem, TrainState

SITES = (SiteSpec("s1", d_in=8, d_out=4, C=6), SiteSpec("s2", d_in=8, d_out=4, C=6))


def make_state(key: jax.Array) -> TrainState:
    k1, k2, k3 = jax.random.split(key, 3)
    vu = {}
    for i, spec in enumerate(SITES):
        V = jax.random.normal(jax.random.fold_in(k1, i), (spec.d_in, spec.C))
        U_full = jax.random.normal(jax.random.fold_in(k2, i), (spec.C, spec.d_out))
        vu[spec.name] = (V, U_full.at[4:, :].set(0.0))  # slots 4,5 exact null
    components = component_stacks_from_sites(vu)
    ci_fn = build_ci_fn(LayerwiseMLPCIArch(hidden_dims=(16,), has_position_axis=False), SITES, k3)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-3, weight_decay=0.0))
    import equinox as eqx

    vu_opt = opt.init(cast(Any, components))  # ComponentStacks is a pytree
    ci_opt = opt.init(eqx.filter(ci_fn, eqx.is_inexact_array))
    return TrainState(
        decomposition=Decomposition(components=components, ci_fn=ci_fn),
        training=TrainingItem(
            components_opt_state=vu_opt,
            ci_fn_opt_state=ci_opt,
            adversaries={},
            step=jnp.zeros((), jnp.int32),
        ),  # fmt: skip
    )


def represented(state: TrainState, site: str) -> jax.Array:
    V, U = state.decomposition.components.site(site)
    return V @ U


def test_find_inactive_slot_and_exhaustion() -> None:
    state = make_state(jax.random.key(0))
    assert find_inactive_slot(state.decomposition.components, "s1") == 4
    V, U = state.decomposition.components.site("s1")
    full = component_stacks_from_sites(
        {"s1": (V, jnp.ones_like(U)), "s2": state.decomposition.components.site("s2")}
    )
    assert find_inactive_slot(full, "s1") is None


def test_birth_preserves_represented_matrix_exactly() -> None:
    state = make_state(jax.random.key(1))
    before = represented(state, "s1")
    direction = jax.random.normal(jax.random.key(2), (8,))
    born = birth_slot(state, "s1", 4, direction)
    assert jnp.array_equal(represented(born, "s1"), before)
    V, U = born.decomposition.components.site("s1")
    assert jnp.allclose(jnp.linalg.norm(V[:, 4]), 1.0, atol=1e-6)
    assert jnp.all(U[4, :] == 0.0)


def test_first_gradient_aligns_with_leading_singular_mode() -> None:
    state = make_state(jax.random.key(3))
    target = jax.random.normal(jax.random.key(4), (8, 4))

    def loss_from_components(components: Any) -> jax.Array:
        V, U = components.site("s1")
        return 0.5 * jnp.sum((V @ U - target) ** 2)

    G = jax.grad(lambda M: 0.5 * jnp.sum((M - target) ** 2))(represented(state, "s1"))
    p, sigma = birth_direction_from_grad(G)
    born = birth_slot(state, "s1", 4, p)
    grads = jax.grad(loss_from_components)(born.decomposition.components)
    _, gU = grads.site("s1")
    # dL/dU[slot] = p^T G = sigma q^T: check norm and alignment with the exact SVD
    assert jnp.allclose(jnp.linalg.norm(gU[4, :]), sigma, rtol=1e-4)
    _, s_exact, Bt = jnp.linalg.svd(G, full_matrices=False)
    cos = jnp.abs(gU[4, :] @ Bt[0]) / (jnp.linalg.norm(gU[4, :]) + 1e-30)
    assert cos > 0.999
    assert jnp.allclose(sigma, s_exact[0], rtol=1e-4)


def test_only_selected_slot_and_moments_change() -> None:
    state = make_state(jax.random.key(5))
    born = birth_slot(state, "s1", 4, jax.random.normal(jax.random.key(6), (8,)))
    # untouched site identical everywhere
    assert jnp.array_equal(
        state.decomposition.components.site("s2")[0], born.decomposition.components.site("s2")[0]
    )
    V0, U0 = state.decomposition.components.site("s1")
    V1, U1 = born.decomposition.components.site("s1")
    keep = jnp.arange(6) != 4
    assert jnp.array_equal(V0[:, keep], V1[:, keep])
    assert jnp.array_equal(U0[keep, :], U1[keep, :])
    # CI head: only column 4 and bias 4 of s1 moved
    w0 = cast(Any, state.decomposition.ci_fn).site_mlps["s1"].weights[-1]
    w1 = cast(Any, born.decomposition.ci_fn).site_mlps["s1"].weights[-1]
    assert jnp.array_equal(w0[:, keep], w1[:, keep])
    assert jnp.all(w1[:, 4] == 0.0)
    assert cast(Any, born.decomposition.ci_fn).site_mlps["s1"].biases[-1][4] == 1.0


def test_rollback_restores_the_entire_pretrial_frontier() -> None:
    import equinox as eqx

    state = make_state(jax.random.key(7))
    snap = snapshot_trial(state, "s1", 4)
    born = birth_slot(state, "s1", 4, jax.random.normal(jax.random.key(8), (8,)))

    # contaminate EVERYTHING an ongoing probe trains: an unrelated slot, a CI hidden
    # layer, both optimizer states, and the step counter
    contaminated = birth_slot(
        # unrelated-slot edit via the public path (slot 5 is also null)
        born, "s1", 5, jax.random.normal(jax.random.key(9), (8,))
    )
    hidden_w = cast(Any, contaminated.decomposition.ci_fn).site_mlps["s2"].weights[0]
    contaminated = eqx.tree_at(
        lambda s: cast(Any, s.decomposition.ci_fn).site_mlps["s2"].weights[0],
        contaminated, hidden_w + 1.0,
    )  # fmt: skip
    contaminated = eqx.tree_at(
        lambda s: s.training.step, contaminated, contaminated.training.step + 7
    )
    bumped_opts = jax.tree_util.tree_map(
        lambda leaf: leaf + 1 if hasattr(leaf, "dtype") else leaf,
        (contaminated.training.components_opt_state, contaminated.training.ci_fn_opt_state),
    )
    contaminated = eqx.tree_at(
        lambda s: (s.training.components_opt_state, s.training.ci_fn_opt_state),
        contaminated, bumped_opts,
    )  # fmt: skip

    restored = rollback_trial(snap)
    for a, b in zip(
        jax.tree_util.tree_leaves(eqx.filter(restored, eqx.is_array)),
        jax.tree_util.tree_leaves(eqx.filter(state, eqx.is_array)),
        strict=True,
    ):
        assert jnp.array_equal(a, b)


def test_birth_refuses_non_null_slot_and_select_site_prefers_larger_signal() -> None:
    state = make_state(jax.random.key(10))
    with pytest.raises(AssertionError):
        birth_slot(state, "s1", 0, jnp.ones((8,)))  # slot 0 is live
    weak = jax.random.normal(jax.random.key(11), (8, 4)) * 0.01
    strong = jax.random.normal(jax.random.key(12), (8, 4))
    site, p, sigma = select_birth_site({"s1": weak, "s2": strong})
    assert site == "s2" and p.shape == (8,) and float(sigma) > 0


def test_protected_mask_shape() -> None:
    state = make_state(jax.random.key(13))
    mask = protected_mask(state.decomposition.components, "s1", 4)
    assert set(mask) == {"s1"} and mask["s1"].shape == (6,) and bool(mask["s1"][4])
    assert int(mask["s1"].sum()) == 1


def test_truncate_active_prefix_nulls_tail_and_is_idempotent() -> None:
    from param_decomp.core.slot_surgery import truncate_active_prefix

    state = make_state(jax.random.key(20))
    V, U = state.decomposition.components.site("s1")
    truncated = truncate_active_prefix(state, {"s1": 2})
    Vt, Ut = truncated.decomposition.components.site("s1")
    assert jnp.array_equal(Ut[:2, :], U[:2, :])  # prefix untouched
    assert jnp.all(Ut[2:, :] == 0.0)
    assert jnp.array_equal(Vt, V)  # V columns preserved
    assert find_inactive_slot(truncated.decomposition.components, "s1") == 2
    # represented matrix is the prefix-only product (allclose: XLA reduction order
    # differs between a 6-wide matmul with zero rows and a 2-wide matmul)
    assert jnp.allclose(represented(truncated, "s1"), V[:, :2] @ U[:2, :], atol=1e-6)
    b = cast(Any, truncated.decomposition.ci_fn).site_mlps["s1"].biases[-1]
    assert jnp.all(b[2:] == 0.0)
    again = truncate_active_prefix(truncated, {"s1": 2})
    for a, c in zip(
        jax.tree_util.tree_leaves((again.decomposition, again.training.components_opt_state)),
        jax.tree_util.tree_leaves((truncated.decomposition, truncated.training.components_opt_state)),
        strict=True,
    ):
        if hasattr(a, "shape"):
            assert jnp.array_equal(a, c)
