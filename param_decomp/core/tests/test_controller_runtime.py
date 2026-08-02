import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from param_decomp.core.ci_fn import LayerwiseMLPCIArch, build_ci_fn
from param_decomp.core.components import SiteSpec, component_stacks_from_sites
from param_decomp.core.controller_runtime import (
    _block_transfer_cosines,
    choose_birth_batch,
    choose_birth_candidate,
)
from param_decomp.core.train import Decomposition, TrainingItem, TrainState


def _state() -> TrainState:
    sites = (SiteSpec("a", 4, 3, 5), SiteSpec("b", 4, 3, 5))
    vu = {
        site.name: (
            jax.random.normal(jax.random.fold_in(jax.random.key(0), i), (4, 5)),
            jax.random.normal(jax.random.fold_in(jax.random.key(1), i), (5, 3)).at[2:].set(0),
        )
        for i, site in enumerate(sites)
    }
    components = component_stacks_from_sites(vu)
    ci = build_ci_fn(
        LayerwiseMLPCIArch(hidden_dims=(8,), has_position_axis=False), sites, jax.random.key(2)
    )
    opt = optax.adam(1e-3)
    return TrainState(
        Decomposition(components, ci),
        TrainingItem(
            opt.init(components),
            opt.init(eqx.filter(ci, eqx.is_array)),
            {},
            jnp.zeros((), jnp.int32),
        ),
    )


def test_choose_birth_candidate_power_iterates_implicit_matrix_gradient() -> None:
    state = _state()
    G = {
        "a": jnp.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.2], [0, 0, 0]]),
        "b": jnp.array([[4.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5], [0, 0, 0]]),
    }

    def probe(state, _batch, _key, _active, _protected):
        grads = {}
        for site, (V, U) in state.decomposition.components.sites_items():
            grads[site] = (G[site] @ U.T, V.T @ G[site])
        return jnp.zeros(()), component_stacks_from_sites(grads)

    active = {site: jnp.array([True, True, False, False, False]) for site in ("a", "b")}
    candidate = choose_birth_candidate(state, active, probe, None, jax.random.key(3), 4)
    assert candidate is not None and candidate.site == "b" and candidate.slot == 2
    assert jnp.abs(candidate.direction @ jnp.array([1.0, 0.0, 0.0, 0.0])) > 0.999
    # The input state is immutable: the scratch slot remains exact-null.
    assert jnp.all(state.decomposition.components.site("b")[1][2] == 0.0)


def test_choose_birth_candidate_returns_none_for_zero_gradient() -> None:
    state = _state()

    def zero_probe(state, _batch, _key, _active, _protected):
        return jnp.zeros(()), jax.tree.map(jnp.zeros_like, state.decomposition.components)

    active = {site: jnp.array([True, True, False, False, False]) for site in ("a", "b")}
    assert choose_birth_candidate(state, active, zero_probe, None, jax.random.key(4), 2) is None


def test_choose_birth_batch_recovers_stable_rank_and_rejects_split_noise() -> None:
    state = _state()
    G_train = {
        "a": jnp.diag(jnp.array([3.0, 2.0, 0.5]))[jnp.array([0, 1, 2, 2])],
        "b": jnp.diag(jnp.array([4.0, 1.0, 0.2]))[jnp.array([0, 1, 2, 2])],
    }
    # The whole b block points uphill on validation, while a transfers as descent.
    # Rejection is blockwise: individual SVD coordinates are not identifiable.
    G_validation = dict(G_train)
    G_validation["b"] = -G_train["b"]

    def probe(state, batch, _key, _active, _protected):
        G = G_train if batch == "train" else G_validation
        grads = {}
        for site, (V, U) in state.decomposition.components.sites_items():
            grads[site] = (G[site] @ U.T, V.T @ G[site])
        return jnp.zeros(()), component_stacks_from_sites(grads)

    active = {site: jnp.array([True, True, False, False, False]) for site in ("a", "b")}
    candidate = choose_birth_batch(
        state,
        active,
        probe,
        "train",
        jax.random.key(10),
        (("validation", jax.random.key(11)),),
        max_slots_per_site=3,
        n_power_iters=4,
    )
    assert candidate is not None
    by_site = {site.site: site for site in candidate.sites}
    assert candidate.size == 3
    assert len(by_site["a"].slots) == 3
    assert "b" not in by_site
    for site in candidate.sites:
        assert jnp.allclose(
            site.directions.T @ site.directions, jnp.eye(len(site.slots)), atol=1e-5
        )
        assert jnp.all(site.validation_cosines > 0.0)
    # Scratch probes never mutate the input state.
    for site in ("a", "b"):
        assert jnp.all(state.decomposition.components.site(site)[1][2:] == 0.0)


def test_block_transfer_referee_is_invariant_to_internal_basis_rotation() -> None:
    # A degenerate training block has no preferred coordinate frame. In one legal frame
    # an individual-diagonal referee rejects the first coordinate; after a 45° rotation
    # it accepts both. The actual whole-block transfer is positive and unchanged.
    training = jnp.eye(2)
    validation = jnp.array([[[-1.0, 0.0], [0.0, 2.0]]])
    rotation = jnp.array([[1.0, -1.0], [1.0, 1.0]]) / jnp.sqrt(2.0)

    coordinate_scores = jnp.diag(validation[0])
    rotated_validation = jnp.einsum("ij,njk->nik", rotation.T, validation)
    rotated_validation = jnp.einsum("nij,jk->nik", rotated_validation, rotation)
    rotated_scores = jnp.diag(rotated_validation[0])
    assert jnp.any(coordinate_scores < 0.0)
    assert jnp.all(rotated_scores > 0.0)

    expected = _block_transfer_cosines(training, validation)
    rotated = _block_transfer_cosines(
        rotation.T @ training @ rotation,
        rotated_validation,
    )
    assert expected[0] > 0.0
    assert jnp.allclose(rotated, expected, atol=1e-6)
