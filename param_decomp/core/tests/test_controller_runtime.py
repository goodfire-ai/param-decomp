import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from param_decomp.core.ci_fn import LayerwiseMLPCIArch, build_ci_fn
from param_decomp.core.components import SiteSpec, component_stacks_from_sites
from param_decomp.core.controller_runtime import choose_birth_candidate
from param_decomp.core.train import Decomposition, TrainingItem, TrainState


def _state() -> TrainState:
    sites = (SiteSpec("a", 4, 3, 3), SiteSpec("b", 4, 3, 3))
    vu = {
        site.name: (
            jax.random.normal(jax.random.fold_in(jax.random.key(0), i), (4, 3)),
            jax.random.normal(jax.random.fold_in(jax.random.key(1), i), (3, 3)).at[2].set(0),
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

    active = {site: jnp.array([True, True, False]) for site in ("a", "b")}
    candidate = choose_birth_candidate(state, active, probe, None, jax.random.key(3), 4)
    assert candidate is not None and candidate.site == "b" and candidate.slot == 2
    assert jnp.abs(candidate.direction @ jnp.array([1.0, 0.0, 0.0, 0.0])) > 0.999
    # The input state is immutable: the scratch slot remains exact-null.
    assert jnp.all(state.decomposition.components.site("b")[1][2] == 0.0)


def test_choose_birth_candidate_returns_none_for_zero_gradient() -> None:
    state = _state()

    def zero_probe(state, _batch, _key, _active, _protected):
        return jnp.zeros(()), jax.tree.map(jnp.zeros_like, state.decomposition.components)

    active = {site: jnp.array([True, True, False]) for site in ("a", "b")}
    assert choose_birth_candidate(state, active, zero_probe, None, jax.random.key(4), 2) is None
