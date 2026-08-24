"""SPEC R1: reconstruction-grid RNG derivation is explicit and disjoint."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from param_decomp.core.recon import ReconLossTerm, StochasticSources
from param_decomp.core.train import ReconGrid


def _key_routes(
    key: PRNGKeyArray, _leading: tuple[int, ...]
) -> tuple[dict[str, Array], dict[str, Array]]:
    """Expose the routing key through each of two draws so the whole chain is pinned."""
    routes = {"site": key}
    return routes, routes


def _term(name: str) -> ReconLossTerm[StochasticSources]:
    return ReconLossTerm(
        name=name,
        coeff=1.0,
        sample_routing=_key_routes,
        sources=StochasticSources(),
        hidden_acts_reconstruction=None,
    )


def test_recon_grid_fold_in_chain_pins_term_and_draw_offsets():
    """Target starts at 1; non-target starts after every target term (SPEC R1)."""
    key = jax.random.PRNGKey(17)
    target_terms = (_term("target-0"), _term("target-1"))
    grids = (
        ReconGrid.of(target_terms, key_offset=1),
        ReconGrid.of((_term("nontarget-0"),), key_offset=1 + len(target_terms)),
    )
    for grid in grids:
        draws_per_term = grid.draws(key, {}, (3,))
        for term_idx, draws in enumerate(draws_per_term):
            term_key = jax.random.fold_in(key, grid.key_offset + term_idx)
            term_draw_key, routing_key = jax.random.split(term_key)
            assert len(draws) == 2
            for draw_idx, (draw_key, routes) in enumerate(draws):
                assert jnp.array_equal(draw_key, jax.random.fold_in(term_draw_key, draw_idx))
                assert routes is not None
                assert jnp.array_equal(routes["site"], routing_key)
