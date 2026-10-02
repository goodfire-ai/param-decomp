"""Reconstruction-grid RNG derivation is explicit and disjoint."""

import jax
import jax.numpy as jnp

from param_decomp.core.components import SelectedCI, site_ci_values
from param_decomp.core.masking import materialize_masking
from param_decomp.core.model import StochasticMasking
from param_decomp.core.recon import ReconLossTerm, StaticProbabilityRouting, StochasticSources
from param_decomp.core.runtime_schedule import RuntimeSchedule
from param_decomp.core.train import ReconGrid


def _term(name: str) -> ReconLossTerm[StochasticSources]:
    return ReconLossTerm(
        name=name,
        coeff=RuntimeSchedule.from_coeff(1.0),
        sample_routing=StaticProbabilityRouting(("site",), 0.5),
        sources=StochasticSources(),
        auxiliaries=(),
    )


def test_recon_grid_fold_in_chain_pins_term_offsets():
    """Target starts at 1; non-target starts after every target term."""
    key = jax.random.PRNGKey(17)
    target_terms = (_term("target-0"), _term("target-1"))
    grids = (
        ReconGrid(target_terms, key_offset=1),
        ReconGrid((_term("nontarget-0"),), key_offset=1 + len(target_terms)),
    )
    for grid in grids:
        for term_idx, draw in enumerate(grid.draws(key, {}, (64,))):
            term_key = jax.random.fold_in(key, grid.key_offset + term_idx)
            draw_key, routing_key = jax.random.split(term_key)
            assert jnp.array_equal(draw.key, draw_key)
            assert draw.routes is not None
            expected = jax.random.bernoulli(jax.random.fold_in(routing_key, 0), 0.5, (64,))
            assert jnp.array_equal(draw.routes["site"], expected)


def test_stochastic_materialization_preserves_site_key_order():
    selected = SelectedCI(
        jnp.full((2, 3, 4), 0.25),
        jnp.broadcast_to(jnp.array([2, 0]), (2, 3, 2)),
        n_blocks=4,
    )
    dense = jnp.full((2, 3, 5), 0.5)
    key = jax.random.key(17)
    masking = materialize_masking(StochasticMasking(ci={"z": selected, "a": dense}, draw_key=key))
    assert masking.weight_delta_masks is not None
    mask_key, delta_key = jax.random.split(key)
    for index, (name, values) in enumerate((("z", selected.values), ("a", dense))):
        source = jax.random.uniform(jax.random.fold_in(mask_key, index), values.shape)
        expected = values + (1.0 - values) * source
        assert jnp.array_equal(site_ci_values(masking.component_masks[name]), expected)
        expected_delta = jax.random.uniform(jax.random.fold_in(delta_key, index), (2, 3))
        assert jnp.array_equal(masking.weight_delta_masks[name], expected_delta)
    selected_mask = masking.component_masks["z"]
    assert isinstance(selected_mask, SelectedCI)
    assert jnp.array_equal(selected_mask.block_indices, selected.block_indices)
