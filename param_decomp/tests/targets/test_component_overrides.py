"""Component activation overrides on the dense transformer's materialized masked forward:
an overridden entry carries its value in place of `(x@V)·m`, so under the pinned delta a
site computes `W x + Σ (v − (x@V)_c) U_c` over its overridden components."""

import jax
import jax.numpy as jnp
import pytest

from param_decomp.core.components import init_component_stacks
from param_decomp.core.linear_plan import uniform_like
from param_decomp.core.model import (
    CaptureKeys,
    ComponentOverrides,
    ForwardResult,
    Masking,
    MaterializedMasking,
    SiteOverrides,
    StochasticMasking,
)
from param_decomp.lm.batch import LMBatchWithDocuments
from param_decomp.targets.lm_output import LMOutput
from param_decomp.targets.testing import materialized_logits, tiny_glu_cfg, tiny_glu_decomposed_lm
from param_decomp.targets.transformer import (
    TransformerDecomposedModel,
    TransformerPreparedWeights,
    _StochasticSiteMask,
    glu_site_specs,
    mlp_family_site_cs,
    parse_site_name,
)

B, T, C = 2, 16, 8


class _Harness:
    def __init__(self) -> None:
        cfg = tiny_glu_cfg()
        sites = glu_site_specs(cfg, mlp_family_site_cs(3, 5, C))
        self.model: TransformerDecomposedModel = tiny_glu_decomposed_lm(
            cfg, sites, jax.random.PRNGKey(0)
        )
        components = init_component_stacks(sites, jax.random.PRNGKey(1))
        self.weights: TransformerPreparedWeights = self.model.prepare_compute_weights(
            components, None
        )
        tokens = jax.random.randint(jax.random.PRNGKey(2), (B, T), 0, cfg.vocab_size)
        self.inputs = LMBatchWithDocuments.from_unsegmented_sequences(tokens)
        self.delta_pinned = MaterializedMasking(
            component_masks={site: jnp.ones((B, T, C)) for site in self.model.site_names},
            weight_delta_masks={site: jnp.ones((B, T)) for site in self.model.site_names},
        )

    def forward(
        self, masking: Masking, capture_keys: CaptureKeys
    ) -> ForwardResult[LMOutput, LMBatchWithDocuments]:
        return self.model.masked_forward(
            self.weights,
            self.inputs,
            masking=self.model.prepare_masking(masking),
            routes=None,
            placement=None,
            capture_keys=capture_keys,
            remat=False,
        )

    def overridden(
        self, masking: Masking, overrides: SiteOverrides, capture_keys: CaptureKeys
    ) -> ForwardResult[LMOutput, LMBatchWithDocuments]:
        return self.model.overridden_forward(
            self.weights,
            self.inputs,
            masking=self.model.prepare_masking(masking),
            overrides=overrides,
            placement=None,
            capture_keys=capture_keys,
            remat=False,
        )

    def site_io_keys(self, site: str) -> tuple[str, str]:
        (output_key,) = self.model.site_output_keys((site,))
        return self.model.anatomy.site_input_key(site), output_key

    def expected_site_output(
        self, site: str, site_input: jax.Array, overrides: ComponentOverrides
    ) -> jax.Array:
        """`W x + Σ (v − (x@V)_c) U_c` over the overridden entries."""
        layer, kind = parse_site_name(site)
        V = self.weights.per_kind[kind]["V"][layer]
        U = self.weights.per_kind[kind]["U"][layer]
        rows = tuple(overrides.indices.T)
        replaced = jnp.zeros((B, T, C)).at[rows].set(overrides.values - (site_input @ V)[rows])
        return site_input @ self.model.frozen_site_weight(site).T + replaced @ U


@pytest.fixture(scope="module")
def harness() -> _Harness:
    return _Harness()


def _random_overrides(key: jax.Array, n: int) -> ComponentOverrides:
    """`n` distinct waist entries with standard-normal values."""
    flat_key, values_key = jax.random.split(key)
    flat = jax.random.choice(flat_key, B * T * C, (n,), replace=False)
    indices = jnp.stack(jnp.unravel_index(flat, (B, T, C)), axis=-1)
    return ComponentOverrides(indices=indices, values=jax.random.normal(values_key, (n,)))


def test_replaying_live_activations_is_bitwise_the_delta_pinned_forward(harness: _Harness):
    live = harness.model.masked_component_activations(
        harness.weights, harness.inputs, harness.delta_pinned, placement=None
    )
    every_entry = jnp.indices((B, T, C)).reshape(3, -1).T
    replay = {
        site: ComponentOverrides(indices=every_entry, values=activation.reshape(-1))
        for site, activation in live.items()
    }
    unoverridden = harness.forward(harness.delta_pinned, frozenset())
    replayed = harness.overridden(harness.delta_pinned, replay, frozenset())
    unoverridden, replayed = (materialized_logits(r.output) for r in (unoverridden, replayed))
    assert jnp.array_equal(replayed, unoverridden)


def test_zero_override_is_the_zero_mask_ablation(harness: _Harness):
    site = "layers.4.mlp.up_proj"
    overrides = _random_overrides(jax.random.PRNGKey(3), 12)
    zeroed = ComponentOverrides(indices=overrides.indices, values=jnp.zeros_like(overrides.values))
    masks = dict(harness.delta_pinned.component_masks)
    masks[site] = jnp.ones((B, T, C)).at[tuple(overrides.indices.T)].set(0.0)
    ablated = MaterializedMasking(
        component_masks=masks, weight_delta_masks=harness.delta_pinned.weight_delta_masks
    )
    assert jnp.array_equal(
        materialized_logits(
            harness.overridden(harness.delta_pinned, {site: zeroed}, frozenset()).output
        ),
        materialized_logits(harness.forward(ablated, frozenset()).output),
    )


def test_value_override_replaces_the_site_output_by_hand(harness: _Harness):
    site = "layers.4.mlp.down_proj"
    overrides = _random_overrides(jax.random.PRNGKey(4), 20)
    input_key, output_key = harness.site_io_keys(site)
    captures = harness.overridden(
        harness.delta_pinned, {site: overrides}, frozenset((input_key, output_key))
    ).captures
    expected = harness.expected_site_output(site, captures[input_key], overrides)
    assert jnp.allclose(captures[output_key], expected, atol=1e-5)
    unoverridden = harness.forward(harness.delta_pinned, frozenset((output_key,))).captures
    assert not jnp.allclose(captures[output_key], unoverridden[output_key], atol=1e-2)


def test_later_override_reads_inputs_edited_by_an_earlier_one(harness: _Harness):
    earlier, later = "layers.3.mlp.down_proj", "layers.5.mlp.gate_proj"
    later_overrides = _random_overrides(jax.random.PRNGKey(5), 20)
    input_key, output_key = harness.site_io_keys(later)
    overrides = {earlier: _random_overrides(jax.random.PRNGKey(6), 64), later: later_overrides}
    captures = harness.overridden(
        harness.delta_pinned, overrides, frozenset((input_key, output_key))
    ).captures
    unedited = harness.forward(harness.delta_pinned, frozenset((input_key,))).captures
    assert not jnp.allclose(captures[input_key], unedited[input_key], atol=1e-2)
    expected = harness.expected_site_output(later, captures[input_key], later_overrides)
    assert jnp.allclose(captures[output_key], expected, atol=1e-5)


def test_overrides_on_stochastic_masks_match_the_same_masks_materialized(harness: _Harness):
    ci = {
        site: jax.random.uniform(key, (B, T, C))
        for site, key in zip(
            harness.model.site_names,
            jax.random.split(jax.random.PRNGKey(7), len(harness.model.site_names)),
            strict=True,
        )
    }
    stochastic = StochasticMasking(ci=ci, draw_key=jax.random.PRNGKey(8))
    drawn = harness.model.prepare_masking(stochastic).per_kind
    component_masks, delta_masks = {}, {}
    for site in harness.model.site_names:
        layer, kind = parse_site_name(site)
        entry = drawn[kind]
        assert isinstance(entry, _StochasticSiteMask)
        site_ci = entry.ci[layer]
        component_masks[site] = site_ci + (1.0 - site_ci) * uniform_like(
            entry.src_key[layer], site_ci
        )
        delta_masks[site] = uniform_like(entry.delta_key[layer], site_ci, drop_last_axis=True)
    same_draw = MaterializedMasking(component_masks=component_masks, weight_delta_masks=delta_masks)
    overrides = {
        "layers.3.mlp.up_proj": _random_overrides(jax.random.PRNGKey(9), 24),
        "layers.5.mlp.down_proj": _random_overrides(jax.random.PRNGKey(10), 24),
    }

    def overridden_logits(masking: Masking) -> jax.Array:
        return materialized_logits(harness.overridden(masking, overrides, frozenset()).output)

    overridden = overridden_logits(stochastic)
    assert jnp.allclose(overridden, overridden_logits(same_draw), atol=1e-5)
    unoverridden = materialized_logits(harness.forward(stochastic, frozenset()).output)
    assert not jnp.allclose(overridden, unoverridden, atol=1e-2)
