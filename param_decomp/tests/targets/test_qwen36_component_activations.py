"""CPU oracle tests for `Qwen36MoeDecomposedModel.component_activation_forward`.

An offline consumer reads each requested site's ``x @ V`` off one clean forward. The
dense sites — the eleven token-mixer projections and the shared expert — emit full
`[B, T, C]` arrays against their captured site inputs (the normed mixer input, the mixer
core, the normed MoE input, the shared hidden); expert sites emit `SelectedCI` bundles on
the forward's routing, computed by the model's expert implementation — job space
(routed) or every expert then gathered (dense) — and each arm is compared against a
per-token dense gather oracle (`V[ids]` einsums over the same captured taps and routing),
so the plumbing (schedule, grouped matmuls, unsort; the all-expert contraction and slot
gather; each kind's V stack slot) is pinned to the arithmetic it claims."""

import jax
import jax.numpy as jnp
import numpy as np

from param_decomp.core.components import (
    BlockSelection,
    ComponentStacks,
    SelectedCI,
    init_component_stacks,
)
from param_decomp.lm.batch import LMBatch
from param_decomp.targets.qwen36_moe import (
    Qwen36MoeDecomposedModel,
    full_site_cs,
    layers_of_kind,
    parse_site_name,
    qwen36_moe_site_specs,
    site_name,
)
from param_decomp.targets.testing import (
    TINY_QWEN36_ALL_CS,
    tiny_qwen36_cfg,
    tiny_qwen36_decomposed_model,
)
from param_decomp.targets.transformer_taps import (
    attention_input_tap_key,
    attention_output_tap_key,
    mlp_input_tap_key,
    resid_tap_key,
    site_output_tap_key,
)

B, T = 2, 12


def _model_and_vu(key: jax.Array) -> tuple[Qwen36MoeDecomposedModel, ComponentStacks]:
    cfg = tiny_qwen36_cfg()
    sites = qwen36_moe_site_specs(cfg, full_site_cs(cfg, TINY_QWEN36_ALL_CS))
    model_key, vu_key = jax.random.split(key)
    return tiny_qwen36_decomposed_model(cfg, sites, model_key), init_component_stacks(sites, vu_key)


def _oracle_taps_and_routing(
    model: Qwen36MoeDecomposedModel, tokens: jax.Array
) -> tuple[dict[str, jax.Array], BlockSelection]:
    cfg = model.cfg
    keys = frozenset(
        key
        for layer in range(cfg.n_layer)
        for key in (
            attention_input_tap_key(layer),
            attention_output_tap_key(layer),
            mlp_input_tap_key(layer),
            site_output_tap_key(site_name(layer, "shared_gate")),
            site_output_tap_key(site_name(layer, "shared_up")),
        )
    )
    clean = model.clean_forward(LMBatch(tokens), keys, placement=None)
    return dict(clean.captures), clean.conditioning.selection


def _expected_site_activation(
    model: Qwen36MoeDecomposedModel,
    prepared: dict[str, dict[str, jax.Array]],
    taps: dict[str, jax.Array],
    routing: BlockSelection,
    site: str,
) -> jax.Array:
    """The per-token oracle for one site's x @ V on the captured site input (expert
    sites at the narrow `[B, T, k·c]` layout, slot m the pinned m-th expert)."""
    cfg = model.cfg
    layer, kind = parse_site_name(site)
    V = prepared[kind]["V"][layers_of_kind(cfg, kind).index(layer)]
    match kind:
        case (
            "gdn_q"
            | "gdn_k"
            | "gdn_v"
            | "gdn_z"
            | "gdn_b"
            | "gdn_a"
            | "attn_q"
            | "attn_k"
            | "attn_v"
        ):
            return taps[attention_input_tap_key(layer)] @ V
        case "gdn_out" | "attn_o":
            return taps[attention_output_tap_key(layer)] @ V
        case "shared_gate" | "shared_up":
            return taps[mlp_input_tap_key(layer)] @ V
        case "shared_down":
            gate = taps[site_output_tap_key(site_name(layer, "shared_gate"))]
            up = taps[site_output_tap_key(site_name(layer, "shared_up"))]
            return (jax.nn.silu(gate) * up) @ V
        case "experts_gate" | "experts_up":
            ids = routing.indices[layer]
            h2 = taps[mlp_input_tap_key(layer)]
            slot_values = jnp.einsum("btd,btkdc->btkc", h2, V[ids])
            return slot_values.reshape(B, T, -1)
        case "experts_down":
            ids = routing.indices[layer]
            h2 = taps[mlp_input_tap_key(layer)]
            weights = routing.weights[layer]
            di = cfg.moe_intermediate
            gate_blocks = model.moe.experts_gate[layer].reshape(cfg.n_experts, di, cfg.n_embd)
            up_blocks = model.moe.experts_up[layer].reshape(cfg.n_experts, di, cfg.n_embd)
            gate = jnp.einsum("btd,btkid->btki", h2, gate_blocks[ids])
            up = jnp.einsum("btd,btkid->btki", h2, up_blocks[ids])
            hidden = jax.nn.silu(gate) * up * weights[..., None]
            return jnp.einsum("btki,btkic->btkc", hidden, V[ids]).reshape(B, T, -1)
        case _:
            raise AssertionError(kind)


def test_component_activations_match_the_dense_gather_oracle():
    model, vu = _model_and_vu(jax.random.PRNGKey(0))
    cfg = model.cfg
    tokens = jax.random.randint(jax.random.PRNGKey(1), (B, T), 0, cfg.vocab_size)
    prepared = model.prepare_compute_weights(vu, None)
    taps, routing = _oracle_taps_and_routing(model, tokens)

    forward, activations = model.component_activation_forward(
        prepared,
        LMBatch(tokens),
        sites=model.site_names,
        capture_keys=frozenset(),
        placement=None,
    )

    np.testing.assert_array_equal(forward.conditioning.selection.indices, routing.indices)
    assert set(activations) == set(model.site_names)
    for spec in model.sites:
        layer, kind = parse_site_name(spec.name)
        expected = _expected_site_activation(model, prepared.per_kind, taps, routing, spec.name)
        value = activations[spec.name]
        if kind.startswith("experts_"):
            assert isinstance(value, SelectedCI), spec.name
            assert value.n_blocks == cfg.n_experts
            np.testing.assert_array_equal(value.block_indices, routing.indices[layer])
            np.testing.assert_allclose(value.values, expected, rtol=2e-4, atol=2e-5)
        else:
            assert isinstance(value, jax.Array), spec.name
            assert value.shape == (B, T, spec.C)
            np.testing.assert_allclose(value, expected, rtol=2e-4, atol=2e-5)


def test_component_activation_forward_returns_only_requested_sites_and_captures():
    model, vu = _model_and_vu(jax.random.PRNGKey(2))
    cfg = model.cfg
    tokens = jax.random.randint(jax.random.PRNGKey(3), (B, T), 0, cfg.vocab_size)
    prepared = model.prepare_compute_weights(vu, None)
    requested_capture = resid_tap_key(0)
    sites = (site_name(0, "experts_gate"), site_name(1, "attn_o"), site_name(1, "shared_down"))

    forward, activations = model.component_activation_forward(
        prepared,
        LMBatch(tokens),
        sites=sites,
        capture_keys=frozenset({requested_capture}),
        placement=None,
    )

    assert tuple(activations) == sites
    assert set(forward.captures) == {requested_capture}
    clean = model.clean_forward(LMBatch(tokens), placement=None)
    np.testing.assert_array_equal(forward.output, clean.output)
