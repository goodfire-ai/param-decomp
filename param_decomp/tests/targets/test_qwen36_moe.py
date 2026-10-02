"""CPU tests for the qwen36_moe target at a tiny config.

The `DecomposedModel` contract over the hybrid DeltaNet/attention + MoE engine:
mask=1 + delta=1 reconstructs the clean forward, ablation changes logits, route=False
takes the frozen path, the stochastic masked forward runs and differentiates, the fused
gate/up taps carry the scattered routed semantics, the routing verbs and taps spell the
released router from one place, emission is fixed per kind at the target boundary, the
pinned routing is validated, and the whole-grid coverage contract fails closed. The
heart is the ROUTED-vs-DENSE decomposed parity block: the production routed decomposed
expert arm (job-space compute over the pinned experts) against a per-layer DENSE
reference written in this module — over the fused all-expert matrices with
the routing folded into the down input — at every layer of the masked forward, and per
layer in values AND gradients (dV/dU, d-masks/CI, d-delta, d-input), materialized and
stochastic, deltas/routes on and off. Direct HF parity lives in
`tests/qwen36_moe_hf_parity/`.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array

from param_decomp.core.components import (
    BlockSelection,
    ComponentStacks,
    SelectedCI,
    SiteC,
    SiteCI,
    SiteDims,
    init_component_stacks,
)
from param_decomp.core.decomposed_linear import BlockedSiteWeights, blocked_site_forward
from param_decomp.core.linear_plan import BlockContraction
from param_decomp.core.model import (
    MaterializedMasking,
    StochasticMasking,
    site_weight_delta,
)
from param_decomp.lm.batch import LMBatch, LMBatchWithRouting
from param_decomp.targets.qwen36_moe import (
    KIND_ORDER,
    FrozenMoE,
    Qwen36MoeConfig,
    Qwen36MoeDecomposedModel,
    _entry_masking,
    _fold_routing_weights,
    _StagedKind,
    _StagedMaterialized,
    _StagedStochastic,
    canonical_site_cs,
    expert_mixing_weights,
    full_site_cs,
    is_expert_kind,
    layer_is_full_attention,
    nonlinearity_aligned_component_initializer,
    parse_site_name,
    qwen36_35b_a3b_config,
    qwen36_moe_site_specs,
    router_probs,
    router_probs_tap_key,
    router_weights_tap_key,
    select_experts,
    site_contraction,
    site_dims,
    site_name,
)
from param_decomp.targets.testing import (
    TINY_QWEN36_CS,
    capture_clean,
    constant_mask_values,
    identity_masking,
    materialized_logits,
    random_mask_values,
    run_masked,
    site_masks,
    tiny_qwen36_cfg,
    tiny_qwen36_decomposed_model,
)
from param_decomp.targets.transformer_taps import (
    mlp_input_tap_key,
    resid_tap_key,
    site_output_tap_key,
)
from param_decomp.tests.core.test_selected_ci import _scatter_to_full


def _full_model_and_vu(key: jax.Array) -> tuple[Qwen36MoeDecomposedModel, ComponentStacks]:
    cfg = tiny_qwen36_cfg()
    sites = qwen36_moe_site_specs(cfg, full_site_cs(cfg, TINY_QWEN36_CS))
    model_key, vu_key = jax.random.split(key)
    model = tiny_qwen36_decomposed_model(cfg, sites, model_key)
    return model, init_component_stacks(sites, vu_key)


def test_nonlinearity_aligned_init_is_exact_and_expert_local():
    cfg = tiny_qwen36_cfg()
    cs = {
        "experts_gate": cfg.n_experts * cfg.moe_intermediate,
        "experts_up": cfg.n_experts * cfg.moe_intermediate,
        "experts_down": cfg.n_experts * cfg.moe_intermediate,
        "shared_gate": cfg.shared_expert_intermediate,
        "shared_up": cfg.shared_expert_intermediate,
        "shared_down": cfg.shared_expert_intermediate,
    }
    sites = qwen36_moe_site_specs(cfg, full_site_cs(cfg, cs))
    model = tiny_qwen36_decomposed_model(cfg, sites, jax.random.PRNGKey(0))

    components = nonlinearity_aligned_component_initializer(model, jax.random.PRNGKey(1))

    for delta in model.weight_deltas(components).values():
        assert jnp.array_equal(delta, jnp.zeros_like(delta))
    for name, site in components.sites_items():
        _layer, kind = parse_site_name(name)
        if kind.startswith("experts_"):
            assert site.V.shape[0] == cfg.n_experts
            assert site.U.shape[0] == cfg.n_experts
        assert jnp.all(jnp.linalg.norm(site.V, axis=-2) > 0)
        assert jnp.all(jnp.linalg.norm(site.U, axis=-1) > 0)


def _tokens(cfg: Qwen36MoeConfig) -> jax.Array:
    return jax.random.randint(jax.random.PRNGKey(7), (2, 12), 0, cfg.vocab_size)


def _mlp_input_keys(cfg: Qwen36MoeConfig) -> frozenset[str]:
    return frozenset(mlp_input_tap_key(layer) for layer in range(cfg.n_layer))


def _routing_tap_keys(cfg: Qwen36MoeConfig) -> frozenset[str]:
    return frozenset(
        key
        for layer in range(cfg.n_layer)
        for key in (router_probs_tap_key(layer), router_weights_tap_key(layer))
    )


def _selected_experts(indices: Array, n_experts: int) -> Array:
    """Which experts each position ran (`[*lead, E]` bool) under a pinned routing."""
    return jnp.any(jax.nn.one_hot(indices, n_experts, dtype=bool), axis=-2)


def _moe_layer(model: Qwen36MoeDecomposedModel, layer: int) -> FrozenMoE:
    """One layer's frozen MoE weights out of the layer-stacked `moe` leaves."""
    return jax.tree.map(lambda leaf: leaf[layer], model.moe)


def _assert_values_close(
    actual: Array | np.ndarray, desired: Array | np.ndarray, err_msg: str = ""
) -> None:
    """fp32 CPU: the compared spellings compute identical math and differ only by
    reassociation — grouped per-expert blocks with fp32 job sums (the routed arms; the
    frozen arm weights its down OUTPUT in the fp32 combine, the decomposed arm folds the
    weight into the down INPUT) vs one dense contraction — compounding through the
    residual stream."""
    np.testing.assert_allclose(actual, desired, rtol=2e-4, atol=2e-4, err_msg=err_msg)


def _assert_close_across_programs(
    actual: Array | np.ndarray, desired: Array | np.ndarray, err_msg: str = ""
) -> None:
    """A routing tap against the same fp32 arithmetic run as a separate XLA program (the
    routing verbs re-applied to a captured input, or another forward's program): the two
    may reassociate the router matmul's reduction differently (XLA CPU vectorizes it per
    the host ISA), and the softmax carries that last-ulp logit difference into the
    probabilities and the weights gathered from them."""
    np.testing.assert_allclose(actual, desired, rtol=1e-5, atol=1e-6, err_msg=err_msg)


# ----------------------------- the DecomposedModel contract -----------------------------


def test_mask_one_delta_one_reconstructs_clean_forward():
    model, vu = _full_model_and_vu(jax.random.PRNGKey(0))
    tokens = _tokens(model.cfg)
    batch = LMBatch(tokens)
    clean = model.clean_forward(batch, placement=None)
    assert clean.conditioning.batch is batch
    assert clean.sequence is None
    prepared = model.prepare_compute_weights(vu, None)
    masked = model.masked_forward(
        prepared,
        clean.conditioning,
        masking=model.prepare_masking(identity_masking(model, clean.conditioning.selection)),
        routes=None,
        placement=None,
        remat=False,
    )
    assert masked.conditioning is clean.conditioning
    assert masked.sequence is None
    _assert_values_close(materialized_logits(masked.output), materialized_logits(clean.output))


def test_zero_masks_change_logits():
    model, vu = _full_model_and_vu(jax.random.PRNGKey(1))
    tokens = _tokens(model.cfg)
    clean = model.clean_forward(LMBatch(tokens), placement=None)
    prepared = model.prepare_compute_weights(vu, None)
    masks = site_masks(
        model, clean.conditioning.selection, constant_mask_values(model, tokens.shape, 0.0)
    )
    ablated = materialized_logits(
        run_masked(
            model,
            prepared,
            clean.conditioning,
            MaterializedMasking(component_masks=masks, weight_delta_masks=None),
            remat=False,
            routes=None,
        )
    )
    assert np.abs(np.asarray(ablated - materialized_logits(clean.output))).max() > 1e-3


def test_route_false_takes_frozen_path():
    model, vu = _full_model_and_vu(jax.random.PRNGKey(2))
    tokens = _tokens(model.cfg)
    clean = model.clean_forward(LMBatch(tokens), placement=None)
    prepared = model.prepare_compute_weights(vu, None)
    masks = site_masks(
        model,
        clean.conditioning.selection,
        random_mask_values(model, tokens.shape, jax.random.PRNGKey(3)),
    )
    routes = {spec.name: jnp.zeros(tokens.shape, bool) for spec in model.sites}
    routed_off = materialized_logits(
        run_masked(
            model,
            prepared,
            clean.conditioning,
            MaterializedMasking(component_masks=masks, weight_delta_masks=None),
            routes=routes,
            remat=False,
        )
    )
    _assert_values_close(routed_off, materialized_logits(clean.output))


def test_stochastic_masked_forward_runs_and_differentiates():
    model, vu = _full_model_and_vu(jax.random.PRNGKey(4))
    tokens = _tokens(model.cfg)
    clean = model.clean_forward(LMBatch(tokens), placement=None)
    ci = site_masks(
        model, clean.conditioning.selection, constant_mask_values(model, tokens.shape, 0.5)
    )
    masking = StochasticMasking(ci=ci, draw_key=jax.random.PRNGKey(5))

    def loss(components: ComponentStacks) -> jax.Array:
        prepared = model.prepare_compute_weights(components, None)
        masked = model.masked_forward(
            prepared,
            clean.conditioning,
            masking=model.prepare_masking(masking),
            routes=None,
            placement=None,
            remat=True,
        )
        return model.recon_loss_fn(masked.output, clean.output)

    value, grads = eqx.filter_value_and_grad(loss)(vu)
    assert jnp.isfinite(value)
    leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
    assert leaves and all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
    assert any(jnp.any(leaf != 0) for leaf in leaves)


def test_captures_cover_the_closed_vocabulary():
    model, _vu = _full_model_and_vu(jax.random.PRNGKey(6))
    cfg = model.cfg
    tokens = _tokens(cfg)
    second, mid, last = 1, cfg.n_layer // 2, cfg.n_layer - 1
    gate_key = site_output_tap_key(site_name(mid, "experts_gate"))
    keys = (
        resid_tap_key(0),
        resid_tap_key(second),
        resid_tap_key(cfg.n_layer),
        f"mlp_in.{mid}",
        f"mlp_in.{last}",
        gate_key,
        site_output_tap_key(site_name(last, "experts_down")),
        site_output_tap_key(site_name(second, "shared_down")),
        router_probs_tap_key(mid),
        router_weights_tap_key(last),
    )
    clean = model.clean_forward(LMBatch(tokens), frozenset(keys), placement=None)
    captures = clean.captures
    assert set(captures) == set(keys)
    np.testing.assert_array_equal(captures[resid_tap_key(0)], model.embed[tokens])
    fused = cfg.n_experts * cfg.moe_intermediate
    assert captures[gate_key].shape == (*tokens.shape, fused)
    assert captures[f"mlp_in.{last}"].shape == (*tokens.shape, cfg.n_embd)
    probs_tap, weights_tap = (
        captures[router_probs_tap_key(mid)],
        captures[router_weights_tap_key(last)],
    )
    assert probs_tap.shape == (*tokens.shape, cfg.n_experts) and probs_tap.dtype == jnp.float32
    assert weights_tap.shape == (*tokens.shape, cfg.n_experts_per_token)
    assert weights_tap.dtype == jnp.float32

    for unknown in ("attn_unknown.0", "router_idx.0"):
        with pytest.raises(AssertionError):
            capture_clean(model, LMBatch(tokens), (unknown,))


def test_routed_frozen_fused_taps_scatter_selected_experts():
    """The clean (routed-frozen) fused gate/up taps hold the scattered job results:
    selected experts match the dense matmul to reduction-order tolerance, unselected
    experts are EXACTLY zero (the routed arm never computes them)."""
    model, _vu = _full_model_and_vu(jax.random.PRNGKey(12))
    cfg = model.cfg
    tokens = _tokens(cfg)
    layer = cfg.n_layer - 1
    keys = (
        mlp_input_tap_key(layer),
        site_output_tap_key(site_name(layer, "experts_gate")),
        site_output_tap_key(site_name(layer, "experts_up")),
    )
    clean = model.clean_forward(LMBatch(tokens), frozenset(keys), placement=None)
    h2 = clean.captures[mlp_input_tap_key(layer)]
    selected = _selected_experts(clean.conditioning.selection.indices[layer], cfg.n_experts)
    selected_wide = np.asarray(jnp.repeat(selected, cfg.moe_intermediate, axis=-1))
    for kind, frozen_fused in (
        ("experts_gate", model.moe.experts_gate[layer]),
        ("experts_up", model.moe.experts_up[layer]),
    ):
        tap = np.asarray(clean.captures[site_output_tap_key(site_name(layer, kind))])
        dense = np.asarray(h2 @ frozen_fused.T)
        np.testing.assert_allclose(tap[selected_wide], dense[selected_wide], rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(tap[~selected_wide], 0.0)


def test_masked_grads_flow_through_routed_frozen_experts():
    """Decomposing only the shared kinds leaves the expert arm routed-frozen inside the
    MASKED forward: gradients must flow through the routed compute (the custom-VJP
    gathers, under remat) to the shared V/U."""
    cfg = tiny_qwen36_cfg()
    shared_cs = {kind: c for kind, c in TINY_QWEN36_CS.items() if kind.startswith("shared_")}
    sites = qwen36_moe_site_specs(cfg, full_site_cs(cfg, shared_cs))
    model = tiny_qwen36_decomposed_model(cfg, sites, jax.random.PRNGKey(13))
    vu = init_component_stacks(sites, jax.random.PRNGKey(14))
    tokens = _tokens(cfg)
    clean = model.clean_forward(LMBatch(tokens), placement=None)
    ci = {spec.name: jnp.full((*tokens.shape, spec.C), 0.5) for spec in model.sites}
    masking = StochasticMasking(ci=ci, draw_key=jax.random.PRNGKey(15))

    def loss(components: ComponentStacks) -> jax.Array:
        prepared = model.prepare_compute_weights(components, None)
        masked = model.masked_forward(
            prepared,
            clean.conditioning,
            masking=model.prepare_masking(masking),
            routes=None,
            placement=None,
            remat=True,
        )
        return model.recon_loss_fn(masked.output, clean.output)

    value, grads = eqx.filter_value_and_grad(loss)(vu)
    assert jnp.isfinite(value)
    leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
    assert leaves and all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
    assert any(jnp.any(leaf != 0) for leaf in leaves)


# ----------------------------- routing: verbs, taps, pinning -----------------------------


def test_clean_routing_taps_spell_the_router_verbs():
    """The clean forward's routing — its pinned `BlockSelection` and the `router_probs` /
    `router_weights` taps — is the three verbs composed on the captured MoE input:
    `router_probs` (fp32 softmax over all experts), `select_experts` (top-k, descending),
    `expert_mixing_weights` (gathered, renormalized to sum 1). The expert selection
    agrees exactly; the probabilities and weights to program-to-program fp32 parity."""
    model, _vu = _full_model_and_vu(jax.random.PRNGKey(31))
    cfg = model.cfg
    tokens = _tokens(cfg)
    k = cfg.n_experts_per_token
    clean = model.clean_forward(
        LMBatch(tokens),
        _mlp_input_keys(cfg) | _routing_tap_keys(cfg),
        placement=None,
    )
    routing = clean.conditioning
    assert (
        routing.selection.indices.shape
        == routing.selection.weights.shape
        == (cfg.n_layer, *tokens.shape, k)
    )
    assert (
        routing.selection.indices.dtype == jnp.int32
        and routing.selection.weights.dtype == jnp.float32
    )
    for layer in range(cfg.n_layer):
        probs = router_probs(model.moe.router[layer], clean.captures[mlp_input_tap_key(layer)])
        _assert_close_across_programs(clean.captures[router_probs_tap_key(layer)], probs)
        indices = select_experts(probs, k)
        np.testing.assert_array_equal(routing.selection.indices[layer], indices)
        _assert_close_across_programs(
            routing.selection.weights[layer], expert_mixing_weights(probs, indices)
        )
        np.testing.assert_array_equal(
            clean.captures[router_weights_tap_key(layer)], routing.selection.weights[layer]
        )
        selected = jnp.take_along_axis(probs, indices, axis=-1)
        assert bool(jnp.all(jnp.diff(selected, axis=-1) <= 0)), layer
    np.testing.assert_allclose(routing.selection.weights.sum(-1), 1.0, rtol=1e-6)


def test_identity_masked_routing_taps_reproduce_the_clean_router():
    """Under the exact identity (all-ones masks, delta≡1) the masked residual reproduces
    the clean one up to fp32 reassociation, so the masked router's softmax and its
    mixing weights recomputed at the pinned indices reproduce the clean taps."""
    model, vu = _full_model_and_vu(jax.random.PRNGKey(32))
    cfg = model.cfg
    tokens = _tokens(cfg)
    keys = _routing_tap_keys(cfg)
    clean = model.clean_forward(LMBatch(tokens), keys, placement=None)
    prepared = model.prepare_compute_weights(vu, None)
    masked = model.masked_forward(
        prepared,
        clean.conditioning,
        masking=model.prepare_masking(identity_masking(model, clean.conditioning.selection)),
        routes=None,
        placement=None,
        capture_keys=keys,
        remat=False,
    )
    for key in sorted(keys):
        _assert_values_close(masked.captures[key], clean.captures[key], err_msg=key)


def test_perturbed_masked_forward_reweights_the_pinned_experts():
    """Ablating every site (zero masks, delta channel off) perturbs the residual below
    every layer but the first: the masked router's softmax moves while the experts stay
    pinned — the masked `router_weights` tap is exactly the masked softmax gathered at
    the CLEAN indices and renormalized, not the clean weights — and the result carries
    the routing it was given."""
    model, vu = _full_model_and_vu(jax.random.PRNGKey(33))
    cfg = model.cfg
    tokens = _tokens(cfg)
    first, last = 0, cfg.n_layer - 1
    keys = frozenset(
        key
        for layer in (first, last)
        for key in (router_probs_tap_key(layer), router_weights_tap_key(layer))
    )
    clean = model.clean_forward(LMBatch(tokens), keys, placement=None)
    prepared = model.prepare_compute_weights(vu, None)
    masks = site_masks(
        model, clean.conditioning.selection, constant_mask_values(model, tokens.shape, 0.0)
    )
    masked = model.masked_forward(
        prepared,
        clean.conditioning,
        masking=model.prepare_masking(MaterializedMasking(component_masks=masks)),
        routes=None,
        placement=None,
        capture_keys=keys,
        remat=False,
    )
    _assert_close_across_programs(
        masked.captures[router_probs_tap_key(first)], clean.captures[router_probs_tap_key(first)]
    )
    masked_probs = masked.captures[router_probs_tap_key(last)]
    clean_probs = clean.captures[router_probs_tap_key(last)]
    assert np.abs(np.asarray(masked_probs - clean_probs)).max() > 1e-3
    masked_weights = expert_mixing_weights(masked_probs, clean.conditioning.selection.indices[last])
    np.testing.assert_array_equal(masked.captures[router_weights_tap_key(last)], masked_weights)
    assert (
        np.abs(np.asarray(masked_weights - clean.conditioning.selection.weights[last])).max() > 1e-3
    )
    np.testing.assert_array_equal(
        masked.conditioning.selection.indices, clean.conditioning.selection.indices
    )
    np.testing.assert_array_equal(
        masked.conditioning.selection.weights, clean.conditioning.selection.weights
    )


def test_masked_forward_refuses_a_mismatched_pinned_routing():
    model, vu = _full_model_and_vu(jax.random.PRNGKey(34))
    tokens = _tokens(model.cfg)
    routing = model.clean_forward(LMBatch(tokens), placement=None).conditioning
    prepared = model.prepare_compute_weights(vu, None)
    masking = identity_masking(model, routing.selection)

    def run(conditioning: BlockSelection) -> None:
        model.masked_forward(
            prepared,
            LMBatchWithRouting(LMBatch(tokens), conditioning),
            masking=model.prepare_masking(masking),
            routes=None,
            placement=None,
            remat=False,
        )

    with pytest.raises(AssertionError, match="pinned routing indices"):
        run(BlockSelection(routing.selection.indices[1:], routing.selection.weights[1:]))
    with pytest.raises(AssertionError, match="pinned routing indices"):
        run(BlockSelection(routing.selection.indices[..., :1], routing.selection.weights[..., :1]))
    with pytest.raises(AssertionError, match="pinned routing weights"):
        run(
            BlockSelection(
                routing.selection.indices, routing.selection.weights.astype(jnp.bfloat16)
            )
        )


def test_kind_emission_is_asserted_at_the_target_boundary():
    """Expert kinds take `SelectedCI`, shared kinds full `[.., C]` arrays: the other
    emission dies naming the kind — at materialized-mask attachment and at the stacked
    CI's attachment (the stochastic recipe)."""
    model, vu = _full_model_and_vu(jax.random.PRNGKey(35))
    cfg = model.cfg
    tokens = _tokens(cfg)
    routing = model.clean_forward(LMBatch(tokens), placement=None).conditioning
    prepared = model.prepare_compute_weights(vu, None)
    masks = identity_masking(model, routing.selection).component_masks
    full_on_expert: dict[str, SiteCI] = {
        **masks,
        site_name(0, "experts_gate"): jnp.ones((*tokens.shape, TINY_QWEN36_CS["experts_gate"])),
    }
    narrow_on_shared: dict[str, SiteCI] = {
        **masks,
        site_name(0, "shared_gate"): SelectedCI(
            jnp.ones((*tokens.shape, cfg.n_experts_per_token)),
            routing.selection.indices[0],
            cfg.n_experts,
        ),
    }
    for wrong_emission, kind in (
        (full_on_expert, "experts_gate"),
        (narrow_on_shared, "shared_gate"),
    ):
        with pytest.raises(AssertionError, match=kind):
            model.masked_forward(
                prepared,
                routing,
                masking=model.prepare_masking(MaterializedMasking(component_masks=wrong_emission)),
                routes=None,
                placement=None,
                remat=False,
            )
    # the stacked recipes attach one emission per KIND: every layer of the kind wrong
    for kind in ("experts_gate", "shared_gate"):
        wrong_kind: dict[str, SiteCI] = {
            **masks,
            **{
                site_name(layer, kind): (
                    full_on_expert if kind == "experts_gate" else narrow_on_shared
                )[site_name(0, kind)]
                for layer in range(cfg.n_layer)
            },
        }
        with pytest.raises(AssertionError, match=kind):
            model.masked_forward(
                prepared,
                routing,
                masking=model.prepare_masking(
                    StochasticMasking(
                        ci=wrong_kind,
                        draw_key=jax.random.PRNGKey(36),
                    )
                ),
                routes=None,
                placement=None,
                remat=False,
            )
    with pytest.raises(AssertionError, match="mixes payload structures"):
        model.prepare_stochastic_masking(full_on_expert)


# ----------------------------- routed vs dense parity -----------------------------


def _assert_grad_leaves_close(routed_grads: object, dense_grads: object) -> None:
    """Per-leaf relative Frobenius error: reassociation noise concentrates on near-zero
    ELEMENTS, so an elementwise bound punishes exactly the entries that carry no
    signal; a wrong or missing term lands O(1) on this metric, not 1e-3. A leaf the
    expert arm structurally never reaches is zero in both spellings."""
    for routed_leaf, dense_leaf in zip(
        jax.tree.leaves(routed_grads), jax.tree.leaves(dense_grads), strict=True
    ):
        routed_np, dense_np = np.asarray(routed_leaf), np.asarray(dense_leaf)
        denom = np.linalg.norm(dense_np)
        error = np.linalg.norm(routed_np - dense_np) / (denom if denom > 0 else 1.0)
        assert error < 2e-3, error


@dataclass(frozen=True)
class _DenseExperts:
    """The dense reference's outputs for one layer: gate/up over EVERY expert at the
    fused width (`[*lead, E·di]`) and the routing-weighted down output (`[*lead, d]`)."""

    gate: Array
    up: Array
    down: Array


def _dense_experts_reference(
    cfg: Qwen36MoeConfig,
    moe: FrozenMoE,
    h2: Array,
    indices: Array,
    weights: Array,
    per_kind: Mapping[str, _StagedKind],
) -> _DenseExperts:
    """Compute a dense reference for one layer's routed expert decomposition.

    Every expert runs. Decomposed kinds use `blocked_site_forward` with scattered
    component masks; undecomposed kinds use frozen matmuls. Routing weights are
    folded into the down-projection input, matching the routed implementation."""
    n_experts, di = cfg.n_experts, cfg.moe_intermediate
    lead = h2.shape[:-1]

    def site(kind: str, x: Array, fused_weight: Array, contraction: BlockContraction) -> Array:
        entry = per_kind.get(kind)
        if entry is None:
            return x @ fused_weight.T
        V, U = entry.v, entry.u
        mask, delta_mask, route = _entry_masking(entry)
        assert isinstance(mask, SelectedCI), kind
        return blocked_site_forward(
            x,
            BlockedSiteWeights(fused_weight, V, U, None, None),
            _scatter_to_full(mask),
            delta_mask,
            route,
            contraction,
        ).output

    gate = site("experts_gate", h2, moe.experts_gate, "fused_output")
    up = site("experts_up", h2, moe.experts_up, "fused_output")
    dense_routing = jnp.sum(
        jax.nn.one_hot(indices, n_experts, dtype=weights.dtype) * weights[..., None], axis=-2
    )
    down_input = _fold_routing_weights(
        gate.reshape(*lead, n_experts, di),
        up.reshape(*lead, n_experts, di),
        dense_routing[..., None],
    ).reshape(*lead, n_experts * di)
    down = site("experts_down", down_input, moe.experts_down, "fused_input")
    return _DenseExperts(gate=gate, up=up, down=down)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _LayerMasking:
    """One layer's DIFFERENTIABLE decomposed-expert inputs — per decomposed expert kind
    its V/U blocks, its narrow mask values (`[*lead, k·c]`, slot m the pinned m-th
    expert), and its per-position deltas when the delta channel is on."""

    vu: dict[str, tuple[Array, Array]]
    mask_values: dict[str, Array]
    deltas: dict[str, Array] | None


def _layer_masking(
    prepared: Mapping[str, Mapping[str, Array]],
    masks: Mapping[str, SiteCI],
    deltas: Mapping[str, Array] | None,
    layer: int,
) -> _LayerMasking:
    """The decomposed expert kinds' inputs at one layer, sliced out of a whole-forward's
    per-kind stacks and per-site masks."""
    kinds = [kind for kind in prepared if is_expert_kind(kind)]

    def selected_values(kind: str) -> Array:
        mask = masks[site_name(layer, kind)]
        assert isinstance(mask, SelectedCI), kind
        return mask.values

    return _LayerMasking(
        vu={kind: (prepared[kind]["V"][layer], prepared[kind]["U"][layer]) for kind in kinds},
        mask_values={kind: selected_values(kind) for kind in kinds},
        deltas=None if deltas is None else {kind: deltas[site_name(layer, kind)] for kind in kinds},
    )


def _layer_routes(
    masking: _LayerMasking, routes: Mapping[str, Array] | None, layer: int
) -> dict[str, Array] | None:
    if routes is None:
        return None
    return {kind: routes[site_name(layer, kind)] for kind in masking.vu}


type _PerKindEntries = dict[str, _StagedKind]


def _materialized_entries(
    model: Qwen36MoeDecomposedModel,
    masking: _LayerMasking,
    indices: Array,
    routes: Mapping[str, Array] | None,
) -> _PerKindEntries:
    """One layer's per-kind entries in the shape `_attach_materialized` hands the stage
    body — V/U, the `SelectedCI` mask on the pinned indices, and the delta/route channels
    when present."""
    return {
        kind: _StagedKind(
            V,
            U,
            _StagedMaterialized(
                SelectedCI(masking.mask_values[kind], indices, model.cfg.n_experts),
                None if masking.deltas is None else masking.deltas[kind],
            ),
            None if routes is None else routes[kind],
        )
        for kind, (V, U) in masking.vu.items()
    }


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _PinnedLayer:
    """The stochastic entries builder's context: the whole forward's pinned routing (the
    recipe prepares whole per-kind stacks) and the layer whose
    slice the arm runs."""

    routing: BlockSelection
    layer: int = field(metadata=dict(static=True))


def _routed_down(
    model: Qwen36MoeDecomposedModel,
    moe: FrozenMoE,
    h2: Array,
    indices: Array,
    weights: Array,
    per_kind: _PerKindEntries,
) -> Array:
    _gate, _up, down = model._routed_decomposed_experts(
        moe, h2, indices, weights, per_kind, frozenset(), "ragged_dot"
    )
    return down


def _dense_down(
    model: Qwen36MoeDecomposedModel,
    moe: FrozenMoE,
    h2: Array,
    indices: Array,
    weights: Array,
    per_kind: _PerKindEntries,
) -> Array:
    return _dense_experts_reference(model.cfg, moe, h2, indices, weights, per_kind).down


def _down_and_grads[MaskingT, ContextT](
    arm: Callable[
        [Qwen36MoeDecomposedModel, FrozenMoE, Array, Array, Array, _PerKindEntries], Array
    ],
    entries: Callable[[Qwen36MoeDecomposedModel, MaskingT, Array, ContextT], _PerKindEntries],
):
    """`(model, moe, h2, masking, indices, weights, context) -> (grads w.r.t. (h2,
    masking), down)` for one arm fed by one entries builder: `entries(model, masking,
    indices, context)` produces the per-kind entries for that layer; `masking`
    contains the differentiable inputs and `context` supplies the rest. Jitted once, shared by every layer and test (the model rides as a traced arg)."""

    def loss(
        model: Qwen36MoeDecomposedModel,
        moe: FrozenMoE,
        h2: Array,
        masking: MaskingT,
        indices: Array,
        weights: Array,
        context: ContextT,
    ) -> tuple[Array, Array]:
        per_kind = entries(model, masking, indices, context)
        down = arm(model, moe, h2, indices, weights, per_kind)
        return jnp.sum(jnp.cos(down)), down

    return jax.jit(jax.grad(loss, argnums=(2, 3), has_aux=True))


_ROUTED_DOWN_AND_GRADS = _down_and_grads(_routed_down, _materialized_entries)
_DENSE_DOWN_AND_GRADS = _down_and_grads(_dense_down, _materialized_entries)


def _assert_routed_matches_dense(
    model: Qwen36MoeDecomposedModel,
    vu: ComponentStacks,
    tokens: Array,
    mask_values: Mapping[str, Array],
    deltas: Mapping[str, Array] | None,
    routes: Mapping[str, Array] | None,
) -> None:
    """The parity block. (1) Through the FULL masked forward: at every layer, the routed
    arm's `experts_down.out` against the dense reference fed that same forward's
    `mlp_in` / `router_weights` taps — so each layer is an oracle whatever the layers
    below did to its input — and the fused gate tap holds the reference's selected
    experts with exact zeros elsewhere. (2) Per layer, the routed arm called directly on
    the clean layer input against the dense reference: values and gradients w.r.t. the
    input, V/U, the mask values, and the deltas (a size-1 broadcast delta or route axis
    gets the cross-lead sum both ways)."""
    cfg = model.cfg
    clean = model.clean_forward(LMBatch(tokens), _mlp_input_keys(cfg), placement=None)
    routing = clean.conditioning
    masks = site_masks(model, routing.selection, mask_values)
    prepared = model.prepare_compute_weights(vu, None)
    keys = frozenset(
        key
        for layer in range(cfg.n_layer)
        for key in (
            mlp_input_tap_key(layer),
            router_weights_tap_key(layer),
            site_output_tap_key(site_name(layer, "experts_gate")),
            site_output_tap_key(site_name(layer, "experts_down")),
        )
    )
    masked = model.masked_forward(
        prepared,
        routing,
        masking=model.prepare_masking(
            MaterializedMasking(component_masks=masks, weight_delta_masks=deltas)
        ),
        routes=routes,
        placement=None,
        capture_keys=keys,
        remat=False,
    )
    for layer in range(cfg.n_layer):
        moe = _moe_layer(model, layer)
        masking = _layer_masking(prepared.per_kind, masks, deltas, layer)
        layer_routes = _layer_routes(masking, routes, layer)
        indices = routing.selection.indices[layer]

        per_kind = _materialized_entries(model, masking, indices, layer_routes)
        reference = _dense_experts_reference(
            cfg,
            moe,
            masked.captures[mlp_input_tap_key(layer)],
            indices,
            masked.captures[router_weights_tap_key(layer)],
            per_kind,
        )
        down_tap = masked.captures[site_output_tap_key(site_name(layer, "experts_down"))]
        _assert_values_close(down_tap, reference.down, err_msg=f"experts_down at layer {layer}")
        selected_wide = np.asarray(
            jnp.repeat(_selected_experts(indices, cfg.n_experts), cfg.moe_intermediate, axis=-1)
        )
        gate_tap = np.asarray(
            masked.captures[site_output_tap_key(site_name(layer, "experts_gate"))]
        )
        np.testing.assert_array_equal(gate_tap[~selected_wide], 0.0)
        _assert_values_close(
            gate_tap[selected_wide],
            np.asarray(reference.gate)[selected_wide],
            err_msg=f"experts_gate at layer {layer}",
        )

        args = (
            model,
            moe,
            clean.captures[mlp_input_tap_key(layer)],
            masking,
            indices,
            routing.selection.weights[layer],
            layer_routes,
        )
        routed_grads, routed_down = _ROUTED_DOWN_AND_GRADS(*args)
        dense_grads, dense_down = _DENSE_DOWN_AND_GRADS(*args)
        _assert_values_close(routed_down, dense_down, err_msg=f"layer {layer}")
        _assert_grad_leaves_close(routed_grads, dense_grads)


@pytest.mark.parametrize(
    ("with_deltas", "with_routes"), [(True, True), (True, False), (False, False)]
)
def test_routed_decomposed_experts_match_the_dense_reference(with_deltas: bool, with_routes: bool):
    """The heart of the training math: the routed decomposed arm against the dense
    reference, with the delta channel and routes on and off."""
    model, vu = _full_model_and_vu(jax.random.PRNGKey(20))
    tokens = _tokens(model.cfg)
    mask_values = random_mask_values(model, tokens.shape, jax.random.PRNGKey(21))
    deltas = (
        {spec.name: jnp.full(tokens.shape, 0.25) for spec in model.sites} if with_deltas else None
    )
    routes = (
        {spec.name: (jnp.arange(tokens.size).reshape(tokens.shape) % 3 > 0) for spec in model.sites}
        if with_routes
        else None
    )
    _assert_routed_matches_dense(model, vu, tokens, mask_values, deltas, routes)


def test_routed_decomposed_experts_match_the_dense_reference_with_broadcast_masking():
    """Delta masks and routes may carry size-1 broadcast lead axes (batch-shared
    persistent sources): the routed arm broadcasts them to the full lead
    before its job gathers — the shape the `sc` sources trace."""
    model, vu = _full_model_and_vu(jax.random.PRNGKey(28))
    tokens = _tokens(model.cfg)
    mask_values = random_mask_values(model, tokens.shape, jax.random.PRNGKey(29))
    deltas = {
        spec.name: jax.random.uniform(jax.random.PRNGKey(30), (1, tokens.shape[1]))
        for spec in model.sites
    }
    routes = {spec.name: (jnp.arange(tokens.shape[1])[None, :] % 2 > 0) for spec in model.sites}
    _assert_routed_matches_dense(model, vu, tokens, mask_values, deltas, routes)


def test_routed_decomposed_experts_match_the_dense_reference_with_undecomposed_expert_kinds():
    """Decomposing a strict subset of the expert kinds (gate + down; up stays frozen)
    runs the frozen grouped matmuls inside the routed decomposed arm — parity against
    the reference's mixed execution."""
    cfg = tiny_qwen36_cfg()
    partial_cs = {k: c for k, c in TINY_QWEN36_CS.items() if k != "experts_up"}
    sites = qwen36_moe_site_specs(cfg, full_site_cs(cfg, partial_cs))
    model = tiny_qwen36_decomposed_model(cfg, sites, jax.random.PRNGKey(25))
    vu = init_component_stacks(sites, jax.random.PRNGKey(26))
    tokens = _tokens(cfg)
    mask_values = random_mask_values(model, tokens.shape, jax.random.PRNGKey(27))
    deltas = {spec.name: jnp.full(tokens.shape, 0.5) for spec in sites}
    _assert_routed_matches_dense(model, vu, tokens, mask_values, deltas, None)


def test_routed_decomposed_experts_match_the_dense_reference_under_stochastic_masking():
    """The in-stage stochastic rebuild through the routed arm against the dense
    reference consuming the SAME `src_key` entry: `_entry_masking` draws the source at
    the narrow shape and the per-position delta from one key in both spellings, so the
    realization is identical and the arms differ by fp32 reassociation only — values and
    gradients (dV/dU, d-CI, d-input) per layer."""
    model, vu = _full_model_and_vu(jax.random.PRNGKey(22))
    cfg = model.cfg
    tokens = _tokens(cfg)
    clean = model.clean_forward(LMBatch(tokens), _mlp_input_keys(cfg), placement=None)
    prepared = model.prepare_compute_weights(vu, None)
    ci_values = random_mask_values(model, tokens.shape, jax.random.PRNGKey(23))
    draw_key = jax.random.PRNGKey(24)

    def stochastic_entries(
        m: Qwen36MoeDecomposedModel,
        inputs: tuple[dict[str, dict[str, Array]], dict[str, Array]],
        _indices: Array,
        conditioning: _PinnedLayer,
    ) -> _PerKindEntries:
        """One layer's per-kind entries in the shape masked execution hands the stage
        body, sliced out of the whole forward's attached stacks — differentiable in the
        prepared V/U stacks and the CI values."""
        prepared_weights, ci = inputs
        masking = StochasticMasking(
            ci=site_masks(m, conditioning.routing, ci),
            draw_key=draw_key,
        )
        prepared_masking = m.prepare_masking(masking)
        attached = {
            kind: _StagedKind(prepared_weights[kind]["V"], prepared_weights[kind]["U"], entry, None)
            for kind, entry in prepared_masking.per_kind.items()
        }
        return jax.tree.map(lambda a: a[conditioning.layer], attached)

    routed_down_and_grads = _down_and_grads(_routed_down, stochastic_entries)
    dense_down_and_grads = _down_and_grads(_dense_down, stochastic_entries)
    for layer in range(cfg.n_layer):
        args = (
            model,
            _moe_layer(model, layer),
            clean.captures[mlp_input_tap_key(layer)],
            (prepared.per_kind, ci_values),
            clean.conditioning.selection.indices[layer],
            clean.conditioning.selection.weights[layer],
            _PinnedLayer(clean.conditioning.selection, layer),
        )
        routed_grads, routed_down = routed_down_and_grads(*args)
        dense_grads, dense_down = dense_down_and_grads(*args)
        _assert_values_close(routed_down, dense_down, err_msg=f"layer {layer}")
        _assert_grad_leaves_close(routed_grads, dense_grads)


# ----------------------------- the rest of the target surface -----------------------------


def test_weight_deltas_and_norms_are_slot_aligned():
    model, vu = _full_model_and_vu(jax.random.PRNGKey(9))
    cfg = model.cfg
    deltas = model.weight_deltas(vu)
    norms = model.target_weight_sq_norms()
    assert set(deltas) == set(norms) == set(TINY_QWEN36_CS)
    for kind in TINY_QWEN36_CS:
        match site_contraction(kind):
            case None:
                dims = site_dims(cfg, kind)
                assert deltas[kind].shape == (cfg.n_layer, dims.d_out, dims.d_in)
            case "fused_output":
                assert deltas[kind].shape == (
                    cfg.n_layer,
                    cfg.n_experts,
                    cfg.moe_intermediate,
                    cfg.n_embd,
                )
            case "fused_input":
                assert deltas[kind].shape == (
                    cfg.n_layer,
                    cfg.n_experts,
                    cfg.n_embd,
                    cfg.moe_intermediate,
                )
        assert norms[kind].shape == (cfg.n_layer,)
    name = site_name(3, "shared_gate")
    site = vu.site(name)
    expected = model.moe.shared_gate[3].astype(jnp.float32) - (site.V @ site.U).T
    np.testing.assert_allclose(site_weight_delta(deltas, vu, name), expected, rtol=1e-6)
    gate_name = site_name(2, "experts_gate")
    gate = vu.site(gate_name)  # V [E, d, c], U [E, c, di]
    frozen_blocks = model.moe.experts_gate[2].reshape(
        cfg.n_experts, cfg.moe_intermediate, cfg.n_embd
    )
    expected_blocks = frozen_blocks.astype(jnp.float32) - jnp.einsum(
        "eic,eco->eoi", gate.V.astype(jnp.float32), gate.U.astype(jnp.float32)
    )
    np.testing.assert_allclose(site_weight_delta(deltas, vu, gate_name), expected_blocks, rtol=1e-6)


def test_weight_deltas_convert_after_relayout_bit_for_bit():
    """The frozen stack converts to fp32 only once it holds the blocked layout, so the
    resident bf16 bytes are never doubled whole. The reference here converts the whole
    resident stack FIRST and relayouts the fp32 copy: the convert commutes exactly with
    reshape and transpose, so the two orders agree bit for bit."""
    model, vu = _full_model_and_vu(jax.random.PRNGKey(12))
    cfg = model.cfg
    model = eqx.tree_at(
        lambda m: m.moe, model, jax.tree.map(lambda a: a.astype(jnp.bfloat16), model.moe)
    )
    deltas = model.weight_deltas(vu)
    for kind, (Vs, Us) in vu.stacks.items():
        frozen32 = getattr(model.moe, kind).astype(jnp.float32)
        v32, u32 = Vs.astype(jnp.float32), Us.astype(jnp.float32)
        match site_contraction(kind):
            case None:
                expected = frozen32 - jnp.einsum("gic,gco->goi", v32, u32)
            case "fused_output":
                blocks = frozen32.reshape(
                    cfg.n_layer, cfg.n_experts, cfg.moe_intermediate, cfg.n_embd
                )
                expected = blocks - jnp.einsum("geic,geco->geoi", v32, u32)
            case "fused_input":
                blocks = frozen32.reshape(
                    cfg.n_layer, cfg.n_embd, cfg.n_experts, cfg.moe_intermediate
                ).transpose(0, 2, 1, 3)
                expected = blocks - jnp.einsum("geic,geco->geoi", v32, u32)
        np.testing.assert_array_equal(deltas[kind], expected, err_msg=kind)


def test_partial_layer_coverage_is_refused():
    cfg = tiny_qwen36_cfg()
    partial = tuple(
        SiteC(site_name(layer, kind), TINY_QWEN36_CS[kind])
        for layer in range(cfg.n_layer // 2)
        for kind in TINY_QWEN36_CS
    )
    sites = qwen36_moe_site_specs(cfg, partial)
    model = tiny_qwen36_decomposed_model(cfg, sites, jax.random.PRNGKey(10))
    vu = init_component_stacks(sites, jax.random.PRNGKey(11))
    with pytest.raises(AssertionError, match="EVERY layer"):
        model.prepare_compute_weights(vu, None)


def test_family_grammar_round_trips_and_fails_closed():
    for layer in (0, 7, 39):
        for kind in KIND_ORDER:
            assert parse_site_name(site_name(layer, kind)) == (layer, kind)
    for bad in (
        "layers.0.self_attn.unknown_proj",
        "layers.0.mlp.experts.q_proj",
        "layers.0.mlp.shared_expert_gate",
        "layers.0.mlp.gate_proj",
    ):
        with pytest.raises(AssertionError):
            parse_site_name(bad)
    shuffled = full_site_cs(tiny_qwen36_cfg(), TINY_QWEN36_CS)[::-1]
    assert canonical_site_cs(shuffled) == full_site_cs(tiny_qwen36_cfg(), TINY_QWEN36_CS)


def test_released_qwen36_architecture():
    cfg = qwen36_35b_a3b_config()
    assert (cfg.n_layer, cfg.n_stages, cfg.rotary_dim) == (40, 10, 64)
    assert site_dims(cfg, "experts_gate") == SiteDims(d_in=2048, d_out=131072)
    assert site_dims(cfg, "experts_down") == SiteDims(d_in=131072, d_out=2048)
    assert site_dims(cfg, "shared_up") == SiteDims(d_in=2048, d_out=512)
    assert layer_is_full_attention(cfg, 39) and not layer_is_full_attention(cfg, 38)
    assert sum(layer_is_full_attention(cfg, i) for i in range(cfg.n_layer)) == 10


def test_routing_fold_rounds_the_fused_down_input_once():
    """bf16 gate/up with fp32 routing weights: the fold is the fp32 product rounded to
    bf16 exactly once (weights rounded to bf16 first, or a bf16 product chain, land off
    that value on some rows)."""
    key_gate, key_up, key_weights = jax.random.split(jax.random.PRNGKey(0), 3)
    gate = jax.random.normal(key_gate, (256, 32)).astype(jnp.bfloat16)
    up = jax.random.normal(key_up, (256, 32)).astype(jnp.bfloat16)
    weights = jax.nn.softmax(jax.random.normal(key_weights, (256, 1)), axis=0) * 256

    folded = _fold_routing_weights(gate, up, weights)
    assert folded.dtype == jnp.bfloat16
    exact = jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32) * weights
    np.testing.assert_array_equal(np.asarray(folded), np.asarray(exact.astype(jnp.bfloat16)))


@pytest.mark.parametrize("n_vocab_chunks", [1, 4])
def test_abstract_model_constructs_the_authored_streamed_output(n_vocab_chunks: int):
    from param_decomp.targets.lm_output import StreamedLinearOutput, StreamedOutputEdge
    from param_decomp.targets.qwen36_moe import abstract_qwen36_moe_model

    cfg = tiny_qwen36_cfg()
    edge = StreamedOutputEdge(n_vocab_chunks=n_vocab_chunks)
    model = abstract_qwen36_moe_model(cfg, (), jnp.float32, "ragged_dot", edge, "xla")
    assert model.output_edge == edge
    assert model.attn.attn.implementation == "xla"
    assert all(isinstance(leaf, jax.ShapeDtypeStruct) for leaf in jax.tree.leaves(model))
    output = eqx.filter_eval_shape(
        lambda target, tokens: target.clean_forward(LMBatch(tokens), placement=None).output,
        model,
        jax.ShapeDtypeStruct((2, 4), jnp.int32),
    )
    assert isinstance(output, StreamedLinearOutput)
    assert output.n_chunks == n_vocab_chunks
    assert output.activations.shape == (2, 4, cfg.n_embd)
    assert output.head.shape == (cfg.vocab_size, cfg.n_embd)


def test_stochastic_draws_share_prepared_selected_ci():
    model, _vu = _full_model_and_vu(jax.random.key(31))
    tokens = _tokens(model.cfg)
    clean = model.clean_forward(LMBatch(tokens), placement=None)
    values = random_mask_values(model, tokens.shape, jax.random.key(32))
    ci = site_masks(model, clean.conditioning.selection, values)
    draw = model.prepare_stochastic_masking(ci)
    first = draw(jax.random.key(33))
    second = draw(jax.random.key(34))
    for kind, entry in first.per_kind.items():
        left_recipe, right_recipe = entry, second.per_kind[kind]
        assert isinstance(left_recipe, _StagedStochastic)
        assert isinstance(right_recipe, _StagedStochastic)
        for left, right in zip(
            jax.tree.leaves(left_recipe.ci),
            jax.tree.leaves(right_recipe.ci),
            strict=True,
        ):
            assert left is right
        assert not jnp.array_equal(left_recipe.src_key, right_recipe.src_key)
