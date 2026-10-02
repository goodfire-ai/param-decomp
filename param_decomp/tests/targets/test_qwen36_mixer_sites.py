"""CPU tests for the qwen36_moe token-mixer sites — the eleven DeltaNet / gated-attention
projections decomposed beside the six MoE kinds — against the model's own frozen forward
at tiny shapes.

Coverage lands on each kind's own layers (DeltaNet kinds on the linear-attention layers,
attention kinds on the full-attention layers, MoE on all) and every kind's V/U stack,
frozen stack, and delta stack read slot s = the s-th such layer; the nonlinearity-aligned init
is exact at width for every kind, the residual writers aligning on their input axis; the
`DecomposedModel` contract holds with all seventeen kinds decomposed (identity
reconstructs, ablating one mixer kind moves the logits, the stochastic forward
differentiates into every mixer's V/U); the MoE CI arch stays chunk-homogeneous with the
mixers' dense slots; the reconstruction-point audit sees the mixer taps; and the placed
masked forward matches the unplaced one with every kind decomposed.
"""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array

from param_decomp.core.ci_fn.implementations.block_selected.arch import (
    _block_selected_slot_signature,
)
from param_decomp.core.components import (
    ComponentStacks,
    SelectedCI,
    SiteCI,
    init_component_stacks,
    nonlinearity_alignments,
)
from param_decomp.core.init_placed import init_component_stacks_placed
from param_decomp.core.losses import nonlinearity_loss
from param_decomp.core.model import (
    ForwardResult,
    MaterializedMasking,
    PlacedModel,
    StochasticMasking,
)
from param_decomp.core.nonlinearity import (
    DeltaNetHeads,
    KVHeads,
    Neurons,
    NonlinearityUnitKind,
    QueryHeads,
)
from param_decomp.core.placement import from_config
from param_decomp.core.sharding import place_target
from param_decomp.core.tools.hlo_census import collective_census
from param_decomp.lm.batch import LMBatch, LMBatchWithRouting
from param_decomp.targets.lm_output import LMOutput
from param_decomp.targets.qwen36_moe import (
    KIND_ORDER,
    Qwen36MoeConfig,
    Qwen36MoeDecomposedModel,
    QwenPreparedMasking,
    QwenPreparedWeights,
    full_site_cs,
    is_expert_kind,
    layer_is_full_attention,
    layers_of_kind,
    nonlinearity_aligned_component_initializer,
    nonlinearity_alignment,
    parse_site_name,
    qwen36_moe_site_specs,
    site_dims,
    site_name,
    sublayer_of,
)
from param_decomp.targets.testing import (
    TINY_QWEN36_ALL_CS,
    TINY_QWEN36_CS,
    constant_mask_values,
    exact_width_cs,
    identity_masking,
    materialized_logits,
    random_mask_values,
    site_masks,
    tiny_qwen36_cfg,
    tiny_qwen36_decomposed_model,
    tiny_qwen36_moe_ci_fn,
    tiny_qwen36_moe_ci_fn_arch,
)
from param_decomp.targets.transformer_taps import (
    attention_output_tap_key,
    resid_tap_key,
    site_output_tap_key,
)
from param_decomp.tests.targets.qwen36_placement_helpers import (
    BATCH,
    CENSUS_CS,
    DATA,
    SEQ,
    TP,
    assert_no_weight_gather_in_any_loop,
    batch_placed,
    placement_mesh,
)
from param_decomp.tests.targets.test_qwen36_moe import _assert_values_close

MIXER_KINDS = tuple(kind for kind in KIND_ORDER if sublayer_of(kind) != "moe")
ROW_KINDS = ("gdn_out", "attn_o")
"""The mixers' residual writers: their components align on the input axis."""


def _tokens(cfg: Qwen36MoeConfig) -> Array:
    return jax.random.randint(jax.random.PRNGKey(7), (2, 12), 0, cfg.vocab_size)


def _model_and_vu(
    cfg: Qwen36MoeConfig, c_of: dict[str, int], key: Array
) -> tuple[Qwen36MoeDecomposedModel, ComponentStacks]:
    sites = qwen36_moe_site_specs(cfg, full_site_cs(cfg, c_of))
    model_key, vu_key = jax.random.split(key)
    return tiny_qwen36_decomposed_model(cfg, sites, model_key), init_component_stacks(sites, vu_key)


def _stored_weight(model: Qwen36MoeDecomposedModel, layer: int, kind: str) -> Array:
    """One site's frozen matrix `[d_out, d_in]`, read straight off the stage-blocked
    storage (`deltanet` leaves `[n_stages, interval−1, …]`, `attn` `[n_stages, …]`, `moe`
    `[n_layer, …]`) — the oracle for every stack that claims slot s = the s-th layer."""
    stage, position = divmod(layer, model.cfg.full_attention_interval)
    deltanet, attn = model.deltanet.mixer, model.attn.attn
    match kind:
        case "gdn_q":
            return deltanet.w_q[stage, position]
        case "gdn_k":
            return deltanet.w_k[stage, position]
        case "gdn_v":
            return deltanet.w_v[stage, position]
        case "gdn_z":
            return deltanet.w_z[stage, position]
        case "gdn_b":
            return deltanet.w_b[stage, position]
        case "gdn_a":
            return deltanet.w_a[stage, position]
        case "gdn_out":
            return deltanet.w_out[stage, position]
        case "attn_q":
            return attn.wq[stage]
        case "attn_k":
            return attn.wk[stage]
        case "attn_v":
            return attn.wv[stage]
        case "attn_o":
            return attn.wo[stage]
        case (
            "experts_gate"
            | "experts_up"
            | "experts_down"
            | "shared_gate"
            | "shared_up"
            | "shared_down"
        ):
            return getattr(model.moe, kind)[layer]
        case _:
            raise AssertionError(kind)


def _assert_one_hot_columns(matrix: Array) -> None:
    """Every column selects exactly one coordinate, and no two select the same."""
    columns = np.asarray(matrix)
    assert set(np.unique(columns)) <= {0.0, 1.0}, np.unique(columns)
    assert np.all(columns.sum(axis=0) == 1), columns.sum(axis=0)
    assert np.all(columns.sum(axis=1) <= 1), columns.sum(axis=1)


# ----------------------------- the nonlinearity-aligned init -----------------------------


def test_nonlinearity_aligned_init_is_exact_at_width_for_every_kind():
    """At exact width every component is one coordinate: zero deltas, nonempty
    factors, the column kinds' U a permutation of the identity on the output axis, the
    residual writers' V a permutation of the identity on the input axis with U's rows
    the rows of the right-mult matrix."""
    cfg = tiny_qwen36_cfg()
    model, _vu = _model_and_vu(cfg, exact_width_cs(cfg, KIND_ORDER), jax.random.PRNGKey(0))
    components = nonlinearity_aligned_component_initializer(model, jax.random.PRNGKey(1))

    for delta in model.weight_deltas(components).values():
        assert jnp.array_equal(delta, jnp.zeros_like(delta))
    for name, site in components.sites_items():
        layer, kind = parse_site_name(name)
        assert jnp.all(jnp.linalg.norm(site.V, axis=-2) > 0), name
        assert jnp.all(jnp.linalg.norm(site.U, axis=-1) > 0), name
        if kind not in MIXER_KINDS:
            continue
        right_mult = _stored_weight(model, layer, kind).astype(jnp.float32).T
        if kind in ROW_KINDS:
            _assert_one_hot_columns(site.V)
            np.testing.assert_array_equal(site.U, site.V.T @ right_mult, err_msg=name)
        else:
            _assert_one_hot_columns(site.U.T)
            np.testing.assert_array_equal(site.V, right_mult @ site.U.T, err_msg=name)


def test_nonlinearity_aligned_init_below_width_owns_distinct_output_coordinates():
    """Below width a column kind samples C distinct output coordinates whole: the delta
    vanishes exactly on the owned rows and is the frozen row elsewhere."""
    cfg = tiny_qwen36_cfg()
    kind = "gdn_v"
    c = site_dims(cfg, kind).d_out // 2
    model, _vu = _model_and_vu(cfg, {kind: c}, jax.random.PRNGKey(2))
    components = nonlinearity_aligned_component_initializer(model, jax.random.PRNGKey(3))
    deltas = model.weight_deltas(components)[kind]
    for slot, layer in enumerate(layers_of_kind(cfg, kind)):
        site = components.site(site_name(layer, kind))
        _assert_one_hot_columns(site.U.T)
        assert jnp.all(jnp.linalg.norm(site.V, axis=0) > 0)
        owned = np.asarray(jnp.any(site.U > 0, axis=0))
        assert owned.sum() == c
        weight = np.asarray(_stored_weight(model, layer, kind), dtype=np.float32)
        delta = np.asarray(deltas[slot])
        np.testing.assert_array_equal(delta[owned], 0.0)
        np.testing.assert_array_equal(delta[~owned], weight[~owned])


# ----------------------------- the DecomposedModel contract, all kinds -----------------------------


def test_identity_masking_reconstructs_clean_with_every_kind_decomposed():
    model, vu = _model_and_vu(tiny_qwen36_cfg(), TINY_QWEN36_ALL_CS, jax.random.PRNGKey(4))
    tokens = _tokens(model.cfg)
    clean = model.clean_forward(LMBatch(tokens), placement=None)
    masked = model.masked_forward(
        model.prepare_compute_weights(vu, None),
        clean.conditioning,
        masking=model.prepare_masking(identity_masking(model, clean.conditioning.selection)),
        routes=None,
        placement=None,
        remat=False,
    ).output
    _assert_values_close(materialized_logits(masked), materialized_logits(clean.output))


@pytest.mark.parametrize("kind", ["gdn_v", "attn_o"])
def test_ablating_one_mixer_kind_changes_logits(kind: str):
    model, vu = _model_and_vu(tiny_qwen36_cfg(), TINY_QWEN36_ALL_CS, jax.random.PRNGKey(5))
    tokens = _tokens(model.cfg)
    clean = model.clean_forward(LMBatch(tokens), placement=None)
    identity = identity_masking(model, clean.conditioning.selection)
    zeroed = {
        **identity.component_masks,
        **{
            site_name(layer, kind): jnp.zeros((*tokens.shape, TINY_QWEN36_ALL_CS[kind]))
            for layer in layers_of_kind(model.cfg, kind)
        },
    }
    ablated = model.masked_forward(
        model.prepare_compute_weights(vu, None),
        clean.conditioning,
        masking=model.prepare_masking(
            MaterializedMasking(
                component_masks=zeroed, weight_delta_masks=identity.weight_delta_masks
            )
        ),
        routes=None,
        placement=None,
        remat=False,
    ).output
    difference = materialized_logits(ablated) - materialized_logits(clean.output)
    assert np.abs(np.asarray(difference)).max() > 1e-3


def test_stochastic_masked_forward_differentiates_into_every_mixer():
    model, vu = _model_and_vu(tiny_qwen36_cfg(), TINY_QWEN36_ALL_CS, jax.random.PRNGKey(6))
    tokens = _tokens(model.cfg)
    clean = model.clean_forward(LMBatch(tokens), placement=None)
    ci = site_masks(
        model, clean.conditioning.selection, constant_mask_values(model, tokens.shape, 0.5)
    )
    masking = StochasticMasking(ci=ci, draw_key=jax.random.PRNGKey(8))

    def loss(components: ComponentStacks) -> Array:
        masked = model.masked_forward(
            model.prepare_compute_weights(components, None),
            clean.conditioning,
            masking=model.prepare_masking(masking),
            routes=None,
            placement=None,
            remat=True,
        )
        return model.recon_loss_fn(masked.output, clean.output)

    value, grads = eqx.filter_value_and_grad(loss)(vu)
    assert jnp.isfinite(value)
    for kind in MIXER_KINDS:
        for leaf in grads.stacks[kind]:
            assert jnp.all(jnp.isfinite(leaf)) and jnp.any(leaf != 0), kind


# ----------------------------- coverage and stack order -----------------------------


def test_coverage_lands_on_each_kinds_own_layers():
    cfg = tiny_qwen36_cfg()
    deltanet = tuple(
        layer for layer in range(cfg.n_layer) if not layer_is_full_attention(cfg, layer)
    )
    attention = tuple(layer for layer in range(cfg.n_layer) if layer_is_full_attention(cfg, layer))
    assert (deltanet, attention) == ((0, 2), (1, 3))
    covered: dict[str, list[int]] = {kind: [] for kind in KIND_ORDER}
    for site in full_site_cs(cfg, TINY_QWEN36_ALL_CS):
        layer, kind = parse_site_name(site.name)
        covered[kind].append(layer)
    for kind in KIND_ORDER:
        match sublayer_of(kind):
            case "deltanet":
                expected = deltanet
            case "attn":
                expected = attention
            case "moe":
                expected = tuple(range(cfg.n_layer))
        assert tuple(covered[kind]) == expected == layers_of_kind(cfg, kind), kind

    model, vu = _model_and_vu(cfg, TINY_QWEN36_ALL_CS, jax.random.PRNGKey(9))
    lengths = vu.group_lengths()
    assert lengths == {kind: len(layers_of_kind(cfg, kind)) for kind in KIND_ORDER}
    assert {lengths[kind] for kind in MIXER_KINDS} == {2}
    assert {lengths[kind] for kind in TINY_QWEN36_CS} == {4}
    model.prepare_compute_weights(vu, None)
    deltas = model.weight_deltas(vu)
    assert {kind: delta.shape[0] for kind, delta in deltas.items()} == lengths


def test_frozen_kind_stacks_read_layer_major():
    """Slot s of a kind's frozen stack, its delta stack, and its target norms is the
    s-th layer that has the kind — pinned at a three-layer interval, where a DeltaNet
    kind's `[n_stages, interval−1]` storage has a position axis to flatten."""
    cfg = replace(tiny_qwen36_cfg(), n_layer=6, full_attention_interval=3)
    assert layers_of_kind(cfg, "gdn_q") == (0, 1, 3, 4)
    model, vu = _model_and_vu(cfg, TINY_QWEN36_ALL_CS, jax.random.PRNGKey(10))
    norms = model.target_weight_sq_norms()
    deltas = model.weight_deltas(vu)
    for kind in KIND_ORDER:
        layers = layers_of_kind(cfg, kind)
        assert norms[kind].shape == (len(layers),) and deltas[kind].shape[0] == len(layers)
        for slot, layer in enumerate(layers):
            weight = _stored_weight(model, layer, kind).astype(jnp.float32)
            np.testing.assert_allclose(norms[kind][slot], jnp.sum(weight**2), rtol=1e-6)
            if kind in MIXER_KINDS:
                site = vu.site(site_name(layer, kind))
                np.testing.assert_allclose(
                    deltas[kind][slot], weight - (site.V @ site.U).T, rtol=1e-6, atol=1e-6
                )


# ----------------------------- the CI arch and the reconstruction points -----------------------------


def test_nonlinearity_partitions_are_the_chunks_attached_to_one_nonlinearity():
    """Each kind's declared chunk at the tiny fixture (2 key heads x 6, 4 value heads x 5,
    2 query heads x 16, 1 kv head): DeltaNet q/k partition by key head, each read by the
    two value heads that repeat it; v/z by value head, used once; b/a are already one
    coordinate per value head; attention q by query head, k/v by kv head read by both
    query heads; attention o reads query heads; GDN out reads value heads; MLP
    projections face the hidden neuron axis on their output or input side."""
    cfg = tiny_qwen36_cfg()
    expected = {
        "gdn_q": DeltaNetHeads(2, 2),
        "gdn_k": DeltaNetHeads(2, 2),
        "gdn_v": DeltaNetHeads(4, 1),
        "gdn_z": DeltaNetHeads(4, 1),
        "gdn_b": Neurons(),
        "gdn_a": Neurons(),
        "gdn_out": DeltaNetHeads(4, 1),
        "attn_q": QueryHeads(2),
        "attn_k": KVHeads(1, 2),
        "attn_v": KVHeads(1, 2),
        "attn_o": QueryHeads(2),
        "experts_gate": Neurons(),
        "experts_up": Neurons(),
        "experts_down": Neurons(),
        "shared_gate": Neurons(),
        "shared_up": Neurons(),
        "shared_down": Neurons(),
    }
    assert tuple(expected) == KIND_ORDER
    assert {kind: nonlinearity_alignment(cfg, kind).partition for kind in KIND_ORDER} == expected
    for spec in qwen36_moe_site_specs(cfg, full_site_cs(cfg, TINY_QWEN36_ALL_CS)):
        assert spec.alignment is not None
        assert spec.alignment.partition == expected[spec.group], spec.name
        expected_side = (
            "input"
            if spec.group in {"gdn_out", "attn_o", "experts_down", "shared_down"}
            else "output"
        )
        assert spec.alignment.side == expected_side, spec.name


def test_locality_loss_matches_exact_init_unit_counts():
    """The locality term over all 17 kinds: one mean per unit kind (`neuron`,
    `attention_head`, `deltanet_head`), each finite; at the exact-width aligned init every
    component reads or writes one aligned coordinate inside one chunk, so every kind's
    soft use count follows the one-coordinate formula at the authored threshold."""
    cfg = tiny_qwen36_cfg()
    sites = qwen36_moe_site_specs(cfg, full_site_cs(cfg, exact_width_cs(cfg, KIND_ORDER)))
    model = tiny_qwen36_decomposed_model(cfg, sites, jax.random.PRNGKey(30))
    components = nonlinearity_aligned_component_initializer(model, jax.random.PRNGKey(31))
    partitions = nonlinearity_alignments(sites)
    coefficients: dict[NonlinearityUnitKind, float] = {
        "neuron": 1.0,
        "attention_head": 1.0,
        "deltanet_head": 1.0,
    }
    total, by_kind = nonlinearity_loss(
        components, partitions, jnp.asarray(4.0), coefficients, normalize_at_one=False
    )
    assert set(by_kind) == set(coefficients)
    assert all(bool(jnp.isfinite(v)) for v in by_kind.values())
    # A one-coordinate component: fraction 1 on its own unit, 0 elsewhere, so its soft
    # count is exactly 1/(1 + t/U) times the multiplicity, and 0 for every other unit.
    threshold = 4.0
    expected_by_kind: dict[NonlinearityUnitKind, float] = {}
    for kind in coefficients:
        counts = []
        for spec in sites:
            if spec.alignment is None or spec.alignment.partition.unit_kind != kind:
                continue
            partition = spec.alignment.partition
            aligned_width = (
                spec.factorization.d_in
                if spec.alignment.side == "input"
                else spec.factorization.d_out
            )
            units = aligned_width if isinstance(partition, Neurons) else partition.head_count
            counts += [partition.use_multiplicity / (1 + threshold / units)] * spec.C
        expected_by_kind[kind] = sum(counts) / len(counts)
    for kind, value in by_kind.items():
        np.testing.assert_allclose(float(value), expected_by_kind[kind], rtol=1e-5, err_msg=kind)
    np.testing.assert_allclose(float(total), sum(expected_by_kind.values()), rtol=1e-5)


def test_moe_ci_arch_stays_chunk_homogeneous_with_mixer_slots():
    model, _vu = _model_and_vu(tiny_qwen36_cfg(), TINY_QWEN36_ALL_CS, jax.random.PRNGKey(11))
    cfg = model.cfg
    arch = tiny_qwen36_moe_ci_fn_arch(model)
    site_spec = {spec.name: spec for spec in model.sites}
    signatures = {
        _block_selected_slot_signature(chunk, site_spec, arch.table_size) for chunk in arch.chunks
    }
    assert len(arch.chunks) == cfg.n_stages and len(signatures) == 1

    ci_fn = tiny_qwen36_moe_ci_fn(model, jax.random.PRNGKey(12))
    tokens = _tokens(cfg)
    clean = model.clean_forward(LMBatch(tokens), ci_fn.capture_keys, placement=None)
    components = init_component_stacks(model.sites, jax.random.PRNGKey(13))
    ci = ci_fn.prepare()(
        dict(clean.captures),
        clean.conditioning,
        components,
        sequence=None,
        remat=False,
    )
    assert set(ci.lower) == set(model.site_names)
    for spec in model.sites:
        value = ci.lower[spec.name]
        _layer, kind = parse_site_name(spec.name)
        if is_expert_kind(kind):
            assert isinstance(value, SelectedCI), spec.name
            narrow = cfg.n_experts_per_token * spec.C // cfg.n_experts
            assert value.values.shape == (*tokens.shape, narrow), spec.name
        else:
            assert isinstance(value, jax.Array) and value.shape == (*tokens.shape, spec.C), (
                spec.name
            )


# ----------------------------- placed -----------------------------

multidevice = pytest.mark.skipif(len(jax.devices()) < 8, reason="requires eight local devices")

PLACED_MIXER_CS: dict[str, int] = {kind: 8 for kind in MIXER_KINDS}
"""The mixer kinds at a C the moe resident preset's master cut (`C: (tp, data)`, ÷8)
tiles; their exact widths at the tiny config (12, 20, 4) do not — as the shared kinds'
exact 20 does not in `CENSUS_CS`."""


@multidevice
@pytest.mark.multidevice
def test_placed_masked_forward_with_every_kind_matches_unplaced():
    """All seventeen kinds decomposed through the placed masked forward (random masks,
    weight-delta masks, routes, remat, mixer captures) on the (data, tp) mesh: logits,
    captures, and V/U gradients match the unplaced run, with the residency census intact.
    Two KV heads so tp=2 splits them whole and the K/V projections take the column row
    (`_kv_target_linear`)."""
    mesh = placement_mesh()
    cfg = replace(tiny_qwen36_cfg(), n_head=4, n_kv_head=2)
    sites = qwen36_moe_site_specs(cfg, full_site_cs(cfg, {**PLACED_MIXER_CS, **CENSUS_CS}))
    model = replace(
        tiny_qwen36_decomposed_model(cfg, sites, jax.random.PRNGKey(0)),
        expert_implementation="dense_masked",
    )
    components = init_component_stacks(sites, jax.random.PRNGKey(1))
    tokens = jax.random.randint(jax.random.PRNGKey(2), (BATCH, SEQ), 0, cfg.vocab_size)
    mask_values = random_mask_values(model, (BATCH, SEQ), jax.random.PRNGKey(3))
    deltas = {spec.name: jnp.full((BATCH, SEQ), 0.25) for spec in sites}
    routes = {spec.name: (jnp.arange(BATCH * SEQ).reshape(BATCH, SEQ) % 3 > 0) for spec in sites}
    capture_keys = frozenset(
        {
            resid_tap_key(cfg.n_layer),
            attention_output_tap_key(1),
            site_output_tap_key(site_name(0, "gdn_v")),
            site_output_tap_key(site_name(1, "attn_k")),
        }
    )

    def loss_and_result(
        target: PlacedModel[
            LMBatch,
            LMOutput,
            QwenPreparedWeights,
            LMBatchWithRouting[LMBatch],
            QwenPreparedMasking,
        ],
        value: ComponentStacks,
        component: dict[str, SiteCI],
        delta: dict[str, Array],
        route: dict[str, Array],
        conditioning: LMBatchWithRouting[LMBatch],
    ) -> tuple[Array, ForwardResult[LMOutput, LMBatchWithRouting[LMBatch]]]:
        result = target.masked_forward(
            target.prepare_compute_weights(value),
            conditioning,
            masking=target.model.prepare_masking(
                MaterializedMasking(component_masks=component, weight_delta_masks=delta)
            ),
            routes=route,
            capture_keys=capture_keys,
            remat=True,
        )
        return jnp.sum(materialized_logits(result.output)), result

    grads_and_result = jax.jit(jax.grad(loss_and_result, argnums=1, has_aux=True))
    unplaced = PlacedModel(model=model, placement=None)
    conditioning = unplaced.clean_forward(LMBatch(tokens)).conditioning
    expected_grads, expected = grads_and_result(
        unplaced,
        components,
        site_masks(model, conditioning.selection, mask_values),
        deltas,
        routes,
        conditioning,
    )

    rules = from_config("zero1-replicated-resident-moe", mesh, sites)
    placed_model = place_target(model, rules)
    placed_components = init_component_stacks_placed(sites, jax.random.PRNGKey(1), rules)
    placed_tokens = batch_placed(tokens, mesh)
    placed_mask_values, placed_deltas, placed_routes = jax.tree.map(
        lambda value: batch_placed(value, mesh), (mask_values, deltas, routes)
    )
    with jax.set_mesh(mesh):
        placed_conditioning = placed_model.clean_forward(LMBatch(placed_tokens)).conditioning
        np.testing.assert_array_equal(
            np.asarray(placed_conditioning.selection.indices),
            np.asarray(conditioning.selection.indices),
        )
        placed_args = (
            placed_model,
            placed_components,
            site_masks(model, placed_conditioning.selection, placed_mask_values),
            placed_deltas,
            placed_routes,
            placed_conditioning,
        )
        compiled = grads_and_result.lower(*placed_args).compile()
        hlo = compiled.as_text()
        got_grads, got = compiled(*placed_args)
    assert hlo is not None

    np.testing.assert_allclose(
        np.asarray(got.output), np.asarray(expected.output), rtol=2e-4, atol=2e-4
    )
    for key in sorted(capture_keys):
        np.testing.assert_allclose(
            np.asarray(got.captures[key]),
            np.asarray(expected.captures[key]),
            rtol=2e-4,
            atol=2e-4,
            err_msg=key,
        )
    # master grads pass through the bf16 compute cast in both arms, and a small element
    # can inherit a whole bf16 ulp of an O(1) upstream intermediate across the tp-split
    # reassociation — far past any elementwise bound at its own scale, negligible on the
    # leaf's norm; a wrong or missing term lands O(1) on the norm
    for group, (got_v, got_u) in got_grads.stacks.items():
        expected_v, expected_u = expected_grads.stacks[group]
        for got_leaf, expected_leaf in ((got_v, expected_v), (got_u, expected_u)):
            got_np, expected_np = np.asarray(got_leaf), np.asarray(expected_leaf)
            error = np.linalg.norm(got_np - expected_np) / np.linalg.norm(expected_np)
            assert error < 2e-2, (group, error)

    census = collective_census(hlo, replica_stride=TP, n_devices=DATA * TP)
    assert census.in_loop_cross_replicate == 0, census.counts
    assert census.exit_reductions > 0, census.counts
    assert census.counts.get("entry:all-gather[xrep]", 0) > 0, census.counts
    assert_no_weight_gather_in_any_loop(hlo, BATCH // DATA)


@multidevice
@pytest.mark.multidevice
def test_placed_forward_refuses_decomposing_a_replicated_kv_projection():
    """One KV head under tp=2: the K/V projections stay replicated, so a placed forward
    asked to decompose `attn_k` dies naming the constraint."""
    cfg = tiny_qwen36_cfg()
    mesh = placement_mesh()
    sites = qwen36_moe_site_specs(cfg, full_site_cs(cfg, {"attn_k": 8, **CENSUS_CS}))
    model = tiny_qwen36_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    rules = from_config("zero1-replicated-resident-moe", mesh, sites)
    placed_model = place_target(model, rules)
    components = init_component_stacks_placed(sites, jax.random.PRNGKey(1), rules)
    tokens = batch_placed(
        jax.random.randint(jax.random.PRNGKey(2), (BATCH, SEQ), 0, cfg.vocab_size), mesh
    )
    mask_values = jax.tree.map(
        lambda value: batch_placed(value, mesh), constant_mask_values(model, (BATCH, SEQ), 1.0)
    )
    with jax.set_mesh(mesh):
        conditioning = placed_model.clean_forward(LMBatch(tokens)).conditioning
        prepared = placed_model.prepare_compute_weights(components)
        with pytest.raises(AssertionError, match="cannot be decomposed here"):
            placed_model.masked_forward(
                prepared,
                conditioning,
                masking=placed_model.model.prepare_masking(
                    MaterializedMasking(
                        component_masks=site_masks(model, conditioning.selection, mask_values)
                    )
                ),
                routes=None,
                remat=True,
            )


@pytest.mark.multidevice
@pytest.mark.skipif(len(jax.devices()) < 2, reason="requires two devices")
@pytest.mark.parametrize("tp", [1, 2])
def test_placed_aligned_initializer_and_zero_padding_preserve_sites(tp: int):
    from jax.sharding import AxisType, Mesh

    from param_decomp.core.components import pad_component_stacks

    cfg = replace(tiny_qwen36_cfg(), n_head=4, n_kv_head=2)
    model, _ = _model_and_vu(cfg, exact_width_cs(cfg, KIND_ORDER), jax.random.PRNGKey(32))
    key = jax.random.PRNGKey(33)
    expected = nonlinearity_aligned_component_initializer(model, key)
    mesh = Mesh(
        np.asarray(jax.devices()[:2]).reshape(2 // tp, tp),
        ("data", "tp"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    rules = from_config("zero1-replicated-resident-moe", mesh, model.sites)
    placed = place_target(model, rules)
    assert isinstance(placed.model, Qwen36MoeDecomposedModel)
    with jax.set_mesh(mesh):
        initialized = jax.jit(nonlinearity_aligned_component_initializer)(placed.model, key)
        padded = jax.jit(lambda x: pad_component_stacks(x, {"gdn_q": 1, "experts_gate": 1}))(
            initialized
        )
        for name, reference in expected.sites_items():
            got = padded.site(name)
            np.testing.assert_array_equal(got.V, reference.V, err_msg=name)
            np.testing.assert_array_equal(got.U, reference.U, err_msg=name)
        for group in ("gdn_q", "experts_gate"):
            for value, original in zip(
                padded.stacks[group], initialized.stacks[group], strict=True
            ):
                assert jnp.array_equal(value[-1], jnp.zeros_like(value[-1]))
                assert value.sharding == original.sharding
