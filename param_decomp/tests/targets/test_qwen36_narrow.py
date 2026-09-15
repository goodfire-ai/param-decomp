"""CPU tests for NARROW CI emission end-to-end on the tiny qwen36_moe target.

The MoE chunkwise CI fn reads the target's captured routing and emits `NarrowCI`
bundles; narrow masks drive the routed decomposed expert arm on the CLEAN forward's
routing (the seam contract: the mask bundle's indices key the jobs schedule). The
scatter-oracle parity block compares narrow masks against the `"dense"` all-expert
oracle consuming the bundle's scattered full-width view — at the first MoE layer, whose
router input no upstream mask can perturb, the two are the same math up to fp32
reassociation. The e2e block runs the REAL `make_train_step` (faithfulness + smooth-L0
imp-min + stochastic recon + persistent-PGD `sc` sources) under both optimizers."""

import dataclasses
from collections.abc import Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from param_decomp.core.adversary import (
    PersistentAdversary,
    Sources,
    SourceStacks,
    init_persistent_sources,
    init_sources_opt_state,
    source_values_to_float,
    store_unit_float,
)
from param_decomp.core.ci_fn import (
    CI,
    MoEChunkwiseTransformerCIFn,
)
from param_decomp.core.components import (
    ComponentStacks,
    NarrowCI,
    SiteCI,
    init_component_stacks,
    map_site_ci,
    site_ci_values,
)
from param_decomp.core.configs import (
    AdamPGDConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    MomentumSgdPGDConfig,
    PersistentPGDReconLossConfig,
    StochasticReconLossConfig,
)
from param_decomp.core.faithfulness import faithfulness_loss_for
from param_decomp.core.masking import _sample_stochastic_masks
from param_decomp.core.model import PlacedModel
from param_decomp.core.muon_stacked import stacked_muon
from param_decomp.core.objective import build_objective
from param_decomp.core.run_state import moe_stacked_muon_dimension_numbers
from param_decomp.core.schedule import Knot, ScheduleConfig
from param_decomp.core.train import (
    Decomposition,
    ForwardSubstrate,
    TrainingItem,
    TrainState,
    make_train_step,
)
from param_decomp.targets.qwen36_moe import (
    Qwen36MoeConfig,
    Qwen36MoeDecomposedModel,
    full_site_cs,
    qwen36_moe_site_specs,
    router_idx_tap_key,
    site_name,
)
from param_decomp.targets.testing import (
    TINY_QWEN36_CS,
    materialized_logits,
    run_clean,
    run_masked,
    tiny_qwen36_cfg,
    tiny_qwen36_decomposed_model,
    tiny_qwen36_moe_ci_fn,
)
from param_decomp.targets.transformer_taps import site_output_tap_key

B, T = 2, 12


def _model_and_vu(key: jax.Array) -> tuple[Qwen36MoeDecomposedModel, ComponentStacks]:
    cfg = tiny_qwen36_cfg()
    sites = qwen36_moe_site_specs(cfg, full_site_cs(cfg, TINY_QWEN36_CS))
    model_key, vu_key = jax.random.split(key)
    return tiny_qwen36_decomposed_model(cfg, sites, model_key), init_component_stacks(sites, vu_key)


def _tokens(cfg: Qwen36MoeConfig) -> jax.Array:
    return jax.random.randint(jax.random.PRNGKey(7), (B, T), 0, cfg.vocab_size)


def _clean_ci(model: Qwen36MoeDecomposedModel, tokens: jax.Array) -> CI:
    ci_fn = tiny_qwen36_moe_ci_fn(model, jax.random.PRNGKey(11))
    taps = model.clean_forward(tokens, ci_fn.capture_keys, placement=None).captures
    return ci_fn(dict(taps), remat=False, placement=None)


def _scatter_to_full(bundle: NarrowCI) -> jax.Array:
    """THE test oracle (this module's twin of `test_narrow_ci._scatter_to_full`)."""
    lead = bundle.values.shape[:-1]
    k = bundle.router_indices.shape[-1]
    values = bundle.values.reshape(*lead, k, bundle.c_per_expert)
    one_hot = jax.nn.one_hot(bundle.router_indices, bundle.n_experts, dtype=values.dtype)
    return jnp.einsum("...kc,...ke->...ec", values, one_hot).reshape(*lead, bundle.C)


def test_narrow_emission_carries_the_captured_routing():
    model, _vu = _model_and_vu(jax.random.PRNGKey(0))
    cfg = model.cfg
    tokens = _tokens(cfg)
    routing_keys = frozenset(router_idx_tap_key(layer) for layer in range(cfg.n_layer))
    captures = model.clean_forward(tokens, routing_keys, placement=None).captures
    ci = _clean_ci(model, tokens)
    for spec in model.sites:
        value = ci.lower[spec.name]
        layer = int(spec.name.split(".")[1])
        if "shared_expert" in spec.name:
            assert site_ci_values(value).shape == (B, T, spec.C)
        else:
            assert isinstance(value, NarrowCI), spec.name
            assert value.n_experts == cfg.n_experts
            assert value.values.shape == (
                B,
                T,
                cfg.n_experts_per_token * spec.factorization.C // cfg.n_experts,
            )
            np.testing.assert_array_equal(value.router_indices, captures[router_idx_tap_key(layer)])


def test_narrow_mask_one_delta_one_reconstructs_clean_forward():
    """All-ones narrow masks + delta 1: coefficients cancel to 0 and the frozen channel
    carries the layer — with the bundle's CLEAN routing and unperturbed inputs, the
    masked forward reproduces the clean forward up to fp32 reassociation."""
    model, vu = _model_and_vu(jax.random.PRNGKey(1))
    tokens = _tokens(model.cfg)
    ci = _clean_ci(model, tokens)
    prepared = model.prepare_compute_weights(vu, None)
    masks = {name: map_site_ci(jnp.ones_like, value) for name, value in ci.lower.items()}
    deltas = {name: jnp.ones(tokens.shape) for name in model.site_names}
    clean = materialized_logits(run_clean(model, tokens))
    masked = materialized_logits(
        run_masked(
            model, prepared, tokens, masks, deltas, None, uses_weight_deltas=True, remat=False
        )
    )
    np.testing.assert_allclose(masked, clean, rtol=2e-4, atol=2e-4)


def test_narrow_masks_match_the_dense_scatter_oracle_at_the_first_moe_layer():
    """The scatter-oracle equivalence for the masked path: random narrow masks against
    the `"dense"` all-expert oracle consuming their scattered `[.., C]` view. Compared at
    layer 0's routed-expert output — the one site whose router input no upstream mask can
    perturb, so both arms route identically and differ by fp32 reassociation only."""
    model, vu = _model_and_vu(jax.random.PRNGKey(2))
    tokens = _tokens(model.cfg)
    ci = _clean_ci(model, tokens)
    prepared = model.prepare_compute_weights(vu, None)
    mask_key = jax.random.PRNGKey(3)
    masks = {}
    full_masks = {}
    for site_idx, (name, value) in enumerate(ci.lower.items()):
        draw = jax.random.uniform(
            jax.random.fold_in(mask_key, site_idx), site_ci_values(value).shape
        )
        narrow_mask = map_site_ci(lambda v, d=draw: jnp.clip(v + 0.5 * d, 0.0, 1.0), value)
        masks[name] = narrow_mask
        match narrow_mask:
            case NarrowCI():
                full_masks[name] = _scatter_to_full(narrow_mask)
            case _:
                full_masks[name] = narrow_mask
    tap = site_output_tap_key(site_name(0, "experts_down"))
    dense_model = dataclasses.replace(model, experts_execution="dense")

    def captured(m: Qwen36MoeDecomposedModel, site_masks: Mapping[str, SiteCI]) -> jax.Array:
        from param_decomp.core.model import MaterializedMasking

        return m.masked_forward(
            prepared,
            tokens,
            masking=MaterializedMasking(component_masks=site_masks),
            placement=None,
            capture_keys=frozenset({tap}),
            remat=False,
        ).captures[tap]

    narrow_out = captured(model, masks)
    dense_out = captured(dense_model, full_masks)
    np.testing.assert_allclose(narrow_out, dense_out, rtol=2e-4, atol=2e-4)


def test_stochastic_narrow_masked_forward_differentiates():
    """The in-stage stochastic rebuild draws at the narrow shape; grads flow to V/U and
    to the CI values through the routed decomposed arm."""
    model, vu = _model_and_vu(jax.random.PRNGKey(4))
    tokens = _tokens(model.cfg)
    ci = _clean_ci(model, tokens)
    prepared = model.prepare_compute_weights(vu, None)

    def loss(
        prepared_weights: dict[str, dict[str, jax.Array]], ci_lower: dict[str, SiteCI]
    ) -> jax.Array:
        from param_decomp.core.model import StochasticMasking

        out = model.masked_forward(
            prepared_weights,
            tokens,
            masking=StochasticMasking(
                ci_stacked=model.stack_ci(ci_lower),
                draw_key=jax.random.PRNGKey(5),
                routes=None,
            ),
            placement=None,
            remat=True,
        ).output
        return jnp.sum(materialized_logits(out).astype(jnp.float32) ** 2)

    # filter_grad: the bundle's int32 router indices carry no cotangent
    grads_prepared, grads_ci = eqx.filter_grad(lambda args: loss(*args), has_aux=False)(
        (prepared, dict(ci.lower))
    )
    v_grad = grads_prepared["experts_gate"]["V"]
    assert float(jnp.max(jnp.abs(v_grad))) > 0.0
    narrow_site = site_name(0, "experts_gate")
    narrow_grad = grads_ci[narrow_site]
    assert isinstance(narrow_grad, NarrowCI)
    assert float(jnp.max(jnp.abs(narrow_grad.values))) > 0.0


def _constant_sources(
    model: Qwen36MoeDecomposedModel, leading: tuple[int, ...], dtype: jnp.dtype
) -> SourceStacks:
    drawn = init_persistent_sources(model.sites, leading, jnp.float32, jax.random.PRNGKey(0))
    return jax.tree.map(lambda a: store_unit_float(jnp.full_like(a, 0.5), dtype), drawn)


PPGD_SC_ADAM = PersistentPGDReconLossConfig(
    coeff=0.5,
    n_warmup_steps=1,
    source_shape="sc",
    optimizer=AdamPGDConfig(
        type="adam",
        beta1=0.01,
        beta2=0.99,
        eps=1e-8,
        lr_schedule=ScheduleConfig(
            max_val=0.01, points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=1.0))
        ),
    ),
)

PPGD_BSC_MOMENTUM_U16 = PersistentPGDReconLossConfig(
    coeff=0.5,
    n_warmup_steps=1,
    source_shape="bsc",
    source_dtype="uint16",
    optimizer=MomentumSgdPGDConfig(momentum=0.9, lr_schedule=ScheduleConfig.constant(0.01)),
)


def _persistent_sources(
    model: Qwen36MoeDecomposedModel, cfg: PersistentPGDReconLossConfig
) -> SourceStacks:
    match cfg.source_shape:
        case "sc":
            leading: tuple[int, ...] = (1, T)
        case "bsc":
            leading = (B, T)
        case other:
            raise AssertionError(other)
    return _constant_sources(model, leading, jnp.dtype(cfg.source_dtype))


@pytest.mark.parametrize(
    ("optimizer_kind", "ppgd_cfg"),
    [
        ("adamw", PPGD_SC_ADAM),
        ("stacked_muon", PPGD_SC_ADAM),
        ("adamw", PPGD_BSC_MOMENTUM_U16),
    ],
    ids=["adamw-sc-adam", "muon-sc-adam", "adamw-bsc-momentum-u16"],
)
def test_e2e_train_step_with_narrow_emission(
    optimizer_kind: str, ppgd_cfg: PersistentPGDReconLossConfig
):
    """The REAL train step at the tiny config: MoE chunkwise CI fn (narrow emission),
    smooth-L0 imp-min (the no-[C]-accumulator lp path), stochastic recon, persistent-PGD
    sources (the full-C-source → narrow-mask gather; `sc`+adam and the large-capacity reference's
    `bsc`+sgd bf16 slots), faithfulness — two steps, finite, V and the CI expert banks
    both move."""
    model, components = _model_and_vu(jax.random.PRNGKey(6))
    ci_fn = tiny_qwen36_moe_ci_fn(model, jax.random.PRNGKey(8))
    match optimizer_kind:
        case "adamw":
            opt_ci = optax.adamw(1e-2, weight_decay=0.0)
        case "stacked_muon":
            opt_ci = stacked_muon(
                1e-2,
                beta=0.95,
                weight_decay=0.0,
                consistent_rms=None,
                muon_weight_dimension_numbers=moe_stacked_muon_dimension_numbers,
                ns_steps=5,
                ns_dtype=jnp.dtype(jnp.float32),
                waypoints=None,
            )
        case _:
            raise AssertionError(optimizer_kind)
    opt_vu = optax.adamw(1e-2, weight_decay=0.0)
    objective = build_objective(
        (
            FaithfulnessLossConfig(coeff=1.0),
            ImportanceMinimalityLossConfig(
                coeff=1e-4,
                gamma=ScheduleConfig(
                    max_val=1.0, points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.01))
                ),
            ),
            StochasticReconLossConfig(coeff=1.0),
            ppgd_cfg,
        ),
        model.site_names,
    )
    sources = _persistent_sources(model, ppgd_cfg)
    state = TrainState(
        decomposition=Decomposition(components=components, ci_fn=ci_fn),
        training=TrainingItem(
            components_opt_state=opt_vu.init(eqx.filter(components, eqx.is_array)),
            ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
            adversaries={
                ppgd_cfg.type: PersistentAdversary(
                    sources=sources,
                    opt_state=init_sources_opt_state(ppgd_cfg.optimizer, sources),
                    state_key=ppgd_cfg.type,
                    optimizer=ppgd_cfg.optimizer,
                    n_warmup=ppgd_cfg.n_warmup_steps,
                )
            },
            freq_ema=None,
            step=jnp.zeros((), jnp.int32),
        ),
    )
    placed = PlacedModel(model=model, placement=None)
    step_fn = make_train_step(
        model_static=placed,
        substrate=ForwardSubstrate.of(
            placed,
            remat_recon_forwards=True,
            remat_ci_fn=True,
            ci_capture_keys=ci_fn.capture_keys,
            ci_placement=None,
        ),
        objective=objective,
        components_optimizer=opt_vu,
        ci_fn_optimizer=opt_ci,
        total_steps=4,
        faithfulness=faithfulness_loss_for(placed),
    )
    tokens = _tokens(model.cfg)
    v_before = np.asarray(components.stacks["experts_gate"][0])
    bank_before = np.asarray(ci_fn.chunks.blocks[0].expert_gate[0])
    for step_index in range(2):
        state, metrics = step_fn(
            placed, state, tokens, jax.random.fold_in(jax.random.PRNGKey(9), step_index)
        )
        assert jnp.isfinite(metrics["total"]), (optimizer_kind, step_index, metrics["total"])
    v_moved = np.asarray(state.decomposition.components.stacks["experts_gate"][0])
    stepped_ci_fn = state.decomposition.ci_fn
    assert isinstance(stepped_ci_fn, MoEChunkwiseTransformerCIFn)
    bank_moved = np.asarray(stepped_ci_fn.chunks.blocks[0].expert_gate[0])
    assert not np.allclose(v_moved, v_before), "V did not move"
    assert not np.allclose(bank_moved, bank_before), "the CI expert bank did not move"


def test_sample_stochastic_masks_draws_at_the_narrow_shape():
    model, _vu = _model_and_vu(jax.random.PRNGKey(10))
    tokens = _tokens(model.cfg)
    ci = _clean_ci(model, tokens)
    masks, deltas = _sample_stochastic_masks(dict(ci.lower), jax.random.PRNGKey(11))
    narrow_site = site_name(0, "experts_gate")
    mask = masks[narrow_site]
    assert isinstance(mask, NarrowCI)
    lower = ci.lower[narrow_site]
    assert isinstance(lower, NarrowCI)
    assert bool(jnp.all(mask.values >= lower.values))
    assert deltas[narrow_site].shape == (B, T)


def test_source_masking_recomposes_bit_identically_to_materialized_masks():
    """`SourceMasking` (masks recomposed inside the checkpointed stage bodies) vs the
    eager spelling (`masks_from_sources` → `MaterializedMasking`) through the REAL
    masked forward: uint16 `bsc` sources, real narrow CI, remat on. The output and the
    gradients w.r.t. V/U, the CI envelope, and the float source view must be
    BIT-identical — the recipe re-spells the same ops, staged not saved, and
    the persistent coeff rides the STACKED CI's cotangents exactly as the eager
    spelling rides the per-site CI's (model-side scaled, source path not)."""
    from param_decomp.core.masking import masks_from_sources, source_value_cis
    from param_decomp.core.model import MaterializedMasking, SourceMasking
    from param_decomp.core.train import model_cotangents_scaled

    model, vu = _model_and_vu(jax.random.PRNGKey(6))
    tokens = _tokens(model.cfg)
    ci = _clean_ci(model, tokens)
    prepared = model.prepare_compute_weights(vu, None)
    stored = init_persistent_sources(model.sites, (B, T), jnp.uint16, jax.random.PRNGKey(12))
    coeff = 0.5

    def out_loss(masking: MaterializedMasking | SourceMasking) -> jax.Array:
        out = model.masked_forward(
            prepared, tokens, masking=masking, placement=None, remat=True
        ).output
        return jnp.sum(materialized_logits(out).astype(jnp.float32) ** 2)

    def loss_eager(args: tuple[dict[str, SiteCI], Sources]) -> jax.Array:
        ci_lower, sources = args
        masks, deltas = masks_from_sources(model_cotangents_scaled(ci_lower, coeff), sources)
        return out_loss(
            MaterializedMasking(component_masks=masks, weight_delta_masks=deltas, routes=None)
        )

    def loss_recipe(args: tuple[dict[str, SiteCI], Sources]) -> jax.Array:
        ci_lower, sources = args
        values, deltas = source_value_cis(ci_lower, sources)
        return out_loss(
            SourceMasking(
                ci_stacked=model_cotangents_scaled(model.stack_ci(ci_lower), coeff),
                source_values_stacked=model.stack_ci(values),
                delta_values_stacked=model.stack_ci(deltas),
                routes=None,
            )
        )

    # One jit around value_and_grad, mask formation inside — the production seam
    # (train.py's loss_fn). Bit-identity is a compiled-program property: op-by-op
    # eager execution rounds each concrete intermediate separately and drifts.
    args = (dict(ci.lower), source_values_to_float(stored).per_site())
    eager_loss, eager_grads = eqx.filter_jit(eqx.filter_value_and_grad(loss_eager))(args)
    recipe_loss, recipe_grads = eqx.filter_jit(eqx.filter_value_and_grad(loss_recipe))(args)
    np.testing.assert_array_equal(np.asarray(eager_loss), np.asarray(recipe_loss))
    eager_leaves = jax.tree.leaves(eager_grads)
    recipe_leaves = jax.tree.leaves(recipe_grads)
    assert eager_leaves and len(eager_leaves) == len(recipe_leaves)
    for eager_leaf, recipe_leaf in zip(eager_leaves, recipe_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(eager_leaf), np.asarray(recipe_leaf))
