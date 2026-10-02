"""CPU tests for NARROW CI emission end-to-end on the tiny qwen36_moe target.

The MoE chunkwise CI fn scores on the clean forward's pinned `BlockSelection` and emits
`SelectedCI` bundles carrying that routing; narrow masks drive the routed decomposed expert
arm under the same pinned routing (the seam contract: the masked forward reproduces the
clean indices, and mask slot m means the m-th pinned expert). The routed arm's parity
against the dense all-expert reference lives in `test_qwen36_moe`. The e2e block runs the
REAL `make_train_step` (faithfulness + smooth-L0 imp-min + stochastic recon +
persistent-PGD `bsc` sources) under both optimizers."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
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
from param_decomp.core.ci_fn.implementations.block_selected.arch import (
    BlockSelectedChunkwiseTransformerCIFn,
)
from param_decomp.core.ci_fn.interface import CI
from param_decomp.core.ci_fn.optimizer import ci_fn_muon_dimension_numbers
from param_decomp.core.components import (
    ComponentStacks,
    SelectedCI,
    SiteCI,
    init_component_stacks,
    map_site_ci,
    site_ci_values,
)
from param_decomp.core.configs import (
    AdamPGDConfig,
    AdamWOptimizerConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    MomentumSgdPGDConfig,
    MuonOptimizerConfig,
    PersistentPGDReconLossConfig,
    StochasticReconLossConfig,
)
from param_decomp.core.faithfulness import faithfulness_loss_for
from param_decomp.core.losses import BatchFrequency
from param_decomp.core.masking import materialize_masking
from param_decomp.core.model import MaterializedMasking, PlacedModel, StochasticMasking
from param_decomp.core.objective import build_objective
from param_decomp.core.run_state import (
    _adamw_optimizer,
    _muon_optimizer,
)
from param_decomp.core.schedule import Knot, ScheduleConfig
from param_decomp.core.train import (
    Decomposition,
    ForwardSubstrate,
    PDTrainingState,
    TrainState,
    make_train_step,
)
from param_decomp.lm.batch import LMBatch, LMBatchWithRouting
from param_decomp.targets.qwen36_moe import (
    Qwen36MoeConfig,
    Qwen36MoeDecomposedModel,
    QwenPreparedWeights,
    full_site_cs,
    qwen36_moe_site_specs,
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

B, T = 2, 12


def _model_and_vu(key: jax.Array) -> tuple[Qwen36MoeDecomposedModel, ComponentStacks]:
    cfg = tiny_qwen36_cfg()
    sites = qwen36_moe_site_specs(cfg, full_site_cs(cfg, TINY_QWEN36_CS))
    model_key, vu_key = jax.random.split(key)
    return tiny_qwen36_decomposed_model(cfg, sites, model_key), init_component_stacks(sites, vu_key)


def _tokens(cfg: Qwen36MoeConfig) -> jax.Array:
    return jax.random.randint(jax.random.PRNGKey(7), (B, T), 0, cfg.vocab_size)


def _clean_routing_and_ci(
    model: Qwen36MoeDecomposedModel, vu: ComponentStacks, tokens: jax.Array
) -> tuple[LMBatchWithRouting[LMBatch], CI]:
    """One clean forward: its pinned routing, and the MoE CI fn's envelope scored on it."""
    ci_fn = tiny_qwen36_moe_ci_fn(model, jax.random.PRNGKey(11))
    clean = model.clean_forward(LMBatch(tokens), ci_fn.capture_keys, placement=None)
    prepared = model.prepare_compute_weights(vu, None)
    return clean.conditioning, ci_fn(
        dict(clean.captures), clean.conditioning, prepared, sequence=None, remat=False
    )


def test_narrow_emission_carries_the_pinned_routing():
    model, vu = _model_and_vu(jax.random.PRNGKey(0))
    cfg = model.cfg
    tokens = _tokens(cfg)
    routing, ci = _clean_routing_and_ci(model, vu, tokens)
    for spec in model.sites:
        value = ci.lower[spec.name]
        layer = int(spec.name.split(".")[1])
        if "shared_expert" in spec.name:
            assert site_ci_values(value).shape == (B, T, spec.C)
        else:
            assert isinstance(value, SelectedCI), spec.name
            assert value.n_blocks == cfg.n_experts
            assert value.values.shape == (
                B,
                T,
                cfg.n_experts_per_token * spec.factorization.C // cfg.n_experts,
            )
            np.testing.assert_array_equal(value.block_indices, routing.selection.indices[layer])


def test_narrow_mask_one_delta_one_reconstructs_clean_forward():
    """All-ones narrow masks + delta 1: coefficients cancel to 0 and the frozen channel
    carries the layer — under the pinned CLEAN routing with unperturbed inputs, the
    masked forward reproduces the clean forward up to fp32 reassociation."""
    model, vu = _model_and_vu(jax.random.PRNGKey(1))
    tokens = _tokens(model.cfg)
    routing, ci = _clean_routing_and_ci(model, vu, tokens)
    prepared = model.prepare_compute_weights(vu, None)
    masks = {name: map_site_ci(jnp.ones_like, value) for name, value in ci.lower.items()}
    deltas = {name: jnp.ones(tokens.shape) for name in model.site_names}
    clean = materialized_logits(run_clean(model, LMBatch(tokens)))
    masked = materialized_logits(
        run_masked(
            model,
            prepared,
            routing,
            MaterializedMasking(component_masks=masks, weight_delta_masks=deltas),
            remat=False,
            routes=None,
        )
    )
    np.testing.assert_allclose(masked, clean, rtol=2e-4, atol=2e-4)


def test_stochastic_narrow_masked_forward_differentiates():
    """The in-stage stochastic rebuild draws at the narrow shape; grads flow to V/U and
    to the CI values through the routed decomposed arm."""
    model, vu = _model_and_vu(jax.random.PRNGKey(4))
    tokens = _tokens(model.cfg)
    routing, ci = _clean_routing_and_ci(model, vu, tokens)
    prepared = model.prepare_compute_weights(vu, None)

    def loss(prepared_weights: QwenPreparedWeights, ci_lower: dict[str, SiteCI]) -> jax.Array:
        from param_decomp.core.model import StochasticMasking

        out = model.masked_forward(
            prepared_weights,
            routing,
            masking=model.prepare_masking(
                StochasticMasking(
                    ci=ci_lower,
                    draw_key=jax.random.PRNGKey(5),
                )
            ),
            routes=None,
            placement=None,
            remat=True,
        ).output
        return jnp.sum(materialized_logits(out).astype(jnp.float32) ** 2)

    # filter_grad: the bundle's int32 router indices carry no cotangent
    grads_prepared, grads_ci = eqx.filter_grad(lambda args: loss(*args), has_aux=False)(
        (prepared, dict(ci.lower))
    )
    v_grad = grads_prepared.per_kind["experts_gate"]["V"]
    assert float(jnp.max(jnp.abs(v_grad))) > 0.0
    narrow_site = site_name(0, "experts_gate")
    narrow_grad = grads_ci[narrow_site]
    assert isinstance(narrow_grad, SelectedCI)
    assert float(jnp.max(jnp.abs(narrow_grad.values))) > 0.0


def _constant_sources(
    model: Qwen36MoeDecomposedModel, leading: tuple[int, ...], dtype: jnp.dtype
) -> SourceStacks:
    drawn = init_persistent_sources(model.sites, leading, jnp.float32, jax.random.PRNGKey(0))
    return jax.tree.map(lambda a: store_unit_float(jnp.full_like(a, 0.5), dtype), drawn)


PPGD_BSC_ADAM = PersistentPGDReconLossConfig(
    coeff=0.5,
    n_warmup_steps=1,
    source_shape="bsc",
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
        case "bc":
            leading: tuple[int, ...] = (B, 1)
        case "bsc":
            leading = (B, T)
    return _constant_sources(model, leading, jnp.dtype(cfg.source_dtype))


@pytest.mark.parametrize(
    ("optimizer_kind", "ppgd_cfg"),
    [
        ("adamw", PPGD_BSC_ADAM),
        ("stacked_muon", PPGD_BSC_ADAM),
        ("adamw", PPGD_BSC_MOMENTUM_U16),
    ],
    ids=["adamw-bsc-adam", "muon-bsc-adam", "adamw-bsc-momentum-u16"],
)
def test_e2e_train_step_with_narrow_emission(
    optimizer_kind: str, ppgd_cfg: PersistentPGDReconLossConfig
):
    """The REAL train step at the tiny config: MoE chunkwise CI fn (narrow emission),
    smooth-L0 imp-min (the no-[C]-accumulator lp path), stochastic recon, persistent-PGD
    sources (the full-C-source → narrow-mask gather; `bsc`+adam and the large-capacity configuration's
    `bsc`+sgd bf16 slots), faithfulness — two steps, finite, V and the CI expert banks
    both move."""
    model, components = _model_and_vu(jax.random.PRNGKey(6))
    ci_fn = tiny_qwen36_moe_ci_fn(model, jax.random.PRNGKey(8))
    match optimizer_kind:
        case "adamw":
            opt_ci = _adamw_optimizer(
                AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(1e-2)), 1
            )
        case "stacked_muon":
            opt_ci = _muon_optimizer(
                MuonOptimizerConfig(
                    type="muon", lr_schedule=ScheduleConfig.constant(1e-2), consistent_rms=None
                ),
                1,
                ci_fn_muon_dimension_numbers,
                None,
            )
        case _:
            raise AssertionError(optimizer_kind)
    opt_vu = _adamw_optimizer(AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(1e-2)), 1)
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
        model.sites,
    )
    sources = _persistent_sources(model, ppgd_cfg)
    state = TrainState(
        decomposition=Decomposition(components=components, ci_fn=ci_fn),
        training=PDTrainingState(
            frequency=BatchFrequency(),
            objective=objective,
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
            step=jnp.zeros((), jnp.int32),
        ),
    )
    placed = PlacedModel(model=model, placement=None)
    step_fn = jax.jit(
        make_train_step(
            model_static=placed,
            substrate=ForwardSubstrate.of(
                placed,
                remat_recon_forwards=True,
                remat_ci_fn=True,
                ci_capture_keys=ci_fn.capture_keys,
            ),
            components_optimizer=opt_vu,
            ci_fn_optimizer=opt_ci,
            total_steps=4,
            faithfulness=faithfulness_loss_for(placed),
        )
    )
    tokens = _tokens(model.cfg)
    v_before = np.asarray(components.stacks["experts_gate"][0])
    bank_before = np.asarray(ci_fn.chunks.blocks[0].selected_gate[0])
    for step_index in range(2):
        state, metrics = step_fn(
            placed,
            state,
            LMBatch(tokens),
            jax.random.fold_in(jax.random.PRNGKey(9), step_index),
        )
        assert jnp.isfinite(metrics["total"]), (optimizer_kind, step_index, metrics["total"])
    v_moved = np.asarray(state.decomposition.components.stacks["experts_gate"][0])
    stepped_ci_fn = state.decomposition.ci_fn
    assert isinstance(stepped_ci_fn, BlockSelectedChunkwiseTransformerCIFn)
    bank_moved = np.asarray(stepped_ci_fn.chunks.blocks[0].selected_gate[0])
    assert not np.allclose(v_moved, v_before), "V did not move"
    assert not np.allclose(bank_moved, bank_before), "the CI expert bank did not move"


def test_stochastic_masking_materializes_at_the_narrow_shape():
    model, vu = _model_and_vu(jax.random.PRNGKey(10))
    tokens = _tokens(model.cfg)
    _routing, ci = _clean_routing_and_ci(model, vu, tokens)
    masking = materialize_masking(
        StochasticMasking(ci=dict(ci.lower), draw_key=jax.random.PRNGKey(11))
    )
    masks = masking.component_masks
    assert masking.weight_delta_masks is not None
    deltas = masking.weight_delta_masks
    narrow_site = site_name(0, "experts_gate")
    mask = masks[narrow_site]
    assert isinstance(mask, SelectedCI)
    lower = ci.lower[narrow_site]
    assert isinstance(lower, SelectedCI)
    assert bool(jnp.all(mask.values >= lower.values))
    assert deltas[narrow_site].shape == (B, T)


def test_source_masking_recomposes_bit_identically_to_materialized_masks():
    """`SourceMasking` (masks recomposed inside the checkpointed stage bodies) vs the
    eager spelling (`materialize_masking` → `MaterializedMasking`) through the REAL
    masked forward: uint16 `bsc` sources, real narrow CI, remat on. The output and the
    gradients w.r.t. V/U, the CI envelope, and the float source view must be
    BIT-identical — the recipe re-spells the same ops, staged not saved, and
    the persistent coeff rides the STACKED CI's cotangents exactly as the eager
    spelling rides the per-site CI's (model-side scaled, source path not)."""
    from param_decomp.core.masking import materialize_masking, source_masking
    from param_decomp.core.model import MaterializedMasking, SourceMasking
    from param_decomp.core.train import model_cotangents_scaled

    model, vu = _model_and_vu(jax.random.PRNGKey(6))
    tokens = _tokens(model.cfg)
    routing, ci = _clean_routing_and_ci(model, vu, tokens)
    prepared = model.prepare_compute_weights(vu, None)
    stored = init_persistent_sources(model.sites, (B, T), jnp.uint16, jax.random.PRNGKey(12))
    coeff = jnp.asarray(0.5, dtype=jnp.float32)

    def out_loss(masking: MaterializedMasking | SourceMasking) -> jax.Array:
        out = model.masked_forward(
            prepared,
            routing,
            masking=model.prepare_masking(masking),
            routes=None,
            placement=None,
            remat=True,
        ).output
        return jnp.sum(materialized_logits(out).astype(jnp.float32) ** 2)

    def loss_eager(args: tuple[dict[str, SiteCI], Sources]) -> jax.Array:
        ci_lower, sources = args
        masking = materialize_masking(
            source_masking(model_cotangents_scaled(ci_lower, coeff), sources)
        )
        return out_loss(masking)

    def loss_recipe(args: tuple[dict[str, SiteCI], Sources]) -> jax.Array:
        ci_lower, sources = args
        return out_loss(source_masking(model_cotangents_scaled(ci_lower, coeff), sources))

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
