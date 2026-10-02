"""Optimizer budgets use real logical parameter shapes and economical update algebra."""

from dataclasses import replace
from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from optax.contrib._muon import orthogonalize_via_newton_schulz

from param_decomp.core.ci_fn.implementations.block_selected.arch import (
    BlockSelectedChunk,
    BlockSelectedChunkwiseTransformerCIFnArch,
    FullSlot,
    SelectedSlot,
)
from param_decomp.core.ci_fn.implementations.chunkwise.arch import (
    Chunk,
    ChunkwiseTransformerCIFnArch,
)
from param_decomp.core.ci_fn.implementations.global_mlp import GlobalMLPCIFnArch
from param_decomp.core.ci_fn.implementations.global_transformer.arch import (
    GlobalTransformerCIFnArch,
)
from param_decomp.core.ci_fn.implementations.layerwise_mlp import LayerwiseMLPCIFnArch
from param_decomp.core.ci_fn.implementations.transformer.component_conditioning import (
    ConditionedCIFnArch,
    InputScaleCalibration,
    SiteInput,
)
from param_decomp.core.ci_fn.implementations.transformer.layers import GQACIFnAttention
from param_decomp.core.ci_fn.interface import TapSpec
from param_decomp.core.components import (
    BlockedFactorization,
    DenseFactorization,
    SiteSpec,
    component_parameters,
)
from param_decomp.core.configs import (
    AdamPGDConfig,
    AdamWOptimizerConfig,
    BatchSourceShape,
    CIMaskedReconLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    MergedStochasticSubsetPooledPPGDReconLossConfig,
    MergedStochasticSubsetPPGDReconLossConfig,
    MomentumSgdPGDConfig,
    MuonOptimizerConfig,
    PDConfig,
    PersistentPGDReconLossConfig,
    SgdPGDConfig,
    SourcePoolConfig,
    TargetedPDConfig,
)
from param_decomp.core.flops.optimizer import (
    OptimizerFlops,
    newton_schulz_flops,
    parameter_optimizer_flops,
    prepare_optimizer_flops,
)
from param_decomp.core.flops.types import MatrixParameters, ParameterCensus
from param_decomp.core.model import Positioned, Positionless
from param_decomp.core.schedule import Knot, ScheduleConfig

SCHEDULE = ScheduleConfig.constant(0.001)
SITES = (
    SiteSpec("first", DenseFactorization(d_in=3, d_out=4, C=2), "first"),
    SiteSpec("second", DenseFactorization(d_in=4, d_out=3, C=3), "second"),
)
SELECTED_SITES = (
    SITES[0],
    SiteSpec(
        "experts", BlockedFactorization(n_blocks=8, d_in=4, d_out=4, c_per_block=3), "experts"
    ),
)
DENSE_ARCH = ChunkwiseTransformerCIFnArch(
    chunks=(Chunk(("input",), ("first", "second")),),
    input_dim=3,
    d_model=8,
    n_blocks=2,
    attention=GQACIFnAttention(2, 1, "xla", "causal"),
    ffn_hidden=7,
    ffn_kind="swiglu",
    learned_norm_scale=True,
)
GLOBAL_ARCH = GlobalTransformerCIFnArch(
    input_taps=(TapSpec("first", 3), TapSpec("second", 4)),
    d_model=8,
    n_blocks=2,
    attention=DENSE_ARCH.attention,
    ffn_hidden=7,
    ffn_kind="swiglu",
    learned_norm_scale=True,
)
SELECTED_ARCH = BlockSelectedChunkwiseTransformerCIFnArch(
    chunks=(
        BlockSelectedChunk(("input",), (0, 1), (FullSlot("first"), SelectedSlot("experts", 0))),
    ),
    input_dim=3,
    d_model=8,
    n_blocks=2,
    attention=DENSE_ARCH.attention,
    table_size=8,
    selected_ffn_hidden=5,
    shared_ffn_hidden=6,
    learned_norm_scale=True,
    expert_implementation="dense_masked",
)


@pytest.mark.parametrize(
    ("arch", "sites"),
    [
        (LayerwiseMLPCIFnArch((5, 7), True, ("first", "second")), SITES),
        (GlobalMLPCIFnArch((5, 7), True, (TapSpec("first", 3), TapSpec("second", 4))), SITES),
        (DENSE_ARCH, SITES),
        (replace(DENSE_ARCH, n_blocks=0), SITES),
        (replace(DENSE_ARCH, ffn_kind="gelu", learned_norm_scale=False), SITES),
        (SELECTED_ARCH, SELECTED_SITES),
        (GLOBAL_ARCH, SITES),
        (replace(GLOBAL_ARCH, ffn_kind="gelu", learned_norm_scale=False), SITES),
        (
            ConditionedCIFnArch(
                GLOBAL_ARCH,
                (SiteInput("first", "first"), SiteInput("second", "second")),
                output_scale_init=0.5,
                calibration=InputScaleCalibration(min_n_tokens=64),
            ),
            SITES,
        ),
    ],
)
def test_census_matches_real_parameter_trees(
    arch: ChunkwiseTransformerCIFnArch
    | BlockSelectedChunkwiseTransformerCIFnArch
    | LayerwiseMLPCIFnArch
    | GlobalMLPCIFnArch
    | GlobalTransformerCIFnArch
    | ConditionedCIFnArch[GlobalTransformerCIFnArch],
    sites: tuple[SiteSpec, ...],
):
    network = arch.initialize(sites, None, jax.random.key(0))
    leaves = [
        leaf
        for path, leaf in jax.tree_util.tree_flatten_with_path(
            eqx.filter(network, eqx.is_inexact_array)
        )[0]
        if path[-1] != jax.tree_util.GetAttrKey("inv_freq")
    ]
    match arch:
        case ChunkwiseTransformerCIFnArch() | BlockSelectedChunkwiseTransformerCIFnArch():
            expected_n_matrix_parameters = sum(leaf.size for leaf in leaves if leaf.ndim in (3, 4))
        case LayerwiseMLPCIFnArch() | GlobalMLPCIFnArch():
            expected_n_matrix_parameters = sum(leaf.size for leaf in leaves if leaf.ndim == 2)
        case GlobalTransformerCIFnArch() | ConditionedCIFnArch():
            # Its depth-stacked biases share the rank of its unstacked input projection.
            expected_n_matrix_parameters = sum(
                matrix.value.size for matrix in network.matrix_parameters()
            )
    expected_n_vector_parameters = sum(leaf.size for leaf in leaves) - expected_n_matrix_parameters
    census = arch.parameter_census(sites)
    assert sum(matrix.n_parameters for matrix in census.matrices) == expected_n_matrix_parameters
    assert census.n_vector_parameters == expected_n_vector_parameters


def test_components_preserve_independent_expert_matrices():
    census = component_parameters(SELECTED_SITES)
    assert census.matrices == (
        MatrixParameters(3, 2, 1),
        MatrixParameters(2, 4, 1),
        MatrixParameters(4, 3, 8),
        MatrixParameters(3, 4, 8),
    )


def test_dead_final_residual_parameters_are_excluded():
    arch = replace(
        SELECTED_ARCH,
        chunks=(BlockSelectedChunk(("input",), (0, 1), (SelectedSlot("experts", 0),)),),
    )
    census = arch.parameter_census(SELECTED_SITES[1:])
    full = SELECTED_ARCH.parameter_census(SELECTED_SITES)
    removed_dense_head = 8 * 2 + 2
    removed_shared_ffn = 3 * 8 * 6
    removed_expert_ffn = (3 * 2 - 2) * 8 * 8 * 5
    assert (
        full.n_parameters - census.n_parameters
        == removed_dense_head + removed_shared_ffn + removed_expert_ffn
    )


def test_symmetric_polynomial_matches_production_newton_schulz():
    x = np.random.default_rng(4).normal(size=(3, 7)).astype(np.float32)
    coefficients = np.array([3.4445, -4.7750, 2.0315], dtype=np.float32)
    expected = np.asarray(
        orthogonalize_via_newton_schulz(jnp.asarray(x), jnp.asarray(coefficients), ns_steps=3)
    )
    economical = x / (np.linalg.norm(x) + 1e-8)
    for _ in range(3):
        gram = economical @ economical.T
        polynomial = coefficients[1] * gram + coefficients[2] * (gram @ gram)
        polynomial += coefficients[0] * np.eye(3, dtype=np.float32)
        economical = polynomial @ economical
    np.testing.assert_allclose(economical, expected, rtol=2e-5, atol=2e-6)
    cost = newton_schulz_flops(MatrixParameters(3, 7, 2), 3, "bfloat16", "ns")
    triangular_entries = 6
    expected_contractions = (
        2 * 3 * (2 * triangular_entries * 7 + 2 * triangular_entries * 3 + 2 * 3 * 3 * 7)
    )
    assert cost.contraction_flops == expected_contractions
    assert cost == newton_schulz_flops(MatrixParameters(7, 3, 2), 3, "bfloat16", "ns")
    assert cost.contraction_flops < 2 * 3 * (4 * 3**2 * 7 + 2 * 3**3)


def test_muon_precision_iterations_adam_fallback_and_decay():
    params = ParameterCensus((MatrixParameters(3, 7, 2),), 5)
    opt = MuonOptimizerConfig(
        type="muon", lr_schedule=SCHEDULE, ns_steps=3, ns_dtype="bfloat16", weight_decay=0.1
    )
    terms = parameter_optimizer_flops("test", opt, params)
    update, ns = terms
    assert update.elementwise_flops == 9 * 42 + 15 * 5
    assert update.dtype == "float32"
    assert ns.dtype == "bfloat16"
    twice = parameter_optimizer_flops("test", opt.model_copy(update={"ns_steps": 6}), params)[1]
    assert twice.contraction_flops == 2 * ns.contraction_flops
    assert twice.elementwise_flops < 2 * ns.elementwise_flops
    adam = OptimizerFlops(
        parameter_optimizer_flops(
            "test", AdamWOptimizerConfig(lr_schedule=SCHEDULE, grad_clip_norm=1.0), params
        )
    )
    assert adam.contraction_flops == 0
    assert adam.elementwise_flops == 12 * 47 + 3 * 47 + 2


def _pd(
    loss: PersistentPGDReconLossConfig
    | MergedStochasticSubsetPooledPPGDReconLossConfig
    | MergedStochasticSubsetPPGDReconLossConfig,
) -> PDConfig:
    return PDConfig(
        components_optimizer=AdamWOptimizerConfig(lr_schedule=SCHEDULE),
        ci_fn_optimizer=AdamWOptimizerConfig(lr_schedule=SCHEDULE),
        batch_size=4,
        steps=10,
        loss_metrics=[
            FaithfulnessLossConfig(coeff=1.0),
            ImportanceMinimalityLossConfig(coeff=1.0, gamma=SCHEDULE),
            loss,
        ],
    )


@pytest.mark.parametrize(("shape", "n_copies"), [("bc", 4), ("bsc", 24)])
def test_persistent_source_shapes_and_all_updates(shape: BatchSourceShape, n_copies: int):
    loss = PersistentPGDReconLossConfig(
        coeff=1.0,
        optimizer=SgdPGDConfig(lr_schedule=SCHEDULE),
        source_shape=shape,
        n_warmup_steps=2,
    )
    cost = prepare_optimizer_flops(_pd(loss), SITES, DENSE_ARCH, Positioned(6))(9)
    sources = sum(term.total for term in cost.terms if term.name.startswith("adversary/"))
    assert sources == 2 * n_copies * sum(site.C + 1 for site in SITES) * 3
    if shape == "bsc":
        with pytest.raises(ValueError, match="positioned"):
            prepare_optimizer_flops(_pd(loss), SITES, DENSE_ARCH, Positionless())


@pytest.mark.parametrize(
    ("optimizer", "active_cost", "inactive_cost"),
    [
        (SgdPGDConfig(lr_schedule=SCHEDULE), 2, 0),
        (MomentumSgdPGDConfig(lr_schedule=SCHEDULE, momentum=0.9), 4, 3),
        (AdamPGDConfig(lr_schedule=SCHEDULE), 12, 7),
    ],
)
def test_pooled_sources_keep_unsampled_momentum_evolution(
    optimizer: SgdPGDConfig | MomentumSgdPGDConfig | AdamPGDConfig,
    active_cost: int,
    inactive_cost: int,
):
    loss = MergedStochasticSubsetPooledPPGDReconLossConfig(
        coeff=1.0,
        optimizer=optimizer,
        pool=SourcePoolConfig(size_per_batch_element=5),
        adv_fraction=ScheduleConfig.constant(1.0),
        n_warmup_steps=2,
    )
    cost = prepare_optimizer_flops(_pd(loss), SITES, DENSE_ARCH, Positioned(6))(9)
    sources = sum(term.total for term in cost.terms if term.name.startswith("adversary/"))
    n_active_updates = 4 * sum(site.C + 1 for site in SITES)
    assert sources == 3 * n_active_updates * (active_cost + 4 * inactive_cost)


def test_source_storage_precision_does_not_change_mathematical_work():
    loss = PersistentPGDReconLossConfig(
        coeff=1.0,
        optimizer=AdamPGDConfig(lr_schedule=SCHEDULE),
        source_shape="bc",
        source_dtype="bfloat16",
    )
    cost = prepare_optimizer_flops(_pd(loss), SITES, DENSE_ARCH, Positioned(6))(9)
    source_terms = [term for term in cost.terms if term.name.startswith("adversary/")]
    assert [term.dtype for term in source_terms] == ["bfloat16", "float32"]
    assert [term.elementwise_flops for term in source_terms] == [8 * 28, 4 * 28]


def test_targeted_ci_scaled_decay_counts_only_arithmetic():
    config = TargetedPDConfig(
        components_optimizer=AdamWOptimizerConfig(lr_schedule=SCHEDULE),
        ci_fn_optimizer=AdamWOptimizerConfig(lr_schedule=SCHEDULE),
        batch_size=4,
        steps=10,
        ci_scaled_weight_decay=0.1,
        loss_metrics=[
            ImportanceMinimalityLossConfig(coeff=1.0, gamma=SCHEDULE),
            CIMaskedReconLossConfig(coeff=1.0),
        ],
    )
    cost = prepare_optimizer_flops(config, SITES, DENSE_ARCH, Positioned(6))(9)
    decay = next(term for term in cost.terms if term.name == "components/ci_scaled_weight_decay")
    assert decay.elementwise_flops == component_parameters(SITES).n_parameters + 2 * sum(
        site.C for site in SITES
    )
    assert decay.contraction_flops == 0


@pytest.mark.parametrize(
    ("shape", "n_copies", "probability"),
    [
        ("bc", 4, 0.5),
        ("bsc", 24, 0.5),
    ],
)
def test_merged_sources_average_final_active_coordinates(
    shape: BatchSourceShape, n_copies: int, probability: float
):
    loss = MergedStochasticSubsetPPGDReconLossConfig(
        coeff=1.0,
        source_shape=shape,
        adv_fraction=ScheduleConfig.constant(0.5),
        optimizer=AdamPGDConfig(lr_schedule=SCHEDULE),
        n_warmup_steps=2,
    )
    cost = prepare_optimizer_flops(_pd(loss), SITES, DENSE_ARCH, Positioned(6))(9)
    width = sum(site.C + 1 for site in SITES)
    n_active_updates = (2 + probability) * n_copies * width
    n_inactive_updates = (1 - probability) * n_copies * width
    assert (
        sum(term.total for term in cost.terms if term.name.startswith("adversary/"))
        == 12 * n_active_updates + 7 * n_inactive_updates
    )


@pytest.mark.parametrize(("step", "probability"), [(0, 0.0), (3, 0.5), (9, 1.0)])
def test_pooled_final_updates_follow_the_adversarial_schedule(step: int, probability: float):
    loss = MergedStochasticSubsetPooledPPGDReconLossConfig(
        coeff=1.0,
        pool=SourcePoolConfig(size_per_batch_element=5),
        adv_fraction=ScheduleConfig(
            max_val=1.0,
            points=(Knot(at=0.0, frac=0.0), Knot(at=2 / 3, frac=1.0), Knot(at=1.0, frac=1.0)),
        ),
        optimizer=AdamPGDConfig(lr_schedule=SCHEDULE),
        n_warmup_steps=2,
    )
    config = _pd(loss)
    cost = prepare_optimizer_flops(config, SITES, DENSE_ARCH, Positioned(6))(step)
    width = sum(site.C + 1 for site in SITES)
    n_active_updates = (2 + probability) * 4 * width
    n_inactive_updates = 3 * 5 * 4 * width - n_active_updates
    assert (
        sum(term.total for term in cost.terms if term.name.startswith("adversary/"))
        == 12 * n_active_updates + 7 * n_inactive_updates
    )


def test_step_evaluation_reuses_parameter_counts_and_source_geometry():
    loss = MergedStochasticSubsetPPGDReconLossConfig(
        coeff=1.0,
        source_shape="bsc",
        optimizer=AdamPGDConfig(lr_schedule=SCHEDULE),
        adv_fraction=ScheduleConfig(
            max_val=1.0,
            points=(Knot(at=0.0, frac=0.0), Knot(at=1.0, frac=1.0)),
        ),
        n_warmup_steps=2,
    )
    config = _pd(loss).model_copy(
        update={"components_optimizer": MuonOptimizerConfig(type="muon", lr_schedule=SCHEDULE)}
    )
    module = "param_decomp.core.flops.optimizer"
    with (
        patch(f"{module}.component_parameters", wraps=component_parameters) as components,
        patch.object(type(DENSE_ARCH), "parameter_census", wraps=DENSE_ARCH.parameter_census) as ci,
        patch(f"{module}.parameter_optimizer_flops", wraps=parameter_optimizer_flops) as updates,
        patch(f"{module}.newton_schulz_flops", wraps=newton_schulz_flops) as newton_schulz,
    ):
        at_step = prepare_optimizer_flops(config, SITES, DENSE_ARCH, Positioned(6))
        components.assert_called_once()
        ci.assert_called_once()
        assert updates.call_count == 2
        assert newton_schulz.call_count == len(component_parameters(SITES).matrices)
        for expensive_work in (components, ci, updates, newton_schulz):
            expensive_work.side_effect = AssertionError("Architecture work repeated during a step")
        with patch(
            f"{module}._n_source_copies",
            side_effect=AssertionError("Source geometry repeated during a step"),
        ):
            initial = at_step(0)
            final = at_step(config.steps - 1)
            assert at_step(0) == initial
            assert at_step(config.steps - 1) == final
    n_source_parameters = 4 * 6 * sum(site.C + 1 for site in SITES)
    assert final.total - initial.total == 5 * n_source_parameters
