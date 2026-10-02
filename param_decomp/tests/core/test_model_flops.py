"""The useful-work budget follows training passes, including adversarial reuse."""

import numpy as np
import pytest

from param_decomp.core.components import BlockedFactorization, DenseFactorization, SiteSpec
from param_decomp.core.configs import (
    AdamWOptimizerConfig,
    AllRoutingConfig,
    AnyReconLossMetricConfig,
    AuxiliaryReconstructionConfig,
    CaptureReconstruction,
    CIMaskedReconLossConfig,
    CIMaskedReconSubsetLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    MergedStochasticSubsetPooledPPGDReconLossConfig,
    MergedStochasticSubsetPPGDReconLossConfig,
    NonlinearityLocalityLossConfig,
    NontargetConfig,
    PDConfig,
    PersistentPGDReconLossConfig,
    PGDReconLossConfig,
    PGDReconSubsetLossConfig,
    SgdPGDConfig,
    SourcePoolConfig,
    StochasticReconLossConfig,
    StochasticReconSubsetLossConfig,
    TargetedPDConfig,
    UnmaskedNoDeltaReconLossConfig,
    UnmaskedReconLossConfig,
)
from param_decomp.core.flops.model import (
    FlopsTerm,
    StreamFlops,
    decomposition_step_flops,
    faithfulness_flops,
    nonlinearity_flops,
    nontarget_reconstruction_plans,
    reconstruction_plan,
    reconstruction_plans,
    targeted_step_flops,
)
from param_decomp.core.flops.types import ForwardBackwardFlops
from param_decomp.core.nonlinearity import Neurons, NonlinearityAlignment, QueryHeads
from param_decomp.core.schedule import ScheduleConfig

SITES = (SiteSpec("linear", DenseFactorization(d_in=3, d_out=5, C=7), "linear"),)
STREAM = StreamFlops(
    clean_forward=100,
    ci=ForwardBackwardFlops(20, 30),
    reconstructions=(FlopsTerm("reconstruction", ForwardBackwardFlops(110, 150), 1),),
)
OPTIMIZER = AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(0.001))
IMPORTANCE = ImportanceMinimalityLossConfig(coeff=1.0, gamma=ScheduleConfig.constant(1.0))
SOURCE_OPTIMIZER = SgdPGDConfig(lr_schedule=ScheduleConfig.constant(0.1))
AUXILIARY = AuxiliaryReconstructionConfig(
    name="residual",
    coeff=1.0,
    comparisons=(CaptureReconstruction(capture="residual", distance="relative_squared_error"),),
)


def faithful_config(reconstruction: AnyReconLossMetricConfig) -> PDConfig:
    return PDConfig(
        components_optimizer=OPTIMIZER,
        ci_fn_optimizer=OPTIMIZER,
        batch_size=4,
        steps=10,
        loss_metrics=[FaithfulnessLossConfig(coeff=1.0), IMPORTANCE, reconstruction],
    )


@pytest.mark.parametrize(
    ("reconstruction", "expected"),
    [
        (CIMaskedReconLossConfig(coeff=1.0), 160),
        (CIMaskedReconSubsetLossConfig(coeff=1.0), 260),
        (CIMaskedReconSubsetLossConfig(coeff=1.0, routing=AllRoutingConfig()), 160),
        (UnmaskedReconLossConfig(coeff=1.0), 160),
        (StochasticReconLossConfig(coeff=1.0), 260),
        (StochasticReconSubsetLossConfig(coeff=1.0), 260),
        (
            PGDReconLossConfig(
                coeff=1.0, init="random", step_size=0.1, n_steps=3, source_shape="bc"
            ),
            260 + 3 * 170,
        ),
        (
            PGDReconSubsetLossConfig(
                coeff=1.0, init="ones", step_size=0.1, n_steps=0, source_shape="bc"
            ),
            260,
        ),
    ],
)
def test_reconstruction_passes(reconstruction: AnyReconLossMetricConfig, expected: int) -> None:
    plan = reconstruction_plan(reconstruction, 9, 10)
    assert (260 if plan.include_frozen_paths else 160) + plan.n_ascent_steps * 170 == expected
    assert not plan.retake_source_gradient
    cost = decomposition_step_flops(faithful_config(reconstruction), SITES, STREAM)
    assert cost.total == 150 + 6 * 3 * 5 * 7 + 260
    assert sum(term.total for term in cost.terms) == cost.total


@pytest.mark.parametrize("objective", ["e2e", "term"])
@pytest.mark.parametrize("has_auxiliary", [False, True])
def test_persistent_final_ascent_reuses_main_backward(objective: str, has_auxiliary: bool) -> None:
    loss = PersistentPGDReconLossConfig.model_validate(
        {
            "coeff": 1.0,
            "optimizer": SOURCE_OPTIMIZER,
            "source_shape": "bsc",
            "n_warmup_steps": 2,
            "adversary_objective": objective,
            "auxiliaries": (AUXILIARY,) if has_auxiliary else (),
        }
    )
    plan = reconstruction_plan(loss, 9, 10)
    assert plan.n_ascent_steps == 2
    assert plan.retake_source_gradient == (objective == "e2e" and has_auxiliary)
    assert plan.source_captures == (
        frozenset({"residual"}) if objective == "term" and has_auxiliary else frozenset()
    )


def test_merged_and_pooled_adversaries_each_use_one_reconstruction() -> None:
    merged = MergedStochasticSubsetPPGDReconLossConfig(
        coeff=1.0,
        optimizer=SOURCE_OPTIMIZER,
        source_shape="bc",
        adv_fraction=ScheduleConfig.constant(0.5),
    )
    pooled = MergedStochasticSubsetPooledPPGDReconLossConfig(
        coeff=1.0,
        optimizer=SOURCE_OPTIMIZER,
        pool=SourcePoolConfig(size_per_batch_element=100),
        adv_fraction=ScheduleConfig.constant(0.2),
    )
    for loss in (merged, pooled):
        cost = decomposition_step_flops(faithful_config(loss), SITES, STREAM)
        assert cost.total == 150 + 630 + 260


def test_targeted_training_accounts_for_both_streams_without_faithfulness() -> None:
    pd = TargetedPDConfig(
        components_optimizer=OPTIMIZER,
        ci_fn_optimizer=OPTIMIZER,
        batch_size=4,
        steps=10,
        loss_metrics=[IMPORTANCE, CIMaskedReconLossConfig(coeff=1.0)],
    )
    nontarget = NontargetConfig(
        batch_size=8,
        impmin_coeff=1.0,
        recon=[CIMaskedReconLossConfig(coeff=1.0), UnmaskedNoDeltaReconLossConfig(coeff=1.0)],
    )
    broad = StreamFlops(
        clean_forward=200,
        ci=ForwardBackwardFlops(40, 60),
        reconstructions=(
            FlopsTerm("nontarget/delta", ForwardBackwardFlops(220, 300), 1),
            FlopsTerm("nontarget/no_delta", ForwardBackwardFlops(140, 180), 1),
        ),
    )
    assert len(reconstruction_plans(pd, 9)) == 1
    cost = targeted_step_flops(STREAM, broad)
    assert cost.total == 150 + 260 + 300 + 520 + 320
    assert [plan.include_frozen_paths for plan in nontarget_reconstruction_plans(nontarget)] == [
        True,
        False,
    ]
    assert not any(term.name == "faithfulness" for term in cost.terms)


def test_faithfulness_reconstructs_all_expert_weights() -> None:
    experts = SiteSpec(
        "experts", BlockedFactorization(n_blocks=11, d_in=3, d_out=5, c_per_block=7), "experts"
    )
    assert faithfulness_flops((experts,)) == ForwardBackwardFlops(11 * 210, 11 * 420)


def test_nonlinearity_partition_norms_require_only_reductions() -> None:
    sites = (
        SiteSpec(
            "heads",
            DenseFactorization(d_in=3, d_out=8, C=7),
            "heads",
            alignment=NonlinearityAlignment("output", QueryHeads(2)),
        ),
        SiteSpec(
            "neurons",
            DenseFactorization(d_in=5, d_out=8, C=13),
            "neurons",
            alignment=NonlinearityAlignment("output", Neurons()),
        ),
    )
    loss = NonlinearityLocalityLossConfig(
        coeff=1.0,
        relative_threshold=ScheduleConfig.constant(0.1),
        unit_kind_coefficients={"attention_head": 1.0, "neuron": 1.0},
    )
    assert nonlinearity_flops(loss, sites) == ForwardBackwardFlops(0, 0)


def test_low_rank_faithfulness_uses_equivalent_gram_loss_and_gradients() -> None:
    rng = np.random.default_rng(0)
    v, u, w = rng.normal(size=(17, 2)), rng.normal(size=(2, 13)), rng.normal(size=(17, 13))
    residual = v @ u - w
    vg, ug, wu = v.T @ v, u @ u.T, w @ u.T
    gram_loss = np.sum(vg * ug) - 2 * np.sum(v * wu) + np.sum(w * w)
    assert gram_loss == pytest.approx(np.sum(residual * residual))
    np.testing.assert_allclose(v @ ug - wu, residual @ u.T)
    np.testing.assert_allclose(vg @ u - v.T @ w, v.T @ residual)
    cost = faithfulness_flops(
        (SiteSpec("low_rank", DenseFactorization(d_in=17, d_out=13, C=2), "low_rank"),)
    )
    assert cost.total < 6 * 17 * 2 * 13
    assert cost.forward == (17 + 13) * 2 * 3 + 2 * 17 * 13 * 2


def test_merged_source_gradient_fraction_and_auxiliaries_follow_the_step() -> None:
    schedule = ScheduleConfig.model_validate(
        {
            "max_val": 1.0,
            "points": [{"at": 0.0, "frac": 0.0}, {"at": 1.0, "frac": 1.0}],
        }
    )
    loss = MergedStochasticSubsetPPGDReconLossConfig(
        coeff=1.0,
        optimizer=SOURCE_OPTIMIZER,
        source_shape="bc",
        adv_fraction=schedule,
        adversary_objective="e2e",
        auxiliaries=(AUXILIARY.model_copy(update={"coeff": schedule}),),
    )
    beginning = reconstruction_plan(loss, 0, 3)
    middle = reconstruction_plan(loss, 1, 3)
    assert beginning.auxiliary_captures == frozenset()
    assert not beginning.retake_source_gradient
    assert middle.auxiliary_captures == frozenset({"residual"})
    assert middle.retake_source_gradient
    assert middle.source_gradient_fraction == 0.5
