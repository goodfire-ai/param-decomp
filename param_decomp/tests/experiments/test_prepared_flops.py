"""Prepared step costs preserve scheduled objectives and adversarial fractions."""

from functools import partial
from typing import Literal
from unittest.mock import patch

import pytest

from param_decomp.core.ci_fn.implementations.layerwise_mlp import LayerwiseMLPCIFnArch
from param_decomp.core.components import SiteC
from param_decomp.core.configs import (
    AdamPGDConfig,
    AdamWOptimizerConfig,
    AuxiliaryReconstructionConfig,
    CaptureReconstruction,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    MergedStochasticSubsetPPGDReconLossConfig,
    MomentumSgdPGDConfig,
    PDConfig,
    SgdPGDConfig,
)
from param_decomp.core.flops.model import (
    decomposition_step_flops,
    reconstruction_plans,
)
from param_decomp.core.flops.optimizer import prepare_optimizer_flops
from param_decomp.core.model import Positioned
from param_decomp.core.schedule import Interp, Knot, ScheduleConfig, get_scheduled_value
from param_decomp.experiments.model_flops import (
    StreamDescription,
    _objective_change_steps,
    prepare_ordinary_step_flops,
    stream_flops,
)
from param_decomp.targets.transformer import (
    GLU_ANATOMY,
    TransformerConfig,
    glu_site_specs,
    transformer_flops,
)


@pytest.mark.parametrize(
    ("n_steps", "scheduled_auxiliary"), [(1, True), (9, True), (10**12, True), (9, False)]
)
@pytest.mark.parametrize("optimizer", ["adam", "sgd", "momentum"])
def test_prepared_lm_costs_match_independent_step_accounting(
    n_steps: int,
    scheduled_auxiliary: bool,
    optimizer: Literal["adam", "sgd", "momentum"],
) -> None:
    rate = ScheduleConfig.constant(0.001)
    match optimizer:
        case "adam":
            source_optimizer = AdamPGDConfig(lr_schedule=rate)
        case "sgd":
            source_optimizer = SgdPGDConfig(lr_schedule=rate)
        case "momentum":
            source_optimizer = MomentumSgdPGDConfig(lr_schedule=rate, momentum=0.9)
    probability = ScheduleConfig(
        max_val=1,
        points=(
            Knot(at=0, frac=0),
            Knot(at=0.5, frac=1),
            Knot(at=1, frac=0, interp="cosine"),
        ),
    )
    auxiliary = ScheduleConfig(
        max_val=1,
        points=(
            Knot(at=0, frac=0),
            Knot(at=0.25, frac=1, interp="hold"),
            Knot(at=0.75, frac=0, interp="hold"),
            Knot(at=1, frac=0),
        ),
    )
    reconstruction = MergedStochasticSubsetPPGDReconLossConfig(
        coeff=1,
        optimizer=source_optimizer,
        n_warmup_steps=2,
        source_shape="bsc",
        adv_fraction=probability,
        adversary_objective="e2e",
        auxiliaries=(
            AuxiliaryReconstructionConfig(
                name="hidden",
                coeff=auxiliary if scheduled_auxiliary else 1.0,
                comparisons=(
                    CaptureReconstruction(
                        capture="layers.0.mlp.gate_proj.out", distance="relative_squared_error"
                    ),
                ),
            ),
            AuxiliaryReconstructionConfig(
                name="output",
                coeff=probability if scheduled_auxiliary else 1.0,
                comparisons=(
                    CaptureReconstruction(
                        capture="layers.0.mlp.down_proj.out", distance="relative_squared_error"
                    ),
                ),
            ),
        ),
    )
    pd = PDConfig(
        steps=n_steps,
        batch_size=4,
        components_optimizer=AdamWOptimizerConfig(lr_schedule=rate),
        ci_fn_optimizer=AdamWOptimizerConfig(lr_schedule=rate),
        loss_metrics=[
            FaithfulnessLossConfig(coeff=1),
            ImportanceMinimalityLossConfig(coeff=1, gamma=rate),
            reconstruction,
        ],
    )
    cfg = TransformerConfig(
        vocab_size=20,
        n_layer=1,
        n_head=2,
        n_kv_head=1,
        n_embd=8,
        n_intermediate=12,
        head_dim=4,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        max_position_embeddings=16,
        tie_word_embeddings=False,
    )
    sites = glu_site_specs(
        cfg, (SiteC("layers.0.mlp.gate_proj", 7), SiteC("layers.0.mlp.down_proj", 6))
    )
    ci = LayerwiseMLPCIFnArch(
        hidden_dims=(8,),
        has_position_axis=True,
        input_names=("mlp_in.0", "mlp_hidden.0"),
    )
    positions = Positioned(3)
    stream = StreamDescription(
        partial(transformer_flops, cfg, GLU_ANATOMY, sequence_length=3),
        sites,
        ci,
        pd.batch_size,
        positions,
        None,
        "",
    )
    with patch(
        "param_decomp.experiments.model_flops.get_scheduled_value", wraps=get_scheduled_value
    ) as evaluate_schedule:
        compute = prepare_ordinary_step_flops(pd, stream)
        assert evaluate_schedule.call_count < 500
    for invalid_step in (-1, n_steps):
        with pytest.raises(ValueError, match="within the configured run"):
            compute(invalid_step)
    steps = sorted(
        {
            0,
            n_steps - 1,
            *(
                step
                for quarter in range(5)
                for step in (
                    quarter * (n_steps - 1) // 4 - 1,
                    quarter * (n_steps - 1) // 4,
                    quarter * (n_steps - 1) // 4 + 1,
                )
                if 0 <= step < n_steps
            ),
        }
    )
    expected = {
        step: (
            decomposition_step_flops(
                pd, sites, stream_flops(stream, reconstruction_plans(pd, step))
            ).total,
            prepare_optimizer_flops(pd, sites, ci, positions)(step).total,
        )
        for step in steps
    }
    with patch(
        "param_decomp.experiments.model_flops.stream_flops",
        side_effect=AssertionError("Model work during a step"),
    ):
        for step, (model_flops, optimizer_flops) in expected.items():
            assert compute(step).model == pytest.approx(model_flops)
            assert compute(step).optimizer == pytest.approx(optimizer_flops)


@pytest.mark.parametrize("n_steps", [1, 2, 9, 103])
@pytest.mark.parametrize("interpolation", ["linear", "cosine", "hold"])
def test_prepared_objective_changes_match_host_schedules(
    n_steps: int, interpolation: Interp
) -> None:
    schedules = tuple(
        ScheduleConfig(
            max_val=1,
            points=tuple(
                Knot(at=at, frac=frac, interp=interpolation)
                for at, frac in zip(positions, (0, 1, 0, 1), strict=True)
            ),
        )
        for positions in ((0, 0.3, 0.8, 1), (0, 1 / 3, 2 / 3, 1))
    )
    expected = {0}
    for schedule in schedules:
        activity = [get_scheduled_value(step, n_steps, schedule) > 0 for step in range(n_steps)]
        expected.update(step for step in range(1, n_steps) if activity[step] != activity[step - 1])
    assert _objective_change_steps(schedules, n_steps) == tuple(sorted(expected))


def test_long_cosine_schedule_preserves_rounded_zero_plateau() -> None:
    schedule = ScheduleConfig(
        max_val=1,
        points=(
            Knot(at=0, frac=0),
            Knot(at=1, frac=1, interp="cosine"),
        ),
    )
    n_steps = 10**12
    with patch(
        "param_decomp.experiments.model_flops.get_scheduled_value", wraps=get_scheduled_value
    ) as evaluate_schedule:
        start, first_active = _objective_change_steps((schedule,), n_steps)
        assert evaluate_schedule.call_count < 100
    assert start == 0
    assert first_active > 1
    assert get_scheduled_value(first_active - 1, n_steps, schedule) == 0
    assert get_scheduled_value(first_active, n_steps, schedule) > 0
