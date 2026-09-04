"""Slowness is a property of an eval, declared in code beside the eval itself.

Pins the three things that makes true: every authored metric states a tier, no config seat
can restate it, and a metric's schedule comes from its own declaration rather than from
whichever family happened to bind it.
"""

from typing import ClassVar, Literal

import pytest
from pydantic import ValidationError

from param_decomp.core.base_config import BaseConfig
from param_decomp.core.configs import (
    CI_L0Config,
    CIHistogramsConfig,
    PGDReconLossConfig,
    UVPlotsConfig,
)
from param_decomp.core.eval_schedule import Every, FirstThenEvery, eval_due
from param_decomp.experiments.eval_config import (
    EVAL_METRIC_CONFIG_TYPES,
    AnyEvalMetricConfig,
    EvalConfig,
    assert_every_metric_declares_its_tier,
    schedule_for,
)

FAST_METRICS = {
    "CEandKLLossesConfig",
    "CIMaskedAttnPatternsReconLossConfig",
    "CI_L0Config",
    "PGDReconLossConfig",
    "StochasticAttnPatternsReconLossConfig",
}
SLOW_METRICS = {
    "ArithmeticCIGridConfig",
    "CIHistogramsConfig",
    "CIMeanPerComponentConfig",
    "ComponentActivationDensityConfig",
    "IdentityCIErrorConfig",
    "PermutedCIPlotsConfig",
    "UVPlotsConfig",
    "WellTemperednessConfig",
}


def _eval_config(*metrics: AnyEvalMetricConfig, slow_on_first_step: bool = True) -> EvalConfig:
    return EvalConfig(
        batch_size=8,
        n_steps=1,
        every=1000,
        slow_every=5000,
        slow_on_first_step=slow_on_first_step,
        metrics=list(metrics),
    )


def test_the_declared_tier_of_every_authored_metric() -> None:
    declared = {cls.__name__: cls.slow for cls in EVAL_METRIC_CONFIG_TYPES}  # pyright: ignore[reportAttributeAccessIssue]
    assert {name for name, slow in declared.items() if not slow} == FAST_METRICS
    assert {name for name, slow in declared.items() if slow} == SLOW_METRICS


def test_a_fast_and_a_slow_metric_authored_together_get_different_schedules() -> None:
    fast = CI_L0Config(groups=None)
    slow = CIHistogramsConfig(n_batches_accum=None)
    eval_config = _eval_config(fast, slow)

    assert schedule_for(fast, eval_config) == Every(1000)
    assert schedule_for(slow, eval_config) == FirstThenEvery(0, 5000)


def test_the_slow_tier_skips_the_first_pass_when_unasked() -> None:
    slow = CIHistogramsConfig(n_batches_accum=None)
    assert schedule_for(slow, _eval_config(slow, slow_on_first_step=False)) == Every(5000)


def test_slow_on_first_step_is_the_untrained_baseline_not_the_first_eval_pass() -> None:
    """The flag's whole point is a pre-training readout; landing it on the first `every`
    pass instead would report an already-trained model and defer any eval-path blowup by
    `every` steps."""
    slow = CIHistogramsConfig(n_batches_accum=None)
    schedule = schedule_for(slow, _eval_config(slow))

    assert eval_due(schedule, 0)
    assert not eval_due(schedule, 1000)
    assert eval_due(schedule, 5000)
    assert not eval_due(Every(1000), 0)
    assert not eval_due(schedule_for(slow, _eval_config(slow, slow_on_first_step=False)), 0)


def test_the_tier_travels_with_the_metric_across_families() -> None:
    """UVPlots is slow wherever it is bound; the toy and LM binders read the same
    declaration rather than each assigning a tier of their own."""
    assert UVPlotsConfig.slow and not PGDReconLossConfig.slow


def test_a_seat_cannot_author_its_own_tier() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        UVPlotsConfig.model_validate(
            {"identity_patterns": None, "dense_patterns": None, "slow": False}
        )


def test_slow_every_must_land_on_a_fast_eval_step() -> None:
    with pytest.raises(ValidationError, match="must be a multiple of every"):
        EvalConfig(batch_size=8, n_steps=1, every=1000, slow_every=1500)


class _UndeclaredConfig(BaseConfig):
    type: Literal["Undeclared"] = "Undeclared"


class _InheritingConfig(CIHistogramsConfig):
    type: Literal["Inheriting"] = "Inheriting"  # pyright: ignore[reportIncompatibleVariableOverride]


class _RedeclaringConfig(CIHistogramsConfig):
    slow: ClassVar[bool] = True
    type: Literal["Redeclaring"] = "Redeclaring"  # pyright: ignore[reportIncompatibleVariableOverride]


@pytest.mark.parametrize("metric_type", [_UndeclaredConfig, _InheritingConfig])
def test_a_metric_that_does_not_state_its_tier_refuses(metric_type: type[BaseConfig]) -> None:
    """Inheriting a tier counts as not stating one: the sweep is what stops a new metric
    from silently taking a sibling's answer."""
    with pytest.raises(AssertionError, match=metric_type.__name__):
        assert_every_metric_declares_its_tier([metric_type])


def test_the_sweep_accepts_a_metric_that_states_its_own_tier() -> None:
    assert_every_metric_declares_its_tier([_RedeclaringConfig])
