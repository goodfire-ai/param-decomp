"""The authored eval-operation union shared by experiment schemas.

Operation schemas live with the layer that owns their semantics; this module only
assembles the closed authoring language used for discriminated parsing.
"""

from collections.abc import Iterable
from typing import Annotated, Self, get_args

from pydantic import Discriminator, Field, PositiveInt, model_validator

from param_decomp.core.base_config import BaseConfig
from param_decomp.core.configs import (
    CI_L0Config,
    CIHistogramsConfig,
    CIMeanPerComponentConfig,
    ComponentActivationDensityConfig,
    EvalPGDReconLossConfig,
    IdentityCIErrorConfig,
    PermutedCIPlotsConfig,
    ReconstructionAuxiliariesMixin,
    SlowPGDReconLossConfig,
    UVPlotsConfig,
)
from param_decomp.core.eval_schedule import EvalSchedule, Every, FirstThenEvery
from param_decomp.experiments.lm.eval_config import (
    ArithmeticCIGridConfig,
    CEandKLLossesConfig,
    CIActiveCountsPerPositionConfig,
    CIMaskedAttnPatternsReconLossConfig,
    RouterDivergenceConfig,
    StochasticAttnPatternsReconLossConfig,
    WellTemperednessConfig,
)

AnyEvalMetricConfig = Annotated[
    ArithmeticCIGridConfig
    | CEandKLLossesConfig
    | CIActiveCountsPerPositionConfig
    | CIHistogramsConfig
    | CI_L0Config
    | CIMaskedAttnPatternsReconLossConfig
    | CIMeanPerComponentConfig
    | ComponentActivationDensityConfig
    | IdentityCIErrorConfig
    | PermutedCIPlotsConfig
    | EvalPGDReconLossConfig
    | RouterDivergenceConfig
    | SlowPGDReconLossConfig
    | StochasticAttnPatternsReconLossConfig
    | UVPlotsConfig
    | WellTemperednessConfig,
    Discriminator("type"),
]

EVAL_METRIC_CONFIG_TYPES: tuple[type[BaseConfig], ...] = get_args(get_args(AnyEvalMetricConfig)[0])


def assert_every_metric_declares_its_tier(metric_types: Iterable[type[BaseConfig]]) -> None:
    """Each metric must declare `slow` ITSELF, not inherit it: a metric that silently picks
    up a shared base's tier has had the decision made for it by an unrelated sibling."""
    for metric_type in metric_types:
        assert "slow" in vars(metric_type), (
            f"{metric_type.__name__} must declare `slow: ClassVar[bool]` beside its own "
            "definition — an eval's cost is a property of the eval, not of a config seat"
        )


assert_every_metric_declares_its_tier(EVAL_METRIC_CONFIG_TYPES)


def validate_eval_metrics(metrics: list[AnyEvalMetricConfig]) -> None:
    """Validate invariants shared by every authored eval operation list.

    Every metric type logs under keys of its own, so listing each type at most once keeps
    every metric's keys distinct.
    """
    types = [metric.type for metric in metrics]
    assert len(types) == len(set(types)), f"eval.metrics lists a metric type twice: {types}"
    for metric in metrics:
        if isinstance(metric, ReconstructionAuxiliariesMixin):
            for auxiliary in metric.auxiliaries:
                assert isinstance(auxiliary.coeff, float), (
                    f"eval metric {metric.type}: {auxiliary.name}.coeff must be a constant float "
                    "— an eval probe has no training step for a schedule to read"
                )


class EvalConfig(BaseConfig):
    """One evaluation callback's data cadence and requested operations.

    Two cadences, one per tier: `every` for the metrics that declare themselves fast,
    `slow_every` for the rest. `slow_on_first_step` fires the slow tier once at step 0 —
    before the first optimizer step — for an untrained baseline against the end state.
    Only a fresh run has a step 0; a resumed one starts past it.
    """

    batch_size: PositiveInt
    n_steps: PositiveInt
    every: PositiveInt
    slow_every: PositiveInt
    slow_on_first_step: bool = True
    metrics: list[AnyEvalMetricConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_eval_contract(self) -> Self:
        validate_eval_metrics(self.metrics)
        assert self.slow_every % self.every == 0, (
            f"slow_every ({self.slow_every}) must be a multiple of every ({self.every}): the "
            "slow tier reuses the fast pass's batches, so it can only fire on a fast-eval step"
        )
        return self


def slow_schedule(eval_config: EvalConfig) -> EvalSchedule:
    """The callback's shared schedule for every slow or standing-slow operation."""
    return (
        FirstThenEvery(0, eval_config.slow_every)
        if eval_config.slow_on_first_step
        else Every(eval_config.slow_every)
    )


def schedule_for(metric: AnyEvalMetricConfig, eval_config: EvalConfig) -> EvalSchedule:
    """When `metric` fires: its own declared tier read against this callback's cadences."""
    return slow_schedule(eval_config) if metric.slow else Every(eval_config.every)
