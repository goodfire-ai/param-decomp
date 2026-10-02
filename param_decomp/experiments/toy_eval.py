"""Binding of authored evaluation operations for toy targets.

The fast-tier scalars come from the shared target-generic binder
(`experiments/fast_eval_operations.py`); only the UV figures are toy-owned, because they
read the toy's single-feature CI probe.
"""

from collections.abc import Callable

import numpy as np
from jax.sharding import Mesh
from jaxtyping import Array

from param_decomp.core.built_run import BuiltRun, TargetSites
from param_decomp.core.configs import (
    CI_L0Config,
    CIHistogramsConfig,
    CIMeanPerComponentConfig,
    ComponentActivationDensityConfig,
    EvalPGDReconLossConfig,
    IdentityCIErrorConfig,
    PDConfig,
    PermutedCIPlotsConfig,
    SlowPGDReconLossConfig,
    UVPlotsConfig,
)
from param_decomp.core.eval_schedule import EvalSchedule
from param_decomp.core.metrics import LogRecord
from param_decomp.core.model import CaptureKeys, ComponentActivations, PlacedModel
from param_decomp.core.run import EvalInvocation, StandaloneOperation, StandaloneOperationPlan
from param_decomp.core.train import Decomposition
from param_decomp.experiments import toy_uv_eval
from param_decomp.experiments.eval_config import EvalConfig, schedule_for
from param_decomp.experiments.fast_eval_operations import (
    make_ci_l0_operation,
    make_fresh_pgd_operation,
)
from param_decomp.experiments.lm.eval_config import (
    ArithmeticCIGridConfig,
    CEandKLLossesConfig,
    CIActiveCountsPerPositionConfig,
    CIMaskedAttnPatternsReconLossConfig,
    RouterDivergenceConfig,
    StochasticAttnPatternsReconLossConfig,
    WellTemperednessConfig,
)
from param_decomp.experiments.toy_config import ToyCIFnArch

type ToyRun[TargetT: TargetSites] = BuiltRun[None, TargetT, PDConfig, ToyCIFnArch]
type ProbeCI[Conditioning] = Callable[[Decomposition[Conditioning]], dict[str, Array]]


def _make_uv_plots_operation[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    metric: UVPlotsConfig,
    schedule: EvalSchedule,
    model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    probe_ci: ProbeCI[Conditioning],
    wandb_configured: bool,
) -> StandaloneOperationPlan[EvalInvocation[Conditioning]]:
    assert wandb_configured, "UVPlots requires a configured wandb transport"
    spec = toy_uv_eval.toy_uv_spec(model, metric)

    def run(context: EvalInvocation[Conditioning]) -> LogRecord:
        return toy_uv_eval.render_uv_metric(
            spec,
            context.decomposition.components,
            probe_ci(context.decomposition),
        )

    return StandaloneOperationPlan(lambda _example: StandaloneOperation(schedule, run))


def make_toy_evaluation_operations[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    eval_config: EvalConfig,
    seed: int,
    compiler_options: dict[str, bool | int | str],
    model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    ci_capture_keys: CaptureKeys,
    mesh: Mesh,
    sample_eval_batch: Callable[[np.uint32], TargetIn],
    probe_ci: ProbeCI[Conditioning],
    wandb_configured: bool,
) -> tuple[StandaloneOperationPlan[EvalInvocation[Conditioning]], ...]:
    """Exhaustively bind each authored toy metric to one executable operation."""
    operations: list[StandaloneOperationPlan[EvalInvocation[Conditioning]]] = []
    for metric in eval_config.metrics:
        schedule = schedule_for(metric, eval_config)
        match metric:
            case EvalPGDReconLossConfig() | SlowPGDReconLossConfig():
                operation = make_fresh_pgd_operation(
                    metric,
                    eval_config,
                    schedule,
                    seed,
                    compiler_options,
                    model,
                    ci_capture_keys,
                    mesh,
                    sample_eval_batch,
                )
            case CI_L0Config():
                operation = make_ci_l0_operation(
                    metric,
                    eval_config,
                    schedule,
                    seed,
                    compiler_options,
                    model,
                    ci_capture_keys,
                    mesh,
                    sample_eval_batch,
                )
            case UVPlotsConfig():
                operation = _make_uv_plots_operation(
                    metric, schedule, model, probe_ci, wandb_configured
                )
            case CEandKLLossesConfig():
                raise AssertionError(
                    "CEandKLLosses scores next-token cross-entropy and KL over a categorical "
                    "output distribution; a toy target emits neither tokens nor logits"
                )
            case WellTemperednessConfig():
                raise AssertionError(
                    "WellTemperedness ablates components at token positions of an LM; a "
                    "positionless toy target has no positions to ablate at"
                )
            case (
                ArithmeticCIGridConfig()
                | CIActiveCountsPerPositionConfig()
                | CIHistogramsConfig()
                | CIMaskedAttnPatternsReconLossConfig()
                | CIMeanPerComponentConfig()
                | ComponentActivationDensityConfig()
                | IdentityCIErrorConfig()
                | PermutedCIPlotsConfig()
                | RouterDivergenceConfig()
                | StochasticAttnPatternsReconLossConfig()
            ):
                raise AssertionError(f"eval metric {metric.type!r} has no toy binding")
        operations.append(operation)
    return tuple(operations)
