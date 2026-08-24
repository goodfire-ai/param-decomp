"""Binding of authored evaluation operations for toy targets.

The fast-tier scalars come from the shared target-generic binder
(`experiments/fast_eval_operations.py`); only the UV figures are toy-owned, because they
read the toy's single-feature CI probe.
"""

from collections.abc import Callable

import jax
from jax.sharding import Mesh
from jaxtyping import Array

from param_decomp.core.built_run import BuiltRun, TargetSites
from param_decomp.core.configs import (
    CI_L0Config,
    PDConfig,
    PGDReconLossConfig,
    UVPlotsConfig,
    WellTemperednessConfig,
)
from param_decomp.core.eval_schedule import EvalSchedule
from param_decomp.core.metrics import LogRecord
from param_decomp.core.model import CaptureKeys, PlacedModel
from param_decomp.core.run import EvalInvocation, EvalOperation
from param_decomp.core.train import TrainState
from param_decomp.core.well_temperedness_eval import make_well_temperedness_operation
from param_decomp.experiments import toy_uv_eval
from param_decomp.experiments.eval_config import EvalConfig, schedule_for
from param_decomp.experiments.fast_eval_operations import (
    make_ci_l0_operation,
    make_fresh_pgd_operation,
)
from param_decomp.experiments.lm.eval_config import CEandKLLossesConfig

type ToyRun[TargetT: TargetSites] = BuiltRun[None, TargetT, PDConfig]
type ProbeCI = Callable[[TrainState], dict[str, Array]]


def _make_uv_plots_operation(
    metric: UVPlotsConfig,
    schedule: EvalSchedule,
    model: PlacedModel,
    probe_ci: ProbeCI,
    wandb_configured: bool,
) -> EvalOperation[EvalInvocation]:
    assert wandb_configured, "UVPlots requires a configured wandb transport"
    spec = toy_uv_eval.toy_uv_spec(model, metric)

    def run(context: EvalInvocation) -> LogRecord:
        return toy_uv_eval.render_uv_metric(
            spec,
            dict(context.state.decomposition.components.sites_items()),
            probe_ci(context.state),
        )

    return EvalOperation(schedule, run)


def make_toy_evaluation_operations(
    eval_config: EvalConfig,
    seed: int,
    compiler_options: dict[str, bool | int | str],
    model: PlacedModel,
    ci_capture_keys: CaptureKeys,
    mesh: Mesh,
    sample_eval_batch: Callable[[int], Array],
    probe_ci: ProbeCI,
    wandb_configured: bool,
) -> tuple[EvalOperation[EvalInvocation], ...]:
    """Exhaustively bind each authored toy metric to one executable operation."""
    operations: list[EvalOperation[EvalInvocation]] = []
    well_temperedness_base_key = jax.random.PRNGKey(seed + 2)

    def well_temperedness_inputs(context: EvalInvocation) -> tuple[Array, jax.Array]:
        pass_index = context.now_step // eval_config.every
        return (
            sample_eval_batch(pass_index * eval_config.n_steps),
            jax.random.fold_in(well_temperedness_base_key, pass_index),
        )

    for metric in eval_config.metrics:
        schedule = schedule_for(metric, eval_config)
        match metric:
            case PGDReconLossConfig():
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
            case WellTemperednessConfig():
                operation = make_well_temperedness_operation(
                    metric,
                    schedule,
                    model,
                    ci_capture_keys,
                    mesh,
                    compiler_options,
                    inputs_for_context=well_temperedness_inputs,
                    figure_rendering="synchronous" if wandb_configured else None,
                )
            case CEandKLLossesConfig():
                raise AssertionError(
                    "CEandKLLosses scores next-token cross-entropy and KL over a categorical "
                    "output distribution; a toy target emits neither tokens nor logits"
                )
            case _:
                raise AssertionError(f"eval metric {metric.type!r} has no toy binding")
        operations.append(operation)
    return tuple(operations)
