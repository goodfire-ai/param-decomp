"""Independent CE/KL, causal-L0, and fresh-PGD LM operations."""

import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh
from jaxtyping import Array, PRNGKeyArray

from param_decomp.core.configs import CI_L0Config, PGDReconLossConfig
from param_decomp.core.eval_schedule import EvalSchedule
from param_decomp.core.metrics import BarChart, LogRecord
from param_decomp.core.model import CaptureKeys, PlacedModel
from param_decomp.core.recon import resolve_reconstruction_spec
from param_decomp.core.recon_eval import FreshPGDReconEval
from param_decomp.core.run import EvalOperation
from param_decomp.experiments.lm.eval import (
    ScalarStep,
    make_ce_kl_step,
    make_ci_l0_step,
    make_fresh_pgd_step,
)
from param_decomp.experiments.lm.eval_config import CEandKLLossesConfig
from param_decomp.experiments.lm.eval_context import LMEvalContext
from param_decomp.experiments.lm.eval_keys import EvalKeyStream

type AnyScalarMetricConfig = CEandKLLossesConfig | CI_L0Config | PGDReconLossConfig


def fresh_pgd_probe(metric: PGDReconLossConfig) -> FreshPGDReconEval:
    assert metric.init == "random" and metric.source_shape == "c", metric
    return FreshPGDReconEval(
        name=metric.name or metric.type,
        n_steps=metric.n_steps,
        step_size=metric.step_size,
        reconstruction=resolve_reconstruction_spec(metric.hidden_acts_reconstruction),
    )


def scalar_step_for(
    metric: AnyScalarMetricConfig,
    model: PlacedModel,
    ci_capture_keys: CaptureKeys,
    mesh: Mesh,
    compiler_options: dict[str, bool | int | str] | None,
) -> ScalarStep:
    """THE config→kernel binding for the scalar tier — the operations below and the AOT
    eval fit check compile the identical step from one spelling."""
    match metric:
        case CEandKLLossesConfig():
            return make_ce_kl_step(
                model, ci_capture_keys, metric.rounding_threshold, mesh, compiler_options
            )
        case CI_L0Config():
            groups = (
                {name: tuple(patterns) for name, patterns in metric.groups.items()}
                if metric.groups is not None
                else None
            )
            return make_ci_l0_step(
                model, ci_capture_keys, metric.ci_alive_threshold, groups, mesh, compiler_options
            )
        case PGDReconLossConfig():
            return make_fresh_pgd_step(
                model, ci_capture_keys, fresh_pgd_probe(metric), mesh, compiler_options
            )


def _make_scalar_operation(
    schedule: EvalSchedule,
    step: ScalarStep,
    prefixes: tuple[str, ...],
    model: PlacedModel,
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
) -> EvalOperation[LMEvalContext]:
    def run(context: LMEvalContext) -> LogRecord:
        sums: dict[str, Array] = {}
        for batch_index, tokens in enumerate(context.batches):
            key = random.fold_in(
                run_key,
                EvalKeyStream.SCALARS * train_steps + context.pass_index * eval_steps + batch_index,
            )
            values = step(
                model,
                context.state.decomposition.components,
                context.placed_ci_fn,
                tokens,
                key,
            )
            for name, value in values.items():
                if name.startswith(prefixes):
                    sums[name] = sums.get(name, jnp.zeros(())) + value
        return {f"eval/{name}": float(value) / eval_steps for name, value in sums.items()}

    return EvalOperation(schedule, run)


def make_ce_kl_operation(
    metric: CEandKLLossesConfig,
    schedule: EvalSchedule,
    model: PlacedModel,
    ci_capture_keys: CaptureKeys,
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    mesh: Mesh,
    compiler_options: dict[str, bool | int | str],
) -> EvalOperation[LMEvalContext]:
    return _make_scalar_operation(
        schedule,
        scalar_step_for(metric, model, ci_capture_keys, mesh, compiler_options),
        ("ce_kl/",),
        model,
        run_key,
        train_steps,
        eval_steps,
    )


def make_ci_l0_operation(
    metric: CI_L0Config,
    schedule: EvalSchedule,
    model: PlacedModel,
    ci_capture_keys: CaptureKeys,
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    mesh: Mesh,
    compiler_options: dict[str, bool | int | str],
) -> EvalOperation[LMEvalContext]:
    scalars = _make_scalar_operation(
        schedule,
        scalar_step_for(metric, model, ci_capture_keys, mesh, compiler_options),
        ("l0/",),
        model,
        run_key,
        train_steps,
        eval_steps,
    )

    def run(context: LMEvalContext) -> LogRecord:
        record = dict(scalars.run(context))
        prefix = f"eval/l0/{metric.ci_alive_threshold}_"
        record["eval/l0/bar_chart"] = BarChart(
            rows=tuple(
                (name.removeprefix(prefix), value)
                for name, value in record.items()
                if name.startswith(prefix) and isinstance(value, float)
            ),
            x_label="layer",
            y_label="l0",
            title=f"L0_{metric.ci_alive_threshold}",
        )
        return record

    return EvalOperation(schedule, run)


def make_fresh_pgd_operation(
    metric: PGDReconLossConfig,
    schedule: EvalSchedule,
    model: PlacedModel,
    ci_capture_keys: CaptureKeys,
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    mesh: Mesh,
    compiler_options: dict[str, bool | int | str],
) -> EvalOperation[LMEvalContext]:
    return _make_scalar_operation(
        schedule,
        scalar_step_for(metric, model, ci_capture_keys, mesh, compiler_options),
        (f"loss/{fresh_pgd_probe(metric).name}",),
        model,
        run_key,
        train_steps,
        eval_steps,
    )
