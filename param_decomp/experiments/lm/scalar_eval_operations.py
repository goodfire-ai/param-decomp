"""Independent CE/KL, causal-L0, and fresh-PGD LM operations."""

from typing import Literal

import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh
from jaxtyping import Array, PRNGKeyArray

from param_decomp.core.configs import CI_L0Config, PGDReconLossConfig
from param_decomp.core.eval_schedule import EvalSchedule
from param_decomp.core.metrics import BarChart, LogRecord
from param_decomp.core.model import CaptureKeys, DecomposedModel
from param_decomp.core.recon import resolve_reconstruction_spec
from param_decomp.core.recon_eval import FreshPGDReconEval
from param_decomp.core.run import EvalOperation
from param_decomp.experiments.lm.eval import (
    MaskingArm,
    ScalarStep,
    make_ce_kl_step,
    make_ci_l0_step,
    make_fresh_pgd_step,
    make_masked_kl_step,
)
from param_decomp.experiments.lm.eval_config import CEandKLLossesConfig
from param_decomp.experiments.lm.eval_context import LMEvalContext
from param_decomp.experiments.lm.eval_keys import EvalKeyStream

type Stream = Literal["nontarget", "target"]
"""Which STREAM an eval operation measures. ONE value, not a (batch source, log prefix)
pair, so target-stream batches cannot be spelled under the nontarget stream's log keys."""


def stream_batches(stream: Stream, context: LMEvalContext) -> tuple[Array, ...]:
    match stream:
        case "nontarget":
            return context.batches
        case "target":
            assert context.target_batches is not None, (
                "target-stream metrics need a tPD run's prompt pool; a plain run has none"
            )
            return context.target_batches


def stream_log_prefix(stream: Stream, context: LMEvalContext) -> str:
    targeted = context.target_batches is not None
    match stream:
        case "nontarget":
            return "eval/nontarget_data/" if targeted else "eval/"
        case "target":
            return "eval/"


def _make_scalar_operation(
    schedule: EvalSchedule,
    step: ScalarStep,
    prefixes: tuple[str, ...],
    model: DecomposedModel,
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    stream: Stream,
) -> EvalOperation[LMEvalContext]:
    def run(context: LMEvalContext) -> LogRecord:
        log_prefix = stream_log_prefix(stream, context)
        sums: dict[str, Array] = {}
        for batch_index, tokens in enumerate(stream_batches(stream, context)):
            key = random.fold_in(
                run_key,
                EvalKeyStream.SCALARS * train_steps + context.pass_index * eval_steps + batch_index,
            )
            values = step(
                model,
                context.state.decomposition.components,
                context.state.decomposition.ci_fn,
                tokens,
                key,
            )
            for name, value in values.items():
                if name.startswith(prefixes):
                    sums[name] = sums.get(name, jnp.zeros(())) + value
        return {f"{log_prefix}{name}": float(value) / eval_steps for name, value in sums.items()}

    return EvalOperation(schedule, run)


def make_ce_kl_operation(
    metric: CEandKLLossesConfig,
    schedule: EvalSchedule,
    stream: Stream,
    model: DecomposedModel,
    ci_capture_keys: CaptureKeys,
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    mesh: Mesh,
    compiler_options: dict[str, bool | int | str],
) -> EvalOperation[LMEvalContext]:
    return _make_scalar_operation(
        schedule,
        make_ce_kl_step(model, ci_capture_keys, metric.rounding_threshold, mesh, compiler_options),
        ("ce_kl/",),
        model,
        run_key,
        train_steps,
        eval_steps,
        stream,
    )


def make_masked_kl_operation(
    arm: MaskingArm,
    schedule: EvalSchedule,
    stream: Stream,
    model: DecomposedModel,
    ci_capture_keys: CaptureKeys,
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    mesh: Mesh,
    compiler_options: dict[str, bool | int | str],
) -> EvalOperation[LMEvalContext]:
    """ONE masking arm, authored as the loss config that names the same construction —
    `CIMaskedReconLoss` / `UnmaskedNoDeltaReconLoss`, as `PGDReconLoss` already is."""
    return _make_scalar_operation(
        schedule,
        make_masked_kl_step(model, ci_capture_keys, arm, mesh, compiler_options),
        (f"ce_kl/kl_{arm}",),
        model,
        run_key,
        train_steps,
        eval_steps,
        stream,
    )


def make_ci_l0_operation(
    metric: CI_L0Config,
    schedule: EvalSchedule,
    stream: Stream,
    model: DecomposedModel,
    ci_capture_keys: CaptureKeys,
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    mesh: Mesh,
    compiler_options: dict[str, bool | int | str],
) -> EvalOperation[LMEvalContext]:
    groups = (
        {name: tuple(patterns) for name, patterns in metric.groups.items()}
        if metric.groups is not None
        else None
    )
    scalars = _make_scalar_operation(
        schedule,
        make_ci_l0_step(
            model, ci_capture_keys, metric.ci_alive_threshold, groups, mesh, compiler_options
        ),
        ("l0/",),
        model,
        run_key,
        train_steps,
        eval_steps,
        stream,
    )

    def run(context: LMEvalContext) -> LogRecord:
        record = dict(scalars.run(context))
        log_prefix = stream_log_prefix(stream, context)
        prefix = f"{log_prefix}l0/{metric.ci_alive_threshold}_"
        record[f"{log_prefix}l0/bar_chart"] = BarChart(
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
    stream: Stream,
    model: DecomposedModel,
    ci_capture_keys: CaptureKeys,
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    mesh: Mesh,
    compiler_options: dict[str, bool | int | str],
) -> EvalOperation[LMEvalContext]:
    assert metric.init == "random" and metric.source_shape == "c", metric
    probe = FreshPGDReconEval(
        name=metric.name or metric.type,
        n_steps=metric.n_steps,
        step_size=metric.step_size,
        reconstruction=resolve_reconstruction_spec(metric.hidden_acts_reconstruction),
    )
    return _make_scalar_operation(
        schedule,
        make_fresh_pgd_step(model, ci_capture_keys, probe, mesh, compiler_options),
        (f"loss/{probe.name}",),
        model,
        run_key,
        train_steps,
        eval_steps,
        stream,
    )
