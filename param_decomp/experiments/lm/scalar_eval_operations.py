"""Independent CE/KL, causal-L0, and fresh-PGD LM operations over the shared batch context."""

import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh
from jaxtyping import Array, PRNGKeyArray

from param_decomp.core.configs import CI_L0Config, PGDReconLossConfig
from param_decomp.core.eval_schedule import EvalSchedule
from param_decomp.core.jit_util import filter_jit
from param_decomp.core.metrics import BarChart, LogRecord
from param_decomp.core.model import EMPTY_CAPTURE_KEYS, CaptureKeys, PlacedModel
from param_decomp.core.recon import resolve_reconstruction_spec
from param_decomp.core.recon_eval import FreshPGDReconEval
from param_decomp.core.run import BatchedOperation, batched_operation
from param_decomp.experiments.lm.eval import (
    ScalarScorer,
    ScalarStep,
    make_ce_kl_scorer,
    make_ce_kl_step,
    make_ci_l0_scorer,
    make_ci_l0_step,
    make_fresh_pgd_scorer,
    make_fresh_pgd_step,
)
from param_decomp.experiments.lm.eval_config import CEandKLLossesConfig
from param_decomp.experiments.lm.eval_context import (
    LMBatchContext,
    LMEvalPass,
    prepared_batch_from_context,
)
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
    scorer: ScalarScorer,
    prefixes: tuple[str, ...],
    model: PlacedModel,
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    compiler_options: dict[str, bool | int | str],
    hidden_acts_capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
) -> BatchedOperation[LMEvalPass, LMBatchContext]:
    score_step = filter_jit(scorer, compiler_options=compiler_options)

    def init() -> dict[str, Array]:
        return {}

    def update(sums: dict[str, Array], context: LMBatchContext) -> dict[str, Array]:
        key = random.fold_in(
            run_key,
            EvalKeyStream.SCALARS * train_steps
            + context.pass_index * eval_steps
            + context.batch_index,
        )
        values = score_step(
            model, prepared_batch_from_context(context, hidden_acts_capture_keys), key
        )
        folded = dict(sums)
        for name, value in values.items():
            if name.startswith(prefixes):
                folded[name] = folded.get(name, jnp.zeros(())) + value
        return folded

    def finish(eval_pass: LMEvalPass, sums: dict[str, Array]) -> LogRecord:
        del eval_pass
        return {f"eval/{name}": float(value) / eval_steps for name, value in sums.items()}

    return batched_operation(schedule, init, update, finish)


def make_ce_kl_operation(
    metric: CEandKLLossesConfig,
    schedule: EvalSchedule,
    model: PlacedModel,
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    mesh: Mesh,
    compiler_options: dict[str, bool | int | str],
) -> BatchedOperation[LMEvalPass, LMBatchContext]:
    return _make_scalar_operation(
        schedule,
        make_ce_kl_scorer(model, metric.rounding_threshold, mesh),
        ("ce_kl/",),
        model,
        run_key,
        train_steps,
        eval_steps,
        compiler_options,
    )


def make_ci_l0_operation(
    metric: CI_L0Config,
    schedule: EvalSchedule,
    model: PlacedModel,
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    compiler_options: dict[str, bool | int | str],
) -> BatchedOperation[LMEvalPass, LMBatchContext]:
    groups = (
        {name: tuple(patterns) for name, patterns in metric.groups.items()}
        if metric.groups is not None
        else None
    )
    scalars = _make_scalar_operation(
        schedule,
        make_ci_l0_scorer(model, metric.ci_alive_threshold, groups),
        ("l0/",),
        model,
        run_key,
        train_steps,
        eval_steps,
        compiler_options,
    )

    def finish(eval_pass: LMEvalPass, sums: dict[str, Array]) -> LogRecord:
        record = dict(scalars.finish(eval_pass, sums))
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

    return BatchedOperation(schedule, scalars.init, scalars.update, finish)


def make_fresh_pgd_operation(
    metric: PGDReconLossConfig,
    schedule: EvalSchedule,
    model: PlacedModel,
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    mesh: Mesh,
    compiler_options: dict[str, bool | int | str],
) -> BatchedOperation[LMEvalPass, LMBatchContext]:
    probe = fresh_pgd_probe(metric)
    return _make_scalar_operation(
        schedule,
        make_fresh_pgd_scorer(model, probe, mesh),
        (f"loss/{probe.name}",),
        model,
        run_key,
        train_steps,
        eval_steps,
        compiler_options,
        hidden_acts_capture_keys=probe.hidden_acts_capture_keys,
    )
