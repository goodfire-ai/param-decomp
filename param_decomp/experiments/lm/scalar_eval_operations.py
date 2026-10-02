"""Independent CE/KL, causal-L0, and fresh-PGD LM operations over the shared batch context."""

import jax
import numpy as np
from jax import random
from jax.sharding import Mesh
from jaxtyping import Array, PRNGKeyArray
from numpy.typing import NDArray

from param_decomp.core.configs import (
    AnyPGDEvalConfig,
    CI_L0Config,
    EvalPGDReconLossConfig,
    SlowPGDReconLossConfig,
)
from param_decomp.core.eval_schedule import EvalSchedule
from param_decomp.core.metrics import BarChart, LogRecord
from param_decomp.core.model import (
    EMPTY_CAPTURE_KEYS,
    CaptureKeys,
    ComponentActivations,
    PlacedModel,
)
from param_decomp.core.recon import resolve_auxiliary_reconstruction
from param_decomp.core.recon_eval import FreshPGDReconEval
from param_decomp.core.run import (
    SharedForwardOperation,
    SharedForwardOperationPlan,
    shared_forward_operation,
)
from param_decomp.experiments.lm.eval import (
    PreparedLMBatch,
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
from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments
from param_decomp.targets.lm_output import LMOutput

type AnyScalarMetricConfig = CEandKLLossesConfig | CI_L0Config | AnyPGDEvalConfig


def fresh_pgd_probe(metric: AnyPGDEvalConfig) -> FreshPGDReconEval:
    return FreshPGDReconEval(
        name=metric.name or metric.type,
        n_steps=metric.n_steps,
        step_size=metric.step_size,
        reconstruction=resolve_auxiliary_reconstruction(metric.auxiliaries),
    )


def _ci_l0_groups(metric: CI_L0Config) -> dict[str, tuple[str, ...]] | None:
    return (
        {name: tuple(patterns) for name, patterns in metric.groups.items()}
        if metric.groups is not None
        else None
    )


def scalar_step_for[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    metric: AnyScalarMetricConfig,
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    ci_capture_keys: CaptureKeys,
    mesh: Mesh,
) -> ScalarStep[TargetIn, PreparedT, Conditioning, PreparedMaskingT]:
    """The scalar tier's STANDALONE step (own clean forward) — what the AOT eval fit check
    compiles; the operations below score the pass's shared context via `scalar_scorer_for`."""
    match metric:
        case CEandKLLossesConfig():
            return make_ce_kl_step(model, ci_capture_keys, metric.rounding_threshold, mesh)
        case CI_L0Config():
            return make_ci_l0_step(
                model,
                ci_capture_keys,
                metric.ci_alive_threshold,
                _ci_l0_groups(metric),
                mesh,
            )
        case EvalPGDReconLossConfig() | SlowPGDReconLossConfig():
            return make_fresh_pgd_step(model, ci_capture_keys, fresh_pgd_probe(metric), mesh)


def scalar_scorer_for[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    metric: AnyScalarMetricConfig,
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    mesh: Mesh,
) -> ScalarScorer[TargetIn, PreparedT, Conditioning, PreparedMaskingT]:
    """THE config→scorer binding for the scalar tier: the pure scorer each operation jits
    over the pass's shared batch context, and what the trace gate lowers."""
    match metric:
        case CEandKLLossesConfig():
            return make_ce_kl_scorer(model, metric.rounding_threshold, mesh)
        case CI_L0Config():
            return make_ci_l0_scorer(model, metric.ci_alive_threshold, _ci_l0_groups(metric))
        case EvalPGDReconLossConfig() | SlowPGDReconLossConfig():
            return make_fresh_pgd_scorer(model, fresh_pgd_probe(metric), mesh)


def _make_scalar_operation[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    schedule: EvalSchedule,
    scorer: ScalarScorer[TargetIn, PreparedT, Conditioning, PreparedMaskingT],
    prefixes: tuple[str, ...],
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    compiler_options: dict[str, bool | int | str],
    reconstruction_capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
) -> SharedForwardOperationPlan[
    LMEvalPass[TargetIn, Conditioning], LMBatchContext[TargetIn, PreparedT, Conditioning]
]:
    def prepare(
        _example_pass: LMEvalPass[TargetIn, Conditioning],
        example_context: LMBatchContext[TargetIn, PreparedT, Conditioning],
    ) -> SharedForwardOperation[
        LMEvalPass[TargetIn, Conditioning], LMBatchContext[TargetIn, PreparedT, Conditioning]
    ]:
        def score(
            model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
            batch: PreparedLMBatch[TargetIn, PreparedT, Conditioning],
            pass_index: NDArray[np.uint32],
            batch_index: NDArray[np.uint32],
        ) -> dict[str, Array]:
            key = random.fold_in(
                run_key, EvalKeyStream.SCALARS * train_steps + pass_index * eval_steps + batch_index
            )
            return scorer(model, batch, key)

        score_step = (
            jax.jit(score, compiler_options=compiler_options)
            .lower(
                model,
                prepared_batch_from_context(example_context, reconstruction_capture_keys),
                np.zeros((), np.uint32),
                np.zeros((), np.uint32),
            )
            .compile()
        )

        def init() -> dict[str, np.ndarray]:
            return {}

        def update(
            sums: dict[str, np.ndarray],
            context: LMBatchContext[TargetIn, PreparedT, Conditioning],
        ) -> dict[str, np.ndarray]:
            values = score_step(
                model,
                prepared_batch_from_context(context, reconstruction_capture_keys),
                np.asarray(context.pass_index, np.uint32),
                np.asarray(context.batch_index, np.uint32),
            )
            folded = dict(sums)
            for name, value in values.items():
                if name.startswith(prefixes):
                    folded[name] = folded.get(name, np.zeros((), np.float32)) + np.asarray(value)
            return folded

        def finish(
            eval_pass: LMEvalPass[TargetIn, Conditioning], sums: dict[str, np.ndarray]
        ) -> LogRecord:
            del eval_pass
            return {f"eval/{name}": float(value) / eval_steps for name, value in sums.items()}

        return shared_forward_operation(schedule, init, update, finish)

    return SharedForwardOperationPlan(prepare)


def make_ce_kl_operation[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    metric: CEandKLLossesConfig,
    schedule: EvalSchedule,
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    mesh: Mesh,
    compiler_options: dict[str, bool | int | str],
) -> SharedForwardOperationPlan[
    LMEvalPass[TargetIn, Conditioning], LMBatchContext[TargetIn, PreparedT, Conditioning]
]:
    return _make_scalar_operation(
        schedule,
        scalar_scorer_for(metric, model, mesh),
        ("ce_kl/",),
        model,
        run_key,
        train_steps,
        eval_steps,
        compiler_options,
    )


def make_ci_l0_operation[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    metric: CI_L0Config,
    schedule: EvalSchedule,
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    mesh: Mesh,
    compiler_options: dict[str, bool | int | str],
) -> SharedForwardOperationPlan[
    LMEvalPass[TargetIn, Conditioning], LMBatchContext[TargetIn, PreparedT, Conditioning]
]:
    def prepare(
        example_pass: LMEvalPass[TargetIn, Conditioning],
        example_context: LMBatchContext[TargetIn, PreparedT, Conditioning],
    ) -> SharedForwardOperation[
        LMEvalPass[TargetIn, Conditioning], LMBatchContext[TargetIn, PreparedT, Conditioning]
    ]:
        scalars = _make_scalar_operation(
            schedule,
            scalar_scorer_for(metric, model, mesh),
            ("l0/",),
            model,
            run_key,
            train_steps,
            eval_steps,
            compiler_options,
        )

        prepared_scalars = scalars.prepare(example_pass, example_context)

        def finish(
            eval_pass: LMEvalPass[TargetIn, Conditioning], sums: dict[str, np.ndarray]
        ) -> LogRecord:
            record = dict(prepared_scalars.finish(eval_pass, sums))
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

        return SharedForwardOperation(
            schedule, prepared_scalars.init, prepared_scalars.update, finish
        )

    return SharedForwardOperationPlan(prepare)


def make_fresh_pgd_operation[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    metric: AnyPGDEvalConfig,
    schedule: EvalSchedule,
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    mesh: Mesh,
    compiler_options: dict[str, bool | int | str],
) -> SharedForwardOperationPlan[
    LMEvalPass[TargetIn, Conditioning], LMBatchContext[TargetIn, PreparedT, Conditioning]
]:
    probe = fresh_pgd_probe(metric)
    return _make_scalar_operation(
        schedule,
        scalar_scorer_for(metric, model, mesh),
        (f"loss/{probe.name}",),
        model,
        run_key,
        train_steps,
        eval_steps,
        compiler_options,
        reconstruction_capture_keys=probe.reconstruction_capture_keys,
    )
