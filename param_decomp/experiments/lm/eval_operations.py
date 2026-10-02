"""LM evaluation operation binding and execution.

Binding is two closed passes over the authored metrics: first each metric declares its
clean-capture demand on the shared batch context (`clean_capture_demand`), then each
binds its operation. The pass's one context step captures the union of those demands, so
every shared-forward operation reads one clean forward + CI envelope per batch.
"""

from collections.abc import Callable, Iterable

import jax
from jax.sharding import Mesh
from jaxtyping import PRNGKeyArray

from param_decomp.core.components import nonlinearity_partitions
from param_decomp.core.configs import (
    CI_L0Config,
    CIHistogramsConfig,
    CIMeanPerComponentConfig,
    ComponentActivationDensityConfig,
    EvalPGDReconLossConfig,
    IdentityCIErrorConfig,
    PermutedCIPlotsConfig,
    SlowPGDReconLossConfig,
    UVPlotsConfig,
)
from param_decomp.core.model import (
    EMPTY_CAPTURE_KEYS,
    CaptureKeys,
    ComponentActivations,
    PlacedModel,
)
from param_decomp.core.run import (
    BackgroundRenderer,
    EvalInvocation,
    EvalOperation,
    EvalOperationPlan,
    Evaluation,
    EvaluationPlan,
    MetricsSink,
    SharedForwardOperationPlan,
    StandaloneOperationPlan,
    no_batch_contexts,
)
from param_decomp.core.slow_eval import component_group_counts
from param_decomp.experiments.eval_config import (
    AnyEvalMetricConfig,
    EvalConfig,
    schedule_for,
    slow_schedule,
)
from param_decomp.experiments.lm.arithmetic_eval_operation import make_arithmetic_operation
from param_decomp.experiments.lm.attn_patterns_eval import attn_output_key_by_site
from param_decomp.experiments.lm.ci_position_eval import make_ci_position_counts_operation
from param_decomp.experiments.lm.diagnostic_eval_operations import (
    make_attention_operation,
    make_nonlinearity_operation,
    make_permutation_operation,
    make_router_divergence_operation,
    make_site_figures_operation,
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
from param_decomp.experiments.lm.eval_context import (
    LMBatchContext,
    LMEvalPass,
    make_lm_batch_context_step,
    make_lm_batch_contexts,
)
from param_decomp.experiments.lm.resolved import LMAnyRun
from param_decomp.experiments.lm.router_divergence_eval import router_probs_capture_keys
from param_decomp.experiments.lm.scalar_eval_operations import (
    fresh_pgd_probe,
    make_ce_kl_operation,
    make_ci_l0_operation,
    make_fresh_pgd_operation,
)
from param_decomp.experiments.lm.well_temperedness_eval import make_well_temperedness_operation
from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments
from param_decomp.targets.lm_output import LMOutput


def clean_capture_demand[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    metric: AnyEvalMetricConfig,
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
) -> CaptureKeys:
    """What this metric reads off the shared clean forward beyond the CI taps."""
    match metric:
        case CIMaskedAttnPatternsReconLossConfig() | StochasticAttnPatternsReconLossConfig():
            return frozenset(attn_output_key_by_site(model).values())
        case EvalPGDReconLossConfig() | SlowPGDReconLossConfig():
            return fresh_pgd_probe(metric).reconstruction_capture_keys
        case RouterDivergenceConfig():
            return frozenset(router_probs_capture_keys(model))
        case (
            CEandKLLossesConfig()
            | CI_L0Config()
            | CIActiveCountsPerPositionConfig()
            | CIHistogramsConfig()
            | ComponentActivationDensityConfig()
            | CIMeanPerComponentConfig()
            | PermutedCIPlotsConfig()
            | UVPlotsConfig()
            | IdentityCIErrorConfig()
            | WellTemperednessConfig()
            | ArithmeticCIGridConfig()
        ):
            return EMPTY_CAPTURE_KEYS


def make_lm_evaluation[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    built: LMAnyRun,
    eval: EvalConfig,
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    run_key: PRNGKeyArray,
    mesh: Mesh,
    n_proc: int,
    sink: MetricsSink,
    compiler_options: dict[str, bool | int | str],
    *,
    sample_batch: Callable[[int], TargetIn],
) -> EvaluationPlan[
    Conditioning,
    LMEvalPass[TargetIn, Conditioning],
    LMBatchContext[TargetIn, PreparedT, Conditioning],
]:
    """Describe the kernels each authored metric needs before any evaluation runs."""
    pd = built.pd
    capture_inputs = built.ci_fn.capture_keys
    renderer = BackgroundRenderer(sink)

    def batches(pass_index: int) -> list[TargetIn]:
        return [sample_batch(pass_index * eval.n_steps + j) for j in range(eval.n_steps)]

    def make_operation(
        metric: AnyEvalMetricConfig,
    ) -> EvalOperationPlan[
        LMEvalPass[TargetIn, Conditioning], LMBatchContext[TargetIn, PreparedT, Conditioning]
    ]:
        schedule = schedule_for(metric, eval)
        match metric:
            case CEandKLLossesConfig():
                return make_ce_kl_operation(
                    metric,
                    schedule,
                    model,
                    run_key,
                    pd.steps,
                    eval.n_steps,
                    mesh,
                    compiler_options,
                )
            case CIActiveCountsPerPositionConfig():
                return make_ci_position_counts_operation(schedule, compiler_options)
            case CI_L0Config():
                return make_ci_l0_operation(
                    metric,
                    schedule,
                    model,
                    run_key,
                    pd.steps,
                    eval.n_steps,
                    mesh,
                    compiler_options,
                )
            case EvalPGDReconLossConfig() | SlowPGDReconLossConfig():
                return make_fresh_pgd_operation(
                    metric,
                    schedule,
                    model,
                    run_key,
                    pd.steps,
                    eval.n_steps,
                    mesh,
                    compiler_options,
                )

            case CIMaskedAttnPatternsReconLossConfig() | StochasticAttnPatternsReconLossConfig():
                return make_attention_operation(
                    metric, schedule, model, run_key, pd.steps, compiler_options
                )
            case RouterDivergenceConfig():
                return make_router_divergence_operation(
                    metric, schedule, model, run_key, pd.steps, mesh, compiler_options
                )
            case (
                CIHistogramsConfig()
                | ComponentActivationDensityConfig()
                | CIMeanPerComponentConfig()
            ):
                return make_site_figures_operation(
                    metric,
                    schedule,
                    component_group_counts(model.model.sites),
                    compiler_options,
                    renderer,
                )
            case PermutedCIPlotsConfig() | UVPlotsConfig() | IdentityCIErrorConfig():
                return make_permutation_operation(
                    metric, schedule, model, compiler_options, renderer
                )

            case WellTemperednessConfig():
                return make_well_temperedness_operation(
                    metric,
                    schedule,
                    model,
                    capture_inputs,
                    mesh,
                    compiler_options,
                    run_key=run_key,
                    train_steps=pd.steps,
                    figure_rendering=renderer if sink.accepts_deferred_media else None,
                )

            case ArithmeticCIGridConfig():
                return make_arithmetic_operation(
                    metric,
                    schedule,
                    built.target,
                    model,
                    capture_inputs,
                    mesh,
                    n_proc,
                    sink,
                    run_key,
                    pd.steps,
                    compiler_options,
                )

    standing_operations = (
        (
            make_nonlinearity_operation(
                slow_schedule(eval), model.model.sites, compiler_options, mesh
            ),
        )
        if nonlinearity_partitions(model.model.sites)
        else ()
    )
    operations = tuple(make_operation(metric) for metric in eval.metrics) + standing_operations

    operation_capture_keys = frozenset().union(
        *(clean_capture_demand(metric, model) for metric in eval.metrics), EMPTY_CAPTURE_KEYS
    )
    context_step = make_lm_batch_context_step(model, capture_inputs, operation_capture_keys, mesh)

    def make_pass(invocation: EvalInvocation[Conditioning]) -> LMEvalPass[TargetIn, Conditioning]:
        pass_index = invocation.now_step // eval.every
        return LMEvalPass(
            decomposition=invocation.decomposition,
            persistent_sources=invocation.persistent_sources,
            now_step=invocation.now_step,
            pass_index=pass_index,
            batches=tuple(batches(pass_index)),
        )

    def prepare(
        invocation: EvalInvocation[Conditioning],
    ) -> Evaluation[
        Conditioning,
        LMEvalPass[TargetIn, Conditioning],
        LMBatchContext[TargetIn, PreparedT, Conditioning],
    ]:
        example_pass = make_pass(invocation)
        example_context = None
        batch_contexts: Callable[
            [LMEvalPass[TargetIn, Conditioning]],
            Iterable[LMBatchContext[TargetIn, PreparedT, Conditioning]],
        ] = no_batch_contexts
        if any(isinstance(operation, SharedForwardOperationPlan) for operation in operations):
            lowered_context = jax.jit(context_step, compiler_options=compiler_options).lower(
                model,
                invocation.decomposition.components,
                invocation.decomposition.ci_fn,
                example_pass.batches[0],
            )
            compiled_context = lowered_context.compile()
            example_context = LMBatchContext(
                pass_index=example_pass.pass_index,
                batch_index=0,
                forward=compiled_context.out_info,
                persistent_sources=invocation.persistent_sources,
            )
            batch_contexts = make_lm_batch_contexts(compiled_context, model)
        prepared_operations: list[
            EvalOperation[
                LMEvalPass[TargetIn, Conditioning],
                LMBatchContext[TargetIn, PreparedT, Conditioning],
            ]
        ] = []
        for operation in operations:
            match operation:
                case StandaloneOperationPlan():
                    prepared_operations.append(operation.prepare(example_pass))
                case SharedForwardOperationPlan():
                    assert example_context is not None
                    prepared_operations.append(operation.prepare(example_pass, example_context))
        return Evaluation(tuple(prepared_operations), make_pass, batch_contexts)

    return EvaluationPlan(prepare)
