"""LM evaluation operation binding and execution.

Binding is two closed passes over the authored metrics: first each metric declares its
clean-capture demand on the shared batch context (`_clean_capture_demand`), then each
binds its operation. The pass's one context step captures the union of those demands, so
every batched operation reads one clean forward + CI envelope per batch.
"""

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import PRNGKeyArray

from param_decomp.core.components import nonlinearity_partitions
from param_decomp.core.configs import (
    CI_L0Config,
    CIHistogramsConfig,
    CIMeanPerComponentConfig,
    ComponentActivationDensityConfig,
    IdentityCIErrorConfig,
    PermutedCIPlotsConfig,
    PGDReconLossConfig,
    UVPlotsConfig,
    WellTemperednessConfig,
)
from param_decomp.core.model import BATCH_AXES, EMPTY_CAPTURE_KEYS, CaptureKeys, PlacedModel
from param_decomp.core.run import (
    BackgroundRenderer,
    EvalInvocation,
    EvalOperation,
    Evaluation,
    MetricsSink,
)
from param_decomp.core.sharding import local_data_parallel_size
from param_decomp.core.well_temperedness_eval import make_well_temperedness_operation
from param_decomp.experiments.eval_config import (
    AnyEvalMetricConfig,
    EvalConfig,
    schedule_for,
    slow_schedule,
)
from param_decomp.experiments.lm.arithmetic_eval_operation import make_arithmetic_operation
from param_decomp.experiments.lm.attn_patterns_eval import attn_output_key_by_site
from param_decomp.experiments.lm.diagnostic_eval_operations import (
    make_attention_operation,
    make_nonlinearity_operation,
    make_permutation_operation,
    make_site_figures_operation,
)
from param_decomp.experiments.lm.eval_config import (
    ArithmeticCIGridConfig,
    CEandKLLossesConfig,
    CIMaskedAttnPatternsReconLossConfig,
    StochasticAttnPatternsReconLossConfig,
)
from param_decomp.experiments.lm.eval_context import (
    LMBatchContext,
    LMEvalPass,
    make_lm_batch_context_step,
    make_lm_batch_contexts,
)
from param_decomp.experiments.lm.eval_keys import EvalKeyStream
from param_decomp.experiments.lm.resolved import LMAnyRun
from param_decomp.experiments.lm.scalar_eval_operations import (
    fresh_pgd_probe,
    make_ce_kl_operation,
    make_ci_l0_operation,
    make_fresh_pgd_operation,
)
from param_decomp.infra.dataset_store import read_dataset_meta
from param_decomp.pretrain.batch_data import BatchSchedule, ShardServer, scan_shards


def global_token_batch(local: np.ndarray, mesh: Mesh, global_batch: int) -> jax.Array:
    sharding = NamedSharding(mesh, P(BATCH_AXES))
    return jax.make_array_from_process_local_data(sharding, local, (global_batch, local.shape[1]))


def make_lm_evaluation(
    built: LMAnyRun,
    eval: EvalConfig,
    model: PlacedModel,
    run_key: PRNGKeyArray,
    mesh: Mesh,
    n_proc: int,
    sink: MetricsSink,
    compiler_options: dict[str, bool | int | str],
) -> Evaluation[LMEvalPass, LMBatchContext]:
    """Construct one executable operation for every authored LM metric."""
    pd = built.pd
    capture_inputs = built.ci_fn.capture_keys
    data = built.data
    schedule = BatchSchedule(scan_shards(data.eval_dir), eval.batch_size, pd.seed + 1)
    seq_len = read_dataset_meta(data.eval_dir).seq_len
    server = ShardServer(schedule, seq_len, jax.process_index(), n_proc)
    assert server.per_process % local_data_parallel_size(mesh) == 0
    renderer = BackgroundRenderer(sink)

    def batches(pass_index: int) -> list[jax.Array]:
        return [
            global_token_batch(
                server.local_batch(pass_index * eval.n_steps + j), mesh, eval.batch_size
            )
            for j in range(eval.n_steps)
        ]

    def well_temperedness_inputs(
        eval_pass: LMEvalPass,
    ) -> tuple[jax.Array, PRNGKeyArray]:
        return eval_pass.batches[0], jax.random.fold_in(
            run_key, EvalKeyStream.WELL_TEMPEREDNESS * pd.steps + eval_pass.pass_index
        )

    def _clean_capture_demand(metric: AnyEvalMetricConfig) -> CaptureKeys:
        """What this metric reads off the shared clean forward beyond the CI taps."""
        match metric:
            case CIMaskedAttnPatternsReconLossConfig() | StochasticAttnPatternsReconLossConfig():
                return frozenset(attn_output_key_by_site(model).values())
            case PGDReconLossConfig():
                return fresh_pgd_probe(metric).hidden_acts_capture_keys
            case _:
                return EMPTY_CAPTURE_KEYS

    def make_operation(metric: AnyEvalMetricConfig) -> EvalOperation[LMEvalPass, LMBatchContext]:
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
            case CI_L0Config():
                return make_ci_l0_operation(
                    metric,
                    schedule,
                    model,
                    run_key,
                    pd.steps,
                    eval.n_steps,
                    compiler_options,
                )
            case PGDReconLossConfig():
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
            case (
                CIHistogramsConfig()
                | ComponentActivationDensityConfig()
                | CIMeanPerComponentConfig()
            ):
                return make_site_figures_operation(metric, schedule, compiler_options, renderer)
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
                    inputs_for_context=well_temperedness_inputs,
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

    partitions = nonlinearity_partitions(model.sites)
    standing_operations = (
        (make_nonlinearity_operation(slow_schedule(eval), partitions, compiler_options),)
        if partitions
        else ()
    )
    operations = tuple(make_operation(metric) for metric in eval.metrics) + standing_operations

    operation_capture_keys = frozenset().union(
        *(_clean_capture_demand(metric) for metric in eval.metrics), EMPTY_CAPTURE_KEYS
    )
    context_step = make_lm_batch_context_step(
        model, capture_inputs, operation_capture_keys, mesh, compiler_options
    )

    def make_pass(invocation: EvalInvocation) -> LMEvalPass:
        pass_index = invocation.now_step // eval.every
        return LMEvalPass(
            state=invocation.state,
            now_step=invocation.now_step,
            placed_ci_fn=invocation.placed_ci_fn,
            pass_index=pass_index,
            batches=tuple(batches(pass_index)),
        )

    return Evaluation(operations, make_pass, make_lm_batch_contexts(context_step, model))
