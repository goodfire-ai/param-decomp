"""Attention and causal-importance diagnostic operations.

Every operation here is a `SharedForwardOperation` folding over the pass's shared batch
contexts: the CI-reduction family is a cheap on-device reduction of the context's CI
envelope, and the masked-forward metrics (attention patterns) run only
their masked side — the clean side comes from the context.
"""

from functools import partial

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array, PRNGKeyArray
from numpy.typing import NDArray

from param_decomp.core.adversary import SourceStacks
from param_decomp.core.components import SiteSpec, nonlinearity_partitions
from param_decomp.core.configs import (
    CIHistogramsConfig,
    CIMeanPerComponentConfig,
    ComponentActivationDensityConfig,
    IdentityCIErrorConfig,
    PermutedCIPlotsConfig,
    UVPlotsConfig,
)
from param_decomp.core.dict_utils import dict_safe_update_
from param_decomp.core.eval_schedule import EvalSchedule
from param_decomp.core.metrics import LogRecord
from param_decomp.core.model import ComponentActivations, PlacedModel
from param_decomp.core.nonlinearity_eval import (
    make_nonlinearity_eval_step,
    nonlinearity_log_entries,
    site_nonlinearity_stats,
)
from param_decomp.core.run import (
    BackgroundRenderer,
    DeferredMediaRecord,
    SharedForwardOperation,
    SharedForwardOperationPlan,
    shared_forward_operation,
)
from param_decomp.core.slow_eval import (
    IDENTITY_CI_ERROR_TOLERANCE,
    VALUE_HISTOGRAM_N_BINS,
    CIReductionStep,
    PermutationMetricSpec,
    PositionCI,
    PositionCIAccumulation,
    SiteReduction,
    SiteReductionAccumulation,
    compute_identity_ci_errors,
    empty_position_ci_accumulation,
    empty_site_reduction_accumulation,
    fold_position_ci,
    fold_site_reduction,
    make_ci_reduction_step,
    make_position_ci_step,
    position_ci,
    render_permutation_figures,
    render_slow_eval_figures,
    resolve_permutation_metrics,
    site_reductions,
)
from param_decomp.experiments.lm.attn_patterns_eval import (
    AttnPatternsStep,
    LayerKLReduction,
    attn_output_key_by_site,
    attn_patterns_log_entries,
    fold_layer_kl,
    make_ci_attn_patterns_step,
    make_stochastic_attn_patterns_step,
)
from param_decomp.experiments.lm.eval_config import (
    CIMaskedAttnPatternsReconLossConfig,
    RouterDivergenceConfig,
    StochasticAttnPatternsReconLossConfig,
)
from param_decomp.experiments.lm.eval_context import (
    LMBatchContext,
    LMBatchForwardProducts,
    LMEvalPass,
)
from param_decomp.experiments.lm.eval_keys import EvalKeyStream
from param_decomp.experiments.lm.router_divergence_eval import (
    RouterDivergenceAccumulation,
    RouterDivergenceSums,
    empty_router_divergence_sums,
    expert_router_model,
    fold_router_divergence,
    make_router_divergence_step,
    router_divergence_log_entries,
    router_probs_capture_keys,
)
from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments
from param_decomp.targets.lm_output import LMOutput


def _render_selected_figures(
    reductions: dict[str, SiteReduction],
    group_counts: dict[str, int],
    wanted: set[str],
    now_step: int,
) -> DeferredMediaRecord:
    figures = render_slow_eval_figures(reductions, group_counts)
    return DeferredMediaRecord(
        step_key="slow_eval/figure_step",
        step=now_step,
        media={f"slow_eval/{name}": figures[name] for name in wanted},
    )


def _render_permutation(
    spec: PermutationMetricSpec,
    position_ci_by_site: dict[str, PositionCI],
    components: dict[str, tuple[np.ndarray, np.ndarray]] | None,
    include_ci_heatmaps: bool,
    now_step: int,
) -> DeferredMediaRecord:
    figures = render_permutation_figures(spec, position_ci_by_site, components)
    if not include_ci_heatmaps:
        figures = {key: value for key, value in figures.items() if key == "figures/uv_matrices"}
    return DeferredMediaRecord(
        step_key="slow_eval/figure_step",
        step=now_step,
        media={f"slow_eval/{name}": value for name, value in figures.items()},
    )


def make_nonlinearity_operation[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
](
    schedule: EvalSchedule,
    sites: tuple[SiteSpec, ...],
    compiler_options: dict[str, bool | int | str],
    mesh: Mesh,
) -> SharedForwardOperationPlan[
    LMEvalPass[TargetIn, Conditioning], LMBatchContext[TargetIn, PreparedT, Conditioning]
]:
    def prepare(
        example_pass: LMEvalPass[TargetIn, Conditioning],
        example_context: LMBatchContext[TargetIn, PreparedT, Conditioning],
    ) -> SharedForwardOperation[
        LMEvalPass[TargetIn, Conditioning], LMBatchContext[TargetIn, PreparedT, Conditioning]
    ]:
        partitions = nonlinearity_partitions(sites)
        reduction_step = (
            jax.jit(make_ci_reduction_step(0.0, None, None), compiler_options=compiler_options)
            .lower(example_context.forward.ci.preactivations)
            .compile()
        )
        nonlinearity_step = (
            jax.jit(
                make_nonlinearity_eval_step(sites, NamedSharding(mesh, PartitionSpec())),
                compiler_options=compiler_options,
            )
            .lower(example_pass.decomposition.components)
            .compile()
        )

        def update(
            accumulation: SiteReductionAccumulation,
            context: LMBatchContext[TargetIn, PreparedT, Conditioning],
        ) -> SiteReductionAccumulation:
            return fold_site_reduction(
                accumulation, reduction_step(context.forward.ci.preactivations)
            )

        def finish(
            eval_pass: LMEvalPass[TargetIn, Conditioning], accumulation: SiteReductionAccumulation
        ) -> LogRecord:
            reductions = site_reductions(accumulation)
            ci_means = {
                name: value.ci_sums / value.n_positions for name, value in reductions.items()
            }
            return nonlinearity_log_entries(
                site_nonlinearity_stats(
                    nonlinearity_step(eval_pass.decomposition.components), sites
                ),
                ci_means,
                partitions,
            )

        return shared_forward_operation(schedule, empty_site_reduction_accumulation, update, finish)

    return SharedForwardOperationPlan(prepare)


type AnyAttnPatternsMetricConfig = (
    CIMaskedAttnPatternsReconLossConfig | StochasticAttnPatternsReconLossConfig
)
type AnySiteFiguresMetricConfig = (
    CIHistogramsConfig | ComponentActivationDensityConfig | CIMeanPerComponentConfig
)


def attn_patterns_step_for[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    metric: AnyAttnPatternsMetricConfig,
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
) -> AttnPatternsStep[TargetIn, PreparedT, Conditioning, PreparedMaskingT]:
    """THE config→kernel binding for the attention-patterns metrics — the operation and
    the trace gate build the identical masked-side step from one spelling."""
    match metric:
        case CIMaskedAttnPatternsReconLossConfig():
            return make_ci_attn_patterns_step(model)
        case StochasticAttnPatternsReconLossConfig():
            return make_stochastic_attn_patterns_step(model)


def make_attention_operation[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    metric: AnyAttnPatternsMetricConfig,
    schedule: EvalSchedule,
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    run_key: PRNGKeyArray,
    train_steps: int,
    compiler_options: dict[str, bool | int | str],
) -> SharedForwardOperationPlan[
    LMEvalPass[TargetIn, Conditioning], LMBatchContext[TargetIn, PreparedT, Conditioning]
]:
    def prepare(
        _example_pass: LMEvalPass[TargetIn, Conditioning],
        example_context: LMBatchContext[TargetIn, PreparedT, Conditioning],
    ) -> SharedForwardOperation[
        LMEvalPass[TargetIn, Conditioning], LMBatchContext[TargetIn, PreparedT, Conditioning]
    ]:
        masked_step = attn_patterns_step_for(metric, model)
        output_key_by_site = attn_output_key_by_site(model)

        def score(
            model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
            forward: LMBatchForwardProducts[TargetIn, PreparedT, Conditioning],
            pass_index: NDArray[np.uint32],
            batch_index: NDArray[np.uint32],
        ) -> tuple[dict[str, Array], dict[str, int]]:
            base_key = jax.random.fold_in(
                run_key, EvalKeyStream.ATTENTION_PATTERNS * train_steps + pass_index
            )
            return masked_step(
                model,
                forward.prepared_weights,
                forward.conditioning,
                forward.ci.lower,
                {site: forward.captures[key] for site, key in output_key_by_site.items()},
                jax.random.fold_in(base_key, batch_index),
            )

        step = (
            jax.jit(score, compiler_options=compiler_options)
            .lower(model, example_context.forward, np.zeros((), np.uint32), np.zeros((), np.uint32))
            .compile()
        )

        def init() -> dict[str, LayerKLReduction]:
            return {}

        def update(
            reductions: dict[str, LayerKLReduction],
            context: LMBatchContext[TargetIn, PreparedT, Conditioning],
        ) -> dict[str, LayerKLReduction]:
            batch_sum, batch_n = step(
                model,
                context.forward,
                np.asarray(context.pass_index, np.uint32),
                np.asarray(context.batch_index, np.uint32),
            )
            return fold_layer_kl(reductions, batch_sum, batch_n)

        def finish(
            eval_pass: LMEvalPass[TargetIn, Conditioning], reductions: dict[str, LayerKLReduction]
        ) -> LogRecord:
            del eval_pass
            return {
                f"eval/loss/{name}": value
                for name, value in attn_patterns_log_entries(metric.type, reductions).items()
            }

        return shared_forward_operation(schedule, init, update, finish)

    return SharedForwardOperationPlan(prepare)


def make_router_divergence_operation[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    metric: RouterDivergenceConfig,
    schedule: EvalSchedule,
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    run_key: PRNGKeyArray,
    train_steps: int,
    mesh: Mesh,
    compiler_options: dict[str, bool | int | str],
) -> SharedForwardOperationPlan[
    LMEvalPass[TargetIn, Conditioning], LMBatchContext[TargetIn, PreparedT, Conditioning]
]:
    def prepare(
        _example_pass: LMEvalPass[TargetIn, Conditioning],
        example_context: LMBatchContext[TargetIn, PreparedT, Conditioning],
    ) -> SharedForwardOperation[
        LMEvalPass[TargetIn, Conditioning], LMBatchContext[TargetIn, PreparedT, Conditioning]
    ]:
        masked_step = make_router_divergence_step(model, metric, mesh)
        layers = expert_router_model(model).expert_router_layers
        probs_keys = router_probs_capture_keys(model)

        def score(
            model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
            forward: LMBatchForwardProducts[TargetIn, PreparedT, Conditioning],
            persistent_sources: dict[str, SourceStacks],
            pass_index: NDArray[np.uint32],
            batch_index: NDArray[np.uint32],
        ) -> dict[str, tuple[RouterDivergenceSums, int]]:
            base_key = jax.random.fold_in(
                run_key, EvalKeyStream.ROUTER_DIVERGENCE * train_steps + pass_index
            )
            return masked_step(
                model,
                forward.prepared_weights,
                forward.conditioning,
                forward.ci.lower,
                {key: forward.captures[key] for key in probs_keys},
                forward.clean_output,
                persistent_sources,
                jax.random.fold_in(base_key, batch_index),
            )

        step = (
            jax.jit(score, compiler_options=compiler_options)
            .lower(
                model,
                example_context.forward,
                example_context.persistent_sources,
                np.zeros((), np.uint32),
                np.zeros((), np.uint32),
            )
            .compile()
        )

        def init() -> dict[str, RouterDivergenceAccumulation]:
            return {
                strategy.kind: empty_router_divergence_sums(len(layers))
                for strategy in metric.strategies
            }

        def update(
            sums_by_kind: dict[str, RouterDivergenceAccumulation],
            context: LMBatchContext[TargetIn, PreparedT, Conditioning],
        ) -> dict[str, RouterDivergenceAccumulation]:
            batch = step(
                model,
                context.forward,
                context.persistent_sources,
                np.asarray(context.pass_index, np.uint32),
                np.asarray(context.batch_index, np.uint32),
            )
            return {
                kind: fold_router_divergence(sums, *batch[kind])
                for kind, sums in sums_by_kind.items()
            }

        def finish(
            eval_pass: LMEvalPass[TargetIn, Conditioning],
            sums_by_kind: dict[str, RouterDivergenceAccumulation],
        ) -> LogRecord:
            del eval_pass
            entries: LogRecord = {}
            for strategy in metric.strategies:
                dict_safe_update_(
                    entries,
                    router_divergence_log_entries(strategy, layers, sums_by_kind[strategy.kind]),
                )
            return entries

        return shared_forward_operation(schedule, init, update, finish)

    return SharedForwardOperationPlan(prepare)


def site_figures_reduction_step(
    metric: AnySiteFiguresMetricConfig,
) -> CIReductionStep:
    """THE config→kernel binding for the CI-reduction figure metrics — the operation and
    the trace gate build the identical per-batch reduction from one spelling."""
    match metric:
        case CIHistogramsConfig():
            assert metric.n_batches_accum in (None, 1), (
                "CIHistograms bins its values exactly over one eval batch (the counts from "
                f"different batches sit on different edges), so n_batches_accum="
                f"{metric.n_batches_accum} cannot be honoured"
            )
            return make_ci_reduction_step(
                0.0, metric.density_heatmap_n_bins, VALUE_HISTOGRAM_N_BINS
            )
        case ComponentActivationDensityConfig():
            return make_ci_reduction_step(metric.ci_alive_threshold, None, None)
        case CIMeanPerComponentConfig():
            return make_ci_reduction_step(0.0, None, None)


def make_site_figures_operation[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
](
    metric: AnySiteFiguresMetricConfig,
    schedule: EvalSchedule,
    group_counts: dict[str, int],
    compiler_options: dict[str, bool | int | str],
    renderer: BackgroundRenderer,
) -> SharedForwardOperationPlan[
    LMEvalPass[TargetIn, Conditioning], LMBatchContext[TargetIn, PreparedT, Conditioning]
]:
    def prepare(
        _example_pass: LMEvalPass[TargetIn, Conditioning],
        example_context: LMBatchContext[TargetIn, PreparedT, Conditioning],
    ) -> SharedForwardOperation[
        LMEvalPass[TargetIn, Conditioning], LMBatchContext[TargetIn, PreparedT, Conditioning]
    ]:
        reduction_step = (
            jax.jit(site_figures_reduction_step(metric), compiler_options=compiler_options)
            .lower(example_context.forward.ci.preactivations)
            .compile()
        )
        match metric:
            case CIHistogramsConfig():
                wanted = {
                    "figures/causal_importance_values",
                    "figures/causal_importance_values_pre_sigmoid",
                    *(
                        {"figures/ci_density_heatmap"}
                        if metric.density_heatmap_n_bins is not None
                        else set()
                    ),
                }
            case ComponentActivationDensityConfig():
                wanted = {
                    "figures/component_activation_density",
                    *({"figures/component_activation_density_groups"} if group_counts else set()),
                }
            case CIMeanPerComponentConfig():
                wanted = {
                    "figures/ci_mean_per_component",
                    "figures/ci_mean_per_component_log",
                    *({"figures/ci_mean_per_component_groups"} if group_counts else set()),
                }

        def update(
            accumulation: SiteReductionAccumulation,
            context: LMBatchContext[TargetIn, PreparedT, Conditioning],
        ) -> SiteReductionAccumulation:
            return fold_site_reduction(
                accumulation, reduction_step(context.forward.ci.preactivations)
            )

        def finish(
            eval_pass: LMEvalPass[TargetIn, Conditioning], accumulation: SiteReductionAccumulation
        ) -> LogRecord:
            renderer.submit(
                partial(
                    _render_selected_figures,
                    site_reductions(accumulation),
                    group_counts,
                    wanted,
                    eval_pass.now_step,
                )
            )
            return {}

        return shared_forward_operation(schedule, empty_site_reduction_accumulation, update, finish)

    return SharedForwardOperationPlan(prepare)


def make_permutation_operation[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    metric: PermutedCIPlotsConfig | UVPlotsConfig | IdentityCIErrorConfig,
    schedule: EvalSchedule,
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    compiler_options: dict[str, bool | int | str],
    renderer: BackgroundRenderer,
) -> SharedForwardOperationPlan[
    LMEvalPass[TargetIn, Conditioning], LMBatchContext[TargetIn, PreparedT, Conditioning]
]:
    def prepare(
        _example_pass: LMEvalPass[TargetIn, Conditioning],
        example_context: LMBatchContext[TargetIn, PreparedT, Conditioning],
    ) -> SharedForwardOperation[
        LMEvalPass[TargetIn, Conditioning], LMBatchContext[TargetIn, PreparedT, Conditioning]
    ]:
        spec = resolve_permutation_metrics(model.model.site_names, [metric])
        position_step = (
            jax.jit(make_position_ci_step(), compiler_options=compiler_options)
            .lower(example_context.forward.ci.preactivations)
            .compile()
        )

        def update(
            accumulation: PositionCIAccumulation,
            context: LMBatchContext[TargetIn, PreparedT, Conditioning],
        ) -> PositionCIAccumulation:
            return fold_position_ci(accumulation, position_step(context.forward.ci.preactivations))

        def finish(
            eval_pass: LMEvalPass[TargetIn, Conditioning], accumulation: PositionCIAccumulation
        ) -> LogRecord:
            position_ci_by_site = position_ci(accumulation)
            match metric:
                case IdentityCIErrorConfig():
                    errors = compute_identity_ci_errors(
                        spec, position_ci_by_site, IDENTITY_CI_ERROR_TOLERANCE
                    )
                    return {f"eval/slow/{name}": value for name, value in errors.items()}
                case UVPlotsConfig():
                    include_ci_heatmaps = False
                    # The naive whole-stack host gather; the per-site read is a HOST slice,
                    # because the stack axis is sharded under the owner presets.
                    stacks = eval_pass.decomposition.components
                    host = {
                        group: (np.asarray(vs), np.asarray(us))
                        for group, (vs, us) in stacks.stacks.items()
                    }
                    components = {
                        name: (host[group][0][slot], host[group][1][slot])
                        for name, group, slot in stacks.site_stack_indices
                    }
                case PermutedCIPlotsConfig():
                    include_ci_heatmaps = True
                    components = None
            renderer.submit(
                partial(
                    _render_permutation,
                    spec,
                    position_ci_by_site,
                    components,
                    include_ci_heatmaps,
                    eval_pass.now_step,
                )
            )
            return {}

        return shared_forward_operation(schedule, empty_position_ci_accumulation, update, finish)

    return SharedForwardOperationPlan(prepare)
