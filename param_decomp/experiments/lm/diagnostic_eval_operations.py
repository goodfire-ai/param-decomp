"""Attention and causal-importance diagnostic operations.

Every operation here is a `BatchedOperation` folding over the pass's shared batch
contexts: the CI-reduction family is a cheap on-device reduction of the context's CI
envelope, and the masked-forward metrics (attention patterns) run only
their masked side — the clean side comes from the context.
"""

from collections.abc import Mapping
from functools import partial

import jax
import numpy as np
from jaxtyping import PRNGKeyArray

from param_decomp.core.configs import (
    CIHistogramsConfig,
    CIMeanPerComponentConfig,
    ComponentActivationDensityConfig,
    IdentityCIErrorConfig,
    PermutedCIPlotsConfig,
    UVPlotsConfig,
)
from param_decomp.core.eval_schedule import EvalSchedule
from param_decomp.core.metrics import LogRecord
from param_decomp.core.model import PlacedModel
from param_decomp.core.nonlinearity import NonlinearityPartition
from param_decomp.core.nonlinearity_eval import (
    make_nonlinearity_eval_step,
    nonlinearity_log_entries,
)
from param_decomp.core.run import (
    BackgroundRenderer,
    BatchedOperation,
    DeferredMediaRecord,
    batched_operation,
)
from param_decomp.core.slow_eval import (
    IDENTITY_CI_ERROR_TOLERANCE,
    VALUE_HISTOGRAM_N_BINS,
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
    LayerKLReduction,
    attn_output_key_by_site,
    attn_patterns_log_entries,
    fold_layer_kl,
    make_ci_attn_patterns_step,
    make_stochastic_attn_patterns_step,
)
from param_decomp.experiments.lm.eval_config import (
    CIMaskedAttnPatternsReconLossConfig,
    StochasticAttnPatternsReconLossConfig,
)
from param_decomp.experiments.lm.eval_context import LMBatchContext, LMEvalPass
from param_decomp.experiments.lm.eval_keys import EvalKeyStream


def _render_selected_figures(
    reductions: dict[str, SiteReduction], wanted: set[str], now_step: int
) -> DeferredMediaRecord:
    figures = render_slow_eval_figures(reductions)
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


def make_nonlinearity_operation(
    schedule: EvalSchedule,
    partitions: Mapping[str, NonlinearityPartition],
    compiler_options: dict[str, bool | int | str],
) -> BatchedOperation[LMEvalPass, LMBatchContext]:
    reduction_step = make_ci_reduction_step(0.0, None, None, compiler_options)
    nonlinearity_step = make_nonlinearity_eval_step(partitions, compiler_options)

    def update(
        accumulation: SiteReductionAccumulation, context: LMBatchContext
    ) -> SiteReductionAccumulation:
        return fold_site_reduction(accumulation, reduction_step(context.ci.preactivations))

    def finish(eval_pass: LMEvalPass, accumulation: SiteReductionAccumulation) -> LogRecord:
        reductions = site_reductions(accumulation)
        ci_means = {name: value.ci_sums / value.n_positions for name, value in reductions.items()}
        return nonlinearity_log_entries(
            nonlinearity_step(eval_pass.state.decomposition.components), ci_means, partitions
        )

    return batched_operation(schedule, empty_site_reduction_accumulation, update, finish)


def make_attention_operation(
    metric: CIMaskedAttnPatternsReconLossConfig | StochasticAttnPatternsReconLossConfig,
    schedule: EvalSchedule,
    model: PlacedModel,
    run_key: PRNGKeyArray,
    train_steps: int,
    compiler_options: dict[str, bool | int | str],
) -> BatchedOperation[LMEvalPass, LMBatchContext]:
    match metric:
        case CIMaskedAttnPatternsReconLossConfig():
            step = make_ci_attn_patterns_step(model, compiler_options)
        case StochasticAttnPatternsReconLossConfig():
            step = make_stochastic_attn_patterns_step(
                model, metric.n_mask_samples, compiler_options
            )
    output_key_by_site = attn_output_key_by_site(model)

    def init() -> dict[str, LayerKLReduction]:
        return {}

    def update(
        reductions: dict[str, LayerKLReduction], context: LMBatchContext
    ) -> dict[str, LayerKLReduction]:
        base_key = jax.random.fold_in(
            run_key, EvalKeyStream.ATTENTION_PATTERNS * train_steps + context.pass_index
        )
        batch_sum, batch_n = step(
            model,
            context.prepared_weights,
            context.tokens,
            context.ci.lower,
            {site: context.captures[key] for site, key in output_key_by_site.items()},
            jax.random.fold_in(base_key, context.batch_index),
        )
        return fold_layer_kl(reductions, batch_sum, batch_n)

    def finish(eval_pass: LMEvalPass, reductions: dict[str, LayerKLReduction]) -> LogRecord:
        del eval_pass
        return {
            f"eval/loss/{name}": value
            for name, value in attn_patterns_log_entries(metric.type, reductions).items()
        }

    return batched_operation(schedule, init, update, finish)


def make_site_figures_operation(
    metric: CIHistogramsConfig | ComponentActivationDensityConfig | CIMeanPerComponentConfig,
    schedule: EvalSchedule,
    compiler_options: dict[str, bool | int | str],
    renderer: BackgroundRenderer,
) -> BatchedOperation[LMEvalPass, LMBatchContext]:
    match metric:
        case CIHistogramsConfig():
            assert metric.n_batches_accum in (None, 1), (
                "CIHistograms bins its values exactly over one eval batch (the counts from "
                f"different batches sit on different edges), so n_batches_accum="
                f"{metric.n_batches_accum} cannot be honoured"
            )
            bins = metric.density_heatmap_n_bins
            wanted = {
                "figures/causal_importance_values",
                "figures/causal_importance_values_pre_sigmoid",
                *({"figures/ci_density_heatmap"} if bins is not None else set()),
            }
            reduction_step = make_ci_reduction_step(
                0.0, bins, VALUE_HISTOGRAM_N_BINS, compiler_options
            )
        case ComponentActivationDensityConfig():
            wanted = {"figures/component_activation_density"}
            reduction_step = make_ci_reduction_step(
                metric.ci_alive_threshold, None, None, compiler_options
            )
        case CIMeanPerComponentConfig():
            wanted = {
                "figures/ci_mean_per_component",
                "figures/ci_mean_per_component_log",
            }
            reduction_step = make_ci_reduction_step(0.0, None, None, compiler_options)

    def update(
        accumulation: SiteReductionAccumulation, context: LMBatchContext
    ) -> SiteReductionAccumulation:
        return fold_site_reduction(accumulation, reduction_step(context.ci.preactivations))

    def finish(eval_pass: LMEvalPass, accumulation: SiteReductionAccumulation) -> LogRecord:
        renderer.submit(
            partial(
                _render_selected_figures, site_reductions(accumulation), wanted, eval_pass.now_step
            )
        )
        return {}

    return batched_operation(schedule, empty_site_reduction_accumulation, update, finish)


def make_permutation_operation(
    metric: PermutedCIPlotsConfig | UVPlotsConfig | IdentityCIErrorConfig,
    schedule: EvalSchedule,
    model: PlacedModel,
    compiler_options: dict[str, bool | int | str],
    renderer: BackgroundRenderer,
) -> BatchedOperation[LMEvalPass, LMBatchContext]:
    spec = resolve_permutation_metrics(model.site_names, [metric])
    position_step = make_position_ci_step(compiler_options)

    def update(
        accumulation: PositionCIAccumulation, context: LMBatchContext
    ) -> PositionCIAccumulation:
        return fold_position_ci(accumulation, position_step(context.ci.preactivations))

    def finish(eval_pass: LMEvalPass, accumulation: PositionCIAccumulation) -> LogRecord:
        position_ci_by_site = position_ci(accumulation)
        match metric:
            case IdentityCIErrorConfig():
                errors = compute_identity_ci_errors(
                    spec, position_ci_by_site, IDENTITY_CI_ERROR_TOLERANCE
                )
                return {f"eval/slow/{name}": value for name, value in errors.items()}
            case UVPlotsConfig():
                include_ci_heatmaps = False
                components = {
                    name: (np.asarray(site_components.V), np.asarray(site_components.U))
                    for name, site_components in (
                        eval_pass.state.decomposition.components.sites_items()
                    )
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

    return batched_operation(schedule, empty_position_ci_accumulation, update, finish)
