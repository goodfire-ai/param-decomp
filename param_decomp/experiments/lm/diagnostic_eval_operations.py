"""Attention, hidden-activation, and causal-importance diagnostic operations."""

from collections.abc import Callable, Mapping
from functools import partial

import jax
import numpy as np
from jaxtyping import PRNGKeyArray

from param_decomp.core.configs import (
    CIHiddenActsReconLossConfig,
    CIHistogramsConfig,
    CIMeanPerComponentConfig,
    ComponentActivationDensityConfig,
    IdentityCIErrorConfig,
    PermutedCIPlotsConfig,
    StochasticHiddenActsReconLossConfig,
    UVPlotsConfig,
)
from param_decomp.core.eval_schedule import EvalSchedule
from param_decomp.core.hidden_acts_eval import (
    accumulate_hidden_acts,
    hidden_acts_log_entries,
    make_ci_hidden_acts_step,
    make_stochastic_hidden_acts_step,
)
from param_decomp.core.metrics import LogRecord
from param_decomp.core.model import CaptureKeys, PlacedModel
from param_decomp.core.nonlinearity import NonlinearityPartition
from param_decomp.core.nonlinearity_eval import (
    make_nonlinearity_eval_step,
    nonlinearity_log_entries,
)
from param_decomp.core.run import (
    BackgroundRenderer,
    DeferredMediaRecord,
    EvalOperation,
)
from param_decomp.core.slow_eval import (
    IDENTITY_CI_ERROR_TOLERANCE,
    VALUE_HISTOGRAM_N_BINS,
    PermutationMetricSpec,
    PositionCI,
    SiteReduction,
    accumulate_position_ci,
    accumulate_site_reductions,
    compute_identity_ci_errors,
    make_position_ci_step,
    make_slow_eval_step,
    render_permutation_figures,
    render_slow_eval_figures,
    resolve_permutation_metrics,
)
from param_decomp.experiments.lm.attn_patterns_eval import (
    accumulate_attn_patterns,
    attn_patterns_log_entries,
    make_ci_attn_patterns_step,
    make_stochastic_attn_patterns_step,
)
from param_decomp.experiments.lm.eval_config import (
    CIMaskedAttnPatternsReconLossConfig,
    StochasticAttnPatternsReconLossConfig,
)
from param_decomp.experiments.lm.eval_context import LMEvalContext
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
    position_ci: dict[str, PositionCI],
    components: dict[str, tuple[np.ndarray, np.ndarray]] | None,
    include_ci_heatmaps: bool,
    now_step: int,
) -> DeferredMediaRecord:
    figures = render_permutation_figures(spec, position_ci, components)
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
) -> EvalOperation[LMEvalContext]:
    nonlinearity_step = make_nonlinearity_eval_step(partitions, compiler_options)

    def run(context: LMEvalContext) -> LogRecord:
        reductions = context.shared_ci_reductions()
        ci_means = {name: value.ci_sums / value.n_positions for name, value in reductions.items()}
        return nonlinearity_log_entries(
            nonlinearity_step(context.state.decomposition.components), ci_means, partitions
        )

    return EvalOperation(schedule, run)


def make_attention_operation(
    metric: CIMaskedAttnPatternsReconLossConfig | StochasticAttnPatternsReconLossConfig,
    schedule: EvalSchedule,
    model: PlacedModel,
    ci_capture_keys: CaptureKeys,
    run_key: PRNGKeyArray,
    train_steps: int,
    compiler_options: dict[str, bool | int | str],
) -> EvalOperation[LMEvalContext]:
    match metric:
        case CIMaskedAttnPatternsReconLossConfig():
            step = make_ci_attn_patterns_step(model, ci_capture_keys, compiler_options)
        case StochasticAttnPatternsReconLossConfig():
            step = make_stochastic_attn_patterns_step(
                model,
                ci_capture_keys,
                metric.n_mask_samples,
                compiler_options,
            )

    def run(context: LMEvalContext) -> LogRecord:
        reductions = accumulate_attn_patterns(
            step,
            model,
            context.state.decomposition.components,
            context.placed_ci_fn,
            list(context.batches),
            jax.random.fold_in(
                run_key, EvalKeyStream.ATTENTION_PATTERNS * train_steps + context.pass_index
            ),
        )
        return {
            f"eval/loss/{name}": value
            for name, value in attn_patterns_log_entries(metric.type, reductions).items()
        }

    return EvalOperation(schedule, run)


def make_hidden_acts_operation(
    metric: CIHiddenActsReconLossConfig | StochasticHiddenActsReconLossConfig,
    schedule: EvalSchedule,
    model: PlacedModel,
    ci_capture_keys: CaptureKeys,
    run_key: PRNGKeyArray,
    train_steps: int,
    compiler_options: dict[str, bool | int | str],
) -> EvalOperation[LMEvalContext]:
    match metric:
        case CIHiddenActsReconLossConfig():
            step = make_ci_hidden_acts_step(model, ci_capture_keys, compiler_options)
        case StochasticHiddenActsReconLossConfig():
            step = make_stochastic_hidden_acts_step(
                model,
                ci_capture_keys,
                metric.n_mask_samples,
                compiler_options,
            )

    def run(context: LMEvalContext) -> LogRecord:
        reductions = accumulate_hidden_acts(
            step,
            model,
            context.state.decomposition.components,
            context.placed_ci_fn,
            list(context.batches),
            jax.random.fold_in(
                run_key, EvalKeyStream.HIDDEN_ACTS * train_steps + context.pass_index
            ),
        )
        return {
            f"eval/slow/loss/{name}": value
            for name, value in hidden_acts_log_entries(metric.type, reductions).items()
        }

    return EvalOperation(schedule, run)


def make_site_figures_operation(
    metric: CIHistogramsConfig | ComponentActivationDensityConfig | CIMeanPerComponentConfig,
    schedule: EvalSchedule,
    model: PlacedModel,
    ci_capture_keys: CaptureKeys,
    compiler_options: dict[str, bool | int | str],
    renderer: BackgroundRenderer,
) -> EvalOperation[LMEvalContext]:
    def own_reductions(
        threshold: float, bins: int | None, value_histogram_n_bins: int | None
    ) -> Callable[[LMEvalContext], dict[str, SiteReduction]]:
        step = make_slow_eval_step(
            model, ci_capture_keys, threshold, bins, value_histogram_n_bins, compiler_options
        )

        def reductions_of(context: LMEvalContext) -> dict[str, SiteReduction]:
            return accumulate_site_reductions(
                step, model, context.placed_ci_fn, list(context.batches)
            )

        return reductions_of

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
            reductions_of = own_reductions(0.0, bins, VALUE_HISTOGRAM_N_BINS)
        case ComponentActivationDensityConfig():
            wanted = {"figures/component_activation_density"}
            reductions_of = own_reductions(metric.ci_alive_threshold, None, None)
        case CIMeanPerComponentConfig():
            wanted = {
                "figures/ci_mean_per_component",
                "figures/ci_mean_per_component_log",
            }
            # threshold 0, no histograms: exactly the pass-shared reduction — no own step
            reductions_of = lambda context: context.shared_ci_reductions()  # noqa: E731

    def run(context: LMEvalContext) -> LogRecord:
        renderer.submit(
            partial(_render_selected_figures, reductions_of(context), wanted, context.now_step)
        )
        return {}

    return EvalOperation(schedule, run)


def make_permutation_operation(
    metric: PermutedCIPlotsConfig | UVPlotsConfig | IdentityCIErrorConfig,
    schedule: EvalSchedule,
    model: PlacedModel,
    ci_capture_keys: CaptureKeys,
    compiler_options: dict[str, bool | int | str],
    renderer: BackgroundRenderer,
) -> EvalOperation[LMEvalContext]:
    spec = resolve_permutation_metrics(model.site_names, [metric])
    position_step = make_position_ci_step(model, ci_capture_keys, compiler_options)

    def run(context: LMEvalContext) -> LogRecord:
        position_ci = accumulate_position_ci(
            position_step,
            model,
            context.placed_ci_fn,
            list(context.batches),
        )
        match metric:
            case IdentityCIErrorConfig():
                errors = compute_identity_ci_errors(spec, position_ci, IDENTITY_CI_ERROR_TOLERANCE)
                return {f"eval/slow/{name}": value for name, value in errors.items()}
            case UVPlotsConfig():
                include_ci_heatmaps = False
                components = {
                    name: (np.asarray(site_components.V), np.asarray(site_components.U))
                    for name, site_components in context.state.decomposition.components.sites_items()
                }
            case PermutedCIPlotsConfig():
                include_ci_heatmaps = True
                components = None
        renderer.submit(
            partial(
                _render_permutation,
                spec,
                position_ci,
                components,
                include_ci_heatmaps,
                context.now_step,
            )
        )
        return {}

    return EvalOperation(schedule, run)
