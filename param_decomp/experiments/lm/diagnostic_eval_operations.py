"""Attention, hidden-activation, and causal-importance diagnostic operations."""

from functools import partial

import jax
import numpy as np
from jaxtyping import PRNGKeyArray

from param_decomp.core.ci_fn import CIFn
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
from param_decomp.core.model import CaptureKeys, DecomposedModel
from param_decomp.core.run import (
    BackgroundRenderer,
    DeferredMediaRecord,
    EvalOperation,
)
from param_decomp.core.slow_eval import (
    IDENTITY_CI_ERROR_TOLERANCE,
    PermutationMetricSpec,
    PositionCI,
    SiteReduction,
    accumulate_position_ci,
    accumulate_site_reductions,
    compute_identity_ci_errors,
    make_position_ci_step,
    make_slow_eval_step,
    mean_cis,
    plot_mean_component_cis_two_streams,
    plot_weight_magnitudes,
    render_permutation_figures,
    render_slow_eval_figures,
    resolve_permutation_metrics,
    weight_magnitudes,
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
from param_decomp.experiments.lm.scalar_eval_operations import (
    Stream,
    stream_batches,
    stream_log_prefix,
)


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


def make_attention_operation(
    metric: CIMaskedAttnPatternsReconLossConfig | StochasticAttnPatternsReconLossConfig,
    schedule: EvalSchedule,
    model: DecomposedModel,
    ci_capture_keys: CaptureKeys,
    run_key: PRNGKeyArray,
    train_steps: int,
    compiler_options: dict[str, bool | int | str],
    stream: Stream,
) -> EvalOperation[LMEvalContext]:
    match metric:
        case CIMaskedAttnPatternsReconLossConfig():
            step = make_ci_attn_patterns_step(model, ci_capture_keys, compiler_options)
        case StochasticAttnPatternsReconLossConfig():
            step = make_stochastic_attn_patterns_step(
                model, ci_capture_keys, metric.n_mask_samples, compiler_options
            )

    def run(context: LMEvalContext) -> LogRecord:
        reductions = accumulate_attn_patterns(
            step,
            model,
            context.state.decomposition.components,
            context.state.decomposition.ci_fn,
            list(stream_batches(stream, context)),
            jax.random.fold_in(
                run_key, EvalKeyStream.ATTENTION_PATTERNS * train_steps + context.pass_index
            ),
        )
        prefix = stream_log_prefix(stream, context)
        return {
            f"{prefix}loss/{name}": value
            for name, value in attn_patterns_log_entries(metric.type, reductions).items()
        }

    return EvalOperation(schedule, run)


def make_hidden_acts_operation(
    metric: CIHiddenActsReconLossConfig | StochasticHiddenActsReconLossConfig,
    schedule: EvalSchedule,
    model: DecomposedModel,
    ci_capture_keys: CaptureKeys,
    run_key: PRNGKeyArray,
    train_steps: int,
    compiler_options: dict[str, bool | int | str],
    stream: Stream,
) -> EvalOperation[LMEvalContext]:
    match metric:
        case CIHiddenActsReconLossConfig():
            step = make_ci_hidden_acts_step(model, ci_capture_keys, compiler_options)
        case StochasticHiddenActsReconLossConfig():
            step = make_stochastic_hidden_acts_step(
                model, ci_capture_keys, metric.n_mask_samples, compiler_options
            )

    def run(context: LMEvalContext) -> LogRecord:
        reductions = accumulate_hidden_acts(
            step,
            model,
            context.state.decomposition.components,
            context.state.decomposition.ci_fn,
            list(stream_batches(stream, context)),
            jax.random.fold_in(
                run_key, EvalKeyStream.HIDDEN_ACTS * train_steps + context.pass_index
            ),
        )
        prefix = stream_log_prefix(stream, context)
        return {
            f"{prefix}slow/loss/{name}": value
            for name, value in hidden_acts_log_entries(metric.type, reductions).items()
        }

    return EvalOperation(schedule, run)


def _render_weight_magnitudes(
    magnitudes: dict[str, np.ndarray], now_step: int
) -> DeferredMediaRecord:
    return DeferredMediaRecord(
        step_key="slow_eval/figure_step",
        step=now_step,
        media={"slow_eval/figures/weight_magnitude": plot_weight_magnitudes(magnitudes)},
    )


def make_weight_magnitude_operation(
    schedule: EvalSchedule, renderer: BackgroundRenderer
) -> EvalOperation[LMEvalContext]:
    """`‖V_c‖·‖U_c‖` per site. Reads the trained V/U only — no model, no batch, no step."""

    def run(context: LMEvalContext) -> LogRecord:
        magnitudes = weight_magnitudes(context.state.decomposition.components)
        renderer.submit(partial(_render_weight_magnitudes, magnitudes, context.now_step))
        return {}

    return EvalOperation(schedule, run)


def _render_two_stream_ci_means(
    target: dict[str, np.ndarray],
    nontarget: dict[str, np.ndarray],
    now_step: int,
) -> DeferredMediaRecord:
    linear, log = plot_mean_component_cis_two_streams(target, nontarget)
    return DeferredMediaRecord(
        step_key="slow_eval/figure_step",
        step=now_step,
        media={
            "slow_eval/figures/ci_mean_per_component_two_streams": linear,
            "slow_eval/figures/ci_mean_per_component_two_streams_log": log,
        },
    )


def make_two_stream_ci_mean_operation(
    schedule: EvalSchedule,
    model: DecomposedModel,
    ci_capture_keys: CaptureKeys,
    compiler_options: dict[str, bool | int | str],
    renderer: BackgroundRenderer,
) -> EvalOperation[LMEvalContext]:
    """Both streams' mean CI per component in one figure, ordered by the target mean."""
    step = make_slow_eval_step(model, ci_capture_keys, 0.0, None, compiler_options)

    def stream_mean_cis(ci_fn: CIFn, batches: tuple[jax.Array, ...]) -> dict[str, np.ndarray]:
        return mean_cis(
            accumulate_site_reductions(step, model, ci_fn, list(batches), n_batches_accum=0)
        )

    def run(context: LMEvalContext) -> LogRecord:
        ci_fn = context.state.decomposition.ci_fn
        renderer.submit(
            partial(
                _render_two_stream_ci_means,
                stream_mean_cis(ci_fn, stream_batches("target", context)),
                stream_mean_cis(ci_fn, stream_batches("nontarget", context)),
                context.now_step,
            )
        )
        return {}

    return EvalOperation(schedule, run)


def make_site_figures_operation(
    metric: CIHistogramsConfig | ComponentActivationDensityConfig | CIMeanPerComponentConfig,
    schedule: EvalSchedule,
    model: DecomposedModel,
    ci_capture_keys: CaptureKeys,
    compiler_options: dict[str, bool | int | str],
    renderer: BackgroundRenderer,
    stream: Stream,
) -> EvalOperation[LMEvalContext]:
    match metric:
        case CIHistogramsConfig():
            threshold = 0.0
            bins = metric.density_heatmap_n_bins
            limit = metric.n_batches_accum
            wanted = {
                "figures/causal_importance_values",
                "figures/causal_importance_values_pre_sigmoid",
                *({"figures/ci_density_heatmap"} if bins is not None else set()),
            }
        case ComponentActivationDensityConfig():
            threshold = metric.ci_alive_threshold
            bins = None
            limit = None
            wanted = {"figures/component_activation_density"}
        case CIMeanPerComponentConfig():
            threshold = 0.0
            bins = None
            limit = None
            wanted = {
                "figures/ci_mean_per_component",
                "figures/ci_mean_per_component_log",
            }
    step = make_slow_eval_step(model, ci_capture_keys, threshold, bins, compiler_options)

    def run(context: LMEvalContext) -> LogRecord:
        reductions = accumulate_site_reductions(
            step,
            model,
            context.state.decomposition.ci_fn,
            list(stream_batches(stream, context)),
            limit,
        )
        renderer.submit(partial(_render_selected_figures, reductions, wanted, context.now_step))
        return {}

    return EvalOperation(schedule, run)


def make_permutation_operation(
    metric: PermutedCIPlotsConfig | UVPlotsConfig | IdentityCIErrorConfig,
    schedule: EvalSchedule,
    model: DecomposedModel,
    ci_capture_keys: CaptureKeys,
    compiler_options: dict[str, bool | int | str],
    renderer: BackgroundRenderer,
    stream: Stream,
) -> EvalOperation[LMEvalContext]:
    spec = resolve_permutation_metrics(model.site_names, [metric])
    position_step = make_position_ci_step(model, ci_capture_keys, compiler_options)

    def run(context: LMEvalContext) -> LogRecord:
        position_ci = accumulate_position_ci(
            position_step,
            model,
            context.state.decomposition.ci_fn,
            list(stream_batches(stream, context)),
        )
        match metric:
            case IdentityCIErrorConfig():
                errors = compute_identity_ci_errors(spec, position_ci, IDENTITY_CI_ERROR_TOLERANCE)
                prefix = stream_log_prefix(stream, context)
                return {f"{prefix}slow/{name}": value for name, value in errors.items()}
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
