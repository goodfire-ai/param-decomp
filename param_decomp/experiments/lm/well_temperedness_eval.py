"""Run and log the well-temperedness measurement during LM evaluation."""

import io
import math
from functools import partial
from typing import Literal

import jax
import numpy as np
from jax.sharding import Mesh
from jaxtyping import PRNGKeyArray
from matplotlib.figure import Figure

from param_decomp.core.ci_fn.interface import CIFn
from param_decomp.core.ci_l0_eval import resolve_site_groups
from param_decomp.core.components import ComponentStacks
from param_decomp.core.eval_schedule import EvalSchedule
from param_decomp.core.metrics import LogRecord, PNGImage
from param_decomp.core.model import CaptureKeys, ComponentActivations, PlacedModel
from param_decomp.core.run import (
    BackgroundRenderer,
    DeferredMediaRecord,
    StandaloneOperation,
    StandaloneOperationPlan,
)
from param_decomp.experiments.lm.eval_config import WellTemperednessConfig
from param_decomp.experiments.lm.eval_context import LMEvalPass
from param_decomp.experiments.lm.eval_keys import EvalKeyStream
from param_decomp.experiments.lm.well_temperedness import (
    REGIONS,
    Ablations,
    Region,
    in_region,
    make_well_temperedness_step,
    well_temperedness_log_entries,
)
from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments
from param_decomp.targets.lm_output import LMOutput

_PREFIX = "eval/slow/well_temperedness/"
_FIGURE_KEY = f"{_PREFIX}figures/preactivation_vs_ablation_damage"
_MAX_FIGURE_LOCATIONS = 48
type FigureRendering = BackgroundRenderer | Literal["synchronous"] | None

_REGION_STYLES: dict[Region, tuple[str, str]] = {
    "below_zero": ("0.55", "preactivation <= 0 (CI = 0)"),
    "zero_to_one": ("tab:blue", "0 < preactivation < 1"),
    "above_one": ("tab:red", "preactivation >= 1 (CI = 1)"),
}


def _resolve_groups(
    site_names: tuple[str, ...], configured_groups: dict[str, list[str]] | None
) -> dict[str, tuple[int, ...]]:
    if configured_groups is None:
        return {}
    assert "all_sites" not in configured_groups, (
        "WellTemperedness group name 'all_sites' is reserved"
    )
    resolved = resolve_site_groups(
        site_names, {name: tuple(patterns) for name, patterns in configured_groups.items()}
    )
    site_indices = {site_name: index for index, site_name in enumerate(site_names)}
    return {
        name: tuple(site_indices[site_name] for site_name in matching_sites)
        for name, matching_sites in resolved.items()
    }


def _plot_preactivation_vs_damage(ablations: Ablations) -> bytes:
    n_locations = ablations.preactivations.shape[1]
    n_rows = min(n_locations, 6)
    n_cols = math.ceil(n_locations / n_rows)
    figure = Figure(figsize=(4.5 * n_cols, 3.5 * n_rows), layout="constrained")
    axes = figure.subplots(n_rows, n_cols, squeeze=False).T.ravel()
    for location_index, axis in enumerate(axes[:n_locations]):
        for region_index, region in enumerate(REGIONS):
            colour, label = _REGION_STYLES[region]
            preactivations = np.asarray(ablations.preactivations[region_index, location_index])
            damage = np.asarray(ablations.damage[region_index, location_index])
            in_region_mask = in_region(preactivations, region)
            axis.scatter(
                preactivations[in_region_mask],
                damage[in_region_mask],
                s=6,
                alpha=0.5,
                c=colour,
                label=label,
            )
        axis.set_yscale("symlog", linthresh=1e-12)
        axis.axvline(0.0, color="k", lw=0.6, ls=":")
        axis.axvline(1.0, color="k", lw=0.6, ls=":")
        axis.set_title(f"input location {location_index}", fontsize=8)
        axis.set_xlabel("causal importance preactivation", fontsize=7)
        axis.set_ylabel("change in reconstruction loss when ablated", fontsize=7)
        axis.tick_params(labelsize=6)
    for axis in axes[n_locations:]:
        axis.set_visible(False)
    axes[0].legend(fontsize=6, loc="lower right")
    figure.suptitle("Components from all heads and layers")
    png_buffer = io.BytesIO()
    figure.savefig(png_buffer, format="png", bbox_inches="tight")
    return png_buffer.getvalue()


def _render_deferred(ablations: Ablations, now_step: int) -> DeferredMediaRecord:
    return DeferredMediaRecord(
        step_key=f"{_PREFIX}figure_step",
        step=now_step,
        media={_FIGURE_KEY: _plot_preactivation_vs_damage(ablations)},
    )


def make_well_temperedness_operation[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    metric: WellTemperednessConfig,
    schedule: EvalSchedule,
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    ci_capture_keys: CaptureKeys,
    mesh: Mesh | None,
    compiler_options: dict[str, bool | int | str],
    run_key: PRNGKeyArray,
    train_steps: int,
    figure_rendering: FigureRendering,
) -> StandaloneOperationPlan[LMEvalPass[TargetIn, Conditioning]]:
    if figure_rendering is not None:
        assert metric.n_locations <= _MAX_FIGURE_LOCATIONS, (
            f"WellTemperedness scatter supports at most {_MAX_FIGURE_LOCATIONS} locations, "
            f"got {metric.n_locations}"
        )
    site_groups = _resolve_groups(model.model.site_names, metric.groups)
    measure_ablations = make_well_temperedness_step(model, ci_capture_keys, metric, mesh)

    def measure(
        model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
        components: ComponentStacks,
        ci_fn: CIFn[Conditioning],
        inputs: TargetIn,
        pass_index: np.uint32,
    ) -> Ablations:
        sampling_key = jax.random.fold_in(
            run_key, EvalKeyStream.WELL_TEMPEREDNESS * train_steps + pass_index
        )
        return measure_ablations(model, components, ci_fn, inputs, sampling_key)

    def prepare(
        example: LMEvalPass[TargetIn, Conditioning],
    ) -> StandaloneOperation[LMEvalPass[TargetIn, Conditioning]]:
        compiled_measurement = (
            jax.jit(measure, compiler_options=compiler_options)
            .lower(
                model,
                example.decomposition.components,
                example.decomposition.ci_fn,
                example.batches[0],
                np.uint32(example.pass_index),
            )
            .compile()
        )

        def run(context: LMEvalPass[TargetIn, Conditioning]) -> LogRecord:
            device_ablations = compiled_measurement(
                model,
                context.decomposition.components,
                context.decomposition.ci_fn,
                context.batches[0],
                np.uint32(context.pass_index),
            )
            ablations = jax.device_get(device_ablations)
            log_record: dict[str, float | PNGImage] = {
                f"{_PREFIX}{name}": value
                for name, value in well_temperedness_log_entries(ablations, site_groups).items()
            }
            match figure_rendering:
                case None:
                    pass
                case "synchronous":
                    log_record[_FIGURE_KEY] = PNGImage(_plot_preactivation_vs_damage(ablations))
                case BackgroundRenderer() as renderer:
                    renderer.submit(partial(_render_deferred, ablations, context.now_step))
            return log_record

        return StandaloneOperation(schedule, run)

    return StandaloneOperationPlan(prepare)
