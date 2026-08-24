"""Pure figure rendering for positionless toy targets.

Toy composition owns the single-feature probe and target-specific permutation. This module
turns those already-host-side values into transport-independent PNG metric values; the shared
metrics sink is the only code that knows about W&B.
"""

from typing import Literal

import numpy as np
from jaxtyping import Array

from param_decomp.core.components import SiteComponents
from param_decomp.core.configs import (
    Checkpointing,
    NoCheckpointing,
    PeriodicCheckpointing,
    UVPlotsConfig,
)
from param_decomp.core.metrics import LogRecord, PNGImage
from param_decomp.core.model import PlacedModel
from param_decomp.core.slow_eval import (
    PermutationMetricSpec,
    PositionCI,
    plot_permuted_ci_heatmaps,
    render_uv_figure,
    resolve_permutation_metrics,
)


def toy_uv_spec(model: PlacedModel, metric: UVPlotsConfig | None) -> PermutationMetricSpec:
    """Resolve the optional typed UV-plot metric over the toy model's sites."""
    return resolve_permutation_metrics(model.site_names, [] if metric is None else [metric])


def render_uv_metric(
    spec: PermutationMetricSpec,
    components_vu: dict[str, SiteComponents],
    probe_ci_upper: dict[str, Array],
) -> LogRecord:
    """Render the authored ``UVPlots`` operation into typed PNG values."""
    assert spec.want_uv_plots, "UVPlots renderer requires an authored UVPlots metric"
    components = {
        name: (np.asarray(site_components.V), np.asarray(site_components.U))
        for name, site_components in components_vu.items()
    }
    perm_source = {name: np.asarray(probe_ci_upper[name]) for name in spec.permutation}
    return {
        f"slow_eval/{key}": PNGImage(encoded)
        for key, encoded in render_uv_figure(spec, components, perm_source).items()
    }


def permuted_ci_heatmap_due(now_step: int, total_steps: int, checkpointing: Checkpointing) -> bool:
    """Emit native recovery figures beside checkpoints and at the final step (final step
    only when the run checkpoints nothing)."""
    match checkpointing:
        case PeriodicCheckpointing(save_every=save_every):
            return now_step == total_steps or now_step % save_every == 0
        case NoCheckpointing():
            return now_step == total_steps


def render_permuted_ci_heatmap(
    ci_lower: dict[str, Array],
    ci_upper: dict[str, Array],
    permutation: dict[str, Literal["identity", "dense"]],
) -> LogRecord:
    """Render both lower and upper-leaky single-feature-probe CI views."""
    position_ci = {
        name: PositionCI(lower=np.asarray(ci_lower[name]), upper=np.asarray(ci_upper[name]))
        for name in ci_lower
    }
    lower_png, upper_png = plot_permuted_ci_heatmaps(position_ci, permutation)
    return {
        "slow_eval/figures/causal_importances": PNGImage(lower_png),
        "slow_eval/figures/causal_importances_upper_leaky": PNGImage(upper_png),
    }
