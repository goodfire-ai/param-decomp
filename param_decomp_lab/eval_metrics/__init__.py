"""Lab eval metrics shipped for the in-repo experiments.

YAML `eval.metrics` entries are validated against `AnyEvalMetricConfig` and dispatched
to the matching `Metric` subclass via `EVAL_METRIC_CLASSES`. External users instantiate
their own eval metrics directly and pass them in `EvalLoop(metrics=...)`.
"""

from typing import Annotated, Any

from pydantic import Discriminator

from param_decomp.metrics.base import Metric
from param_decomp.metrics.dispatch import LOSS_METRIC_CLASSES
from param_decomp.metrics.pgd_masked_recon import PGDReconLoss, PGDReconLossConfig
from param_decomp.metrics.stochastic_hidden_acts_recon import (
    StochasticHiddenActsReconLoss,
    StochasticHiddenActsReconLossConfig,
)
from param_decomp_lab.eval_metrics.attn_patterns_recon_loss import (
    CIMaskedAttnPatternsReconLoss,
    CIMaskedAttnPatternsReconLossConfig,
    StochasticAttnPatternsReconLoss,
    StochasticAttnPatternsReconLossConfig,
)
from param_decomp_lab.eval_metrics.autointerp_labels import (
    AutointerpLabels,
    AutointerpLabelsConfig,
    AutointerpRunContext,
)
from param_decomp_lab.eval_metrics.ce_and_kl_losses import CEandKLLosses, CEandKLLossesConfig
from param_decomp_lab.eval_metrics.ci_hidden_acts_recon_loss import (
    CIHiddenActsReconLoss,
    CIHiddenActsReconLossConfig,
)
from param_decomp_lab.eval_metrics.ci_histograms import CIHistograms, CIHistogramsConfig
from param_decomp_lab.eval_metrics.ci_l0 import CI_L0, CI_L0Config
from param_decomp_lab.eval_metrics.ci_mean_per_component import (
    CIMeanPerComponent,
    CIMeanPerComponentConfig,
)
from param_decomp_lab.eval_metrics.component_activation_density import (
    ComponentActivationDensity,
    ComponentActivationDensityConfig,
)
from param_decomp_lab.eval_metrics.identity_ci_error import IdentityCIError, IdentityCIErrorConfig
from param_decomp_lab.eval_metrics.permuted_ci_plots import PermutedCIPlots, PermutedCIPlotsConfig
from param_decomp_lab.eval_metrics.uv_plots import UVPlots, UVPlotsConfig

AnyEvalMetricConfig = Annotated[
    AutointerpLabelsConfig
    | CEandKLLossesConfig
    | CIHiddenActsReconLossConfig
    | CIHistogramsConfig
    | CI_L0Config
    | CIMaskedAttnPatternsReconLossConfig
    | CIMeanPerComponentConfig
    | ComponentActivationDensityConfig
    | IdentityCIErrorConfig
    | PermutedCIPlotsConfig
    | PGDReconLossConfig
    | StochasticAttnPatternsReconLossConfig
    | StochasticHiddenActsReconLossConfig
    | UVPlotsConfig,
    Discriminator("type"),
]

EVAL_METRIC_CLASSES: dict[str, type[Metric[Any]]] = {
    cls.__name__: cls
    for cls in (
        AutointerpLabels,
        CEandKLLosses,
        CIHiddenActsReconLoss,
        CIHistograms,
        CI_L0,
        CIMaskedAttnPatternsReconLoss,
        CIMeanPerComponent,
        ComponentActivationDensity,
        IdentityCIError,
        PermutedCIPlots,
        PGDReconLoss,
        StochasticAttnPatternsReconLoss,
        StochasticHiddenActsReconLoss,
        UVPlots,
    )
}


def build_eval_metrics(
    configs: list[AnyEvalMetricConfig],
    *,
    autointerp_run_context: AutointerpRunContext | None,
) -> list[Metric[Any]]:
    """Instantiate eval metrics from their configs.

    Most metrics are built from config alone. `AutointerpLabels` additionally needs
    run/data context (`AutointerpRunContext`) the generic dispatch can't supply, so it
    is constructed explicitly; pass `autointerp_run_context` whenever such a metric may
    appear (asserted present if one does).
    """
    metrics: list[Metric[Any]] = []
    for c in configs:
        if isinstance(c, AutointerpLabelsConfig):
            assert autointerp_run_context is not None, (
                "AutointerpLabels eval metric requires an AutointerpRunContext"
            )
            metrics.append(AutointerpLabels(c, autointerp_run_context))
        else:
            metrics.append(EVAL_METRIC_CLASSES[c.type](c))
    return metrics


def metric_short_names() -> dict[str, str]:
    """Map metric class name (== config `type`) to `short_name`, for wandb key prettifying.

    Covers both loss and eval metrics. Lives here (not in `infra/wandb`) so the infra
    layer doesn't depend upward on the metric registries.
    """
    return {
        cls.__name__: cls.short_name
        for cls in (*LOSS_METRIC_CLASSES.values(), *EVAL_METRIC_CLASSES.values())
        if cls.short_name is not None
    }
