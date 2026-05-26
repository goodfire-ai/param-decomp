"""Lab eval metrics shipped for the in-repo experiments.

YAML `eval.metrics` entries are validated against `AnyEvalMetricConfig` and dispatched
to the matching `Metric` subclass via `EVAL_METRIC_CLASSES`. External users instantiate
their own eval metrics directly and pass them in `EvalLoop(metrics=...)`.
"""

from typing import Annotated, Any

from pydantic import Discriminator

from param_decomp.metrics.base import Metric
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
    CEandKLLossesConfig
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
