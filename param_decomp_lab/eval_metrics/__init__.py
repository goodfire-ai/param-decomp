"""Lab eval metrics.

These ship for the in-repo experiments and are referenced from YAML
`logging.eval_metrics` blocks. The YAML form is a list of dicts, each carrying a `type:
"<ClassName>"` discriminator (mirrors the loss-metrics pattern). `AnyEvalMetricConfig` is
the pydantic discriminated union used to validate each entry; `EVAL_METRIC_CLASSES` is the
runtime dispatch from the `type` literal to the matching `Metric` subclass.

External users defining their own eval metric instantiate it directly in `run.py` and
include it in the `EvalLoop(metrics=...)` they pass to `optimize`.
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
