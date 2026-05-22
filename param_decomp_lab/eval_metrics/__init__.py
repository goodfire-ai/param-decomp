"""Lab eval metrics.

These ship for the in-repo experiments and are referenced from YAML `logging.eval_metrics`
blocks by class name (see `EVAL_METRICS` below). External users can build their own metric
lists and pass them directly to `optimize(eval_metrics=...)`.
"""

from typing import Any

from param_decomp.metrics.base import Metric
from param_decomp.metrics.pgd_masked_recon import PGDReconLoss
from param_decomp.metrics.stochastic_hidden_acts_recon import StochasticHiddenActsReconLoss
from param_decomp_lab.eval_metrics.attn_patterns_recon_loss import (
    CIMaskedAttnPatternsReconLoss,
    StochasticAttnPatternsReconLoss,
)
from param_decomp_lab.eval_metrics.ce_and_kl_losses import CEandKLLosses
from param_decomp_lab.eval_metrics.ci_hidden_acts_recon_loss import CIHiddenActsReconLoss
from param_decomp_lab.eval_metrics.ci_histograms import CIHistograms
from param_decomp_lab.eval_metrics.ci_l0 import CI_L0
from param_decomp_lab.eval_metrics.ci_mean_per_component import CIMeanPerComponent
from param_decomp_lab.eval_metrics.component_activation_density import ComponentActivationDensity
from param_decomp_lab.eval_metrics.identity_ci_error import IdentityCIError
from param_decomp_lab.eval_metrics.permuted_ci_plots import PermutedCIPlots
from param_decomp_lab.eval_metrics.uv_plots import UVPlots

EVAL_METRICS: dict[str, type[Metric[Any]]] = {
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
