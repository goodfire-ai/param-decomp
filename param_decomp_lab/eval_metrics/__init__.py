"""Lab eval metrics shipped for the in-repo experiments.

YAML `eval.metrics` entries are validated against `AnyEvalMetricConfig` (in
`param_decomp_config.eval_metrics`) and dispatched to the matching `Metric` subclass
via `EVAL_METRIC_CLASSES`. External users instantiate
their own eval metrics directly and pass them in `EvalLoop(metrics=...)`.
"""

from typing import Any

from param_decomp.metrics.base import Metric
from param_decomp.metrics.dispatch import LOSS_METRIC_CLASSES
from param_decomp.metrics.pgd_masked_recon import PGDReconLoss
from param_decomp.metrics.stochastic_hidden_acts_recon import StochasticHiddenActsReconLoss
from param_decomp_config.base import BaseConfig
from param_decomp_config.wandb_config import flatten_typed_lists
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


def metric_short_names() -> dict[str, str]:
    """Map metric class name (== config `type`) to `short_name`, derived from the loss +
    eval metric registries. The canonical wandb-facing copy is the torch-free
    `param_decomp_config.wandb_config.METRIC_SHORT_NAMES`; a test guards the two against
    drift.
    """
    return {
        cls.__name__: cls.short_name
        for cls in (*LOSS_METRIC_CLASSES.values(), *EVAL_METRIC_CLASSES.values())
        if cls.short_name is not None
    }


def wandb_config_dict(config: BaseConfig) -> dict[str, Any]:
    """Render `config` to the dict logged to `wandb.config`.

    Nested lists-of-typed-dicts (loss/eval metric lists) are flattened into queryable
    flat keys addressed by metric `short_name`, and the raw lists dropped so wandb
    doesn't also log them as opaque JSON blobs. The flattening (and the `short_name`
    table) live in `param_decomp_config.wandb_config` so the JAX trainer can produce the
    identical key layout without reaching into the torch metric registry; the table there
    is guarded against this registry-derived one by a test.
    """
    return flatten_typed_lists(config.model_dump(mode="json"))
