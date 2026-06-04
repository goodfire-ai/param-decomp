"""Lab eval metrics shipped for the in-repo experiments.

YAML `eval.metrics` entries are validated against `AnyEvalMetricConfig` and dispatched
to the matching `Metric` subclass via `EVAL_METRIC_CLASSES`. External users instantiate
their own eval metrics directly and pass them in `EvalLoop(metrics=...)`.
"""

from typing import Annotated, Any

from pydantic import Discriminator

from param_decomp.base_config import BaseConfig
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


def _metric_short_names() -> dict[str, str]:
    """Map metric class name (== config `type`) to `short_name`, for wandb key prettifying.

    Covers both loss and eval metrics.
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
    doesn't also log them as opaque JSON blobs. Lives here (not in `infra/wandb`) so the
    infra layer doesn't depend upward on the metric registries.
    """
    short_names = _metric_short_names()
    config_dict = config.model_dump(mode="json")
    flattened: dict[str, Any] = {}

    def is_typed_list(obj: Any) -> bool:
        return (
            isinstance(obj, list)
            and len(obj) > 0
            and all(isinstance(x, dict) and "type" in x for x in obj)
        )

    def walk(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for key in list(obj.keys()):
                child = obj[key]
                child_path = f"{path}.{key}" if path else key
                if is_typed_list(child):
                    for entry in child:
                        short = short_names.get(entry["type"], entry["type"])
                        for k, v in entry.items():
                            if k == "type":
                                continue
                            flattened[f"{child_path}.{short}.{k}"] = v
                    del obj[key]
                else:
                    walk(child, child_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                walk(item, f"{path}.{i}")

    walk(config_dict, "")
    config_dict.update(flattened)
    return config_dict
