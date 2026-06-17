"""Torch-free shaping of a config dict for `wandb.config`.

Both impls log the same flat key layout so cross-impl config queries line up: the torch
trainer dumps a `BaseConfig`, the JAX trainer reuses the raw torch YAML dict. The
flattening rule and the metric `short_name` table both live here (in the shared,
torch-free config package) so neither side has to reach into the torch metric registry.

`METRIC_SHORT_NAMES` is the canonical `type -> short_name` map; the torch metric classes
carry the same `short_name` attribute and a test guards the two against drift.
"""

from typing import Any

METRIC_SHORT_NAMES: dict[str, str] = {
    "CIMaskedReconLayerwiseLoss": "CIMaskReconLayer",
    "CIMaskedReconLoss": "CIMaskRecon",
    "CIMaskedReconSubsetLoss": "CIMaskReconSub",
    "FaithfulnessLoss": "Faith",
    "ImportanceMinimalityLoss": "ImpMin",
    "PersistentPGDReconLoss": "PersistPGDRecon",
    "PGDReconLayerwiseLoss": "PGDReconLayer",
    "PGDReconLoss": "PGDRecon",
    "PGDReconSubsetLoss": "PGDReconSub",
    "StochasticHiddenActsReconLoss": "StochHiddenActRecon",
    "StochasticReconLayerwiseLoss": "StochReconLayer",
    "StochasticReconLoss": "StochRecon",
    "StochasticReconSubsetLoss": "StochReconSub",
    "UnmaskedReconLoss": "UnmaskedRecon",
    "CEandKLLosses": "CEandKL",
    "CIHiddenActsReconLoss": "CIHiddenActRecon",
    "CIHistograms": "CIHist",
    "CI_L0": "CI_L0",
    "CIMaskedAttnPatternsReconLoss": "CIAttnRecon",
    "CIMeanPerComponent": "CIMeanPerComp",
    "ComponentActivationDensity": "CompActDens",
    "IdentityCIError": "IdCIErr",
    "PermutedCIPlots": "PermCIPlots",
    "StochasticAttnPatternsReconLoss": "StochAttnRecon",
    "UVPlots": "UVPlots",
}


def flatten_typed_lists(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested lists-of-typed-dicts (loss/eval metric lists) into queryable flat
    keys addressed by metric `short_name`, returning a copy with the raw lists dropped.

    Example: `pd.loss_metrics: [{type: "ImportanceMinimalityLoss", coeff: 0.1}]`
    becomes `pd.loss_metrics.ImpMin.coeff: 0.1`, and the raw `pd.loss_metrics` list is
    removed so wandb doesn't also log it as an opaque JSON blob. A metric type with no
    entry in `METRIC_SHORT_NAMES` falls back to its raw type string.
    """
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
                        short = METRIC_SHORT_NAMES.get(entry["type"], entry["type"])
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

    out = dict(config_dict)
    walk(out, "")
    out.update(flattened)
    return out
