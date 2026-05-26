"""Dispatch from `PDConfig.loss_metrics` entries to bound `Metric` instances.

The `type` literal -> class table is `LOSS_METRIC_CLASSES`.
"""

from typing import Any

from param_decomp.component_model import ComponentModel
from param_decomp.configs import PDConfig
from param_decomp.metrics.base import Metric
from param_decomp.metrics.ci_masked_recon import CIMaskedReconLoss
from param_decomp.metrics.ci_masked_recon_layerwise import CIMaskedReconLayerwiseLoss
from param_decomp.metrics.ci_masked_recon_subset import CIMaskedReconSubsetLoss
from param_decomp.metrics.faithfulness import FaithfulnessLoss
from param_decomp.metrics.importance_minimality import ImportanceMinimalityLoss
from param_decomp.metrics.persistent_pgd_recon import (
    PersistentPGDReconLoss,
    PersistentPGDReconSubsetLoss,
)
from param_decomp.metrics.pgd_masked_recon import PGDReconLoss
from param_decomp.metrics.pgd_masked_recon_layerwise import PGDReconLayerwiseLoss
from param_decomp.metrics.pgd_masked_recon_subset import PGDReconSubsetLoss
from param_decomp.metrics.stochastic_hidden_acts_recon import StochasticHiddenActsReconLoss
from param_decomp.metrics.stochastic_recon import StochasticReconLoss
from param_decomp.metrics.stochastic_recon_layerwise import StochasticReconLayerwiseLoss
from param_decomp.metrics.stochastic_recon_subset import StochasticReconSubsetLoss
from param_decomp.metrics.unmasked_recon import UnmaskedReconLoss

LOSS_METRIC_CLASSES: dict[str, type[Metric[Any]]] = {
    cls.__name__: cls
    for cls in (
        CIMaskedReconLayerwiseLoss,
        CIMaskedReconLoss,
        CIMaskedReconSubsetLoss,
        FaithfulnessLoss,
        ImportanceMinimalityLoss,
        PersistentPGDReconLoss,
        PersistentPGDReconSubsetLoss,
        PGDReconLayerwiseLoss,
        PGDReconLoss,
        PGDReconSubsetLoss,
        StochasticHiddenActsReconLoss,
        StochasticReconLayerwiseLoss,
        StochasticReconLoss,
        StochasticReconSubsetLoss,
        UnmaskedReconLoss,
    )
}


def instantiate_metrics(
    pd_config: PDConfig,
    component_model: ComponentModel,
    device: str,
    eval_metrics: list[Metric[Any]] | None = None,
) -> tuple[dict[str, Metric[Any]], dict[str, Metric[Any]]]:
    """Instantiate loss metrics from config and bind caller-supplied eval metrics.

    Loss metrics are auto-evaluated alongside dedicated eval metrics, so eval metrics
    whose class name collides with a loss metric are rejected. Returns
    `(loss_instances, eval_instances)`: `loss_instances` is keyed by `type` literal;
    `eval_instances` is the full eval-pass set (loss + eval-only) keyed by class name.
    """
    loss_instances: dict[str, Metric[Any]] = {}
    for cfg in pd_config.loss_metrics:
        assert cfg.type not in loss_instances, f"duplicate loss metric {cfg.type!r}"
        m = LOSS_METRIC_CLASSES[cfg.type](cfg)
        m.bind(model=component_model, device=device)
        loss_instances[cfg.type] = m

    eval_only_instances: dict[str, Metric[Any]] = {}
    if eval_metrics is not None:
        for m in eval_metrics:
            m.bind(model=component_model, device=device)
            metric_name = type(m).__name__
            assert metric_name not in eval_only_instances, f"duplicate eval metric {metric_name!r}"
            eval_only_instances[metric_name] = m
        overlap = sorted(set(loss_instances) & set(eval_only_instances))
        assert not overlap, (
            f"eval metrics overlap with pd_config.loss_metrics: {overlap}. Loss metrics "
            "are automatically evaluated; remove the duplicates from eval metrics."
        )
    eval_instances = {**loss_instances, **eval_only_instances}
    return loss_instances, eval_instances
