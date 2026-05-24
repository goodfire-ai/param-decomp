"""Dispatch from PDConfig.loss_metrics entries to bound Metric instances.

`PDConfig.loss_metrics` is a discriminated union keyed by each config's `type` literal.
`LOSS_METRIC_CLASSES` maps that literal to the matching `Metric` subclass, and
`instantiate_loss_metrics` walks `pd_config.loss_metrics` to build one bound metric per
entry — the form the training loop actually consumes.
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


def instantiate_loss_metrics(
    pd_config: PDConfig,
    component_model: ComponentModel,
    device: str,
) -> dict[str, Metric[Any]]:
    """Instantiate and bind one `Metric` per entry in `pd_config.loss_metrics`.

    Args:
        pd_config: The validated PD config; its `loss_metrics` list drives instantiation.
        component_model: Live `ComponentModel` passed to each metric's `bind`.
        device: Device string passed to each metric's `bind`.

    Returns:
        Dict keyed by each config's `type` literal (e.g. `"FaithfulnessLoss"`).
        Duplicate `type` literals are rejected.
    """
    instances: dict[str, Metric[Any]] = {}
    for cfg in pd_config.loss_metrics:
        assert cfg.type not in instances, f"duplicate loss metric {cfg.type!r}"
        m = LOSS_METRIC_CLASSES[cfg.type](cfg)
        m.bind(model=component_model, device=device)
        instances[cfg.type] = m
    return instances
