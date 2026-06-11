"""Lab-side loss-metric dispatch: the core table plus lab-only metric classes.

Lab loss metrics depend on lab code (the vendored `LMComponentModel`, the recon-plan
machinery) that core cannot import, so they can't join core's `LOSS_METRIC_CLASSES`.
Their *configs* live in core (so they validate in `pd.loss_metrics`); this table maps
those `type` literals to the lab impl classes. Lab trainers use `instantiate_lab_metrics`
in place of core's `instantiate_metrics`.
"""

from typing import Any

from param_decomp.component_model import ComponentModelProtocol
from param_decomp.metrics.base import Metric
from param_decomp.metrics.dispatch import LOSS_METRIC_CLASSES
from param_decomp_config.pd import PDConfig
from param_decomp_lab.metrics.chunkwise_subset_recon import ChunkwiseSubsetReconLoss
from param_decomp_lab.metrics.fused_persistent_pgd_recon import (
    PersistentPGDReconLoss,
    PersistentPGDReconSubsetLoss,
)

LAB_LOSS_METRIC_CLASSES: dict[str, type[Metric[Any]]] = {
    cls.__name__: cls
    for cls in (ChunkwiseSubsetReconLoss, PersistentPGDReconLoss, PersistentPGDReconSubsetLoss)
}

ALL_LOSS_METRIC_CLASSES: dict[str, type[Metric[Any]]] = {
    **LOSS_METRIC_CLASSES,
    **LAB_LOSS_METRIC_CLASSES,
}


def instantiate_lab_metrics(
    pd_config: PDConfig,
    component_model: ComponentModelProtocol,
    device: str,
) -> dict[str, Metric[Any]]:
    """Build + bind loss metrics from `pd_config.loss_metrics`, resolving lab classes.

    Returns the loss instances keyed by `type` literal (matches core
    `instantiate_metrics`' first return value).
    """
    loss_instances: dict[str, Metric[Any]] = {}
    for cfg in pd_config.loss_metrics:
        assert cfg.type not in loss_instances, f"duplicate loss metric {cfg.type!r}"
        m = ALL_LOSS_METRIC_CLASSES[cfg.type](cfg)
        m.bind(model=component_model, device=device)
        loss_instances[cfg.type] = m
    return loss_instances
