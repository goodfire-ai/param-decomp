"""Type-literal to Metric-class dispatch table for loss-capable metrics.

`PDConfig.loss_metrics` is a discriminated union keyed by each config's `type` literal.
This table maps the same literal to the matching `Metric` subclass so that `optimize()`
can instantiate the right metric for each configured entry.
"""

from typing import Any

from param_decomp.metrics.base import Metric
from param_decomp.metrics.ci_masked_recon_layerwise_loss import CIMaskedReconLayerwiseLoss
from param_decomp.metrics.ci_masked_recon_loss import CIMaskedReconLoss
from param_decomp.metrics.ci_masked_recon_subset_loss import CIMaskedReconSubsetLoss
from param_decomp.metrics.faithfulness_loss import FaithfulnessLoss
from param_decomp.metrics.hidden_acts_recon_loss import StochasticHiddenActsReconLoss
from param_decomp.metrics.importance_minimality_loss import ImportanceMinimalityLoss
from param_decomp.metrics.persistent_pgd_recon import (
    PersistentPGDReconLoss,
    PersistentPGDReconSubsetLoss,
)
from param_decomp.metrics.pgd_masked_recon_layerwise_loss import PGDReconLayerwiseLoss
from param_decomp.metrics.pgd_masked_recon_loss import PGDReconLoss
from param_decomp.metrics.pgd_masked_recon_subset_loss import PGDReconSubsetLoss
from param_decomp.metrics.stochastic_recon_layerwise_loss import StochasticReconLayerwiseLoss
from param_decomp.metrics.stochastic_recon_loss import StochasticReconLoss
from param_decomp.metrics.stochastic_recon_subset_loss import StochasticReconSubsetLoss
from param_decomp.metrics.unmasked_recon_loss import UnmaskedReconLoss

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
