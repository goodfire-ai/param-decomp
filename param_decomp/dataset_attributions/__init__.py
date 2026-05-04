"""Dataset attributions module.

Computes component-to-component attribution strengths aggregated over the
training dataset.
"""

from param_decomp.dataset_attributions.config import DatasetAttributionConfig
from param_decomp.dataset_attributions.harvest import harvest_attributions
from param_decomp.dataset_attributions.repo import AttributionRepo
from param_decomp.dataset_attributions.storage import (
    DatasetAttributionEntry,
    DatasetAttributionStorage,
)

__all__ = [
    "AttributionRepo",
    "DatasetAttributionConfig",
    "DatasetAttributionEntry",
    "DatasetAttributionStorage",
    "harvest_attributions",
]
