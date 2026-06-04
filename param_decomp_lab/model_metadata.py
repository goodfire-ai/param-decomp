"""Build autointerp `ModelMetadata` from a transformer target model + run config.

Single source of truth shared by `PDAdapter.model_metadata` and the in-training
`AutointerpLabels` metric, so the two can't drift.
"""

import torch.nn as nn

from param_decomp_lab.autointerp.schemas import ModelMetadata
from param_decomp_lab.topology import TransformerTopology


def build_model_metadata(
    target_model: nn.Module,
    target_module_paths: list[str],
    *,
    model_class: str,
    dataset_name: str,
    seq_len: int,
) -> ModelMetadata:
    topology = TransformerTopology(target_model)
    return ModelMetadata(
        n_blocks=topology.n_blocks,
        model_class=model_class,
        dataset_name=dataset_name,
        layer_descriptions={path: topology.target_to_canon(path) for path in target_module_paths},
        seq_len=seq_len,
        decomposition_method="pd",
    )
