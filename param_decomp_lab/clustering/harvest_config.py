"""Configuration for harvesting component activations into membership snapshots."""

from pydantic import PositiveInt

from param_decomp.base_config import BaseConfig, Probability
from param_decomp_lab.clustering.formatting import (
    DeadComponentFilterStat,
    ModuleFilterFunc,
    ModuleFilterSource,
)
from param_decomp_lab.infra.paths import ModelPath


def _to_module_filter(source: ModuleFilterSource) -> ModuleFilterFunc:
    if source is None:
        return lambda _: True
    if isinstance(source, str):
        return lambda name: name.startswith(source)
    if isinstance(source, set):
        return lambda name: name in source
    assert callable(source)
    return source


class HarvestConfig(BaseConfig):
    """Settings for an LM-run clustering harvest.

    Clustering currently only consumes LM PD runs — `model_path` must point at one
    (a local checkpoint directory or a W&B run reference).
    """

    model_path: ModelPath
    batch_size: PositiveInt
    n_tokens: PositiveInt
    n_tokens_per_seq: PositiveInt | None = None
    use_all_tokens_per_seq: bool = False
    dataset_seed: int = 0
    activation_threshold: Probability
    filter_dead_threshold: float = 0.001
    filter_dead_stat: DeadComponentFilterStat = "max"
    module_name_filter: ModuleFilterSource = None

    @property
    def filter_modules(self) -> ModuleFilterFunc:
        return _to_module_filter(self.module_name_filter)
