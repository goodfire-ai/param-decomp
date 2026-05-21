"""Configuration for harvesting component activations into membership snapshots."""

from pydantic import PositiveInt, field_validator

from param_decomp.base_config import BaseConfig
from param_decomp.types import Probability
from param_decomp_lab.clustering.util import (
    DeadComponentFilterStat,
    ModuleFilterFunc,
    ModuleFilterSource,
)


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
    model_path: str
    batch_size: PositiveInt
    n_samples: PositiveInt | None = None
    n_tokens: PositiveInt | None = None
    n_tokens_per_seq: PositiveInt | None = None
    use_all_tokens_per_seq: bool = False
    dataset_seed: int = 0
    activation_threshold: Probability
    filter_dead_threshold: float = 0.001
    filter_dead_stat: DeadComponentFilterStat = "max"
    module_name_filter: ModuleFilterSource = None

    @field_validator("model_path")
    def validate_model_path(cls, v: str) -> str:
        assert v.startswith("wandb:"), f"model_path must start with 'wandb:', got: {v}"
        return v

    @property
    def filter_modules(self) -> ModuleFilterFunc:
        return _to_module_filter(self.module_name_filter)
