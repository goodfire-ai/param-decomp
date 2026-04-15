"""Configuration for harvesting component activations into membership snapshots."""

from typing import Any

from pydantic import PositiveInt, field_validator, model_validator

from spd.base_config import BaseConfig
from spd.clustering.util import DeadComponentFilterStat, ModuleFilterFunc, ModuleFilterSource
from spd.registry import EXPERIMENT_REGISTRY
from spd.spd_types import Probability


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

    @model_validator(mode="before")
    def process_experiment_key(cls, values: dict[str, Any]) -> dict[str, Any]:
        experiment_key: str | None = values.get("experiment_key")
        if experiment_key:
            model_path = EXPERIMENT_REGISTRY[experiment_key].canonical_run
            assert model_path is not None
            values["model_path"] = model_path
            del values["experiment_key"]
        return values

    @field_validator("model_path")
    def validate_model_path(cls, v: str) -> str:
        assert v.startswith("wandb:"), f"model_path must start with 'wandb:', got: {v}"
        return v

    @property
    def filter_modules(self) -> ModuleFilterFunc:
        return _to_module_filter(self.module_name_filter)
