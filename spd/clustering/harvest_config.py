"""Configuration for harvesting component activations into membership snapshots."""

from typing import Any

from pydantic import Field, PositiveInt, field_validator, model_validator

from spd.base_config import BaseConfig
from spd.clustering.merge_config import _to_module_filter
from spd.clustering.util import DeadComponentFilterStat, ModuleFilterFunc, ModuleFilterSource
from spd.registry import EXPERIMENT_REGISTRY
from spd.spd_types import Probability


class HarvestConfig(BaseConfig):
    model_path: str
    batch_size: PositiveInt
    n_samples: PositiveInt | None = Field(
        default=None,
        description="Number of activation samples (non-LM tasks). Defaults to batch_size.",
    )
    n_tokens: PositiveInt | None = Field(
        default=None, description="Number of token samples to collect (LM only)"
    )
    n_tokens_per_seq: PositiveInt | None = Field(
        default=None, description="Random token positions per sequence (LM only)"
    )
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
