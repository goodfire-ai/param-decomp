"""Dataset attribution configuration.

DatasetAttributionConfig: tuning params for the attribution pipeline.
AttributionsSlurmConfig: DatasetAttributionConfig + SLURM submission params.
"""

from typing import Any, Literal

from pydantic import PositiveInt, model_validator

from param_decomp.base_config import BaseConfig
from param_decomp.settings import DEFAULT_PARTITION_NAME


class DatasetAttributionConfig(BaseConfig):
    wandb_path: str
    n_batches: int | Literal["whole_dataset"] = 10_000
    batch_size: int = 32
    ci_threshold: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_keys(cls, data: dict[str, Any]) -> dict[str, Any]:
        if "spd_run_wandb_path" in data:
            data["wandb_path"] = data.pop("spd_run_wandb_path")
        return data


class AttributionsSlurmConfig(BaseConfig):
    """Config for dataset attributions SLURM submission."""

    config: DatasetAttributionConfig
    n_gpus: PositiveInt = 8
    partition: str = DEFAULT_PARTITION_NAME
    time: str = "48:00:00"
    merge_time: str = "01:00:00"
    merge_mem: str = "200G"
