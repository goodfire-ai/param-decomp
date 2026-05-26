"""Dataset attribution configuration.

DatasetAttributionConfig: tuning params for the attribution pipeline.
AttributionsSlurmConfig: DatasetAttributionConfig + SLURM submission params.
"""

from typing import Literal

from pydantic import PositiveInt

from param_decomp.base_config import BaseConfig
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME


class DatasetAttributionConfig(BaseConfig):
    wandb_path: str
    n_batches: int | Literal["whole_dataset"] = 10_000
    batch_size: int = 32
    ci_threshold: float = 0.0


class AttributionsSlurmConfig(BaseConfig):
    """Config for dataset attributions SLURM submission."""

    config: DatasetAttributionConfig
    n_gpus: PositiveInt = 8
    partition: str | None = DEFAULT_PARTITION_NAME
    time: str = "48:00:00"
    merge_time: str = "01:00:00"
    merge_mem: str = "200G"
