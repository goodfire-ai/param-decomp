"""Dataset attribution configuration.

DatasetAttributionConfig: tuning params for the attribution pipeline.
AttributionsSlurmConfig: DatasetAttributionConfig + SLURM submission params.
"""

from typing import Literal

from pydantic import PositiveInt

from spd.base_config import BaseConfig
from spd.settings import DEFAULT_PARTITION_NAME


class DatasetAttributionConfig(BaseConfig):
    spd_run_wandb_path: str
    harvest_subrun_id: str | None = None
    n_batches: int | Literal["whole_dataset"] = 10_000
    batch_size: int = 32
    ci_threshold: float = 0.0


class AttributionsSlurmConfig(BaseConfig):
    """Config for dataset attributions SLURM submission."""

    config: DatasetAttributionConfig
    n_gpus: PositiveInt = 8
    partition: str = DEFAULT_PARTITION_NAME
    time: str = "48:00:00"
    merge_time: str = "01:00:00"
    merge_mem: str = "200G"
