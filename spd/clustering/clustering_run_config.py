"""ClusteringRunConfig — combines harvest + merge config with orchestration settings."""

from pydantic import Field, PositiveInt

from spd.base_config import BaseConfig
from spd.clustering.harvest_config import HarvestConfig
from spd.clustering.merge_config import MergeConfig


class LoggingIntervals(BaseConfig):
    stat: PositiveInt = 1
    tensor: PositiveInt = 100
    plot: PositiveInt = 100
    artifact: PositiveInt = 100


class ClusteringRunConfig(BaseConfig):
    harvest: HarvestConfig
    merge: MergeConfig = Field(default_factory=MergeConfig)
    ensemble_id: str | None = None
    logging_intervals: LoggingIntervals = Field(default_factory=LoggingIntervals)
    wandb_project: str | None = None
    wandb_entity: str = "goodfire"

    @property
    def wandb_decomp_model(self) -> str:
        parts = self.harvest.model_path.replace("wandb:", "").split("/")
        assert len(parts) >= 3, f"Invalid wandb path format: {self.harvest.model_path}"
        return parts[-1] if parts[-1] != "runs" else parts[-2]
