"""ClusteringRunConfig — combines harvest + merge config with orchestration settings."""

from pydantic import Field, PositiveInt

from param_decomp.base_config import BaseConfig
from param_decomp_lab.clustering.harvest_config import (
    HarvestConfig,
)
from param_decomp_lab.clustering.merge_config import MergeConfig
from param_decomp_lab.infra.wandb import parse_wandb_run_path


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
        """W&B run-id slug used in the clustering run's wandb tags.

        Only valid when `harvest.model_path` is a W&B reference — raises
        `ValueError` for local checkpoint paths.
        """
        _, _, run_id = parse_wandb_run_path(str(self.harvest.model_path))
        return run_id
