"""Harvest configuration.

HarvestConfig: tuning params for the harvest pipeline.
HarvestSlurmConfig: HarvestConfig + SLURM submission params.
"""

from typing import Annotated, Any, Literal, override

from pydantic import Field, PositiveInt

from spd.autointerp.providers import LLMConfig, OpenRouterLLMConfig
from spd.base_config import BaseConfig
from spd.settings import DEFAULT_PARTITION_NAME
from spd.utils.wandb_utils import parse_wandb_run_path

# -- Method-specific harvest configs ------------------------------------------


class SPDHarvestConfig(BaseConfig):
    type: Literal["SPDHarvestConfig"] = "SPDHarvestConfig"
    wandb_path: str
    activation_threshold: float = 0.0

    @property
    def id(self) -> str:
        _, _, run_id = parse_wandb_run_path(self.wandb_path)
        return run_id

    @override
    def model_post_init(self, __context: Any) -> None:
        parse_wandb_run_path(self.wandb_path)


class CLTHarvestConfig(BaseConfig):
    type: Literal["CLTHarvestConfig"] = "CLTHarvestConfig"

    wandb_path: str

    @property
    def id(self) -> str:
        return "clt"


class MOLTHarvestConfig(BaseConfig):
    type: Literal["MOLTHarvestConfig"] = "MOLTHarvestConfig"

    wandb_path: str

    @property
    def id(self) -> str:
        return "molt"


DecompositionMethodHarvestConfig = SPDHarvestConfig | CLTHarvestConfig | MOLTHarvestConfig


# -- Pipeline configs ----------------------------------------------------------


class IntruderEvalConfig(BaseConfig):
    """Config for intruder detection eval (decomposition quality, not label quality)."""

    llm: LLMConfig = OpenRouterLLMConfig(reasoning_effort="none")
    n_real: int = 4
    n_trials: int = 10
    density_tolerance: float = 0.05
    max_concurrent: int = 50
    limit: int | None = None
    cost_limit_usd: float | None = None
    max_requests_per_minute: int = 500


class IntruderSlurmConfig(BaseConfig):
    """Config for intruder eval SLURM submission."""

    config: IntruderEvalConfig = IntruderEvalConfig()
    partition: str = DEFAULT_PARTITION_NAME
    time: str = "4:00:00"


class HarvestConfig(BaseConfig):
    method_config: Annotated[DecompositionMethodHarvestConfig, Field(discriminator="type")]
    n_batches: int | Literal["whole_dataset"] = 20_000
    batch_size: int = 32
    activation_examples_per_component: int = 400
    activation_context_tokens_per_side: int = 20
    pmi_token_top_k: int = 40
    max_examples_per_batch_per_component: int = 5


class HarvestSlurmConfig(BaseConfig):
    """Config for harvest SLURM submission."""

    config: HarvestConfig
    n_gpus: PositiveInt = 8
    partition: str = DEFAULT_PARTITION_NAME
    time: str = "12:00:00"
    merge_time: str = "04:00:00"
    merge_mem: str = "200G"
