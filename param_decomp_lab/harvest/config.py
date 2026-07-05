"""Harvest configuration.

HarvestConfig: tuning params for the harvest pipeline.
HarvestSlurmConfig: HarvestConfig + SLURM submission params.
"""

from typing import Any, Literal, override

from pydantic import PositiveInt

from param_decomp.base_config import BaseConfig
from param_decomp_lab.autointerp.config import LLMConfig, OpenRouterLLMConfig
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME
from param_decomp_lab.infra.wandb import parse_wandb_run_path

# -- Method-specific harvest configs ------------------------------------------


class ParamDecompHarvestConfig(BaseConfig):
    type: Literal["ParamDecompHarvestConfig"] = "ParamDecompHarvestConfig"
    wandb_path: str
    activation_threshold: float = 0.0

    @property
    def id(self) -> str:
        _, _, run_id = parse_wandb_run_path(self.wandb_path)
        return run_id

    @override
    def model_post_init(self, __context: Any) -> None:
        parse_wandb_run_path(self.wandb_path)


# -- Pipeline configs ----------------------------------------------------------


class IntruderEvalConfig(BaseConfig):
    """Config for intruder detection eval (decomposition quality, not label quality)."""

    llm: LLMConfig = OpenRouterLLMConfig(reasoning_effort="none")
    n_real: int = 4
    n_trials: int = 10
    density_tolerance: float = 0.05
    limit: int | None = None
    cost_limit_usd: float | None = None


class IntruderSlurmConfig(BaseConfig):
    """Config for intruder eval SLURM submission."""

    config: IntruderEvalConfig = IntruderEvalConfig()
    partition: str | None = DEFAULT_PARTITION_NAME
    time: str = "10:00:00"


class HarvestConfig(BaseConfig):
    method_config: ParamDecompHarvestConfig
    n_batches: int | Literal["whole_dataset"] = 20_000
    local_batch_size: int = 16
    """Sequences per forward on EACH worker; a sharded run's global batch is
    `local_batch_size * world_size`, so total sequences harvested =
    `n_batches * local_batch_size * world_size`."""
    activation_examples_per_component: int = 400
    activation_context_tokens_per_side: int = 20
    pmi_token_top_k: int = 40
    max_examples_per_batch_per_component: int = 5
    collect_component_cooccurrence: bool = True
    """Accumulate the dense component×component co-occurrence matrix (powers the app's
    component-correlation view). It is O(C²) in memory — at ~10⁵ components it needs tens
    of GB resident on the harvest GPU. Set false to skip it for large-C decompositions."""


class HarvestSlurmConfig(BaseConfig):
    """Config for harvest SLURM submission."""

    config: HarvestConfig
    n_gpus: PositiveInt = 8
    partition: str | None = DEFAULT_PARTITION_NAME
    time: str = "12:00:00"
    merge_time: str = "04:00:00"
    merge_mem: str = "200G"
