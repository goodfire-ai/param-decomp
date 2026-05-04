"""Harvest configuration.

HarvestConfig: tuning params for the harvest pipeline.
HarvestSlurmConfig: HarvestConfig + SLURM submission params.
"""

from typing import Annotated, Any, Literal, override

from pydantic import Field, PositiveInt, model_validator

from param_decomp.autointerp.providers import LLMConfig, OpenRouterLLMConfig
from param_decomp.base_config import BaseConfig
from param_decomp.settings import DEFAULT_PARTITION_NAME
from param_decomp.utils.wandb_utils import parse_wandb_run_path

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


class CLTHarvestConfig(BaseConfig):
    type: Literal["CLTHarvestConfig"] = "CLTHarvestConfig"
    base_model_path: str
    artifact_path: str
    """Wandb artifact path for the CLT checkpoint (single artifact covering all layers)."""

    @property
    def id(self) -> str:
        import hashlib

        return "clt-" + hashlib.sha256(self.artifact_path.encode()).hexdigest()[:8]


class TranscoderHarvestConfig(BaseConfig):
    type: Literal["TranscoderHarvestConfig"] = "TranscoderHarvestConfig"
    base_model_path: str
    artifact_paths: dict[str, str]
    """Maps module paths (e.g. "h.0.mlp") to wandb artifact paths."""

    @property
    def id(self) -> str:
        import hashlib

        key = str(sorted(self.artifact_paths.items()))
        return "tc-" + hashlib.sha256(key.encode()).hexdigest()[:8]


DecompositionMethodHarvestConfig = (
    ParamDecompHarvestConfig | CLTHarvestConfig | TranscoderHarvestConfig
)


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
    partition: str = DEFAULT_PARTITION_NAME
    time: str = "10:00:00"


class HarvestConfig(BaseConfig):
    method_config: Annotated[DecompositionMethodHarvestConfig, Field(discriminator="type")]
    n_batches: int | Literal["whole_dataset"] = 20_000
    batch_size: int = 32
    activation_examples_per_component: int = 400
    activation_context_tokens_per_side: int = 20
    pmi_token_top_k: int = 40
    max_examples_per_batch_per_component: int = 5

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_keys(cls, data: dict[str, Any]) -> dict[str, Any]:
        method = data.get("method_config")
        if isinstance(method, dict) and method.get("type") == "SPDHarvestConfig":
            method["type"] = "ParamDecompHarvestConfig"
        return data


class HarvestSlurmConfig(BaseConfig):
    """Config for harvest SLURM submission."""

    config: HarvestConfig
    n_gpus: PositiveInt = 8
    partition: str = DEFAULT_PARTITION_NAME
    time: str = "12:00:00"
    merge_time: str = "04:00:00"
    merge_mem: str = "200G"
