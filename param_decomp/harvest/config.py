"""Harvest configuration.

HarvestConfig: tuning params for the harvest pipeline.
"""

from typing import Any, Literal, override

from param_decomp.autointerp.config import LLMConfig, OpenRouterLLMConfig
from param_decomp.core.base_config import BaseConfig
from param_decomp.infra.wandb import parse_wandb_run_path

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


class HarvestConfig(BaseConfig):
    method_config: ParamDecompHarvestConfig
    n_batches: int | Literal["whole_dataset"] = 20_000
    batch_size: int = 32
    activation_examples_per_component: int = 400
    activation_context_tokens_per_side: int = 20
    pmi_token_top_k: int = 40
    max_examples_per_batch_per_component: int = 5
    collect_component_cooccurrence: bool = True
    """Accumulate the dense component×component co-occurrence matrix (powers the app's
    component-correlation view). It is O(C²) in memory — at ~10⁵ components it needs tens
    of GB resident on the harvest GPU. Set false to skip it for large-C decompositions."""
