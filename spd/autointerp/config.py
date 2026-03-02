"""Autointerp configuration.

CompactSkepticalConfig: interpretation strategy config.
AutointerpEvalConfig: eval job config (detection, fuzzing).
AutointerpSlurmConfig: CompactSkepticalConfig + eval + SLURM submission params.
"""

from typing import Annotated, Literal

from openrouter.components import Effort
from pydantic import Field

from spd.base_config import BaseConfig
from spd.settings import DEFAULT_PARTITION_NAME

FORBIDDEN_WORDS_DEFAULT = [
    "narrative",
    "story",
    "character",
    "theme",
    "descriptive",
    "content",
    "transition",
    "scene",
]


class CompactSkepticalConfig(BaseConfig):
    """Current default strategy: compact prompt, skeptical tone, structured JSON output."""

    type: Literal["compact_skeptical"] = "compact_skeptical"
    max_examples: int = 30
    include_pmi: bool = True
    include_spd_context: bool = True
    include_dataset_description: bool = True
    label_max_words: int = 8
    forbidden_words: list[str] | None = None


class DualViewConfig(BaseConfig):
    """Dual-view strategy: presents both input and output evidence with dual example views.

    Key differences from compact_skeptical:
    - Output data presented first
    - Two example sections: "fires on" (current token) and "produces" (next token)
    - Task asks for functional description, not detection label
    """

    type: Literal["dual_view"] = "dual_view"
    max_examples: int = 30
    include_pmi: bool = True
    include_dataset_description: bool = True
    label_max_words: int = 8
    forbidden_words: list[str] | None = None


StrategyConfig = CompactSkepticalConfig | DualViewConfig


class AutointerpConfig(BaseConfig):
    model: str = "google/gemini-3-flash-preview"
    reasoning_effort: Effort = "low"
    limit: int | None = None
    cost_limit_usd: float | None = None
    max_requests_per_minute: int = 500
    max_concurrent: int = 50
    template_strategy: Annotated[StrategyConfig, Field(discriminator="type")]


class DetectionEvalConfig(BaseConfig):
    type: Literal["detection"] = "detection"
    n_activating: int = 5
    n_non_activating: int = 5
    n_trials: int = 5


class FuzzingEvalConfig(BaseConfig):
    type: Literal["fuzzing"] = "fuzzing"
    n_correct: int = 5
    n_incorrect: int = 2
    n_trials: int = 5


class AutointerpEvalConfig(BaseConfig):
    """Config for label-based autointerp evals (detection, fuzzing)."""

    model: str = "google/gemini-3-flash-preview"
    reasoning_effort: Effort = "none"
    detection_config: DetectionEvalConfig
    fuzzing_config: FuzzingEvalConfig
    limit: int | None = None
    cost_limit_usd: float | None = None
    max_requests_per_minute: int = 500
    max_concurrent: int = 50


class AutointerpSlurmConfig(BaseConfig):
    """Config for the autointerp functional unit (interpret + evals).

    Dependency graph within autointerp:
        interpret         (depends on harvest merge)
        ├── detection     (depends on interpret)
        └── fuzzing       (depends on interpret)
    """

    config: AutointerpConfig
    partition: str = DEFAULT_PARTITION_NAME
    time: str = "12:00:00"
    evals: AutointerpEvalConfig | None
    evals_time: str = "12:00:00"
