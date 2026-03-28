"""Autointerp configuration."""

from typing import Annotated, Literal

from pydantic import Field

from spd.autointerp.providers import LLMConfig, OpenRouterLLMConfig
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


class LegacyDelimitedExamplesConfig(BaseConfig):
    format: Literal["legacy_delimited"] = "legacy_delimited"


class SingleLineExamplesConfig(BaseConfig):
    format: Literal["single_line"] = "single_line"
    annotation_style: Literal["none", "activation"] = "none"
    highlight_delimiter: Literal["brackets", "angle"] = "brackets"


class XmlExamplesConfig(BaseConfig):
    format: Literal["xml"] = "xml"
    annotation_style: Literal["none", "activation"] = "none"
    highlight_delimiter: Literal["brackets", "angle"] = "brackets"
    sanitize_raw: bool = False
    sanitize_highlighted: bool = False


ExampleRenderingConfig = Annotated[
    LegacyDelimitedExamplesConfig | SingleLineExamplesConfig | XmlExamplesConfig,
    Field(discriminator="format"),
]
RichExampleRenderingConfig = Annotated[
    SingleLineExamplesConfig | XmlExamplesConfig,
    Field(discriminator="format"),
]


def default_example_rendering() -> ExampleRenderingConfig:
    return LegacyDelimitedExamplesConfig()


def default_rich_example_rendering() -> RichExampleRenderingConfig:
    return SingleLineExamplesConfig(annotation_style="activation")


CANON_RENDERING = XmlExamplesConfig(
    format="xml",
    annotation_style="activation",
    highlight_delimiter="brackets",
    sanitize_raw=False,
    sanitize_highlighted=False,
)


class CompactSkepticalConfig(BaseConfig):
    """Current default strategy: compact prompt, skeptical tone, structured JSON output."""

    type: Literal["compact_skeptical"] = "compact_skeptical"
    max_examples: int = 30
    include_pmi: bool = True
    include_dataset_description: bool = True
    label_max_words: int = 8
    forbidden_words: list[str] | None = None
    example_rendering: ExampleRenderingConfig = Field(default_factory=default_example_rendering)


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
    example_rendering: ExampleRenderingConfig = Field(default_factory=default_example_rendering)


class RichExamplesConfig(BaseConfig):
    """Rich examples strategy: drops token statistics, shows per-token CI and activation values.

    Supports both compact one-line rendering and an XML dual-block rendering
    with separate raw and highlighted views.
    """

    type: Literal["rich_examples"] = "rich_examples"
    max_examples: int = 30
    include_dataset_description: bool = True
    label_max_words: int = 8
    output_pmi_min_count: float = 2.0
    example_rendering: RichExampleRenderingConfig = Field(
        default_factory=default_rich_example_rendering
    )


class CanonConfig(BaseConfig):
    """Canon strategy: detailed SPD explanation, sign convention, CI-vs-act guidance,
    output PMI, and XML dual-view examples.

    Uses a fixed XML rendering (brackets, activation annotations, no sanitization).
    """

    type: Literal["canon"] = "canon"
    max_examples: int = 30
    label_max_words: int = 8


StrategyConfig = CompactSkepticalConfig | DualViewConfig | RichExamplesConfig | CanonConfig


class AutointerpConfig(BaseConfig):
    llm: LLMConfig = OpenRouterLLMConfig()
    limit: int | None = None
    component_keys_path: str | None = None
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

    llm: LLMConfig = OpenRouterLLMConfig(reasoning_effort="none")
    detection_config: DetectionEvalConfig
    fuzzing_config: FuzzingEvalConfig
    limit: int | None = None
    component_keys_path: str | None = None
    seed: int = 0
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
