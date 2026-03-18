"""Graph interpretation configuration."""

from spd.autointerp.providers import LLMConfig, OpenRouterLLMConfig
from spd.base_config import BaseConfig
from spd.dataset_attributions.storage import AttrMetric
from spd.settings import DEFAULT_PARTITION_NAME


class GraphInterpConfig(BaseConfig):
    llm: LLMConfig = OpenRouterLLMConfig()
    attr_metric: AttrMetric = "attr_abs"
    top_k_attributed: int = 8
    max_examples: int = 20
    label_max_words: int = 8
    cost_limit_usd: float | None = None
    max_requests_per_minute: int = 500
    max_concurrent: int = 50
    limit: int | None = None


class GraphInterpSlurmConfig(BaseConfig):
    config: GraphInterpConfig
    partition: str = DEFAULT_PARTITION_NAME
    time: str = "24:00:00"
