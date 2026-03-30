import functools
import hashlib
from typing import Any

from pydantic import Field, PositiveInt

from spd.base_config import BaseConfig
from spd.clustering.consts import ClusterCoactivationShaped, MergePair
from spd.clustering.math.merge_pair_samplers import (
    MERGE_PAIR_SAMPLERS,
    MergePairSampler,
    MergePairSamplerKey,
)


class MergeConfig(BaseConfig):
    alpha: float = Field(default=1.0)
    iters: PositiveInt | None = Field(
        default=100,
        description="Max merge iterations. If None, set to n_components - 1.",
    )
    merge_pair_sampling_method: MergePairSamplerKey = Field(default="range")
    merge_pair_sampling_kwargs: dict[str, Any] = Field(
        default_factory=lambda: {"threshold": 0.05},
    )

    @property
    def merge_pair_sample_func(self) -> MergePairSampler:
        return functools.partial(
            MERGE_PAIR_SAMPLERS[self.merge_pair_sampling_method],
            **self.merge_pair_sampling_kwargs,
        )

    def merge_pair_sample(self, costs: ClusterCoactivationShaped) -> MergePair:
        return self.merge_pair_sample_func(costs=costs)

    def get_num_iters(self, n_components: int) -> PositiveInt:
        if self.iters is None:
            return n_components - 1
        return self.iters

    @property
    def stable_hash(self) -> str:
        return hashlib.md5(self.model_dump_json().encode()).hexdigest()[:6]
