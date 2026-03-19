import functools
import hashlib
from typing import Any, Literal

from pydantic import (
    Field,
    PositiveInt,
)

from spd.base_config import BaseConfig
from spd.clustering.consts import ClusterCoactivationShaped, MergePair
from spd.clustering.math.merge_pair_samplers import (
    MERGE_PAIR_SAMPLERS,
    MergePairSampler,
    MergePairSamplerKey,
)
from spd.clustering.util import DeadComponentFilterStat, ModuleFilterFunc, ModuleFilterSource
from spd.spd_types import Probability

MergeConfigKey = Literal[
    "activation_threshold",
    "alpha",
    "iters",
    "merge_pair_sampling_method",
    "merge_pair_sampling_kwargs",
    "filter_dead_threshold",
    "filter_dead_stat",
]


def _to_module_filter(
    filter_modules: ModuleFilterSource,
) -> ModuleFilterFunc:
    """Convert the filter_modules argument to a callable."""
    if filter_modules is None:
        return lambda _: True
    elif isinstance(filter_modules, str):
        return lambda module_name: module_name.startswith(filter_modules)
    elif isinstance(filter_modules, set):
        return lambda module_name: module_name in filter_modules
    elif callable(filter_modules):
        return filter_modules
    else:
        raise TypeError(f"filter_modules must be str, set, or callable, got {type(filter_modules)}")  # pyright: ignore[reportUnreachable]


class MergeConfig(BaseConfig):
    activation_threshold: Probability | None = Field(
        default=0.01,
        description="Threshold for considering a component active in a group. If None, use raw scalar causal importances",
    )
    alpha: float = Field(
        default=1.0,
        description="rank weight factor. Higher values mean a higher penalty on 'sending' the component weights",
    )
    iters: PositiveInt | None = Field(
        default=100,
        description="max number of iterations to run the merge algorithm for. If `None`, set to number of components (after filtering) minus one.",
    )
    merge_pair_sampling_method: MergePairSamplerKey = Field(
        default="range",
        description="Method for sampling merge pairs. Options: 'range', 'mcmc'.",
    )
    merge_pair_sampling_kwargs: dict[str, Any] = Field(
        default_factory=lambda: {"threshold": 0.05},
        description="Keyword arguments for the merge pair sampling method.",
    )
    filter_dead_threshold: float = Field(
        default=0.001,
        description="Threshold for filtering out dead components using the statistic selected by filter_dead_stat.",
    )
    filter_dead_stat: DeadComponentFilterStat = Field(
        default="max",
        description="Statistic used to determine whether a component is dead before clustering.",
    )
    module_name_filter: ModuleFilterSource = Field(
        default=None,
        description="Filter for module names. Can be a string prefix, a set of names, or a callable that returns True for modules to include.",
    )

    @property
    def merge_pair_sample_func(self) -> MergePairSampler:
        return functools.partial(
            MERGE_PAIR_SAMPLERS[self.merge_pair_sampling_method],
            **self.merge_pair_sampling_kwargs,
        )

    def merge_pair_sample(
        self,
        costs: ClusterCoactivationShaped,
    ) -> MergePair:
        """do merge sampling based on the configured method and kwargs

        has signature `MergePairSampler = Callable[[ClusterCoactivationShaped], MergePair]`
        """
        return self.merge_pair_sample_func(costs=costs)

    @property
    def filter_modules(self) -> ModuleFilterFunc:
        """Get the module filter function based on the provided source."""
        return _to_module_filter(self.module_name_filter)

    def get_num_iters(self, n_components: int) -> PositiveInt:
        """Get the number of iterations to run the merge algorithm for.

        Args:
            n_components: Number of components (after filtering)

        Returns:
            Number of iterations to run
        """
        if self.iters is None:
            return n_components - 1
        else:
            return self.iters

    @property
    def stable_hash(self) -> str:
        return hashlib.md5(self.model_dump_json().encode()).hexdigest()[:6]
