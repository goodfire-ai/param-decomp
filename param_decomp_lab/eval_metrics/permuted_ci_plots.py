from typing import ClassVar, override

from param_decomp.base_config import BaseConfig
from param_decomp.masks import SamplingType
from param_decomp.metrics.base import Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp_lab.eval_metrics.plotting import plot_causal_importance_vals


class PermutedCIPlotsConfig(BaseConfig):
    identity_patterns: list[str] | None
    dense_patterns: list[str] | None


class PermutedCIPlots(Metric[PermutedCIPlotsConfig]):
    section = "figures"
    config_type = PermutedCIPlotsConfig
    slow = True
    short_name = "PermCIPlots"

    input_magnitude: ClassVar[float] = 0.75

    @override
    def reset(self) -> None:
        self.batch_shape: tuple[int, ...] | None = None
        self.sampling: SamplingType | None = None

    @override
    def update(self, ctx: MetricContext) -> None:
        if self.batch_shape is None:
            input_tensor = ctx.batch[0] if isinstance(ctx.batch, tuple) else ctx.batch
            self.batch_shape = tuple(input_tensor.shape)
            self.sampling = ctx.sampling
        return None

    @override
    def compute(self) -> MetricResult:
        assert self.batch_shape is not None, "haven't seen any inputs yet"
        assert self.sampling is not None
        figures = plot_causal_importance_vals(
            model=self.model,
            batch_shape=self.batch_shape,
            input_magnitude=self.input_magnitude,
            identity_patterns=self.cfg.identity_patterns,
            dense_patterns=self.cfg.dense_patterns,
            sampling=self.sampling,
        )[0]
        return dict(figures)
