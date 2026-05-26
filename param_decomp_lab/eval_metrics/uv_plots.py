from typing import ClassVar, Literal, override

from param_decomp.base_config import BaseConfig
from param_decomp.masks import SamplingType
from param_decomp.metrics.base import Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp_lab.eval_metrics.plotting import plot_causal_importance_vals, plot_UV_matrices


class UVPlotsConfig(BaseConfig):
    """fnmatch patterns for layers permuted to align with the corresponding target solution.

    `identity_patterns` and `dense_patterns` are matched separately against the model.
    """

    type: Literal["UVPlots"] = "UVPlots"
    identity_patterns: list[str] | None
    dense_patterns: list[str] | None


class UVPlots(Metric[UVPlotsConfig]):
    """Component U/V matrix plots, with columns permuted to match CI alignment."""

    log_namespace = "figures"
    slow: ClassVar[bool] = True
    short_name = "UVPlots"

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
        all_perm_indices = plot_causal_importance_vals(
            model=self.model,
            batch_shape=self.batch_shape,
            input_magnitude=self.input_magnitude,
            identity_patterns=self.cfg.identity_patterns,
            dense_patterns=self.cfg.dense_patterns,
            sampling=self.sampling,
        )[1]
        uv_matrices = plot_UV_matrices(
            components=self.model.components, all_perm_indices=all_perm_indices
        )
        return {"uv_matrices": uv_matrices}
