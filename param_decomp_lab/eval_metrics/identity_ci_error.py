from typing import ClassVar, override

from param_decomp.base_config import BaseConfig
from param_decomp.metrics.base import Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp.routing import SamplingType
from param_decomp.utils.target_ci_solutions import compute_target_metrics, make_target_ci_solution
from param_decomp_lab.eval_metrics.plotting import get_single_feature_causal_importances


class IdentityCIErrorConfig(BaseConfig):
    identity_ci: list[dict[str, str | int]] | None
    dense_ci: list[dict[str, str | int]] | None


class IdentityCIError(Metric[IdentityCIErrorConfig]):
    """Error between the CI values and an Identity or Dense CI pattern."""

    section = "target_solution_error"
    config_type = IdentityCIErrorConfig
    slow = True
    short_name = "IdCIErr"

    input_magnitude: ClassVar[float] = 0.75

    @override
    def reset(self) -> None:
        self.batch_shape: tuple[int, ...] | None = None
        self.sampling: SamplingType | None = None

    @override
    def update(self, ctx: MetricContext) -> None:
        # `compute` ignores eval-batch contents and instead synthesizes a single-feature probe
        # from `batch_shape` + `sampling`, so only the first batch's metadata is needed.
        if self.batch_shape is None:
            input_tensor = ctx.batch[0] if isinstance(ctx.batch, tuple) else ctx.batch
            self.batch_shape = tuple(input_tensor.shape)
            self.sampling = ctx.config.sampling
        return None

    @override
    def compute(self) -> MetricResult:
        assert self.batch_shape is not None, "haven't seen any inputs yet"
        assert self.sampling is not None
        target_solution = make_target_ci_solution(
            identity_ci=self.cfg.identity_ci, dense_ci=self.cfg.dense_ci
        )
        if target_solution is None:
            return {}
        ci = get_single_feature_causal_importances(
            model=self.model,
            batch_shape=self.batch_shape,
            input_magnitude=self.input_magnitude,
            sampling=self.sampling,
        )
        return compute_target_metrics(
            causal_importances=ci.lower_leaky, target_solution=target_solution
        )
