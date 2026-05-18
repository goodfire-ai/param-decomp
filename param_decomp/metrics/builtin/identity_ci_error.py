from typing import ClassVar

from param_decomp.configs import SamplingType
from param_decomp.metrics.base import MetricConfig
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.registry import register_metric
from param_decomp.models.component_model import ComponentModel
from param_decomp.plotting import get_single_feature_causal_importances
from param_decomp.utils.target_ci_solutions import compute_target_metrics, make_target_ci_solution


class IdentityCIErrorConfig(MetricConfig):
    identity_ci: list[dict[str, str | int]] | None
    dense_ci: list[dict[str, str | int]] | None


@register_metric
class IdentityCIError:
    """Error between the CI values and an Identity or Dense CI pattern."""

    name = "identity_ci_error"
    section = "target_solution_error"
    config_type = IdentityCIErrorConfig
    slow = True
    short_name = "IdCIErr"

    input_magnitude: ClassVar[float] = 0.75

    def __init__(self, cfg: IdentityCIErrorConfig, *, model: ComponentModel, device: str) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.batch_shape: tuple[int, ...] | None = None
        self.sampling: SamplingType | None = None

    def update(self, ctx: MetricContext) -> None:
        if self.batch_shape is None:
            input_tensor = ctx.batch[0] if isinstance(ctx.batch, tuple) else ctx.batch
            self.batch_shape = tuple(input_tensor.shape)
            self.sampling = ctx.config.sampling
        return None

    def compute(self) -> dict[str, float]:
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
