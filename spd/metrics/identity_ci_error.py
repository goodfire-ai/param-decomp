from typing import Any, ClassVar, override

from torch import Tensor

from spd.configs import SamplingType
from spd.metrics.base import Metric
from spd.models.component_model import ComponentModel
from spd.plotting import get_single_feature_causal_importances
from spd.utils.target_ci_solutions import compute_target_metrics, make_target_ci_solution


class IdentityCIError(Metric):
    """Error between the CI values and an Identity or Dense CI pattern."""

    slow: ClassVar[bool] = True
    input_magnitude: ClassVar[float] = 0.75

    metric_section: ClassVar[str] = "target_solution_error"

    def __init__(
        self,
        model: ComponentModel,
        sampling: SamplingType,
        identity_ci: list[dict[str, str | int]] | None = None,
        dense_ci: list[dict[str, str | int]] | None = None,
    ) -> None:
        self.model = model
        self.sampling: SamplingType = sampling
        self.identity_ci = identity_ci
        self.dense_ci = dense_ci

        self.batch_shape: tuple[int, ...] | None = None

    @override
    def update(self, *, batch: Tensor | tuple[Tensor, ...], **_: Any) -> None:
        if self.batch_shape is None:
            input_tensor = batch[0] if isinstance(batch, tuple) else batch
            self.batch_shape = tuple(input_tensor.shape)

    @override
    def compute(self) -> dict[str, float]:
        assert self.batch_shape is not None, "haven't seen any inputs yet"

        target_solution = make_target_ci_solution(
            identity_ci=self.identity_ci, dense_ci=self.dense_ci
        )
        if target_solution is None:
            return {}

        ci = get_single_feature_causal_importances(
            model=self.model,
            batch_shape=self.batch_shape,
            input_magnitude=self.input_magnitude,
            sampling=self.sampling,
        )

        target_metrics = compute_target_metrics(
            causal_importances=ci.lower_leaky, target_solution=target_solution
        )
        return target_metrics
