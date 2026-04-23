from typing import Any, ClassVar, override

from PIL import Image
from torch import Tensor

from spd.configs import SamplingType
from spd.metrics.base import Metric
from spd.models.component_model import ComponentModel
from spd.plotting import plot_causal_importance_vals, plot_UV_matrices


class UVPlots(Metric):
    metric_section: ClassVar[str] = "figures"

    slow: ClassVar[bool] = True
    input_magnitude: ClassVar[float] = 0.75

    def __init__(
        self,
        model: ComponentModel,
        sampling: SamplingType,
        identity_patterns: list[str] | None = None,
        dense_patterns: list[str] | None = None,
    ) -> None:
        self.model = model
        self.sampling: SamplingType = sampling
        self.identity_patterns = identity_patterns
        self.dense_patterns = dense_patterns

        self.batch_shape: tuple[int, ...] | None = None

    @override
    def update(self, *, batch: Tensor | tuple[Tensor, ...], **_: Any) -> None:
        if self.batch_shape is None:
            input_tensor = batch[0] if isinstance(batch, tuple) else batch
            self.batch_shape = tuple(input_tensor.shape)

    @override
    def compute(self) -> dict[str, Image.Image]:
        assert self.batch_shape is not None, "haven't seen any inputs yet"

        all_perm_indices = plot_causal_importance_vals(
            model=self.model,
            batch_shape=self.batch_shape,
            input_magnitude=self.input_magnitude,
            identity_patterns=self.identity_patterns,
            dense_patterns=self.dense_patterns,
            sampling=self.sampling,
        )[1]

        uv_matrices = plot_UV_matrices(
            components=self.model.components, all_perm_indices=all_perm_indices
        )

        return {"uv_matrices": uv_matrices}
