from typing import Any, ClassVar, override

from PIL import Image
from torch import Tensor

from param_decomp.configs import SamplingType
from param_decomp.metrics.base import Metric
from param_decomp.models.component_model import ComponentModel
from param_decomp.plotting import (
    _plot_causal_importances_figure,
    get_single_feature_causal_importances,
)


class OneHotCIPlots(Metric):
    """Plot per-layer causal importance for one-hot inputs of fixed magnitude.

    For each layer, produces a (D, C) heatmap showing the causal importance of each
    component (columns) when the input is a one-hot vector at each input position (rows).
    Unlike `PermutedCIPlots`, components are not permuted: this exposes whether each
    component naturally responds to a single unique input dimension.
    """

    slow: ClassVar[bool] = True
    metric_section: ClassVar[str] = "figures"

    def __init__(
        self,
        model: ComponentModel,
        sampling: SamplingType,
        input_magnitude: float = 0.5,
    ) -> None:
        self.model = model
        self.sampling: SamplingType = sampling
        self.input_magnitude = input_magnitude
        self.batch_shape: tuple[int, ...] | None = None

    @override
    def update(self, *, batch: Tensor | tuple[Tensor, ...], **_: Any) -> None:
        if self.batch_shape is None:
            input_tensor = batch[0] if isinstance(batch, tuple) else batch
            self.batch_shape = tuple(input_tensor.shape)

    @override
    def compute(self) -> dict[str, Image.Image]:
        assert self.batch_shape is not None, "haven't seen any inputs yet"

        ci_output = get_single_feature_causal_importances(
            model=self.model,
            batch_shape=self.batch_shape,
            input_magnitude=self.input_magnitude,
            sampling=self.sampling,
        )
        has_pos_dim = len(self.batch_shape) == 3

        ci_lower = _plot_causal_importances_figure(
            ci_vals=ci_output.lower_leaky,
            title_prefix="one-hot importance values lower leaky relu",
            colormap="Blues",
            input_magnitude=self.input_magnitude,
            has_pos_dim=has_pos_dim,
        )
        ci_upper = _plot_causal_importances_figure(
            ci_vals=ci_output.upper_leaky,
            title_prefix="one-hot importance values",
            colormap="Reds",
            input_magnitude=self.input_magnitude,
            has_pos_dim=has_pos_dim,
        )
        return {
            "one_hot_causal_importances": ci_lower,
            "one_hot_causal_importances_upper_leaky": ci_upper,
        }
