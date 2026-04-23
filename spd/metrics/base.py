"""Metric interface for distributed metric computation.

All metrics implement Metric and typically handle distributed synchronization directly in their
compute() methods.
"""

from typing import Any, ClassVar, Protocol

from jaxtyping import Float
from torch import Tensor

from spd.models.component_model import CIOutputs


class Metric(Protocol):
    """Interface for metrics that can be used in training and/or evaluation."""

    slow: ClassVar[bool] = False
    metric_section: ClassVar[str]

    def update(
        self,
        *,
        batch: Any,
        target_out: Tensor,
        pre_weight_acts: dict[str, Float[Tensor, "..."]],
        ci: CIOutputs,
        current_frac_of_training: float,
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
    ) -> None:
        """Update metric state with a batch of data."""
        ...

    def compute(self) -> Any:
        """Compute the final metric value(s)."""
        ...
