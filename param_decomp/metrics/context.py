"""Per-step state passed to every metric's `update()`.

Built once per training step (after the DDP forward + CI calc) and once per eval batch. Replaces
the multi-kwarg `update(...)` signature each metric used to take.
"""

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from jaxtyping import Float
from torch import Tensor


class MetricRuntimeConfig(Protocol):
    steps: int
    use_delta_component: bool
    sampling: Literal["continuous", "binomial"]
    n_mask_samples: int


class MetricCIOutputs(Protocol):
    lower_leaky: dict[str, Float[Tensor, "... C"]]
    upper_leaky: dict[str, Float[Tensor, "... C"]]
    pre_sigmoid: dict[str, Tensor]


class MetricReconstructionLoss(Protocol):
    def __call__(self, pred: Tensor, target: Tensor) -> tuple[Float[Tensor, ""], int]: ...


@dataclass(frozen=True)
class MetricContext:
    model: Any
    config: MetricRuntimeConfig
    batch: Any
    target_out: Tensor
    pre_weight_acts: dict[str, Float[Tensor, "..."]]
    ci: MetricCIOutputs
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]]
    step: int
    reconstruction_loss: MetricReconstructionLoss
    is_eval: bool

    @property
    def current_frac_of_training(self) -> float:
        return self.step / self.config.steps if self.config.steps > 0 else 1.0
