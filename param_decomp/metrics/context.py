"""Per-step state passed to every metric's `update()`.

Built once per training step (after the DDP forward + CI calc) and once per eval batch. Replaces
the multi-kwarg `update(...)` signature each metric used to take.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jaxtyping import Float
from torch import Tensor

if TYPE_CHECKING:
    from param_decomp.configs import PDConfig
    from param_decomp.models.batch_and_loss_fns import ReconstructionLoss
    from param_decomp.models.component_model import CIOutputs, ComponentModel


@dataclass(frozen=True)
class MetricContext:
    model: "ComponentModel"
    config: "PDConfig"
    batch: Any
    target_out: Tensor
    pre_weight_acts: dict[str, Float[Tensor, "..."]]
    ci: "CIOutputs"
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]]
    step: int
    reconstruction_loss: "ReconstructionLoss"
    is_eval: bool

    @property
    def current_frac_of_training(self) -> float:
        return self.step / self.config.steps if self.config.steps > 0 else 1.0
