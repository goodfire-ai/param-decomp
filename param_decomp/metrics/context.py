"""Per-step state passed to every metric's `update()`.

Built once per training step (after the DDP forward + CI calc) and once per eval batch.
"""

from dataclasses import dataclass
from typing import Any

from jaxtyping import Float
from torch import Tensor

from param_decomp.batch_and_loss_fns import ReconstructionLoss
from param_decomp.component_model import CIOutputs, ComponentModel
from param_decomp.masks import SamplingType


@dataclass(frozen=True)
class MetricContext:
    """Per-step bundle handed to every `Metric.update(ctx)`.

    Built once per training step (after the DDP forward + CI calc) and once per eval
    batch.
    """

    model: ComponentModel
    batch: Any
    target_out: Tensor
    pre_weight_acts: dict[str, Float[Tensor, "..."]]
    ci: CIOutputs
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]]
    step: int
    total_steps: int
    use_delta_component: bool
    sampling: SamplingType
    n_mask_samples: int
    reconstruction_loss: ReconstructionLoss
    is_eval: bool

    @property
    def current_frac_of_training(self) -> float:
        return self.step / self.total_steps if self.total_steps > 0 else 1.0
