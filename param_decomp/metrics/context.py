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
    """Per-step bundle handed to every `Metric.update(ctx)` call.

    Built once per training step (after the DDP forward + CI calc) and once per eval
    batch.

    Attributes:
        model: The `ComponentModel` being trained.
        batch: The raw batch object consumed by the model and the reconstruction loss.
        target_out: Target-model output for this batch, used as the reconstruction
            reference.
        pre_weight_acts: Cached input activations into each target module, keyed by
            module path.
        ci: Causal-importance outputs (raw + lower-leaky + upper-leaky).
        weight_deltas: Per-target-module weight residual `target_weight - sum(components)`,
            keyed by module path.
        step: Current training step (or eval step) index.
        total_steps: Total number of training steps in the run.
        use_delta_component: Whether the weight-delta is included as an extra component
            during masking.
        sampling: Component-mask sampling strategy applied in stochastic metrics.
        n_mask_samples: Number of stochastic mask samples drawn per batch.
        reconstruction_loss: The user-supplied reconstruction loss callable.
        is_eval: True for an eval batch, False for a training step.
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
        """Return `step / total_steps`, or 1.0 when `total_steps` is non-positive."""
        return self.step / self.total_steps if self.total_steps > 0 else 1.0
