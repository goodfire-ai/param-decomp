from typing import Any, ClassVar, override

import torch
from jaxtyping import Float
from torch import Tensor
from torch.distributed import ReduceOp

from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel
from spd.utils.distributed_utils import all_reduce, get_distributed_state


def _get_linear_annealed_p(
    current_frac_of_training: float,
    initial_p: float,
    p_anneal_start_frac: float,
    p_anneal_final_p: float | None,
    p_anneal_end_frac: float,
) -> float:
    """Calculate the linearly annealed p value for L_p sparsity loss.

    Args:
        current_frac_of_training: Current fraction of training
        initial_p: Starting p value
        p_anneal_start_frac: Fraction of training after which to start annealing
        p_anneal_final_p: Final p value to anneal to
        p_anneal_end_frac: Fraction of training when annealing ends. We stay at the final p value from this point onward

    Returns:
        Current p value based on linear annealing schedule
    """
    if p_anneal_final_p is None or p_anneal_start_frac >= 1.0:
        return initial_p

    assert p_anneal_end_frac >= p_anneal_start_frac, (
        f"p_anneal_end_frac ({p_anneal_end_frac}) must be >= "
        f"p_anneal_start_frac ({p_anneal_start_frac})"
    )

    if current_frac_of_training < p_anneal_start_frac:
        return initial_p
    elif current_frac_of_training >= p_anneal_end_frac:
        return p_anneal_final_p
    else:
        # Linear interpolation between start and end fractions
        progress = (current_frac_of_training - p_anneal_start_frac) / (
            p_anneal_end_frac - p_anneal_start_frac
        )
        return initial_p + (p_anneal_final_p - initial_p) * progress


def _importance_minimality_loss_update(
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
    pnorm: float,
    eps: float,
    p_anneal_start_frac: float,
    p_anneal_final_p: float | None,
    p_anneal_end_frac: float,
    current_frac_of_training: float,
) -> tuple[dict[str, Float[Tensor, " C"]], int]:
    """Calculate per-component sums of (ci_upper_leaky + eps) ** pnorm over batch/seq.

    Returns per-layer per-component sums and the number of batch/seq elements.
    These are used to compute the final loss by averaging over batch/seq
    and summing over components.

    NOTE: We don't normalize over the number of layers because a change in the number of layers
    should not change the ci loss that an ablation of a single component in a single layer might
    have. That said, we're unsure about this, perhaps we do want to normalize over n_layers.
    """
    assert ci_upper_leaky, "Empty ci_upper_leaky"
    pnorm = _get_linear_annealed_p(
        current_frac_of_training=current_frac_of_training,
        initial_p=pnorm,
        p_anneal_start_frac=p_anneal_start_frac,
        p_anneal_final_p=p_anneal_final_p,
        p_anneal_end_frac=p_anneal_end_frac,
    )
    per_component_sums: dict[str, Float[Tensor, " C"]] = {}
    for layer_name, layer_ci_upper_leaky in ci_upper_leaky.items():
        # NOTE: layer_ci_upper_leaky already >= 0, with shape [... C] where ... is batch/seq
        pnorm_result = (layer_ci_upper_leaky + eps) ** pnorm
        # Sum over batch/seq to get per-component sum [C]
        per_component_sums[layer_name] = pnorm_result.sum(dim=tuple(range(pnorm_result.dim() - 1)))
    n_examples = next(iter(ci_upper_leaky.values())).shape[:-1].numel()
    return per_component_sums, n_examples


def _importance_minimality_loss_compute(
    per_component_sums: dict[str, Float[Tensor, " C"]],
    n_examples: int,
    beta: float,
    world_size: int,
) -> Float[Tensor, ""]:
    """Compute final loss from accumulated per-component sums.

    For each layer:
    1. Divide per-component sums by n_examples to get means over batch/seq (i.e. per_component_mean)
    2. Calculate (per_component_mean + beta * per_component_mean * log2(1 + layer_sums * world_size)).sum()

    Then sum contributions from all layers.

    The log2 term uses layer_sums * world_size to estimate the global sum. When sums are already
    globally reduced (eval), pass world_size=1. When sums are local per-rank (training), pass the
    actual world_size.
    """
    total_loss = torch.tensor(0.0, device=next(iter(per_component_sums.values())).device)
    for layer_sums in per_component_sums.values():
        per_component_mean = layer_sums / n_examples
        layer_loss = (
            per_component_mean + beta * per_component_mean * torch.log2(1 + layer_sums * world_size)
        ).sum()
        total_loss += layer_loss
    return total_loss


def importance_minimality_loss(
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
    current_frac_of_training: float,
    eps: float,
    pnorm: float,
    beta: float,
    p_anneal_start_frac: float,
    p_anneal_final_p: float | None,
    p_anneal_end_frac: float,
) -> Float[Tensor, ""]:
    """Compute importance minimality loss."""

    per_component_sums, n_examples = _importance_minimality_loss_update(
        ci_upper_leaky=ci_upper_leaky,
        pnorm=pnorm,
        eps=eps,
        p_anneal_start_frac=p_anneal_start_frac,
        p_anneal_final_p=p_anneal_final_p,
        p_anneal_end_frac=p_anneal_end_frac,
        current_frac_of_training=current_frac_of_training,
    )
    dist_state = get_distributed_state()
    world_size = dist_state.world_size if dist_state is not None else 1
    return _importance_minimality_loss_compute(
        per_component_sums=per_component_sums,
        n_examples=n_examples,
        beta=beta,
        world_size=world_size,
    )


class ImportanceMinimalityLoss(Metric):
    """L_p loss on the sum of CI values.

    NOTE: We don't normalize over the number of layers because a change in the number of layers
    should not change the ci loss that an ablation of a single component in a single layer might
    have. That said, we're unsure about this, perhaps we do want to normalize over n_layers.

    Args:
        pnorm: The p value for the L_p norm applied element-wise before averaging
        p_anneal_start_frac: The fraction of training after which to start annealing p
            (1.0 = no annealing)
        p_anneal_final_p: The final p value to anneal to (None = no annealing)
        p_anneal_end_frac: The fraction of training when annealing ends. We stay at the final p
            value from this point onward (default 1.0 = anneal until end)
        eps: The epsilon value for numerical stability.
    """

    metric_section: ClassVar[str] = "loss"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        pnorm: float,
        beta: float,
        p_anneal_start_frac: float = 1.0,
        p_anneal_final_p: float | None = None,
        p_anneal_end_frac: float = 1.0,
        eps: float = 1e-12,
    ) -> None:
        self.pnorm = pnorm
        self.beta = beta
        self.eps = eps
        self.p_anneal_start_frac = p_anneal_start_frac
        self.p_anneal_final_p = p_anneal_final_p
        self.p_anneal_end_frac = p_anneal_end_frac
        self.device = device
        # Track per-layer per-component sums for proper aggregation
        self.per_component_sums: dict[str, Float[Tensor, " C"]] = {}
        self.n_examples = torch.tensor(0, device=device)

    @override
    def update(
        self,
        *,
        ci: CIOutputs,
        current_frac_of_training: float,
        **_: Any,
    ) -> None:
        per_component_sums, n_examples = _importance_minimality_loss_update(
            ci_upper_leaky=ci.upper_leaky,
            pnorm=self.pnorm,
            eps=self.eps,
            current_frac_of_training=current_frac_of_training,
            p_anneal_start_frac=self.p_anneal_start_frac,
            p_anneal_final_p=self.p_anneal_final_p,
            p_anneal_end_frac=self.p_anneal_end_frac,
        )
        # Accumulate per-layer per-component sums across batches
        for layer_name, layer_sums in per_component_sums.items():
            if layer_name not in self.per_component_sums:
                self.per_component_sums[layer_name] = torch.zeros_like(layer_sums)
            self.per_component_sums[layer_name] += layer_sums
        self.n_examples += n_examples

    @override
    def compute(self) -> Float[Tensor, ""]:
        reduced_sums: dict[str, Float[Tensor, " C"]] = {}
        for layer_name, layer_sums in self.per_component_sums.items():
            reduced_sums[layer_name] = all_reduce(layer_sums, op=ReduceOp.SUM)
        n_examples = int(all_reduce(self.n_examples, op=ReduceOp.SUM))

        return _importance_minimality_loss_compute(
            per_component_sums=reduced_sums,
            n_examples=n_examples,
            beta=self.beta,
            world_size=1,  # sums are already all_reduced
        )
