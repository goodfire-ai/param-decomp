from typing import Literal, override

import einops
import torch
from jaxtyping import Float
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.components import Components
from param_decomp.distributed import all_reduce
from param_decomp.metrics.base import LossMetricConfig, Metric, MetricResult
from param_decomp.metrics.context import MetricContext


class MaskedWeightNormLossConfig(LossMetricConfig):
    type: Literal["MaskedWeightNormLoss"] = "MaskedWeightNormLoss"


def _total_target_weights(components: dict[str, Components]) -> int:
    """Number of weights across every decomposed target matrix (`d_out * d_in` summed)."""
    return sum(c.V.shape[0] * c.U.shape[1] for c in components.values())


def _masked_weight_norm_sum(
    ci: dict[str, Float[Tensor, "... C"]],
    components: dict[str, Components],
) -> tuple[Float[Tensor, ""], int]:
    """Squared Frobenius norm of the CI-masked weight, summed over layers and datapoints.

    The CI-masked weight at one layer for one datapoint is
    `W[i, j] = sum_c ci_c * V[i, c] * U[c, j]`, and the penalty is `sum_{i,j} W[i, j]^2`.
    Rather than materialise the `[..., d_out, d_in]` weight per datapoint, expand the
    square and contract the spatial dims first:

        sum_{i,j} (sum_c ci_c V_ic U_cj)^2
          = sum_{c,c'} ci_c ci_c' (V^T V)[c,c'] (U U^T)[c,c']
          = ci @ ((V^T V) ⊙ (U U^T)) @ ci

    so the per-datapoint cost is `O(C^2)` against a `[C, C]` Gram product computed once
    per layer, independent of the spatial dims.

    Returns the sum over layers and datapoints, and the number of datapoints (leading
    dims of `ci`).
    """
    assert ci, "Empty ci"
    total = torch.zeros((), device=next(iter(ci.values())).device)
    for layer_name, layer_ci in ci.items():
        comp = components[layer_name]
        gram_v = einops.einsum(comp.V, comp.V, "v_dim c1, v_dim c2 -> c1 c2")
        gram_u = einops.einsum(comp.U, comp.U, "c1 u_dim, c2 u_dim -> c1 c2")
        gram = gram_v * gram_u
        ci_gram = einops.einsum(layer_ci, gram, "... c1, c1 c2 -> ... c2")
        per_datapoint = einops.einsum(ci_gram, layer_ci, "... c, ... c -> ...")
        total = total + per_datapoint.sum()
    n_examples = next(iter(ci.values())).shape[:-1].numel()
    return total, n_examples


def masked_weight_norm_loss(
    ci: dict[str, Float[Tensor, "... C"]],
    components: dict[str, Components],
) -> Float[Tensor, ""]:
    """Compute the masked-weight-norm loss directly (helper for tests/notebooks)."""
    sum_sq, n_examples = _masked_weight_norm_sum(ci=ci, components=components)
    return sum_sq / (_total_target_weights(components) * n_examples)


class MaskedWeightNormLoss(Metric[MaskedWeightNormLossConfig]):
    """Penalty on the squared Frobenius norm of the per-datapoint CI-masked weight.

    For each layer the CI-masked weight is `sum_c ci_c * (V[:, c] ⊗ U[c, :])` using
    `ci.lower_leaky` as the mask. The penalty sums the squared entries over each layer's
    weight, sums across layers, normalises by the total number of decomposed target
    weights, and averages over the batch and sequence dims. Computed via the Gram-matrix
    identity in `_masked_weight_norm_sum` rather than materialising the masked weight.
    """

    log_namespace = "loss"
    short_name = "MaskWNorm"

    @override
    def reset(self) -> None:
        self.sum_sq = torch.zeros((), device=self.device)
        self.n_examples = torch.zeros((), device=self.device, dtype=torch.long)

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        sum_sq, n = _masked_weight_norm_sum(ci=ctx.ci.lower_leaky, components=self.model.components)
        self.sum_sq += sum_sq.detach()
        self.n_examples += n
        return sum_sq / (_total_target_weights(self.model.components) * n)

    @override
    def compute(self) -> MetricResult:
        sum_sq = all_reduce(self.sum_sq, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return sum_sq / (_total_target_weights(self.model.components) * n_examples)
