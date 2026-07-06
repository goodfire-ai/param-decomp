"""Bounded smooth-L0 (Geman–McClure) importance-minimality penalty on CI values.

A drop-in alternative to the `L_p` `ImportanceMinimalityLoss`: penalty per CI value is
`φ(c) = c² / (c² + γ²)` instead of `(c + eps)^p`. It is flat at 0 (`φ'(0)=0`) and bounded
(`~0.65/γ` near the threshold `c≈γ`), so it has no gradient cliff at the accumulation
point where most components live, while the per-component sum saturates to ≈ the active
count. Shares the `L_p` variant's entropy term and exact-DDP reduction; the additive
`(sum, entropy)` split (`lp_and_entropy_terms`) is penalty-agnostic, so it's reused here.
"""

from typing import cast, override

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_fn
from jaxtyping import Float
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.distributed import all_reduce, get_distributed_state
from param_decomp.metrics.base import Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.importance_minimality import finalize_imp_min, lp_and_entropy_terms
from param_decomp_config.losses import SmoothL0ImportanceMinimalityLossConfig


def annealed_gamma(
    current_frac_of_training: float,
    initial_gamma: float,
    gamma_anneal_start_frac: float,
    gamma_final: float | None,
    gamma_anneal_end_frac: float,
) -> float:
    """The linearly-annealed smooth-L0 threshold `γ` at the given fraction of training."""
    if gamma_final is None or gamma_anneal_start_frac >= 1.0:
        return initial_gamma
    assert gamma_anneal_end_frac >= gamma_anneal_start_frac, (
        f"gamma_anneal_end_frac ({gamma_anneal_end_frac}) must be >= "
        f"gamma_anneal_start_frac ({gamma_anneal_start_frac})"
    )
    if current_frac_of_training < gamma_anneal_start_frac:
        return initial_gamma
    elif current_frac_of_training >= gamma_anneal_end_frac:
        return gamma_final
    progress = (current_frac_of_training - gamma_anneal_start_frac) / (
        gamma_anneal_end_frac - gamma_anneal_start_frac
    )
    return initial_gamma + (gamma_final - initial_gamma) * progress


def per_component_smooth_l0_sums(
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
    gamma: float,
) -> tuple[dict[str, Float[Tensor, " C"]], int]:
    """Per-component `Σ` over positions of `c² / (c² + γ²)`, and the position count."""
    assert ci_upper_leaky, "Empty ci_upper_leaky"
    assert gamma > 0.0, f"gamma must be positive, got {gamma}"
    gamma_sq = gamma * gamma
    out: dict[str, Float[Tensor, " C"]] = {}
    for layer_name, layer_ci in ci_upper_leaky.items():
        ci_sq = layer_ci * layer_ci
        phi = ci_sq / (ci_sq + gamma_sq)
        out[layer_name] = phi.sum(dim=tuple(range(phi.dim() - 1)))
    n_examples = next(iter(ci_upper_leaky.values())).shape[:-1].numel()
    return out, n_examples


def smooth_l0_importance_minimality_loss(
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
    current_frac_of_training: float,
    gamma: float,
    beta: float,
    gamma_anneal_start_frac: float,
    gamma_final: float | None,
    gamma_anneal_end_frac: float,
) -> Float[Tensor, ""]:
    """Compute the smooth-L0 importance-minimality loss directly (single-process / external callers).

    Operates on the given (un-reduced) sums — correct as-is single-process. Under DDP use
    `SmoothL0ImportanceMinimalityLoss` (which reduces to the global sum) for the exact loss.
    """
    g = annealed_gamma(
        current_frac_of_training=current_frac_of_training,
        initial_gamma=gamma,
        gamma_anneal_start_frac=gamma_anneal_start_frac,
        gamma_final=gamma_final,
        gamma_anneal_end_frac=gamma_anneal_end_frac,
    )
    per_component_sums, n_examples = per_component_smooth_l0_sums(
        ci_upper_leaky=ci_upper_leaky, gamma=g
    )
    return finalize_imp_min(
        per_component_sums=per_component_sums,
        n_examples=n_examples,
        beta=beta,
    )


class SmoothL0ImportanceMinimalityLoss(Metric[SmoothL0ImportanceMinimalityLossConfig]):
    """Bounded smooth-L0 (Geman–McClure) penalty driving CI sparsity.

    `c² / (c² + γ²)` summed across components plus a `beta`-weighted
    `mean * log2(1 + sum)` term.
    """

    log_namespace = "loss"
    short_name = "SmoothL0ImpMin"

    @override
    def reset(self) -> None:
        self.per_component_sums: dict[str, Float[Tensor, " C"]] = {}
        self.n_examples = torch.zeros((), device=self.device, dtype=torch.long)

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        gamma = annealed_gamma(
            current_frac_of_training=ctx.current_frac_of_training,
            initial_gamma=self.cfg.gamma,
            gamma_anneal_start_frac=self.cfg.gamma_anneal_start_frac,
            gamma_final=self.cfg.gamma_final,
            gamma_anneal_end_frac=self.cfg.gamma_anneal_end_frac,
        )
        per_component_sums, n = per_component_smooth_l0_sums(
            ci_upper_leaky=ctx.ci.upper_leaky,
            gamma=gamma,
        )
        for layer_name, layer_sums in per_component_sums.items():
            if layer_name not in self.per_component_sums:
                self.per_component_sums[layer_name] = torch.zeros_like(layer_sums)
            self.per_component_sums[layer_name] += layer_sums.detach()
        self.n_examples += n

        # Exact global sums for the live loss: SUM-reduce per_component_sums across ranks
        # with the autograd-aware all_reduce so the convex log2 term sees the true
        # full-batch sum (a per-rank local sum would give a Jensen upward bias), and so
        # gradient flows back through each rank's local CI values. n_examples is uniform
        # across ranks under DP, so we multiply rather than reduce.
        dist_state = get_distributed_state()
        if dist_state is not None:
            per_component_sums = {
                k: cast(Tensor, dist_fn.all_reduce(v, op=dist.ReduceOp.SUM))
                for k, v in per_component_sums.items()
            }
            n = n * dist_state.world_size
        return finalize_imp_min(
            per_component_sums=per_component_sums,
            n_examples=n,
            beta=self.cfg.beta,
        )

    @override
    def compute(self) -> MetricResult:
        reduced_sums = {
            k: all_reduce(v, op=ReduceOp.SUM) for k, v in self.per_component_sums.items()
        }
        n_examples = int(all_reduce(self.n_examples, op=ReduceOp.SUM))
        smooth_l0, entropy = lp_and_entropy_terms(reduced_sums, n_examples)
        name = self.instance_key
        # Both go under `imp_min/` (fully-qualified keys, so they skip the `loss/`
        # namespace): the `smooth_l0` proxy (≈ active-component count) isn't a loss term,
        # and grouping the two keeps the headline beside its proxy, off the loss panel.
        return {
            f"imp_min/{name}": smooth_l0 + self.cfg.beta * entropy,
            f"imp_min/{name}_no_beta": smooth_l0,
        }
