from typing import Literal, override

import torch
from jaxtyping import Float
from pydantic import NonNegativeFloat
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.base_config import Probability
from param_decomp.distributed import all_reduce, get_distributed_state
from param_decomp.metrics.base import LossMetricConfig, Metric, MetricResult
from param_decomp.metrics.context import MetricContext


class ImportanceMinimalityLossConfig(LossMetricConfig):
    """Config for the `L_p`-style importance-minimality penalty on upper-leaky CI values.

    `pnorm` is the initial `p`; `beta` weights the entropy-like `mean * log2(1 + sum)`
    term added on top of the `L_p` term. `pnorm` is linearly annealed toward
    `p_anneal_final_p` between `p_anneal_start_frac` and `p_anneal_end_frac` of training
    (no-op when `p_anneal_final_p is None` or `p_anneal_start_frac == 1.0`).
    """

    type: Literal["ImportanceMinimalityLoss"] = "ImportanceMinimalityLoss"
    pnorm: NonNegativeFloat
    beta: NonNegativeFloat
    p_anneal_start_frac: Probability = 1.0
    p_anneal_final_p: NonNegativeFloat | None = None
    p_anneal_end_frac: Probability = 1.0
    eps: NonNegativeFloat = 1e-12


def _get_linear_annealed_p(
    current_frac_of_training: float,
    initial_p: float,
    p_anneal_start_frac: float,
    p_anneal_final_p: float | None,
    p_anneal_end_frac: float,
) -> float:
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
    progress = (current_frac_of_training - p_anneal_start_frac) / (
        p_anneal_end_frac - p_anneal_start_frac
    )
    return initial_p + (p_anneal_final_p - initial_p) * progress


def _per_component_sums(
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
    pnorm: float,
    eps: float,
) -> tuple[dict[str, Float[Tensor, " C"]], int]:
    assert ci_upper_leaky, "Empty ci_upper_leaky"
    out: dict[str, Float[Tensor, " C"]] = {}
    for layer_name, layer_ci in ci_upper_leaky.items():
        result = (layer_ci + eps) ** pnorm
        out[layer_name] = result.sum(dim=tuple(range(result.dim() - 1)))
    n_examples = next(iter(ci_upper_leaky.values())).shape[:-1].numel()
    return out, n_examples


def _finalize(
    per_component_sums: dict[str, Float[Tensor, " C"]],
    n_examples: int,
    beta: float,
    world_size: int,
) -> Float[Tensor, ""]:
    total_loss = torch.zeros((), device=next(iter(per_component_sums.values())).device)
    for layer_sums in per_component_sums.values():
        per_component_mean = layer_sums / n_examples
        layer_loss = (
            per_component_mean + beta * per_component_mean * torch.log2(1 + layer_sums * world_size)
        ).sum()
        total_loss = total_loss + layer_loss
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
    """Compute the importance-minimality loss directly (helper for external callers)."""
    annealed_p = _get_linear_annealed_p(
        current_frac_of_training=current_frac_of_training,
        initial_p=pnorm,
        p_anneal_start_frac=p_anneal_start_frac,
        p_anneal_final_p=p_anneal_final_p,
        p_anneal_end_frac=p_anneal_end_frac,
    )
    per_component_sums, n_examples = _per_component_sums(
        ci_upper_leaky=ci_upper_leaky, pnorm=annealed_p, eps=eps
    )
    dist_state = get_distributed_state()
    world_size = dist_state.world_size if dist_state is not None else 1
    return _finalize(
        per_component_sums=per_component_sums,
        n_examples=n_examples,
        beta=beta,
        world_size=world_size,
    )


class ImportanceMinimalityLoss(Metric[ImportanceMinimalityLossConfig]):
    """`L_p`-style penalty driving CI sparsity.

    `(ci + eps)^p` summed across components plus a `beta`-weighted
    `mean * log2(1 + sum)` term.
    """

    log_namespace = "loss"
    short_name = "ImpMin"

    @override
    def reset(self) -> None:
        self.per_component_sums: dict[str, Float[Tensor, " C"]] = {}
        self.n_examples = torch.zeros((), device=self.device, dtype=torch.long)

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        pnorm = _get_linear_annealed_p(
            current_frac_of_training=ctx.current_frac_of_training,
            initial_p=self.cfg.pnorm,
            p_anneal_start_frac=self.cfg.p_anneal_start_frac,
            p_anneal_final_p=self.cfg.p_anneal_final_p,
            p_anneal_end_frac=self.cfg.p_anneal_end_frac,
        )
        per_component_sums, n = _per_component_sums(
            ci_upper_leaky=ctx.ci.upper_leaky,
            pnorm=pnorm,
            eps=self.cfg.eps,
        )
        for layer_name, layer_sums in per_component_sums.items():
            if layer_name not in self.per_component_sums:
                self.per_component_sums[layer_name] = torch.zeros_like(layer_sums)
            self.per_component_sums[layer_name] += layer_sums.detach()
        self.n_examples += n

        dist_state = get_distributed_state()
        world_size = dist_state.world_size if dist_state is not None else 1
        return _finalize(
            per_component_sums=per_component_sums,
            n_examples=n,
            beta=self.cfg.beta,
            world_size=world_size,
        )

    @override
    def compute(self) -> MetricResult:
        reduced_sums = {
            k: all_reduce(v, op=ReduceOp.SUM) for k, v in self.per_component_sums.items()
        }
        n_examples = int(all_reduce(self.n_examples, op=ReduceOp.SUM))
        return _finalize(
            per_component_sums=reduced_sums,
            n_examples=n_examples,
            beta=self.cfg.beta,
            world_size=1,
        )
