from typing import Literal, cast, override

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_fn
from jaxtyping import Float
from pydantic import NonNegativeFloat, PositiveInt
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.base_config import Probability
from param_decomp.distributed import all_reduce, get_distributed_state
from param_decomp.metrics.base import LossMetricConfig, Metric, MetricResult
from param_decomp.metrics.context import MetricContext


class ImportanceMinimalityLossConfig(LossMetricConfig):
    """Config for the bare `L_p` mean term on upper-leaky CI values.

    `pnorm` is the initial `p`; it is linearly annealed toward `p_anneal_final_p`
    between `p_anneal_start_frac` and `p_anneal_end_frac` of training (no-op when
    `p_anneal_final_p is None` or `p_anneal_start_frac == 1.0`). The frequency-weighted
    `log2` term that used to live here is now `FrequencyMinimalityLoss`.
    """

    type: Literal["ImportanceMinimalityLoss"] = "ImportanceMinimalityLoss"
    pnorm: NonNegativeFloat
    p_anneal_start_frac: Probability = 1.0
    p_anneal_final_p: NonNegativeFloat | None = None
    p_anneal_end_frac: Probability = 1.0
    eps: NonNegativeFloat = 1e-12


class FrequencyMinimalityLossConfig(LossMetricConfig):
    """Config for the batch-invariant frequency-minimality penalty.

    Penalizes a component's per-token firing frequency `f_c` (over the whole global
    batch) by `f_c * log2(1 + reference_token_count * f_c)`, summed over components.
    The `f=0 -> 0` cutoff is inherent to the form. `reference_token_count` (`a'`) is the
    token count the penalty is normalized against: setting it to the run's `B*T`
    reproduces the implicit `B*T` the old rolled `ImportanceMinimalityLoss` used inside
    its `log2`, so coefficients transfer as `freq.coeff = old imp.coeff * old beta`.

    `pnorm` and the anneal fields parameterize the same `(ci + eps)^p` per-token power
    that feeds `f_c` (shared in spirit with `ImportanceMinimalityLoss`, configured
    independently here so the two terms can use different `p`).
    """

    type: Literal["FrequencyMinimalityLoss"] = "FrequencyMinimalityLoss"
    pnorm: NonNegativeFloat
    reference_token_count: PositiveInt
    p_anneal_start_frac: Probability = 1.0
    p_anneal_final_p: NonNegativeFloat | None = None
    p_anneal_end_frac: Probability = 1.0
    eps: NonNegativeFloat = 1e-12


def annealed_pnorm(
    current_frac_of_training: float,
    initial_p: float,
    p_anneal_start_frac: float,
    p_anneal_final_p: float | None,
    p_anneal_end_frac: float,
) -> float:
    """The linearly-annealed ``p`` for the ``L_p`` sparsity term at the given fraction of training."""
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


def per_component_lp_sums(
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
    pnorm: float,
    eps: float,
) -> tuple[dict[str, Float[Tensor, " C"]], int]:
    """Per-component ``Σ`` over positions of ``(ci + eps) ** pnorm``, and the position count."""
    assert ci_upper_leaky, "Empty ci_upper_leaky"
    out: dict[str, Float[Tensor, " C"]] = {}
    for layer_name, layer_ci in ci_upper_leaky.items():
        result = (layer_ci + eps) ** pnorm
        out[layer_name] = result.sum(dim=tuple(range(result.dim() - 1)))
    n_examples = next(iter(ci_upper_leaky.values())).shape[:-1].numel()
    return out, n_examples


def finalize_imp_min(
    per_component_sums: dict[str, Float[Tensor, " C"]],
    n_examples: int,
) -> Float[Tensor, ""]:
    """Bare mean term ``Σ_c f_c`` from per-component ``L_p`` sums (``f_c = sum_c / n``)."""
    total_loss = torch.zeros((), device=next(iter(per_component_sums.values())).device)
    for layer_sums in per_component_sums.values():
        total_loss = total_loss + (layer_sums / n_examples).sum()
    return total_loss


def finalize_freq_min(
    per_component_sums: dict[str, Float[Tensor, " C"]],
    n_examples: int,
    reference_token_count: int,
) -> Float[Tensor, ""]:
    """Frequency-minimality ``Σ_c f_c * log2(1 + a' * f_c)`` (``f_c = sum_c / n``, ``a' = reference_token_count``).

    Batch-invariant: ``f_c`` is a per-token frequency, so the same firing rate at a
    different batch size yields the same value. Pass the globally-reduced per-component
    sums (and the corresponding global ``n_examples``) so ``f_c`` is the true full-batch
    frequency — a per-rank local ``f_c`` would give a Jensen bias through the convex log.
    """
    total_loss = torch.zeros((), device=next(iter(per_component_sums.values())).device)
    for layer_sums in per_component_sums.values():
        f = layer_sums / n_examples
        total_loss = total_loss + (f * torch.log2(1 + reference_token_count * f)).sum()
    return total_loss


def importance_minimality_loss(
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
    current_frac_of_training: float,
    eps: float,
    pnorm: float,
    p_anneal_start_frac: float,
    p_anneal_final_p: float | None,
    p_anneal_end_frac: float,
) -> Float[Tensor, ""]:
    """Compute the importance-minimality loss directly (single-process / external callers).

    Operates on the given (un-reduced) sums — correct as-is single-process. Under
    DDP use ``ImportanceMinimalityLoss`` (which reduces to the global sum) for the
    exact loss.
    """
    annealed_p = annealed_pnorm(
        current_frac_of_training=current_frac_of_training,
        initial_p=pnorm,
        p_anneal_start_frac=p_anneal_start_frac,
        p_anneal_final_p=p_anneal_final_p,
        p_anneal_end_frac=p_anneal_end_frac,
    )
    per_component_sums, n_examples = per_component_lp_sums(
        ci_upper_leaky=ci_upper_leaky, pnorm=annealed_p, eps=eps
    )
    return finalize_imp_min(per_component_sums=per_component_sums, n_examples=n_examples)


def _reduce_to_global_sums(
    per_component_sums: dict[str, Float[Tensor, " C"]], n_local: int
) -> tuple[dict[str, Float[Tensor, " C"]], int]:
    """SUM-reduce per-component sums across the DP world (autograd-aware), and scale ``n``.

    The autograd-aware all_reduce so gradient flows back through each rank's local CI
    values; ``n_examples`` is uniform across ranks under DP, so it scales by world size.
    No-op single-process.
    """
    dist_state = get_distributed_state()
    if dist_state is None:
        return per_component_sums, n_local
    reduced = {
        k: cast(Tensor, dist_fn.all_reduce(v, op=dist.ReduceOp.SUM))
        for k, v in per_component_sums.items()
    }
    return reduced, n_local * dist_state.world_size


class ImportanceMinimalityLoss(Metric[ImportanceMinimalityLossConfig]):
    """`L_p`-style penalty driving CI sparsity: the bare per-component mean ``Σ_c f_c``."""

    log_namespace = "loss"
    short_name = "ImpMin"

    @override
    def reset(self) -> None:
        self.per_component_sums: dict[str, Float[Tensor, " C"]] = {}
        self.n_examples = torch.zeros((), device=self.device, dtype=torch.long)

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        pnorm = annealed_pnorm(
            current_frac_of_training=ctx.current_frac_of_training,
            initial_p=self.cfg.pnorm,
            p_anneal_start_frac=self.cfg.p_anneal_start_frac,
            p_anneal_final_p=self.cfg.p_anneal_final_p,
            p_anneal_end_frac=self.cfg.p_anneal_end_frac,
        )
        per_component_sums, n = per_component_lp_sums(
            ci_upper_leaky=ctx.ci.upper_leaky,
            pnorm=pnorm,
            eps=self.cfg.eps,
        )
        for layer_name, layer_sums in per_component_sums.items():
            if layer_name not in self.per_component_sums:
                self.per_component_sums[layer_name] = torch.zeros_like(layer_sums)
            self.per_component_sums[layer_name] += layer_sums.detach()
        self.n_examples += n

        global_sums, n_global = _reduce_to_global_sums(per_component_sums, n)
        return finalize_imp_min(per_component_sums=global_sums, n_examples=n_global)

    @override
    def compute(self) -> MetricResult:
        reduced_sums = {
            k: all_reduce(v, op=ReduceOp.SUM) for k, v in self.per_component_sums.items()
        }
        n_examples = int(all_reduce(self.n_examples, op=ReduceOp.SUM))
        return finalize_imp_min(per_component_sums=reduced_sums, n_examples=n_examples)


class FrequencyMinimalityLoss(Metric[FrequencyMinimalityLossConfig]):
    """Batch-invariant frequency penalty: ``Σ_c f_c * log2(1 + a' * f_c)``."""

    log_namespace = "loss"
    short_name = "FreqMin"

    @override
    def reset(self) -> None:
        self.per_component_sums: dict[str, Float[Tensor, " C"]] = {}
        self.n_examples = torch.zeros((), device=self.device, dtype=torch.long)

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        pnorm = annealed_pnorm(
            current_frac_of_training=ctx.current_frac_of_training,
            initial_p=self.cfg.pnorm,
            p_anneal_start_frac=self.cfg.p_anneal_start_frac,
            p_anneal_final_p=self.cfg.p_anneal_final_p,
            p_anneal_end_frac=self.cfg.p_anneal_end_frac,
        )
        per_component_sums, n = per_component_lp_sums(
            ci_upper_leaky=ctx.ci.upper_leaky,
            pnorm=pnorm,
            eps=self.cfg.eps,
        )
        for layer_name, layer_sums in per_component_sums.items():
            if layer_name not in self.per_component_sums:
                self.per_component_sums[layer_name] = torch.zeros_like(layer_sums)
            self.per_component_sums[layer_name] += layer_sums.detach()
        self.n_examples += n

        global_sums, n_global = _reduce_to_global_sums(per_component_sums, n)
        return finalize_freq_min(
            per_component_sums=global_sums,
            n_examples=n_global,
            reference_token_count=self.cfg.reference_token_count,
        )

    @override
    def compute(self) -> MetricResult:
        reduced_sums = {
            k: all_reduce(v, op=ReduceOp.SUM) for k, v in self.per_component_sums.items()
        }
        n_examples = int(all_reduce(self.n_examples, op=ReduceOp.SUM))
        return finalize_freq_min(
            per_component_sums=reduced_sums,
            n_examples=n_examples,
            reference_token_count=self.cfg.reference_token_count,
        )
