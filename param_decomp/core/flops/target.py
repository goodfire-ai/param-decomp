"""Explicit matrix arithmetic for target forwards and their required backwards."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from param_decomp.core.components import BlockedFactorization, DenseFactorization, SiteSpec
from param_decomp.core.flops.types import ForwardBackwardFlops

GradientTarget = Literal["none", "components", "sources"]


@dataclass(frozen=True)
class TargetPassFlops:
    """A target pass and the contractions that other objectives can reuse.

    Shared forwards also occur in the clean pass. Reusable component projections
    consume unchanged inputs; their gradients can accumulate before one backward.
    An independent output-only source objective repeats `additional_backward` where
    output and auxiliary gradients would otherwise share the same contraction.
    """

    flops: ForwardBackwardFlops
    shared_forward: int
    reusable_components: dict[str, ForwardBackwardFlops]
    additional_backward: int

    def __post_init__(self) -> None:
        reusable_forward = sum(cost.forward for cost in self.reusable_components.values())
        reusable_backward = sum(cost.backward for cost in self.reusable_components.values())
        if not 0 <= self.shared_forward <= self.flops.forward - reusable_forward:
            raise ValueError("Shared and reusable forwards must be disjoint parts of the pass")
        if reusable_backward > self.flops.backward:
            raise ValueError("Reusable backwards must be part of the pass")
        if not 0 <= self.additional_backward <= self.flops.backward:
            raise ValueError("Repeated backwards must be part of the pass")


def linear_flops(
    name_for_err: str,
    n_rows: int,
    d_in: int,
    d_out: int,
    site: SiteSpec | None,
    *,
    gradients: GradientTarget,
    input_changed: bool,
    include_frozen_paths: bool,
    output_gradient: bool,
    auxiliary_gradient: bool,
    component_key: str,
) -> TargetPassFlops:
    match gradients:
        case "none":
            input_gradient, component_gradient, mask_gradient = False, False, False
        case "components":
            input_gradient, component_gradient, mask_gradient = input_changed, True, True
        case "sources":
            input_gradient, component_gradient, mask_gradient = input_changed, False, True
    frozen = 2 * n_rows * d_in * d_out
    reusable: dict[str, ForwardBackwardFlops] = {}
    if site is None:
        forward = frozen
        backward = frozen * input_gradient
        shared = frozen * (not input_changed)
    else:
        if (site.factorization.d_in, site.factorization.d_out) != (d_in, d_out):
            raise ValueError(f"Site {name_for_err} has incompatible factorization dimensions")
        match site.factorization:
            case DenseFactorization(C=n_components):
                pass
            case BlockedFactorization(c_per_block=n_components):
                pass
        input_projection = 2 * n_rows * d_in * n_components
        output_projection = 2 * n_rows * n_components * d_out
        forward = input_projection + output_projection + frozen * include_frozen_paths
        backward = (
            input_projection * (input_gradient + component_gradient)
            + output_projection * ((input_gradient or mask_gradient) + component_gradient)
            + frozen * include_frozen_paths * input_gradient
        )
        shared = frozen * include_frozen_paths * (not input_changed)
        if not input_changed:
            reusable[component_key] = ForwardBackwardFlops(
                input_projection,
                input_projection * component_gradient * (output_gradient or auxiliary_gradient),
            )
    return TargetPassFlops(
        ForwardBackwardFlops(forward, backward * (output_gradient or auxiliary_gradient)),
        shared,
        reusable,
        backward * (output_gradient and auxiliary_gradient),
    )


def frozen_product_flops(
    flops: int,
    *,
    gradients: GradientTarget,
    left_changed: bool,
    right_changed: bool,
    output_gradient: bool,
    auxiliary_gradient: bool,
) -> TargetPassFlops:
    match gradients:
        case "none":
            backward = 0
        case "components" | "sources":
            backward = flops * (left_changed + right_changed)
    return TargetPassFlops(
        ForwardBackwardFlops(flops, backward * (output_gradient or auxiliary_gradient)),
        flops * (not left_changed and not right_changed),
        {},
        backward * (output_gradient and auxiliary_gradient),
    )


def causal_attention_flops(
    batch_size: int,
    sequence_length: int,
    n_heads: int,
    head_dim: int,
    *,
    gradients: GradientTarget,
    query_changed: bool,
    key_changed: bool,
    value_changed: bool,
    auxiliary_gradient: bool,
) -> TargetPassFlops:
    n_causal_pairs = sequence_length * (sequence_length + 1) // 2
    product = 2 * batch_size * n_heads * n_causal_pairs * head_dim
    scores_changed = query_changed or key_changed
    return sum_target_flops(
        (
            frozen_product_flops(
                product,
                gradients=gradients,
                left_changed=query_changed,
                right_changed=key_changed,
                output_gradient=True,
                auxiliary_gradient=auxiliary_gradient,
            ),
            frozen_product_flops(
                product,
                gradients=gradients,
                left_changed=scores_changed,
                right_changed=value_changed,
                output_gradient=True,
                auxiliary_gradient=auxiliary_gradient,
            ),
        )
    )


def sum_target_flops(terms: Sequence[TargetPassFlops]) -> TargetPassFlops:
    reusable: dict[str, ForwardBackwardFlops] = {}
    for term in terms:
        if reusable.keys() & term.reusable_components.keys():
            raise ValueError("Each reusable projection must have its own identity")
        reusable.update(term.reusable_components)
    return TargetPassFlops(
        ForwardBackwardFlops(
            sum(term.flops.forward for term in terms), sum(term.flops.backward for term in terms)
        ),
        sum(term.shared_forward for term in terms),
        reusable,
        sum(term.additional_backward for term in terms),
    )


def validate_batch(batch_size: int, sequence_length: int) -> None:
    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("Batch size and sequence length must be positive")
