"""Validate CI inputs and dispatch to each architecture's useful contraction rules.

Counts omit elementwise operations, optimizer updates and rematerialization. Attention
assumes one uninterrupted sequence per batch row; causal attention counts its triangle.
Selected experts count only active blocks, irrespective of the execution backend.
"""

from param_decomp.core.ci_fn.architecture import CIFnArchitectureFootprint
from param_decomp.core.components import SiteSpec
from param_decomp.core.flops.types import ForwardBackwardFlops
from param_decomp.core.model import PositionAxis


def ci_fn_flops(
    arch: CIFnArchitectureFootprint,
    sites: tuple[SiteSpec, ...],
    batch_size: int,
    positions: PositionAxis,
    *,
    n_selected_blocks_per_token: int | None,
) -> ForwardBackwardFlops:
    """Count one CI evaluation and its gradient with respect to CI parameters.

    Clean target taps and routing are frozen: the input projection computes weight
    gradients but no input gradient. Subsequent contractions differentiate both inputs.
    The target supplies its top-k when routed; dense CI ignores this routing metadata.
    """
    return arch.useful_flops(
        sites, batch_size, positions, n_selected_blocks_per_token=n_selected_blocks_per_token
    )
