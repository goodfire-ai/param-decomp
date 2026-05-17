from jaxtyping import Float
from torch import Tensor

from param_decomp.metrics.base import Metric
from param_decomp.metrics.context import MetricContext


def compute_losses(
    loss_instances: dict[str, Metric],
    ctx: MetricContext,
) -> dict[str, Float[Tensor, ""] | None]:
    """Compute per-metric live loss tensors for the current training step.

    Each metric's `update(ctx)` returns the per-batch scalar (a graph-attached tensor that the
    caller will backprop through), or None if the metric is gated off (e.g. PPGD before its
    `start_frac`).
    """
    return {slug: m.update(ctx) for slug, m in loss_instances.items()}
