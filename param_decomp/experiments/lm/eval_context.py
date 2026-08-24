"""Runtime inputs shared by bound LM evaluation operations."""

from collections.abc import Callable
from dataclasses import dataclass

import jax

from param_decomp.core.run import EvalInvocation
from param_decomp.core.slow_eval import SiteReduction


@dataclass(frozen=True)
class LMEvalContext(EvalInvocation):
    pass_index: int
    batches: tuple[jax.Array, ...]
    shared_ci_reductions: Callable[[], dict[str, SiteReduction]]
    """This pass's per-site CI reductions at threshold 0 with no histograms — the common
    denominator several slow operations read. Accumulated over `batches` on first call
    and cached, so consumers on the same pass share one forward sweep."""
