"""Eval metric: per-layer histograms of the worst-case PGD adversarial source values.

Runs the same sign-PGD attack as `PGDReconLoss` and plots the distribution of the final
adversarial `source` values (the mask box variable, in `[0, 1]`). Answers "what does the
worst-case mask the attack converges to look like" — all-zeros, all-ones, bimodal at the
rails, or soft/interior.
"""

from collections import defaultdict
from typing import Literal, override

import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor

from param_decomp.distributed import gather_all_tensors
from param_decomp.masks import AllLayersRouter
from param_decomp.metrics.base import Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.pgd_utils import PGDConfig, pgd_attack
from param_decomp_lab.eval_metrics.plotting import plot_ci_values_histograms


class PGDSourceHistogramConfig(PGDConfig):
    """`n_batches_accum=None` accumulates every batch in the eval pass."""

    type: Literal["PGDSourceHistogram"] = "PGDSourceHistogram"
    n_batches_accum: int | None = 1
    bins: int = 100


class PGDSourceHistogram(Metric[PGDSourceHistogramConfig]):
    """Per-layer histograms of the final sign-PGD adversarial source values."""

    log_namespace = "figures"
    slow = True
    short_name = "PGDSourceHist"

    @override
    def reset(self) -> None:
        self.batches_seen = 0
        self.sources = defaultdict[str, list[Float[Tensor, "... mask_c"]]](list)

    @override
    def update(self, ctx: MetricContext) -> None:
        if self.cfg.n_batches_accum is not None and self.batches_seen >= self.cfg.n_batches_accum:
            return None
        self.batches_seen += 1
        weight_deltas = ctx.weight_deltas if ctx.use_delta_component else None
        sources, _, _ = pgd_attack(
            model=self.model,
            batch=ctx.batch,
            ci=ctx.ci.lower_leaky,
            weight_deltas=weight_deltas,
            target_out=ctx.target_out,
            router=AllLayersRouter(),
            pgd_config=self.cfg,
            reconstruction_loss=ctx.reconstruction_loss,
        )
        for module_name, source in sources.items():
            self.sources[module_name].append(source)
        return None

    @override
    def compute(self) -> MetricResult:
        if self.batches_seen == 0:
            raise RuntimeError("No batches seen yet")
        gathered: dict[str, Float[Tensor, "... mask_c"]] = {}
        for module_name, source_list in self.sources.items():
            local = torch.cat(source_list, dim=0)
            gathered[module_name] = torch.cat(gather_all_tensors(local), dim=0)
        fig: Image.Image = plot_ci_values_histograms(gathered, bins=self.cfg.bins)
        return {"pgd_adversarial_source_values": fig}
