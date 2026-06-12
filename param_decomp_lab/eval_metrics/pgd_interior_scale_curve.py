"""Eval diagnostic: recon loss at interior scalings of the eval-PGD endpoint direction.

Decision-boundary-distortion detector (Kim, Lee & Lee, AAAI 2021, transplanted from the
image domain): runs the same sign-PGD attack as `PGDReconLoss` to get endpoint sources
`s*`, then evaluates recon at `s = c * s*` for `c` on a grid in `[0, 1]` (c=0 is the
mask=ci baseline). A defender that is robust at the attack's *endpoint* but vulnerable
at *interior* magnitudes along the same direction (loss(c) > loss(1) for some c < 1) has
the distorted-surface signature of catastrophic overfitting — it has hardened against
where cheap attacks land (saturated/corner masks), not against the threat set.

Logs `loss_at_scale/c{c}` scalars plus `interior_max_over_endpoint_ratio` =
max_c loss(c) / loss(1.0); values meaningfully above 1 indicate distortion.
"""

from typing import Literal, override

import torch
from pydantic import PositiveInt
from torch import Tensor

from param_decomp.distributed import all_reduce
from param_decomp.masks import AllLayersRouter
from param_decomp.metrics.base import Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.persistent_pgd_state import get_ppgd_mask_infos
from param_decomp.metrics.pgd_utils import PGDConfig, pgd_attack


class PGDInteriorScaleCurveConfig(PGDConfig):
    """PGD fields should mirror the eval `PGDReconLoss` config so the attack matches."""

    type: Literal["PGDInteriorScaleCurve"] = "PGDInteriorScaleCurve"
    n_scales: PositiveInt = 10
    n_batches_accum: int | None = 1


class PGDInteriorScaleCurve(Metric[PGDInteriorScaleCurveConfig]):
    """Recon loss along interior scalings of the worst-case PGD source direction."""

    log_namespace = "diagnostics"
    short_name = "PGDInteriorScale"

    @override
    def reset(self) -> None:
        n_points = self.cfg.n_scales + 1  # + the c=0 (mask=ci) baseline
        self._loss_sums = torch.zeros(n_points, device=self.device)
        self._n_examples = torch.zeros((), device=self.device)
        self._batches_seen = 0

    def _scales(self) -> list[float]:
        return [i / self.cfg.n_scales for i in range(self.cfg.n_scales + 1)]

    @override
    def update(self, ctx: MetricContext) -> None:
        if self.cfg.n_batches_accum is not None and self._batches_seen >= self.cfg.n_batches_accum:
            return None
        self._batches_seen += 1
        weight_deltas = ctx.weight_deltas if ctx.use_delta_component else None
        batch_dims = ctx.target_out.shape[:-1]

        endpoint_sources, _, _ = pgd_attack(
            model=self.model,
            batch=ctx.batch,
            ci=ctx.ci.lower_leaky,
            weight_deltas=weight_deltas,
            target_out=ctx.target_out,
            router=AllLayersRouter(),
            pgd_config=self.cfg,
            reconstruction_loss=ctx.reconstruction_loss,
        )

        n_at_scale: int | None = None
        for i, scale in enumerate(self._scales()):
            scaled = {k: v * scale for k, v in endpoint_sources.items()}
            mask_infos = get_ppgd_mask_infos(
                ci=ctx.ci.lower_leaky,
                weight_deltas=weight_deltas,
                ppgd_sources=scaled,
                routing_masks="all",
                batch_dims=batch_dims,
            )
            out = self.model(ctx.batch, mask_infos=mask_infos)
            sum_loss, n = ctx.reconstruction_loss(pred=out, target=ctx.target_out)
            self._loss_sums[i] += sum_loss.detach()
            n_at_scale = n
        assert n_at_scale is not None
        self._n_examples += n_at_scale
        return None

    @override
    def compute(self) -> MetricResult:
        cls = type(self).__name__
        loss_sums = all_reduce(self._loss_sums.clone())
        n = all_reduce(self._n_examples.clone()).clamp(min=1)
        losses = loss_sums / n
        out: dict[str, Tensor] = {
            f"{cls}/loss_at_scale/c{scale:.1f}": losses[i] for i, scale in enumerate(self._scales())
        }
        endpoint = losses[-1].clamp(min=1e-12)
        out[f"{cls}/interior_max_over_endpoint_ratio"] = losses.max() / endpoint
        return out
