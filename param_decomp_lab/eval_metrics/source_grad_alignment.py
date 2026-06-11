"""Eval diagnostic: local gradient alignment of the recon loss in mask-source space.

Catastrophic-overfitting detector — GradAlign (Andriushchenko & Flammarion, NeurIPS
2020) transplanted from the image eps-ball to the mask box. Samples a random shared
source `s1` in the box and a nearby point `s2 = clamp(s1 + eta)` with
`||eta||_inf <= perturb_radius` (the scale of ~1-2 eval-PGD steps), and measures
`cos(grad_s L(s1), grad_s L(s2))` per module and pooled across modules.

If the defender silences shallow training attacks by warping the local loss surface
(rather than by genuine robustness), this alignment collapses; the CO hypothesis
predicts collapse exactly when eval `PGDReconLoss` spikes. Alignment is intentionally
local — recon is legitimately nonlinear across the whole box, so only step-scale
alignment (what sign-PGD needs to make progress) is diagnostic.
"""

from typing import Literal, override

import torch
import torch.nn.functional as F
from jaxtyping import Float
from pydantic import PositiveInt
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.base_config import BaseConfig
from param_decomp.distributed import all_reduce, broadcast_tensor
from param_decomp.metrics.base import Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.persistent_pgd_state import get_ppgd_mask_infos

SharedSources = dict[str, Float[Tensor, "*ones mask_c"]]


class SourceGradAlignmentConfig(BaseConfig):
    """`perturb_radius` should match the eval PGD step scale (default 2x step 0.1)."""

    type: Literal["SourceGradAlignment"] = "SourceGradAlignment"
    perturb_radius: float = 0.2
    n_pairs: PositiveInt = 2
    n_batches_accum: int | None = 1


class SourceGradAlignment(Metric[SourceGradAlignmentConfig]):
    """Step-scale cosine alignment of source-space recon gradients at random box points."""

    log_namespace = "diagnostics"
    short_name = "SrcGradAlign"

    @override
    def reset(self) -> None:
        self._cos_sum: dict[str, Tensor] = {}
        self._pooled_sum = torch.zeros((), device=self.device)
        self._n_pairs_seen = torch.zeros((), device=self.device)
        self._batches_seen = 0

    def _source_grads(
        self,
        ctx: MetricContext,
        sources: SharedSources,
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    ) -> SharedSources:
        """Recon-loss gradient w.r.t. shared sources (averaged across ranks)."""
        batch_dims = ctx.target_out.shape[:-1]
        with torch.enable_grad():
            mask_infos = get_ppgd_mask_infos(
                ci={k: v.detach() for k, v in ctx.ci.lower_leaky.items()},
                weight_deltas=weight_deltas,
                ppgd_sources=sources,
                routing_masks="all",
                batch_dims=batch_dims,
            )
            out = self.model(ctx.batch, mask_infos=mask_infos)
            sum_loss, n = ctx.reconstruction_loss(pred=out, target=ctx.target_out)
            loss = sum_loss / n
        grads = torch.autograd.grad(loss, list(sources.values()))
        return {
            k: all_reduce(g, op=ReduceOp.AVG)
            for k, g in zip(sources.keys(), grads, strict=True)
        }

    @override
    def update(self, ctx: MetricContext) -> None:
        if self.cfg.n_batches_accum is not None and self._batches_seen >= self.cfg.n_batches_accum:
            return None
        self._batches_seen += 1
        weight_deltas = ctx.weight_deltas if ctx.use_delta_component else None
        batch_dims = ctx.target_out.shape[:-1]
        singleton_dims = [1] * len(batch_dims)

        for _ in range(self.cfg.n_pairs):
            s1: SharedSources = {}
            s2: SharedSources = {}
            for module_name in self.model.target_module_paths:
                mask_c = self.model.module_to_c[module_name] + (
                    1 if ctx.use_delta_component else 0
                )
                shape = torch.Size([*singleton_dims, mask_c])
                base = broadcast_tensor(torch.rand(shape, device=self.device))
                eta = broadcast_tensor(
                    (torch.rand(shape, device=self.device) * 2 - 1) * self.cfg.perturb_radius
                )
                s1[module_name] = base.clone().requires_grad_(True)
                s2[module_name] = (base + eta).clamp(0.0, 1.0).requires_grad_(True)

            g1 = self._source_grads(ctx, s1, weight_deltas)
            g2 = self._source_grads(ctx, s2, weight_deltas)

            for module_name in g1:
                cos = F.cosine_similarity(
                    g1[module_name].flatten(), g2[module_name].flatten(), dim=0
                )
                if module_name not in self._cos_sum:
                    self._cos_sum[module_name] = torch.zeros((), device=self.device)
                self._cos_sum[module_name] += cos.detach()
            order = sorted(g1)
            pooled1 = torch.cat([g1[m].flatten() for m in order])
            pooled2 = torch.cat([g2[m].flatten() for m in order])
            self._pooled_sum += F.cosine_similarity(pooled1, pooled2, dim=0).detach()
            self._n_pairs_seen += 1
        return None

    @override
    def compute(self) -> MetricResult:
        cls = type(self).__name__
        n = self._n_pairs_seen.clamp(min=1)
        out: dict[str, Tensor] = {
            f"{cls}/{module_name}": cos_sum / n for module_name, cos_sum in self._cos_sum.items()
        }
        out[f"{cls}/pooled"] = self._pooled_sum / n
        return out
