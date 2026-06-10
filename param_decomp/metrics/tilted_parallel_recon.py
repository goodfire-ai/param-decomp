"""Tilted parallel-sampling recon loss: a stateless adversary that replaces serial PGD.

Draws `n_candidates` vertex-biased joint mask candidates per (batch, position), scores
each under per-position KL, and combines them with a soft worst-case tilt
`tau * logsumexp(L / tau)`. As `tau -> 0` this approaches the per-position worst case (a
strong adversary); as `tau -> inf` it approaches the mean (ordinary stochastic recon).

Stateless by construction: nothing persists across steps, so there is no optimizer,
warmup, backward hook, or checkpoint state. The `k` candidates batch along the leading
dim into a single wide forward (cost is width, not depth).
"""

from typing import ClassVar, Literal, override

import torch
import torch.nn.functional as F
from jaxtyping import Float
from pydantic import Field, PositiveFloat, PositiveInt
from torch import Tensor

from param_decomp.base_config import Probability
from param_decomp.distributed import all_reduce
from param_decomp.masks import AllLayersRouter
from param_decomp.metrics.base import LossMetricConfig, Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.persistent_pgd_state import PPGDSources, get_ppgd_mask_infos


class TiltedParallelReconLossConfig(LossMetricConfig):
    type: Literal["TiltedParallelReconLoss"] = "TiltedParallelReconLoss"
    n_candidates: PositiveInt = Field(
        default=8, description="Number of parallel joint-mask candidates k drawn per position."
    )
    temperature: PositiveFloat = Field(
        default=0.1,
        description=(
            "Soft worst-case tilt tau in `tau * logsumexp(L_i / tau)`. Small tau ~ per-position"
            " max (strong adversary); large tau ~ mean (stochastic recon)."
        ),
    )
    bernoulli_p: Probability = Field(
        default=0.5,
        description="P(source=1) for vertex (Bernoulli) candidate coordinates; mask = ci or 1.",
    )
    uniform_frac: Probability = Field(
        default=0.0,
        description="Per-coordinate probability of drawing continuous U[0,1] instead of a vertex.",
    )
    start_frac: Probability = 0.0


class TiltedParallelReconLoss(Metric[TiltedParallelReconLossConfig]):
    """Stateless tilted parallel-sampling adversarial recon loss (routes to all layers)."""

    log_namespace: ClassVar[str] = "loss"
    slow: ClassVar[bool] = True
    short_name = "TiltedRecon"

    def __init__(self, cfg: TiltedParallelReconLossConfig) -> None:
        super().__init__(cfg)
        self._router = AllLayersRouter()

    @override
    def reset(self) -> None:
        self._recon_sum_loss = torch.zeros((), device=self.device)
        self._recon_n_examples = torch.zeros((), device=self.device, dtype=torch.long)

    def _draw_sources(
        self, k: int, batch_dims: tuple[int, ...], source_c: int, device: torch.device
    ) -> Float[Tensor, "k *batch_dims source_c"]:
        shape = (k, *batch_dims, source_c)
        vertex = torch.bernoulli(torch.full(shape, float(self.cfg.bernoulli_p), device=device))
        if self.cfg.uniform_frac == 0.0:
            return vertex
        uniform = torch.rand(shape, device=device)
        use_uniform = torch.rand(shape, device=device) < self.cfg.uniform_frac
        return torch.where(use_uniform, uniform, vertex)

    def _tilted_recon_sum_and_n(
        self,
        ctx: MetricContext,
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    ) -> tuple[Float[Tensor, ""], int]:
        ci = ctx.ci.lower_leaky
        batch_dims = tuple(ctx.target_out.shape[:-1])
        assert len(batch_dims) == 2, f"expected (batch, seq) target dims, got {batch_dims}"
        b, t = batch_dims
        k = self.cfg.n_candidates
        vocab = ctx.target_out.shape[-1]
        device = ctx.target_out.device

        wide_dims = (k * b, t)
        sources: PPGDSources = {}
        for module_name, c in self.model.module_to_c.items():
            source_c = c + 1 if ctx.use_delta_component else c
            drawn = self._draw_sources(k, batch_dims, source_c, device)
            sources[module_name] = drawn.reshape(k * b, t, source_c)

        ci_wide = {name: v.repeat(k, 1, 1) for name, v in ci.items()}
        routing_masks = self._router.get_masks(
            module_names=self.model.target_module_paths, mask_shape=wide_dims
        )
        mask_infos = get_ppgd_mask_infos(
            ci=ci_wide,
            weight_deltas=weight_deltas,
            ppgd_sources=sources,
            routing_masks=routing_masks,
            batch_dims=wide_dims,
        )
        batch_wide = ctx.batch.repeat(k, *([1] * (ctx.batch.ndim - 1)))
        out = self.model(batch_wide, mask_infos=mask_infos)

        log_q = F.log_softmax(out.view(k, b, t, vocab), dim=-1)
        p = F.softmax(ctx.target_out, dim=-1).unsqueeze(0)  # [1, b, t, vocab]
        kl = (p * (p.clamp_min(1e-12).log() - log_q)).sum(-1)  # [k, b, t]

        tau = self.cfg.temperature
        tilted_per_position = tau * torch.logsumexp(kl / tau, dim=0)  # [b, t]
        return tilted_per_position.sum(), b * t

    @override
    def update(self, ctx: MetricContext) -> Tensor | None:
        if ctx.current_frac_of_training < self.cfg.start_frac:
            return None
        wd = ctx.weight_deltas if ctx.use_delta_component else None
        sum_loss, n_examples = self._tilted_recon_sum_and_n(ctx, wd)
        if ctx.is_eval:
            self._recon_sum_loss += sum_loss.detach()
            self._recon_n_examples += n_examples
        return sum_loss / n_examples

    @override
    def compute(self) -> MetricResult:
        out: dict[str, Float[Tensor, ""]] = {}
        if self._recon_n_examples.item() > 0:
            sum_loss = all_reduce(self._recon_sum_loss)
            n = all_reduce(self._recon_n_examples)
            out[f"{type(self).__name__}/output_recon"] = sum_loss / n
        return out
