"""Fused-KL PPGD recon metrics for the vendored-LM (flat FSDP) path.

Same `type` literals as the core PPGD metrics — the lab dispatch table overrides core
by class name, and with `use_fused_kl: false` these behave exactly like core. With
`use_fused_kl: true`, the whole PPGD phase — the clean target forward, every warmup
ascent forward, and the live recon forward — runs under the vendored model's
`bypass_lm_head`, and the recon is the fused linear+KL kernel. This mirrors the 3-pool
PPGD pool's regime (`three_pool/step_ppgd.py` wraps the phase in `strategy.context()`)
and never materializes a vocab-scale tensor (at Llama vocab 128k that is ~7.8 GiB fp32
per forward at per-rank batch 8 — the bl8 OOM site).

The bypass target is recomputed here (one no-grad suffix forward) because
`ctx.target_out` is the unbypassed full-logits forward and cannot serve as the fused
target — same trade `ChunkwiseSubsetReconLoss` makes.
"""

from dataclasses import replace
from typing import ClassVar, override

import torch
from torch import Tensor

from param_decomp.component_model import ComponentModelProtocol
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.persistent_pgd_recon import _PersistentPGDReconBase
from param_decomp_config.losses import (
    PersistentPGDReconLossConfig,
    PersistentPGDReconSubsetLossConfig,
)
from param_decomp_lab.metrics.chunkwise_subset_recon import _as_lm_component_model
from param_decomp_lab.three_pool.recon_loss_strategy import ReconLossStrategy


class _FusedPersistentPGDReconBase[
    TConfig: PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig
](_PersistentPGDReconBase[TConfig]):
    supports_fused_kl: ClassVar[bool] = True

    @override
    def bind(self, *, model: ComponentModelProtocol, device: str) -> None:
        super().bind(model=model, device=device)
        self._strategy = (
            ReconLossStrategy.fused(_as_lm_component_model(model))
            if self.cfg.use_fused_kl
            else None
        )

    @property
    @override
    def needs_target_out(self) -> bool:
        return not self.cfg.use_fused_kl

    @override
    def update(self, ctx: MetricContext) -> Tensor | None:
        if self._strategy is None:
            return super().update(ctx)
        with self._strategy.context():
            with torch.no_grad():
                target_hidden = self.model(ctx.batch)
            assert isinstance(target_hidden, Tensor)
            fused_ctx = replace(
                ctx,
                target_out=target_hidden.detach(),
                reconstruction_loss=self._strategy.recon_loss,
            )
            return super().update(fused_ctx)


class PersistentPGDReconLoss(_FusedPersistentPGDReconBase[PersistentPGDReconLossConfig]):
    short_name = "PersistPGDRecon"


class PersistentPGDReconSubsetLoss(
    _FusedPersistentPGDReconBase[PersistentPGDReconSubsetLossConfig]
):
    short_name = "PersistPGDReconSub"
