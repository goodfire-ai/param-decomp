"""Persistent PGD reconstruction metric.

Merges what used to be two separate metrics (PersistentPGDReconLoss + PPGDReconEval): the same
metric instance owns the persistent adversarial state, returns the live training loss, and at
eval time additionally tracks hidden-activation MSE breakdowns. The optimizer loop's
`before_backward(loss)` and `after_backward()` hooks orchestrate the source-grad / source-step
that needs to bracket `total_loss.backward()`.
"""

from typing import Any, ClassVar

import torch
from jaxtyping import Float
from torch import Tensor

from param_decomp.configs import (
    PersistentPGDReconLossConfig,
    PersistentPGDReconSubsetLossConfig,
    _PersistentPGDBaseConfig,
)
from param_decomp.metrics.builtin.hidden_acts_recon_loss import (
    calc_hidden_acts_mse,
    compute_per_module_metrics,
)
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.registry import register_metric
from param_decomp.models.component_model import ComponentModel
from param_decomp.persistent_pgd import PersistentPGDState, get_ppgd_mask_infos
from param_decomp.utils.distributed_utils import all_reduce


class _PersistentPGDReconBase:
    """Shared logic between all-layers and subset PPGD recon metrics."""

    section: ClassVar[str] = "loss"
    slow: ClassVar[bool] = True

    def __init__(
        self, cfg: _PersistentPGDBaseConfig, *, model: ComponentModel, device: str
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device
        self.state: PersistentPGDState | None = None
        self._pending_source_grads: Any = None
        self.reset()

    def _ensure_state(self, ctx: MetricContext) -> None:
        if self.state is not None:
            return
        batch_dims = ctx.target_out.shape[:-1]
        # cfg is one of PersistentPGDReconLossConfig / PersistentPGDReconSubsetLossConfig (the two
        # concrete subclasses); the type is fixed in each registered metric class below.
        assert isinstance(
            self.cfg, PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig
        )
        self.state = PersistentPGDState(
            module_to_c=self.model.module_to_c,
            batch_dims=batch_dims,
            device=self.device,
            use_delta_component=ctx.config.use_delta_component,
            cfg=self.cfg,
            reconstruction_loss=ctx.reconstruction_loss,
        )

    def reset(self) -> None:
        self._recon_sum_loss = torch.zeros((), device=self.device)
        self._recon_n_examples = torch.zeros((), device=self.device, dtype=torch.long)
        self._hidden_sum_mse: dict[str, Tensor] = {}
        self._hidden_n: dict[str, Tensor] = {}

    def update(self, ctx: MetricContext) -> Tensor | None:
        if ctx.current_frac_of_training < self.cfg.start_frac:
            return None
        self._ensure_state(ctx)
        assert self.state is not None
        # The optimizer-loop calls `update_lr` once per step in run_pd.py today.
        # The schedule is keyed on training step, so we only step it when not in eval.
        if not ctx.is_eval:
            self.state.update_lr(step=ctx.step, total_steps=ctx.config.steps)

        wd = ctx.weight_deltas if ctx.config.use_delta_component else None

        if not ctx.is_eval:
            self.state.warmup(
                model=self.model,
                batch=ctx.batch,
                target_out=ctx.target_out,
                ci=ctx.ci.lower_leaky,
                weight_deltas=wd,
            )

        loss = self.state.compute_recon_loss(
            model=self.model,
            batch=ctx.batch,
            target_out=ctx.target_out,
            ci=ctx.ci.lower_leaky,
            weight_deltas=wd,
        )

        if ctx.is_eval:
            self._recon_sum_loss += loss.detach()
            self._recon_n_examples += 1
            self._accum_hidden_acts(ctx, wd)

        return loss

    def _accum_hidden_acts(
        self,
        ctx: MetricContext,
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    ) -> None:
        assert self.state is not None
        target_acts = self.model(ctx.batch, cache_type="output").cache
        batch_dims = ctx.target_out.shape[:-1]
        mask_infos = get_ppgd_mask_infos(
            ci=ctx.ci.lower_leaky,
            weight_deltas=weight_deltas,
            ppgd_sources=self.state.get_effective_sources(),
            routing_masks="all",
            batch_dims=batch_dims,
        )
        per_module, _ = calc_hidden_acts_mse(
            model=self.model, batch=ctx.batch, mask_infos=mask_infos, target_acts=target_acts
        )
        for key, (mse, n) in per_module.items():
            if key not in self._hidden_sum_mse:
                self._hidden_sum_mse[key] = torch.zeros((), device=self.device)
                self._hidden_n[key] = torch.zeros((), device=self.device, dtype=torch.long)
            self._hidden_sum_mse[key] += mse.detach()
            self._hidden_n[key] += n

    def compute(self) -> dict[str, Float[Tensor, ""]]:
        out: dict[str, Float[Tensor, ""]] = {}
        if self._hidden_sum_mse:
            class_name = f"{type(self).__name__}/hidden_acts"
            out.update(
                compute_per_module_metrics(
                    class_name=class_name,
                    per_module_sum_mse=self._hidden_sum_mse,
                    per_module_n_examples=self._hidden_n,
                )
            )
        if self._recon_n_examples.item() > 0:
            sum_loss = all_reduce(self._recon_sum_loss)
            n = all_reduce(self._recon_n_examples)
            out[f"{type(self).__name__}/output_recon"] = sum_loss / n
        return out

    def before_backward(self, live_loss: Tensor | None) -> None:
        if live_loss is None or self.state is None:
            return
        self._pending_source_grads = self.state.get_grads(live_loss, retain_graph=True)

    def after_backward(self) -> None:
        if self._pending_source_grads is None:
            return
        assert self.state is not None
        self.state.step(self._pending_source_grads)
        self._pending_source_grads = None


@register_metric
class PersistentPGDReconLoss(_PersistentPGDReconBase):
    """Persistent PGD adversarial-mask reconstruction loss (routes to all layers)."""

    config_type = PersistentPGDReconLossConfig
    short_name = "PersistPGDRecon"


@register_metric
class PersistentPGDReconSubsetLoss(_PersistentPGDReconBase):
    """Persistent PGD adversarial-mask reconstruction loss (subset routing)."""

    config_type = PersistentPGDReconSubsetLossConfig
    short_name = "PersistPGDReconSub"
