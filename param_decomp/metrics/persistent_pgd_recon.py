"""PPGD `Metric` subclasses and their configs.

The metric returns the live training loss and, at eval time, additionally tracks
hidden-activation MSE breakdowns. `before_backward(loss)` and `after_backward()`
orchestrate the source-grad / source-step around `total_loss.backward()`. Persistent
state + optimizer state machine live in `persistent_pgd_state`.
"""

from collections.abc import Iterable
from typing import Any, ClassVar, override

import torch
import torch.distributed as dist
from jaxtyping import Float
from torch import Tensor

from param_decomp.distributed import active_reduction_group, all_reduce, is_distributed
from param_decomp.masks import AllLayersRouter, Router, get_subset_router
from param_decomp.metrics.base import Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.persistent_pgd_state import (
    PersistentPGDState,
    PPGDSources,
    get_ppgd_mask_infos,
    scope_needs_replica_sync,
)
from param_decomp.metrics.stochastic_hidden_acts_recon import (
    calc_hidden_acts_mse,
    compute_per_module_metrics,
)
from param_decomp_config.losses import (
    LossMetricConfig,
    NSCScope,
    PersistentPGDReconLossConfig,
    PersistentPGDReconSubsetLossConfig,
)


def _router_for_cfg(
    cfg: PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig,
    device: torch.device | str,
) -> Router:
    match cfg:
        case PersistentPGDReconLossConfig():
            return AllLayersRouter()
        case PersistentPGDReconSubsetLossConfig(routing=routing):
            return get_subset_router(routing, device)


def validate_pgd_scope(
    loss_metrics: Iterable[LossMetricConfig],
    *,
    batch_size: int,
    world_size: int,
) -> None:
    """Assert persistent-PGD `repeat_across_batch` divides the per-rank training batch size.

    Takes `world_size` as an int (not a `DistributedState`) to avoid pulling distributed
    plumbing into this module.
    """
    assert batch_size % world_size == 0, (
        f"batch_size {batch_size} not divisible by world size {world_size}"
    )
    per_rank = batch_size // world_size
    for cfg in loss_metrics:
        if isinstance(
            cfg, PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig
        ) and isinstance(cfg.scope, NSCScope):
            n = cfg.scope.n_sources
            assert per_rank % n == 0, (
                f"{cfg.type}: repeat_across_batch n_sources={n} must divide "
                f"per-rank batch_size={per_rank}"
            )


class _PersistentPGDReconBase[
    TConfig: PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig
](Metric[TConfig]):
    """Shared logic between all-layers and subset PPGD recon metrics.

    Lazily constructs the `PersistentPGDState` on the first `update` so it can snapshot
    the live batch shape. Returns the live recon loss on training steps and, on eval
    batches, additionally accumulates output and per-module hidden-activation MSE for
    `compute()`. The outer optimizer loop drives source updates via `before_backward`
    and `after_backward`.
    """

    log_namespace: ClassVar[str] = "loss"
    slow: ClassVar[bool] = True
    supports_fused_kl: ClassVar[bool] = False

    def __init__(self, cfg: TConfig) -> None:
        super().__init__(cfg)
        assert not cfg.use_fused_kl or self.supports_fused_kl, (
            f"{type(self).__name__} cannot run use_fused_kl=true: the fused-KL LM-head bypass "
            f"needs the vendored-LM metric class (lab dispatch). Use a lab trainer or set "
            f"use_fused_kl: false."
        )
        self.state: PersistentPGDState | None = None
        self._pending_source_grads: PPGDSources | None = None
        # Stash from `load_state_dict` if called before the first `update()` —
        # `PersistentPGDState` needs batch_dims, which we only learn from a live ctx.
        self._pending_resume_state: dict[str, Any] | None = None

    def _ensure_state(self, ctx: MetricContext) -> None:
        if self.state is not None:
            return
        batch_dims = ctx.target_out.shape[:-1]
        # Replicated scopes keep their shared sources in lockstep across the active
        # data-parallel group: the context group set by ``use_reduction_group`` (pooled
        # eval scopes it to that pool), else the whole-world default group.
        replica_sync_group: dist.ProcessGroup | None = None
        if scope_needs_replica_sync(self.cfg.scope) and is_distributed():
            replica_sync_group = active_reduction_group() or dist.group.WORLD
        self.state = PersistentPGDState(
            module_to_c=self.model.module_to_c,
            batch_dims=batch_dims,
            device=self.device,
            use_delta_component=ctx.use_delta_component,
            optimizer_cfg=self.cfg.optimizer,
            scope=self.cfg.scope,
            use_sigmoid_parameterization=self.cfg.use_sigmoid_parameterization,
            n_warmup_steps=self.cfg.n_warmup_steps,
            n_samples=self.cfg.n_samples,
            router=_router_for_cfg(self.cfg, self.device),
            reconstruction_loss=ctx.reconstruction_loss,
            replica_sync_group=replica_sync_group,
        )
        if self._pending_resume_state is not None:
            self.state.load_state_dict(self._pending_resume_state)
            self._pending_resume_state = None

    @override
    def reset(self) -> None:
        self._recon_sum_loss = torch.zeros((), device=self.device)
        self._recon_n_examples = torch.zeros((), device=self.device, dtype=torch.long)
        self._hidden_sum_mse: dict[str, Tensor] = {}
        self._hidden_n: dict[str, Tensor] = {}

    @override
    def update(self, ctx: MetricContext) -> Tensor | None:
        if ctx.current_frac_of_training < self.cfg.start_frac:
            return None
        self._ensure_state(ctx)
        assert self.state is not None
        # The schedule is keyed on training step, so we only step it when not in eval.
        if not ctx.is_eval:
            self.state.update_lr(step=ctx.step, total_steps=ctx.total_steps)

        wd = ctx.weight_deltas if ctx.use_delta_component else None

        if not ctx.is_eval:
            self.state.warmup(
                model=self.model,
                batch=ctx.batch,
                target_out=ctx.target_out,
                ci=ctx.ci.lower_leaky,
                weight_deltas=wd,
            )

        sum_loss, n_examples = self.state.compute_recon_sum_and_n(
            model=self.model,
            batch=ctx.batch,
            target_out=ctx.target_out,
            ci=ctx.ci.lower_leaky,
            weight_deltas=wd,
        )

        if ctx.is_eval:
            self._recon_sum_loss += sum_loss.detach()
            self._recon_n_examples += n_examples
            self._accum_hidden_acts(ctx, wd)

        return sum_loss / n_examples

    def _accum_hidden_acts(
        self,
        ctx: MetricContext,
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    ) -> None:
        assert self.state is not None
        _out, target_acts = self.model.forward_with_output_acts(ctx.batch)
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

    @override
    def compute(self) -> MetricResult:
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

    @override
    def before_backward(self, live_loss: Tensor | None) -> None:
        if live_loss is None or self.state is None:
            return
        grads = self.state.get_grads(live_loss, retain_graph=True)
        self._pending_source_grads = self.state.reduce_source_grads(grads)

    @override
    def after_backward(self) -> None:
        if self._pending_source_grads is None:
            return
        assert self.state is not None
        self.state.step(self._pending_source_grads)
        self._pending_source_grads = None

    @override
    def state_dict(self) -> dict[str, Any]:
        if self.state is None:
            return {}
        return self.state.state_dict()

    @override
    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not state:
            self._pending_resume_state = None
            return
        if self.state is None:
            # `PersistentPGDState` needs batch_dims, which only arrives with the first
            # `update()` ctx. Defer the load until `_ensure_state` constructs the state.
            self._pending_resume_state = state
        else:
            self.state.load_state_dict(state)


class PersistentPGDReconLoss(_PersistentPGDReconBase[PersistentPGDReconLossConfig]):
    """PPGD adversarial-mask recon loss (routes to all layers).

    Drives components to reconstruct the target output under adversarially-optimised
    masks whose source tensors persist across training steps.
    """

    short_name = "PersistPGDRecon"


class PersistentPGDReconSubsetLoss(_PersistentPGDReconBase[PersistentPGDReconSubsetLossConfig]):
    """`PersistentPGDReconLoss` variant that masks only a routed subset of layers per forward."""

    short_name = "PersistPGDReconSub"
