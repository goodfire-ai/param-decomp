"""Persistent PGD: Persistent adversarial sources that evolve across training steps.

Instead of reinitializing PGD sources each training step and running N optimization steps,
PersistentPGD maintains persistent sources that receive one gradient update per training step.
Over many steps, these sources converge to strong adversarial configurations.

The key insight is that this amortizes PGD optimization across training steps - getting the
benefit of many PGD steps without the per-step computational cost.

This module owns the PPGD config types (`PersistentPGDReconLossConfig`,
`PersistentPGDReconSubsetLossConfig`, plus the nested optimizer/scope helpers) and the
`validate_pgd_scope` invariant check that the optimizer applies before training.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Annotated, Literal, override

import torch
from jaxtyping import Float, Int
from pydantic import Field, NonNegativeFloat, NonNegativeInt, PositiveInt
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.base_config import BaseConfig
from param_decomp.metrics.base import LossMetricConfig
from param_decomp.models.batch_and_loss_fns import ReconstructionLoss
from param_decomp.models.component_model import ComponentModel
from param_decomp.models.components import ComponentsMaskInfo, RoutingMasks, make_mask_infos
from param_decomp.routing import (
    AllLayersRouter,
    Router,
    SubsetRoutingType,
    UniformKSubsetRoutingConfig,
    get_subset_router,
)
from param_decomp.schedule import ScheduleConfig
from param_decomp.types import Probability
from param_decomp.utils.distributed_utils import all_reduce, broadcast_tensor
from param_decomp.utils.general_utils import get_scheduled_value

# --- PPGD config types --------------------------------------------------------------


class SignPGDConfig(BaseConfig):
    type: Literal["sign"] = "sign"
    lr_schedule: ScheduleConfig


class AdamPGDConfig(BaseConfig):
    type: Literal["adam"] = "adam"
    beta1: Probability = Field(default=0.9, description="Adam beta1 for masks")
    beta2: Probability = Field(default=0.999, description="Adam beta2 for masks")
    eps: NonNegativeFloat = Field(default=1e-8, description="Adam epsilon for masks")
    lr_schedule: ScheduleConfig


PGDOptimizerConfig = SignPGDConfig | AdamPGDConfig


class SingleSourceScope(BaseConfig):
    type: Literal["single_source"] = "single_source"


class BroadcastAcrossBatchScope(BaseConfig):
    type: Literal["broadcast_across_batch"] = "broadcast_across_batch"


class RepeatAcrossBatchScope(BaseConfig):
    type: Literal["repeat_across_batch"] = "repeat_across_batch"
    n_sources: PositiveInt


class PerBatchPerPositionScope(BaseConfig):
    type: Literal["per_batch_per_position"] = "per_batch_per_position"


PersistentPGDSourceScope = Annotated[
    SingleSourceScope
    | BroadcastAcrossBatchScope
    | RepeatAcrossBatchScope
    | PerBatchPerPositionScope,
    Field(discriminator="type"),
]


class _PersistentPGDBaseConfig(LossMetricConfig):
    """Shared fields for persistent PGD configs."""

    optimizer: Annotated[PGDOptimizerConfig, Field(discriminator="type")]
    scope: PersistentPGDSourceScope
    use_sigmoid_parameterization: bool = False
    n_warmup_steps: NonNegativeInt = Field(
        default=0,
        description=(
            "Extra inner PGD source-optimization steps on each train batch before the final loss"
            " computation."
        ),
    )
    start_frac: Probability = 0.0
    n_samples: PositiveInt = 1


class PersistentPGDReconLossConfig(_PersistentPGDBaseConfig):
    type: Literal["PersistentPGDReconLoss"] = "PersistentPGDReconLoss"


class PersistentPGDReconSubsetLossConfig(_PersistentPGDBaseConfig):
    type: Literal["PersistentPGDReconSubsetLoss"] = "PersistentPGDReconSubsetLoss"
    routing: Annotated[
        SubsetRoutingType, Field(discriminator="type", default=UniformKSubsetRoutingConfig())
    ]


# --- PPGD state ---------------------------------------------------------------------

PPGDSources = dict[str, Float[Tensor, " source_c"]]


class PPGDOptimizer(ABC):
    """Interface for persistent PGD optimizers."""

    @abstractmethod
    def init_state(self, sources: PPGDSources) -> None:
        """Initialize any optimizer-specific state for the given sources."""

    @abstractmethod
    def step(self, sources: PPGDSources, grads: PPGDSources) -> None:
        """Perform one update step on sources using gradients. Updates sources in-place."""

    @abstractmethod
    def set_lr(self, lr: float) -> None:
        """Update the learning rate / step size."""


class SignPGDOptimizer(PPGDOptimizer):
    def __init__(self, cfg: SignPGDConfig) -> None:
        self._step_size = cfg.lr_schedule.start_val

    @override
    def init_state(self, sources: PPGDSources) -> None:
        pass

    @override
    def step(self, sources: PPGDSources, grads: PPGDSources) -> None:
        for module_name in sources:
            sources[module_name].add_(self._step_size * grads[module_name].sign())

    @override
    def set_lr(self, lr: float) -> None:
        self._step_size = lr


class AdamPGDOptimizer(PPGDOptimizer):
    def __init__(self, cfg: AdamPGDConfig) -> None:
        self._lr = cfg.lr_schedule.start_val
        self._beta1 = cfg.beta1
        self._beta2 = cfg.beta2
        self._eps = cfg.eps
        self._step_count = 0
        self._m: PPGDSources = {}
        self._v: PPGDSources = {}

    @override
    def init_state(self, sources: PPGDSources) -> None:
        for module_name, source in sources.items():
            self._m[module_name] = torch.zeros_like(source)
            self._v[module_name] = torch.zeros_like(source)

    @override
    def step(self, sources: PPGDSources, grads: PPGDSources) -> None:
        self._step_count += 1
        bias_correction1 = 1 - self._beta1**self._step_count
        bias_correction2 = 1 - self._beta2**self._step_count
        for module_name, source in sources.items():
            grad = grads[module_name]
            m = self._m[module_name]
            v = self._v[module_name]
            m.mul_(self._beta1).add_(grad, alpha=1 - self._beta1)
            v.mul_(self._beta2).addcmul_(grad, grad, value=1 - self._beta2)
            m_hat = m / bias_correction1
            v_hat = v / bias_correction2
            denom = v_hat.sqrt().add_(self._eps)
            source.add_(self._lr * m_hat / denom)

    @override
    def set_lr(self, lr: float) -> None:
        self._lr = lr


def make_ppgd_optimizer(cfg: PGDOptimizerConfig) -> PPGDOptimizer:
    match cfg:
        case SignPGDConfig():
            return SignPGDOptimizer(cfg)
        case AdamPGDConfig():
            return AdamPGDOptimizer(cfg)


class PersistentPGDState:
    """Persistent state for persistent PGD optimization.

    Holds adversarial sources per module that persist across training steps.
    Source shape depends on scope: shared across batch (SingleSource, BroadcastAcrossBatch),
    repeated along batch dim (RepeatAcrossBatch), or per-batch-element-per-position with no
    cross-rank synchronization (PerBatchPerPosition).
    """

    def __init__(
        self,
        module_to_c: dict[str, int],
        batch_dims: tuple[int, ...],
        device: torch.device | str,
        use_delta_component: bool,
        cfg: PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig,
        reconstruction_loss: ReconstructionLoss,
    ) -> None:
        self.optimizer = make_ppgd_optimizer(cfg.optimizer)
        self._skip_all_reduce = isinstance(cfg.scope, PerBatchPerPositionScope)
        self._use_sigmoid_parameterization = cfg.use_sigmoid_parameterization
        self._router = _get_router_for_ppgd_config(cfg, device)
        self._n_warmup_steps = cfg.n_warmup_steps
        self._n_samples = cfg.n_samples
        self._reconstruction_loss = reconstruction_loss
        self._lr_schedule = cfg.optimizer.lr_schedule

        self.sources: PPGDSources = {}

        match cfg.scope:
            case SingleSourceScope():
                source_leading_dims = [1] * len(batch_dims)
            case BroadcastAcrossBatchScope():
                source_leading_dims = [1] + list(batch_dims[1:])
            case RepeatAcrossBatchScope(n_sources=n):
                assert batch_dims[0] % n == 0, (
                    f"n_sources={n} must divide the per-rank microbatch size "
                    f"{batch_dims[0]}, not the global batch size. "
                    f"Adjust n_sources or batch_size to satisfy this."
                )
                source_leading_dims = [n] + list(batch_dims[1:])
            case PerBatchPerPositionScope():
                source_leading_dims = list(batch_dims)

        init_fn = torch.randn if self._use_sigmoid_parameterization else torch.rand
        for module_name, module_c in module_to_c.items():
            source_c = module_c + 1 if use_delta_component else module_c
            source_shape = source_leading_dims + [source_c]
            source_data = init_fn(source_shape, device=device)
            if not self._skip_all_reduce:
                broadcast_tensor(source_data)
            self.sources[module_name] = source_data.requires_grad_(True)

        self.optimizer.init_state(self.sources)

    def get_grads(self, loss: Float[Tensor, ""], retain_graph: bool = True) -> PPGDSources:
        grads = torch.autograd.grad(loss, list(self.sources.values()), retain_graph=retain_graph)

        if self._skip_all_reduce:
            return dict(zip(self.sources.keys(), grads, strict=True))
        return {
            k: all_reduce(g, op=ReduceOp.AVG)
            for k, g in zip(self.sources.keys(), grads, strict=True)
        }

    def step(self, grads: PPGDSources) -> None:
        """Perform one PGD update step using the provided gradients.

        Updates sources in-place, then clamps to [0, 1] (or leaves unbounded when using sigmoid
        parameterization, where sigmoid is applied when reading effective sources).
        """
        with torch.no_grad():
            self.optimizer.step(self.sources, grads)

            if not self._use_sigmoid_parameterization:
                for source in self.sources.values():
                    source.clamp_(0.0, 1.0)

    def get_effective_sources(self) -> PPGDSources:
        """Return sources in [0, 1] range.

        If using sigmoid parameterization, applies sigmoid to unconstrained values. Otherwise
        returns raw sources (already clamped to [0, 1]).
        """
        if self._use_sigmoid_parameterization:
            return {k: torch.sigmoid(v) for k, v in self.sources.items()}
        return self.sources

    def update_lr(self, step: int, total_steps: int) -> None:
        lr = get_scheduled_value(step, total_steps, self._lr_schedule)
        self.optimizer.set_lr(lr)

    def warmup(
        self,
        model: ComponentModel,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    ) -> None:
        """Run extra PGD steps to refine adversarial sources before the final loss computation.

        Each step computes the recon loss, extracts gradients, and updates sources in-place.
        When n_warmup_steps=0 (default), this is a no-op.
        """
        all_layers = AllLayersRouter()
        for _ in range(self._n_warmup_steps):
            sum_loss, n = self.compute_recon_sum_and_n(
                model, batch, target_out, ci, weight_deltas, router=all_layers
            )
            grads = self.get_grads(sum_loss / n, retain_graph=False)
            self.step(grads)

    def compute_recon_sum_and_n(
        self,
        model: ComponentModel,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
        router: Router | None = None,
    ) -> tuple[Float[Tensor, ""], int]:
        """Pure forward pass returning (sum_loss, n_examples) over all mask samples.

        Returning the unreduced pair lets eval accumulators weight by example count
        across batches; training callers divide locally to get a scalar loss.
        """
        batch_dims = next(iter(ci.values())).shape[:-1]
        router = router or self._router
        ppgd_sources = self.get_effective_sources()

        device = next(iter(ci.values())).device
        sum_loss = torch.tensor(0.0, device=device)
        n_examples = 0
        for _ in range(self._n_samples):
            routing_masks = router.get_masks(
                module_names=model.target_module_paths, mask_shape=batch_dims
            )
            loss, n = _compute_ppgd_recon_loss(
                model=model,
                ppgd_sources=ppgd_sources,
                reconstruction_loss=self._reconstruction_loss,
                batch=batch,
                target_out=target_out,
                ci=ci,
                weight_deltas=weight_deltas,
                routing_masks=routing_masks,
            )
            sum_loss = sum_loss + loss
            n_examples += n
        return sum_loss, n_examples


def _get_router_for_ppgd_config(
    cfg: PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig,
    device: torch.device | str,
) -> Router:
    match cfg:
        case PersistentPGDReconLossConfig():
            return AllLayersRouter()
        case PersistentPGDReconSubsetLossConfig(routing=routing):
            return get_subset_router(routing, device)


def get_ppgd_mask_infos(
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    ppgd_sources: dict[str, Float[Tensor, "*batch_dims source_c"]],
    routing_masks: RoutingMasks,
    batch_dims: tuple[int, ...],
) -> dict[str, ComponentsMaskInfo]:
    """Get mask infos for persistent PGD."""

    expanded_adv_sources: dict[str, Float[Tensor, "*batch_dims source_c"]] = {}
    for module_name, source in ppgd_sources.items():
        B = batch_dims[0]
        N = source.shape[0]
        if N == 1 or N == B:
            expanded_adv_sources[module_name] = source.expand(*batch_dims, -1)
        else:
            assert B % N == 0, f"source leading dim {N} must divide batch dim {B}"
            repeat_dims = (B // N,) + (1,) * (source.ndim - 1)
            expanded_adv_sources[module_name] = source.repeat(*repeat_dims)

    # Split into component sources and weight delta sources
    adv_sources_components: dict[str, Float[Tensor, "*batch_dims C"]]
    weight_deltas_and_masks: (
        dict[str, tuple[Float[Tensor, "d_out d_in"], Float[Tensor, ...]]] | None
    )
    match weight_deltas:
        case None:
            weight_deltas_and_masks = None
            adv_sources_components = expanded_adv_sources
        case dict():
            weight_deltas_and_masks = {
                k: (weight_deltas[k], expanded_adv_sources[k][..., -1]) for k in weight_deltas
            }
            adv_sources_components = {k: v[..., :-1] for k, v in expanded_adv_sources.items()}

    component_masks = _interpolate_component_mask(ci, adv_sources_components)

    return make_mask_infos(
        component_masks=component_masks,
        weight_deltas_and_masks=weight_deltas_and_masks,
        routing_masks=routing_masks,
    )


def _interpolate_component_mask(
    ci: dict[str, Float[Tensor, "... C"]],
    adv_sources: dict[str, Float[Tensor, "... C"]],
) -> dict[str, Float[Tensor, "... C"]]:
    """Interpolate CI with adversarial sources: mask = ci + (1 - ci) * adv."""
    return {name: ci[name] + (1 - ci[name]) * adv_sources[name] for name in ci}


def _compute_ppgd_recon_loss(
    model: ComponentModel,
    ppgd_sources: PPGDSources,
    reconstruction_loss: ReconstructionLoss,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    target_out: Float[Tensor, "... vocab"],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    routing_masks: RoutingMasks,
) -> tuple[Float[Tensor, ""], int]:
    assert ci, "Empty ci"
    batch_dims = next(iter(ci.values())).shape[:-1]

    mask_infos = get_ppgd_mask_infos(ci, weight_deltas, ppgd_sources, routing_masks, batch_dims)
    out = model(batch, mask_infos=mask_infos)
    loss, n_examples = reconstruction_loss(pred=out, target=target_out)
    return loss, n_examples


def validate_pgd_scope(
    loss_metrics: Iterable[LossMetricConfig],
    *,
    batch_size: int,
    world_size: int,
) -> None:
    """Assert persistent-PGD `repeat_across_batch` divides the per-rank training batch size.

    Takes ``world_size`` directly (not a ``DistributedState``) so this module
    doesn't have to know about distributed plumbing. Callers pass
    ``dist_state.world_size if dist_state is not None else 1``.
    """
    assert batch_size % world_size == 0, (
        f"batch_size {batch_size} not divisible by world size {world_size}"
    )
    per_rank = batch_size // world_size
    for cfg in loss_metrics:
        if isinstance(
            cfg, PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig
        ) and isinstance(cfg.scope, RepeatAcrossBatchScope):
            n = cfg.scope.n_sources
            assert per_rank % n == 0, (
                f"{cfg.type}: repeat_across_batch n_sources={n} must divide "
                f"per-rank batch_size={per_rank}"
            )
