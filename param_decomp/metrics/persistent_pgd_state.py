"""Persistent PGD state machine.

Owns per-step adversarial source tensors, the optimizer that updates them, and the
recon-forward used to score them. The metric layer (`persistent_pgd_recon.py`) composes
these primitives.
"""

from abc import ABC, abstractmethod
from typing import Any, override

import torch
import torch.distributed as dist
from jaxtyping import Float, Int
from torch import Tensor

from param_decomp.batch_and_loss_fns import ReconstructionLoss
from param_decomp.component_model import ComponentModelProtocol
from param_decomp.masks import (
    AllLayersRouter,
    ComponentsMaskInfo,
    Router,
    RoutingMasks,
    interpolate_component_mask,
    make_mask_infos,
)
from param_decomp_config.losses import (
    AdamPGDConfig,
    BSCScope,
    CScope,
    NSCScope,
    PersistentPGDSourceScope,
    PGDOptimizerConfig,
    SCScope,
    SignPGDConfig,
)
from param_decomp_config.schedule import get_scheduled_value

PPGDSources = dict[str, Float[Tensor, " source_c"]]


def scope_needs_replica_sync(scope: PersistentPGDSourceScope) -> bool:
    """Whether data-parallel replicas of this scope's sources must be kept in sync.

    Replicated scopes (single / broadcast / repeat) hold the same sources on every
    replica, so splitting the batch across DP ranks needs broadcast-init + grad
    reduction to keep them identical. ``per_batch_per_position`` sources are
    independent per element, so a batch split is just slicing — no sync needed.
    """
    return not isinstance(scope, BSCScope)


class PPGDOptimizer(ABC):
    """Interface for persistent PGD optimizers."""

    @abstractmethod
    def init_state(self, sources: PPGDSources) -> None: ...

    @abstractmethod
    def step(self, sources: PPGDSources, grads: PPGDSources) -> None:
        """One update step on `sources` in-place using `grads`."""

    @abstractmethod
    def set_lr(self, lr: float) -> None: ...

    def state_dict(self) -> dict[str, Any]:
        """Return trajectory-dependent optimizer state.

        Default empty — stateless optimizers (e.g. SignPGD) need nothing. Override
        in optimizers that carry momentum or step counts across training steps.
        """
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:  # noqa: B027 — intentional no-op default
        """Restore optimizer state produced by a prior :meth:`state_dict` call."""
        del state


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

    @override
    def state_dict(self) -> dict[str, Any]:
        return {
            "step_count": self._step_count,
            "m": dict(self._m),
            "v": dict(self._v),
        }

    @override
    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._step_count = state["step_count"]
        with torch.no_grad():
            for k, t in state["m"].items():
                self._m[k].copy_(t.to(self._m[k].device))
            for k, t in state["v"].items():
                self._v[k].copy_(t.to(self._v[k].device))


def make_ppgd_optimizer(cfg: PGDOptimizerConfig) -> PPGDOptimizer:
    match cfg:
        case SignPGDConfig():
            return SignPGDOptimizer(cfg)
        case AdamPGDConfig():
            return AdamPGDOptimizer(cfg)


class PersistentPGDState:
    """Per-module adversarial sources that persist across training steps.

    Source shape depends on scope (`CScope`, `SCScope`,
    `NSCScope`, `BSCScope`).
    """

    def __init__(
        self,
        *,
        module_to_c: dict[str, int],
        batch_dims: tuple[int, ...],
        device: torch.device | str,
        use_delta_component: bool,
        optimizer_cfg: PGDOptimizerConfig,
        scope: PersistentPGDSourceScope,
        use_sigmoid_parameterization: bool,
        n_warmup_steps: int,
        n_samples: int,
        router: Router,
        reconstruction_loss: ReconstructionLoss,
        replica_sync_group: dist.ProcessGroup | None,
    ) -> None:
        """``replica_sync_group`` keeps data-parallel replicas of a *shared* source
        (single / broadcast / repeat scope, ``scope_needs_replica_sync``) in lockstep
        across the ranks of that group: every rank broadcast-inits from the group's
        first rank, and every source-grad is AVG-reduced over the group (in ``warmup``
        and via ``reduce_source_grads`` for the final step). Identical init + a
        deterministic AVG + an identical optimizer step keeps the sources bit-identical
        without per-step re-broadcast. Must be ``None`` for ``per_batch_per_position``
        (independent per-element sources need no sync) and for single-process runs."""
        if replica_sync_group is not None:
            assert scope_needs_replica_sync(scope), (
                f"replica_sync_group passed for scope {type(scope).__name__}, which holds "
                f"independent per-element sources and must not be synced across replicas."
            )
        self._replica_sync_group = replica_sync_group
        self.optimizer = make_ppgd_optimizer(optimizer_cfg)
        self._use_sigmoid_parameterization = use_sigmoid_parameterization
        self._router = router
        self._n_warmup_steps = n_warmup_steps
        self._n_samples = n_samples
        self._reconstruction_loss = reconstruction_loss
        self._lr_schedule = optimizer_cfg.lr_schedule

        self.sources: PPGDSources = {}

        match scope:
            case CScope():
                source_leading_dims = [1] * len(batch_dims)
            case SCScope():
                source_leading_dims = [1] + list(batch_dims[1:])
            case NSCScope(n_sources=n):
                assert batch_dims[0] % n == 0, (
                    f"n_sources={n} must divide the per-rank microbatch size "
                    f"{batch_dims[0]}, not the global batch size. "
                    f"Adjust n_sources or batch_size to satisfy this."
                )
                source_leading_dims = [n] + list(batch_dims[1:])
            case BSCScope():
                source_leading_dims = list(batch_dims)

        init_fn = torch.randn if use_sigmoid_parameterization else torch.rand
        for module_name, module_c in module_to_c.items():
            source_c = module_c + 1 if use_delta_component else module_c
            source_shape = source_leading_dims + [source_c]
            source_data = init_fn(source_shape, device=device)
            self.sources[module_name] = source_data.requires_grad_(True)

        # A shared source must start identical on every replica; each rank inits its own
        # above, so broadcast the group's first rank's value onto the rest before training.
        if self._replica_sync_group is not None:
            src_rank = dist.get_global_rank(self._replica_sync_group, 0)
            with torch.no_grad():
                for source in self.sources.values():
                    dist.broadcast(source, src=src_rank, group=self._replica_sync_group)

        self.optimizer.init_state(self.sources)

    def get_grads(self, loss: Float[Tensor, ""]) -> PPGDSources:
        """Raw per-module ``∂loss/∂source``, freeing the graph. No cross-rank reduction —
        apply ``reduce_source_grads`` for replicated scopes (see ``scope_needs_replica_sync``)."""
        grads = torch.autograd.grad(loss, list(self.sources.values()))
        return dict(zip(self.sources.keys(), grads, strict=True))

    def reduce_source_grads(self, grads: PPGDSources) -> PPGDSources:
        """AVG-reduce per-rank source grads over the replica-sync group, else a no-op.

        Each rank's grad is ``∂(sum_loss_local / n_examples_local)/∂source`` (the per-rank
        recon mean). For a source shared across the global batch, uniform batch slicing
        makes ``mean_r(sum_loss_local / n_examples_local) == sum_loss_global / n_examples_global``,
        so AVG over the group reproduces the shared-source full-batch grad. Combined with
        identical init + an identical optimizer step, replicas stay in lockstep."""
        if self._replica_sync_group is None:
            return grads
        for grad in grads.values():
            dist.all_reduce(grad, op=dist.ReduceOp.AVG, group=self._replica_sync_group)
        return grads

    def step(self, grads: PPGDSources) -> None:
        """One PGD update step using `grads`.

        Sources are clamped to `[0, 1]` after, unless sigmoid parameterization is on
        (then left unbounded and sigmoid is applied when reading effective sources).
        """
        with torch.no_grad():
            self.optimizer.step(self.sources, grads)

            if not self._use_sigmoid_parameterization:
                for source in self.sources.values():
                    source.clamp_(0.0, 1.0)

    def get_effective_sources(self) -> PPGDSources:
        """Sources in `[0, 1]` range.

        Under sigmoid parameterization, applies sigmoid to unconstrained values;
        otherwise returns the raw clamped sources.
        """
        if self._use_sigmoid_parameterization:
            return {k: torch.sigmoid(v) for k, v in self.sources.items()}
        return self.sources

    def update_lr(self, step: int, total_steps: int) -> None:
        lr = get_scheduled_value(step, total_steps, self._lr_schedule)
        self.optimizer.set_lr(lr)

    def state_dict(self) -> dict[str, Any]:
        """Round-trip the persistent adversary trajectory: sources + optimizer state."""
        return {
            "sources": {k: v.detach() for k, v in self.sources.items()},
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore sources + optimizer state in-place. Shapes must already match."""
        with torch.no_grad():
            for k, src in self.sources.items():
                src.copy_(state["sources"][k].to(src.device))
        self.optimizer.load_state_dict(state["optimizer"])

    def warmup(
        self,
        model: ComponentModelProtocol,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    ) -> None:
        """Run ``n_warmup_steps`` supplemental source-only PGD iterations.

        "Warmup" is the historical name; semantically these are **supplemental**
        PGD source-update iterations that happen before the final fwd+bwd of
        the training step. The final iter (the metric's ``after_backward`` on the
        flat path, ``param_decomp_lab/three_pool/step_ppgd.py`` on the pool path)
        additionally backprops to V/U + CI and uses its own source gradient to
        apply one more PGD step. So total source updates per training step =
        ``n_warmup_steps + 1``.

        Each supplemental iter: fwd through the masked target → source-only
        backward (graph freed; it is rebuilt next iter from the updated sources) →
        ``reduce_source_grads`` (AVG over the replica-sync group for a shared
        source; a no-op otherwise) → PGD step.
        """
        # WARNING: warmup ascends against the route-ALL forward regardless of the
        # configured router. A no-op for the base (route-everywhere) variant — i.e.
        # every production run to date — but for the Subset variant this means
        # sources are warmed against an objective they are never scored on (the
        # loss forward uses the configured router). Inherited from the warmup
        # centralization in #486, not a documented decision; parameterize before
        # relying on Subset PPGD.
        all_layers = AllLayersRouter()
        for _ in range(self._n_warmup_steps):
            sum_loss, n = self.compute_recon_sum_and_n(
                model, batch, target_out, ci, weight_deltas, router=all_layers
            )
            grads = self.reduce_source_grads(self.get_grads(sum_loss / n))
            self.step(grads)

    def compute_recon_sum_and_n(
        self,
        model: ComponentModelProtocol,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
        router: Router | None = None,
    ) -> tuple[Float[Tensor, ""], int]:
        """Recon forward returning `(sum_loss, n_examples)` over all mask samples.

        Returning the unreduced pair lets eval accumulators weight by example count
        across batches.
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


def get_ppgd_mask_infos(
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    ppgd_sources: dict[str, Float[Tensor, "*batch_dims source_c"]],
    routing_masks: RoutingMasks,
    batch_dims: tuple[int, ...],
) -> dict[str, ComponentsMaskInfo]:
    """Build per-module mask infos from PPGD sources, CI values, and routing masks.

    Expands sources to match the per-batch shape (broadcasting or repeating), splits off
    the weight-delta source channel when present, and interpolates
    `mask = ci + (1 - ci) * source`.
    """

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

    component_masks = interpolate_component_mask(ci, adv_sources_components)

    return make_mask_infos(
        component_masks=component_masks,
        weight_deltas_and_masks=weight_deltas_and_masks,
        routing_masks=routing_masks,
    )


def _compute_ppgd_recon_loss(
    model: ComponentModelProtocol,
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
