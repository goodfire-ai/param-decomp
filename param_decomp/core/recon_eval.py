"""Target-generic reconstruction evaluation kernels."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jaxtyping import Array, Float, PRNGKeyArray

from param_decomp.core.adversary import Sources, init_fresh_pgd_sources
from param_decomp.core.ci_fn.interface import CIFn
from param_decomp.core.ci_fn.runtime import evaluate_ci_from_captures
from param_decomp.core.components import ComponentStacks, SiteCI, SiteSpec
from param_decomp.core.decomposed_linear import constrain_component_activation
from param_decomp.core.losses import reconstruction_loss
from param_decomp.core.masking import materialize_masking, source_masking
from param_decomp.core.model import (
    CaptureKeys,
    ComponentActivations,
    MaterializedMasking,
    PlacedModel,
    select_captures,
)
from param_decomp.core.recon import (
    AuxiliaryReconstructionSpec,
    auxiliary_capture_keys,
    reconstruction_observations,
)
from param_decomp.core.sharding import batch_shard_leading

type FreshPGDStep[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
] = Callable[
    [
        PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
        ComponentStacks,
        CIFn[Conditioning],
        TargetIn,
        PRNGKeyArray,
    ],
    Float[Array, ""],
]
"""One batch's fresh-PGD reconstruction scalar. `inputs` is the target's own opaque batch
(token ids, a feature matrix, a dict of tensors) — exactly what `masked_forward` takes."""


@dataclass(frozen=True)
class FreshPGDReconEval:
    """Resolved fresh sign-PGD reconstruction probe."""

    n_steps: int
    step_size: float
    reconstruction: AuxiliaryReconstructionSpec = ()
    name: str = "PGDReconLoss"
    """Output-key suffix; overridden only when the authored metric sets ``name``."""

    @property
    def reconstruction_capture_keys(self) -> CaptureKeys:
        return auxiliary_capture_keys(self.reconstruction)


def fresh_pgd_recon_sources(
    sites: tuple[SiteSpec, ...],
    ci_lower: Mapping[str, SiteCI],
    leading: tuple[int, ...],
    key: PRNGKeyArray,
    fresh_pgd: FreshPGDReconEval,
    loss_at_masking: Callable[[MaterializedMasking], Array],
) -> Sources:
    """Ascend fresh sources against a caller-owned recon objective."""
    initial_sources = init_fresh_pgd_sources(sites, "random", "c", leading, key)

    def loss_at_sources(sources: Sources) -> Array:
        return loss_at_masking(materialize_masking(source_masking(ci_lower, sources)))

    def ascend(sources: Sources, _: None) -> tuple[Sources, None]:
        gradients = jax.grad(loss_at_sources)(sources)
        return jax.tree.map(
            lambda source, gradient: jnp.clip(
                source + fresh_pgd.step_size * jnp.sign(gradient), 0.0, 1.0
            ),
            sources,
            gradients,
        ), None

    final_sources, _ = jax.lax.scan(ascend, initial_sources, None, length=fresh_pgd.n_steps)
    return final_sources


def fresh_pgd_recon_loss(
    sites: tuple[SiteSpec, ...],
    ci_lower: Mapping[str, SiteCI],
    leading: tuple[int, ...],
    key: PRNGKeyArray,
    fresh_pgd: FreshPGDReconEval,
    loss_at_masking: Callable[[MaterializedMasking], Array],
) -> Array:
    """Fresh sign-PGD over generic model masks, scored by a caller-owned recon metric."""
    sources = fresh_pgd_recon_sources(sites, ci_lower, leading, key, fresh_pgd, loss_at_masking)
    return loss_at_masking(materialize_masking(source_masking(ci_lower, sources)))


def make_fresh_pgd_eval_step[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    model_static: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    fresh_pgd: FreshPGDReconEval,
    ci_capture_keys: CaptureKeys,
    mesh: Mesh | None = None,
) -> FreshPGDStep[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT]:
    """Build fresh-PGD reconstruction eval for any target. Only static topology and the
    target's array-free output operations close over the factory; `model` is the jit ARG."""
    sites = model_static.model.sites
    recon_loss_fn = model_static.recon_loss_fn
    pin_output_batch = model_static.pin_output_batch
    placement = model_static.placement
    leading_rank = 2 if model_static.model.has_position_axis else 1
    reconstruction_capture_keys = fresh_pgd.reconstruction_capture_keys
    clean_capture_keys = ci_capture_keys | reconstruction_capture_keys

    def shard_batch_tree[T](tree: T) -> T:
        return jax.tree.map(lambda x: batch_shard_leading(x, mesh), tree)

    def eval_step(
        model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
        components: ComponentStacks,
        ci_fn: CIFn[Conditioning],
        inputs: TargetIn,
        key: PRNGKeyArray,
    ) -> Array:
        sharded_inputs = shard_batch_tree(inputs)
        clean_forward_result = model.clean_forward(sharded_inputs, clean_capture_keys)
        ci_input_activations = select_captures(clean_forward_result.captures, ci_capture_keys)
        clean = reconstruction_observations(
            clean_forward_result,
            pin_output_batch,
            capture_keys=reconstruction_capture_keys,
            mesh=mesh,
        )
        leading = clean_forward_result.leading_shape
        assert len(leading) == leading_rank, (leading, model_static.model.has_position_axis)

        prepared_weights = model.prepare_compute_weights(components)
        ci_lower = {
            site: constrain_component_activation(
                value, None if placement is None else placement.activations.component
            )
            for site, value in evaluate_ci_from_captures(
                ci_fn.prepare(),
                ci_input_activations,
                clean_forward_result.conditioning,
                prepared_weights,
                sequence=clean_forward_result.sequence,
                remat=False,
            ).lower.items()
        }

        def loss_at_masking(masking: MaterializedMasking) -> Array:
            masked_forward_result = model.masked_forward(
                prepared_weights,
                clean_forward_result.conditioning,
                masking=model.model.prepare_masking(masking),
                routes=None,
                capture_keys=reconstruction_capture_keys,
                remat=False,
            )
            masked = reconstruction_observations(
                masked_forward_result,
                pin_output_batch,
                capture_keys=reconstruction_capture_keys,
                mesh=mesh,
            )
            return reconstruction_loss(
                recon_loss_fn,
                masked=masked,
                clean=clean,
                reconstruction=fresh_pgd.reconstruction,
            ).total

        return fresh_pgd_recon_loss(sites, ci_lower, leading, key, fresh_pgd, loss_at_masking)

    return eval_step
