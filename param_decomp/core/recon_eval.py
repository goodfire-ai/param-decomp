"""Target-generic reconstruction evaluation kernels."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jaxtyping import Array, Float, PRNGKeyArray

from param_decomp.core.adversary import Sources, init_fresh_pgd_sources
from param_decomp.core.ci_fn import PlacedCIFn, evaluate_ci
from param_decomp.core.components import ComponentStacks, SiteSpec
from param_decomp.core.decomposed_linear import constrain_component_activation
from param_decomp.core.jit_util import filter_jit
from param_decomp.core.losses import reconstruction_loss
from param_decomp.core.masking import masks_from_sources
from param_decomp.core.model import (
    CaptureKeys,
    MaterializedMasking,
    PlacedModel,
    prepare_compute_weights,
    select_captures,
)
from param_decomp.core.recon import (
    OutputOnlyReconstruction,
    ReconstructionSpec,
    hidden_acts_capture_keys,
    reconstruction_observations,
)
from param_decomp.core.sharding import batch_shard_leading

type FreshPGDStep = Callable[
    [PlacedModel, ComponentStacks, PlacedCIFn, Any, PRNGKeyArray], Float[Array, ""]
]
"""One batch's fresh-PGD reconstruction scalar. `inputs` is the target's own opaque batch
(token ids, a feature matrix, a dict of tensors) — exactly what `masked_forward` takes."""


@dataclass(frozen=True)
class FreshPGDReconEval:
    """Resolved fresh sign-PGD reconstruction probe."""

    n_steps: int
    step_size: float
    reconstruction: ReconstructionSpec = OutputOnlyReconstruction()
    name: str = "PGDReconLoss"
    """Output-key suffix; overridden only when the authored metric sets ``name``."""

    @property
    def hidden_acts_capture_keys(self) -> CaptureKeys:
        return hidden_acts_capture_keys(self.reconstruction)


def fresh_pgd_recon_sources(
    sites: tuple[SiteSpec, ...],
    ci_lower: dict[str, Array],
    leading: tuple[int, ...],
    key: PRNGKeyArray,
    fresh_pgd: FreshPGDReconEval,
    loss_at_masks: Callable[[dict[str, Array], dict[str, Array]], Array],
) -> Sources:
    """Ascend fresh sources against a caller-owned recon objective."""
    initial_sources = init_fresh_pgd_sources(sites, "random", "c", leading, key)

    def loss_at_sources(sources: Sources) -> Array:
        masks, delta_masks = masks_from_sources(ci_lower, sources)
        return loss_at_masks(masks, delta_masks)

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
    ci_lower: dict[str, Array],
    leading: tuple[int, ...],
    key: PRNGKeyArray,
    fresh_pgd: FreshPGDReconEval,
    loss_at_masks: Callable[[dict[str, Array], dict[str, Array]], Array],
) -> Array:
    """Fresh sign-PGD over generic model masks, scored by a caller-owned recon metric."""
    sources = fresh_pgd_recon_sources(sites, ci_lower, leading, key, fresh_pgd, loss_at_masks)
    masks, delta_masks = masks_from_sources(ci_lower, sources)
    return loss_at_masks(masks, delta_masks)


def make_fresh_pgd_eval_step(
    model_static: PlacedModel,
    fresh_pgd: FreshPGDReconEval,
    ci_capture_keys: CaptureKeys,
    mesh: Mesh | None = None,
    compiler_options: dict[str, bool | int | str] | None = None,
) -> FreshPGDStep:
    """Build fresh-PGD reconstruction eval for any target."""
    sites = model_static.sites
    recon_loss_fn = model_static.recon_loss_fn
    placement = model_static.placement
    leading_rank = 2 if model_static.has_position_axis else 1
    reconstruction_capture_keys = fresh_pgd.hidden_acts_capture_keys
    model_static.assert_hidden_acts_reconstruction_points(
        tuple(sorted(reconstruction_capture_keys))
    )
    clean_capture_keys = ci_capture_keys | reconstruction_capture_keys

    def shard_batch_tree[T](tree: T) -> T:
        return jax.tree.map(lambda x: batch_shard_leading(x, mesh), tree)

    def eval_step(
        model: PlacedModel,
        components: ComponentStacks,
        placed_ci_fn: PlacedCIFn,
        inputs: Any,
        key: PRNGKeyArray,
    ) -> Array:
        sharded_inputs = shard_batch_tree(inputs)
        clean_forward_result = model.clean_forward(sharded_inputs, clean_capture_keys)
        ci_input_activations = select_captures(clean_forward_result.captures, ci_capture_keys)
        clean = reconstruction_observations(
            clean_forward_result,
            hidden_acts_capture_keys=reconstruction_capture_keys,
            mesh=mesh,
        )
        leading = next(iter(ci_input_activations.values())).shape[:-1]
        assert len(leading) == leading_rank, (leading, model_static.has_position_axis)

        prepared_weights = prepare_compute_weights(model, components)
        ci_lower = {
            site: constrain_component_activation(value, placement)
            for site, value in evaluate_ci(
                placed_ci_fn, ci_input_activations, remat=False
            ).lower.items()
        }

        def loss_at_masks(masks: dict[str, Array], delta_masks: dict[str, Array]) -> Array:
            masked_forward_result = model.masked_forward(
                prepared_weights,
                sharded_inputs,
                masking=MaterializedMasking(
                    component_masks=masks,
                    weight_delta_masks=delta_masks,
                ),
                capture_keys=reconstruction_capture_keys,
                remat=False,
            )
            masked = reconstruction_observations(
                masked_forward_result,
                hidden_acts_capture_keys=reconstruction_capture_keys,
                mesh=mesh,
            )
            return reconstruction_loss(
                recon_loss_fn,
                masked=masked,
                clean=clean,
                reconstruction=fresh_pgd.reconstruction,
            ).total

        return fresh_pgd_recon_loss(sites, ci_lower, leading, key, fresh_pgd, loss_at_masks)

    return filter_jit(eval_step, compiler_options=compiler_options)
