"""Target-generic reconstruction evaluation kernels."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, PRNGKeyArray

from param_decomp.core.adversary import init_fresh_pgd_sources, source_masks
from param_decomp.core.ci_fn import CIFn, protect_ci
from param_decomp.core.components import ComponentStacks, SiteSpec
from param_decomp.core.jit_util import filter_jit
from param_decomp.core.model import DecomposedModel
from param_decomp.core.sharding import batch_shard_leading
from param_decomp.core.train import COMPUTE_DT, cast_floating

type FreshPGDStep = Callable[
    [DecomposedModel, ComponentStacks, CIFn, Any, PRNGKeyArray], Float[Array, ""]
]
"""One batch's fresh-PGD reconstruction scalar. `inputs` is the target's own opaque batch
(token ids, a feature matrix, a dict of tensors) — exactly what `masked_output` takes."""


type FreshPGDComponentGradStep = Callable[
    [
        DecomposedModel,
        ComponentStacks,
        CIFn,
        Any,
        PRNGKeyArray,
        dict[str, Array] | None,
    ],
    tuple[Float[Array, ""], ComponentStacks],
]
"""Fresh-PGD scalar plus its gradient with respect to V/U, with optional protected CI
slots. Used only by settle-gated lifecycle probes, never by the ordinary train step."""


@dataclass(frozen=True)
class FreshPGDReconEval:
    """Resolved fresh sign-PGD reconstruction probe."""

    n_steps: int
    step_size: float
    name: str = "PGDReconLoss"
    """Output-key suffix; overridden only when the authored metric sets ``name``."""


def fresh_pgd_recon_loss(
    sites: tuple[SiteSpec, ...],
    ci_lower: dict[str, Array],
    leading: tuple[int, ...],
    key: PRNGKeyArray,
    fresh_pgd: FreshPGDReconEval,
    loss_at_masks: Callable[[dict[str, Array], dict[str, Array]], Array],
) -> Array:
    """Fresh sign-PGD over generic model masks, scored by a caller-owned recon metric."""
    initial_sources = init_fresh_pgd_sources(sites, "random", "c", leading, key)
    site_names = tuple(site.name for site in sites)

    def loss_at_sources(sources: dict[str, Array]) -> Array:
        masks, delta_masks = source_masks(ci_lower, sources, site_names)
        return loss_at_masks(masks, delta_masks)

    def ascend(sources: dict[str, Array], _: None) -> tuple[dict[str, Array], None]:
        gradients = jax.grad(loss_at_sources)(sources)
        return {
            site: jnp.clip(
                sources[site] + fresh_pgd.step_size * jnp.sign(gradients[site]), 0.0, 1.0
            )
            for site in sources
        }, None

    final_sources, _ = jax.lax.scan(ascend, initial_sources, None, length=fresh_pgd.n_steps)
    # The attack is an inner argmax, not a differentiable optimizer. Stopping its selected
    # sources is value-preserving for ordinary eval and lets the lifecycle gradient probe
    # take a first-order envelope gradient instead of tracing accidental second derivatives
    # through sign-PGD.
    return loss_at_sources(jax.lax.stop_gradient(final_sources))


def make_fresh_pgd_eval_step(
    model_static: DecomposedModel,
    fresh_pgd: FreshPGDReconEval,
    mesh: Mesh | None,
    compiler_options: dict[str, bool | int | str],
) -> FreshPGDStep:
    """Build fresh-PGD reconstruction eval for ANY target.

    Generic over all three model edges: the input rides through as the target's opaque
    pytree, the output is only ever handed to the target's own `recon_loss_fn`, and both
    waist geometries are served — `has_position_axis` fixes the leading rank the CI taps
    must present.
    """
    sites = model_static.sites
    site_names = model_static.site_names
    recon_loss_fn = model_static.recon_loss_fn
    leading_rank = 2 if model_static.has_position_axis else 1

    def batch_sharded[T](tree: T) -> T:
        return jax.tree.map(lambda x: batch_shard_leading(x, mesh), tree)

    def ci_shard(x: Array) -> Array:
        if mesh is None:
            return x
        return jax.lax.with_sharding_constraint(
            x, NamedSharding(mesh, P(("replicate", "fsdp"), *((None,) * (x.ndim - 1))))
        )

    def eval_step(
        model: DecomposedModel,
        components: ComponentStacks,
        ci_fn: CIFn,
        inputs: Any,
        key: PRNGKeyArray,
    ) -> Array:
        inputs = batch_sharded(inputs)
        clean_output, taps = model.clean_output_and_activations(inputs, ci_fn.input_names)
        clean_output = batch_sharded(clean_output)
        leading = next(iter(taps.values())).shape[:-1]
        assert len(leading) == leading_rank, (leading, model_static.has_position_axis)

        prepared = model.prepare_compute_weights(cast_floating(components, COMPUTE_DT))
        ci_lower = {
            site: ci_shard(value)
            for site, value in cast_floating(ci_fn, COMPUTE_DT)(taps, remat=False).lower.items()
        }

        def loss_at_masks(masks: dict[str, Array], delta_masks: dict[str, Array]) -> Array:
            masked = model.masked_output(
                prepared,
                inputs,
                masks,
                delta_masks,
                None,
                site_names,
                True,
                remat=False,
            )
            return recon_loss_fn(batch_sharded(masked), clean_output)

        return fresh_pgd_recon_loss(sites, ci_lower, leading, key, fresh_pgd, loss_at_masks)

    return filter_jit(eval_step, compiler_options=compiler_options)


def make_fresh_pgd_component_grad_step(
    model_static: DecomposedModel,
    fresh_pgd: FreshPGDReconEval,
    mesh: Mesh | None,
    compiler_options: dict[str, bool | int | str],
) -> FreshPGDComponentGradStep:
    """Build the first-order V/U gradient of the authored fresh-PGD referee.

    A protected null slot can alternate ``(V=v,U=0)`` and ``(V=0,U=q)`` between calls;
    its returned factor gradients are respectively ``v^T G`` and ``G q`` for the implicit
    represented-matrix gradient ``G``. This supplies GradMax power iteration without
    materializing a dense per-site ``G`` or adding target-specific model plumbing.
    """
    sites = model_static.sites
    site_names = model_static.site_names
    recon_loss_fn = model_static.recon_loss_fn
    leading_rank = 2 if model_static.has_position_axis else 1

    def batch_sharded[T](tree: T) -> T:
        return jax.tree.map(lambda x: batch_shard_leading(x, mesh), tree)

    def ci_shard(x: Array) -> Array:
        if mesh is None:
            return x
        return jax.lax.with_sharding_constraint(
            x, NamedSharding(mesh, P(("replicate", "fsdp"), *((None,) * (x.ndim - 1))))
        )

    def grad_step(
        model: DecomposedModel,
        components: ComponentStacks,
        ci_fn: CIFn,
        inputs: Any,
        key: PRNGKeyArray,
        protected: dict[str, Array] | None,
    ) -> tuple[Array, ComponentStacks]:
        inputs = batch_sharded(inputs)
        clean_output, taps = model.clean_output_and_activations(inputs, ci_fn.input_names)
        clean_output = batch_sharded(clean_output)
        leading = next(iter(taps.values())).shape[:-1]
        assert len(leading) == leading_rank, (leading, model_static.has_position_axis)
        ci_lower = {
            site: ci_shard(value)
            for site, value in protect_ci(
                cast_floating(ci_fn, COMPUTE_DT)(taps, remat=False), protected
            ).lower.items()
        }

        def loss(candidate: ComponentStacks) -> Array:
            prepared = model.prepare_compute_weights(cast_floating(candidate, COMPUTE_DT))

            def loss_at_masks(masks: dict[str, Array], delta_masks: dict[str, Array]) -> Array:
                masked = model.masked_output(
                    prepared,
                    inputs,
                    masks,
                    delta_masks,
                    None,
                    site_names,
                    True,
                    remat=False,
                )
                return recon_loss_fn(batch_sharded(masked), clean_output)

            return fresh_pgd_recon_loss(sites, ci_lower, leading, key, fresh_pgd, loss_at_masks)

        return jax.value_and_grad(loss)(components)

    return filter_jit(grad_step, compiler_options=compiler_options)
