"""Target-generic reconstruction evaluation kernels."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from operator import itemgetter

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jaxtyping import Array, Int32, PRNGKeyArray

from param_decomp.core.adversary import Sources, init_fresh_pgd_sources
from param_decomp.core.ci_fn.interface import CIFn
from param_decomp.core.ci_fn.runtime import evaluate_ci_from_captures
from param_decomp.core.components import ComponentStacks, SiteCI, SiteSpec
from param_decomp.core.configs import AnyPGDEvalConfig, AnyPGDEvalType
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
    resolve_auxiliary_reconstruction,
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
    dict[str, Array],
]
"""One batch's fresh-PGD reconstruction scalar per read-out, keyed `loss/<read-out name>`
(`FreshPGDReconEval.read_out_names`).
`inputs` is the target's own opaque batch (token ids, a feature matrix, a dict of tensors) —
exactly what `masked_forward` takes."""


@dataclass(frozen=True)
class FreshPGDAttack:
    """Sign-PGD from one fresh random draw shared across the batch, `step_size` per step,
    read out after each of `read_out_steps`. The step has no schedule, so a shallower
    read-out is exactly a prefix of a deeper one and one ascent serves them all."""

    step_size: float
    read_out_steps: tuple[int, ...]

    def __post_init__(self) -> None:
        steps = self.read_out_steps
        assert steps and steps[0] >= 0, steps
        assert all(a < b for a, b in pairwise(steps)), f"read-out steps must ascend: {steps}"


@dataclass(frozen=True)
class FreshPGDReconEval:
    """A resolved fresh-PGD reconstruction probe: the attack, the reconstruction objective
    it ascends and reports, and the metric type its read-outs log under."""

    attack: FreshPGDAttack
    reconstruction: AuxiliaryReconstructionSpec
    metric_type: AnyPGDEvalType

    @property
    def read_out_names(self) -> tuple[str, ...]:
        """Each read-out's logged name, `<metric_type>_<n>step`, in `attack.read_out_steps`
        order."""
        return tuple(f"{self.metric_type}_{n}step" for n in self.attack.read_out_steps)

    @property
    def reconstruction_capture_keys(self) -> CaptureKeys:
        return auxiliary_capture_keys(self.reconstruction)


def fresh_pgd_probe(metric: AnyPGDEvalConfig) -> FreshPGDReconEval:
    return FreshPGDReconEval(
        attack=FreshPGDAttack(step_size=metric.step_size, read_out_steps=metric.read_out_steps),
        reconstruction=resolve_auxiliary_reconstruction(metric.auxiliaries),
        metric_type=metric.type,
    )


def _masking_at(ci_lower: Mapping[str, SiteCI], sources: Sources) -> MaterializedMasking:
    return materialize_masking(source_masking(ci_lower, sources))


def _sign_ascent_step(
    ci_lower: Mapping[str, SiteCI],
    step_size: float,
    loss_at_masking: Callable[[MaterializedMasking], Array],
) -> Callable[[Sources], Sources]:
    """One projected sign step up `loss_at_masking`'s source gradient."""

    def step(sources: Sources) -> Sources:
        gradients = jax.grad(lambda s: loss_at_masking(_masking_at(ci_lower, s)))(sources)
        return jax.tree.map(
            lambda source, gradient: jnp.clip(source + step_size * jnp.sign(gradient), 0.0, 1.0),
            sources,
            gradients,
        )

    return step


def fresh_pgd_masking(
    sites: tuple[SiteSpec, ...],
    ci_lower: Mapping[str, SiteCI],
    leading: tuple[int, ...],
    key: PRNGKeyArray,
    step_size: float,
    n_steps: int,
    loss_at_masking: Callable[[MaterializedMasking], Array],
) -> MaterializedMasking:
    """The masking after `n_steps` of fresh sign-PGD against a caller-owned objective."""
    step = _sign_ascent_step(ci_lower, step_size, loss_at_masking)
    sources, _ = jax.lax.scan(
        lambda s, _: (step(s), None),
        init_fresh_pgd_sources(sites, "random", "c", leading, key),
        None,
        length=n_steps,
    )
    return _masking_at(ci_lower, sources)


def fresh_pgd_read_outs[T](
    sites: tuple[SiteSpec, ...],
    ci_lower: Mapping[str, SiteCI],
    leading: tuple[int, ...],
    key: PRNGKeyArray,
    attack: FreshPGDAttack,
    loss_at_masking: Callable[[MaterializedMasking], Array],
    read_out: Callable[[MaterializedMasking], T],
) -> tuple[T, ...]:
    """`read_out` of the masking after each of `attack.read_out_steps` along ONE ascent of
    `loss_at_masking`, in that order. An outer scan over the read-out-to-read-out segments
    runs each segment's steps as a dynamic-length loop, so the ascent step and `read_out`
    each compile once however many read-outs there are."""
    step = _sign_ascent_step(ci_lower, attack.step_size, loss_at_masking)

    def segment(sources: Sources, bounds: Int32[Array, "2"]) -> tuple[Sources, T]:
        sources = jax.lax.fori_loop(bounds[0], bounds[1], lambda _, s: step(s), sources)
        return sources, read_out(_masking_at(ci_lower, sources))

    _, stacked = jax.lax.scan(
        segment,
        init_fresh_pgd_sources(sites, "random", "c", leading, key),
        jnp.asarray(tuple(pairwise((0, *attack.read_out_steps))), jnp.int32),
    )
    return tuple(jax.tree.map(itemgetter(i), stacked) for i in range(len(attack.read_out_steps)))


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
    ) -> dict[str, Array]:
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

        totals = fresh_pgd_read_outs(
            sites, ci_lower, leading, key, fresh_pgd.attack, loss_at_masking, loss_at_masking
        )
        return {
            f"loss/{name}": total
            for name, total in zip(fresh_pgd.read_out_names, totals, strict=True)
        }

    return eval_step
