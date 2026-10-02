"""Target-generic causal-importance L0 (`CI_L0`).

Counts components whose causal importance clears `ci_alive_threshold`, per site and per
authored group, averaged over the waist's leading axes. Pure CI arithmetic: it reads
nothing but the CI envelope, so it is blind to the target's input, output and recon
metric, and to which waist geometry the run uses.

Log keys (`l0/<threshold>_<site|group>`) are shared with the LM binder, which supplies its
own row-masked reduction over the leading axes; the key shapes must stay identical because
the torch reference's `CI_L0` is compared against them.
"""

from collections.abc import Callable
from fnmatch import fnmatch

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jaxtyping import Array, Float, PRNGKeyArray

from param_decomp.core.base_config import Probability
from param_decomp.core.ci_fn.interface import CIFn
from param_decomp.core.ci_fn.runtime import evaluate_ci_from_captures
from param_decomp.core.components import ComponentStacks, SelectedCI, SiteCI
from param_decomp.core.model import CaptureKeys, ComponentActivations, PlacedModel
from param_decomp.core.sharding import batch_shard_leading

type CI_L0Step[TargetIn, Out, PreparedT: ComponentActivations, Conditioning, PreparedMaskingT] = (
    Callable[
        [
            PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
            ComponentStacks,
            CIFn[Conditioning],
            TargetIn,
            PRNGKeyArray,
        ],
        dict[str, Array],
    ]
)


def resolve_site_groups(
    site_names: tuple[str, ...], patterns_by_group: dict[str, tuple[str, ...]] | None
) -> dict[str, tuple[str, ...]]:
    """Expand each authored group's fnmatch patterns into matching site names."""
    if patterns_by_group is None:
        return {}
    groups: dict[str, tuple[str, ...]] = {}
    for name, patterns in patterns_by_group.items():
        members = tuple(
            site for site in site_names if any(fnmatch(site, pattern) for pattern in patterns)
        )
        assert members, f"site group {name!r} matches no sites: {patterns}"
        groups[name] = members
    return groups


def alive_component_counts(ci: SiteCI, threshold: Probability) -> Float[Array, "*leading"]:
    """Per-position count of components whose CI is strictly above `threshold` — exact
    for EVERY threshold on both emissions: a selected-emitting site's unselected
    components have CI exactly 0.0 by definition, so each contributes the analytic
    `[threshold < 0.0]` (the `psi(0)` move of the selected-emission loss spellings),
    `C − k·c` of them per position."""
    match ci:
        case SelectedCI():
            selected = (ci.values > threshold).astype(jnp.float32).sum(-1)
            n_unselected = ci.C - ci.values.shape[-1]
            return selected + n_unselected * float(threshold < 0.0)
        case jax.Array():
            return (ci > threshold).astype(jnp.float32).sum(-1)


def ci_l0_scalars(
    ci_lower: dict[str, SiteCI],
    site_names: tuple[str, ...],
    ci_alive_threshold: Probability,
    groups: dict[str, tuple[str, ...]],
    reduce_leading: Callable[[Float[Array, "*leading"]], Float[Array, ""]],
) -> dict[str, Array]:
    """Per-site alive-component counts plus per-group sums of them.

    `reduce_leading` contracts the waist's leading axes to a scalar — a plain mean, or the
    LM's row-masked mean when the batch is partially padded.
    """
    site_l0 = {
        site: reduce_leading(alive_component_counts(ci_lower[site], ci_alive_threshold))
        for site in site_names
    }
    return {f"l0/{ci_alive_threshold}_{site}": value for site, value in site_l0.items()} | {
        f"l0/{ci_alive_threshold}_{name}": sum(
            (site_l0[site] for site in members), start=jnp.zeros((), jnp.float32)
        )
        for name, members in groups.items()
    }


def make_ci_l0_eval_step[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    model_static: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    ci_capture_keys: CaptureKeys,
    ci_alive_threshold: Probability,
    groups: dict[str, tuple[str, ...]] | None,
    mesh: Mesh | None = None,
) -> CI_L0Step[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT]:
    """Build `CI_L0` for ANY target: the CI envelope is all it reads, so only the taps
    forward is target-specific. No clean output, no masked re-forward, no recon metric."""
    site_names = model_static.model.site_names
    resolved_groups = resolve_site_groups(site_names, groups)
    leading_rank = 2 if model_static.model.has_position_axis else 1

    def eval_step(
        model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
        components: ComponentStacks,
        ci_fn: CIFn[Conditioning],
        inputs: TargetIn,
        key: PRNGKeyArray,
    ) -> dict[str, Array]:
        del key
        sharded_inputs = jax.tree.map(lambda x: batch_shard_leading(x, mesh), inputs)
        clean = model.clean_forward(sharded_inputs, ci_capture_keys)
        ci_lower = evaluate_ci_from_captures(
            ci_fn.prepare(),
            clean.captures,
            clean.conditioning,
            model.prepare_compute_weights(components),
            sequence=clean.sequence,
            remat=False,
        ).lower
        leading = clean.leading_shape
        assert len(leading) == leading_rank, (leading, model_static.model.has_position_axis)
        return ci_l0_scalars(ci_lower, site_names, ci_alive_threshold, resolved_groups, jnp.mean)

    return eval_step
