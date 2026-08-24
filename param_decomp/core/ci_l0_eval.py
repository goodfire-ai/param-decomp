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
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jaxtyping import Array, Float, PRNGKeyArray

from param_decomp.core.ci_fn import PlacedCIFn, evaluate_ci
from param_decomp.core.components import ComponentStacks
from param_decomp.core.jit_util import filter_jit
from param_decomp.core.model import CaptureKeys, PlacedModel
from param_decomp.core.sharding import batch_shard_leading

type CI_L0Step = Callable[
    [PlacedModel, ComponentStacks, PlacedCIFn, Any, PRNGKeyArray], dict[str, Array]
]


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


def ci_l0_scalars(
    ci_lower: dict[str, Float[Array, "*leading C"]],
    site_names: tuple[str, ...],
    ci_alive_threshold: float,
    groups: dict[str, tuple[str, ...]],
    reduce_leading: Callable[[Float[Array, "*leading"]], Float[Array, ""]],
) -> dict[str, Array]:
    """Per-site alive-component counts plus per-group sums of them.

    `reduce_leading` contracts the waist's leading axes to a scalar — a plain mean, or the
    LM's row-masked mean when the batch is partially padded.
    """
    site_l0 = {
        site: reduce_leading((ci_lower[site] > ci_alive_threshold).astype(jnp.float32).sum(-1))
        for site in site_names
    }
    return {f"l0/{ci_alive_threshold}_{site}": value for site, value in site_l0.items()} | {
        f"l0/{ci_alive_threshold}_{name}": sum(
            (site_l0[site] for site in members), start=jnp.zeros((), jnp.float32)
        )
        for name, members in groups.items()
    }


def make_ci_l0_eval_step(
    model_static: PlacedModel,
    ci_capture_keys: CaptureKeys,
    ci_alive_threshold: float,
    groups: dict[str, tuple[str, ...]] | None,
    mesh: Mesh | None = None,
    compiler_options: dict[str, bool | int | str] | None = None,
) -> CI_L0Step:
    """Build `CI_L0` for ANY target: the CI envelope is all it reads, so only the taps
    forward is target-specific. No clean output, no masked re-forward, no recon metric."""
    site_names = model_static.site_names
    resolved_groups = resolve_site_groups(site_names, groups)
    leading_rank = 2 if model_static.has_position_axis else 1

    def eval_step(
        model: PlacedModel,
        components: ComponentStacks,
        placed_ci_fn: PlacedCIFn,
        inputs: Any,
        key: PRNGKeyArray,
    ) -> dict[str, Array]:
        del components, key  # L0 reads the CI envelope alone
        sharded_inputs = jax.tree.map(lambda x: batch_shard_leading(x, mesh), inputs)
        ci_input_activations = model.clean_forward(sharded_inputs, ci_capture_keys).captures
        ci_lower = evaluate_ci(placed_ci_fn, ci_input_activations, remat=False).lower
        leading = next(iter(ci_lower.values())).shape[:-1]
        assert len(leading) == leading_rank, (leading, model_static.has_position_axis)
        return ci_l0_scalars(ci_lower, site_names, ci_alive_threshold, resolved_groups, jnp.mean)

    return filter_jit(eval_step, compiler_options=compiler_options)
