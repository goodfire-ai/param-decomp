"""Standing nonlinearity eval.

For each partitioned site it reports two measures of how many nonlinearity uses each
component's aligned vectors touch — under GQA a kv block is used `n_head / n_kv_head` times, so
both statistics scale by the partition's use multiplicity. The device step reduces each
partitioned persistence group's whole aligned factor stack to per-component statistics; a site's `[C]`
vector is its stack index into that small reduced stack, read on the host. The stack axis
is never sliced on device (under the `owner` presets it is sharded, and a static slice of
a sharded axis is unplaceable), and full component stacks are never gathered to the host.
"""

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import get_args

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jaxtyping import Array, Float

from param_decomp.core.components import (
    ComponentStacks,
    SiteSpec,
    aligned_component_vectors,
    site_stack_indices_for,
    stack_index_by_site,
)
from param_decomp.core.dict_utils import dict_safe_update_
from param_decomp.core.losses import nonlinearity_unit_squared_norm_fractions, soft_unit_count
from param_decomp.core.nonlinearity import (
    NonlinearityAlignment,
    NonlinearityPartition,
    NonlinearityUnitKind,
)

NONLINEARITY_EVAL_RELATIVE_THRESHOLD = 4.0
NONLINEARITY_EVAL_SOFT_COUNT_KEY = (
    f"soft_use_count_relative_threshold_{NONLINEARITY_EVAL_RELATIVE_THRESHOLD:g}"
)
NONLINEARITY_EVAL_EFFECTIVE_COUNT_KEY = "effective_use_count_per_subcomponent"
_NONLINEARITY_EVAL_METRIC_KEYS = (
    NONLINEARITY_EVAL_SOFT_COUNT_KEY,
    NONLINEARITY_EVAL_EFFECTIVE_COUNT_KEY,
)
NONLINEARITY_EVAL_MEAN_CI_CUTOFF = 0.0
NONLINEARITY_EVAL_MEAN_CI_STRATUM = f"mean_ci_gt_{NONLINEARITY_EVAL_MEAN_CI_CUTOFF:g}"
_UNIT_KINDS: tuple[NonlinearityUnitKind, ...] = get_args(NonlinearityUnitKind)


@jax.tree_util.register_dataclass
@dataclass(frozen=True, kw_only=True)
class ComponentNonlinearityStats:
    """Per-component statistics: a group's whole aligned factor stack reduces to `[g, C]` (dense) or `[g, E, c]` (block-factored, block dims)."""

    soft_use_count: Float[Array, "*components"]
    effective_use_count_per_subcomponent: Float[Array, "*components"]


@dataclass(frozen=True, kw_only=True)
class SiteNonlinearityStats:
    """One site's statistics on the host, in the flat C order every per-component consumer
    emits (`selected_component_sums`, the CI means): a block-factored site's component
    `(e, j)` at `e·c + j`."""

    soft_use_count: Float[np.ndarray, " C"]
    effective_use_count_per_subcomponent: Float[np.ndarray, " C"]


NonlinearityEvalStep = Callable[[ComponentStacks], dict[str, ComponentNonlinearityStats]]
"""Per partitioned persistence group, the statistics of its whole U stack."""


def component_nonlinearity_stats(
    vectors: Float[Array, "*components d"], partition: NonlinearityPartition
) -> ComponentNonlinearityStats:
    """Return the fixed-threshold soft use count and L1 effective use count per component.

    For unit-block norms `r_u`, the effective block count is `(Σ_u r_u)² / Σ_u r_u²`;
    both statistics scale by the partition's use multiplicity to count uses.
    """
    fractions = nonlinearity_unit_squared_norm_fractions(vectors, partition)
    return ComponentNonlinearityStats(
        soft_use_count=partition.use_multiplicity
        * soft_unit_count(fractions, NONLINEARITY_EVAL_RELATIVE_THRESHOLD, normalize_at_one=False),
        effective_use_count_per_subcomponent=partition.use_multiplicity
        * jnp.sqrt(fractions).sum(-1) ** 2,
    )


def _group_alignments(sites: tuple[SiteSpec, ...]) -> dict[str, NonlinearityAlignment]:
    """The one partition each group's partitioned sites share — a group is a matrix kind,
    and one kind faces one nonlinearity on one side."""
    by_group: defaultdict[str, set[NonlinearityAlignment]] = defaultdict(set)
    for site in sites:
        if site.alignment is not None:
            by_group[site.group].add(site.alignment)
    partitions: dict[str, NonlinearityAlignment] = {}
    for group, found in by_group.items():
        assert len(found) == 1, f"group {group!r} mixes nonlinearity partitions: {found}"
        (partitions[group],) = found
    return partitions


def make_nonlinearity_eval_step(
    sites: tuple[SiteSpec, ...],
    output_sharding: NamedSharding | None,
) -> NonlinearityEvalStep:
    """Reduce aligned factors and place the small statistics for their consumer."""
    partitions = _group_alignments(sites)

    def nonlinearity_eval_step(
        components: ComponentStacks,
    ) -> dict[str, ComponentNonlinearityStats]:
        stats = {
            group: component_nonlinearity_stats(
                aligned_component_vectors(components.stacks[group], alignment.side),
                alignment.partition,
            )
            for group, alignment in partitions.items()
        }
        if output_sharding is None:
            return stats
        return jax.tree.map(lambda value: jax.sharding.reshard(value, output_sharding), stats)

    return nonlinearity_eval_step


def site_nonlinearity_stats(
    stack_stats: Mapping[str, ComponentNonlinearityStats], sites: tuple[SiteSpec, ...]
) -> dict[str, SiteNonlinearityStats]:
    """Each partitioned site's `[C]` statistics: its stack index into the group's reduced stack,
    the block dims of a block-factored group flattened to the flat block-major C order."""
    host = {
        group: (
            np.asarray(stats.soft_use_count),
            np.asarray(stats.effective_use_count_per_subcomponent),
        )
        for group, stats in stack_stats.items()
    }
    stack_indices = stack_index_by_site(site_stack_indices_for(sites))
    per_site: dict[str, SiteNonlinearityStats] = {}
    for site in sites:
        if site.alignment is None:
            continue
        group, index = stack_indices[site.name]
        soft, effective = host[group]
        per_site[site.name] = SiteNonlinearityStats(
            soft_use_count=soft[index].reshape(-1),
            effective_use_count_per_subcomponent=effective[index].reshape(-1),
        )
    return per_site


def _metric_values(stat: SiteNonlinearityStats) -> dict[str, np.ndarray]:
    return {
        NONLINEARITY_EVAL_SOFT_COUNT_KEY: stat.soft_use_count,
        NONLINEARITY_EVAL_EFFECTIVE_COUNT_KEY: stat.effective_use_count_per_subcomponent,
    }


def _mean_entries(
    prefix: str, metrics: Mapping[str, np.ndarray], ci_alive: np.ndarray
) -> dict[str, float]:
    assert all(value.shape == ci_alive.shape for value in metrics.values())
    entries = {f"{prefix}/all/{key}": float(value.mean()) for key, value in metrics.items()}
    ci_alive_prefix = f"{prefix}/{NONLINEARITY_EVAL_MEAN_CI_STRATUM}"
    entries[f"{ci_alive_prefix}/n_components"] = float(ci_alive.sum())
    if ci_alive.any():
        dict_safe_update_(
            entries,
            {
                f"{ci_alive_prefix}/{key}": float(value[ci_alive].mean())
                for key, value in metrics.items()
            },
        )
    return entries


def nonlinearity_log_entries(
    stats: Mapping[str, SiteNonlinearityStats],
    ci_means: Mapping[str, np.ndarray],
    partitions: Mapping[str, NonlinearityPartition],
) -> dict[str, float]:
    """Log site and unit-kind means over all and mean-CI-positive components."""
    assert stats.keys() == partitions.keys()
    assert stats.keys() <= ci_means.keys()
    entries: dict[str, float] = {}
    metrics_by_site = {name: _metric_values(stat) for name, stat in stats.items()}
    ci_alive_by_site = {
        name: np.asarray(ci_means[name]) > NONLINEARITY_EVAL_MEAN_CI_CUTOFF for name in stats
    }

    for kind in _UNIT_KINDS:
        names = [name for name, part in partitions.items() if part.unit_kind == kind]
        if not names:
            continue
        metrics = {
            key: np.concatenate([metrics_by_site[name][key] for name in names])
            for key in _NONLINEARITY_EVAL_METRIC_KEYS
        }
        ci_alive = np.concatenate([ci_alive_by_site[name] for name in names])
        dict_safe_update_(
            entries, _mean_entries(f"eval/nonlinearity/aggregates/{kind}", metrics, ci_alive)
        )

    for name, metrics in metrics_by_site.items():
        dict_safe_update_(
            entries,
            _mean_entries(f"eval/nonlinearity/sites/{name}", metrics, ci_alive_by_site[name]),
        )

    return entries
