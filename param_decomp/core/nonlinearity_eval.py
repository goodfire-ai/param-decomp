"""Standing nonlinearity eval.

For each partitioned site it reports two measures of how many nonlinearity uses each
component's writes feed — under GQA a kv block is used `n_head / n_kv_head` times, so
both statistics scale by the partition's use multiplicity. The device step reduces each
partitioned persistence group's WHOLE U stack to per-component statistics; a site's `[C]`
vector is its slot of that small reduced stack, read on the host. The stack axis is never
sliced on device (under the `owner` presets it is sharded, and a static per-slot slice of
a sharded axis is unplaceable), and full component stacks are never gathered to the host.
"""

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import get_args

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jaxtyping import Array, Float

from param_decomp.core.components import ComponentStacks, SiteSpec, site_slots_for, slot_index
from param_decomp.core.jit_util import filter_jit
from param_decomp.core.losses import nonlinearity_unit_squared_norm_fractions, soft_unit_count
from param_decomp.core.nonlinearity import NonlinearityPartition, NonlinearityUnitKind

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
    """Per-component statistics in the components' own layout — a group's whole U stack
    reduces to `[g, C]` (dense) or `[g, E, c]` (expert-blocked, block dims)."""

    soft_use_count: Float[Array, "*components"]
    effective_use_count_per_subcomponent: Float[Array, "*components"]


@dataclass(frozen=True, kw_only=True)
class SiteNonlinearityStats:
    """One site's statistics on the host, in the flat C order every per-component consumer
    emits (`narrow_component_sums`, the CI means): an expert-blocked site's component
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
        * soft_unit_count(fractions, NONLINEARITY_EVAL_RELATIVE_THRESHOLD),
        effective_use_count_per_subcomponent=partition.use_multiplicity
        * jnp.sqrt(fractions).sum(-1) ** 2,
    )


def _group_partitions(sites: tuple[SiteSpec, ...]) -> dict[str, NonlinearityPartition]:
    """The one partition each group's partitioned sites share — a group is a matrix kind,
    and one kind writes into one nonlinearity."""
    by_group: defaultdict[str, set[NonlinearityPartition]] = defaultdict(set)
    for site in sites:
        if site.nonlinearity_partition is not None:
            by_group[site.group].add(site.nonlinearity_partition)
    partitions: dict[str, NonlinearityPartition] = {}
    for group, found in by_group.items():
        assert len(found) == 1, f"group {group!r} mixes nonlinearity partitions: {found}"
        (partitions[group],) = found
    return partitions


def make_nonlinearity_eval_step(
    sites: tuple[SiteSpec, ...],
    compiler_options: dict[str, bool | int | str],
) -> NonlinearityEvalStep:
    """Reduce every partitioned group's whole U stack to per-component statistics."""
    partitions = _group_partitions(sites)

    def nonlinearity_eval_step(
        components: ComponentStacks,
    ) -> dict[str, ComponentNonlinearityStats]:
        return {
            group: component_nonlinearity_stats(components.stacks[group][1], partition)
            for group, partition in partitions.items()
        }

    return filter_jit(nonlinearity_eval_step, compiler_options=compiler_options)


def _host_array(value: Array) -> np.ndarray:
    """Materialize a small diagnostic reduction that may span multiple processes."""
    if not value.is_fully_addressable:
        value = multihost_utils.process_allgather(value, tiled=True)
    return np.asarray(value)


def site_nonlinearity_stats(
    stack_stats: Mapping[str, ComponentNonlinearityStats], sites: tuple[SiteSpec, ...]
) -> dict[str, SiteNonlinearityStats]:
    """Each partitioned site's `[C]` statistics: its slot of the group's reduced stack,
    the block dims of an expert-blocked group flattened to the flat expert-major C order."""
    host = {
        group: (
            _host_array(stats.soft_use_count),
            _host_array(stats.effective_use_count_per_subcomponent),
        )
        for group, stats in stack_stats.items()
    }
    slots = slot_index(site_slots_for(sites))
    per_site: dict[str, SiteNonlinearityStats] = {}
    for site in sites:
        if site.nonlinearity_partition is None:
            continue
        group, slot = slots[site.name]
        soft, effective = host[group]
        per_site[site.name] = SiteNonlinearityStats(
            soft_use_count=soft[slot].reshape(-1),
            effective_use_count_per_subcomponent=effective[slot].reshape(-1),
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
        entries |= {
            f"{ci_alive_prefix}/{key}": float(value[ci_alive].mean())
            for key, value in metrics.items()
        }
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
        entries |= _mean_entries(f"eval/nonlinearity/aggregates/{kind}", metrics, ci_alive)

    for name, metrics in metrics_by_site.items():
        entries |= _mean_entries(f"eval/nonlinearity/sites/{name}", metrics, ci_alive_by_site[name])

    return entries
