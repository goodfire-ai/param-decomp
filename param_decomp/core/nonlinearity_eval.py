"""Standing nonlinearity eval (SPEC S36).

For each partitioned site it reports two measures of how many nonlinearity uses each
component's writes feed — under GQA a kv block is used `n_head / n_kv_head` times, so
both statistics scale by the partition's use multiplicity. The device step reduces U to
`[C]` vectors so full component stacks are never gathered to the host.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import get_args

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jaxtyping import Array, Float

from param_decomp.core.components import (
    ComponentStacks,
)
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
    soft_use_count: Float[Array, " C"]
    effective_use_count_per_subcomponent: Float[Array, " C"]


NonlinearityEvalStep = Callable[[ComponentStacks], dict[str, ComponentNonlinearityStats]]


def component_nonlinearity_stats(
    vectors: Float[Array, "C d"], partition: NonlinearityPartition
) -> ComponentNonlinearityStats:
    """Return the fixed-threshold soft use count and L1 effective use count per component.

    For unit-block norms `r_u`, the effective block count is `(Σ_u r_u)² / Σ_u r_u²`;
    both statistics scale by the partition's use multiplicity to count uses (SPEC S36).
    """
    fractions = nonlinearity_unit_squared_norm_fractions(vectors, partition)
    return ComponentNonlinearityStats(
        soft_use_count=partition.use_multiplicity
        * soft_unit_count(fractions, NONLINEARITY_EVAL_RELATIVE_THRESHOLD),
        effective_use_count_per_subcomponent=partition.use_multiplicity
        * jnp.sqrt(fractions).sum(-1) ** 2,
    )


def make_nonlinearity_eval_step(
    partitions: Mapping[str, NonlinearityPartition],
    compiler_options: dict[str, bool | int | str],
) -> NonlinearityEvalStep:
    def nonlinearity_eval_step(
        components: ComponentStacks,
    ) -> dict[str, ComponentNonlinearityStats]:
        return {
            name: component_nonlinearity_stats(components.site(name).U, part)
            for name, part in partitions.items()
        }

    return filter_jit(nonlinearity_eval_step, compiler_options=compiler_options)


def _host_array(value: Array) -> np.ndarray:
    """Materialize a small diagnostic reduction that may span multiple processes."""
    if not value.is_fully_addressable:
        value = multihost_utils.process_allgather(value, tiled=True)
    return np.asarray(value)


def _metric_values(stat: ComponentNonlinearityStats) -> dict[str, np.ndarray]:
    return {
        NONLINEARITY_EVAL_SOFT_COUNT_KEY: _host_array(stat.soft_use_count),
        NONLINEARITY_EVAL_EFFECTIVE_COUNT_KEY: _host_array(
            stat.effective_use_count_per_subcomponent
        ),
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
    stats: Mapping[str, ComponentNonlinearityStats],
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
