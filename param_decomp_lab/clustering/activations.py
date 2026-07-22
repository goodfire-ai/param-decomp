"""Dense activation filtering — the reference path the streaming `MembershipBuilder` is
checked against. Not used in production harvest (the JAX worker streams memberships)."""

from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
from jaxtyping import Bool, Float

from param_decomp_lab.clustering.formatting import (
    DeadComponentFilterStat,
    ModuleFilterFunc,
)
from param_decomp_lab.clustering.types import ActivationsArray, ComponentLabels


def _get_component_filter_values(
    activations: ActivationsArray,
    filter_stat: DeadComponentFilterStat,
) -> Float[np.ndarray, " c"]:
    if filter_stat == "max":
        return activations.max(axis=0)

    assert filter_stat == "mean", f"Unsupported dead component filter stat: {filter_stat}"
    return activations.mean(axis=0)


class FilteredActivations(NamedTuple):
    activations: ActivationsArray
    "activations after filtering dead components"

    labels: ComponentLabels
    "list of length c with labels for each preserved component"

    dead_components_labels: ComponentLabels | None
    "list of labels for dead components, or None if no filtering was applied"

    @property
    def n_alive(self) -> int:
        n_alive: int = len(self.labels)
        assert n_alive == self.activations.shape[1], (
            f"{n_alive = } != {self.activations.shape[1] = }"
        )
        return n_alive

    @property
    def n_dead(self) -> int:
        return len(self.dead_components_labels) if self.dead_components_labels else 0


def filter_dead_components(
    activations: ActivationsArray,
    labels: ComponentLabels,
    filter_dead_threshold: float = 0.01,
    filter_dead_stat: DeadComponentFilterStat = "max",
) -> FilteredActivations:
    """Filter out dead components based on a threshold.

    if `filter_dead_threshold` is 0, no filtering is applied: activations and labels are
    returned as is, `dead_components_labels` is `None`. Otherwise, components whose
    aggregate activation statistic (selected by `filter_dead_stat`) is below the threshold
    are dropped, and their labels returned in `dead_components_labels` (also `None` if none
    were below the threshold).
    """
    dead_components_lst: ComponentLabels | None = None
    if filter_dead_threshold > 0:
        dead_components_lst = ComponentLabels(list())
        filter_values: Float[np.ndarray, " c"] = _get_component_filter_values(
            activations=activations,
            filter_stat=filter_dead_stat,
        )
        dead_components: Bool[np.ndarray, " c"] = filter_values < filter_dead_threshold

        if dead_components.any():
            activations = activations[:, ~dead_components]
            alive_labels: list[tuple[str, bool]] = [
                (lbl, bool(keep)) for lbl, keep in zip(labels, ~dead_components, strict=False)
            ]
            labels = ComponentLabels([label for label, keep in alive_labels if keep])
            dead_components_lst = ComponentLabels(
                [label for label, keep in alive_labels if not keep]
            )

    return FilteredActivations(
        activations=activations,
        labels=labels,
        dead_components_labels=dead_components_lst if dead_components_lst else None,
    )


@dataclass(frozen=True)
class ProcessedActivations:
    """Filtered, concatenated dense activations with per-module bookkeeping."""

    module_component_counts: dict[str, int]
    "total component count per module (including dead), preserving module order"

    module_alive_counts: dict[str, int]
    "alive component count per module, preserving module order"

    activations: ActivationsArray
    "activations after filtering and concatenation"

    labels: ComponentLabels
    "labels for each preserved component, format `{module_name}:{component_index}`"

    dead_components_lst: ComponentLabels | None
    "labels for dead components, or None if no filtering was applied"

    @property
    def n_components_original(self) -> int:
        return sum(self.module_component_counts.values())

    @property
    def n_components_alive(self) -> int:
        n_alive: int = len(self.labels)
        assert n_alive + self.n_components_dead == self.n_components_original, (
            f"({n_alive = }) + ({self.n_components_dead = }) != ({self.n_components_original = })"
        )
        assert n_alive == self.activations.shape[1], (
            f"{n_alive = } != {self.activations.shape[1] = }"
        )
        return n_alive

    @property
    def n_components_dead(self) -> int:
        return len(self.dead_components_lst) if self.dead_components_lst else 0


def process_activations(
    activations: dict[
        str,
        Float[np.ndarray, "samples C"] | Float[np.ndarray, " n_sample n_ctx C"],
    ],
    filter_dead_threshold: float,
    filter_dead_stat: DeadComponentFilterStat = "max",
    seq_mode: Literal["concat", "seq_mean", None] = None,
    filter_modules: ModuleFilterFunc | None = None,
) -> ProcessedActivations:
    """Concatenate per-module activations and filter dead components in one pass."""
    activations_: dict[str, ActivationsArray]
    if seq_mode == "concat":
        activations_ = {
            key: act.reshape(act.shape[0] * act.shape[1], act.shape[2])
            for key, act in activations.items()
        }
    elif seq_mode == "seq_mean":
        activations_ = {
            key: act.mean(axis=1) if act.ndim == 3 else act for key, act in activations.items()
        }
    else:
        activations_ = activations

    if filter_modules is not None:
        activations_ = {key: act for key, act in activations_.items() if filter_modules(key)}

    module_component_counts: dict[str, int] = {}
    alive_masks: dict[str, Bool[np.ndarray, " c"]] = {}
    total_alive = 0
    for key, act in activations_.items():
        c = act.shape[-1]
        module_component_counts[key] = c
        if filter_dead_threshold > 0:
            filter_values: Float[np.ndarray, " c"] = _get_component_filter_values(
                activations=act,
                filter_stat=filter_dead_stat,
            )
            alive = filter_values >= filter_dead_threshold
            alive_masks[key] = alive
            total_alive += int(alive.sum())
        else:
            total_alive += c

    total_c = sum(module_component_counts.values())

    first_act = next(iter(activations_.values()))
    n_samples = first_act.shape[0]
    dtype = first_act.dtype
    act_filtered = np.empty((n_samples, total_alive), dtype=dtype)

    offset = 0
    alive_labels = ComponentLabels(list())
    dead_labels = ComponentLabels(list())
    module_alive_counts: dict[str, int] = {}

    for key in list(activations_.keys()):
        tensor = activations_.pop(key)
        c = tensor.shape[-1]

        if filter_dead_threshold > 0:
            alive = alive_masks[key]
            n_alive = int(alive.sum())
            for i in range(c):
                label = f"{key}:{i}"
                if alive[i]:
                    alive_labels.append(label)
                else:
                    dead_labels.append(label)
            if n_alive > 0:
                act_filtered[:, offset : offset + n_alive] = tensor[:, alive]
        else:
            n_alive = c
            alive_labels.extend([f"{key}:{i}" for i in range(c)])
            act_filtered[:, offset : offset + n_alive] = tensor

        module_alive_counts[key] = n_alive
        offset += n_alive

    assert offset == total_alive
    assert list(module_alive_counts.keys()) == list(module_component_counts.keys())
    assert len(alive_labels) + len(dead_labels) == total_c

    return ProcessedActivations(
        module_component_counts=module_component_counts,
        module_alive_counts=module_alive_counts,
        activations=act_filtered,
        labels=alive_labels,
        dead_components_lst=dead_labels if dead_labels else None,
    )
