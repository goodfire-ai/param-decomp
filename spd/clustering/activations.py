from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal, NamedTuple

import torch
from jaxtyping import Bool, Float, Float16
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.clustering.consts import (
    ActivationsTensor,
    BoolActivationsTensor,
    ClusterCoactivationShaped,
    ComponentLabels,
)
from spd.clustering.util import ModuleFilterFunc
from spd.log import logger
from spd.models.component_model import ComponentModel, OutputWithCache


def component_activations(
    model: ComponentModel,
    device: torch.device | str,
    batch: Tensor,
) -> dict[str, ActivationsTensor]:
    """Get the component activations over a **single** batch."""
    causal_importances: dict[str, ActivationsTensor]
    with torch.no_grad():
        model_output: OutputWithCache = model(
            batch.to(device),
            cache_type="input",
        )

        causal_importances = model.calc_causal_importances(
            pre_weight_acts=model_output.cache,
            sampling="continuous",
            detach_inputs=False,
        ).lower_leaky

    return causal_importances


def collect_activations(
    model: ComponentModel,
    dataloader: DataLoader[Any],
    n_tokens: int,
    n_tokens_per_seq: int,
    device: torch.device | str,
    seed: int,
) -> dict[str, Float[Tensor, "n_tokens C"]]:
    """Collect activation samples by picking random tokens per sequence.

    Iterates through batches from dataloader, runs component_activations on each,
    then selects n_tokens_per_seq random token positions per sequence. Collects until
    n_tokens samples are gathered.

    Args:
        model: ComponentModel to get activations from
        dataloader: DataLoader yielding batches of sequences
        n_tokens: Total number of token samples to collect
        n_tokens_per_seq: Number of random token positions to sample per sequence
        device: Device to run on
        seed: Random seed for reproducible token position selection
    """
    rng = torch.Generator().manual_seed(seed)
    collected: dict[str, list[Tensor]] = {}
    n_collected = 0

    pbar = tqdm(dataloader, desc="Collecting activations", unit="batch")
    for batch_data in pbar:
        input_ids = batch_data["input_ids"]
        batch_size, n_ctx = input_ids.shape

        activations = component_activations(model=model, batch=input_ids, device=device)

        # Pick n_tokens_per_seq random token positions per sequence
        positions = torch.randint(0, n_ctx, (batch_size, n_tokens_per_seq), generator=rng)
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand_as(positions)

        for key, act in activations.items():
            # act shape: (batch_size, n_ctx, C)
            sampled = act[batch_indices, positions]  # (batch_size, n_tokens_per_seq, C)
            sampled = sampled.reshape(batch_size * n_tokens_per_seq, -1)
            if key not in collected:
                collected[key] = []
            collected[key].append(sampled.cpu())

        n_collected += batch_size * n_tokens_per_seq
        pbar.set_postfix(tokens=f"{min(n_collected, n_tokens)}/{n_tokens}")
        if n_collected >= n_tokens:
            break

    assert n_collected >= n_tokens, (
        f"Dataloader exhausted: collected {n_collected} tokens but needed {n_tokens}"
    )

    logger.info(f"Collected {n_collected} token activations (requested {n_tokens})")

    # Concatenate and truncate to exactly n_tokens. Pop chunks to free memory before next module.
    result: dict[str, Float[Tensor, "n_tokens C"]] = {}
    for key in list(collected.keys()):
        result[key] = torch.cat(collected.pop(key), dim=0)[:n_tokens]
    return result


def compute_coactivatons(
    activations: ActivationsTensor | BoolActivationsTensor,
) -> ClusterCoactivationShaped:
    """Compute the coactivations matrix from the activations."""
    # TODO: this works for both boolean and continuous activations,
    # but we could do better by just using OR for boolean activations
    # and maybe even some bitshift hacks. but for now, we convert to float16
    activations_f16: Float16[Tensor, "samples C"] = activations.to(torch.float16)
    return activations_f16.T @ activations_f16


class FilteredActivations(NamedTuple):
    activations: ActivationsTensor
    "activations after filtering dead components"

    labels: ComponentLabels
    "list of length c with labels for each preserved component"

    dead_components_labels: ComponentLabels | None
    "list of labels for dead components, or None if no filtering was applied"

    @property
    def n_alive(self) -> int:
        """Number of alive components after filtering."""
        n_alive: int = len(self.labels)
        assert n_alive == self.activations.shape[1], (
            f"{n_alive = } != {self.activations.shape[1] = }"
        )
        return n_alive

    @property
    def n_dead(self) -> int:
        """Number of dead components after filtering."""
        return len(self.dead_components_labels) if self.dead_components_labels else 0


def filter_dead_components(
    activations: ActivationsTensor,
    labels: ComponentLabels,
    filter_dead_threshold: float = 0.01,
) -> FilteredActivations:
    """Filter out dead components based on a threshold

    if `filter_dead_threshold` is 0, no filtering is applied.
    activations and labels are returned as is, `dead_components_labels` is `None`.

    otherwise, components whose **maximum** activations across all samples is below the threshold
    are considered dead and filtered out. The labels of these components are returned in `dead_components_labels`.
    `dead_components_labels` will also be `None` if no components were below the threshold.
    """
    dead_components_lst: ComponentLabels | None = None
    if filter_dead_threshold > 0:
        dead_components_lst = ComponentLabels(list())
        max_act: Float[Tensor, " c"] = activations.max(dim=0).values
        dead_components: Bool[Tensor, " c"] = max_act < filter_dead_threshold

        if dead_components.any():
            activations = activations[:, ~dead_components]
            alive_labels: list[tuple[str, bool]] = [
                (lbl, bool(keep.item()))
                for lbl, keep in zip(labels, ~dead_components, strict=False)
            ]
            # re-assign labels only if we are filtering
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
    """Processed activations after filtering and concatenation"""

    module_component_counts: dict[str, int]
    "total component count per module (including dead), preserving module order"

    module_alive_counts: dict[str, int]
    "alive component count per module, preserving module order"

    activations: ActivationsTensor
    "activations after filtering and concatenation"

    labels: ComponentLabels
    "list of length c with labels for each preserved component, format `{module_name}:{component_index}`"

    dead_components_lst: ComponentLabels | None
    "list of labels for dead components, or None if no filtering was applied"

    def validate(self) -> None:
        """Validate the processed activations"""
        # getting this property will also perform a variety of other checks
        assert self.n_components_alive > 0

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

    @cached_property
    def label_index(self) -> dict[str, int | None]:
        """Create a mapping from label to alive index (`None` if dead)"""
        return {
            **{label: i for i, label in enumerate(self.labels)},
            **(
                {label: None for label in self.dead_components_lst}
                if self.dead_components_lst
                else {}
            ),
        }

    def get_label_index(self, label: str) -> int | None:
        """Get the index of a label in the activations, or None if it is dead"""
        return self.label_index[label]

    def get_label_index_alive(self, label: str) -> int:
        """Get the index of a label in the activations, or raise if it is dead"""
        idx: int | None = self.get_label_index(label)
        if idx is None:
            raise ValueError(f"Label '{label}' is dead and has no index in the activations.")
        return idx

    @property
    def module_keys(self) -> list[str]:
        return list(self.module_component_counts.keys())

    def get_module_indices(self, module_key: str) -> list[int | None]:
        """given a module key, return a list len "num components in that module", with int index in alive components, or None if dead"""
        num_components: int = self.module_component_counts[module_key]
        return [self.label_index[f"{module_key}:{i}"] for i in range(num_components)]

    def get_module_activations(self) -> dict[str, ActivationsTensor]:
        """Reconstruct per-module activation views (alive components only) from the concatenated tensor."""
        result: dict[str, ActivationsTensor] = {}
        offset = 0
        for key, n_alive in self.module_alive_counts.items():
            if n_alive > 0:
                result[key] = self.activations[:, offset : offset + n_alive]
            offset += n_alive
        return result


def process_activations(
    activations: dict[
        str,  # module name to
        Float[Tensor, "samples C"]  # (sample x component gate activations)
        | Float[Tensor, " n_sample n_ctx C"],  # (sample x seq index x component gate activations)
    ],
    filter_dead_threshold: float,
    seq_mode: Literal["concat", "seq_mean", None] = None,
    filter_modules: ModuleFilterFunc | None = None,
) -> ProcessedActivations:
    """Concatenate per-module activations and filter dead components.

    Fuses concatenation and filtering into a single pass to avoid holding two full
    copies (~2x total components * n_samples) in memory simultaneously.
    """

    # reshape -- special cases for llms
    # ============================================================
    activations_: dict[str, ActivationsTensor]
    if seq_mode == "concat":
        activations_ = {
            key: act.reshape(act.shape[0] * act.shape[1], act.shape[2])
            for key, act in activations.items()
        }
    elif seq_mode == "seq_mean":
        activations_ = {
            key: act.mean(dim=1) if act.ndim == 3 else act for key, act in activations.items()
        }
    else:
        activations_ = activations

    # filter activations for only the modules we want
    if filter_modules is not None:
        activations_ = {key: act for key, act in activations_.items() if filter_modules(key)}

    # First pass: compute per-module component counts and alive masks
    module_component_counts: dict[str, int] = {}
    alive_masks: dict[str, Bool[Tensor, " c"]] = {}
    total_alive = 0
    for key, act in activations_.items():
        c = act.shape[-1]
        module_component_counts[key] = c
        if filter_dead_threshold > 0:
            max_act: Float[Tensor, " c"] = act.max(dim=0).values
            alive = max_act >= filter_dead_threshold
            alive_masks[key] = alive
            total_alive += int(alive.sum().item())
        else:
            total_alive += c

    total_c = sum(module_component_counts.values())

    # Second pass: pre-allocate output and copy alive components one module at a time,
    # freeing each module's tensor after copying to keep peak memory ~= 1x total size.
    first_act = next(iter(activations_.values()))
    n_samples = first_act.shape[0]
    dtype = first_act.dtype
    act_filtered = torch.empty(n_samples, total_alive, dtype=dtype)

    offset = 0
    alive_labels = ComponentLabels(list())
    dead_labels = ComponentLabels(list())
    module_alive_counts: dict[str, int] = {}

    for key in list(activations_.keys()):
        tensor = activations_.pop(key)
        c = tensor.shape[-1]

        if filter_dead_threshold > 0:
            alive = alive_masks[key]
            n_alive = int(alive.sum().item())
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
        del tensor

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
