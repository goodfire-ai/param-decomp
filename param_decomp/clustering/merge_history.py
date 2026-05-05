import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

import numpy as np
import torch
from jaxtyping import Float, Int

from param_decomp.clustering.consts import (
    ComponentLabels,
    DistancesArray,
    DistancesMethod,
    MergePair,
    MergesArray,
    SaveableObject,
)
from param_decomp.clustering.math.merge_distances import compute_distances
from param_decomp.clustering.math.merge_matrix import BatchedGroupMerge, GroupMerge
from param_decomp.clustering.merge_config import MergeConfig


@dataclass(frozen=True)
class IterationInfo:
    """Information about a single merge iteration."""

    idx: int
    selected_pair: list[int]
    merges: GroupMerge


def _zip_save_arr(zf: zipfile.ZipFile, name: str, arr: np.ndarray) -> None:
    """Save a numpy array to a zip file."""
    buf: io.BytesIO = io.BytesIO()
    np.save(buf, arr)
    zf.writestr(name, buf.getvalue())


def _zip_save_arr_dict(zf: zipfile.ZipFile, data: dict[str, np.ndarray]) -> None:
    """Save a dictionary of numpy arrays to a zip file, {key}.npy used as path"""
    key: str
    arr: np.ndarray
    for key, arr in data.items():
        _zip_save_arr(zf, f"{key}.npy", arr)


@dataclass(kw_only=True)
class MergeHistory(SaveableObject):
    """Track merge iteration history"""

    merges: BatchedGroupMerge
    selected_pairs: Int[np.ndarray, " n_iters 2"]
    labels: ComponentLabels
    merge_config: MergeConfig
    n_iters_current: int

    meta: dict[str, Any] | None = None

    @property
    def c_components(self) -> int:
        return len(self.labels)

    @classmethod
    def from_config(
        cls,
        merge_config: MergeConfig,
        labels: ComponentLabels,
    ) -> "MergeHistory":
        n_components: int = len(labels)
        n_iters_target: int = merge_config.get_num_iters(n_components)
        return MergeHistory(
            labels=labels,
            n_iters_current=0,
            selected_pairs=np.full((n_iters_target, 2), -1, dtype=np.int16),
            merges=BatchedGroupMerge.init_empty(
                batch_size=n_iters_target, n_components=n_components
            ),
            merge_config=merge_config,
        )

    def summary(self) -> dict[str, str | int | None | dict[str, int | str | None]]:
        return dict(
            c_components=self.c_components,
            n_iters_current=self.n_iters_current,
            total_iters=len(self.merges.k_groups),
            len_labels=len(self.labels),
            # wandb_url=self.wandb_url,
            merge_config=self.merge_config.model_dump(mode="json"),
            merges_summary=self.merges.summary(),
        )

    @override
    def __str__(self) -> str:
        out: list[str] = [f"  {key} = {value}" for key, value in self.summary().items()]
        return "MergeHistory(\n" + "\n".join(out) + "\n)"

    @override
    def __repr__(self) -> str:
        return self.__str__()

    def add_iteration(
        self,
        idx: int,
        selected_pair: MergePair,
        current_merge: GroupMerge,
    ) -> None:
        """Add data for one iteration."""
        self.selected_pairs[idx] = np.array(selected_pair, dtype=np.int16)
        self.merges[idx] = current_merge

        assert self.n_iters_current == idx
        self.n_iters_current += 1

    def __getitem__(self, idx: int) -> IterationInfo:
        """Get data for a specific iteration."""
        if idx < 0 or idx >= self.n_iters_current:
            raise IndexError(
                f"Index {idx} out of range for history with {self.n_iters_current} iterations"
            )

        return IterationInfo(
            idx=idx,
            selected_pair=self.selected_pairs[idx].tolist(),
            merges=self.merges[idx],
        )

    def __len__(self) -> int:
        """Get the number of iterations in the history."""
        return self.n_iters_current

    def latest(self) -> IterationInfo:
        """Get the latest values."""
        if self.n_iters_current == 0:
            raise ValueError("No history available")
        latest_idx: int = self.n_iters_current - 1
        return self[latest_idx]

    def get_unique_clusters(self, iteration: int) -> list[int]:
        """Get unique cluster IDs at a given iteration.

        Args:
            iteration: Iteration index (negative indexes from end)

        Returns:
            List of unique cluster IDs
        """
        if iteration < 0:
            iteration = self.n_iters_current + iteration
        assert 0 <= iteration < self.n_iters_current, (
            f"Invalid iteration: {iteration = }, {self.n_iters_current = }"
        )
        merge: GroupMerge = self.merges[iteration]
        return torch.unique(merge.group_idxs).tolist()

    def get_cluster_component_labels(self, iteration: int, cluster_id: int) -> ComponentLabels:
        """Get component labels for a specific cluster at a given iteration.

        Args:
            iteration: Iteration index (negative indexes from end)
            cluster_id: Cluster ID to query

        Returns:
            List of component labels in the cluster
        """
        if iteration < 0:
            iteration = self.n_iters_current + iteration
        assert 0 <= iteration < self.n_iters_current, (
            f"Invalid iteration: {iteration = }, {self.n_iters_current = }"
        )
        merge: GroupMerge = self.merges[iteration]
        component_indices: list[int] = merge.components_in_group(cluster_id)
        return ComponentLabels([self.labels[idx] for idx in component_indices])

    def get_cluster_components_info(self, iteration: int, cluster_id: int) -> list[dict[str, Any]]:
        """Get detailed component information for a cluster.

        Args:
            iteration: Iteration index (negative indexes from end)
            cluster_id: Cluster ID to query

        Returns:
            List of dicts with keys: module, index, label
        """
        component_labels: list[str] = self.get_cluster_component_labels(iteration, cluster_id)
        result: list[dict[str, Any]] = []
        for label in component_labels:
            module: str
            idx_str: str
            module, idx_str = label.rsplit(":", 1)
            result.append({"module": module, "index": int(idx_str), "label": label})
        return result

    # Convenience properties for sweep analysis
    @property
    def total_iterations(self) -> int:
        """Total number of iterations performed."""
        return self.n_iters_current

    @property
    def final_k_groups(self) -> int:
        """Final number of groups after merging."""
        if self.n_iters_current == 0:
            return self.c_components
        return int(self.merges.k_groups[self.n_iters_current - 1].item())

    @property
    def initial_k_groups(self) -> int:
        """Initial number of groups before merging."""
        if self.n_iters_current == 0:
            return self.c_components
        return int(self.merges.k_groups[0].item())

    @override
    def save(self, path: Path) -> None:
        zf: zipfile.ZipFile
        with zipfile.ZipFile(path, "w") as zf:
            # save arrays
            _zip_save_arr_dict(
                zf=zf,
                data={
                    "merge.group_idxs": self.merges.group_idxs.cpu().numpy(),
                    "merge.k_groups": self.merges.k_groups.cpu().numpy(),
                    "selected_pairs": self.selected_pairs,
                },
            )
            # Save labels
            zf.writestr("labels.txt", "\n".join(self.labels))
            # Save metadata
            zf.writestr(
                "metadata.json",
                json.dumps(
                    dict(
                        merge_config=self.merge_config.model_dump(mode="json"),
                        c_components=self.c_components,
                        n_iters_current=self.n_iters_current,
                        labels=self.labels,
                    )
                ),
            )

    @override
    @classmethod
    def read(cls, path: Path) -> "MergeHistory":
        zf: zipfile.ZipFile
        with zipfile.ZipFile(path, "r") as zf:
            group_idxs: np.ndarray = np.load(io.BytesIO(zf.read("merge.group_idxs.npy")))
            k_groups: np.ndarray = np.load(io.BytesIO(zf.read("merge.k_groups.npy")))
            selected_pairs: np.ndarray = np.load(io.BytesIO(zf.read("selected_pairs.npy")))
            merges: BatchedGroupMerge = BatchedGroupMerge(
                group_idxs=torch.from_numpy(group_idxs),
                k_groups=torch.from_numpy(k_groups),
            )
            labels_raw: list[str] = zf.read("labels.txt").decode("utf-8").splitlines()
            labels: ComponentLabels = ComponentLabels(labels_raw)
            metadata: dict[str, Any] = json.loads(zf.read("metadata.json").decode("utf-8"))
            merge_config: MergeConfig = MergeConfig.model_validate(metadata["merge_config"])

        metadata["origin_path"] = path

        return cls(
            merges=merges,
            selected_pairs=selected_pairs,
            labels=labels,
            merge_config=merge_config,
            n_iters_current=metadata["n_iters_current"],
            meta=metadata,
        )


@dataclass
class MergeHistoryEnsemble:
    data: list[MergeHistory]

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, idx: int) -> MergeHistory:
        return self.data[idx]

    def _validate_configs_match(self) -> None:
        """Ensure all histories have the same merge config."""
        if not self.data:
            return
        first_config: MergeConfig = self.data[0].merge_config
        for history in self.data[1:]:
            if history.merge_config != first_config:
                raise ValueError("All histories must have the same merge config")

    @property
    def config(self) -> MergeConfig:
        """Get the merge config used in the ensemble."""
        self._validate_configs_match()
        return self.data[0].merge_config

    @property
    def n_iters_min(self) -> int:
        """Minimum number of iterations across all histories in the ensemble."""
        return min(len(history.merges.k_groups) for history in self.data)

    @property
    def n_iters_max(self) -> int:
        """Maximum number of iterations across all histories in the ensemble."""
        return max(len(history.merges.k_groups) for history in self.data)

    @property
    def n_iters_range(self) -> tuple[int, int]:
        """Range of iterations (min, max) across all histories in the ensemble."""
        iter_counts = [len(history.merges.k_groups) for history in self.data]
        return (min(iter_counts), max(iter_counts))

    @property
    def n_ensemble(self) -> int:
        """Number of ensemble members."""
        return len(self.data)

    @property
    def c_components(self) -> int:
        """Number of components in each history."""
        c_components: int = self.data[0].c_components
        assert all(history.c_components == c_components for history in self.data), (
            "All histories must have the same number of components"
        )
        return c_components

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of the ensemble data."""
        return (self.n_ensemble, self.n_iters_min, self.c_components)

    @property
    def merges_array(self) -> MergesArray:
        n_ens: int = self.n_ensemble
        n_iters: int = self.n_iters_min
        c_components: int = self.c_components

        output: MergesArray = np.full(
            (n_ens, n_iters, c_components),
            fill_value=-1,
            dtype=np.int16,
            # if you have more than 32k components, change this to np.int32
            # if you have more than 2.1b components, rethink your life choices
        )
        for i_ens, history in enumerate(self.data):
            for i_iter, merge in enumerate(history.merges):
                output[i_ens, i_iter] = merge.group_idxs

        return output

    def normalized(self) -> tuple[MergesArray, dict[str, Any]]:
        """Normalize the component labels across all histories.

        if different histories see different batches, then they might have different dead
        components, and are hence not directly comparable. So, we find the union of all
        component labels across all histories, and then any component missing from a history
        is put into it's own group in that history
        """

        unique_labels_set: set[str] = set()
        for history in self.data:
            unique_labels_set.update(history.labels)

        unique_labels_list: list[str] = sorted(unique_labels_set)
        unique_labels: ComponentLabels = ComponentLabels(unique_labels_list)
        c_components: int = len(unique_labels)
        component_label_idxs: dict[str, int] = {
            label: idx for idx, label in enumerate(unique_labels)
        }

        try:
            merges_array: MergesArray = np.full(
                (self.n_ensemble, self.n_iters_min, c_components),
                fill_value=-1,
                dtype=np.int16,
            )
        except Exception as e:
            err_msg = (
                f"failed to create merge array, probably due to issues with getting shape.\n"
                f"{self = }\n"
                f"{self.data = }\n"
            )
            raise RuntimeError(err_msg) from e

        overlap_stats: Float[np.ndarray, " n_ens"] = np.full(
            self.n_ensemble,
            fill_value=float("nan"),
            dtype=np.float32,
        )
        i_ens: int
        history: MergeHistory
        for i_ens, history in enumerate(self.data):
            hist_c_labels: list[str] = history.labels
            hist_n_components: int = len(hist_c_labels)
            overlap_stats[i_ens] = hist_n_components / c_components
            # map from old component indices to new component indices
            i_comp_old: int
            comp_label: str
            for i_comp_old, comp_label in enumerate(hist_c_labels):
                i_comp_new: int = component_label_idxs[comp_label]
                merges_array[i_ens, :, i_comp_new] = history.merges.group_idxs[
                    : self.n_iters_min, i_comp_old
                ]

            # assert np.max(merges_array[i_ens]) == hist_n_components - 1, (
            #     f"Max component index in history {i_ens} should be {hist_n_components - 1}, "
            #     f"but got {np.max(merges_array[i_ens])}"
            # )

            # put each missing label into its own group
            hist_missing_labels: set[str] = unique_labels_set - set(hist_c_labels)
            assert len(hist_missing_labels) == c_components - hist_n_components
            idx_missing: int
            missing_label: str
            for idx_missing, missing_label in enumerate(hist_missing_labels):
                i_comp_new_relabel: int = component_label_idxs[missing_label]
                merges_array[i_ens, :, i_comp_new_relabel] = np.full(
                    self.n_iters_min,
                    fill_value=idx_missing + hist_n_components,
                    dtype=np.int16,
                )

        # TODO: double check this
        # Convert any Path objects to strings for JSON serialization
        history_metadatas: list[dict[str, Any] | None] = []
        for history in self.data:
            if history.meta is not None:
                meta_copy = history.meta.copy()
                # Convert Path objects to strings
                for key, value in meta_copy.items():
                    if isinstance(value, Path):
                        meta_copy[key] = str(value)
                history_metadatas.append(meta_copy)
            else:
                history_metadatas.append(None)

        return (
            # TODO: dataclass this
            merges_array,
            dict(
                component_labels=unique_labels,
                n_ensemble=self.n_ensemble,
                n_iters_min=self.n_iters_min,
                n_iters_max=self.n_iters_max,
                n_iters_range=self.n_iters_range,
                c_components=c_components,
                config=self.config.model_dump(mode="json"),
                history_metadatas=history_metadatas,
            ),
        )

    def get_distances(self, method: DistancesMethod = "perm_invariant_hamming") -> DistancesArray:
        merges_array: MergesArray = self.merges_array
        return compute_distances(
            normalized_merge_array=merges_array,
            method=method,
        )
