"""Rank-one component scoring with an optional Rust backend.

Python/torch code should export the activation matrix for the decomposed linear
map, reference logits, labels, and rank-one component factors. The hot loop here
scores single-component ablations:

    ablated_logits = reference_logits - (inputs @ v)[:, None] * u[None, :]

That keeps decomposition training/orchestration in Python while giving bootstrap
scoring a compact seam that can be moved to Rust.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

Backend = Literal["auto", "python", "rust"]


class AccelBackendUnavailable(RuntimeError):
    """Raised when an explicitly requested acceleration backend cannot run."""


def score_rank_one_linear_components(
    *,
    inputs: np.ndarray,
    labels: np.ndarray,
    reference_logits: np.ndarray,
    components_u: np.ndarray,
    components_v: np.ndarray,
    component_ids: Sequence[str] | None = None,
    row_indices: np.ndarray | Sequence[int] | None = None,
    slice_names: Sequence[str] | None = None,
    slice_offsets: np.ndarray | Sequence[int] | None = None,
    slice_indices: np.ndarray | Sequence[int] | None = None,
    backend: Backend | None = None,
    rust_threads: int | None = None,
) -> list[dict[str, Any]]:
    """Score rank-one linear component ablations.

    Args:
        inputs: Float array with shape ``[n_rows, in_dim]``.
        labels: Int array with shape ``[n_rows]``.
        reference_logits: Float array with shape ``[n_rows, out_dim]``.
        components_u: Float array with shape ``[n_components, out_dim]``.
        components_v: Float array with shape ``[n_components, in_dim]``.
        component_ids: Stable component IDs. Defaults to stringified indices.
        row_indices: Optional global row IDs for the sampled rows.
        slice_names/slice_offsets/slice_indices: Optional CSR slice encoding.
        backend: ``python`` is the default. ``rust`` opts into the extension and
            fails if unavailable. ``auto`` uses Rust if installed, else Python.
        rust_threads: Optional Rayon thread count for the Rust backend.
    """

    selected_backend = _resolve_backend(backend)
    arrays = _coerce_inputs(
        inputs=inputs,
        labels=labels,
        reference_logits=reference_logits,
        components_u=components_u,
        components_v=components_v,
        row_indices=row_indices,
        slice_offsets=slice_offsets,
        slice_indices=slice_indices,
    )
    ids = list(component_ids) if component_ids is not None else [str(i) for i in range(arrays["components_u"].shape[0])]
    _validate_bundle(
        arrays["inputs"],
        arrays["labels"],
        arrays["reference_logits"],
        arrays["components_u"],
        arrays["components_v"],
        ids,
        arrays["row_indices"],
        slice_names,
        arrays["slice_offsets"],
        arrays["slice_indices"],
    )

    if selected_backend in ("auto", "rust"):
        try:
            import param_decomp_accel as rust_accel
        except ImportError as exc:
            if selected_backend == "rust":
                raise AccelBackendUnavailable(
                    "Rust acceleration requested but param_decomp_accel is not installed. "
                    "Build it with: python -m pip install maturin && "
                    "python -m maturin develop --manifest-path crates/param_decomp_accel/Cargo.toml"
                ) from exc
        else:
            return rust_accel.score_rank_one_linear_components(
                arrays["inputs"],
                arrays["labels"],
                arrays["reference_logits"],
                arrays["components_u"],
                arrays["components_v"],
                ids,
                arrays["row_indices"],
                list(slice_names or []),
                arrays["slice_offsets"],
                arrays["slice_indices"],
                int(rust_threads or 0),
            )

    return _score_rank_one_linear_components_python(
        inputs=arrays["inputs"],
        labels=arrays["labels"],
        reference_logits=arrays["reference_logits"],
        components_u=arrays["components_u"],
        components_v=arrays["components_v"],
        component_ids=ids,
        row_indices=arrays["row_indices"],
        slice_names=list(slice_names or []),
        slice_offsets=arrays["slice_offsets"],
        slice_indices=arrays["slice_indices"],
    )


def _resolve_backend(backend: Backend | None) -> Backend:
    raw = backend or os.environ.get("PARAM_DECOMP_ACCEL", "python")
    if raw not in {"auto", "python", "rust"}:
        raise ValueError(f"Unsupported acceleration backend: {raw!r}")
    return raw  # type: ignore[return-value]


def _coerce_inputs(**kwargs: Any) -> dict[str, np.ndarray]:
    return {
        "inputs": np.asarray(kwargs["inputs"], dtype=np.float32, order="C"),
        "labels": np.asarray(kwargs["labels"], dtype=np.int64, order="C"),
        "reference_logits": np.asarray(kwargs["reference_logits"], dtype=np.float32, order="C"),
        "components_u": np.asarray(kwargs["components_u"], dtype=np.float32, order="C"),
        "components_v": np.asarray(kwargs["components_v"], dtype=np.float32, order="C"),
        "row_indices": np.asarray(
            np.arange(np.asarray(kwargs["inputs"]).shape[0]) if kwargs["row_indices"] is None else kwargs["row_indices"],
            dtype=np.int64,
            order="C",
        ),
        "slice_offsets": np.asarray([] if kwargs["slice_offsets"] is None else kwargs["slice_offsets"], dtype=np.int64, order="C"),
        "slice_indices": np.asarray([] if kwargs["slice_indices"] is None else kwargs["slice_indices"], dtype=np.int64, order="C"),
    }


def _validate_bundle(
    inputs: np.ndarray,
    labels: np.ndarray,
    reference_logits: np.ndarray,
    components_u: np.ndarray,
    components_v: np.ndarray,
    component_ids: Sequence[str],
    row_indices: np.ndarray,
    slice_names: Sequence[str] | None,
    slice_offsets: np.ndarray,
    slice_indices: np.ndarray,
) -> None:
    if inputs.ndim != 2:
        raise ValueError("inputs must have shape [n_rows, in_dim]")
    if labels.ndim != 1 or labels.shape[0] != inputs.shape[0]:
        raise ValueError("labels must have shape [n_rows]")
    if components_u.ndim != 2 or components_v.ndim != 2:
        raise ValueError("component factors must be rank-2 arrays")
    if reference_logits.shape != (inputs.shape[0], components_u.shape[1]):
        raise ValueError("reference_logits must have shape [n_rows, out_dim]")
    if components_u.shape[0] != components_v.shape[0] or components_v.shape[1] != inputs.shape[1]:
        raise ValueError("component factors must have shapes [n_components, out_dim] and [n_components, in_dim]")
    if len(component_ids) != components_u.shape[0]:
        raise ValueError("component_ids length must equal n_components")
    if row_indices.shape != (inputs.shape[0],):
        raise ValueError("row_indices must have shape [n_rows]")
    if np.any(labels < 0) or np.any(labels >= reference_logits.shape[1]):
        raise ValueError("labels must be valid class indices for reference_logits")
    if bool(slice_names) != bool(len(slice_offsets)):
        raise ValueError("slice_names and slice_offsets must either both be present or both be omitted")
    if slice_names and len(slice_offsets) != len(slice_names) + 1:
        raise ValueError("slice_offsets length must be len(slice_names) + 1")
    if len(slice_offsets) and (slice_offsets[0] != 0 or slice_offsets[-1] != len(slice_indices)):
        raise ValueError("slice_offsets must start at 0 and end at len(slice_indices)")
    if len(slice_indices) and (np.any(slice_indices < 0) or np.any(slice_indices >= inputs.shape[0])):
        raise ValueError("slice_indices are local row indices and must be in [0, n_rows)")


def _score_rank_one_linear_components_python(
    *,
    inputs: np.ndarray,
    labels: np.ndarray,
    reference_logits: np.ndarray,
    components_u: np.ndarray,
    components_v: np.ndarray,
    component_ids: Sequence[str],
    row_indices: np.ndarray,
    slice_names: Sequence[str],
    slice_offsets: np.ndarray,
    slice_indices: np.ndarray,
) -> list[dict[str, Any]]:
    ref_pred = np.argmax(reference_logits, axis=1)
    records: list[dict[str, Any]] = []
    for component_index, component_id in enumerate(component_ids):
        contribution_scale = inputs @ components_v[component_index]
        ablated_logits = reference_logits - contribution_scale[:, None] * components_u[component_index][None, :]
        metrics = _metrics(reference_logits, ablated_logits, labels, ref_pred)
        record: dict[str, Any] = {
            "component_id": component_id,
            "component_index": component_index,
            "sample_row_count": int(inputs.shape[0]),
            **metrics,
        }
        if slice_names:
            record["slice_metrics"] = _slice_metrics(
                reference_logits,
                ablated_logits,
                labels,
                ref_pred,
                slice_names,
                slice_offsets,
                slice_indices,
                row_indices,
            )
        else:
            record["slice_metrics"] = {}
        records.append(record)
    return records


def _metrics(reference_logits: np.ndarray, ablated_logits: np.ndarray, labels: np.ndarray, ref_pred: np.ndarray) -> dict[str, float]:
    pred = np.argmax(ablated_logits, axis=1)
    return {
        "ablated_behavior_mse": round(float(np.mean((ablated_logits - reference_logits) ** 2)), 10),
        "ablated_task_loss": round(float(_cross_entropy(ablated_logits, labels)), 8),
        "ablated_accuracy": round(float(np.mean(pred == labels)), 8),
        "ablated_original_pred_match_rate": round(float(np.mean(pred == ref_pred)), 8),
    }


def _cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=1)) + np.max(logits, axis=1)
    return float(np.mean(logsumexp - logits[np.arange(labels.shape[0]), labels]))


def _slice_metrics(
    reference_logits: np.ndarray,
    ablated_logits: np.ndarray,
    labels: np.ndarray,
    ref_pred: np.ndarray,
    slice_names: Sequence[str],
    slice_offsets: np.ndarray,
    slice_indices: np.ndarray,
    row_indices: np.ndarray,
) -> dict[str, dict[str, float | int | list[int]]]:
    output: dict[str, dict[str, float | int | list[int]]] = {}
    for index, name in enumerate(slice_names):
        start = int(slice_offsets[index])
        end = int(slice_offsets[index + 1])
        local_rows = slice_indices[start:end]
        if len(local_rows) == 0:
            continue
        metrics = _metrics(reference_logits[local_rows], ablated_logits[local_rows], labels[local_rows], ref_pred[local_rows])
        output[name] = {
            "count": int(len(local_rows)),
            "global_rows": [int(row_indices[row]) for row in local_rows],
            **metrics,
        }
    return output
