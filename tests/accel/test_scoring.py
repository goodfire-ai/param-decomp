import importlib.util
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from pytest import MonkeyPatch

from param_decomp.accel import AccelBackendUnavailable, score_rank_one_linear_components
from param_decomp.accel.scoring import Backend

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]


def _fixture() -> tuple[FloatArray, IntArray, FloatArray, FloatArray, FloatArray]:
    inputs = np.array([[1.0, 2.0], [0.5, -1.0], [1.5, 0.0]], dtype=np.float32)
    labels = np.array([0, 1, 0], dtype=np.int64)
    weights = np.array([[1.0, 0.25], [-0.5, 1.0]], dtype=np.float32)
    bias = np.array([0.1, -0.2], dtype=np.float32)
    reference_logits = inputs @ weights.T + bias
    components_u = np.array([[0.25, -0.5], [1.0, 0.25]], dtype=np.float32)
    components_v = np.array([[0.5, 1.0], [-0.25, 0.75]], dtype=np.float32)
    return inputs, labels, reference_logits, components_u, components_v


def _rust_extension_available() -> bool:
    return importlib.util.find_spec("param_decomp_accel") is not None


def _score(
    *,
    inputs: FloatArray,
    labels: IntArray,
    reference_logits: FloatArray,
    components_u: FloatArray,
    components_v: FloatArray,
    component_ids: Sequence[str] | None = None,
    metric_reference_logits: FloatArray | None = None,
    row_indices: IntArray | None = None,
    slice_names: Sequence[str] | None = None,
    slice_offsets: IntArray | None = None,
    slice_indices: IntArray | None = None,
    backend: Backend = "python",
    rust_threads: int | None = None,
) -> list[dict[str, Any]]:
    return score_rank_one_linear_components(
        inputs=inputs,
        labels=labels,
        reference_logits=reference_logits,
        metric_reference_logits=metric_reference_logits,
        components_u=components_u,
        components_v=components_v,
        component_ids=component_ids,
        row_indices=row_indices,
        slice_names=slice_names,
        slice_offsets=slice_offsets,
        slice_indices=slice_indices,
        backend=backend,
        rust_threads=rust_threads,
    )


def test_python_rank_one_linear_component_scoring_has_expected_shape() -> None:
    inputs, labels, reference_logits, components_u, components_v = _fixture()

    records = _score(
        inputs=inputs,
        labels=labels,
        reference_logits=reference_logits,
        components_u=components_u,
        components_v=components_v,
        component_ids=["layer:0", "layer:1"],
        backend="python",
    )

    assert [record["component_id"] for record in records] == ["layer:0", "layer:1"]
    assert records[0]["sample_row_count"] == 3
    assert records[0]["ablated_behavior_mse"] > 0
    assert 0 <= records[0]["ablated_accuracy"] <= 1
    assert 0 <= records[0]["ablated_original_pred_match_rate"] <= 1
    assert records[0]["slice_metrics"] == {}


def test_default_backend_is_python_even_if_rust_is_installed(monkeypatch: MonkeyPatch) -> None:
    inputs, labels, reference_logits, components_u, components_v = _fixture()
    monkeypatch.delenv("PARAM_DECOMP_ACCEL", raising=False)

    default_records = score_rank_one_linear_components(
        inputs=inputs,
        labels=labels,
        reference_logits=reference_logits,
        components_u=components_u,
        components_v=components_v,
        component_ids=["layer:0", "layer:1"],
    )
    python_records = score_rank_one_linear_components(
        inputs=inputs,
        labels=labels,
        reference_logits=reference_logits,
        components_u=components_u,
        components_v=components_v,
        component_ids=["layer:0", "layer:1"],
        backend="python",
    )

    assert default_records == python_records


def test_python_rank_one_linear_component_scoring_emits_slice_metrics_with_global_rows() -> None:
    inputs, labels, reference_logits, components_u, components_v = _fixture()

    records = _score(
        inputs=inputs,
        labels=labels,
        reference_logits=reference_logits,
        components_u=components_u,
        components_v=components_v,
        component_ids=["layer:0", "layer:1"],
        row_indices=np.array([20, 21, 22], dtype=np.int64),
        slice_names=["first_two", "last"],
        slice_offsets=np.array([0, 2, 3], dtype=np.int64),
        slice_indices=np.array([0, 1, 2], dtype=np.int64),
        backend="python",
    )

    slice_metrics = records[0]["slice_metrics"]
    assert slice_metrics["first_two"]["count"] == 2
    assert slice_metrics["first_two"]["global_rows"] == [20, 21]
    assert slice_metrics["last"]["count"] == 1
    assert slice_metrics["last"]["global_rows"] == [22]


def test_metric_reference_logits_can_differ_from_base_logits() -> None:
    inputs, labels, reference_logits, components_u, components_v = _fixture()
    metric_reference_logits = reference_logits + np.array([[0.25, 0.0], [0.0, -0.5], [0.5, 0.0]], dtype=np.float32)

    python_records = _score(
        inputs=inputs,
        labels=labels,
        reference_logits=reference_logits,
        metric_reference_logits=metric_reference_logits,
        components_u=components_u,
        components_v=components_v,
        component_ids=["layer:0", "layer:1"],
        backend="python",
    )

    if not _rust_extension_available():
        return

    rust_records = _score(
        inputs=inputs,
        labels=labels,
        reference_logits=reference_logits,
        metric_reference_logits=metric_reference_logits,
        components_u=components_u,
        components_v=components_v,
        component_ids=["layer:0", "layer:1"],
        backend="rust",
        rust_threads=2,
    )
    for rust_record, python_record in zip(rust_records, python_records, strict=True):
        assert rust_record["ablated_behavior_mse"] == pytest.approx(
            python_record["ablated_behavior_mse"],
            abs=1e-6,
        )


def test_python_rank_one_linear_component_scoring_validates_shapes() -> None:
    inputs, labels, reference_logits, components_u, components_v = _fixture()

    with pytest.raises(ValueError, match="component_ids"):
        score_rank_one_linear_components(
            inputs=inputs,
            labels=labels,
            reference_logits=reference_logits,
            components_u=components_u,
            components_v=components_v,
            component_ids=["only-one"],
            backend="python",
        )


def test_explicit_rust_backend_fails_cleanly_when_extension_missing() -> None:
    inputs, labels, reference_logits, components_u, components_v = _fixture()

    if not _rust_extension_available():
        with pytest.raises(AccelBackendUnavailable):
            _score(
                inputs=inputs,
                labels=labels,
                reference_logits=reference_logits,
                components_u=components_u,
                components_v=components_v,
                backend="rust",
            )
    else:
        records = _score(
            inputs=inputs,
            labels=labels,
            reference_logits=reference_logits,
            components_u=components_u,
            components_v=components_v,
            backend="rust",
        )
        assert len(records) == 2


def test_rust_backend_matches_python_when_installed() -> None:
    inputs, labels, reference_logits, components_u, components_v = _fixture()

    if not _rust_extension_available():
        pytest.skip("param_decomp_accel is not installed")

    row_indices = np.array([20, 21, 22], dtype=np.int64)
    slice_offsets = np.array([0, 2, 3], dtype=np.int64)
    slice_indices = np.array([0, 1, 2], dtype=np.int64)
    python_records = _score(
        inputs=inputs,
        labels=labels,
        reference_logits=reference_logits,
        components_u=components_u,
        components_v=components_v,
        component_ids=["layer:0", "layer:1"],
        row_indices=row_indices,
        slice_names=["first_two", "last"],
        slice_offsets=slice_offsets,
        slice_indices=slice_indices,
        backend="python",
    )
    rust_records = _score(
        inputs=inputs,
        labels=labels,
        reference_logits=reference_logits,
        components_u=components_u,
        components_v=components_v,
        component_ids=["layer:0", "layer:1"],
        row_indices=row_indices,
        slice_names=["first_two", "last"],
        slice_offsets=slice_offsets,
        slice_indices=slice_indices,
        backend="rust",
        rust_threads=2,
    )

    assert len(rust_records) == len(python_records)
    for rust_record, python_record in zip(rust_records, python_records, strict=True):
        assert rust_record["component_id"] == python_record["component_id"]
        assert rust_record["component_index"] == python_record["component_index"]
        assert rust_record["sample_row_count"] == python_record["sample_row_count"]
        for key in (
            "ablated_behavior_mse",
            "ablated_task_loss",
            "ablated_accuracy",
            "ablated_original_pred_match_rate",
        ):
            assert rust_record[key] == pytest.approx(python_record[key], abs=1e-6)
        assert rust_record["slice_metrics"].keys() == python_record["slice_metrics"].keys()
        for slice_name, rust_slice in rust_record["slice_metrics"].items():
            python_slice = python_record["slice_metrics"][slice_name]
            assert rust_slice["count"] == python_slice["count"]
            assert rust_slice["global_rows"] == python_slice["global_rows"]
            for key in (
                "ablated_behavior_mse",
                "ablated_task_loss",
                "ablated_accuracy",
                "ablated_original_pred_match_rate",
            ):
                assert rust_slice[key] == pytest.approx(python_slice[key], abs=1e-6)
