"""Each input format validates its own corpus representation."""

from pathlib import Path

import jax
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from jax.sharding import AxisType, Mesh
from pydantic import ValidationError

from param_decomp.experiments.lm.input_format import (
    DocumentInput,
    TokenInput,
    input_sampler,
    transformer_input_format,
)
from param_decomp.infra.dataset_store import DatasetIdentity, DatasetMeta
from param_decomp.targets.testing import (
    tiny_glu_cfg,
    tiny_glu_decomposed_lm,
    tiny_simple_mlp_cfg,
    tiny_simple_mlp_decomposed_model,
)
from param_decomp.tests.targets.test_qwen3 import _tiny_decomposed_qwen, _tiny_qwen_cfg


def _write_corpus(directory: Path, metadata: DatasetIdentity, documents: bool) -> None:
    (directory / "meta.json").write_text(metadata.model_dump_json())
    columns = {"input_ids": pa.array([[1, 2, 3, 4]] * 4, type=pa.list_(pa.int32()))}
    if documents:
        columns["document_ids"] = pa.array([[0, 0, 1, 1]] * 4, type=pa.list_(pa.int32()))
    pq.write_table(pa.table(columns), directory / "train-00000-of-00001.parquet")


def _mesh() -> Mesh:
    return Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1),
        ("data", "tp"),
        axis_types=(AxisType.Explicit,) * 2,
    )


def test_token_input_reads_legacy_metadata_without_rewriting_it(tmp_path: Path) -> None:
    _write_corpus(tmp_path, DatasetIdentity(seq_len=4, tokenizer_name="test"), documents=False)
    before = (tmp_path / "meta.json").read_bytes()
    mesh = _mesh()
    with jax.set_mesh(mesh):
        sample = input_sampler(TokenInput(), tmp_path, 2, 3, mesh, 10)
        np.testing.assert_array_equal(sample(0).token_ids, [[1, 2, 3, 4]] * 2)
        np.testing.assert_array_equal(sample(5).token_ids, [[1, 2, 3, 4]] * 2)
    assert (tmp_path / "meta.json").read_bytes() == before


@pytest.mark.parametrize("versioned", [False, True])
def test_token_input_rejects_document_columns_regardless_of_metadata(
    tmp_path: Path, versioned: bool
) -> None:
    metadata = (
        DatasetMeta(seq_len=4, tokenizer_name="test", format_version=2, preprocessing_name="test")
        if versioned
        else DatasetIdentity(seq_len=4, tokenizer_name="test")
    )
    _write_corpus(tmp_path, metadata, documents=True)
    with pytest.raises(AssertionError, match="token-only inputs require exactly input_ids"):
        input_sampler(TokenInput(), tmp_path, 2, 3, _mesh(), 10)


def test_document_input_rejects_unversioned_metadata(tmp_path: Path) -> None:
    _write_corpus(tmp_path, DatasetIdentity(seq_len=4, tokenizer_name="test"), documents=True)
    with pytest.raises(ValidationError, match="format_version"):
        input_sampler(DocumentInput(), tmp_path, 2, 3, _mesh(), 10)


def test_document_input_rejects_missing_document_columns(tmp_path: Path) -> None:
    _write_corpus(
        tmp_path,
        DatasetMeta(seq_len=4, tokenizer_name="test", format_version=2, preprocessing_name="test"),
        documents=False,
    )
    sample = input_sampler(DocumentInput(), tmp_path, 2, 3, _mesh(), 10)
    with pytest.raises(AssertionError, match="shards require input_ids and document_ids"):
        sample(0)


def test_document_input_preserves_paired_inputs(tmp_path: Path) -> None:
    _write_corpus(
        tmp_path,
        DatasetMeta(seq_len=4, tokenizer_name="test", format_version=2, preprocessing_name="test"),
        documents=True,
    )
    mesh = _mesh()
    with jax.set_mesh(mesh):
        batch = input_sampler(DocumentInput(), tmp_path, 2, 3, mesh, 10)(0)
        np.testing.assert_array_equal(batch.batch.token_ids, [[1, 2, 3, 4]] * 2)
        np.testing.assert_array_equal(batch.sequence.document_ids, [[0, 0, 1, 1]] * 2)


@pytest.mark.parametrize("documents", [False, True])
def test_dense_qwen_input_requires_stored_document_ids(tmp_path: Path, documents: bool) -> None:
    model = _tiny_decomposed_qwen(_tiny_qwen_cfg(), (), jax.random.key(9))
    metadata = DatasetMeta(
        seq_len=4, tokenizer_name="test", format_version=2, preprocessing_name="test"
    )
    _write_corpus(tmp_path, metadata, documents=documents)
    mesh = _mesh()
    with jax.set_mesh(mesh):
        if documents:
            batch = input_sampler(transformer_input_format(model), tmp_path, 2, 3, mesh, 10)(0)
            np.testing.assert_array_equal(batch.batch.token_ids, [[1, 2, 3, 4]] * 2)
            np.testing.assert_array_equal(batch.sequence.document_ids, [[0, 0, 1, 1]] * 2)
        else:
            with pytest.raises(AssertionError, match="shards require input_ids and document_ids"):
                input_sampler(transformer_input_format(model), tmp_path, 2, 3, mesh, 10)(0)


def test_llama_input_preserves_documents(tmp_path: Path) -> None:
    model = tiny_glu_decomposed_lm(tiny_glu_cfg(), (), jax.random.key(9))
    _write_corpus(
        tmp_path,
        DatasetMeta(seq_len=4, tokenizer_name="test", format_version=2, preprocessing_name="test"),
        documents=True,
    )
    mesh = _mesh()
    with jax.set_mesh(mesh):
        batch = input_sampler(transformer_input_format(model), tmp_path, 2, 3, mesh, 10)(0)
        np.testing.assert_array_equal(batch.sequence.document_ids, [[0, 0, 1, 1]] * 2)


@pytest.mark.parametrize("versioned", [False, True])
def test_simple_mlp_input_preserves_unsegmented_token_rows(tmp_path: Path, versioned: bool) -> None:
    model = tiny_simple_mlp_decomposed_model(tiny_simple_mlp_cfg(), (), jax.random.key(9))
    metadata = (
        DatasetMeta(seq_len=4, tokenizer_name="test", format_version=2, preprocessing_name="test")
        if versioned
        else DatasetIdentity(seq_len=4, tokenizer_name="test")
    )
    _write_corpus(tmp_path, metadata, documents=False)
    before = (tmp_path / "meta.json").read_bytes()
    mesh = _mesh()
    with jax.set_mesh(mesh):
        sample = input_sampler(transformer_input_format(model), tmp_path, 2, 3, mesh, 10)
        for step in (0, 5):
            batch = sample(step)
            np.testing.assert_array_equal(batch.batch.token_ids, [[1, 2, 3, 4]] * 2)
            np.testing.assert_array_equal(batch.sequence.document_ids, np.zeros((2, 4)))
            np.testing.assert_array_equal(batch.sequence.position_ids(), [[0, 1, 2, 3]] * 2)
    assert (tmp_path / "meta.json").read_bytes() == before


def test_simple_mlp_input_rejects_document_columns(tmp_path: Path) -> None:
    model = tiny_simple_mlp_decomposed_model(tiny_simple_mlp_cfg(), (), jax.random.key(9))
    _write_corpus(
        tmp_path,
        DatasetMeta(seq_len=4, tokenizer_name="test", format_version=2, preprocessing_name="test"),
        documents=True,
    )
    with pytest.raises(AssertionError, match="does not support document-aware inputs"):
        input_sampler(transformer_input_format(model), tmp_path, 2, 3, _mesh(), 10)
