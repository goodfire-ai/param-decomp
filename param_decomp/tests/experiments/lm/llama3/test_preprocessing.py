"""Declared Llama preprocessing preserves text, document boundaries, and artifact identity."""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import Dataset
from pydantic import ValidationError
from tokenizers import Regex, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from param_decomp.experiments.lm.data import pack_documents
from param_decomp.experiments.lm.llama3.data import (
    Llama3Tokenizer,
    load_llama3_tokenizer,
    preprocess_llama3,
    preprocess_llama3_streaming,
)
from param_decomp.experiments.lm.llama3.prestage_tokenized import _prepare_output, prestage
from param_decomp.infra.dataset_store import DatasetMeta, read_dataset_meta, write_dataset_meta

_BOS = "<|begin_of_text|>"
_EOS = "<|end_of_text|>"
_EOT = "<|eot_id|>"


def _tokenizer(vocabulary: dict[str, int]) -> PreTrainedTokenizerFast:
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Split(Regex(r"[\s\S]"), behavior="isolated")
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token=_BOS,
        eos_token=_EOS,
        additional_special_tokens=[token for token, index in vocabulary.items() if index >= 128002],
    )


@pytest.fixture(scope="module")
def raw_tokenizer() -> PreTrainedTokenizerFast:
    """A character vocabulary with Llama's special-token IDs, not a Meta BPE replica."""
    vocabulary = {chr(index): index for index in range(128)}
    vocabulary["<unk>"] = 128
    vocabulary.update({f"piece_{index}": index for index in range(129, 128000)})
    vocabulary.update(
        {f"<|reserved_special_token_{index}|>": 128000 + index for index in range(256)}
    )
    for token, offset in ((_BOS, 0), (_EOS, 1), (_EOT, 9)):
        del vocabulary[f"<|reserved_special_token_{offset}|>"]
        vocabulary[token] = 128000 + offset
    return _tokenizer(vocabulary)


@pytest.fixture(scope="module")
def tokenizer(raw_tokenizer: PreTrainedTokenizerFast) -> Llama3Tokenizer:
    return Llama3Tokenizer(raw_tokenizer)


@pytest.mark.parametrize("text", ["", " alpha beta ", "\n\talpha\r\n", "alpha " * 10000])
def test_document_framing_preserves_exact_text(tokenizer: Llama3Tokenizer, text: str) -> None:
    assert tokenizer.encode_document(text) == (128000, *map(ord, text), 128001)


@pytest.mark.parametrize("literal", [_BOS, _EOS, _EOT, "<|reserved_special_token_42|>"])
def test_literal_special_spellings_are_ordinary_text(
    tokenizer: Llama3Tokenizer, literal: str
) -> None:
    text = f"alpha {literal} beta"
    encoded = tokenizer.encode_document(text)
    assert encoded == (128000, *map(ord, text), 128001)
    assert all(token_id < 128000 for token_id in encoded[1:-1])


@pytest.mark.parametrize("invalid", ["bos", "eos", "eot", "vocabulary"])
def test_non_llama_tokenizer_refuses(
    raw_tokenizer: PreTrainedTokenizerFast,
    invalid: Literal["bos", "eos", "eot", "vocabulary"],
) -> None:
    vocabulary = raw_tokenizer.get_vocab()
    match invalid:
        case "bos":
            vocabulary[_BOS], vocabulary[_EOS] = vocabulary[_EOS], vocabulary[_BOS]
        case "eos":
            other = "<|reserved_special_token_2|>"
            vocabulary[_EOS], vocabulary[other] = vocabulary[other], vocabulary[_EOS]
        case "eot":
            other = "<|reserved_special_token_10|>"
            vocabulary[_EOT], vocabulary[other] = vocabulary[other], vocabulary[_EOT]
        case "vocabulary":
            vocabulary["additional_token"] = 128256
    with pytest.raises(AssertionError):
        Llama3Tokenizer(_tokenizer(vocabulary))


def test_packer_preserves_documents_without_inserting_tokens() -> None:
    packed = pack_documents(((7, 8, 9), (10, 11), (12, 13, 14, 15)), max_length=4)
    np.testing.assert_array_equal(packed.token_ids, [[7, 8, 9, 10], [11, 12, 13, 14]])
    np.testing.assert_array_equal(packed.document_ids, [[0, 0, 0, 1], [1, 2, 2, 2]])
    assert packed.token_ids.dtype == packed.document_ids.dtype == np.int32


@pytest.mark.parametrize("length", [0, -1])
def test_packer_requires_positive_sequence_length(length: int) -> None:
    with pytest.raises(AssertionError):
        pack_documents(((1, 2),), max_length=length)


def test_incomplete_pack_has_typed_empty_rows() -> None:
    packed = pack_documents(((1, 2),), max_length=4)
    for values in (packed.token_ids, packed.document_ids):
        assert values.shape == (0, 4)
        assert values.dtype == np.int32


def test_document_tokenization_and_identity(tokenizer: Llama3Tokenizer) -> None:
    texts = ["alpha " * 137, f"beta {_EOS} alpha", "", "beta " * 93]
    dataset = Dataset.from_dict({"text": texts, "unused": [0] * len(texts)})
    packed = preprocess_llama3(dataset, tokenizer, column_name="text", max_length=7, num_proc=1)
    expected_tokens: list[int] = []
    expected_ids: list[int] = []
    for document_id, text in enumerate(texts):
        document = [128000, *map(ord, text), 128001]
        expected_tokens.extend(document)
        expected_ids.extend([document_id] * len(document))
    length = len(expected_tokens) // 7 * 7
    np.testing.assert_array_equal(np.asarray(packed["input_ids"]).ravel(), expected_tokens[:length])
    np.testing.assert_array_equal(np.asarray(packed["document_ids"]).ravel(), expected_ids[:length])
    assert set(packed.column_names) == {"input_ids", "document_ids"}


def test_streaming_packing_is_lazy_and_matches_materialized(tokenizer: Llama3Tokenizer) -> None:
    texts = ["alpha " * 137, f"beta {_EOS} alpha", "", "beta " * 93]
    consumed: list[str] = []

    def observe(row: dict[str, str]) -> dict[str, str]:
        consumed.append(row["text"])
        return row

    materialized = Dataset.from_dict({"text": texts})
    source = materialized.to_iterable_dataset().map(observe, features=materialized.features)
    packed = preprocess_llama3_streaming(source, tokenizer, column_name="text", max_length=7)
    assert consumed == []
    rows = list(packed)
    assert consumed == texts
    expected = preprocess_llama3(
        Dataset.from_dict({"text": texts}), tokenizer, column_name="text", max_length=7, num_proc=1
    )
    assert set(rows[0]) == {"input_ids", "document_ids"}
    for column in ("input_ids", "document_ids"):
        np.testing.assert_array_equal([row[column] for row in rows], expected[column])


@pytest.mark.parametrize("length", [0, -1])
def test_preprocessing_requires_positive_sequence_length(
    tokenizer: Llama3Tokenizer, length: int
) -> None:
    source = Dataset.from_dict({"text": ["alpha"]})
    with pytest.raises(AssertionError):
        preprocess_llama3(source, tokenizer, column_name="text", max_length=length, num_proc=1)
    with pytest.raises(AssertionError):
        preprocess_llama3_streaming(
            source.to_iterable_dataset(), tokenizer, column_name="text", max_length=length
        )


def test_incomplete_batch_has_no_rows(tokenizer: Llama3Tokenizer) -> None:
    packed = preprocess_llama3(
        Dataset.from_dict({"text": ["a"]}), tokenizer, column_name="text", max_length=8, num_proc=1
    )
    assert len(packed) == 0


def test_staging_refuses_unversioned_directory(tmp_path: Path) -> None:
    shard = tmp_path / "shard_00000.parquet"
    shard.write_bytes(b"old artifact")
    with pytest.raises(AssertionError, match="without document-aware metadata"):
        _prepare_output(
            tmp_path,
            DatasetMeta(
                format_version=2, preprocessing_name="fixture", seq_len=4, tokenizer_name="test"
            ),
        )
    assert not (tmp_path / "meta.json").exists()
    assert shard.read_bytes() == b"old artifact"


def test_staging_refuses_old_metadata_without_overwriting(tmp_path: Path) -> None:
    path = tmp_path / "meta.json"
    old = '{"seq_len": 4, "tokenizer_name": "test"}\n'
    path.write_text(old)
    with pytest.raises(ValidationError, match="format_version"):
        _prepare_output(
            tmp_path,
            DatasetMeta(
                format_version=2, preprocessing_name="fixture", seq_len=4, tokenizer_name="test"
            ),
        )
    assert path.read_text() == old


def test_staging_refuses_token_only_shards(tmp_path: Path) -> None:
    meta = DatasetMeta(
        format_version=2, preprocessing_name="fixture", seq_len=4, tokenizer_name="test"
    )
    _prepare_output(tmp_path, meta)
    pq.write_table(
        pa.table({"input_ids": pa.array([[1, 2, 3, 4]], type=pa.list_(pa.int32(), 4))}),
        tmp_path / "shard_00000.parquet",
    )
    with pytest.raises(AssertionError, match="require input_ids and document_ids"):
        _prepare_output(tmp_path, meta)


def test_staging_resumes_matching_document_shards(tmp_path: Path) -> None:
    meta = DatasetMeta(
        format_version=2, preprocessing_name="fixture", seq_len=4, tokenizer_name="test"
    )
    _prepare_output(tmp_path, meta)
    pq.write_table(
        pa.table(
            {
                "input_ids": pa.array([[3, 2, 4, 2]], type=pa.list_(pa.int32(), 4)),
                "document_ids": pa.array([[0, 0, 1, 1]], type=pa.list_(pa.int32(), 4)),
            }
        ),
        tmp_path / "shard_00000.parquet",
    )
    _prepare_output(tmp_path, meta)
    with pytest.raises(AssertionError, match="metadata differs"):
        _prepare_output(
            tmp_path,
            DatasetMeta(
                format_version=2, preprocessing_name="fixture", seq_len=8, tokenizer_name="test"
            ),
        )
    assert read_dataset_meta(tmp_path) == meta


def test_concurrent_workers_publish_complete_metadata(tmp_path: Path) -> None:
    meta = DatasetMeta(
        format_version=2, preprocessing_name="fixture", seq_len=4, tokenizer_name="test"
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: _prepare_output(tmp_path, meta), range(16)))
    assert read_dataset_meta(tmp_path) == meta


def test_worker_accepts_metadata_published_during_directory_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    meta = DatasetMeta(
        format_version=2, preprocessing_name="fixture", seq_len=4, tokenizer_name="test"
    )
    original_iterdir = Path.iterdir

    def publish_then_list(path: Path) -> Iterator[Path]:
        if path == tmp_path:
            write_dataset_meta(path, meta)
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", publish_then_list)
    _prepare_output(tmp_path, meta)
    assert read_dataset_meta(tmp_path) == meta


def test_staging_refuses_changed_preprocessing(tmp_path: Path) -> None:
    original = DatasetMeta(
        format_version=2, preprocessing_name="fixture-v1", seq_len=4, tokenizer_name="test"
    )
    _prepare_output(tmp_path, original)
    changed = DatasetMeta(
        format_version=2, preprocessing_name="fixture-v2", seq_len=4, tokenizer_name="test"
    )
    with pytest.raises(AssertionError, match="metadata differs"):
        _prepare_output(tmp_path, changed)
    assert read_dataset_meta(tmp_path) == original


@pytest.mark.parametrize(
    "name", ["Qwen/Qwen3-8B-Base", "meta-llama/Llama-3.1-8B-Instruct", "unknown"]
)
def test_loader_refuses_unsupported_tokenizer_names(name: str) -> None:
    with pytest.raises(AssertionError, match="unsupported Llama 3 base tokenizer"):
        load_llama3_tokenizer(name)


def test_preprocessing_refuses_nontext_columns(tokenizer: Llama3Tokenizer) -> None:
    source = Dataset.from_dict({"text": [123]})
    with pytest.raises(AssertionError, match="string text column"):
        preprocess_llama3(source, tokenizer, column_name="text", max_length=8, num_proc=1)
    with pytest.raises(AssertionError, match="string text column"):
        preprocess_llama3_streaming(
            source.to_iterable_dataset(), tokenizer, column_name="text", max_length=8
        )


@pytest.mark.parametrize("token_id", [-1, 2**31])
def test_packer_refuses_tokens_outside_int32(token_id: int) -> None:
    with pytest.raises(AssertionError, match="nonnegative int32"):
        pack_documents(((token_id,),), max_length=1)


@pytest.mark.parametrize("column", ["input_ids", "document_ids"])
@pytest.mark.parametrize("width", [3, 5])
def test_staging_refuses_fixed_rows_with_wrong_width(
    tmp_path: Path, column: str, width: int
) -> None:
    meta = DatasetMeta(
        format_version=2, preprocessing_name="fixture", seq_len=4, tokenizer_name="test"
    )
    _prepare_output(tmp_path, meta)
    columns = {
        "input_ids": pa.array([[1, 2, 3, 4]], type=pa.list_(pa.int32(), 4)),
        "document_ids": pa.array([[0, 0, 1, 1]], type=pa.list_(pa.int32(), 4)),
    }
    columns[column] = pa.array([[0] * width], type=pa.list_(pa.int32(), width))
    pq.write_table(pa.table(columns), tmp_path / "shard_00000.parquet")
    with pytest.raises(AssertionError, match=column):
        _prepare_output(tmp_path, meta)


def test_staging_refuses_variable_length_lists(tmp_path: Path) -> None:
    meta = DatasetMeta(
        format_version=2, preprocessing_name="fixture", seq_len=4, tokenizer_name="test"
    )
    _prepare_output(tmp_path, meta)
    pq.write_table(
        pa.table(
            {
                "input_ids": pa.array([[1, 2, 3], [4, 5, 6, 7, 8]], type=pa.list_(pa.int32())),
                "document_ids": pa.array([[0, 0, 1], [1, 1, 1, 2, 2]], type=pa.list_(pa.int32())),
            }
        ),
        tmp_path / "shard_00000.parquet",
    )
    with pytest.raises(AssertionError, match="input_ids"):
        _prepare_output(tmp_path, meta)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_files", 0),
        ("num_files", -1),
        ("num_tasks", 0),
        ("num_tasks", -1),
        ("num_proc", 0),
        ("num_proc", -1),
        ("seq_len", 0),
        ("seq_len", -1),
        ("skip_files", -1),
        ("task_id", -1),
        ("task_id", 1),
    ],
)
def test_staging_validates_cli_bounds_before_io(tmp_path: Path, field: str, value: int) -> None:
    bounds = {
        "num_files": 1,
        "num_tasks": 1,
        "num_proc": 1,
        "seq_len": 4,
        "skip_files": 0,
        "task_id": 0,
    }
    bounds[field] = value
    output = tmp_path / "output"
    with pytest.raises(AssertionError, match=field):
        prestage(
            out_dir=str(output),
            num_files=bounds["num_files"],
            skip_files=bounds["skip_files"],
            task_id=bounds["task_id"],
            num_tasks=bounds["num_tasks"],
            dataset_repo="fixture/dataset",
            subdir="train",
            revision="0" * 40,
            tokenizer_name="missing-tokenizer",
            seq_len=bounds["seq_len"],
            column_name="text",
            num_proc=bounds["num_proc"],
        )
    assert not output.exists()
