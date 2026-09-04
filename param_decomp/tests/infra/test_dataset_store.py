"""The `DatasetRef` schema: store names resolve under `data_root`, ad-hoc dirs are
absolute, and datasets self-describe via `meta.json`."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from param_decomp.infra.dataset_store import (
    DatasetDir,
    DatasetMeta,
    NamedDataset,
    dataset_dir,
    read_dataset_meta,
    write_dataset_meta,
)


def test_store_name_resolves_under_data_root() -> None:
    assert dataset_dir(Path("/project"), "pile") == Path("/project/datasets/pile")


def test_dataset_names_are_flat() -> None:
    with pytest.raises(ValidationError, match="flat store names"):
        NamedDataset.model_validate({"kind": "name", "name": "datasets/pile"})


def test_ad_hoc_dirs_are_absolute() -> None:
    with pytest.raises(ValidationError, match="absolute"):
        DatasetDir.model_validate({"kind": "dir", "dir": "relative/shards"})


def test_dataset_meta_round_trips(tmp_path: Path) -> None:
    meta = DatasetMeta(seq_len=512, tokenizer_name="EleutherAI/gpt-neox-20b")
    write_dataset_meta(tmp_path, meta)
    assert read_dataset_meta(tmp_path) == meta


def test_dataset_meta_missing_refuses(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="self-describing"):
        read_dataset_meta(tmp_path)
