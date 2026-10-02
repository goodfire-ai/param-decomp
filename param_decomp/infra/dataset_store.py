"""The dataset-store artifact contract: layout, the reference a config names shards by,
and the self-describing `meta.json`.

A store dataset is a directory of pre-tokenized `*.parquet` shards plus a `meta.json`
carrying the dataset's own facts. Composition roots and consumers read the meta here
and thread the values into the core loader as explicit parameters
(`ShardServer(seq_len=...)`).

`DatasetRef` is the one way any config names shards — the pretrainer and the
decomposition trainer share it. It lives beside the layout because resolving its `name`
arm IS the layout.
"""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Literal, Self

from pydantic import Discriminator, Field, PositiveInt, model_validator

from param_decomp.core.base_config import BaseConfig

DATASET_META_FILENAME = "meta.json"


def dataset_dir(data_root: Path, name: str) -> Path:
    """The store layout: a named dataset's shards live at `<data_root>/datasets/<name>`."""
    return data_root / "datasets" / name


class NamedDataset(BaseConfig):
    """A dataset in the store: shards + `meta.json` at `<data_root>/datasets/<name>`.
    Names are immutable versions — a changed dataset is a new name."""

    kind: Literal["name"] = "name"
    name: str

    @model_validator(mode="after")
    def _flat_name(self) -> Self:
        assert self.name and "/" not in self.name and "*" not in self.name, (
            f"dataset names are flat store names: {self.name!r}"
        )
        return self


class DatasetDir(BaseConfig):
    """Ad-hoc escape hatch: an explicit directory of `*.parquet` shards. Machine-specific
    by nature, so the path must be absolute; a named store dataset is the portable form."""

    kind: Literal["dir"] = "dir"
    dir: Path

    @model_validator(mode="after")
    def _absolute(self) -> Self:
        assert self.dir.is_absolute(), f"ad-hoc shard dirs are absolute paths: {self.dir}"
        return self


DatasetRef = Annotated[NamedDataset | DatasetDir, Discriminator("kind")]


def resolve_dataset_ref(ref: DatasetRef, data_root: Path) -> Path:
    match ref:
        case NamedDataset(name=name):
            return dataset_dir(data_root, name)
        case DatasetDir(dir=dir):
            return dir


class DatasetIdentity(BaseConfig):
    """Tokenizer and staged row width, independent of the shard storage format.

    Enough to tokenize new prompts or decode stored token IDs; this does not certify
    that a dataset can be read by the current training loader.
    """

    seq_len: PositiveInt
    tokenizer_name: str = Field(min_length=1)


class DatasetMeta(DatasetIdentity):
    """`seq_len` is the exact staged row width, including any final next-token label.
    `tokenizer_name` is the tokenizer that produced the ids — the decode authority for
    consumers rendering harvested tokens. Version 2 requires paired int32 `input_ids`
    and `document_ids` columns. `preprocessing_name` records the producing recipe;
    it is provenance, not a runtime behavior switch. Older artifacts must be tokenized again."""

    format_version: Literal[2]
    preprocessing_name: str = Field(min_length=1)


def read_dataset_identity(data_dir: Path) -> DatasetIdentity:
    """Read only tokenizer/context facts; never authorize reading or rewriting shards."""
    raw = json.loads((data_dir / DATASET_META_FILENAME).read_text())
    return DatasetIdentity(seq_len=raw["seq_len"], tokenizer_name=raw["tokenizer_name"])


def read_dataset_meta(data_dir: Path) -> DatasetMeta:
    path = data_dir / DATASET_META_FILENAME
    assert path.exists(), (
        f"no {DATASET_META_FILENAME} in {data_dir}: dataset dirs are self-describing "
        "(prestage writes document-aware metadata; older datasets must be tokenized again)"
    )
    return DatasetMeta.model_validate_json(path.read_text())


def write_dataset_meta(data_dir: Path, meta: DatasetMeta) -> None:
    """Publish immutable metadata atomically, including concurrent prestage workers."""
    path = data_dir / DATASET_META_FILENAME
    with NamedTemporaryFile(mode="w", dir=data_dir.parent, suffix=".meta.json") as temporary:
        temporary.write(meta.model_dump_json() + "\n")
        temporary.flush()
        try:
            path.hardlink_to(temporary.name)
        except FileExistsError:
            assert read_dataset_meta(data_dir) == meta, f"dataset metadata differs in {data_dir}"
