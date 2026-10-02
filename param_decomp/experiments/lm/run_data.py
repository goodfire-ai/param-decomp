"""A finished LM run's data config, read from its run directory without loading JAX."""

from pathlib import Path

import yaml

from param_decomp.core.base_config import BaseConfig
from param_decomp.core.run_files import LAUNCH_CONFIG_FILENAME
from param_decomp.infra.dataset_store import DatasetRef

DELIVERABLE_FILENAME = "deliverable.yaml"


class LMDataConfig(BaseConfig):
    """The run's data: `train` feeds the trainer; `eval` is the held-out split the eval
    pass reads. Each ref's facts (seq_len, tokenizer) ride with its shards as `meta.json`
    (`param_decomp.infra.dataset_store`), read at load."""

    train: DatasetRef
    eval: DatasetRef


def product_document(run_dir: Path) -> Path:
    normalized = run_dir / DELIVERABLE_FILENAME
    return normalized if normalized.is_file() else run_dir / LAUNCH_CONFIG_FILENAME


def load_data_config(run_dir: Path) -> LMDataConfig:
    """The datasets a finished run trained on and held out, as its product names them."""
    raw = yaml.safe_load(product_document(run_dir).read_text())
    assert isinstance(raw, dict), f"stored run config must be a mapping: {run_dir}"
    return LMDataConfig.model_validate(raw["data"])
