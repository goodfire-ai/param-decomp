"""Shared utilities for loading and sampling prompts across scripts."""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from param_decomp.data import DatasetConfig, create_data_loader
from param_decomp.experiments.lm.experiment import LMExperimentConfig
from param_decomp.models.component_model import PDRunInfo


def load_prompts(path: Path) -> list[str]:
    """Read a JSON file containing a list of prompt strings."""
    assert path.exists(), f"Prompts file not found: {path}"
    with open(path) as f:
        prompts: list[str] = json.load(f)
    return prompts


def sample_prompts_from_dataset(run_info: PDRunInfo, n_samples: int) -> list[str]:
    """Sample n_samples sequences from the dataset and decode to strings."""
    exp = run_info.config
    assert isinstance(exp, LMExperimentConfig), "Run is not an LM experiment"
    data = exp.data

    tokenizer = AutoTokenizer.from_pretrained(data.tokenizer_name)

    dataset_config = DatasetConfig(
        name=data.dataset_name,
        hf_tokenizer_path=data.tokenizer_name,
        split=data.eval_split,
        n_ctx=data.max_seq_len,
        is_tokenized=data.is_tokenized,
        streaming=data.streaming,
        column_name=data.column_name,
        shuffle_each_epoch=False,
    )
    loader, _ = create_data_loader(
        dataset_config=dataset_config,
        batch_size=1,
        buffer_size=1000,
    )

    prompts: list[str] = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_samples:
                break
            input_ids = batch[data.column_name][0]
            text = tokenizer.decode(input_ids, skip_special_tokens=False)  # pyright: ignore[reportAttributeAccessIssue]
            prompts.append(text)

    return prompts
