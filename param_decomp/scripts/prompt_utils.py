"""Shared utilities for loading and sampling prompts across scripts."""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from param_decomp.experiments.lm.data import create_lm_data_loader
from param_decomp.experiments.lm.experiment import LMRunConfig
from param_decomp.saved_run import PDRun


def load_prompts(path: Path) -> list[str]:
    """Read a JSON file containing a list of prompt strings."""
    assert path.exists(), f"Prompts file not found: {path}"
    with open(path) as f:
        prompts: list[str] = json.load(f)
    return prompts


def sample_prompts_from_dataset(pd_run: PDRun, n_samples: int) -> list[str]:
    """Sample n_samples sequences from the dataset and decode to strings."""
    exp = pd_run.run_cfg
    assert isinstance(exp, LMRunConfig), "Run is not an LM experiment"
    data = exp.data

    tokenizer = AutoTokenizer.from_pretrained(data.tokenizer_name)

    loader, _ = create_lm_data_loader(
        data,
        split=data.eval_split,
        batch_size=1,
        seed=pd_run.pd_config.seed + 1,
    )

    token_column = data.column_name if data.is_tokenized else "input_ids"
    prompts: list[str] = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_samples:
                break
            input_ids = batch[token_column][0]
            text = tokenizer.decode(input_ids, skip_special_tokens=False)  # pyright: ignore[reportAttributeAccessIssue]
            prompts.append(text)

    return prompts
