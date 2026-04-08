"""Sample prompts from the dataset and save as JSON for plot_prompt_attention.

Usage:
    python -m spd.scripts.plot_prompt_attention.sample_dataset_prompts \
        wandb:goodfire/spd/runs/<run_id> --n_samples 5
"""

import json
from pathlib import Path

import fire
import torch
from transformers import AutoTokenizer

from spd.configs import LMTaskConfig
from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.models.component_model import SPDRunInfo
from spd.spd_types import ModelPath

SCRIPT_DIR = Path(__file__).parent


def sample_dataset_prompts(wandb_path: ModelPath, n_samples: int = 5) -> None:
    run_info = SPDRunInfo.from_path(wandb_path)

    config = run_info.config
    task_config = config.task_config
    assert isinstance(task_config, LMTaskConfig)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    dataset_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=config.tokenizer_name,
        split=task_config.eval_data_split,
        n_ctx=task_config.max_seq_len,
        is_tokenized=task_config.is_tokenized,
        streaming=task_config.streaming,
        column_name=task_config.column_name,
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
            input_ids = batch[task_config.column_name][0]
            text = tokenizer.decode(input_ids, skip_special_tokens=False)  # pyright: ignore[reportAttributeAccessIssue]
            prompts.append(text)

    out_path = SCRIPT_DIR / "dataset_prompts.json"
    with open(out_path, "w") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(prompts)} prompts to {out_path}")


if __name__ == "__main__":
    fire.Fire(sample_dataset_prompts)
