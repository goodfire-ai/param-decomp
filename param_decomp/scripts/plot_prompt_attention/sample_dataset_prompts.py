"""Sample prompts from the dataset and save as JSON for plot_prompt_attention.

Usage:
    python -m param_decomp.scripts.plot_prompt_attention.sample_dataset_prompts \
        wandb:goodfire/spd/runs/<run_id> --n_samples 5
"""

import json
from pathlib import Path

import fire

from param_decomp.log import logger
from param_decomp.models.component_model import ParamDecompRunInfo
from param_decomp.param_decomp_types import ModelPath
from param_decomp.scripts.prompt_utils import sample_prompts_from_dataset

SCRIPT_DIR = Path(__file__).parent


def sample_dataset_prompts(wandb_path: ModelPath, n_samples: int = 5) -> None:
    run_info = ParamDecompRunInfo.from_path(wandb_path)
    prompts = sample_prompts_from_dataset(run_info, n_samples)

    out_path = SCRIPT_DIR / "dataset_prompts.json"
    with open(out_path, "w") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(prompts)} prompts to {out_path}")


if __name__ == "__main__":
    fire.Fire(sample_dataset_prompts)
