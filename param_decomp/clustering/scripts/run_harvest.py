"""Harvest component activations into a compressed membership snapshot.

Output:
    <PARAM_DECOMP_OUT_DIR>/clustering/harvests/<harvest_id>/
        ├── harvest_config.json
        ├── memberships.npz
        ├── metadata.json
        └── preview.pt (optional)
"""

import argparse
import gc
import os
from pathlib import Path
from typing import cast

import torch

from param_decomp.clustering.harvest_config import HarvestConfig
from param_decomp.clustering.memberships import collect_memberships
from param_decomp.clustering.paths import clustering_harvest_dir, new_harvest_id
from param_decomp.load import load_pd
from param_decomp.log import logger
from param_decomp.models.component_model import PDRunInfo
from param_decomp.param_decomp_types import TaskName
from param_decomp.utils.distributed_utils import get_device

os.environ["WANDB_QUIET"] = "true"


def harvest(config: HarvestConfig) -> Path:
    run_id = new_harvest_id()
    out = clustering_harvest_dir(run_id)
    out.mkdir(parents=True, exist_ok=True)
    logger.info(f"Harvest {run_id} → {out}")

    config.to_file(out / "harvest_config.json")

    device = get_device()
    pd_run = PDRunInfo.from_path(config.model_path)
    exp = pd_run.spec
    task_name = cast(TaskName, exp.kind)
    target = pd_run.load_target()
    model = load_pd(config.model_path, target=target).to(device)
    dataloader, _ = pd_run.build_dataloaders(
        seed=config.dataset_seed,
        train_batch_size=config.batch_size,
        eval_batch_size=config.batch_size,
    )

    processed = collect_memberships(model, dataloader, task_name, device, config)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    logger.info(f"Saving: {processed.n_components_alive} alive, {processed.n_samples} samples")
    processed.save(out)

    logger.info(f"Harvest complete: {out}")
    return out


def cli() -> None:
    parser = argparse.ArgumentParser(description="Harvest activations into membership snapshot.")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    harvest(HarvestConfig.from_file(args.config))


if __name__ == "__main__":
    cli()
