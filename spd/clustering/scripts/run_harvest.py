"""Harvest component activations into a compressed membership snapshot.

Output:
    <SPD_OUT_DIR>/clustering/harvests/<harvest_id>/
        ├── harvest_config.json
        ├── memberships.npz
        ├── metadata.json
        └── preview.pt (optional)
"""

import argparse
import gc
import os
from pathlib import Path

import torch

from spd.clustering.dataset import create_clustering_dataloader
from spd.clustering.harvest_config import HarvestConfig
from spd.clustering.memberships import collect_memberships
from spd.clustering.paths import clustering_harvest_dir, new_harvest_id
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device

os.environ["WANDB_QUIET"] = "true"


def harvest(config: HarvestConfig) -> Path:
    run_id = new_harvest_id()
    out = clustering_harvest_dir(run_id)
    out.mkdir(parents=True, exist_ok=True)
    logger.info(f"Harvest {run_id} → {out}")

    config.to_file(out / "harvest_config.json")

    device = get_device()
    spd_run = SPDRunInfo.from_path(config.model_path)
    task_name = spd_run.config.task_config.task_name
    model = ComponentModel.from_run_info(spd_run).to(device)
    dataloader = create_clustering_dataloader(
        model_path=config.model_path,
        task_name=task_name,
        batch_size=config.batch_size,
        seed=config.dataset_seed,
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
