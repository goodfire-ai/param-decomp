"""Harvest component activations into a compressed membership snapshot.

Output:
    <SPD_OUT_DIR>/clustering/harvests/<harvest_id>/
        ├── harvest_config.json
        ├── memberships.npz     # scipy sparse CSC matrix
        └── metadata.json       # labels, n_samples, n_components
"""

import argparse
import gc
import os
from pathlib import Path

import torch

from spd.clustering.activations import collect_memberships
from spd.clustering.dataset import create_clustering_dataloader
from spd.clustering.harvest_config import HarvestConfig
from spd.clustering.storage import StorageBase
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.run_utils import ExecutionStamp

os.environ["WANDB_QUIET"] = "true"


class HarvestStorage(StorageBase):
    _CONFIG = "harvest_config.json"

    def __init__(self, execution_stamp: ExecutionStamp) -> None:
        super().__init__(execution_stamp)
        self.config_path: Path = self.base_dir / self._CONFIG


def harvest(config: HarvestConfig) -> Path:
    execution_stamp = ExecutionStamp.create(run_type="clustering/harvests", create_snapshot=False)
    storage = HarvestStorage(execution_stamp)
    logger.info(f"Harvest {execution_stamp.run_id} → {storage.base_dir}")

    config.to_file(storage.config_path)

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
    processed.save(storage.base_dir)

    logger.info(f"Harvest complete: {storage.base_dir}")
    return storage.base_dir


def cli() -> None:
    parser = argparse.ArgumentParser(description="Harvest activations into membership snapshot.")
    parser.add_argument("config", type=Path, help="Path to HarvestConfig JSON/YAML.")
    args = parser.parse_args()
    harvest(HarvestConfig.from_file(args.config))


if __name__ == "__main__":
    cli()
