"""Harvest component activations into a compressed membership snapshot.

Collects CI activations from a decomposed model, thresholds them, and saves
a sparse boolean membership matrix to disk. The snapshot can then be fed to
`run_merge.py` with different merge configs without re-running the GPU work.

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

from spd.clustering.activations import (
    ProcessedMemberships,
    collect_memberships_lm,
    collect_memberships_resid_mlp,
)
from spd.clustering.dataset import create_clustering_dataloader
from spd.clustering.harvest_config import HarvestConfig
from spd.clustering.membership_snapshot import save_membership_snapshot
from spd.clustering.storage import StorageBase
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.spd_types import TaskName
from spd.utils.distributed_utils import get_device
from spd.utils.run_utils import ExecutionStamp

os.environ["WANDB_QUIET"] = "true"


class HarvestStorage(StorageBase):
    _CONFIG = "harvest_config.json"

    def __init__(self, execution_stamp: ExecutionStamp) -> None:
        super().__init__(execution_stamp)
        self.config_path: Path = self.base_dir / self._CONFIG
        self.snapshot_dir: Path = self.base_dir


def harvest(config: HarvestConfig) -> Path:
    execution_stamp = ExecutionStamp.create(
        run_type="clustering/harvests",
        create_snapshot=False,
    )
    storage = HarvestStorage(execution_stamp)
    harvest_id = execution_stamp.run_id
    logger.info(f"Harvest ID: {harvest_id}")
    logger.info(f"Output: {storage.base_dir}")

    config.to_file(storage.config_path)

    device = get_device()
    spd_run = SPDRunInfo.from_path(config.model_path)
    task_name: TaskName = spd_run.config.task_config.task_name

    logger.info(f"Loading dataset (seed={config.dataset_seed})")
    dataloader = create_clustering_dataloader(
        model_path=config.model_path,
        task_name=task_name,
        batch_size=config.batch_size,
        seed=config.dataset_seed,
    )

    logger.info("Loading model")
    model = ComponentModel.from_run_info(spd_run).to(device)

    logger.info("Collecting memberships")
    processed: ProcessedMemberships
    if task_name == "lm":
        assert config.n_tokens is not None
        assert config.n_tokens_per_seq is not None
        processed = collect_memberships_lm(
            model=model,
            dataloader=dataloader,
            n_tokens=config.n_tokens,
            n_tokens_per_seq=config.n_tokens_per_seq,
            device=device,
            seed=config.dataset_seed,
            activation_threshold=config.activation_threshold,
            filter_dead_threshold=config.filter_dead_threshold,
            filter_dead_stat=config.filter_dead_stat,
            filter_modules=config.filter_modules,
        )
    else:
        processed = collect_memberships_resid_mlp(
            model=model,
            dataloader=dataloader,
            n_samples=config.n_samples or config.batch_size,
            device=device,
            activation_threshold=config.activation_threshold,
            filter_dead_threshold=config.filter_dead_threshold,
            filter_dead_stat=config.filter_dead_stat,
            filter_modules=config.filter_modules,
        )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    logger.info(
        f"Saving snapshot: {processed.n_components_alive} alive components, "
        f"{processed.n_samples} samples"
    )
    save_membership_snapshot(
        storage.snapshot_dir,
        memberships=processed.memberships,
        labels=processed.labels,
        n_samples=processed.n_samples,
    )

    logger.info(f"Harvest complete: {storage.base_dir}")
    return storage.base_dir


def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="spd-cluster-harvest",
        description="Harvest component activations into a compressed membership snapshot.",
    )
    parser.add_argument("config", type=Path, help="Path to HarvestConfig JSON/YAML.")
    args = parser.parse_args()

    config = HarvestConfig.from_file(args.config)
    harvest(config)


if __name__ == "__main__":
    cli()
