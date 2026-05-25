"""Harvest LM component activations into a compressed membership snapshot.

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

import torch

from param_decomp.log import logger
from param_decomp_lab.clustering.harvest_config import HarvestConfig
from param_decomp_lab.clustering.memberships import collect_memberships
from param_decomp_lab.clustering.paths import clustering_harvest_dir, new_harvest_id
from param_decomp_lab.distributed import get_device
from param_decomp_lab.experiments.lm.run import SavedLMRun, build_lm_loader

os.environ["WANDB_QUIET"] = "true"


def harvest(config: HarvestConfig) -> Path:
    run_id = new_harvest_id()
    out = clustering_harvest_dir(run_id)
    out.mkdir(parents=True, exist_ok=True)
    logger.info(f"Harvest {run_id} → {out}")

    config.to_file(out / "harvest_config.json")

    device = get_device()
    pd_run = SavedLMRun.from_path(config.model_path)
    dataloader = build_lm_loader(
        pd_run.cfg.target,
        pd_run.cfg.data,
        split="train",
        device=device,
        batch_size=config.batch_size,
        seed=config.dataset_seed,
    )
    model = pd_run.load_model().to(device)

    processed = collect_memberships(model, dataloader, device, config)

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
