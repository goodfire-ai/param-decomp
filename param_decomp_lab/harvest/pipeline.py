"""Harvest merge: combine partial worker states into final harvest results."""

from pathlib import Path

import tqdm

from param_decomp.log import logger
from param_decomp_lab.adapters.pd import PDAdapter
from param_decomp_lab.harvest.accumulator import Harvester
from param_decomp_lab.harvest.config import HarvestConfig
from param_decomp_lab.harvest.repo import HarvestRepo


def merge_harvest(output_dir: Path, config: HarvestConfig) -> None:
    """Merge partial harvest results from parallel workers.

    Looks for worker_*.npz state files in output_dir/worker_states/ and merges them
    into final harvest results written to output_dir.
    """
    state_dir = output_dir / "worker_states"

    worker_files = sorted(state_dir.glob("worker_*.npz"))
    assert worker_files, f"No worker state files found in {state_dir}"
    logger.info(f"Found {len(worker_files)} worker state files to merge")

    first_worker_file, *rest_worker_files = worker_files

    logger.info(f"Loading worker 0: {first_worker_file.name}")
    harvester = Harvester.load(first_worker_file)
    logger.info(f"Loaded worker 0: {harvester.total_tokens_processed:,} tokens")

    for worker_file in tqdm.tqdm(rest_worker_files, desc="Merging worker states"):
        other = Harvester.load(worker_file)
        harvester.merge(other)
        del other

    logger.info(f"Merge complete. Total tokens: {harvester.total_tokens_processed:,}")

    subrun_id = output_dir.name
    run_id = output_dir.parent.parent.name
    tokenizer_name = PDAdapter(run_id).tokenizer_name
    HarvestRepo.save_results(harvester, config, run_id, subrun_id, tokenizer_name)
    db_path = output_dir / "harvest.db"
    assert db_path.exists() and db_path.stat().st_size > 0, f"Merge output is empty: {db_path}"
    logger.info(f"Saved merged results to {output_dir}")

    for worker_file in worker_files:
        worker_file.unlink()
    state_dir.rmdir()
    logger.info(f"Deleted {len(worker_files)} worker state files")
