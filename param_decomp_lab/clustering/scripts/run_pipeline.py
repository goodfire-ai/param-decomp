"""Ensemble clustering pipeline (`pd-clustering`).

Fans a single decomposition out into a seeded ensemble of independent clustering runs, then
computes their cross-run consensus. Each member is a seeded JAX harvest
(`run_worker`, 1 GPU) feeding a CPU merge (`run_merge` / `pd-cluster-merge`); a final
consensus job (`calc_distances`) normalizes the members' labels and computes per-iteration
pairwise distances + a stability plot.

Three dependency tiers, submitted as SLURM jobs:

    harvest array (N × 1 GPU, seeded)
        └─ merge array (N × CPU, seeded)            [afterok harvest array]
              └─ consensus job per distance method  [afterok merge array]

Output:
    PARAM_DECOMP_OUT_DIR/clustering/ensembles/<ensemble_id>/
        ├── ensemble_config.yaml
        └── (consensus artifacts, written by calc_distances.py)
    PARAM_DECOMP_OUT_DIR/clustering/harvests/<harvest_id>/   (per member)
    PARAM_DECOMP_OUT_DIR/clustering/runs/<run_id>/           (per member)
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field, PositiveInt, model_validator

from param_decomp.base_config import BaseConfig
from param_decomp.log import logger
from param_decomp_lab.clustering.harvest_config import HarvestConfig
from param_decomp_lab.clustering.merge_config import MergeConfig
from param_decomp_lab.clustering.paths import (
    clustering_ensemble_dir,
    clustering_harvest_dir,
    new_harvest_id,
    new_run_id,
)
from param_decomp_lab.clustering.scripts import calc_distances, run_merge, run_worker
from param_decomp_lab.clustering.types import DistancesMethod
from param_decomp_lab.infra.git import create_git_snapshot
from param_decomp_lab.infra.run_files import generate_run_id, run_locally
from param_decomp_lab.infra.slurm import (
    SlurmArrayConfig,
    SlurmConfig,
    generate_array_script,
    generate_script,
    submit_slurm_job,
)

os.environ["WANDB_QUIET"] = "true"


class ClusteringEnsembleConfig(BaseConfig):
    """Seeded ensemble of harvest→merge clustering runs plus their consensus."""

    harvest: HarvestConfig
    merge: MergeConfig = Field(default_factory=MergeConfig)
    n_runs: PositiveInt
    distances_methods: list[DistancesMethod] = Field(
        default_factory=lambda: ["perm_invariant_hamming"]
    )
    base_seed: int = 0
    step: int | None = Field(default=None, description="checkpoint step (default: latest)")
    plot_members: bool = Field(default=False, description="emit per-member diagnostic plots")
    partition: str | None = None
    merge_mem: str | None = None
    create_git_snapshot: bool = False

    @model_validator(mode="after")
    def _validate_methods(self) -> "ClusteringEnsembleConfig":
        assert self.distances_methods, "distances_methods must be non-empty"
        assert all(m in DistancesMethod.__args__ for m in self.distances_methods), (
            f"invalid distances_methods: {self.distances_methods}"
        )
        return self


@dataclass(frozen=True)
class EnsembleMember:
    harvest_id: str
    run_id: str
    seed: int


def _members(config: ClusteringEnsembleConfig) -> list[EnsembleMember]:
    return [
        EnsembleMember(
            harvest_id=new_harvest_id(),
            run_id=new_run_id(),
            seed=config.base_seed + i,
        )
        for i in range(config.n_runs)
    ]


def submit(config: ClusteringEnsembleConfig, local: bool) -> str:
    """Submit (or run locally) the full ensemble pipeline. Returns the ensemble id."""
    ensemble_id = generate_run_id("clustering/ensembles")
    ensemble_dir = clustering_ensemble_dir(ensemble_id)
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    config.to_file(ensemble_dir / "ensemble_config.yaml")
    logger.info(f"Ensemble {ensemble_id} → {ensemble_dir}")

    harvest_config_path = ensemble_dir / "harvest_config.json"
    merge_config_path = ensemble_dir / "merge_config.json"
    config.harvest.to_file(harvest_config_path)
    config.merge.to_file(merge_config_path)

    members = _members(config)

    harvest_commands = [
        run_worker.get_command(harvest_config_path, m.harvest_id, m.seed) for m in members
    ]
    merge_commands = [
        run_merge.get_command(
            snapshot_path=clustering_harvest_dir(m.harvest_id),
            merge_config_path=merge_config_path,
            run_id=m.run_id,
            seed=m.seed,
            plot=config.plot_members,
        )
        for m in members
    ]
    consensus_commands = [
        calc_distances.get_command(ensemble_id, [m.run_id for m in members], method)
        for method in config.distances_methods
    ]

    if local:
        run_locally(harvest_commands)
        run_locally(merge_commands)
        run_locally(consensus_commands)
        logger.section("ensemble complete (local)")
        logger.values(
            {
                "Ensemble ID": ensemble_id,
                "Ensemble dir": str(ensemble_dir),
                "N members": config.n_runs,
            }
        )
        return ensemble_id

    assert config.partition is not None, "partition required for SLURM submission"
    snapshot_ref: str | None = None
    if config.create_git_snapshot:
        snapshot_ref, commit_hash = create_git_snapshot(snapshot_id=ensemble_id)
        logger.info(f"Snapshot: {snapshot_ref} ({commit_hash[:8]})")

    harvest_array = SlurmArrayConfig(
        job_name="pd-clustering-harvest",
        partition=config.partition,
        n_gpus=1,
        snapshot_ref=snapshot_ref,
        comment=ensemble_id,
    )
    harvest_result = submit_slurm_job(
        generate_array_script(harvest_array, harvest_commands),
        "clustering_harvest",
        n_array_tasks=config.n_runs,
    )

    merge_array = SlurmArrayConfig(
        job_name="pd-clustering-merge",
        partition=config.partition,
        n_gpus=0,
        mem=config.merge_mem,
        snapshot_ref=snapshot_ref,
        dependency_job_id=harvest_result.job_id,
        comment=ensemble_id,
    )
    merge_result = submit_slurm_job(
        generate_array_script(merge_array, merge_commands),
        "clustering_merge",
        n_array_tasks=config.n_runs,
    )

    consensus_job_ids: list[str] = []
    for method, cmd in zip(config.distances_methods, consensus_commands, strict=True):
        consensus_config = SlurmConfig(
            job_name=f"pd-clustering-consensus-{method}",
            partition=config.partition,
            n_gpus=0,
            mem=config.merge_mem,
            snapshot_ref=snapshot_ref,
            dependency_job_id=merge_result.job_id,
            comment=ensemble_id,
        )
        consensus_result = submit_slurm_job(
            generate_script(consensus_config, cmd), f"clustering_consensus_{method}"
        )
        consensus_job_ids.append(consensus_result.job_id)

    logger.section("ensemble submitted")
    logger.values(
        {
            "Ensemble ID": ensemble_id,
            "Ensemble dir": str(ensemble_dir),
            "N members": config.n_runs,
            "Harvest array job": harvest_result.job_id,
            "Merge array job": merge_result.job_id,
            "Consensus jobs": ", ".join(consensus_job_ids),
        }
    )
    return ensemble_id


def cli() -> None:
    parser = argparse.ArgumentParser(prog="pd-clustering", description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="ClusteringEnsembleConfig file")
    parser.add_argument("--n-runs", type=int, default=None, help="override n_runs")
    parser.add_argument(
        "--local",
        action="store_true",
        help="run sequentially in-process instead of submitting to SLURM",
    )
    args = parser.parse_args()

    config = ClusteringEnsembleConfig.from_file(args.config)
    if args.n_runs is not None:
        config = config.model_copy(update={"n_runs": args.n_runs})
    submit(config, local=args.local)


if __name__ == "__main__":
    cli()
