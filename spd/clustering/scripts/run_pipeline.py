"""Submit clustering runs to SLURM as separate jobs in a SLURM array.

This script submits independent clustering runs as a SLURM job array,
where each run gets its own dataset (seeded), WandB run, and merge history output.

Also submits a job to calculate distances between the clustering runs, which will run after
the clustering runs (the SLURM job depends on the previous array job).

Output structure (only pipeline_config.json is saved to directly in this script. The files under
<runs> are saved by run_clustering.py which is called in SLURM jobs deployed by this script.):
    <ExecutionStamp.out_dir>/                 # from execution stamp
        |── pipeline_config.json              # Saved in this script
        |── clustering_run_config.json        # make copy of the file pointed to by pipeline config
        ├── ensemble_meta.json                # (Saved by calc_distances.py) Ensemble metadata
        ├── ensemble_merge_array.npz          # (Saved by calc_distances.py) Normalized merge array
        ├── distances_<distances_method>.npz  # (Saved by calc_distances.py) Distance array for each method
        └── distances_<distances_method>.png  # (Saved by calc_distances.py) Distance distribution plot
"""

import argparse
import os
from pathlib import Path
from typing import Any

import wandb_workspaces.workspaces as ws
from pydantic import Field, PositiveInt, field_validator, model_validator

from spd.base_config import BaseConfig
from spd.clustering.clustering_run_config import ClusteringRunConfig
from spd.clustering.consts import DistancesMethod
from spd.clustering.paths import clustering_ensemble_dir, new_ensemble_id, new_run_id
from spd.clustering.scripts.calc_distances import get_command as distances_command
from spd.clustering.scripts.run_clustering import get_command as clustering_command
from spd.log import logger
from spd.utils.general_utils import replace_pydantic_model
from spd.utils.git_utils import create_git_snapshot
from spd.utils.run_utils import (
    _NO_ARG_PARSSED_SENTINEL,
    read_noneable_str,
    run_locally,
)
from spd.utils.slurm import (
    SlurmArrayConfig,
    SlurmConfig,
    generate_array_script,
    generate_script,
    submit_slurm_job,
)

os.environ["WANDB_QUIET"] = "true"


class ClusteringPipelineConfig(BaseConfig):
    """Configuration for submitting an ensemble of clustering runs to SLURM."""

    clustering_run_config_path: Path = Field(
        description="Path to ClusteringRunConfig file.",
    )
    n_runs: PositiveInt = Field(description="Number of clustering runs in the ensemble")
    distances_methods: list[DistancesMethod] = Field(
        description="List of method(s) to use for calculating distances"
    )
    slurm_job_name_prefix: str | None = Field(
        default=None, description="Prefix for SLURM job names"
    )
    slurm_partition: str | None = Field(default=None, description="SLURM partition to use")
    slurm_mem: str | None = Field(default=None, description="Memory limit per job (e.g. '300G')")
    wandb_project: str | None = Field(
        default=None,
        description="Weights & Biases project name (set to None to disable WandB logging)",
    )
    wandb_entity: str = Field(default="goodfire", description="WandB entity (team/user) name")
    create_git_snapshot: bool = Field(
        default=False, description="Create a git snapshot for the run"
    )

    @model_validator(mode="after")
    def validate_crc(self) -> "ClusteringPipelineConfig":
        """Validate that exactly one of clustering_run_config_path points to a valid `ClusteringRunConfig`."""
        assert self.clustering_run_config_path.exists(), (
            f"clustering_run_config_path does not exist: {self.clustering_run_config_path}"
        )
        # Try to load ClusteringRunConfig
        assert ClusteringRunConfig.from_file(self.clustering_run_config_path)

        return self

    @field_validator("distances_methods")
    @classmethod
    def validate_distances_methods(cls, v: list[DistancesMethod]) -> list[DistancesMethod]:
        """Validate that distances_methods is non-empty and contains valid methods."""
        assert all(method in DistancesMethod.__args__ for method in v), (
            f"Invalid distances_methods: {v}"
        )

        return v


def create_clustering_workspace_view(ensemble_id: str, project: str, entity: str) -> str:
    """Create WandB workspace view for clustering runs.

    TODO: Use a template workspace which actually shows some panels
    TODO: since the run_id here is the same as the wandb id, can we take advantage of that?

    Args:
        ensemble_id: Unique identifier for this ensemble
        project: WandB project name
        entity: WandB entity (team/user) name

    Returns:
        URL to workspace view
    """
    workspace = ws.Workspace(entity=entity, project=project)
    workspace.name = f"Clustering - {ensemble_id}"

    workspace.runset_settings.filters = [
        ws.Tags("tags").isin([f"ensemble_id:{ensemble_id}"]),
    ]

    try:
        workspace.save_as_new_view()
        return workspace.url
    except Exception as e:
        logger.warning(
            f"Failed to create WandB workspace view: {workspace=}, {workspace.name=}, {ensemble_id=}, {project=}, {entity=}, {e}"
        )
        raise e


def main(
    pipeline_config: ClusteringPipelineConfig,
    local: bool = False,
    local_clustering_parallel: bool = False,
    local_calc_distances_parallel: bool = False,
    track_resources_calc_distances: bool = False,
) -> None:
    """Submit clustering runs to SLURM.

    Args:
        pipeline_config_path: Path to ClusteringPipelineConfig file
        n_runs: Number of clustering runs in the ensemble. Will override value in the config file.
    """
    # setup
    # ==========================================================================================

    logger.set_format("console", "terse")

    if local_clustering_parallel or local_calc_distances_parallel or track_resources_calc_distances:
        assert local, (
            "local_clustering_parallel, local_calc_distances_parallel, track_resources_calc_distances "
            "can only be set when running locally\n"
            f"{local_clustering_parallel=}, {local_calc_distances_parallel=}, {track_resources_calc_distances=}, {local=}"
        )

    pipeline_run_id = new_ensemble_id()
    pipeline_dir = clustering_ensemble_dir(pipeline_run_id)
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Pipeline {pipeline_run_id} → {pipeline_dir}")

    # Git snapshot
    snapshot_branch: str | None = None
    if pipeline_config.create_git_snapshot:
        snapshot_branch, commit_hash = create_git_snapshot(snapshot_id=pipeline_run_id)
        logger.info(f"Created git snapshot: {snapshot_branch} ({commit_hash[:8]})")

    # Save pipeline config
    pipeline_config.to_file(pipeline_dir / "pipeline_config.yaml")
    logger.info(f"Pipeline config saved to {pipeline_dir / 'pipeline_config.yaml'}")

    # Create WandB workspace if requested
    if pipeline_config.wandb_project is not None:
        workspace_url = create_clustering_workspace_view(
            ensemble_id=pipeline_run_id,
            project=pipeline_config.wandb_project,
            entity=pipeline_config.wandb_entity,
        )
        logger.info(f"WandB workspace: {workspace_url}")

    # Pre-generate run IDs for each clustering task
    clustering_run_ids = [new_run_id() for _ in range(pipeline_config.n_runs)]

    # Generate commands
    clustering_commands = [
        clustering_command(
            config_path=pipeline_config.clustering_run_config_path,
            run_id=run_id,
            seed_offset=idx,
            ensemble_id=pipeline_run_id,
            wandb_project=pipeline_config.wandb_project,
            wandb_entity=pipeline_config.wandb_entity
            if pipeline_config.wandb_entity != "goodfire"
            else None,
        )
        for idx, run_id in enumerate(clustering_run_ids)
    ]

    calc_distances_commands = [
        distances_command(pipeline_run_id, clustering_run_ids, method)
        for method in pipeline_config.distances_methods
    ]

    # Submit to SLURM
    if local:
        # submit clustering array job
        run_locally(
            commands=clustering_commands,
            parallel=local_clustering_parallel,
        )

        # submit calc_distances jobs in parallel
        logger.info("Calculating distances...")
        run_locally(
            commands=calc_distances_commands,
            parallel=local_calc_distances_parallel,
            track_resources=track_resources_calc_distances,
        )

        logger.section("complete!")

        # Build distances plot paths dict
        distances_plots = {
            f"distances via {method}": str(pipeline_dir / "plots" / f"distances_{method}.png")
            for method in pipeline_config.distances_methods
        }

        logger.values(
            {
                "Total clustering runs": len(clustering_commands),
                "Pipeline run ID": pipeline_run_id,
                "Pipeline output dir": str(pipeline_dir),
                **distances_plots,
            }
        )

    else:
        assert pipeline_config.slurm_job_name_prefix is not None, (
            "must specify slurm_job_name_prefix if not running locally"
        )
        assert pipeline_config.slurm_partition is not None, (
            "must specify slurm_partition if not running locally"
        )

        # Submit clustering array job
        clustering_config = SlurmArrayConfig(
            job_name=f"{pipeline_config.slurm_job_name_prefix}_cluster",
            partition=pipeline_config.slurm_partition,
            n_gpus=1,  # Always 1 GPU per run
            snapshot_branch=snapshot_branch,
            max_concurrent_tasks=pipeline_config.n_runs,  # Run all concurrently
            mem=pipeline_config.slurm_mem,
        )
        clustering_script = generate_array_script(clustering_config, clustering_commands)
        clustering_result = submit_slurm_job(
            clustering_script,
            "clustering",
            is_array=True,
            n_array_tasks=len(clustering_commands),
        )
        array_job_id = clustering_result.job_id

        # Submit calc_distances jobs (one per method) with dependency on array job
        calc_distances_job_ids: list[str] = []
        calc_distances_logs: list[str] = []

        for method, cmd in zip(
            pipeline_config.distances_methods, calc_distances_commands, strict=True
        ):
            dist_config = SlurmConfig(
                job_name=f"{pipeline_config.slurm_job_name_prefix}_dist_{method}",
                partition=pipeline_config.slurm_partition,
                n_gpus=1,
                snapshot_branch=snapshot_branch,
                dependency_job_id=array_job_id,
            )
            dist_script = generate_script(dist_config, cmd)
            dist_result = submit_slurm_job(dist_script, f"calc_distances_{method}")
            calc_distances_job_ids.append(dist_result.job_id)
            calc_distances_logs.append(dist_result.log_pattern)

        logger.section("Jobs submitted successfully!")

        # Build distances plot paths dict
        distances_plots = {
            method: str(pipeline_dir / "plots" / f"distances_{method}.png")
            for method in pipeline_config.distances_methods
        }

        logger.values(
            {
                "Clustering Array Job ID": array_job_id,
                "Calc Distances Job IDs": ", ".join(calc_distances_job_ids),
                "Total clustering runs": len(clustering_commands),
                "Pipeline run ID": pipeline_run_id,
                "Pipeline output dir": str(pipeline_dir),
                "Clustering logs": clustering_result.log_pattern,
                "Calc Distances logs": ", ".join(calc_distances_logs),
            }
        )
        logger.info("Distances plots will be saved to:")
        for method, path in distances_plots.items():
            logger.info(f"  {method}: {path}")


def cli():
    """CLI for spd-clustering command."""
    parser = argparse.ArgumentParser(
        prog="spd-clustering",
        description="Submit clustering runs to SLURM. Arguments specified here will override the "
        "corresponding value in the config file.",
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to pipeline config file",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        help="Number of clustering runs in the ensemble (overrides value in config file)",
    )
    parser.add_argument(
        "--wandb-project",
        type=read_noneable_str,
        default=_NO_ARG_PARSSED_SENTINEL,
        help="WandB project name (if not provided, WandB logging is disabled)",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB entity name (user or team)",
    )
    parser.add_argument(
        "--distances-methods",
        type=str,
        default=None,
        help="Comma-separated list of distance methods (e.g., 'perm_invariant_hamming,matching_dist')",
    )
    parser.add_argument(
        "--local",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run locally instead of submitting to SLURM (required if slurm_job_name_prefix and slurm_partition are None in config)",
    )
    parser.add_argument(
        "--local-clustering-parallel",
        action="store_true",
        help="If running locally, whether to run clustering runs in parallel",
    )
    parser.add_argument(
        "--local-calc-distances-parallel",
        action="store_true",
        help="If running locally, whether to run distance calculations in parallel",
    )
    parser.add_argument(
        "--track-resources-calc-distances",
        action="store_true",
        help="If running locally, whether to track resource usage during distance calculations",
    )

    args = parser.parse_args()

    pipeline_config = ClusteringPipelineConfig.from_file(args.config)
    overrides: dict[str, Any] = {}

    if args.n_runs is not None:
        overrides["n_runs"] = args.n_runs
    if args.wandb_project is not _NO_ARG_PARSSED_SENTINEL:
        overrides["wandb_project"] = args.wandb_project
    if args.wandb_entity is not None:
        overrides["wandb_entity"] = args.wandb_entity
    if args.distances_methods is not None:
        # Parse comma-separated list of distance methods
        methods = [method.strip() for method in args.distances_methods.split(",")]
        overrides["distances_methods"] = methods

    pipeline_config = replace_pydantic_model(pipeline_config, overrides)

    main(
        pipeline_config=pipeline_config,
        local=args.local,
        local_clustering_parallel=args.local_clustering_parallel,
        local_calc_distances_parallel=args.local_calc_distances_parallel,
        track_resources_calc_distances=args.track_resources_calc_distances,
    )


if __name__ == "__main__":
    cli()
