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
import shlex
from pathlib import Path
from typing import Any

import wandb_workspaces.workspaces as ws
from pydantic import Field, PositiveInt, field_validator, model_validator

from spd.base_config import BaseConfig
from spd.clustering.clustering_run_config import ClusteringRunConfig
from spd.clustering.consts import DistancesMethod
from spd.clustering.storage import StorageBase
from spd.log import logger
from spd.utils.general_utils import replace_pydantic_model
from spd.utils.run_utils import (
    _NO_ARG_PARSSED_SENTINEL,
    ExecutionStamp,
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


class ClusteringPipelineStorage(StorageBase):
    """Storage paths for clustering pipeline (ensemble).

    All paths are relative to ExecutionStamp.out_dir.
    """

    # Relative path constants
    _PIPELINE_CONFIG = "pipeline_config.yaml"
    _RUN_IDS = "run_ids.json"
    _ENSEMBLE_META = "ensemble_meta.json"
    _ENSEMBLE_MERGE_ARRAY = "ensemble_merge_array.npz"

    def __init__(self, execution_stamp: ExecutionStamp) -> None:
        super().__init__(execution_stamp)
        self.pipeline_config_path: Path = self.base_dir / self._PIPELINE_CONFIG
        self.run_ids_path: Path = self.base_dir / self._RUN_IDS
        self.ensemble_meta_path: Path = self.base_dir / self._ENSEMBLE_META
        self.ensemble_merge_array_path: Path = self.base_dir / self._ENSEMBLE_MERGE_ARRAY

    def distances_path(self, method: DistancesMethod) -> Path:
        return self.base_dir / f"distances_{method}.npz"


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


def generate_clustering_commands(
    pipeline_config: ClusteringPipelineConfig,
    pipeline_run_id: str,
) -> list[str]:
    """Generate commands for each clustering run.

    Args:
        pipeline_config: Pipeline configuration
        pipeline_run_id: Pipeline run ID (each run will create its own ExecutionStamp)

    Returns:
        List of shell-safe command strings
    """
    commands: list[str] = []

    for idx in range(pipeline_config.n_runs):
        cmd_parts = [
            "python",
            "spd/clustering/scripts/run_clustering.py",
            "--config",
            pipeline_config.clustering_run_config_path.as_posix(),
            "--pipeline-run-id",
            pipeline_run_id,
            "--idx-in-ensemble",
            str(idx),
            "--wandb-project",
            str(pipeline_config.wandb_project),
            "--wandb-entity",
            pipeline_config.wandb_entity,
        ]

        commands.append(shlex.join(cmd_parts))

    return commands


def generate_calc_distances_commands(
    pipeline_run_id: str, distances_methods: list[DistancesMethod]
) -> list[str]:
    """Generate commands for calculating distances.

    Args:
        pipeline_run_id: Pipeline run ID (will query registry for clustering runs)
        distances_methods: List of methods for calculating distances

    Returns:
        List of shell-safe command strings, one per method
    """
    commands: list[str] = []
    for method in distances_methods:
        commands.append(
            shlex.join(
                [
                    "python",
                    "spd/clustering/scripts/calc_distances.py",
                    "--pipeline-run-id",
                    pipeline_run_id,
                    "--distances-method",
                    method,
                ]
            )
        )
    return commands


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

    # Create ExecutionStamp for pipeline
    execution_stamp: ExecutionStamp = ExecutionStamp.create(
        run_type="clustering/ensembles",
        create_snapshot=pipeline_config.create_git_snapshot,
    )
    pipeline_run_id: str = execution_stamp.run_id
    logger.info(f"Pipeline run ID: {pipeline_run_id}")

    # Initialize storage
    storage = ClusteringPipelineStorage(execution_stamp)
    logger.info(f"Pipeline output directory: {storage.base_dir}")

    # Save pipeline config
    pipeline_config.to_file(storage.pipeline_config_path)
    logger.info(f"Pipeline config saved to {storage.pipeline_config_path}")

    # Create WandB workspace if requested
    if pipeline_config.wandb_project is not None:
        workspace_url = create_clustering_workspace_view(
            ensemble_id=pipeline_run_id,
            project=pipeline_config.wandb_project,
            entity=pipeline_config.wandb_entity,
        )
        logger.info(f"WandB workspace: {workspace_url}")

    # Generate commands for clustering runs
    clustering_commands = generate_clustering_commands(
        pipeline_config=pipeline_config,
        pipeline_run_id=pipeline_run_id,
    )

    # Generate commands for calculating distances
    calc_distances_commands = generate_calc_distances_commands(
        pipeline_run_id=pipeline_run_id,
        distances_methods=pipeline_config.distances_methods,
    )

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
            f"distances via {method}": str(storage.plots_dir / f"distances_{method}.png")
            for method in pipeline_config.distances_methods
        }

        logger.values(
            {
                "Total clustering runs": len(clustering_commands),
                "Pipeline run ID": pipeline_run_id,
                "Pipeline output dir": str(storage.base_dir),
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
            snapshot_branch=execution_stamp.snapshot_branch,
            max_concurrent_tasks=pipeline_config.n_runs,  # Run all concurrently
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
                snapshot_branch=execution_stamp.snapshot_branch,
                dependency_job_id=array_job_id,
            )
            dist_script = generate_script(dist_config, cmd)
            dist_result = submit_slurm_job(dist_script, f"calc_distances_{method}")
            calc_distances_job_ids.append(dist_result.job_id)
            calc_distances_logs.append(dist_result.log_pattern)

        logger.section("Jobs submitted successfully!")

        # Build distances plot paths dict
        distances_plots = {
            method: str(storage.plots_dir / f"distances_{method}.png")
            for method in pipeline_config.distances_methods
        }

        logger.values(
            {
                "Clustering Array Job ID": array_job_id,
                "Calc Distances Job IDs": ", ".join(calc_distances_job_ids),
                "Total clustering runs": len(clustering_commands),
                "Pipeline run ID": pipeline_run_id,
                "Pipeline output dir": str(storage.base_dir),
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
