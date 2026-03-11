"""Perform a single clustering run.

This can be run as a standalone script, or called via `spd-clustering`
(i.e. clustering/scripts/run_pipeline.py). If called via spd-clustering, the ensemble-key is passed
in to identify the run within the pipeline ensemble.

Output structure:
    <ExecutionStamp.out_dir>/  # from execution stamp (run_type="clustering/runs")
    ├── clustering_run_config.json
    └── history.npz
"""

import argparse
import gc
import os
import tempfile
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import wandb
from jaxtyping import Float, Int
from matplotlib.figure import Figure
from torch import Tensor
from wandb.sdk.wandb_run import Run

from spd.clustering.activations import (
    ProcessedActivations,
    collect_activations,
    component_activations,
    process_activations,
)
from spd.clustering.clustering_run_config import ClusteringRunConfig
from spd.clustering.consts import (
    ActivationsTensor,
    ClusterCoactivationShaped,
    ComponentLabels,
)
from spd.clustering.dataset import create_clustering_dataloader
from spd.clustering.ensemble_registry import _ENSEMBLE_REGISTRY_DB, register_clustering_run
from spd.clustering.math.merge_matrix import GroupMerge
from spd.clustering.math.semilog import semilog
from spd.clustering.merge import merge_iteration
from spd.clustering.merge_history import MergeHistory
from spd.clustering.plotting.activations import plot_activations
from spd.clustering.plotting.merge import plot_merge_history_cluster_sizes, plot_merge_iteration
from spd.clustering.storage import StorageBase
from spd.clustering.wandb_tensor_info import wandb_log_tensor
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.spd_types import TaskName
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import replace_pydantic_model
from spd.utils.run_utils import _NO_ARG_PARSSED_SENTINEL, ExecutionStamp, read_noneable_str

os.environ["WANDB_QUIET"] = "true"


class ClusteringRunStorage(StorageBase):
    """Storage paths for a single clustering run.

    All paths are relative to ExecutionStamp.out_dir.
    """

    # Relative path constants
    _CONFIG = "clustering_run_config.json"
    # we are saving a zip file with things in it besides npy files -- hence, `.zip` and not `.npz`
    _HISTORY = "history.zip"

    def __init__(self, execution_stamp: ExecutionStamp) -> None:
        super().__init__(execution_stamp)
        self.config_path: Path = self.base_dir / self._CONFIG
        self.history_path: Path = self.base_dir / self._HISTORY


LogCallback = Callable[
    [
        ClusterCoactivationShaped,
        ComponentLabels,
        GroupMerge,
        ClusterCoactivationShaped,
        MergeHistory,
        int,
        int,
        float,
        float,
        float,
        Float[Tensor, " k_groups"],
    ],
    None,
]


def _log_merge_history_plots(run: Run, history: MergeHistory) -> None:
    """Log merge history plots to WandB."""
    fig_cs: Figure = plot_merge_history_cluster_sizes(history=history)
    run.log(
        {"plots/merge_history_cluster_sizes": wandb.Image(fig_cs)},
        step=history.n_iters_current,
    )
    plt.close(fig_cs)


def _save_merge_history_artifact(
    run: Run,
    history_path: Path,
    history: MergeHistory,
) -> None:
    """Save merge history as WandB artifact."""
    artifact: wandb.Artifact = wandb.Artifact(
        name="merge_history",
        type="merge_history",
        description="Merge history",
        metadata={"n_iters_current": history.n_iters_current, "filename": str(history_path)},
    )
    artifact.add_file(str(history_path))
    run.log_artifact(artifact)


def _log_callback(
    run: Run,
    run_config: ClusteringRunConfig,
    current_coact: ClusterCoactivationShaped,
    component_labels: ComponentLabels,
    current_merge: GroupMerge,
    costs: ClusterCoactivationShaped,
    merge_history: MergeHistory,
    iter_idx: int,
    k_groups: int,
    merge_pair_cost: float,
    mdl_loss: float,
    mdl_loss_norm: float,
    diag_acts: Float[Tensor, " k_groups"],
) -> None:
    """Callback for logging during merge iteration."""
    if iter_idx % run_config.logging_intervals.stat == 0:
        run.log(
            {
                "k_groups": int(k_groups),
                "merge_pair_cost": merge_pair_cost,
                "merge_pair_cost_semilog[1e-3]": semilog(merge_pair_cost, epsilon=1e-3),
                "mdl_loss": float(mdl_loss),
                "mdl_loss_norm": float(mdl_loss_norm),
            },
            step=iter_idx,
        )

    if iter_idx % run_config.logging_intervals.tensor == 0:
        group_sizes: Int[Tensor, " k_groups"] = current_merge.components_per_group

        tensor_data: dict[str, Tensor] = {
            "coactivation": current_coact,
            "costs": costs,
            "group_sizes": group_sizes,
            "group_activations": diag_acts,
            "group_activations_over_sizes": (
                diag_acts / group_sizes.to(device=diag_acts.device).float()
            ),
        }

        fraction_singleton_groups: float = (group_sizes == 1).float().mean().item()
        if fraction_singleton_groups > 0:
            tensor_data["group_sizes.log1p"] = torch.log1p(group_sizes.float())

        fraction_zero_coacts: float = (current_coact == 0).float().mean().item()
        if fraction_zero_coacts > 0:
            tensor_data["coactivation.log1p"] = torch.log1p(current_coact.float())

        wandb_log_tensor(run, tensor_data, name="iters", step=iter_idx)

        run.log(
            {
                "fraction_singleton_groups": float(fraction_singleton_groups),
                "num_nonsingleton_groups": int((group_sizes > 1).sum().item()),
                "fraction_zero_coacts": float(fraction_zero_coacts),
            },
            step=iter_idx,
        )

    if iter_idx > 0 and iter_idx % run_config.logging_intervals.artifact == 0:
        with tempfile.NamedTemporaryFile() as tmp_file:
            file: Path = Path(tmp_file.name)
            merge_history.save(file)
            artifact: wandb.Artifact = wandb.Artifact(
                name=f"merge_hist_iter.iter_{iter_idx}",
                type="merge_hist_iter",
                description=f"Group indices at iteration {iter_idx}",
                metadata={
                    "iteration": iter_idx,
                    "config": merge_history.merge_config.model_dump(mode="json"),
                },
            )
            artifact.add_file(str(file))
            run.log_artifact(artifact)

    if iter_idx % run_config.logging_intervals.plot == 0:
        fig: Figure = plot_merge_iteration(
            current_merge=current_merge,
            current_coact=current_coact,
            costs=costs,
            iteration=iter_idx,
            component_labels=component_labels,
            show=False,
        )
        run.log({"plots/merges": wandb.Image(fig)}, step=iter_idx)
        plt.close(fig)


def main(run_config: ClusteringRunConfig) -> Path:
    """A single clustering run.

    Args:
        run_config: Runtime parameters for this clustering run

    Returns:
        Path to saved merge history file
    """
    # Create ExecutionStamp and storage
    # don't create git snapshot -- if we are part of an ensemble, the snapshot should be created by the pipeline
    execution_stamp = ExecutionStamp.create(
        run_type="clustering/runs",
        create_snapshot=False,
    )
    storage = ClusteringRunStorage(execution_stamp)
    clustering_run_id = execution_stamp.run_id
    logger.info(f"Clustering run ID: {clustering_run_id}")

    # Register with ensemble if this is part of a pipeline
    assigned_idx: int | None
    if run_config.ensemble_id:
        assigned_idx = register_clustering_run(
            pipeline_run_id=run_config.ensemble_id,
            clustering_run_id=clustering_run_id,
        )

        logger.info(
            f"Registered with pipeline {run_config.ensemble_id} at index {assigned_idx} in {_ENSEMBLE_REGISTRY_DB}"
        )
        # IMPORTANT: set dataset seed based on assigned index
        run_config = replace_pydantic_model(
            run_config,
            {"dataset_seed": run_config.dataset_seed + assigned_idx},
        )
    else:
        assigned_idx = None

    # save config
    run_config.to_file(storage.config_path)
    logger.info(f"Config saved to {storage.config_path}")

    # start
    logger.info("Starting clustering run")
    logger.info(f"Output directory: {storage.base_dir}")
    device = get_device()

    spd_run = SPDRunInfo.from_path(run_config.model_path)
    task_name: TaskName = spd_run.config.task_config.task_name

    # 1. Create dataloader
    logger.info(f"Loading dataset (seed={run_config.dataset_seed})")
    dataloader = create_clustering_dataloader(
        model_path=run_config.model_path,
        task_name=task_name,
        batch_size=run_config.batch_size,
        seed=run_config.dataset_seed,
    )

    # 2. Setup WandB for this run
    wandb_run: Run | None = None
    if run_config.wandb_project is not None:
        wandb_run = wandb.init(
            id=clustering_run_id,
            entity=run_config.wandb_entity,
            project=run_config.wandb_project,
            group=run_config.ensemble_id,
            config=run_config.model_dump(mode="json"),
            tags=[
                "clustering",
                f"task:{task_name}",
                f"model:{run_config.wandb_decomp_model}",
                f"ensemble_id:{run_config.ensemble_id}",
                f"assigned_idx:{assigned_idx}",
            ],
        )

    # 3. Load model
    logger.info("Loading model")
    model = ComponentModel.from_run_info(spd_run).to(device)

    # 4. Compute activations
    logger.info("Computing activations")
    if task_name == "lm":
        assert run_config.n_tokens is not None, "n_tokens must be set for LM tasks"
        assert run_config.n_tokens_per_seq is not None, "n_tokens_per_seq must be set for LM tasks"
        activations_dict = collect_activations(
            model=model,
            dataloader=dataloader,
            n_tokens=run_config.n_tokens,
            n_tokens_per_seq=run_config.n_tokens_per_seq,
            device=device,
            seed=run_config.dataset_seed,
        )
    else:
        # resid_mlp: single batch, no sequence dimension
        batch_data = next(iter(dataloader))
        batch, _ = batch_data  # DatasetGeneratedDataLoader yields (batch, labels)
        activations_dict = component_activations(
            model=model,
            batch=batch,
            device=device,
        )

    # 5. Process activations
    logger.info("Processing activations")
    processed_activations: ProcessedActivations = process_activations(
        activations=activations_dict,
        filter_dead_threshold=run_config.merge_config.filter_dead_threshold,
        seq_mode=None,
        filter_modules=run_config.merge_config.filter_modules,
    )

    # 6. Log activations (if WandB enabled)
    if wandb_run is not None:
        logger.info("Plotting activations")
        plot_activations(
            processed_activations=processed_activations,
            save_dir=None,  # Don't save to disk, only WandB
            n_samples_max=256,
            wandb_run=wandb_run,
        )
        wandb_log_tensor(
            wandb_run,
            processed_activations.activations,
            "activations",
            0,
            single=True,
        )

    # Extract what we need, then free the model and temporary objects
    activations: ActivationsTensor = processed_activations.activations.to(device)
    component_labels: ComponentLabels = ComponentLabels(processed_activations.labels.copy())
    del processed_activations
    del activations_dict
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 7. Run merge iteration
    logger.info("Starting merging")
    log_callback: LogCallback | None = (
        partial(_log_callback, run=wandb_run, run_config=run_config)
        if wandb_run is not None
        else None
    )

    history: MergeHistory = merge_iteration(
        merge_config=run_config.merge_config,
        activations=activations,
        component_labels=component_labels,
        log_callback=log_callback,
    )

    # 8. Save merge history

    history.save(storage.history_path)
    logger.info(f"History saved to {storage.history_path}")

    # 9. Log to WandB
    if wandb_run is not None:
        _log_merge_history_plots(wandb_run, history)
        _save_merge_history_artifact(wandb_run, storage.history_path, history)
        wandb_run.finish()
        logger.info("WandB run finished")

    return storage.history_path


def cli() -> None:
    """CLI for running a single clustering run."""
    parser = argparse.ArgumentParser(description="Run clustering on a single dataset")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to ClusteringRunConfig file",
    )
    parser.add_argument(
        "--pipeline-run-id",
        type=str,
        default=None,
        help="Pipeline run ID (ensemble identifier). If provided with --idx-in-ensemble, registers run.",
    )
    parser.add_argument(
        "--idx-in-ensemble",
        type=int,
        default=None,
        help="Index of this run in the ensemble",
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
    args: argparse.Namespace = parser.parse_args()

    # Load base config
    run_config = ClusteringRunConfig.from_file(args.config)

    # Override config values from CLI
    overrides: dict[str, Any] = {}

    # Handle ensemble-related overrides
    if args.pipeline_run_id is not None:
        overrides["ensemble_id"] = args.pipeline_run_id

    if args.wandb_project is not _NO_ARG_PARSSED_SENTINEL:
        overrides["wandb_project"] = args.wandb_project
    if args.wandb_entity is not None:
        overrides["wandb_entity"] = args.wandb_entity

    run_config = replace_pydantic_model(run_config, overrides)

    # Run clustering
    main(run_config)


if __name__ == "__main__":
    cli()
