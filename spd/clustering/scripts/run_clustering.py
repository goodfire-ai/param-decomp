"""Perform a single clustering run (harvest + merge in one process).

Called standalone or via `spd-clustering` (run_pipeline.py) for ensemble runs.
The ensemble pipeline varies dataset seeds across runs for stability analysis.

Output:
    <SPD_OUT_DIR>/clustering/runs/<run_id>/
        ├── clustering_run_config.json
        └── history.zip
"""

import argparse
import gc
import os
import tempfile
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

from spd.clustering.activations import collect_memberships
from spd.clustering.clustering_run_config import ClusteringRunConfig
from spd.clustering.consts import ClusterCoactivationShaped, ComponentLabels
from spd.clustering.dataset import create_clustering_dataloader
from spd.clustering.ensemble_registry import _ENSEMBLE_REGISTRY_DB, register_clustering_run
from spd.clustering.math.merge_matrix import GroupMerge
from spd.clustering.math.semilog import semilog
from spd.clustering.merge import LogCallback, merge_iteration_memberships
from spd.clustering.merge_history import MergeHistory
from spd.clustering.plotting.activations import plot_activations
from spd.clustering.plotting.merge import plot_merge_history_cluster_sizes, plot_merge_iteration
from spd.clustering.storage import StorageBase
from spd.clustering.wandb_tensor_info import wandb_log_tensor
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import replace_pydantic_model
from spd.utils.run_utils import _NO_ARG_PARSSED_SENTINEL, ExecutionStamp, read_noneable_str

os.environ["WANDB_QUIET"] = "true"


class ClusteringRunStorage(StorageBase):
    _CONFIG = "clustering_run_config.json"
    _HISTORY = "history.zip"

    def __init__(self, execution_stamp: ExecutionStamp) -> None:
        super().__init__(execution_stamp)
        self.config_path: Path = self.base_dir / self._CONFIG
        self.history_path: Path = self.base_dir / self._HISTORY


# ── WandB logging ──────────────────────────────────────────────────────────


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
    intervals = run_config.logging_intervals

    if iter_idx % intervals.stat == 0:
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

    if iter_idx % intervals.tensor == 0:
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

    if iter_idx > 0 and iter_idx % intervals.artifact == 0:
        with tempfile.NamedTemporaryFile() as tmp_file:
            file = Path(tmp_file.name)
            merge_history.save(file)
            artifact = wandb.Artifact(
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

    if iter_idx % intervals.plot == 0:
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


# ── Main ───────────────────────────────────────────────────────────────────


def main(run_config: ClusteringRunConfig) -> Path:
    execution_stamp = ExecutionStamp.create(run_type="clustering/runs", create_snapshot=False)
    storage = ClusteringRunStorage(execution_stamp)
    clustering_run_id = execution_stamp.run_id
    logger.info(f"Clustering run {clustering_run_id} → {storage.base_dir}")

    # Ensemble registration (seed varies per run)
    if run_config.ensemble_id:
        assigned_idx = register_clustering_run(
            pipeline_run_id=run_config.ensemble_id,
            clustering_run_id=clustering_run_id,
        )
        logger.info(
            f"Registered with pipeline {run_config.ensemble_id} "
            f"at index {assigned_idx} in {_ENSEMBLE_REGISTRY_DB}"
        )
        run_config = replace_pydantic_model(
            run_config, {"dataset_seed": run_config.dataset_seed + assigned_idx}
        )

    run_config.to_file(storage.config_path)

    device = get_device()
    spd_run = SPDRunInfo.from_path(run_config.model_path)
    task_name = spd_run.config.task_config.task_name
    model = ComponentModel.from_run_info(spd_run).to(device)
    dataloader = create_clustering_dataloader(
        model_path=run_config.model_path,
        task_name=task_name,
        batch_size=run_config.batch_size,
        seed=run_config.dataset_seed,
    )

    # WandB
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
            ],
        )

    # Harvest
    mc = run_config.merge_config
    processed = collect_memberships(
        model=model,
        dataloader=dataloader,
        task_name=task_name,
        device=device,
        activation_threshold=mc.activation_threshold,
        filter_dead_threshold=mc.filter_dead_threshold,
        filter_dead_stat=mc.filter_dead_stat,
        filter_modules=mc.filter_modules,
        n_tokens=run_config.n_tokens,
        n_tokens_per_seq=run_config.n_tokens_per_seq,
        use_all_tokens_per_seq=run_config.use_all_tokens_per_seq,
        n_samples=run_config.n_samples or run_config.batch_size,
        dataset_seed=run_config.dataset_seed,
    )

    if wandb_run is not None and processed.preview is not None:
        plot_activations(
            processed_activations=processed.preview,
            save_dir=None,
            n_samples_max=256,
            wandb_run=wandb_run,
        )
        wandb_log_tensor(wandb_run, processed.preview.activations, "activations", 0, single=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Merge
    log_callback: LogCallback | None = (
        partial(_log_callback, run=wandb_run, run_config=run_config)
        if wandb_run is not None
        else None
    )
    history = merge_iteration_memberships(
        merge_config=mc,
        memberships=processed.memberships,
        n_samples=processed.n_samples,
        component_labels=ComponentLabels(processed.labels.copy()),
        log_callback=log_callback,
    )

    history.save(storage.history_path)
    logger.info(f"History saved to {storage.history_path}")

    if wandb_run is not None:
        fig_cs: Figure = plot_merge_history_cluster_sizes(history=history)
        wandb_run.log(
            {"plots/merge_history_cluster_sizes": wandb.Image(fig_cs)},
            step=history.n_iters_current,
        )
        plt.close(fig_cs)

        artifact = wandb.Artifact(
            name="merge_history",
            type="merge_history",
            metadata={"n_iters_current": history.n_iters_current},
        )
        artifact.add_file(str(storage.history_path))
        wandb_run.log_artifact(artifact)
        wandb_run.finish()

    return storage.history_path


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run a single clustering run")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pipeline-run-id", type=str, default=None)
    parser.add_argument("--wandb-project", type=read_noneable_str, default=_NO_ARG_PARSSED_SENTINEL)
    parser.add_argument("--wandb-entity", type=str, default=None)
    args = parser.parse_args()

    run_config = ClusteringRunConfig.from_file(args.config)
    overrides: dict[str, Any] = {}
    if args.pipeline_run_id is not None:
        overrides["ensemble_id"] = args.pipeline_run_id
    if args.wandb_project is not _NO_ARG_PARSSED_SENTINEL:
        overrides["wandb_project"] = args.wandb_project
    if args.wandb_entity is not None:
        overrides["wandb_entity"] = args.wandb_entity
    if overrides:
        run_config = replace_pydantic_model(run_config, overrides)

    main(run_config)


if __name__ == "__main__":
    cli()
