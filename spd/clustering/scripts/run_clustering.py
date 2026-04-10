"""Perform a single clustering run: harvest → merge with wandb logging.

Called standalone or via `spd-clustering` (run_pipeline.py) for ensemble runs.
The pipeline pre-assigns run IDs and creates the git snapshot;
each SLURM task checks out that snapshot and receives its run ID via --run-id.

Output:
    <SPD_OUT_DIR>/clustering/harvests/<harvest_id>/   (from harvest)
    <SPD_OUT_DIR>/clustering/runs/<run_id>/            (from merge)
        ├── merge_config.json
        └── history.zip
"""

import argparse
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

from spd.clustering.clustering_run_config import ClusteringRunConfig
from spd.clustering.consts import ClusterCoactivationShaped, ComponentLabels
from spd.clustering.math.merge_matrix import GroupMerge
from spd.clustering.math.semilog import semilog
from spd.clustering.memberships import ProcessedMemberships
from spd.clustering.merge import LogCallback
from spd.clustering.merge_history import MergeHistory
from spd.clustering.paths import new_run_id
from spd.clustering.plotting.activations import plot_activations
from spd.clustering.plotting.merge import plot_merge_history_cluster_sizes, plot_merge_iteration
from spd.clustering.scripts.run_harvest import harvest as harvest_fn
from spd.clustering.scripts.run_merge import merge
from spd.clustering.wandb_tensor_info import wandb_log_tensor
from spd.utils.general_utils import replace_pydantic_model

os.environ["WANDB_QUIET"] = "true"


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


def main(
    run_config: ClusteringRunConfig,
    run_id: str,
    seed_offset: int = 0,
) -> Path:
    if seed_offset != 0:
        hc = run_config.harvest
        run_config = replace_pydantic_model(
            run_config, {"harvest": {"dataset_seed": hc.dataset_seed + seed_offset}}
        )

    # Harvest
    snapshot_path = harvest_fn(run_config.harvest)

    # WandB
    wandb_run: Run | None = None
    if run_config.wandb_project is not None:
        wandb_run = wandb.init(
            id=run_id,
            entity=run_config.wandb_entity,
            project=run_config.wandb_project,
            group=run_config.ensemble_id,
            config=run_config.model_dump(mode="json"),
            tags=["clustering", f"model:{run_config.wandb_decomp_model}"],
        )

    # Log activation preview
    if wandb_run is not None:
        loaded = ProcessedMemberships.load(snapshot_path)
        if loaded.preview is not None:
            plot_activations(
                processed_activations=loaded.preview,
                save_dir=None,
                n_samples_max=256,
                wandb_run=wandb_run,
            )
            wandb_log_tensor(wandb_run, loaded.preview.activations, "activations", 0, single=True)
        del loaded

    # Merge
    log_callback: LogCallback | None = (
        partial(_log_callback, run=wandb_run, run_config=run_config)
        if wandb_run is not None
        else None
    )
    history_path = merge(
        snapshot_path=snapshot_path,
        merge_config=run_config.merge,
        run_id=run_id,
        log_callback=log_callback,
    )

    if wandb_run is not None:
        history = MergeHistory.read(history_path)
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
        artifact.add_file(str(history_path))
        wandb_run.log_artifact(artifact)
        wandb_run.finish()

    return history_path


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run a single clustering run")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--ensemble-id", type=str, default=None)
    args = parser.parse_args()

    run_config = ClusteringRunConfig.from_file(args.config)
    overrides: dict[str, Any] = {}
    if args.wandb_project is not None:
        overrides["wandb_project"] = args.wandb_project
    if args.wandb_entity is not None:
        overrides["wandb_entity"] = args.wandb_entity
    if args.ensemble_id is not None:
        overrides["ensemble_id"] = args.ensemble_id
    if overrides:
        run_config = replace_pydantic_model(run_config, overrides)

    run_id = args.run_id or new_run_id()
    main(run_config, run_id=run_id, seed_offset=args.seed_offset)


def get_command(
    config_path: Path,
    run_id: str,
    seed_offset: int,
    ensemble_id: str,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
) -> str:
    import shlex

    parts = [
        "python",
        "spd/clustering/scripts/run_clustering.py",
        "--config",
        config_path.as_posix(),
        "--run-id",
        run_id,
        "--seed-offset",
        str(seed_offset),
        "--ensemble-id",
        ensemble_id,
    ]
    if wandb_project is not None:
        parts += ["--wandb-project", wandb_project]
    if wandb_entity is not None:
        parts += ["--wandb-entity", wandb_entity]
    return shlex.join(parts)


if __name__ == "__main__":
    cli()
