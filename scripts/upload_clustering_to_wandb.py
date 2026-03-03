"""Retroactively upload a clustering run to W&B.

Reconstructs metrics from the saved MergeHistory and uploads them to W&B.

Usage:
    python scripts/upload_clustering_to_wandb.py \
        /mnt/polished-lake/artifacts/mechanisms/spd/clustering/runs/c-3f209c9e \
        --project spd
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import wandb
from wandb.sdk.wandb_run import Run

from spd.clustering.clustering_run_config import ClusteringRunConfig
from spd.clustering.math.merge_matrix import GroupMerge
from spd.clustering.merge_history import MergeHistory
from spd.clustering.plotting.merge import plot_merge_history_cluster_sizes
from spd.models.component_model import SPDRunInfo
from spd.spd_types import TaskName


def main(run_dir: Path, project: str) -> None:
    assert run_dir.is_dir(), f"Not a directory: {run_dir}"
    run_id = run_dir.name

    # Load config and history
    config = ClusteringRunConfig.from_file(run_dir / "clustering_run_config.json")
    history_path = run_dir / "history.zip"
    history = MergeHistory.read(history_path)
    print(f"Loaded history: {history.n_iters_current} iterations, {history.c_components} components")

    # Get task name from the SPD run
    spd_run = SPDRunInfo.from_path(config.model_path)
    task_name: TaskName = spd_run.config.task_config.task_name

    # Init W&B
    wandb_run: Run = wandb.init(
        id=run_id,
        entity=config.wandb_entity,
        project=project,
        config=config.model_dump(mode="json"),
        tags=[
            "clustering",
            f"task:{task_name}",
            f"model:{config.wandb_decomp_model}",
            "retroactive_upload",
        ],
    )

    # Log metrics
    stat_interval = config.logging_intervals.stat
    tensor_interval = config.logging_intervals.tensor

    for iter_idx in range(0, history.n_iters_current, stat_interval):
        k_groups = int(history.merges.k_groups[iter_idx].item())
        metrics: dict[str, float | int] = {"k_groups": k_groups}

        if iter_idx % tensor_interval == 0:
            merge: GroupMerge = history.merges[iter_idx]
            group_sizes = merge.components_per_group
            fraction_singleton = float((group_sizes == 1).float().mean().item())
            num_nonsingleton = int((group_sizes > 1).sum().item())
            metrics["fraction_singleton_groups"] = fraction_singleton
            metrics["num_nonsingleton_groups"] = num_nonsingleton

        wandb_run.log(metrics, step=iter_idx)

    n_logged = len(range(0, history.n_iters_current, stat_interval))
    print(f"Logged {n_logged} metric steps")

    # Log cluster sizes plot
    fig = plot_merge_history_cluster_sizes(history=history)
    wandb_run.log(
        {"plots/merge_history_cluster_sizes": wandb.Image(fig)},
        step=history.n_iters_current,
    )
    plt.close(fig)
    print("Logged cluster sizes plot")

    # Upload merge history artifact
    artifact = wandb.Artifact(
        name="merge_history",
        type="merge_history",
        description="Merge history",
        metadata={"n_iters_current": history.n_iters_current, "filename": str(history_path)},
    )
    artifact.add_file(str(history_path))
    wandb_run.log_artifact(artifact)
    print("Uploaded merge_history artifact")

    wandb_run.finish()
    print(f"Done. Run: https://wandb.ai/{config.wandb_entity}/{project}/runs/{run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload clustering run to W&B retroactively")
    parser.add_argument("run_dir", type=Path, help="Path to clustering run directory")
    parser.add_argument("--project", required=True, help="W&B project name")
    args = parser.parse_args()
    main(args.run_dir, args.project)
