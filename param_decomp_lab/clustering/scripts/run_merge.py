"""Run merge iteration on a pre-harvested membership snapshot.

No GPU required — purely CPU work.

Output:
    <PARAM_DECOMP_OUT_DIR>/clustering/runs/<run_id>/
        ├── merge_config.json
        ├── history.zip
        └── plots/                  # only when --plot is passed
            ├── cluster_sizes.png
            └── iter_<idx>.png
"""

import argparse
import json
import os
import random
import shlex
from pathlib import Path

import numpy as np
from jaxtyping import Float
from matplotlib import pyplot as plt

from param_decomp.log import logger
from param_decomp_lab.clustering.math.merge_matrix import GroupMerge
from param_decomp_lab.clustering.memberships import ProcessedMemberships
from param_decomp_lab.clustering.merge import LogCallback, merge_iteration_memberships
from param_decomp_lab.clustering.merge_config import MergeConfig
from param_decomp_lab.clustering.merge_history import MergeHistory
from param_decomp_lab.clustering.paths import clustering_run_dir, new_run_id
from param_decomp_lab.clustering.plotting.merge import (
    plot_coact_and_costs,
    plot_merge_history_cluster_sizes,
)
from param_decomp_lab.clustering.types import (
    ClusterCoactivationShaped,
    ComponentLabels,
)

os.environ["WANDB_QUIET"] = "true"


def _make_iteration_plot_callback(plot_dir: Path, plot_every: int) -> LogCallback:
    plot_dir.mkdir(parents=True, exist_ok=True)

    def callback(
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
        diag_acts: "Float[np.ndarray, ' k_groups']",
    ) -> None:
        del component_labels, current_merge, merge_history, k_groups
        del merge_pair_cost, mdl_loss, mdl_loss_norm, diag_acts
        if iter_idx % plot_every != 0:
            return
        fig = plot_coact_and_costs(current_coact, costs, iteration=iter_idx)
        fig.savefig(plot_dir / f"iter_{iter_idx:05d}.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

    return callback


def merge(
    snapshot_path: Path,
    merge_config: MergeConfig,
    run_id: str,
    seed: int,
    plot_dir: Path | None,
) -> Path:
    """Run merge iteration, return history path.

    `seed` seeds the stdlib `random` the stochastic merge-pair samplers draw from, so
    ensemble members with distinct seeds produce independent merge trajectories.
    `plot_dir` (when given) receives per-iteration coactivation/cost heatmaps and a final
    cluster-sizes plot.
    """
    random.seed(seed)
    out = clustering_run_dir(run_id)
    out.mkdir(parents=True, exist_ok=True)
    logger.info(f"Merge run {run_id} → {out}")

    (out / "merge_config.json").write_text(
        json.dumps(
            {
                "snapshot_path": str(snapshot_path),
                "seed": seed,
                "merge_config": merge_config.model_dump(mode="json"),
            },
            indent=2,
        )
    )

    processed = ProcessedMemberships.load(snapshot_path)
    logger.info(f"Loaded: {processed.n_components_alive} components, {processed.n_samples} samples")

    log_callback: LogCallback | None = None
    if plot_dir is not None:
        log_every = max(1, merge_config.get_num_iters(processed.n_components_alive) // 10)
        log_callback = _make_iteration_plot_callback(plot_dir, plot_every=log_every)

    history = merge_iteration_memberships(
        merge_config=merge_config,
        memberships=processed.memberships,
        n_samples=processed.n_samples,
        component_labels=ComponentLabels(list(processed.labels)),
        log_callback=log_callback,
    )

    history_path = out / "history.zip"
    history.save(history_path)
    logger.info(f"History saved to {history_path}")

    if plot_dir is not None:
        fig = plot_merge_history_cluster_sizes(history)
        fig.savefig(plot_dir / "cluster_sizes.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info(f"Diagnostic plots saved to {plot_dir}")

    return history_path


def get_command(
    snapshot_path: Path,
    merge_config_path: Path,
    run_id: str,
    seed: int,
    plot: bool,
) -> str:
    """Shell command for one ensemble member's merge (depends on its harvest)."""
    parts = [
        "python",
        "-m",
        "param_decomp_lab.clustering.scripts.run_merge",
        snapshot_path.as_posix(),
        merge_config_path.as_posix(),
        "--run-id",
        run_id,
        "--seed",
        str(seed),
    ]
    if plot:
        parts.append("--plot")
    return shlex.join(parts)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Merge from a membership snapshot.")
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("merge_config", type=Path)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", action="store_true", help="emit per-run diagnostic plots")
    args = parser.parse_args()
    run_id = args.run_id or new_run_id()
    plot_dir = clustering_run_dir(run_id) / "plots" if args.plot else None
    merge(
        snapshot_path=args.snapshot,
        merge_config=MergeConfig.from_file(args.merge_config),
        run_id=run_id,
        seed=args.seed,
        plot_dir=plot_dir,
    )


if __name__ == "__main__":
    cli()
