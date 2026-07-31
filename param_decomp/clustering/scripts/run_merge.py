"""Run merge iteration on a pre-harvested membership snapshot.

No GPU required — purely CPU work.

Output:
    <data_root>/clustering/runs/<run_id>/
        ├── merge_config.json
        ├── history.zip
        └── plots/                  # only when --plot is passed
            ├── cluster_sizes.png
            └── iter_<idx>.png
"""

import argparse
import json
import random
import shlex
from pathlib import Path

import numpy as np
from jaxtyping import Float
from matplotlib import pyplot as plt

from param_decomp.clustering.math.merge_matrix import GroupMerge
from param_decomp.clustering.memberships import ProcessedMemberships
from param_decomp.clustering.merge import LogCallback, merge_iteration_memberships
from param_decomp.clustering.merge_config import MergeConfig
from param_decomp.clustering.merge_history import MergeHistory
from param_decomp.clustering.paths import clustering_run_dir, new_run_id
from param_decomp.clustering.plotting.merge import (
    plot_coact_and_costs,
    plot_merge_history_cluster_sizes,
)
from param_decomp.clustering.types import (
    ClusterCoactivationShaped,
    ComponentLabels,
)
from param_decomp.core.log import logger


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
    data_root: Path,
) -> Path:
    """Run merge iteration, return history path.

    `seed` seeds the stdlib `random` the stochastic merge-pair samplers draw from, so
    ensemble members with distinct seeds produce independent merge trajectories.
    `plot_dir` (when given) receives per-iteration coactivation/cost heatmaps and a final
    cluster-sizes plot.
    """
    random.seed(seed)
    out = clustering_run_dir(data_root, run_id)
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
    data_root: Path,
) -> str:
    """Shell command for one ensemble member's merge (depends on its harvest)."""
    parts = [
        "python",
        "-m",
        "param_decomp.clustering.scripts.run_merge",
        snapshot_path.as_posix(),
        merge_config_path.as_posix(),
        "--run-id",
        run_id,
        "--seed",
        str(seed),
        "--data-root",
        data_root.as_posix(),
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
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--plot", action="store_true", help="emit per-run diagnostic plots")
    args = parser.parse_args()
    run_id = args.run_id or new_run_id()
    plot_dir = clustering_run_dir(args.data_root, run_id) / "plots" if args.plot else None
    merge(
        snapshot_path=args.snapshot,
        merge_config=MergeConfig.from_file(args.merge_config),
        run_id=run_id,
        seed=args.seed,
        plot_dir=plot_dir,
        data_root=args.data_root,
    )


if __name__ == "__main__":
    cli()
