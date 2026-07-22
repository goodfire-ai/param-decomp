"""Consensus driver: cross-run label normalization, per-iteration pairwise distance
matrices, and the ensemble stability plot.

Loads an ensemble of merge histories (one per `pd-clustering` member), normalizes their
component labels into a shared space (`MergeHistoryEnsemble.normalized`), computes
pairwise distances per iteration (`math.compute_distances`), and writes:

    PARAM_DECOMP_OUT_DIR/clustering/ensembles/<ensemble_id>/
        ├── ensemble_meta.json                # normalization metadata
        ├── ensemble_merge_array.npz          # normalized merge array
        ├── distances_<method>.npz            # per-iteration distance tensor
        └── plots/distances_<method>.png      # stability plot
"""

import argparse
import json
import shlex

import numpy as np
from matplotlib import pyplot as plt

from param_decomp.log import logger
from param_decomp_lab.clustering.math.merge_distances import compute_distances
from param_decomp_lab.clustering.merge_history import MergeHistory, MergeHistoryEnsemble
from param_decomp_lab.clustering.paths import clustering_ensemble_dir, clustering_run_dir
from param_decomp_lab.clustering.plotting.merge import plot_dists_distribution
from param_decomp_lab.clustering.types import DistancesArray, DistancesMethod


def calc_distances(
    ensemble_id: str,
    clustering_run_ids: list[str],
    distances_method: DistancesMethod,
) -> None:
    assert clustering_run_ids, "no clustering run ids provided"
    logger.info(f"Consensus for ensemble {ensemble_id}: {len(clustering_run_ids)} runs")

    histories: list[MergeHistory] = []
    for run_id in clustering_run_ids:
        history_path = clustering_run_dir(run_id) / "history.zip"
        assert history_path.exists(), f"history not found for run {run_id}: {history_path}"
        histories.append(MergeHistory.read(history_path))
        logger.info(f"Loaded history for run {run_id}")

    ensemble = MergeHistoryEnsemble(data=histories)
    merge_array, merge_meta = ensemble.normalized()

    pipeline_dir = clustering_ensemble_dir(ensemble_id)
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    (pipeline_dir / "ensemble_meta.json").write_text(json.dumps(merge_meta, indent=2))
    np.savez_compressed(pipeline_dir / "ensemble_merge_array.npz", merge_array=merge_array)
    logger.info(f"Saved normalized ensemble ({merge_array.shape}) to {pipeline_dir}")

    distances: DistancesArray = compute_distances(
        normalized_merge_array=merge_array,
        method=distances_method,
    )
    np.savez_compressed(pipeline_dir / f"distances_{distances_method}.npz", distances=distances)
    logger.info(f"Distances ({distances.shape}) via {distances_method}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    plot_dists_distribution(
        distances=distances, mode="points", label=f"{distances_method} distances", ax=ax
    )
    ax.set_title(f"Stability — distance distribution ({distances_method})")
    plots_dir = pipeline_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig_path = plots_dir / f"distances_{distances_method}.png"
    fig.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Stability plot → {fig_path}")


def get_command(
    ensemble_id: str,
    clustering_run_ids: list[str],
    distances_method: DistancesMethod,
) -> str:
    return shlex.join(
        [
            "python",
            "-m",
            "param_decomp_lab.clustering.scripts.calc_distances",
            "--ensemble-id",
            ensemble_id,
            "--clustering-run-ids",
            ",".join(clustering_run_ids),
            "--distances-method",
            distances_method,
        ]
    )


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-id", type=str, required=True)
    parser.add_argument(
        "--clustering-run-ids",
        type=str,
        required=True,
        help="comma-separated clustering run ids",
    )
    parser.add_argument(
        "--distances-method",
        choices=DistancesMethod.__args__,
        default="perm_invariant_hamming",
    )
    args = parser.parse_args()
    calc_distances(
        ensemble_id=args.ensemble_id,
        clustering_run_ids=args.clustering_run_ids.split(","),
        distances_method=args.distances_method,
    )


if __name__ == "__main__":
    cli()
