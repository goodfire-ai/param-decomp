"""Calculate distances between clustering runs in an ensemble.

Output structure:
    SPD_OUT_DIR/clustering/ensembles/{pipeline_run_id}/
        ├── pipeline_config.yaml              # Created by run_pipeline.py
        ├── ensemble_meta.json                # Ensemble metadata
        ├── ensemble_merge_array.npz          # Normalized merge array
        ├── distances_<distances_method>.npz  # Distance array for each method
        └── plots/
            └── distances_<distances_method>.png  # Distance distribution plot
"""

import argparse
import json
import multiprocessing

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from spd.clustering.consts import DistancesArray, DistancesMethod
from spd.clustering.math.merge_distances import compute_distances
from spd.clustering.merge_history import MergeHistory, MergeHistoryEnsemble
from spd.clustering.plotting.merge import plot_dists_distribution
from spd.clustering.scripts.run_merge import _run_dir
from spd.log import logger
from spd.settings import SPD_OUT_DIR

# Set spawn method for CUDA compatibility with multiprocessing
# Must be done before any CUDA operations
if torch.cuda.is_available():
    try:  # noqa: SIM105
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        # Already set, ignore
        pass


def main(pipeline_run_id: str, run_ids: list[str], distances_method: DistancesMethod) -> None:
    logger.info(f"Calculating distances for pipeline run: {pipeline_run_id}")
    assert run_ids, "No run IDs provided"
    logger.info(f"Loading {len(run_ids)} clustering runs")

    histories: list[MergeHistory] = []
    for run_id in run_ids:
        history_path = _run_dir(run_id) / "history.zip"
        assert history_path.exists(), f"History not found for run {run_id}: {history_path}"
        histories.append(MergeHistory.read(history_path))
        logger.info(f"Loaded history for run {run_id}")

    # Compute normalized ensemble
    ensemble: MergeHistoryEnsemble = MergeHistoryEnsemble(data=histories)
    merge_array, merge_meta = ensemble.normalized()

    # Get pipeline output directory
    pipeline_dir = SPD_OUT_DIR / "clustering" / "ensembles" / pipeline_run_id

    # Save ensemble metadata and merge array
    ensemble_meta_path = pipeline_dir / "ensemble_meta.json"
    ensemble_meta_path.write_text(json.dumps(merge_meta, indent=2))
    logger.info(f"Saved ensemble metadata to {ensemble_meta_path}")

    ensemble_array_path = pipeline_dir / "ensemble_merge_array.npz"
    np.savez_compressed(ensemble_array_path, merge_array=merge_array)
    logger.info(f"Saved ensemble merge array to {ensemble_array_path}")

    # Compute distances
    logger.info(f"Computing distances using method: {distances_method}")
    distances: DistancesArray = compute_distances(
        normalized_merge_array=merge_array,
        method=distances_method,
    )

    distances_path = pipeline_dir / f"distances_{distances_method}.npz"
    np.savez_compressed(distances_path, distances=distances)
    logger.info(f"Distances computed and saved: shape={distances.shape}, path={distances_path}")

    # Create and save distances distribution plot
    ax: Axes = plot_dists_distribution(
        distances=distances, mode="points", label=f"{distances_method} distances"
    )
    plt.title(f"Distance Distribution ({distances_method})")

    # Only add legend if there are labeled artists
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        plt.legend()

    plots_dir = pipeline_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig_path = plots_dir / f"distances_{distances_method}.png"
    plt.savefig(fig_path)
    plt.close()
    logger.info(f"Saved distances distribution plot to {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate distances between clustering runs")
    parser.add_argument("--pipeline-run-id", type=str, required=True)
    parser.add_argument("--run-ids", type=str, required=True, help="Comma-separated run IDs")
    parser.add_argument(
        "--distances-method",
        choices=DistancesMethod.__args__,
        default="perm_invariant_hamming",
    )
    args = parser.parse_args()
    main(
        pipeline_run_id=args.pipeline_run_id,
        run_ids=args.run_ids.split(","),
        distances_method=args.distances_method,
    )
