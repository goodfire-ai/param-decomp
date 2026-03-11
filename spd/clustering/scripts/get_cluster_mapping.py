"""Extract cluster mapping from a clustering run at a specific iteration.

Usage:
    python -m spd.clustering.scripts.get_cluster_mapping /path/to/clustering_run --iteration 299
    python -m spd.clustering.scripts.get_cluster_mapping /path/to/clustering_run --iteration 299 --notes "some notes"

Output format:
    {
        "clustering_run_id": "cr-5f228e5f",
        "notes": "",
        "spd_run": "spd/goodfire/5cr21lbs",
        "clusters": {"h.0.mlp.down_proj:1": 0, "h.0.mlp.down_proj:2": null, ...}
    }

    Note: Singleton clusters (clusters with only one member) have null values.
"""

import json
import sys
from pathlib import Path

import fire
import numpy as np

from spd.clustering.merge_history import MergeHistory
from spd.utils.wandb_utils import parse_wandb_run_path


def get_cluster_mapping(
    run_dir: Path,
    iteration: int,
) -> tuple[dict[str, int | None], list[str]]:
    """Get mapping from component labels to cluster indices at a specific iteration.

    Args:
        run_dir: Path to clustering run directory containing history.zip
        iteration: Iteration index to extract clusters from

    Returns:
        Tuple of (mapping, labels) where mapping maps component label
        (e.g. "h.0.mlp.down_proj:42") to cluster index, or None for
        singleton clusters (clusters with only one member).
    """
    history_path = run_dir / "history.zip"
    assert history_path.exists(), f"History not found: {history_path}"

    history = MergeHistory.read(history_path)

    assert 0 <= iteration < history.n_iters_current, (
        f"iteration {iteration} out of bounds [0, {history.n_iters_current})"
    )

    merge = history.merges[iteration]
    assignments = merge.group_idxs.numpy()
    labels = list(history.labels)

    # Count members per cluster to identify singletons
    unique_ids, counts = np.unique(assignments, return_counts=True)
    singleton_clusters = set(unique_ids[counts == 1].tolist())

    mapping = {
        label: None if int(cluster_id) in singleton_clusters else int(cluster_id)
        for label, cluster_id in zip(labels, assignments, strict=True)
    }

    return mapping, labels


def get_spd_run_path(run_dir: Path) -> str:
    """Extract the SPD run path from the clustering run config.

    Returns:
        Formatted path like "spd/goodfire/5cr21lbs"
    """
    config_path = run_dir / "clustering_run_config.json"
    assert config_path.exists(), f"Clustering run config not found: {config_path}"

    with open(config_path) as f:
        config = json.load(f)

    model_path = config["model_path"]
    entity, project, run_id = parse_wandb_run_path(model_path)

    return f"{entity}/{project}/{run_id}"


def main(
    run_dir: str,
    iteration: int,
    notes: str = "",
    output: str | None = None,
) -> None:
    """Extract cluster mapping with metadata and output as JSON.

    Args:
        run_dir: Path to clustering run directory (containing history.zip)
        iteration: Iteration index to extract clusters from
        notes: Optional notes to include in the output
        output: Optional output file path. If not provided, writes to
            {run_dir}/cluster_mapping.json
    """
    run_path = Path(run_dir)

    clusters, _ = get_cluster_mapping(run_dir=run_path, iteration=iteration)

    clustering_run_id = run_path.name
    spd_run = get_spd_run_path(run_path)

    result = {
        "clustering_run_id": clustering_run_id,
        "notes": notes,
        "spd_run": spd_run,
        "iteration": iteration,
        "clusters": clusters,
    }

    json_str = json.dumps(result, indent=2)

    out_path = run_path / "cluster_mapping.json" if output is None else Path(output)

    out_path.write_text(json_str)
    print(f"Wrote mapping ({len(clusters)} components) to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    fire.Fire(main)
