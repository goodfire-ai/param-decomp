"""Create cluster mapping JSON from clustering run.

The most recently finished clustering run with accessible merge history is:
- c-d8b3750d (2026-03-18, ensemble e-9c716814, alpha=10, n_iters=9974)

Note: 4 more recent runs from 2026-03-23 exist (c-9ef2f07f etc.) but they don't
have merge_history artifacts in WandB and disk is inaccessible, so we use c-d8b3750d.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["WANDB_QUIET"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import wandb

from spd.clustering.merge_history import MergeHistory

RUN_ID = "c-d8b3750d"
PROJECT = "goodfire/spd"
SPD_RUN = "goodfire/spd/s-55ea3f9b"
ENSEMBLE_ID = "e-9c716814"


def get_cluster_mapping_from_history(
    history: MergeHistory,
    n_iterations: int,
) -> dict[str, int | None]:
    """Extract cluster mapping at a specific iteration.

    Args:
        history: MergeHistory from a clustering run
        n_iterations: Number of iterations to use (index into history)

    Returns:
        Mapping from component label to cluster index, None for singletons
    """
    assert 0 <= n_iterations < history.n_iters_current, (
        f"n_iterations {n_iterations} out of bounds [0, {history.n_iters_current})"
    )

    merge = history.merges[n_iterations]
    assignments = merge.group_idxs.cpu().numpy()

    cluster_ids, counts = np.unique(assignments, return_counts=True)
    singleton_clusters = set(cluster_ids[counts == 1])

    return {
        label: None if int(cluster_id) in singleton_clusters else int(cluster_id)
        for label, cluster_id in zip(history.labels, assignments, strict=True)
    }


def main() -> None:
    api = wandb.Api(timeout=60)
    run = api.run(f"{PROJECT}/{RUN_ID}")

    print(f"Run: {RUN_ID}")
    print(f"State: {run.state}")
    print(f"Steps completed: {run.summary.get('_step', 'unknown')}")
    print(f"Final k_groups: {run.summary.get('k_groups', 'unknown')}")
    print(
        f"Config: alpha={run.config.get('merge_config', {}).get('alpha')}, n_tokens={run.config.get('n_tokens')}"
    )

    # Download merge_history artifact
    print("\nDownloading merge_history artifact...")
    artifacts = list(run.logged_artifacts())
    merge_history_artifact = next((a for a in artifacts if a.type == "merge_history"), None)
    assert merge_history_artifact is not None, f"No merge_history artifact for run {RUN_ID}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_dir = merge_history_artifact.download(root=tmp_dir)
        history_files = list(Path(artifact_dir).glob("*.zip"))
        assert len(history_files) == 1, f"Expected 1 history file, got {len(history_files)}"
        history_path = history_files[0]

        print(f"Loading merge history from {history_path}...")
        history = MergeHistory.read(history_path)

    print("\nMergeHistory loaded:")
    print(f"  n_iters_current: {history.n_iters_current}")
    print(f"  n_components: {history.c_components}")
    print(f"  final k_groups: {history.final_k_groups}")

    # Use final iteration
    n_iterations = history.n_iters_current - 1
    print(f"\nCreating cluster mapping at final iteration {n_iterations}...")

    clusters = get_cluster_mapping_from_history(history, n_iterations)

    non_null = sum(1 for v in clusters.values() if v is not None)
    null_count = sum(1 for v in clusters.values() if v is None)
    unique_clusters = len(set(v for v in clusters.values() if v is not None))

    print(f"  Total components: {len(clusters)}")
    print(f"  In multi-member clusters: {non_null}")
    print(f"  Singletons (null): {null_count}")
    print(f"  Unique non-singleton clusters: {unique_clusters}")

    result = {
        "run_id": RUN_ID,
        "ensemble_id": ENSEMBLE_ID,
        "notes": "Most recently finished clustering run with accessible merge history (2026-03-18, alpha=10, n_tokens=500K tokens, 9974 iterations)",
        "spd_run": SPD_RUN,
        "n_iterations": n_iterations,
        "alpha": run.config.get("merge_config", {}).get("alpha"),
        "n_tokens": run.config.get("n_tokens"),
        "clusters": clusters,
    }

    output_path = Path(__file__).parent / f"cluster_mapping_{RUN_ID}.json"
    output_path.write_text(json.dumps(result, indent=2))
    print(f"\nCluster mapping saved to {output_path}")


if __name__ == "__main__":
    main()
