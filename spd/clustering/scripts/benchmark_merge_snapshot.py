"""Benchmark exact merge performance from a cached membership snapshot."""

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from spd.clustering.clustering_run_config import ClusteringRunConfig
from spd.clustering.membership_snapshot import load_membership_snapshot
from spd.clustering.merge import merge_iteration_memberships
from spd.utils.general_utils import replace_pydantic_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--profile-overlap", action="store_true")
    args = parser.parse_args()

    snapshot = load_membership_snapshot(args.snapshot_dir)
    memberships = snapshot.to_memberships()

    if args.profile_overlap:
        row_csr: Any = snapshot.to_csr()
        merged = memberships[0].union(memberships[1])
        merged_indices = merged.to_sample_indices().astype(np.int64, copy=False)

        st = time.time()
        current = np.array([merged.intersection_count(m) for m in memberships], dtype=np.int64)
        current_s = time.time() - st

        st = time.time()
        row_sum = np.asarray(row_csr[merged_indices].sum(axis=0)).ravel().astype(np.int64)
        row_sum_s = time.time() - st

        assert np.array_equal(current, row_sum)
        print(
            {
                "phase": "overlap_profile",
                "merged_size": int(merged_indices.size),
                "current_s": round(current_s, 4),
                "row_sum_s": round(row_sum_s, 4),
                "speedup": round(current_s / row_sum_s, 2),
            }
        )

    run_config = ClusteringRunConfig.from_file(args.config)
    merge_config = replace_pydantic_model(run_config.merge_config, {"iters": args.iters})

    st = time.time()
    history = merge_iteration_memberships(
        merge_config=merge_config,
        memberships=memberships,
        n_samples=snapshot.n_samples,
        component_labels=snapshot.labels,
        log_callback=None,
    )
    elapsed = time.time() - st
    print(
        {
            "phase": "merge_benchmark",
            "snapshot_dir": str(args.snapshot_dir),
            "n_samples": snapshot.n_samples,
            "n_components": snapshot.n_components,
            "iters": history.n_iters_current,
            "elapsed_s": round(elapsed, 2),
            "sec_per_iter": round(elapsed / history.n_iters_current, 4)
            if history.n_iters_current
            else None,
        }
    )


if __name__ == "__main__":
    main()
