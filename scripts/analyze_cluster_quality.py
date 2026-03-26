"""Analyze cluster quality for a single clustering run at a given iteration.

Usage:
    python scripts/analyze_cluster_quality.py \
        --run_id c-e8fb48bb --iter 7999 --alpha 2.0 --decay 0.8 \
        --snapshot_path /path/to/ch-XXXXX \
        --output_dir /path/to/output
"""

import json
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import fire
import numpy as np
from scipy import sparse


def analyze(
    run_id: str,
    iter: int,
    alpha: float,
    decay: float,
    snapshot_path: str,
    output_dir: str,
    clustering_base: str = "/mnt/polished-lake/artifacts/mechanisms/spd/clustering/runs",
) -> None:
    snapshot = Path(snapshot_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load snapshot
    matrix_csc = sparse.load_npz(snapshot / "memberships.npz").tocsc()
    metadata = json.loads((snapshot / "metadata.json").read_text())
    labels = metadata["labels"]
    n_samples = metadata["n_samples"]
    label_to_idx = {l: i for i, l in enumerate(labels)}
    matrix_int = matrix_csc.astype(np.int32)
    firing_counts = np.array(matrix_int.sum(axis=0)).ravel()

    # Read history
    history_path = Path(clustering_base) / run_id / "history.zip"
    with ZipFile(history_path) as zf:
        meta = json.loads(zf.read("metadata.json"))
        group_idxs_all = np.load(BytesIO(zf.read("merge.group_idxs.npy")))

    run_labels = meta["labels"]
    assignments = group_idxs_all[iter]

    # Build cluster mapping (filter singletons)
    unique_groups, counts = np.unique(assignments, return_counts=True)
    singleton_groups = set(unique_groups[counts == 1].tolist())

    cluster_to_cols: dict[int, list[int]] = defaultdict(list)
    for comp_idx, gid in enumerate(assignments):
        gid = int(gid)
        if gid in singleton_groups:
            continue
        comp_label = run_labels[comp_idx]
        if comp_label in label_to_idx:
            cluster_to_cols[gid].append(label_to_idx[comp_label])

    multi_clusters = {k: v for k, v in cluster_to_cols.items() if len(v) >= 2}

    # Within-cluster Jaccard
    all_within: list[float] = []
    cluster_means: list[float] = []
    n_zero_clusters = 0

    for cid, cols in multi_clusters.items():
        cols_arr = np.array(cols)
        sub = matrix_int[:, cols_arr]
        coact = (sub.T @ sub).toarray()

        n = len(cols)
        jaccards: list[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                ca, cb, cab = coact[i, i], coact[j, j], coact[i, j]
                union = ca + cb - cab
                jaccards.append(cab / union if union > 0 else 0.0)

        if jaccards:
            all_within.extend(jaccards)
            m = float(np.mean(jaccards))
            cluster_means.append(m)
            if any(j == 0 for j in jaccards):
                n_zero_clusters += 1

    within = np.array(all_within) if all_within else np.array([0.0])
    cmeans = np.array(cluster_means) if cluster_means else np.array([0.0])

    # Between-cluster baseline
    all_cols_list, all_cids_list = [], []
    for cid, cols in multi_clusters.items():
        for c in cols:
            all_cols_list.append(c)
            all_cids_list.append(cid)
    all_cols_arr = np.array(all_cols_list)
    all_cids_arr = np.array(all_cids_list)

    rng = np.random.default_rng(42)
    n_sample = 5000
    idx_pairs = rng.integers(0, len(all_cols_arr), size=(n_sample * 3, 2))
    mask = all_cids_arr[idx_pairs[:, 0]] != all_cids_arr[idx_pairs[:, 1]]
    idx_pairs = idx_pairs[mask][:n_sample]

    unique_cols = np.unique(
        np.concatenate([all_cols_arr[idx_pairs[:, 0]], all_cols_arr[idx_pairs[:, 1]]])
    )
    col_remap = {int(c): i for i, c in enumerate(unique_cols)}
    sub = matrix_int[:, unique_cols]
    between_j: list[float] = []
    for ip in range(len(idx_pairs)):
        c1 = int(all_cols_arr[idx_pairs[ip, 0]])
        c2 = int(all_cols_arr[idx_pairs[ip, 1]])
        r1, r2 = col_remap[c1], col_remap[c2]
        ca, cb = firing_counts[c1], firing_counts[c2]
        cab = sub[:, r1].T @ sub[:, r2]
        cab = cab.toarray()[0, 0] if sparse.issparse(cab) else int(cab)
        union = ca + cb - cab
        between_j.append(cab / union if union > 0 else 0.0)
    between = np.array(between_j)

    n_singletons = int((counts == 1).sum())
    avg_size = float(np.mean([len(v) for v in multi_clusters.values()])) if multi_clusters else 0
    n_in_clusters = sum(len(v) for v in multi_clusters.values())

    result = {
        "run": run_id,
        "alpha": alpha,
        "decay": decay,
        "iter": iter,
        "n_clusters": len(multi_clusters),
        "n_singletons": n_singletons,
        "n_in_clusters": n_in_clusters,
        "avg_size": avg_size,
        "within_mean": float(within.mean()),
        "within_median": float(np.median(within)),
        "between_mean": float(between.mean()),
        "ratio": float(within.mean() / between.mean()) if between.mean() > 0 else 0,
        "frac_zero_pairs": float((within == 0).mean()),
        "frac_clusters_zero": n_zero_clusters / len(multi_clusters) if multi_clusters else 0,
        "frac_clusters_lt05": float((cmeans < 0.05).mean()),
        "frac_clusters_gt50": float((cmeans > 0.50).mean()),
        "cluster_mean_median": float(np.median(cmeans)),
    }

    out_file = out / f"{run_id}.json"
    out_file.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    fire.Fire(analyze)
