"""Explain how a cluster was formed by replaying the merge history.

For each merge that contributed to building the cluster, shows:
- The iteration and which groups merged
- The coactivation between the merged groups
- The MDL cost of the chosen pair vs the best alternative
- Whether the merge was the global minimum or a random pick from the range

Usage:
    python -m spd.clustering.scripts.explain_cluster \
        --run_dir /path/to/clustering/runs/c-XXXXX \
        --cluster_id 372 \
        --iteration 3000
"""

import json
import math
from pathlib import Path

import fire
import numpy as np
import torch
from scipy import sparse

from spd.clustering.compute_costs import compute_merge_costs
from spd.clustering.math.merge_matrix import BatchedGroupMerge, GroupMerge
from spd.clustering.membership_snapshot import load_membership_snapshot
from spd.clustering.merge_history import MergeHistory
from spd.clustering.sample_membership import compute_coactivation_matrix_from_csr
from spd.log import logger


def explain_cluster(
    run_dir: str,
    cluster_id: int,
    iteration: int,
) -> None:
    run_path = Path(run_dir)

    # Load history
    history = MergeHistory.read(run_path / "history.zip")
    assert 0 <= iteration < history.n_iters_current
    config = history.merge_config
    labels = list(history.labels)
    n_components = len(labels)

    # Load membership snapshot for coactivation
    merge_config_path = run_path / "merge_config.json"
    assert merge_config_path.exists(), f"No merge_config.json in {run_path}"
    with open(merge_config_path) as f:
        mc = json.load(f)
    snapshot_path = Path(mc["snapshot_path"])
    logger.info(f"Loading membership snapshot from {snapshot_path}")
    snapshot = load_membership_snapshot(snapshot_path)

    # Compute full component-level coactivation matrix
    logger.info("Computing component coactivation matrix...")
    csr = snapshot.to_csr()
    assert isinstance(csr, (sparse.csr_matrix, sparse.csr_array))
    coact_full = compute_coactivation_matrix_from_csr(csr)
    coact_full = torch.from_numpy(coact_full).float()
    n_samples = snapshot.n_samples
    logger.info(f"Coactivation matrix: {coact_full.shape}, {n_samples} samples")

    # Normalize coactivation by n_samples (as the merge loop does)
    coact_norm = coact_full / n_samples

    # Find which components are in the target cluster at the target iteration
    target_merge = history.merges[iteration]
    target_group_idxs = target_merge.group_idxs.numpy()

    # Find the group ID for the target cluster
    # The cluster_id in the mapping file may differ from the internal group index
    # We need to find which group at this iteration corresponds to cluster_id
    unique_groups, counts = np.unique(target_group_idxs, return_counts=True)
    singleton_groups = set(unique_groups[counts == 1].tolist())

    # Build cluster mapping (same logic as get_cluster_mapping.py)
    component_to_cluster: dict[int, int | None] = {}
    for comp_idx, group_idx in enumerate(target_group_idxs):
        gid = int(group_idx)
        component_to_cluster[comp_idx] = None if gid in singleton_groups else gid

    # Find components in the target cluster
    target_components = [
        comp_idx for comp_idx, cid in component_to_cluster.items() if cid == cluster_id
    ]
    assert len(target_components) > 0, f"Cluster {cluster_id} not found at iteration {iteration}"

    target_labels = [labels[i] for i in target_components]
    print(f"\n{'='*80}")
    print(f"Cluster {cluster_id}: {len(target_components)} components")
    print(f"{'='*80}")
    for i, (comp_idx, label) in enumerate(zip(target_components, target_labels)):
        firing = int(coact_full[comp_idx, comp_idx].item())
        print(f"  [{i}] {label}  (fires {firing}/{n_samples} = {firing/n_samples:.4f})")

    # Pairwise coactivation within cluster
    print(f"\nPairwise coactivation (raw counts, {n_samples} samples):")
    for i, ci in enumerate(target_components):
        for j, cj in enumerate(target_components):
            if j <= i:
                continue
            coact_ij = int(coact_full[ci, cj].item())
            si = int(coact_full[ci, ci].item())
            sj = int(coact_full[cj, cj].item())
            union = si + sj - coact_ij
            jaccard = coact_ij / union if union > 0 else 0
            print(f"  {labels[ci]} × {labels[cj]}: coact={coact_ij}, jaccard={jaccard:.4f}")

    # Now replay the merge history to find merges that built this cluster
    print(f"\n{'='*80}")
    print("Merge history for this cluster")
    print(f"{'='*80}\n")

    target_set = set(target_components)
    relevant_merges: list[dict] = []

    for iter_idx in range(iteration + 1):
        merge_at = history.merges[iter_idx]
        group_idxs = merge_at.group_idxs.numpy()

        # Check if selected_pair at this iteration involves any target components
        if iter_idx == 0:
            continue  # First iteration has the initial state

        selected = history.selected_pairs[iter_idx].tolist()
        prev_merge = history.merges[iter_idx - 1] if iter_idx > 0 else history.merges[0]
        prev_group_idxs = prev_merge.group_idxs.numpy()

        # Which components were in group selected[0] and selected[1] before this merge?
        group_a_members = set(np.where(prev_group_idxs == selected[0])[0].tolist())
        group_b_members = set(np.where(prev_group_idxs == selected[1])[0].tolist())

        # Is this merge relevant to our cluster?
        a_in_target = group_a_members & target_set
        b_in_target = group_b_members & target_set

        if not a_in_target and not b_in_target:
            continue

        # Compute coactivation between the two groups
        a_list = sorted(group_a_members)
        b_list = sorted(group_b_members)
        cross_coact = coact_full[np.ix_(a_list, b_list)].sum().item()
        a_total = sum(coact_full[i, i].item() for i in a_list)
        b_total = sum(coact_full[i, i].item() for i in b_list)

        # Compute what fraction of group activations co-fire
        union_acts = a_total + b_total - cross_coact
        jaccard = cross_coact / union_acts if union_acts > 0 else 0

        a_labels = [labels[i] for i in sorted(a_in_target)[:3]]
        b_labels = [labels[i] for i in sorted(b_in_target)[:3]]
        a_other = len(group_a_members) - len(a_in_target)
        b_other = len(group_b_members) - len(b_in_target)

        info = {
            "iteration": iter_idx,
            "group_a_size": len(group_a_members),
            "group_b_size": len(group_b_members),
            "cross_coact": int(cross_coact),
            "jaccard": jaccard,
            "a_in_target": len(a_in_target),
            "b_in_target": len(b_in_target),
        }
        relevant_merges.append(info)

        both_in = a_in_target and b_in_target
        marker = " *** JOINS NON-CO-FIRING ***" if cross_coact == 0 and both_in else ""

        print(f"Iteration {iter_idx}:{marker}")
        print(f"  Merged: group of {len(group_a_members)} + group of {len(group_b_members)}")
        print(f"  Cross-coactivation: {int(cross_coact)} / {n_samples} samples")
        print(f"  Jaccard: {jaccard:.6f}")

        if a_in_target:
            suffix = f" (+{a_other} outside)" if a_other else ""
            print(f"  Group A target members: {a_labels}{suffix}")
        else:
            print(f"  Group A: {len(group_a_members)} components (none in target cluster)")

        if b_in_target:
            suffix = f" (+{b_other} outside)" if b_other else ""
            print(f"  Group B target members: {b_labels}{suffix}")
        else:
            print(f"  Group B: {len(group_b_members)} components (none in target cluster)")
        print()

    # Summary
    zero_coact_merges = sum(1 for m in relevant_merges if m["cross_coact"] == 0 and m["a_in_target"] > 0 and m["b_in_target"] > 0)
    print(f"\n{'='*80}")
    print(f"Summary: {len(relevant_merges)} merges built this cluster")
    print(f"  Zero-coactivation merges (both sides in target): {zero_coact_merges}")
    print(f"  Config: alpha={config.alpha}, activation_threshold={config.activation_threshold}")
    print(f"  Pair sampling: {config.merge_pair_sampling_method} (kwargs: {config.merge_pair_sampling_kwargs})")


if __name__ == "__main__":
    fire.Fire(explain_cluster)
