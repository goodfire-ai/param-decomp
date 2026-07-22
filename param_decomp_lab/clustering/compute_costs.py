import math

import numpy as np
from jaxtyping import Float
from scipy import sparse

from param_decomp_lab.clustering.math.merge_matrix import GroupMerge
from param_decomp_lab.clustering.sample_membership import (
    CompressedMembership,
    count_group_overlaps_from_component_rows,
)
from param_decomp_lab.clustering.types import ClusterCoactivationShaped, MergePair


def compute_mdl_cost(
    acts: Float[np.ndarray, " k_groups"],
    merges: GroupMerge,
    alpha: float = 1.0,
) -> float:
    r"""Compute MDL costs for merge matrices

    $$
        MDL = \sum_{i \in \N_k} s_i ( \log(k) + \alpha r(P_i) )
    $$

    where:
     - $s_i$ activation of component $i$, $s_j$ activation of component $j$
     - $r(P_i)$ rank of component $i$, $r(P_j)$ rank of component $j$
     - $k$ is the total number of components
    """

    k_groups: int = acts.shape[0]
    assert k_groups == merges.k_groups, "Merges must match activation vector shape"

    return float((acts * (math.log2(k_groups) + alpha * merges.components_per_group)).sum())


def compute_merge_costs(
    coact: ClusterCoactivationShaped,
    merges: GroupMerge,
    alpha: float = 1.0,
) -> ClusterCoactivationShaped:
    r"""Compute MDL costs for merge matrices

    $$
        (s_\Sigma - s_i - s_j) log((c-1)/c)
        + s_{i,j} log(c-1) - s_i log(c) - s_j log(c)
        + alpha ( s_{i,j} r(P_{i,j}) - s_i r(P_i) - s_j r(P_j) )
    $$
    where:
     - $s_\Sigma$ average activation of all components
     - $s_i$ activation of component $i$, $s_j$ activation of component $j$
     - $s_{i,j}$ activation of the merged component $i,j$
     - $r(P_i)$ rank of component $i$, $r(P_j)$ rank of component $j$
     - $r(P_{i,j})$ rank of the merged component $i,j$
    """
    k_groups: int = coact.shape[0]
    assert coact.shape[1] == k_groups, "Coactivation matrix must be square"
    assert merges.k_groups == k_groups, "Merges must match coactivation matrix shape"

    ranks: Float[np.ndarray, " k_groups"] = merges.components_per_group.astype(np.float32)
    s_diag: Float[np.ndarray, " k_groups"] = np.diag(coact)
    term_si_rpi: Float[np.ndarray, " k_groups"] = s_diag * ranks
    rank_sum: ClusterCoactivationShaped = ranks.reshape(-1, 1) + ranks.reshape(1, -1)

    coact_OR: ClusterCoactivationShaped = s_diag.reshape(-1, 1) + s_diag.reshape(1, -1) - coact

    s_other: ClusterCoactivationShaped = (
        s_diag.sum() - s_diag.reshape(-1, 1) - s_diag.reshape(1, -1)
    ) * math.log2((k_groups - 1) / k_groups)

    bits_local: ClusterCoactivationShaped = (
        coact_OR * math.log2(k_groups - 1)
        - s_diag.reshape(-1, 1) * math.log2(k_groups)
        - s_diag.reshape(1, -1) * math.log2(k_groups)
    )

    penalty: ClusterCoactivationShaped = (
        coact_OR * rank_sum  # s_{i,j} r(P_{i,j})
        - term_si_rpi.reshape(-1, 1)  # s_i r(P_i)
        - term_si_rpi.reshape(1, -1)  # s_j r(P_j)
    )

    return s_other + bits_local + alpha * penalty


def recompute_coacts_merge_pair_memberships(
    coact: ClusterCoactivationShaped,
    merges: GroupMerge,
    merge_pair: MergePair,
    memberships: list[CompressedMembership],
    component_activity_csr: sparse.csr_matrix,
) -> tuple[
    GroupMerge,
    Float[np.ndarray, "k_groups-1 k_groups-1"],
    list[CompressedMembership],
]:
    """Recompute coactivations after a merge using compressed memberships."""
    k_groups: int = coact.shape[0]
    assert coact.shape[1] == k_groups, "Coactivation matrix must be square"
    assert len(memberships) == k_groups, "Memberships must match coactivation matrix shape"

    new_group_idx: int = min(merge_pair)
    remove_idx: int = max(merge_pair)
    merged_membership = memberships[merge_pair[0]].union(memberships[merge_pair[1]])

    merge_new: GroupMerge = merges.merge_groups(merge_pair[0], merge_pair[1])

    keep_mask = np.ones(coact.shape[0], dtype=np.bool_)
    keep_mask[remove_idx] = False
    coact_new: Float[np.ndarray, "k_groups-1 k_groups-1"] = coact[keep_mask][:, keep_mask].copy()

    merged_rows = merged_membership.to_sample_indices()
    coact_with_merge = count_group_overlaps_from_component_rows(
        merged_rows=merged_rows,
        component_activity_csr=component_activity_csr,
        group_idxs=merge_new.group_idxs,
        n_groups=merge_new.k_groups,
    ).astype(coact.dtype, copy=False)

    coact_new[new_group_idx, :] = coact_with_merge
    coact_new[:, new_group_idx] = coact_with_merge
    coact_new[new_group_idx, new_group_idx] = float(merged_membership.count())

    memberships_new = memberships.copy()
    memberships_new[new_group_idx] = merged_membership
    memberships_new.pop(remove_idx)

    return merge_new, coact_new, memberships_new
