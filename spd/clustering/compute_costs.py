import math

import torch
from jaxtyping import Bool, Float
from scipy import sparse
from torch import Tensor

from spd.clustering.consts import ClusterCoactivationShaped, MergePair
from spd.clustering.math.merge_matrix import GroupMerge
from spd.clustering.sample_membership import (
    CompressedMembership,
    count_group_overlaps_from_component_rows,
)


def compute_mdl_cost(
    acts: Float[Tensor, " k_groups"],
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

    return (
        (acts * (math.log2(k_groups) + alpha * merges.components_per_group.to(device=acts.device)))
        .sum()
        .item()
    )


def compute_merge_costs(
    coact: ClusterCoactivationShaped,
    merges: GroupMerge,
    alpha: float = 1.0,
) -> ClusterCoactivationShaped:
    r"""Compute MDL costs for merge matrices

    $$
        F(P_i, P_j)
        = \alpha |s_i| r(P_i) + \alpha |s_j| r(P_j)
            - s_i s_j ( \alpha r(P_i) + \alpha r(P_j) + c )
        = \alpha (
            |s_i| r(P_i)
            + |s_j| r(P_j)
            - s_i s_j ( r(P_i) + r(P_j) + c/\alpha )
        )
    $$

    new version from nathu 2025-08-11 16:48

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

    device: torch.device = coact.device
    ranks: Float[Tensor, " k_groups"] = merges.components_per_group.to(device=device).float()
    s_diag: Float[Tensor, " k_groups"] = torch.diag(coact).to(device=device)
    # term_si_rpj: Float[Tensor, "k_groups k_groups"] = s_diag.view(-1, 1) * ranks.view(1, -1)
    # term_si_rpj: Float[Tensor, "k_groups k_groups"] = s_diag.view(-1, 1) * (ranks.view(1, -1) + 1/alpha)
    term_si_rpi: Float[Tensor, " k_groups"] = s_diag * ranks
    # dbg_auto(term_si_rpi)
    rank_sum: ClusterCoactivationShaped = ranks.view(-1, 1) + ranks.view(1, -1)
    # TODO: use dynamic rank computation
    # return alpha * (
    #     term_si_rpj  # |s_i| r(P_j)
    #     + term_si_rpj.T  # |s_j| r(P_i)
    #     - coact * ( # s_i s_j
    #         rank_sum  # r(P_i) + r(P_j)
    #         + (rank_cost(merges.k_groups) / alpha) # c / alpha
    #     )
    # )

    coact_OR: ClusterCoactivationShaped = s_diag.view(-1, 1) + s_diag.view(1, -1) - coact

    # reduce penalty for sending dictionary by 1
    # (s_\Sigma - s_i - s_j) log((c-1)/c)
    # delta of cost for sending index, in expectation
    # + s_{i,j} log(c-1) - s_i log(c) - s_j log(c)
    # delta of cost for sending ranks, in expectation
    # + alpha ( s_{i,j} r(P_{i,j}) - s_i r(P_i) - s_j r(P_j)

    s_other: ClusterCoactivationShaped = (
        s_diag.sum() - s_diag.view(-1, 1) - s_diag.view(1, -1)
    ) * math.log2((k_groups - 1) / k_groups)

    bits_local: ClusterCoactivationShaped = (
        coact_OR * math.log2(k_groups - 1)
        - s_diag.view(-1, 1) * math.log2(k_groups)
        - s_diag.view(1, -1) * math.log2(k_groups)
    )

    penalty: ClusterCoactivationShaped = (
        coact_OR * rank_sum  # s_{i,j} r(P_{i,j})
        - term_si_rpi.view(-1, 1)  # s_i r(P_i)
        - term_si_rpi.view(1, -1)  # s_j r(P_j)
    )

    output: ClusterCoactivationShaped = s_other + bits_local + alpha * penalty
    return output


def recompute_coacts_merge_pair_memberships(
    coact: ClusterCoactivationShaped,
    merges: GroupMerge,
    merge_pair: MergePair,
    memberships: list[CompressedMembership],
    component_activity_csr: sparse.csr_matrix,
) -> tuple[
    GroupMerge,
    Float[Tensor, "k_groups-1 k_groups-1"],
    list[CompressedMembership],
]:
    """Recompute coactivations after a merge using compressed memberships."""
    k_groups: int = coact.shape[0]
    assert coact.shape[1] == k_groups, "Coactivation matrix must be square"
    assert len(memberships) == k_groups, "Memberships must match coactivation matrix shape"

    new_group_idx: int = min(merge_pair)
    remove_idx: int = max(merge_pair)
    merged_membership = memberships[merge_pair[0]].union(memberships[merge_pair[1]])

    merge_new: GroupMerge = merges.merge_groups(
        merge_pair[0],
        merge_pair[1],
    )

    mask: Bool[Tensor, " k_groups"] = torch.ones(
        coact.shape[0], dtype=torch.bool, device=coact.device
    )
    mask[remove_idx] = False
    coact_new: Float[Tensor, "k_groups-1 k_groups-1"] = coact[mask, :][:, mask].clone()

    merged_rows = merged_membership.to_sample_indices()
    coact_with_merge_np = count_group_overlaps_from_component_rows(
        merged_rows=merged_rows,
        component_activity_csr=component_activity_csr,
        group_idxs=merge_new.group_idxs.cpu().numpy(),
        n_groups=merge_new.k_groups,
    )
    coact_with_merge = torch.tensor(
        coact_with_merge_np,
        dtype=coact.dtype,
        device=coact.device,
    )

    coact_new[new_group_idx, :] = coact_with_merge
    coact_new[:, new_group_idx] = coact_with_merge
    coact_new[new_group_idx, new_group_idx] = float(merged_membership.count())

    memberships_new = memberships.copy()
    memberships_new[new_group_idx] = merged_membership
    memberships_new.pop(remove_idx)

    return merge_new, coact_new, memberships_new
