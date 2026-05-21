"""Sampling and statistics utilities for harvest pipeline."""

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def sample_at_most_n_per_group(
    group_ids: Int[Tensor, " N"],
    max_per_group: int,
    generator: torch.Generator | None = None,
) -> Bool[Tensor, " N"]:
    """Randomly sample at most N elements per group.

    Uses a vectorized algorithm: sort by (group, random), compute rank within
    each group via cummax trick, keep elements with rank <= max_per_group.

    Args:
        group_ids: Integer tensor assigning each element to a group.
        max_per_group: Maximum elements to keep per group.
        generator: Optional random generator for reproducibility.

    Returns:
        Boolean mask indicating which elements to keep.
    """
    if len(group_ids) == 0:
        return torch.zeros(0, dtype=torch.bool, device=group_ids.device)

    device = group_ids.device

    # Assign a random number to each element, shuffle via sorting by random key, then stably sort
    # the shuffled indices by group id. This produces a random order within each group while
    # keeping all items of the same group contiguous. "sort_idx" is the final index mapping.
    rand = torch.rand(len(group_ids), device=device, generator=generator)
    rand_order = torch.argsort(rand)
    sort_idx = rand_order[torch.argsort(group_ids[rand_order], stable=True)]
    sorted_groups = group_ids[sort_idx]

    # Compute rank within each group using cummax trick:
    # - Mark where groups change
    # - Use cummax to propagate group start positions forward
    # - Rank = current_position - group_start + 1
    group_change = torch.cat(
        [
            torch.ones(1, device=device, dtype=torch.long),
            (sorted_groups[1:] != sorted_groups[:-1]).long(),
        ]
    )
    positions = torch.arange(1, len(sorted_groups) + 1, device=device)
    group_starts = torch.where(group_change.bool(), positions, torch.zeros_like(positions))
    group_starts_propagated = torch.cummax(group_starts, dim=0)[0]
    rank_within_group = positions - group_starts_propagated + 1

    # Map back to original indices
    keep_mask = torch.zeros(len(group_ids), dtype=torch.bool, device=device)
    keep_mask[sort_idx[rank_within_group <= max_per_group]] = True

    return keep_mask


def compute_pmi(
    cooccurrence_counts: Float[Tensor, " V"],
    marginal_counts: Float[Tensor, " V"],
    target_count: float,
    total_count: int,
) -> Float[Tensor, " V"]:
    """Compute pointwise mutual information (PMI) for each item.

    PMI(x, y) = log(P(x, y) / (P(x) * P(y)))
              = log(count(x, y) * total / (count(x) * count(y)))

    Args:
        cooccurrence_counts: Count (or mass) of each item co-occurring with target.
        marginal_counts: Total count (or mass) of each item across all targets.
        target_count: Total count of the target (e.g., component firings).
        total_count: Total number of observations.

    Returns:
        PMI values for each item. Items with zero counts get -inf.
    """
    valid = (cooccurrence_counts > 0) & (marginal_counts > 0)

    # PMI = log(P(co) / (P(target) * P(item)))
    #     = log(cooccurrence * total / (target_count * marginal))
    pmi = torch.log(cooccurrence_counts * total_count / (target_count * marginal_counts + 1e-10))

    return torch.where(valid, pmi, torch.full_like(pmi, float("-inf")))


def top_k_pmi(
    cooccurrence_counts: Float[Tensor, " V"],
    marginal_counts: Float[Tensor, " V"],
    target_count: float,
    total_count: int,
    top_k: int,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Compute top-k and bottom-k items by PMI.

    Args:
        cooccurrence_counts: Count (or mass) of each item co-occurring with target.
        marginal_counts: Total count (or mass) of each item across all targets.
        target_count: Total count of the target.
        total_count: Total number of observations.
        top_k: Number of top/bottom items to return.

    Returns:
        Tuple of (top_k_items, bottom_k_items), each a list of (index, pmi_value).
    """
    pmi = compute_pmi(cooccurrence_counts, marginal_counts, target_count, total_count)

    n_valid = int((pmi > float("-inf")).sum())
    k = min(top_k, n_valid)

    if k == 0:
        return [], []

    top = torch.topk(pmi, k, largest=True)
    bottom = torch.topk(pmi, k, largest=False)

    top_items = [
        (int(idx), float(val))
        for idx, val in zip(top.indices.tolist(), top.values.tolist(), strict=True)
        if val > float("-inf")
    ]
    bottom_items = [
        (int(idx), float(val))
        for idx, val in zip(bottom.indices.tolist(), bottom.values.tolist(), strict=True)
        if val > float("-inf")
    ]

    return top_items, bottom_items
