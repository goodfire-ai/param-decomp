"""Sampling and statistics utilities for harvest pipeline."""

import numpy as np
from jaxtyping import Bool, Float, Int


def sample_at_most_n_per_group(
    group_ids: Int[np.ndarray, " N"],
    max_per_group: int,
    rng: np.random.Generator,
) -> Bool[np.ndarray, " N"]:
    """Boolean keep-mask: randomly sample at most `max_per_group` elements per group.

    Vectorised: sort by `(group, random)`, compute within-group rank via the cummax
    trick, keep entries with rank `<= max_per_group`.
    """
    if len(group_ids) == 0:
        return np.zeros(0, dtype=np.bool_)

    # Assign a random number to each element, shuffle via sorting by random key, then stably sort
    # the shuffled indices by group id. This produces a random order within each group while
    # keeping all items of the same group contiguous. "sort_idx" is the final index mapping.
    rand = rng.random(len(group_ids))
    rand_order = np.argsort(rand)
    sort_idx = rand_order[np.argsort(group_ids[rand_order], kind="stable")]
    sorted_groups = group_ids[sort_idx]

    # Compute rank within each group using cummax trick:
    # - Mark where groups change
    # - Use cummax to propagate group start positions forward
    # - Rank = current_position - group_start + 1
    group_change = np.concatenate(
        [np.ones(1, dtype=np.int64), (sorted_groups[1:] != sorted_groups[:-1]).astype(np.int64)]
    )
    positions = np.arange(1, len(sorted_groups) + 1)
    group_starts = np.where(group_change.astype(np.bool_), positions, np.zeros_like(positions))
    group_starts_propagated = np.maximum.accumulate(group_starts)
    rank_within_group = positions - group_starts_propagated + 1

    keep_mask = np.zeros(len(group_ids), dtype=np.bool_)
    keep_mask[sort_idx[rank_within_group <= max_per_group]] = True

    return keep_mask


def compute_pmi(
    cooccurrence_counts: Float[np.ndarray, " V"],
    marginal_counts: Float[np.ndarray, " V"],
    target_count: float,
    total_count: int,
) -> Float[np.ndarray, " V"]:
    """Pointwise mutual information per item.

    `PMI(x, y) = log(count(x, y) * total / (count(x) * count(y)))`. Items with zero
    counts get `-inf`.
    """
    valid = (cooccurrence_counts > 0) & (marginal_counts > 0)

    # PMI = log(P(co) / (P(target) * P(item)))
    #     = log(cooccurrence * total / (target_count * marginal))
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(cooccurrence_counts * total_count / (target_count * marginal_counts + 1e-10))

    return np.where(valid, pmi, np.full_like(pmi, float("-inf")))


def top_k_pmi(
    cooccurrence_counts: Float[np.ndarray, " V"],
    marginal_counts: Float[np.ndarray, " V"],
    target_count: float,
    total_count: int,
    top_k: int,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Top-k and bottom-k items by PMI; returns `(top, bottom)` lists of `(index, pmi_value)`."""
    pmi = compute_pmi(cooccurrence_counts, marginal_counts, target_count, total_count)

    n_valid = int((pmi > float("-inf")).sum())
    k = min(top_k, n_valid)

    if k == 0:
        return [], []

    top_indices = np.argsort(pmi)[::-1][:k]
    bottom_indices = np.argsort(pmi)[:k]

    top_items = [(int(idx), float(pmi[idx])) for idx in top_indices if pmi[idx] > float("-inf")]
    bottom_items = [
        (int(idx), float(pmi[idx])) for idx in bottom_indices if pmi[idx] > float("-inf")
    ]

    return top_items, bottom_items
