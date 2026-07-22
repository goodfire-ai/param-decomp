import math
import random
from typing import Any, Literal, Protocol

import numpy as np
from jaxtyping import Bool, Float, Int

from param_decomp_lab.clustering.types import ClusterCoactivationShaped, MergePair

MergePairSamplerKey = Literal["range", "mcmc", "exp_rank"]


class MergePairSamplerConfigurable(Protocol):
    def __call__(
        self,
        costs: ClusterCoactivationShaped,
        **kwargs: Any,
    ) -> MergePair: ...


class MergePairSampler(Protocol):
    def __call__(
        self,
        costs: ClusterCoactivationShaped,
    ) -> MergePair: ...


def range_sampler(
    costs: ClusterCoactivationShaped,
    threshold: float = 0.05,
    **kwargs: Any,
) -> MergePair:
    """Sample a merge pair using threshold-based range selection.

    Considers all pairs with costs below a threshold defined as a fraction
    of the range of non-diagonal costs, then randomly selects one.
    threshold=0 keeps only the minimum-cost pair; threshold=1 keeps all pairs.
    """
    assert not kwargs
    k_groups: int = costs.shape[0]
    assert costs.shape[1] == k_groups, "Cost matrix must be square"

    off_diagonal: Bool[np.ndarray, "k_groups k_groups"] = ~np.eye(k_groups, dtype=np.bool_)
    non_diag_costs: Float[np.ndarray, " k_groups_squared_minus_k"] = costs[off_diagonal]
    min_cost: float = float(non_diag_costs.min())
    max_cost: float = float(non_diag_costs.max())

    max_considered_cost: float = (max_cost - min_cost) * threshold + min_cost

    considered_idxs: Int[np.ndarray, "n_considered 2"] = np.stack(
        np.where(costs <= max_considered_cost), axis=1
    )
    considered_idxs = considered_idxs[considered_idxs[:, 0] != considered_idxs[:, 1]]

    selected_idx: int = random.randint(0, considered_idxs.shape[0] - 1)
    row, col = considered_idxs[selected_idx].tolist()
    return MergePair((row, col))


def mcmc_sampler(
    costs: ClusterCoactivationShaped,
    temperature: float = 1.0,
    **kwargs: Any,
) -> MergePair:
    """Sample a merge pair with probability proportional to exp(-cost/temperature).

    Higher temperature gives more uniform sampling.
    """
    assert not kwargs
    k_groups: int = costs.shape[0]
    assert costs.shape[1] == k_groups, "Cost matrix must be square"

    valid_mask: Bool[np.ndarray, "k_groups k_groups"] = ~np.eye(k_groups, dtype=np.bool_)

    costs_masked: ClusterCoactivationShaped = costs.copy()
    costs_masked[~valid_mask] = np.inf  # diagonal -> exp gives 0

    min_cost: float = float(costs_masked[valid_mask].min())
    probs: ClusterCoactivationShaped = np.exp((min_cost - costs_masked) / temperature) * valid_mask
    probs_flatten: Float[np.ndarray, " k_groups_squared"] = probs.flatten()
    probs_flatten = probs_flatten / probs_flatten.sum()

    idx: int = random.choices(range(probs_flatten.size), weights=probs_flatten.tolist())[0]
    row: int = idx // k_groups
    col: int = idx % k_groups

    return MergePair((row, col))


def exp_rank_sampler(
    costs: ClusterCoactivationShaped,
    decay: float = 0.2,
    **kwargs: Any,
) -> MergePair:
    """Sample a merge pair with probability exponentially decaying in rank.

    P(rank r) ∝ exp(-decay * r), where rank 0 is the lowest-cost pair.
    decay=0 is uniform, decay→∞ is greedy.
    """
    assert not kwargs
    k_groups = costs.shape[0]

    valid_mask = ~np.eye(k_groups, dtype=np.bool_)
    upper_mask = np.triu(valid_mask, k=1)
    upper_indices = np.stack(np.where(upper_mask), axis=1)
    upper_costs = costs[upper_mask]

    sorted_indices = np.argsort(upper_costs)

    # P(rank r) ∝ exp(-decay * r) → sample via inverse CDF
    n = len(sorted_indices)
    u = random.random()
    exp_neg_decay_n = math.exp(-decay * n)
    rank = int(-math.log(1 - u * (1 - exp_neg_decay_n)) / decay)
    rank = min(rank, n - 1)

    pair_idx = sorted_indices[rank]
    row = int(upper_indices[pair_idx, 0])
    col = int(upper_indices[pair_idx, 1])

    return MergePair((row, col))


MERGE_PAIR_SAMPLERS: dict[MergePairSamplerKey, MergePairSamplerConfigurable] = {
    "range": range_sampler,
    "mcmc": mcmc_sampler,
    "exp_rank": exp_rank_sampler,
}
