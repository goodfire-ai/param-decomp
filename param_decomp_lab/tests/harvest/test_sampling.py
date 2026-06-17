"""Tests for harvest sampling utilities."""

import math

import numpy as np

from param_decomp_lab.harvest.sampling import (
    compute_pmi,
    sample_at_most_n_per_group,
    top_k_pmi,
)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


class TestSampleAtMostNPerGroup:
    def test_empty_input(self) -> None:
        group_ids = np.array([], dtype=np.int64)
        mask = sample_at_most_n_per_group(group_ids, max_per_group=5, rng=_rng())
        assert mask.shape == (0,)
        assert mask.dtype == np.bool_

    def test_all_kept_when_under_limit(self) -> None:
        # 3 elements per group, limit is 5 -> all should be kept
        group_ids = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        mask = sample_at_most_n_per_group(group_ids, max_per_group=5, rng=_rng())
        assert mask.all()

    def test_exactly_n_kept_per_group(self) -> None:
        # 10 elements per group, limit is 3
        group_ids = np.array([0] * 10 + [1] * 10 + [2] * 10)
        mask = sample_at_most_n_per_group(group_ids, max_per_group=3, rng=_rng())

        for group in [0, 1, 2]:
            group_mask = group_ids == group
            assert mask[group_mask].sum() == 3

    def test_mixed_group_sizes(self) -> None:
        # Group 0: 2 (under), group 1: 5 (at), group 2: 10 (over)
        group_ids = np.array([0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        mask = sample_at_most_n_per_group(group_ids, max_per_group=5, rng=_rng())

        assert mask[group_ids == 0].sum() == 2
        assert mask[group_ids == 1].sum() == 5
        assert mask[group_ids == 2].sum() == 5

    def test_single_element_groups(self) -> None:
        group_ids = np.array([0, 1, 2, 3, 4])
        mask = sample_at_most_n_per_group(group_ids, max_per_group=3, rng=_rng())
        assert mask.all()

    def test_single_group(self) -> None:
        group_ids = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        mask = sample_at_most_n_per_group(group_ids, max_per_group=3, rng=_rng())
        assert mask.sum() == 3

    def test_deterministic_with_seed(self) -> None:
        group_ids = np.array([0] * 100 + [1] * 100)

        mask1 = sample_at_most_n_per_group(group_ids, max_per_group=5, rng=_rng(42))
        mask2 = sample_at_most_n_per_group(group_ids, max_per_group=5, rng=_rng(42))

        np.testing.assert_array_equal(mask1, mask2)

    def test_different_seeds_give_different_results(self) -> None:
        group_ids = np.array([0] * 100)

        mask1 = sample_at_most_n_per_group(group_ids, max_per_group=5, rng=_rng(42))
        mask2 = sample_at_most_n_per_group(group_ids, max_per_group=5, rng=_rng(123))

        assert mask1.sum() == mask2.sum() == 5
        assert not np.array_equal(mask1, mask2)

    def test_non_contiguous_group_ids(self) -> None:
        group_ids = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        mask = sample_at_most_n_per_group(group_ids, max_per_group=2, rng=_rng())

        assert mask[group_ids == 0].sum() == 2
        assert mask[group_ids == 1].sum() == 2

    def test_large_group_ids(self) -> None:
        group_ids = np.array([100, 100, 100, 500, 500, 500, 999, 999, 999])
        mask = sample_at_most_n_per_group(group_ids, max_per_group=2, rng=_rng())

        assert mask[group_ids == 100].sum() == 2
        assert mask[group_ids == 500].sum() == 2
        assert mask[group_ids == 999].sum() == 2

    def test_max_per_group_one(self) -> None:
        group_ids = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        mask = sample_at_most_n_per_group(group_ids, max_per_group=1, rng=_rng())

        assert mask[group_ids == 0].sum() == 1
        assert mask[group_ids == 1].sum() == 1
        assert mask[group_ids == 2].sum() == 1


class TestComputePMI:
    def test_basic_pmi_calculation(self) -> None:
        # Token 0: 50 total, co-occurs 25; token 1: 100 total, co-occurs 10
        # Target fires 50 times out of 1000 total
        cooccurrence = np.array([25.0, 10.0])
        marginal = np.array([50.0, 100.0])
        target_count = 50.0
        total_count = 1000

        pmi = compute_pmi(cooccurrence, marginal, target_count, total_count)

        # PMI(token0) = log(25 * 1000 / (50 * 50)) = log(10)
        # PMI(token1) = log(10 * 1000 / (50 * 100)) = log(2)
        assert math.isclose(float(pmi[0]), math.log(10), rel_tol=1e-5)
        assert math.isclose(float(pmi[1]), math.log(2), rel_tol=1e-5)

    def test_zero_cooccurrence_gives_neg_inf(self) -> None:
        cooccurrence = np.array([0.0, 10.0])
        marginal = np.array([50.0, 100.0])

        pmi = compute_pmi(cooccurrence, marginal, 50.0, 1000)

        assert float(pmi[0]) == float("-inf")
        assert float(pmi[1]) > float("-inf")

    def test_zero_marginal_gives_neg_inf(self) -> None:
        cooccurrence = np.array([10.0, 10.0])
        marginal = np.array([0.0, 100.0])

        pmi = compute_pmi(cooccurrence, marginal, 50.0, 1000)

        assert float(pmi[0]) == float("-inf")
        assert float(pmi[1]) > float("-inf")

    def test_negative_pmi_for_underrepresented(self) -> None:
        # Token appears 500 times, co-occurs 5 with target (50 firings)
        # Expected if independent: 25; actual 5 -> negative PMI
        cooccurrence = np.array([5.0])
        marginal = np.array([500.0])

        pmi = compute_pmi(cooccurrence, marginal, 50.0, 1000)

        assert float(pmi[0]) < 0


class TestTopKPMI:
    def test_returns_top_and_bottom(self) -> None:
        cooccurrence = np.array([100.0, 10.0, 1.0, 50.0])
        marginal = np.array([100.0, 100.0, 100.0, 100.0])

        top, bottom = top_k_pmi(cooccurrence, marginal, 100.0, 1000, top_k=2)

        assert len(top) == 2
        assert len(bottom) == 2
        assert top[0][0] == 0
        assert bottom[0][0] == 2

    def test_top_k_larger_than_valid(self) -> None:
        cooccurrence = np.array([10.0, 20.0, 5.0])
        marginal = np.array([100.0, 100.0, 100.0])

        top, bottom = top_k_pmi(cooccurrence, marginal, 50.0, 1000, top_k=10)

        assert len(top) == 3
        assert len(bottom) == 3
        assert top[0][0] == 1

    def test_all_zeros_returns_empty(self) -> None:
        cooccurrence = np.array([0.0, 0.0, 0.0])
        marginal = np.array([100.0, 100.0, 100.0])

        top, bottom = top_k_pmi(cooccurrence, marginal, 50.0, 1000, top_k=5)

        assert top == []
        assert bottom == []
