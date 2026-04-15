"""Tests for harvest sampling utilities."""

import math

import torch

from spd.harvest.sampling import compute_pmi, sample_at_most_n_per_group, top_k_pmi


class TestSampleAtMostNPerGroup:
    def test_empty_input(self) -> None:
        group_ids = torch.tensor([], dtype=torch.long)
        mask = sample_at_most_n_per_group(group_ids, max_per_group=5)
        assert mask.shape == (0,)
        assert mask.dtype == torch.bool

    def test_all_kept_when_under_limit(self) -> None:
        # 3 elements per group, limit is 5 -> all should be kept
        group_ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        mask = sample_at_most_n_per_group(group_ids, max_per_group=5)
        assert mask.all()

    def test_exactly_n_kept_per_group(self) -> None:
        # 10 elements per group, limit is 3
        group_ids = torch.tensor([0] * 10 + [1] * 10 + [2] * 10)
        mask = sample_at_most_n_per_group(group_ids, max_per_group=3)

        # Check each group has exactly 3
        for group in [0, 1, 2]:
            group_mask = group_ids == group
            assert mask[group_mask].sum() == 3

    def test_mixed_group_sizes(self) -> None:
        # Group 0: 2 elements (under limit)
        # Group 1: 5 elements (at limit)
        # Group 2: 10 elements (over limit)
        group_ids = torch.tensor([0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        mask = sample_at_most_n_per_group(group_ids, max_per_group=5)

        assert mask[group_ids == 0].sum() == 2  # all kept
        assert mask[group_ids == 1].sum() == 5  # all kept
        assert mask[group_ids == 2].sum() == 5  # capped

    def test_single_element_groups(self) -> None:
        group_ids = torch.tensor([0, 1, 2, 3, 4])
        mask = sample_at_most_n_per_group(group_ids, max_per_group=3)
        assert mask.all()

    def test_single_group(self) -> None:
        group_ids = torch.tensor([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        mask = sample_at_most_n_per_group(group_ids, max_per_group=3)
        assert mask.sum() == 3

    def test_deterministic_with_generator(self) -> None:
        group_ids = torch.tensor([0] * 100 + [1] * 100)

        gen1 = torch.Generator().manual_seed(42)
        mask1 = sample_at_most_n_per_group(group_ids, max_per_group=5, generator=gen1)

        gen2 = torch.Generator().manual_seed(42)
        mask2 = sample_at_most_n_per_group(group_ids, max_per_group=5, generator=gen2)

        assert torch.equal(mask1, mask2)

    def test_different_seeds_give_different_results(self) -> None:
        group_ids = torch.tensor([0] * 100)

        gen1 = torch.Generator().manual_seed(42)
        mask1 = sample_at_most_n_per_group(group_ids, max_per_group=5, generator=gen1)

        gen2 = torch.Generator().manual_seed(123)
        mask2 = sample_at_most_n_per_group(group_ids, max_per_group=5, generator=gen2)

        # Same count, but different elements selected
        assert mask1.sum() == mask2.sum() == 5
        assert not torch.equal(mask1, mask2)

    def test_non_contiguous_group_ids(self) -> None:
        # Groups don't need to be contiguous in input
        group_ids = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        mask = sample_at_most_n_per_group(group_ids, max_per_group=2)

        assert mask[group_ids == 0].sum() == 2
        assert mask[group_ids == 1].sum() == 2

    def test_large_group_ids(self) -> None:
        # Group IDs don't need to be 0-indexed or contiguous
        group_ids = torch.tensor([100, 100, 100, 500, 500, 500, 999, 999, 999])
        mask = sample_at_most_n_per_group(group_ids, max_per_group=2)

        assert mask[group_ids == 100].sum() == 2
        assert mask[group_ids == 500].sum() == 2
        assert mask[group_ids == 999].sum() == 2

    def test_max_per_group_one(self) -> None:
        group_ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        mask = sample_at_most_n_per_group(group_ids, max_per_group=1)

        assert mask[group_ids == 0].sum() == 1
        assert mask[group_ids == 1].sum() == 1
        assert mask[group_ids == 2].sum() == 1

    def test_preserves_device(self) -> None:
        group_ids = torch.tensor([0, 0, 1, 1], device="cpu")
        mask = sample_at_most_n_per_group(group_ids, max_per_group=1)
        assert mask.device == group_ids.device


class TestComputePMI:
    def test_basic_pmi_calculation(self) -> None:
        # Token 0: appears 50 times total, co-occurs 25 times with target
        # Token 1: appears 100 times total, co-occurs 10 times with target
        # Target fires 50 times out of 1000 total
        cooccurrence = torch.tensor([25.0, 10.0])
        marginal = torch.tensor([50.0, 100.0])
        target_count = 50.0
        total_count = 1000

        pmi = compute_pmi(cooccurrence, marginal, target_count, total_count)

        # PMI(token0) = log(25 * 1000 / (50 * 50)) = log(10) ≈ 2.30
        # PMI(token1) = log(10 * 1000 / (50 * 100)) = log(2) ≈ 0.69
        assert math.isclose(pmi[0].item(), math.log(10), rel_tol=1e-5)
        assert math.isclose(pmi[1].item(), math.log(2), rel_tol=1e-5)

    def test_zero_cooccurrence_gives_neg_inf(self) -> None:
        cooccurrence = torch.tensor([0.0, 10.0])
        marginal = torch.tensor([50.0, 100.0])

        pmi = compute_pmi(cooccurrence, marginal, 50.0, 1000)

        assert pmi[0].item() == float("-inf")
        assert pmi[1].item() > float("-inf")

    def test_zero_marginal_gives_neg_inf(self) -> None:
        cooccurrence = torch.tensor([10.0, 10.0])
        marginal = torch.tensor([0.0, 100.0])

        pmi = compute_pmi(cooccurrence, marginal, 50.0, 1000)

        assert pmi[0].item() == float("-inf")
        assert pmi[1].item() > float("-inf")

    def test_negative_pmi_for_underrepresented(self) -> None:
        # Token appears 500 times but only co-occurs 5 times with target (50 firings)
        # Expected co-occurrence if independent: 500 * 50 / 1000 = 25
        # Actual: 5, so underrepresented -> negative PMI
        cooccurrence = torch.tensor([5.0])
        marginal = torch.tensor([500.0])

        pmi = compute_pmi(cooccurrence, marginal, 50.0, 1000)

        assert pmi[0].item() < 0


class TestTopKPMI:
    def test_returns_top_and_bottom(self) -> None:
        # Create tokens with varying PMI
        cooccurrence = torch.tensor([100.0, 10.0, 1.0, 50.0])
        marginal = torch.tensor([100.0, 100.0, 100.0, 100.0])

        top, bottom = top_k_pmi(cooccurrence, marginal, 100.0, 1000, top_k=2)

        assert len(top) == 2
        assert len(bottom) == 2

        # Top should have highest PMI (token 0 with 100/100 cooccurrence)
        assert top[0][0] == 0
        # Bottom should have lowest PMI (token 2 with 1/100 cooccurrence)
        assert bottom[0][0] == 2

    def test_top_k_larger_than_valid(self) -> None:
        # All items have non-zero cooccurrence so all PMIs are finite
        cooccurrence = torch.tensor([10.0, 20.0, 5.0])
        marginal = torch.tensor([100.0, 100.0, 100.0])

        top, bottom = top_k_pmi(cooccurrence, marginal, 50.0, 1000, top_k=10)

        # Only 3 valid items, so we get at most 3
        assert len(top) == 3
        assert len(bottom) == 3
        # Top should be sorted descending by PMI
        assert top[0][0] == 1  # highest cooccurrence -> highest PMI

    def test_all_zeros_returns_empty(self) -> None:
        cooccurrence = torch.tensor([0.0, 0.0, 0.0])
        marginal = torch.tensor([100.0, 100.0, 100.0])

        top, bottom = top_k_pmi(cooccurrence, marginal, 50.0, 1000, top_k=5)

        assert top == []
        assert bottom == []
