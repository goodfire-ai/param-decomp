import pytest
import torch

from param_decomp.routing import rand_perm, sample_uniform_k_subset_routing_masks


class TestRandPerm:
    @pytest.mark.parametrize("shape", [(2, 3, 4), (100, 200, 134), (5, 6, 7)])
    @pytest.mark.parametrize("dim", [0, 1, 2])
    def test_creates_correct_shape(self, shape: tuple[int, ...], dim: int):
        assert rand_perm(shape, dim=dim).shape == shape

    def test_creates_permutations(self):
        # Given a random permutation of shape (100, 200), along dim 1
        shape = (100, 200)
        perm = rand_perm(shape, dim=1)

        # when we sort along dim 1
        sorted_perm = perm.sort(dim=1).values

        # then, we should get a simple arange for each row
        sorted_target = torch.arange(200).repeat(100, 1)
        assert torch.equal(sorted_perm, sorted_target)

    def test_creates_permutations_along_correct_indices(self):
        # Given a random permutation of shape (100, 200), along dim 1
        shape = (100, 200)
        perm = rand_perm(shape, dim=1)

        # when we sort along dim 0 (not dim 1)
        sorted_perm_wrong_axis = perm.sort(dim=0).values

        # then we should NOT get a simple arange for each row (technically it's possible, but
        # the probability is 1/(100!) which is ≈never)
        sorted_target_wrong_axis = torch.arange(100).repeat(200, 1)
        assert not torch.equal(sorted_perm_wrong_axis, sorted_target_wrong_axis)


class TestSampleUniformKSubsetRoutingMasks:
    def test_creates_correct_shape(self):
        mask_shape = (100, 200)
        modules = ["a", "b"]
        masks = sample_uniform_k_subset_routing_masks(mask_shape, modules)
        assert masks["a"].shape == (100, 200)
        assert masks["b"].shape == (100, 200)

    def test_creates_correct_distribution(self):
        S = 1000  # large number for statistical stability
        gen = torch.Generator().manual_seed(1)
        n_modules = 4
        mask_shape = (S,)
        modules = [str(i) for i in range(n_modules)]
        masks = sample_uniform_k_subset_routing_masks(mask_shape, modules, generator=gen)
        mask_matrix = torch.stack(list(masks.values()))
        assert mask_matrix.shape == (n_modules, S)
        position_wise_mask_sum = mask_matrix.sum(dim=0)
        hist = position_wise_mask_sum.float().histogram(bins=n_modules).hist

        # roughly 1/4 of the positions should have 1 module routed to, # 1/4 of the positions should have 2
        # modules routed to, etc.
        # this is a recorded output under the given seed. may need to rewrite test if the implementation changes.
        assert torch.equal(hist, torch.tensor([265, 241, 257, 237]).float())

    @pytest.mark.parametrize("mask_shape", [(16,), (4, 8)])
    @pytest.mark.parametrize("n_modules", [1, 2, 5])
    def test_n_routed(self, mask_shape: tuple[int, ...], n_modules: int):
        # This test relies on the implementation details of the function, so it may need to be
        # rewritten if the implementation changes.
        # Specifically, it relies on torch.randint being the first thing called with the generator,
        # and that being the source of the k noise.

        modules = [f"m{i}" for i in range(n_modules)]
        seed = 42

        # Call the function with a generator
        gen1 = torch.Generator().manual_seed(seed)
        masks1 = sample_uniform_k_subset_routing_masks(mask_shape, modules, generator=gen1)
        mask_block = torch.stack(list(masks1.values()))
        n_routed = mask_block.sum(dim=0)

        # call with an identical generator
        # We should get the same random k noise, resulting in the same number of modules routed to
        # for each position
        gen2 = torch.Generator().manual_seed(seed)
        expected_n_routed = torch.randint(
            low=1,
            high=len(modules) + 1,
            size=mask_shape,
            generator=gen2,
        )

        assert torch.equal(n_routed, expected_n_routed)
