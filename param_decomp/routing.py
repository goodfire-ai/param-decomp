from abc import ABC, abstractmethod
from typing import Literal, override

import torch
from jaxtyping import Bool, Int
from torch import Tensor

from param_decomp.configs import (
    StaticProbabilityRoutingConfig,
    SubsetRoutingType,
    UniformKSubsetRoutingConfig,
)
from param_decomp.models.components import RoutingMasks


class Router(ABC):
    @abstractmethod
    def get_masks(self, module_names: list[str], mask_shape: tuple[int, ...]) -> RoutingMasks:
        pass


class UniformKSubsetRouter(Router):
    """for each position, sample k from [1, n_modules], then route to components for k out of
    `n_modules` modules"""

    def __init__(self, device: torch.device | str):
        self.device = device

    @override
    def get_masks(
        self, module_names: list[str], mask_shape: tuple[int, ...]
    ) -> dict[str, Bool[Tensor, "..."]]:
        return sample_uniform_k_subset_routing_masks(mask_shape, module_names, self.device)


class AllLayersRouter(Router):
    @override
    def get_masks(self, module_names: list[str], mask_shape: tuple[int, ...]) -> Literal["all"]:
        return "all"


class StaticProbabilityRouter(Router):
    def __init__(self, p: float, device: torch.device | str):
        self.p = p
        self.device = device

    @override
    def get_masks(
        self, module_names: list[str], mask_shape: tuple[int, ...]
    ) -> dict[str, Bool[Tensor, "..."]]:
        """returns a { <layer>: [batch, seq] } dict of tensors, where each batch (batch_idx,
        seq_idx) is routed to with probability p"""
        return {mod: torch.rand(*mask_shape, device=self.device) < self.p for mod in module_names}


class LayerRouter(Router):
    def __init__(self, device: torch.device | str, layer_name: str):
        self.device = device
        self.layer_name = layer_name

    @override
    def get_masks(
        self, module_names: list[str], mask_shape: tuple[int, ...]
    ) -> dict[str, Bool[Tensor, "..."]]:
        out = {}
        for mod in module_names:
            f = torch.ones if mod == self.layer_name else torch.zeros
            out[mod] = f(*mask_shape, device=self.device, dtype=torch.bool)
        return out


def rand_perm(
    shape: tuple[int, ...],
    dim: int,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> Int[Tensor, "... k"]:
    """Create a LongTensor of shape `shape` containing random permutations along dimension `dim`.
    For example, if shape is (2, 3) and dim is 1, the returned tensor will be a 2x3 tensor with
    each row having a random permutation of [0, 1, 2].

    Args:
        shape: Shape of the tensor to create
        dim: Dimension along which to make the permutations
        device: Device to create the tensor on
        generator: Generator to use for the random values

    Returns:
        LongTensor of shape `shape` with randomly ordered permutation along dimension `dim`.
    """

    noise = torch.rand(shape, device=device, generator=generator)
    # turn values into ranks via double argsort trick. (for example: [0.8, 0.2, 0.3] -> [2, 0, 1])
    return noise.argsort(dim=dim).argsort(dim=dim)


def sample_uniform_k_subset_routing_masks(
    mask_shape: tuple[int, ...],
    module_names: list[str],
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> dict[str, Bool[Tensor, "..."]]:
    """Creates routing masks for each module such that the number of modules routed to for each
    position is independent and uniformly sampled from [1, len(module_names)]

    Achieves this by:
    - for each position, k is independent and uniformly sampled from [1, len(module_names)]
    - for each position, a k-sized random subset of modules are routed to

    Args:
        mask_shape: Shape of the routing masks, likely (batch,) or (batch, seq_len)
        module_names: List of module names to route to

    Returns:
        Dict mapping module names to routing masks of shape `mask_shape`.
    """
    k_modules_to_route: Int[Tensor, " ..."] = torch.randint(
        low=1,
        high=len(module_names) + 1,
        size=mask_shape,
        device=device,
        generator=generator,
    )

    perms: Int[Tensor, "k_modules ..."] = rand_perm(
        shape=(len(module_names), *mask_shape),
        dim=0,
        device=device,
        generator=generator,
    )

    return {mod: perms[i] < k_modules_to_route for i, mod in enumerate(module_names)}


def get_subset_router(routing: SubsetRoutingType, device: torch.device | str) -> Router:
    match routing:
        case UniformKSubsetRoutingConfig():
            return UniformKSubsetRouter(device=device)
        case StaticProbabilityRoutingConfig(p=p):
            return StaticProbabilityRouter(p=p, device=device)
