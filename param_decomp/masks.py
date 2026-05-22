from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, override

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from param_decomp.base_config import BaseConfig
from param_decomp.types import Probability

WeightDeltaAndMask = tuple[Float[Tensor, "d_out d_in"], Float[Tensor, "..."]]
RoutingMasks = dict[str, Bool[Tensor, "..."]] | Literal["all"]


@dataclass
class ComponentsMaskInfo:
    """Specifies the mask information that will be applied to a component module."""

    component_mask: Float[Tensor, "... C"]
    """when components are routed to, this specifies which subcomponents to use"""

    routing_mask: Bool[Tensor, "..."] | Literal["all"] = "all"
    """Which (batch,) or (batch, seq_len) positions to route to components vs target modules.
    If "all", all positions are routed to components."""

    weight_delta_and_mask: WeightDeltaAndMask | None = None


class UniformKSubsetRoutingConfig(BaseConfig):
    type: Literal["uniform_k_subset"] = "uniform_k_subset"


class StaticProbabilityRoutingConfig(BaseConfig):
    type: Literal["static_probability"] = "static_probability"
    p: Probability


SubsetRoutingType = UniformKSubsetRoutingConfig | StaticProbabilityRoutingConfig


SamplingType = Literal["continuous", "binomial"]


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


def interpolate_component_mask(
    ci: dict[str, Float[Tensor, "*batch_dims C"]],
    sources: dict[str, Float[Tensor, "*batch_dims C"]],
) -> dict[str, Float[Tensor, "*batch_dims C"]]:
    """Set mask values to ci + (1 - ci) * source."""
    component_masks: dict[str, Float[Tensor, "*batch_dims C"]] = {}
    for module_name in ci:
        source = sources[module_name]
        assert ci[module_name].shape[-1] == source.shape[-1]
        component_masks[module_name] = ci[module_name] + (1 - ci[module_name]) * source
    return component_masks


def make_mask_infos(
    component_masks: dict[str, Float[Tensor, "... C"]],
    routing_masks: RoutingMasks = "all",
    weight_deltas_and_masks: dict[str, WeightDeltaAndMask] | None = None,
) -> dict[str, ComponentsMaskInfo]:
    """Create ComponentsMaskInfo dict from dicts of component masks, and optionally routing masks,
    weight deltas, and weight delta masks.
    Keys of all dicts must be the same.

    Args:
        component_masks: Dict mapping module names to component masks. routing_masks: Dict mapping
        module names to routing masks. weight_deltas_and_masks: Dict mapping module names to tuples
        of weight deltas and masks for each module to be decomposed. Defaults to None (disable
        weight delta component) if not provided.
    Returns:
        Dict mapping module names to ComponentsMaskInfo objects.
    """
    if isinstance(routing_masks, dict):
        assert set(routing_masks) == set(component_masks)

    if weight_deltas_and_masks is not None:
        assert set(weight_deltas_and_masks) == set(component_masks)

    result: dict[str, ComponentsMaskInfo] = {}
    for name in component_masks:
        routing_mask = routing_masks[name] if isinstance(routing_masks, dict) else "all"

        weight_delta_and_mask = (
            weight_deltas_and_masks[name] if weight_deltas_and_masks is not None else None
        )

        result[name] = ComponentsMaskInfo(
            component_mask=component_masks[name],
            routing_mask=routing_mask,
            weight_delta_and_mask=weight_delta_and_mask,
        )

    return result


def calc_stochastic_component_mask_info(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    component_mask_sampling: SamplingType,
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    router: Router,
) -> dict[str, ComponentsMaskInfo]:
    ci_sample = next(iter(causal_importances.values()))
    leading_dims = ci_sample.shape[:-1]
    device = ci_sample.device
    dtype = ci_sample.dtype

    component_masks: dict[str, Float[Tensor, "... C"]] = {}
    for layer, ci in causal_importances.items():
        match component_mask_sampling:
            case "binomial":
                stochastic_source = torch.randint(0, 2, ci.shape, device=device).float()
            case "continuous":
                stochastic_source = torch.rand_like(ci)
        component_masks[layer] = ci + (1 - ci) * stochastic_source

    weight_deltas_and_masks: dict[str, WeightDeltaAndMask] | None = None
    if weight_deltas is not None:
        weight_deltas_and_masks = {}
        for layer in causal_importances:
            weight_deltas_and_masks[layer] = (
                weight_deltas[layer],
                torch.rand(leading_dims, device=device, dtype=dtype),
            )

    routing_masks = router.get_masks(
        module_names=list(causal_importances.keys()),
        mask_shape=leading_dims,
    )

    return make_mask_infos(
        component_masks=component_masks,
        weight_deltas_and_masks=weight_deltas_and_masks,
        routing_masks=routing_masks,
    )
