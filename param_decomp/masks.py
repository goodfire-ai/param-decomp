"""Runtime mask payloads and routing for stochastic parameter decomposition.

``SamplingType`` selects between continuous (``rand_like``) and binomial (Bernoulli)
stochastic sources used by the recon losses. ``SubsetRoutingType`` is the
discriminated-union of subset routing configs accepted by metrics that randomly
route a subset of modules per position.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, override

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from param_decomp.base_config import BaseConfig, Probability

WeightDeltaAndMask = tuple[Float[Tensor, "d_out d_in"], Float[Tensor, "..."]]
"""Pair of ``(weight_delta, delta_mask)``.

``weight_delta`` is the residual ``W_target - sum(components)``; ``delta_mask`` is the per-position
scalar that gates how much of the delta is applied on each position.
"""

RoutingMasks = dict[str, Bool[Tensor, "..."]] | Literal["all"]
"""Per-module boolean routing masks, or the sentinel ``"all"`` meaning route everywhere."""


@dataclass
class ComponentsMaskInfo:
    """Mask payload applied to a single component module during a forward pass.

    Attributes:
        component_mask: ``[..., C]`` per-component mask. When the position is routed to components,
            this selects which subcomponents are active.
        routing_mask: Which ``(batch,)`` or ``(batch, seq_len)`` positions route to components vs
            the target module. ``"all"`` routes every position to components.
        weight_delta_and_mask: Optional ``(delta_weight, delta_mask)`` for the residual weight
            delta component. ``None`` disables the delta component.
    """

    component_mask: Float[Tensor, "... C"]
    routing_mask: Bool[Tensor, "..."] | Literal["all"] = "all"
    weight_delta_and_mask: WeightDeltaAndMask | None = None


class UniformKSubsetRoutingConfig(BaseConfig):
    """Subset-routing config: route to a uniformly-sized random subset per position."""

    type: Literal["uniform_k_subset"] = "uniform_k_subset"


class StaticProbabilityRoutingConfig(BaseConfig):
    """Subset-routing config: each position independently routes with probability ``p``.

    Attributes:
        p: Per-position routing probability.
    """

    type: Literal["static_probability"] = "static_probability"
    p: Probability


# Discriminated union over the subset-routing configs (keyed by ``type``).
SubsetRoutingType = UniformKSubsetRoutingConfig | StaticProbabilityRoutingConfig


# ``"continuous"`` draws uniform [0, 1) sources; ``"binomial"`` draws Bernoulli sources.
SamplingType = Literal["continuous", "binomial"]


class Router(ABC):
    """Strategy that produces per-module routing masks for a given leading shape.

    Implementations decide which positions route to component modules versus the original
    target modules. Returning the sentinel ``"all"`` is a fast path meaning "route everywhere".
    """

    @abstractmethod
    def get_masks(self, module_names: list[str], mask_shape: tuple[int, ...]) -> RoutingMasks:
        """Return routing masks for ``module_names`` at the given leading shape."""
        pass


class UniformKSubsetRouter(Router):
    """For each position, sample ``k`` from ``[1, n_modules]`` and route to a random ``k``-subset.

    The number of modules routed at each position is independent and uniform; the chosen subset
    is a uniformly random ``k``-sized subset of ``module_names``.
    """

    def __init__(self, device: torch.device | str):
        self.device = device

    @override
    def get_masks(
        self, module_names: list[str], mask_shape: tuple[int, ...]
    ) -> dict[str, Bool[Tensor, "..."]]:
        return sample_uniform_k_subset_routing_masks(mask_shape, module_names, self.device)


class AllLayersRouter(Router):
    """Route every position to every module (returns the ``"all"`` sentinel)."""

    @override
    def get_masks(self, module_names: list[str], mask_shape: tuple[int, ...]) -> Literal["all"]:
        return "all"


class StaticProbabilityRouter(Router):
    """Route each position to each module independently with fixed probability ``p``."""

    def __init__(self, p: float, device: torch.device | str):
        self.p = p
        self.device = device

    @override
    def get_masks(
        self, module_names: list[str], mask_shape: tuple[int, ...]
    ) -> dict[str, Bool[Tensor, "..."]]:
        return {mod: torch.rand(*mask_shape, device=self.device) < self.p for mod in module_names}


class LayerRouter(Router):
    """Route every position to a single named layer only.

    The mask for ``layer_name`` is all-ones; masks for the other modules are all-zeros.
    """

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
    """Return a LongTensor of shape ``shape`` with random permutations along ``dim``.

    For example, with ``shape=(2, 3)`` and ``dim=1`` each row is a random permutation of
    ``[0, 1, 2]``.

    Args:
        shape: Shape of the tensor to create.
        dim: Dimension along which to make the permutations.
        device: Device to create the tensor on.
        generator: Generator to use for the random values.
    """

    noise = torch.rand(shape, device=device, generator=generator)
    return noise.argsort(dim=dim).argsort(dim=dim)


def sample_uniform_k_subset_routing_masks(
    mask_shape: tuple[int, ...],
    module_names: list[str],
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> dict[str, Bool[Tensor, "..."]]:
    """Sample routing masks where each position routes to a uniform-``k`` random subset.

    For each position, ``k`` is drawn independently and uniformly from
    ``[1, len(module_names)]``, then a ``k``-sized random subset of modules is chosen.

    Args:
        mask_shape: Shape of the routing masks, likely ``(batch,)`` or ``(batch, seq_len)``.
        module_names: Module names to route to.
        device: Device to place the masks on.
        generator: Optional generator for reproducibility.

    Returns:
        Mapping from module name to a boolean routing mask of shape ``mask_shape``.
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
    """Bundle component masks, routing masks, and weight deltas into ``ComponentsMaskInfo``s.

    All inputs must share the same set of module-name keys.

    Args:
        component_masks: Per-module ``[..., C]`` component masks.
        routing_masks: Per-module routing masks, or the ``"all"`` sentinel.
        weight_deltas_and_masks: Per-module ``(delta_weight, delta_mask)`` tuples. ``None``
            disables the weight-delta component.

    Returns:
        Mapping from module name to its ``ComponentsMaskInfo``.
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
