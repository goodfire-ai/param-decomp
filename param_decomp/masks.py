"""Runtime mask payloads and routing for stochastic parameter decomposition."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, override

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from param_decomp.base_config import BaseConfig, Probability

WeightDeltaAndMask = tuple[Float[Tensor, "d_out d_in"], Float[Tensor, "..."]]
"""`(weight_delta, delta_mask)`.

`weight_delta` is `W_target - sum(components)`; `delta_mask` is the per-position scalar
gating how much of the delta is applied.
"""

RoutingMasks = dict[str, Bool[Tensor, "..."]] | Literal["all"]
"""Per-module boolean routing masks, or the sentinel `"all"` meaning route everywhere."""


@dataclass
class ComponentsMaskInfo:
    """Mask payload applied to a single component module during a forward pass.

    `component_mask` (`[..., C]`) selects which subcomponents are active where the
    position routes to components. `routing_mask` picks which positions route to
    components vs the target module (`"all"` routes every position). Optional
    `weight_delta_and_mask` adds the residual weight-delta component.
    """

    component_mask: Float[Tensor, "... C"]
    routing_mask: Bool[Tensor, "..."] | Literal["all"] = "all"
    weight_delta_and_mask: WeightDeltaAndMask | None = None


class UniformKSubsetRoutingConfig(BaseConfig):
    """Route each position to a uniformly-sized random subset."""

    type: Literal["uniform_k_subset"] = "uniform_k_subset"


class StaticProbabilityRoutingConfig(BaseConfig):
    """Each position independently routes to each module with probability `p`."""

    type: Literal["static_probability"] = "static_probability"
    p: Probability


# Discriminated union over the subset-routing configs (keyed by ``type``).
SubsetRoutingType = UniformKSubsetRoutingConfig | StaticProbabilityRoutingConfig


# ``"continuous"`` draws uniform [0, 1) sources; ``"binomial"`` draws Bernoulli sources.
SamplingType = Literal["continuous", "binomial"]


class Router(ABC):
    """Strategy that produces per-module routing masks for a given leading shape.

    Implementations decide which positions route to component modules vs. the original
    target modules. Returning `"all"` is a fast path meaning "route everywhere".
    """

    @abstractmethod
    def get_masks(self, module_names: list[str], mask_shape: tuple[int, ...]) -> RoutingMasks: ...


class UniformKSubsetRouter(Router):
    """For each position, sample `k` from `[1, n_modules]` and route to a random `k`-subset.

    The chosen `k`-subset of `module_names` is uniform.
    """

    def __init__(self, device: torch.device | str):
        self.device = device

    @override
    def get_masks(
        self, module_names: list[str], mask_shape: tuple[int, ...]
    ) -> dict[str, Bool[Tensor, "..."]]:
        return sample_uniform_k_subset_routing_masks(mask_shape, module_names, self.device)


class AllLayersRouter(Router):
    """Route every position to every module (returns the `"all"` sentinel)."""

    @override
    def get_masks(self, module_names: list[str], mask_shape: tuple[int, ...]) -> Literal["all"]:
        return "all"


class StaticProbabilityRouter(Router):
    """Route each position to each module independently with fixed probability `p`."""

    def __init__(self, p: float, device: torch.device | str):
        self.p = p
        self.device = device

    @override
    def get_masks(
        self, module_names: list[str], mask_shape: tuple[int, ...]
    ) -> dict[str, Bool[Tensor, "..."]]:
        return {mod: torch.rand(*mask_shape, device=self.device) < self.p for mod in module_names}


class LayerRouter(Router):
    """Route every position to a single named layer only (other modules' masks are all-zeros)."""

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
    """LongTensor of `shape` with random permutations along `dim`.

    Example: `shape=(2, 3), dim=1` gives two rows, each a random permutation of `[0, 1, 2]`.
    """

    noise = torch.rand(shape, device=device, generator=generator)
    return noise.argsort(dim=dim).argsort(dim=dim)


def sample_uniform_k_subset_routing_masks(
    mask_shape: tuple[int, ...],
    module_names: list[str],
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> dict[str, Bool[Tensor, "..."]]:
    """Routing masks where each position routes to a uniform-`k` random subset of modules.

    `k` is drawn from `[1, len(module_names)]`, then a `k`-sized random subset is chosen.
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
    """Bundle component masks, routing masks, and weight deltas into `ComponentsMaskInfo`s.

    All inputs must share the same set of module-name keys.
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
