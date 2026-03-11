"""Discover gradient connectivity between layers of a ComponentModel."""

from collections import defaultdict
from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor, nn

from spd.configs import SamplingType
from spd.models.component_model import ComponentModel, OutputWithCache
from spd.models.components import make_mask_infos
from spd.topology.topology import TransformerTopology
from spd.utils.general_utils import bf16_autocast


def get_sources_by_target(
    model: ComponentModel,
    topology: TransformerTopology,
    device: str,
    sampling: SamplingType,
) -> dict[str, list[str]]:
    """Find valid gradient connections grouped by target layer.

    Includes embedding as a source and unembed as a target, using the topology's
    actual module paths (not pseudo-names).

    Returns:
        Dict mapping out_layer -> list of in_layers that have gradient flow to it.
    """
    # Use a small dummy batch - we only need to trace gradient connections
    batch: Float[Tensor, "batch seq"] = torch.zeros(2, 3, dtype=torch.long, device=device)

    with torch.no_grad(), bf16_autocast():
        output_with_cache: OutputWithCache = model(batch, cache_type="input")

        ci = model.calc_causal_importances(
            pre_weight_acts=output_with_cache.cache,
            sampling=sampling,
            detach_inputs=False,
        )

    # Create masks so we can use all components
    mask_infos = make_mask_infos(
        component_masks={k: torch.ones_like(v) for k, v in ci.lower_leaky.items()},
        routing_masks="all",
    )

    embed_path = topology.path_schema.embedding_path
    unembed_path = topology.path_schema.unembed_path

    # Hook to capture embedding output with gradients
    embed_cache: dict[str, Tensor] = {}

    def embed_hook(
        _module: nn.Module, _args: tuple[Any, ...], _kwargs: dict[Any, Any], output: Tensor
    ) -> Any:
        output.requires_grad_(True)
        embed_cache[f"{embed_path}_post_detach"] = output
        return output

    embed_handle = topology.embedding_module.register_forward_hook(embed_hook, with_kwargs=True)

    with torch.enable_grad(), bf16_autocast():
        comp_output_with_cache: OutputWithCache = model(
            batch,
            mask_infos=mask_infos,
            cache_type="component_acts",
        )

    embed_handle.remove()

    cache = comp_output_with_cache.cache
    cache[f"{embed_path}_post_detach"] = embed_cache[f"{embed_path}_post_detach"]
    cache[f"{unembed_path}_pre_detach"] = comp_output_with_cache.output

    source_layers = [embed_path, *model.target_module_paths]  # Don't include "output" as source
    target_layers = [*model.target_module_paths, unembed_path]  # Don't include embed as target

    # Test all distinct pairs for gradient flow
    test_pairs = []
    for source_layer in source_layers:
        for target_layer in target_layers:
            if source_layer != target_layer:
                test_pairs.append((source_layer, target_layer))

    sources_by_target: dict[str, list[str]] = defaultdict(list)
    for source_layer, target_layer in test_pairs:
        out_pre_detach = cache[f"{target_layer}_pre_detach"]
        in_post_detach = cache[f"{source_layer}_post_detach"]
        out_value = out_pre_detach[0, 0, 0]
        grads = torch.autograd.grad(
            outputs=out_value,
            inputs=in_post_detach,
            retain_graph=True,
            allow_unused=True,
        )
        assert len(grads) == 1
        grad = grads[0]
        if grad is not None:  # pyright: ignore[reportUnnecessaryComparison]
            sources_by_target[target_layer].append(source_layer)
    return dict(sources_by_target)
