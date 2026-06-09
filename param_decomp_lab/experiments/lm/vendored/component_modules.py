"""In-tree component modules: pure, checkpointable replacements for target Linear / Embedding.

These re-home `ComponentModel._components_and_cache_hook`'s replace-and-route logic into a
module's own `forward`, so the masked forward is a pure function of `(x, mask_info)` with no
side-channel hooks — making it compatible with activation checkpointing, FSDP, and
torch.compile. The V/U math is delegated to core `param_decomp.components`; these classes only
add the frozen-target-weight path and the routing between component output and target output.

A `mask_info` of `None` means "no decomposition here" — the module behaves as the frozen
target. Threading `mask_info` as an argument (rather than reading module-level state) is what
lets `torch.utils.checkpoint` save and replay it across the recompute boundary.
"""

from abc import ABC, abstractmethod
from typing import override

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor, nn

from param_decomp.components import EmbeddingComponents, LinearComponents
from param_decomp.masks import ComponentsMaskInfo
from param_decomp_lab.experiments.lm.vendored.fp8_frozen import (
    _quantize_tensorwise,
    fp8_frozen_enabled,
    fp8_frozen_target_forward,
)


class _ComponentModule(ABC, nn.Module):
    """Shared replace-and-route forward; subclasses supply the frozen-target path."""

    def __init__(self, components: LinearComponents | EmbeddingComponents):
        super().__init__()
        self.components = components
        # Submodule path (e.g. "h.0.attn.q_proj"), set when swapped into a model; used to look
        # up this module's mask from a path-keyed mask_infos dict during a threaded forward.
        self.path: str = ""

    @abstractmethod
    def target_forward(self, x: Tensor) -> Tensor: ...

    @override
    def forward(
        self,
        x: Tensor,
        mask_info: ComponentsMaskInfo | None,
        component_acts_cache: dict[str, Float[Tensor, "... C"]] | None = None,
    ) -> Tensor:
        if mask_info is None:
            assert component_acts_cache is None, "component_acts_cache needs an active mask"
            return self.target_forward(x)

        components_out = self.components(
            x,
            mask=mask_info.component_mask,
            weight_delta_and_mask=mask_info.weight_delta_and_mask,
            component_acts_cache=component_acts_cache,
        )
        if mask_info.routing_mask == "all":
            return components_out
        return torch.where(
            mask_info.routing_mask[..., None], components_out, self.target_forward(x)
        )


class ComponentLinear(_ComponentModule):
    """Replacement for a target `nn.Linear` / Radford `Conv1D`.

    `target_weight` is in PD's `[d_out, d_in]` row-major convention (Conv1D weights must be
    transposed by the caller). The frozen bias is carried over from the target module.
    """

    target_weight: Float[Tensor, "d_out d_in"]
    bias: Float[Tensor, "... d_out"] | None
    # Registered (and read) only when `_fp8_frozen` — the pre-quantized frozen-target weight.
    target_weight_fp8: Float[Tensor, "d_out d_in"]
    target_weight_scale: Float[Tensor, "1 1"]

    def __init__(self, components: LinearComponents, target_weight: Float[Tensor, "d_out d_in"]):
        super().__init__(components)
        assert target_weight.shape == (components.d_out, components.d_in)
        self.register_buffer("target_weight", target_weight)
        self.register_buffer("bias", components.bias)
        self._fp8_frozen = False
        if fp8_frozen_enabled():
            wq, scale = _quantize_tensorwise(target_weight)
            self.register_buffer("target_weight_fp8", wq.contiguous())
            self.register_buffer("target_weight_scale", scale)
            self._fp8_frozen = True

    @override
    def target_forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        if self._fp8_frozen:
            return fp8_frozen_target_forward(
                x,
                self.target_weight_fp8,
                self.target_weight_scale,
                self.bias,
            )
        return F.linear(x, self.target_weight, self.bias)


class ComponentEmbedding(_ComponentModule):
    """Replacement for a target `nn.Embedding`."""

    target_weight: Float[Tensor, "vocab dim"]

    def __init__(self, components: EmbeddingComponents, target_weight: Float[Tensor, "vocab dim"]):
        super().__init__(components)
        assert target_weight.shape == (components.vocab_size, components.embedding_dim)
        self.register_buffer("target_weight", target_weight)

    @override
    def target_forward(self, x: Int[Tensor, "..."]) -> Float[Tensor, "... dim"]:
        return self.target_weight[x]
