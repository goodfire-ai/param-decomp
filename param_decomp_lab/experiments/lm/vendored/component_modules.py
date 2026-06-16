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
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Literal, Protocol, override, runtime_checkable

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor, nn

from param_decomp.components import EmbeddingComponents, LinearComponents
from param_decomp.masks import ComponentsMaskInfo

CaptureMode = Literal["none", "input", "output"]


@runtime_checkable
class CapturableComponent(Protocol):
    """A decomposed leaf that stashes its capture on itself (see `_ComponentModule`).

    Both vendored families (the shared `_ComponentModule` used by GPT-2 and Llama's
    self-contained `ComponentLinear`) implement this; the capture helpers below thread it.
    """

    path: str
    capture_mode: CaptureMode
    captured: Tensor | None


@contextmanager
def capture_acts(components: Iterable[CapturableComponent], mode: Literal["input", "output"]):
    """Set `capture_mode` on every component leaf for the duration, then collect each leaf's
    stash into a `{path: tensor}` dict yielded as a single-element list (so the caller reads it
    after the `with` body runs the forward). Clears the stash + mode on exit.

    Module-instance capture (not a threaded output dict) is mandatory under FSDP2: a wrapped
    block rebuilds its forward args whenever a tensor arg requires grad, copying any threaded
    dict and dropping every site past the first decomposed block. See `_ComponentModule`.
    """
    leaves = list(components)
    collected: dict[str, Tensor] = {}
    out: list[dict[str, Tensor]] = [collected]
    for leaf in leaves:
        leaf.capture_mode = mode
    try:
        yield out
        for leaf in leaves:
            assert leaf.captured is not None, (
                f"capture mode '{mode}' produced no act at {leaf.path}"
            )
            collected[leaf.path] = leaf.captured
    finally:
        for leaf in leaves:
            leaf.capture_mode = "none"
            leaf.captured = None


class _ComponentModule(ABC, nn.Module):
    """Shared replace-and-route forward; subclasses supply the frozen-target path.

    Capture (`pre_weight_acts` / output-acts) is stored on the module rather than threaded
    out through a forward argument: an FSDP2-wrapped block reconstructs its forward args via
    `tree_unflatten` whenever a tensor arg requires grad, which would build a fresh copy of a
    threaded output dict and silently drop every site past the first decomposed block. A
    module-instance stash is invisible to that arg reconstruction, so it survives `fully_shard`.
    """

    capture_mode: CaptureMode
    captured: Tensor | None

    def __init__(self, components: LinearComponents | EmbeddingComponents):
        super().__init__()
        self.components = components
        # Submodule path (e.g. "h.0.attn.q_proj"), set when swapped into a model; used to look
        # up this module's mask from a path-keyed mask_infos dict during a threaded forward.
        self.path: str = ""
        self.capture_mode = "none"
        self.captured = None

    @abstractmethod
    def target_forward(self, x: Tensor) -> Tensor: ...

    @override
    def forward(
        self,
        x: Tensor,
        mask_info: ComponentsMaskInfo | None,
        component_acts_cache: dict[str, Float[Tensor, "... C"]] | None = None,
    ) -> Tensor:
        if self.capture_mode == "input":
            self.captured = x
        out = self._routed_forward(x, mask_info, component_acts_cache)
        if self.capture_mode == "output":
            self.captured = out
        return out

    def _routed_forward(
        self,
        x: Tensor,
        mask_info: ComponentsMaskInfo | None,
        component_acts_cache: dict[str, Float[Tensor, "... C"]] | None,
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
        # `isinstance(..., str)` (static, dynamo-specializable) rather than `== "all"`, which
        # on a tensor routing_mask is a non-Tensor torch op that breaks torch.compile.
        if isinstance(mask_info.routing_mask, str):
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

    def __init__(self, components: LinearComponents, target_weight: Float[Tensor, "d_out d_in"]):
        super().__init__(components)
        assert target_weight.shape == (components.d_out, components.d_in)
        self.register_buffer("target_weight", target_weight)
        self.register_buffer("bias", components.bias)

    @override
    def target_forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
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
