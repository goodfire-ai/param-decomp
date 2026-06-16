"""Componentization of the vendored Llama-3.1 target: `componentize_llama` swaps the
decomposition-target leaves for in-tree `ComponentLinear` and re-points the mlp / attn / block /
model forwards (via `__class__` reassignment) at variants that thread a path-keyed `mask_infos`
dict down to those leaves. Threading masks as a forward argument — not via hooks — is what makes
the masked forward checkpoint / FSDP / compile friendly.

Self-contained: `ComponentLinear` / `_proj` are vendored here, not shared with the gpt2 vendoring.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast, override

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from param_decomp.components import Components, EmbeddingComponents, LinearComponents
from param_decomp.masks import ComponentsMaskInfo
from param_decomp_lab.experiments.lm.vendored.component_modules import CaptureMode, capture_acts
from param_decomp_lab.experiments.lm.vendored.llama_3_1.model import (
    LlamaAttention,
    LlamaBlock,
    LlamaMLP,
    VendoredLlama,
)

# target_weight / bias buffers are register_buffer-initialized (pyright can't model that here).
# pyright: reportUninitializedInstanceVariable=false

MaskInfos = dict[str, ComponentsMaskInfo]
PreWeightActs = dict[str, Tensor]


class ComponentLinear(nn.Module):
    """In-tree, checkpointable replacement for a target `nn.Linear`: routes between the V/U
    component output and the frozen-target output as a pure function of `(x, mask_info)` (no
    side-channel hooks). `mask_info is None` → behave as the frozen target.

    Capture (`pre_weight_acts` / output-acts) is stashed on the module (`capture_mode` /
    `captured`), not threaded out through a forward arg: an FSDP2-wrapped block rebuilds its
    forward args via `tree_unflatten` whenever a tensor arg requires grad, which copies a
    threaded output dict and drops every site past the first decomposed block. See
    `component_modules._ComponentModule` for the shared rationale."""

    target_weight: Float[Tensor, "d_out d_in"]
    bias: Float[Tensor, "... d_out"] | None
    capture_mode: CaptureMode
    captured: Tensor | None

    def __init__(self, components: LinearComponents, target_weight: Float[Tensor, "d_out d_in"]):
        super().__init__()
        self.components = components
        self.path = ""  # submodule path, set at swap time; keys this leaf's mask in mask_infos
        self.capture_mode = "none"
        self.captured = None
        assert target_weight.shape == (components.d_out, components.d_in)
        self.register_buffer("target_weight", target_weight)
        self.register_buffer("bias", components.bias)

    def target_forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return F.linear(x, self.target_weight, self.bias)

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
        if isinstance(mask_info.routing_mask, str):  # "all" sentinel; isinstance is compile-safe
            return components_out
        return torch.where(
            mask_info.routing_mask[..., None], components_out, self.target_forward(x)
        )


def _proj(
    module: nn.Module,
    x: Tensor,
    mask_infos: MaskInfos | None,
    acts_out: dict[str, Tensor] | None = None,
) -> Tensor:
    """Apply a (possibly component-decomposed) leaf, routing its mask in by path. With `acts_out`
    set, a component leaf's input act is recorded there (compile-traceable pre-weight-acts return,
    no module stash). Plain leaves are applied unchanged."""
    if isinstance(module, ComponentLinear):
        if acts_out is not None:
            acts_out[module.path] = x
        mask_info = None if mask_infos is None else mask_infos.get(module.path)
        return module(x, mask_info)
    return module(x)


class ComponentLlamaMLP(LlamaMLP):
    @override
    def forward(
        self,
        x: Float[Tensor, "... dim"],
        mask_infos: MaskInfos | None = None,
        acts_out: dict[str, Tensor] | None = None,
    ) -> Float[Tensor, "... dim"]:
        gate = _proj(self.gate_proj, x, mask_infos, acts_out)
        up = _proj(self.up_proj, x, mask_infos, acts_out)
        return _proj(self.down_proj, F.silu(gate) * up, mask_infos, acts_out)


class ComponentLlamaAttention(LlamaAttention):
    @override
    def forward(
        self,
        x: Float[Tensor, "b t d"],
        mask_infos: MaskInfos | None = None,
        acts_out: dict[str, Tensor] | None = None,
    ) -> Float[Tensor, "b t d"]:
        q = _proj(self.q_proj, x, mask_infos, acts_out)
        k = _proj(self.k_proj, x, mask_infos, acts_out)
        v = _proj(self.v_proj, x, mask_infos, acts_out)
        return _proj(self.o_proj, self._attend(q, k, v), mask_infos, acts_out)


class ComponentLlamaBlock(LlamaBlock):
    @override
    def forward(
        self,
        x: Float[Tensor, "b t d"],
        mask_infos: MaskInfos | None = None,
        acts_out: dict[str, Tensor] | None = None,
    ) -> Float[Tensor, "b t d"]:
        attn: ComponentLlamaAttention = self.self_attn  # pyright: ignore[reportAssignmentType]
        mlp: ComponentLlamaMLP = self.mlp  # pyright: ignore[reportAssignmentType]
        x = x + attn(self.input_layernorm(x), mask_infos, acts_out)
        x = x + mlp(self.post_attention_layernorm(x), mask_infos, acts_out)
        return x


class ComponentLlama(VendoredLlama):
    """`VendoredLlama` whose forward threads a path-keyed `mask_infos` to in-tree components.

    Adopted via `__class__` reassignment by `componentize_llama` (no `__init__` of its own).
    `forward` returns logits, or the post-final-norm hidden state under `bypass_lm_head()`.
    """

    _bypass_lm_head: bool = False
    _cached_residual: tuple[Tensor, int] | None = None

    @override
    def forward(
        self,
        idx: Int[Tensor, "b t"],
        mask_infos: MaskInfos | None = None,
    ) -> Float[Tensor, "b t vocab"] | Float[Tensor, "b t d"]:
        if self._cached_residual is not None:
            residual, start = self._cached_residual
            return self.forward_from_residual(residual, start, mask_infos)
        _b, t = idx.size()
        assert t <= self.config.max_position_embeddings, (
            f"seq len {t} > max_position_embeddings {self.config.max_position_embeddings}"
        )
        return self.forward_from_residual(self.embed_tokens(idx), 0, mask_infos)

    @contextmanager
    def use_cached_residual(self, idx: Int[Tensor, "b t"]) -> Iterator[None]:
        """Within this context, `forward` runs the masked suffix from a clean residual entering
        `decomposition_start_layer` (computed once from `idx` here), skipping the frozen prefix —
        instead of embedding `idx` and running all blocks. Every forward inside (clean target,
        masked recon, the PPGD inner loop, CI harvest) reuses the one cached residual. Output is
        identical to the full forward (the prefix is frozen + component-free).
        `PD_DISABLE_RESIDUAL_START=1` makes it a no-op (full forward) — A/B + scale escape hatch."""
        if os.environ.get("PD_DISABLE_RESIDUAL_START", "").strip() in ("1", "true", "yes"):
            yield
            return
        assert self._cached_residual is None, "use_cached_residual does not nest"
        start = self.decomposition_start_layer
        self._cached_residual = (self.residual_at(idx, start), start)
        try:
            yield
        finally:
            self._cached_residual = None

    @property
    def decomposition_start_layer(self) -> int:
        """Lowest block index holding a decomposition target. Blocks below it are frozen and
        component-free, so the residual entering this block is identical across masked forwards
        and can be cached once (see `residual_at` / `forward_from_residual`)."""
        return min(int(p.split("layers.")[1].split(".")[0]) for p in self.component_modules)

    @torch.no_grad()
    def residual_at(self, idx: Int[Tensor, "b t"], layer: int) -> Float[Tensor, "b t d"]:
        """Clean residual entering block `layer`, run un-checkpointed under no_grad. Valid only
        for `layer <= decomposition_start_layer` (the prefix is frozen + component-free, so no
        masks/grad are needed and it is constant across this step's masked forwards)."""
        assert layer <= self.decomposition_start_layer, "prefix must be below all component sites"
        x = self.embed_tokens(idx)
        for block in self._layers[:layer]:
            x = block(x)
        return x

    def forward_from_residual(
        self,
        residual: Float[Tensor, "b t d"],
        start_layer: int,
        mask_infos: MaskInfos | None = None,
    ) -> Float[Tensor, "b t vocab"] | Float[Tensor, "b t d"]:
        """Run blocks `[start_layer:]` + final norm + head on a cached `residual`, threading
        masks. With `start_layer == 0` and `residual = embed_tokens(idx)` this is the full
        forward; with `start_layer == decomposition_start_layer` and a cached `residual_at` it
        skips the frozen prefix. Bit-identical either way (same ops on the suffix). Capture (when
        active) goes through the un-checkpointed path so the stashed act is the live forward's."""
        x = residual
        blocks: list[ComponentLlamaBlock] = self._layers[start_layer:]  # pyright: ignore[reportAssignmentType]
        if self._use_activation_checkpointing and not self._capturing:
            for block in blocks:
                x = checkpoint(block, x, mask_infos, use_reentrant=False)
        else:
            for block in blocks:
                x = block(x, mask_infos)
        x = self.norm(x)
        return x if self._bypass_lm_head else self.lm_head(x)

    def forward_acts(
        self,
        idx: Int[Tensor, "b t"],
        mask_infos: MaskInfos | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Traceable forward returning `(output, pre_weight_acts)` — each decomposed site's input
        act in a returned dict (no capture_acts stash/hooks). For torch.compile(fullgraph) loss.
        Runs un-checkpointed and ignores residual-start (compiler CSEs the shared prefix)."""
        acts_out: dict[str, Tensor] = {}
        x = _proj(self.embed_tokens, idx, mask_infos, acts_out)
        blocks: list[ComponentLlamaBlock] = self._layers  # pyright: ignore[reportAssignmentType]
        for block in blocks:
            x = block(x, mask_infos, acts_out)
        x = self.norm(x)
        out = x if self._bypass_lm_head else self.lm_head(x)
        return out, acts_out

    @property
    def _capturing(self) -> bool:
        return any(m.capture_mode != "none" for m in self.component_modules.values())

    def forward_with_pre_weight_acts(
        self, idx: Int[Tensor, "b t"], mask_infos: MaskInfos | None = None
    ) -> tuple[Tensor, PreWeightActs]:
        with capture_acts(self.component_modules.values(), "input") as collected:
            out = self(idx, mask_infos)
        return out, collected[0]

    def pre_weight_acts(self, idx: Int[Tensor, "b t"]) -> PreWeightActs:
        return self.forward_with_pre_weight_acts(idx)[1]

    def forward_with_output_acts(
        self, idx: Int[Tensor, "b t"], mask_infos: MaskInfos | None = None
    ) -> tuple[Tensor, PreWeightActs]:
        with capture_acts(self.component_modules.values(), "output") as collected:
            out = self(idx, mask_infos)
        return out, collected[0]

    @contextmanager
    def bypass_lm_head(self) -> Iterator[Float[Tensor, "vocab d_model"]]:
        self._bypass_lm_head = True
        try:
            yield self.lm_head.weight
        finally:
            self._bypass_lm_head = False

    @property
    def component_modules(self) -> dict[str, ComponentLinear]:
        return {m.path: m for m in self.modules() if isinstance(m, ComponentLinear)}

    @property
    def components(self) -> dict[str, Components]:
        return {path: m.components for path, m in self.component_modules.items()}

    @property
    def module_to_c(self) -> dict[str, int]:
        return {path: m.components.C for path, m in self.component_modules.items()}

    @property
    def target_module_paths(self) -> list[str]:
        return list(self.component_modules)

    def target_weight(self, module_name: str) -> Float[Tensor, "rows cols"]:
        return self.component_modules[module_name].target_weight

    def calc_weight_deltas(self) -> dict[str, Float[Tensor, "d_out d_in"]]:
        return {
            path: self.target_weight(path) - m.components.weight
            for path, m in self.component_modules.items()
        }

    def drop_components(self) -> None:
        """Free the per-site V/U params — for the CI pool, which holds no V/U.
        Only safe with a global CI fn (asserted: no embedding sites). See
        `ComponentGPT2.drop_components`."""
        assert not any(
            isinstance(m.components, EmbeddingComponents) for m in self.component_modules.values()
        ), "drop_components() needs V to convert token ids to acts for an embedding site"
        for m in self.component_modules.values():
            comp = m.components
            for pname in ("V", "U", "bias"):
                if hasattr(comp, pname) and getattr(comp, pname) is not None:
                    delattr(comp, pname)


def componentize_llama(model: VendoredLlama, components: dict[str, Components]) -> ComponentLlama:
    """In-place: freeze the target, swap decomposition-target leaves for component modules, and
    re-point the mlp / attn / block / model forwards to mask-threading variants.

    `components` is keyed by submodule path (e.g. `layers.18.mlp.gate_proj`)."""
    for param in model.parameters():
        param.requires_grad_(False)

    for path, comp in components.items():
        parent_path, _, attr = path.rpartition(".")
        parent = model.get_submodule(parent_path)
        target_module = getattr(parent, attr)
        assert isinstance(comp, LinearComponents), (
            f"vendored Llama only decomposes nn.Linear leaves; got {type(comp)} at {path}"
        )
        new = ComponentLinear(comp, target_module.weight.data)
        new.path = path
        setattr(parent, attr, new)

    for block in model._layers:
        block.self_attn.__class__ = ComponentLlamaAttention
        block.mlp.__class__ = ComponentLlamaMLP
        block.__class__ = ComponentLlamaBlock
    model.__class__ = ComponentLlama
    return cast(ComponentLlama, model)
