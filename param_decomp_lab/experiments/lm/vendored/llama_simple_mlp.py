"""Component-aware LlamaSimpleMLP: a pure, checkpointable masked forward for the decomposition.

The LlamaSimpleMLP sibling of `vendored/gpt2.py`. `componentize_llama_simple_mlp` takes a frozen
`LlamaSimpleMLP` (Llama-style attention — RMSNorm, RoPE, GQA q/k/v/o — with a GPT-2-style GELU
2-matrix MLP), swaps the decomposition-target leaves for in-tree `ComponentLinear`, and re-points
the attention / MLP / block / model forwards (via `__class__` reassignment) at variants that
thread a path-keyed `mask_infos` dict to those leaves.

Same contract as `ComponentGPT2`: masks threaded as a forward arg (checkpoint/FSDP/compile
friendly), a missing site (or `mask_infos is None`) routes through the frozen target, and each
decomposed site stashes its pre/post-weight act on itself (`_ComponentModule` + `capture_acts`).
Differences from GPT-2: positions are RoPE (no learned `wpe`), norms are RMSNorm, attention is
grouped-query. `LlamaSimpleMLP` has no `enable_activation_checkpointing`, so this subclass adds it.
"""

import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import override

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from param_decomp.components import Components, EmbeddingComponents, LinearComponents
from param_decomp.masks import ComponentsMaskInfo
from param_decomp_config.base import runtime_cast
from param_decomp_lab.experiments.lm.pretrain.models.llama_simple_mlp import (
    MLP,
    Block,
    CausalSelfAttention,
    LlamaSimpleMLP,
)
from param_decomp_lab.experiments.lm.vendored.component_modules import (
    ComponentEmbedding,
    ComponentLinear,
    _ComponentModule,
    capture_acts,
)

# self.bias is the registered causal-mask buffer; indexing it trips reportIndexIssue.
# pyright: reportIndexIssue=false

MaskInfos = dict[str, ComponentsMaskInfo]
PreWeightActs = dict[str, Tensor]


def _proj(
    module: nn.Module,
    x: Tensor,
    mask_infos: MaskInfos | None,
    acts_out: dict[str, Tensor] | None = None,
) -> Tensor:
    """Apply a (possibly component-decomposed) leaf, routing its mask in by path. Component
    leaves stash their pre/post-weight acts on themselves when capture is active; plain leaves
    are applied unchanged."""
    if isinstance(module, _ComponentModule):
        if acts_out is not None:
            acts_out[module.path] = x
        mask_info = None if mask_infos is None else mask_infos.get(module.path)
        return module(x, mask_info)
    return module(x)


class ComponentCausalSelfAttention(CausalSelfAttention):
    """Grouped-query Llama attention threading masks through q/k/v/o; RoPE unchanged.

    Reproduces the parent forward's GQA + RoPE math, swapping only the four projections for the
    mask-routed `_proj`. Assumes the grouped-query path (separate q/k/v projections) — the only
    one the decomposition targets (`*.attn.{q,k,v,o}_proj`) exist under."""

    @override
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        x: Float[Tensor, "batch pos d_model"],
        mask_infos: MaskInfos | None = None,
        acts_out: dict[str, Tensor] | None = None,
    ) -> Float[Tensor, "batch pos d_model"]:
        assert self.use_grouped_query_attention, "component path supports only the GQA projections"
        B, T, C = x.size()
        q = _proj(self.q_proj, x, mask_infos, acts_out)
        k = _proj(self.k_proj, x, mask_infos, acts_out)
        v = _proj(self.v_proj, x, mask_infos, acts_out)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_key_value_heads, self.head_dim).transpose(1, 2)

        position_ids = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)
        position_ids = position_ids.clamp(max=self.n_ctx - 1)
        cos = self.rotary_cos[position_ids].to(q.dtype)
        sin = self.rotary_sin[position_ids].to(q.dtype)
        q, k = self.apply_rotary_pos_emb(q, k, cos, sin)

        if self.repeat_kv_heads > 1:
            k = k.repeat_interleave(self.repeat_kv_heads, dim=1)
            v = v.repeat_interleave(self.repeat_kv_heads, dim=1)

        if self.flash_attention:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return _proj(self.o_proj, y, mask_infos, acts_out)


class ComponentMLP(MLP):
    @override
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        x: Float[Tensor, "... dim"],
        mask_infos: MaskInfos | None = None,
        acts_out: dict[str, Tensor] | None = None,
    ) -> Float[Tensor, "... dim"]:
        x = _proj(self.c_fc, x, mask_infos, acts_out)
        x = self.gelu(x)
        return _proj(self.down_proj, x, mask_infos, acts_out)


class ComponentBlock(Block):
    @override
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        x: Float[Tensor, "batch pos d_model"],
        mask_infos: MaskInfos | None = None,
        acts_out: dict[str, Tensor] | None = None,
    ) -> Float[Tensor, "batch pos d_model"]:
        attn: ComponentCausalSelfAttention = self.attn  # pyright: ignore[reportAssignmentType]
        mlp: ComponentMLP = self.mlp  # pyright: ignore[reportAssignmentType]
        x = x + attn(self.rms_1(x), mask_infos, acts_out)
        x = x + mlp(self.rms_2(x), mask_infos, acts_out)
        return x


class ComponentLlamaSimpleMLP(LlamaSimpleMLP):
    """`LlamaSimpleMLP` whose forward threads a path-keyed `mask_infos` to in-tree components.

    Built by `componentize_llama_simple_mlp`, not constructed directly (it adopts a populated
    `LlamaSimpleMLP` via `__class__` reassignment, so it has no `__init__`). `forward` returns
    logits (not the parent's `(logits, loss)` tuple), or the post-final-norm hidden state while
    `bypass_lm_head()` is active. The activation-checkpointing flag lives here because the parent
    `LlamaSimpleMLP` has no such mechanism."""

    _bypass_lm_head: bool = False
    _cached_residual: tuple[Tensor, int] | None = None
    _use_activation_checkpointing: bool = False

    def enable_activation_checkpointing(self) -> None:
        self._use_activation_checkpointing = True

    @override
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        idx: Int[Tensor, "batch pos"],
        mask_infos: MaskInfos | None = None,
    ) -> Float[Tensor, "batch pos vocab"] | Float[Tensor, "batch pos d_model"]:
        if self._cached_residual is not None:
            residual, start = self._cached_residual
            return self.forward_from_residual(residual, start, mask_infos)
        _b, t = idx.size()
        assert t <= self.config.block_size, (
            f"sequence length {t} exceeds block size {self.config.block_size}"
        )
        embed = _proj(self.wte, idx, mask_infos)
        return self.forward_from_residual(embed, 0, mask_infos)

    @property
    def decomposition_start_layer(self) -> int:
        """Lowest block index holding a decomposition target. Asserts all sites are block sites
        (`h.<i>....`) — residual-start can't skip an embedding decomposition."""
        idxs: list[int] = []
        for p in self.component_modules:
            assert p.startswith("h."), f"residual-start needs block sites; got embedding-ish {p!r}"
            idxs.append(int(p.split(".")[1]))
        return min(idxs)

    @torch.no_grad()
    def residual_at(
        self, idx: Int[Tensor, "batch pos"], layer: int
    ) -> Float[Tensor, "batch pos d_model"]:
        """Clean residual entering block `layer` (`wte` then blocks `[:layer]`), un-checkpointed
        under no_grad. Valid only for `layer <= decomposition_start_layer`."""
        assert layer <= self.decomposition_start_layer, "prefix must be below all component sites"
        x = self.wte(idx)
        blocks: list[ComponentBlock] = self._h  # pyright: ignore[reportAssignmentType]
        for block in blocks[:layer]:
            x = block(x)
        return x

    def forward_from_residual(
        self,
        residual: Float[Tensor, "batch pos d_model"],
        start_layer: int,
        mask_infos: MaskInfos | None = None,
    ) -> Float[Tensor, "batch pos vocab"] | Float[Tensor, "batch pos d_model"]:
        """Run blocks `[start_layer:]` + final RMSNorm + head on a cached `residual`, threading
        masks. With `start_layer == 0` and `residual = wte(idx)` this is the full forward. Capture
        (when active) goes through the un-checkpointed path so the stashed act is the live one."""
        x = residual
        blocks: list[ComponentBlock] = self._h[start_layer:]  # pyright: ignore[reportAssignmentType]
        if self._use_activation_checkpointing and not self._capturing:
            for block in blocks:
                x = checkpoint(block, x, mask_infos, use_reentrant=False)
        else:
            for block in blocks:
                x = block(x, mask_infos)
        x = self.ln_f(x)
        return x if self._bypass_lm_head else self.lm_head(x)

    def forward_acts(
        self,
        idx: Int[Tensor, "batch pos"],
        mask_infos: MaskInfos | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Traceable forward returning `(output, pre_weight_acts)` as graph outputs — each
        decomposed site's input act collected into a returned dict (no `capture_acts` module
        stash, no hooks). For the `torch.compile(fullgraph=True)` loss path. Runs un-checkpointed
        and ignores residual-start (the compiler CSEs the shared prefix)."""
        acts_out: dict[str, Tensor] = {}
        embed = _proj(self.wte, idx, mask_infos, acts_out)
        x = embed
        blocks: list[ComponentBlock] = self._h  # pyright: ignore[reportAssignmentType]
        for block in blocks:
            x = block(x, mask_infos, acts_out)
        x = self.ln_f(x)
        out = x if self._bypass_lm_head else self.lm_head(x)
        return out, acts_out

    @property
    def _capturing(self) -> bool:
        return any(m.capture_mode != "none" for m in self.component_modules.values())

    @contextmanager
    def use_cached_residual(self, idx: Int[Tensor, "batch pos"]) -> Iterator[None]:
        """Within this context, `forward` runs the masked suffix from a clean residual entering
        `decomposition_start_layer` (computed once from `idx`), skipping the frozen prefix.
        `PD_DISABLE_RESIDUAL_START=1` makes it a no-op (full forward)."""
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

    def forward_with_pre_weight_acts(
        self, idx: Int[Tensor, "batch pos"], mask_infos: MaskInfos | None = None
    ) -> tuple[Tensor, PreWeightActs]:
        """Forward that also returns each decomposed site's input activation. Runs un-checkpointed."""
        with capture_acts(self.component_modules.values(), "input") as collected:
            out = self(idx, mask_infos)
        return out, collected[0]

    def pre_weight_acts(self, idx: Int[Tensor, "batch pos"]) -> PreWeightActs:
        return self.forward_with_pre_weight_acts(idx)[1]

    def forward_with_output_acts(
        self, idx: Int[Tensor, "batch pos"], mask_infos: MaskInfos | None = None
    ) -> tuple[Tensor, PreWeightActs]:
        """Forward that also returns each decomposed site's OUTPUT activation. Runs un-checkpointed."""
        with capture_acts(self.component_modules.values(), "output") as collected:
            out = self(idx, mask_infos)
        return out, collected[0]

    @contextmanager
    def bypass_lm_head(self) -> Iterator[Float[Tensor, "vocab d_model"]]:
        """Within this context, `forward` returns the post-final-norm hidden state instead of
        logits; yields `lm_head.weight` for fused linear-KL reconstruction."""
        self._bypass_lm_head = True
        try:
            yield self.lm_head.weight
        finally:
            self._bypass_lm_head = False

    @property
    def component_modules(self) -> dict[str, _ComponentModule]:
        return {m.path: m for m in self.modules() if isinstance(m, _ComponentModule)}

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
        return runtime_cast(Tensor, self.component_modules[module_name].target_weight)

    def calc_weight_deltas(self) -> dict[str, Float[Tensor, "d_out d_in"]]:
        return {
            path: self.target_weight(path) - m.components.weight
            for path, m in self.component_modules.items()
        }

    def drop_components(self) -> None:
        """Free the per-site V/U params — for the CI pool, which holds no V/U. See
        `ComponentGPT2.drop_components`."""
        assert not any(
            isinstance(m.components, EmbeddingComponents) for m in self.component_modules.values()
        ), "drop_components() needs V to convert token ids to acts for an embedding site"
        for m in self.component_modules.values():
            comp = m.components
            for pname in ("V", "U", "bias"):
                if hasattr(comp, pname) and getattr(comp, pname) is not None:
                    delattr(comp, pname)


def _target_weight(target_module: nn.Module) -> Float[Tensor, "rows cols"]:
    """Frozen target weight in PD's `[d_out, d_in]` (Linear) / `[vocab, dim]` (Embedding)."""
    match target_module:
        case nn.Linear() | nn.Embedding():
            return target_module.weight.data
        case _:
            raise ValueError(f"unsupported target module {type(target_module)}")


def componentize_llama_simple_mlp(
    model: LlamaSimpleMLP, components: dict[str, Components]
) -> ComponentLlamaSimpleMLP:
    """In-place: freeze the target, swap decomposition-target leaves for component modules, and
    re-point the attention / MLP / block / model forwards to mask-threading variants.

    `components` is keyed by submodule path (as returned by `make_components`). The returned value
    is the same object, now a `ComponentLlamaSimpleMLP`."""
    for param in model.parameters():
        param.requires_grad_(False)

    for path, comp in components.items():
        parent_path, _, attr = path.rpartition(".")
        parent = model.get_submodule(parent_path)
        target_module = getattr(parent, attr)
        match comp:
            case LinearComponents():
                new: _ComponentModule = ComponentLinear(comp, _target_weight(target_module))
            case EmbeddingComponents():
                new = ComponentEmbedding(comp, _target_weight(target_module))
            case _:
                raise ValueError(f"unsupported components type {type(comp)} at {path}")
        new.path = path
        setattr(parent, attr, new)

    for block in model._h:
        block.attn.__class__ = ComponentCausalSelfAttention
        block.mlp.__class__ = ComponentMLP
        block.__class__ = ComponentBlock
    model.__class__ = ComponentLlamaSimpleMLP
    return model  # pyright: ignore[reportReturnType]
