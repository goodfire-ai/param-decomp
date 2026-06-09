"""Component-aware GPT2: a pure, checkpointable masked forward for the decomposition.

`componentize_gpt2` takes a frozen `GPT2Simple` (e.g. from `GPT2Simple.from_hf_pretrained`)
and a set of core `Components`, swaps the decomposition-target leaves for in-tree
`ComponentLinear` / `ComponentEmbedding` modules, and re-points the attention / MLP / block /
model forwards (via `__class__` reassignment) at variants that thread a path-keyed
`mask_infos` dict down to those leaves.

Threading the masks as a forward argument — rather than via side-channel hooks — is what makes
the masked forward checkpointable (`torch.utils.checkpoint` saves and replays the `mask_infos`
arg across the recompute boundary; verified bit-for-bit on grads in
`scripts/derisk_ckpt_grad_equivalence.py`), as well as FSDP- and compile-friendly.

A site absent from `mask_infos` (or `mask_infos is None`) routes through the frozen target —
so the clean forward reproduces the original `GPT2Simple` exactly.

Two further forward facets, both threaded explicitly rather than via hooks:
  - `collect`: an output dict that captures each decomposed site's input activation (the
    `pre_weight_acts` feeding the CI fn). Populated on a non-checkpointed clean forward.
  - `bypass_lm_head()`: a context under which `forward` returns the post-final-LN hidden state
    instead of logits (for fused-KL layerwise/PPGD reconstruction), and yields `lm_head.weight`.
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
from transformers.pytorch_utils import Conv1D as RadfordConv1D

from param_decomp.base_config import runtime_cast
from param_decomp.components import Components, EmbeddingComponents, LinearComponents
from param_decomp.masks import ComponentsMaskInfo
from param_decomp_lab.experiments.lm.pretrain.models.gpt2_simple import (
    MLP,
    Block,
    CausalSelfAttention,
    GPT2Simple,
)
from param_decomp_lab.experiments.lm.vendored.component_modules import (
    ComponentEmbedding,
    ComponentLinear,
    _ComponentModule,
)
from param_decomp_lab.experiments.lm.vendored.fp8_frozen import (
    convert_frozen_linears_to_fp8,
    fp8_frozen_enabled,
)

# self.bias is the registered causal-mask buffer; indexing it trips reportIndexIssue (as in
# the parent gpt2_simple.py).
# pyright: reportIndexIssue=false

MaskInfos = dict[str, ComponentsMaskInfo]
PreWeightActs = dict[str, Tensor]


def _proj(
    module: nn.Module,
    x: Tensor,
    mask_infos: MaskInfos | None,
    collect: PreWeightActs | None,
    collect_outputs: PreWeightActs | None = None,
) -> Tensor:
    """Apply a (possibly component-decomposed) leaf, routing its mask in by path.

    For component leaves, optionally records the leaf's input into `collect` (the
    pre-weight-acts capture) and the leaf's output into `collect_outputs` (the
    post-weight-acts capture used by the hidden-acts eval metrics). Plain leaves are
    applied unchanged.
    """
    if isinstance(module, _ComponentModule):
        if collect is not None:
            collect[module.path] = x
        mask_info = None if mask_infos is None else mask_infos.get(module.path)
        out = module(x, mask_info)
        if collect_outputs is not None:
            collect_outputs[module.path] = out
        return out
    return module(x)


class ComponentCausalSelfAttention(CausalSelfAttention):
    @override
    def forward(
        self,
        x: Float[Tensor, "batch pos d_model"],
        mask_infos: MaskInfos | None = None,
        collect: PreWeightActs | None = None,
        collect_outputs: PreWeightActs | None = None,
    ) -> Float[Tensor, "batch pos d_model"]:
        B, T, C = x.size()
        q = _proj(self.q_proj, x, mask_infos, collect, collect_outputs)
        k = _proj(self.k_proj, x, mask_infos, collect, collect_outputs)
        v = _proj(self.v_proj, x, mask_infos, collect, collect_outputs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash_attention:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return _proj(self.o_proj, y, mask_infos, collect, collect_outputs)


class ComponentMLP(MLP):
    @override
    def forward(
        self,
        x: Float[Tensor, "... dim"],
        mask_infos: MaskInfos | None = None,
        collect: PreWeightActs | None = None,
        collect_outputs: PreWeightActs | None = None,
    ) -> Float[Tensor, "... dim"]:
        x = _proj(self.c_fc, x, mask_infos, collect, collect_outputs)
        x = self.gelu(x)
        return _proj(self.down_proj, x, mask_infos, collect, collect_outputs)


class ComponentBlock(Block):
    @override
    def forward(
        self,
        x: Float[Tensor, "batch pos d_model"],
        mask_infos: MaskInfos | None = None,
        collect: PreWeightActs | None = None,
        collect_outputs: PreWeightActs | None = None,
    ) -> Float[Tensor, "batch pos d_model"]:
        attn: ComponentCausalSelfAttention = self.attn  # pyright: ignore[reportAssignmentType]
        mlp: ComponentMLP = self.mlp  # pyright: ignore[reportAssignmentType]
        x = x + attn(self.ln_1(x), mask_infos, collect, collect_outputs)
        x = x + mlp(self.ln_2(x), mask_infos, collect, collect_outputs)
        return x


class ComponentGPT2(GPT2Simple):
    """`GPT2Simple` whose forward threads a path-keyed `mask_infos` to in-tree components.

    Built by `componentize_gpt2`, not constructed directly (it adopts a populated `GPT2Simple`
    via `__class__` reassignment, so it has no `__init__` of its own). `forward` returns logits,
    or the post-final-LN hidden state while `bypass_lm_head()` is active.
    """

    _bypass_lm_head: bool = False
    _cached_residual: tuple[Tensor, int] | None = None

    @override
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        idx: Int[Tensor, "batch pos"],
        mask_infos: MaskInfos | None = None,
        collect: PreWeightActs | None = None,
        collect_outputs: PreWeightActs | None = None,
    ) -> Float[Tensor, "batch pos vocab"] | Float[Tensor, "batch pos d_model"]:
        if self._cached_residual is not None:
            residual, start = self._cached_residual
            return self.forward_from_residual(residual, start, mask_infos, collect, collect_outputs)
        _b, t = idx.size()
        assert t <= self.config.block_size, (
            f"sequence length {t} exceeds block size {self.config.block_size}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        embed = _proj(self.wte, idx, mask_infos, collect, collect_outputs) + self.wpe(pos)
        return self.forward_from_residual(embed, 0, mask_infos, collect, collect_outputs)

    @property
    def decomposition_start_layer(self) -> int:
        """Lowest block index holding a decomposition target. For GPT-2 q/k this is 0 (every
        layer is decomposed), so residual-start is a no-op (empty prefix) but stays bit-identical.
        Asserts all sites are block sites (`h.<i>....`) — residual-start can't skip an embedding
        decomposition (the cached residual is already past the embedding)."""
        idxs: list[int] = []
        for p in self.component_modules:
            assert p.startswith("h."), f"residual-start needs block sites; got embedding-ish {p!r}"
            idxs.append(int(p.split(".")[1]))
        return min(idxs)

    @torch.no_grad()
    def residual_at(
        self, idx: Int[Tensor, "batch pos"], layer: int
    ) -> Float[Tensor, "batch pos d_model"]:
        """Clean residual entering block `layer` (`wte + wpe` then blocks `[:layer]`), run
        un-checkpointed under no_grad. Valid only for `layer <= decomposition_start_layer`."""
        assert layer <= self.decomposition_start_layer, "prefix must be below all component sites"
        _b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        blocks: list[ComponentBlock] = self._h  # pyright: ignore[reportAssignmentType]
        for block in blocks[:layer]:
            x = block(x)
        return x

    def forward_from_residual(
        self,
        residual: Float[Tensor, "batch pos d_model"],
        start_layer: int,
        mask_infos: MaskInfos | None = None,
        collect: PreWeightActs | None = None,
        collect_outputs: PreWeightActs | None = None,
    ) -> Float[Tensor, "batch pos vocab"] | Float[Tensor, "batch pos d_model"]:
        """Run blocks `[start_layer:]` + final LN + head on a cached `residual`, threading masks.
        With `start_layer == 0` and `residual = wte+wpe` this is the full forward."""
        x = residual
        blocks: list[ComponentBlock] = self._h[start_layer:]  # pyright: ignore[reportAssignmentType]
        if self._use_activation_checkpointing and collect is None and collect_outputs is None:
            for block in blocks:
                x = checkpoint(block, x, mask_infos, use_reentrant=False)
        else:
            for block in blocks:
                x = block(x, mask_infos, collect, collect_outputs)
        x = self.ln_f(x)
        return x if self._bypass_lm_head else self.lm_head(x)

    @contextmanager
    def use_cached_residual(self, idx: Int[Tensor, "batch pos"]) -> Iterator[None]:
        """Within this context, `forward` runs the masked suffix from a clean residual entering
        `decomposition_start_layer` (computed once from `idx`), skipping the frozen prefix. Every
        forward inside reuses the one cached residual. Output is identical to the full forward.
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

    def forward_with_pre_weight_acts(
        self, idx: Int[Tensor, "batch pos"], mask_infos: MaskInfos | None = None
    ) -> tuple[Tensor, PreWeightActs]:
        """Forward that also returns each decomposed site's input activation (token-ids for an
        embedding site). Runs un-checkpointed (the capture is a clean diagnostic pass)."""
        collect: PreWeightActs = {}
        out = self(idx, mask_infos, collect)
        return out, collect

    def pre_weight_acts(self, idx: Int[Tensor, "batch pos"]) -> PreWeightActs:
        return self.forward_with_pre_weight_acts(idx)[1]

    def forward_with_output_acts(
        self, idx: Int[Tensor, "batch pos"], mask_infos: MaskInfos | None = None
    ) -> tuple[Tensor, PreWeightActs]:
        """Forward that also returns each decomposed site's OUTPUT activation — the
        post-weight (post-component-replacement when masked) act read by the hidden-acts
        eval metrics. Replaces the core model's `cache_type="output"`. Runs un-checkpointed."""
        collect_outputs: PreWeightActs = {}
        out = self(idx, mask_infos, None, collect_outputs)
        return out, collect_outputs

    @contextmanager
    def bypass_lm_head(self) -> Iterator[Float[Tensor, "vocab d_model"]]:
        """Within this context, `forward` returns the post-final-LN hidden state instead of
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
        """Free the per-site V/U params — for the CI pool, which holds no V/U.

        The CI pool only runs the clean forward (target path, via the frozen
        `target_weight` buffer), `pre_weight_acts`, and `calc_causal_importances` (CI fn
        only) — none of which touch V/U. Deleting the V/U/bias `nn.Parameter`s drops them
        from `.parameters()` / `.state_dict()` and keeps them off the GPU after `.to`.
        `C` survives on the `Components` object so `module_to_c` / `target_module_paths`
        still answer. Post-init, pre-device — same lifecycle as `LMComponentModel.drop_ci_fn`.

        Only safe with a global CI fn: a layerwise CI fn reads `V` via
        `get_component_acts` on every site, and `EmbeddingComponents` need `V` even under
        a global CI fn (token-id → acts). Both are asserted against here.
        """
        from param_decomp.components import EmbeddingComponents

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
        case RadfordConv1D():
            return target_module.weight.data.T
        case nn.Linear() | nn.Embedding():
            return target_module.weight.data
        case _:
            raise ValueError(f"unsupported target module {type(target_module)}")


def componentize_gpt2(model: GPT2Simple, components: dict[str, Components]) -> ComponentGPT2:
    """In-place: freeze the target, swap decomposition-target leaves for component modules, and
    re-point the attention / MLP / block / model forwards to mask-threading variants.

    `components` is keyed by submodule path (as returned by `make_components`). The returned
    value is the same object, now a `ComponentGPT2`.
    """
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
    model.__class__ = ComponentGPT2

    if fp8_frozen_enabled():
        start = model.decomposition_start_layer
        n = sum(convert_frozen_linears_to_fp8(block) for block in model._h[start:])
        print(
            f"[fp8] swapped {n} frozen nn.Linear leaves to Fp8FrozenLinear "
            f"in suffix blocks [{start}:]",
            flush=True,
        )

    return model  # pyright: ignore[reportReturnType]
