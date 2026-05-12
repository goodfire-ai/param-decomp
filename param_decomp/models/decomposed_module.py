"""Fused decomposition-site modules.

A `DecomposedLinear` is a drop-in replacement for an `nn.Linear` (or a Radford
`Conv1D`, or our `Identity` shim) in the target model tree. It owns:
  - the frozen target module (`self.linear`)
  - the trainable component matrices `V` and `U`
  - per-call mask/cache slots set by the surrounding `ComponentModel`

When `set_mask_info(None)` is in effect the forward is identical to the wrapped
target module's forward. When a `ComponentsMaskInfo` is bound, the forward
follows the same math the legacy hook-based path used:
  - `component_acts = x @ V`
  - optional mask: `component_acts *= mask`
  - `out = component_acts @ U`
  - optional delta term: `out += delta_mask * (target_out - full_components_out)`
  - bias (if any): `out += bias`
  - optional routing mask: `where(routing_mask, out, target_out)`

The reason this is a fused module (rather than a hook on the target) is so FSDP
can wrap each site as a single sharding unit — the components' V/U live in the
*same* FSDP unit as the target submodule whose output they replace, so the
"hook crosses FSDP units" problem from §5b of `fsdp_scaling_report.html` goes
away.

See `fsdp_implementation_plan.md` for the full context.
"""

from contextlib import contextmanager
from typing import Literal, override

import einops
import torch
from jaxtyping import Float
from torch import Tensor, nn
from transformers.pytorch_utils import Conv1D as RadfordConv1D

from param_decomp.models.mask_info import ComponentsMaskInfo
from param_decomp.utils.module_utils import init_param_

CacheType = Literal["component_acts", "input", "output"]


class DecomposedLinear(nn.Module):
    """A fused decomposition site wrapping a single linear-like target module."""

    def __init__(self, base: nn.Linear | RadfordConv1D, C: int, site_name: str):
        super().__init__()
        match base:
            case nn.Linear():
                d_in, d_out = base.in_features, base.out_features
            case RadfordConv1D():
                # Conv1D stores weight as (d_in, d_out); convert callers downstream.
                d_in, d_out = base.weight.shape

        self.linear: nn.Linear | RadfordConv1D = base
        self.d_in = d_in
        self.d_out = d_out
        self.C = C
        self.site_name = site_name

        self.V = nn.Parameter(torch.empty(d_in, C))
        self.U = nn.Parameter(torch.empty(C, d_out))
        init_param_(self.V, fan_val=d_in, nonlinearity="linear")
        init_param_(self.U, fan_val=C, nonlinearity="linear")

        self._mask_info: ComponentsMaskInfo | None = None
        self._cache: dict[str, Tensor] | None = None
        self._cache_type: CacheType | None = None

    @contextmanager
    def bind(
        self,
        mask_info: ComponentsMaskInfo | None,
        cache: dict[str, Tensor] | None,
        cache_type: CacheType | None,
    ):
        """Context manager that sets per-call slots and restores afterward.

        ComponentModel uses this on every site before invoking `target_model(x)`.
        """
        prev_mask, prev_cache, prev_type = (
            self._mask_info,
            self._cache,
            self._cache_type,
        )
        self._mask_info = mask_info
        self._cache = cache
        self._cache_type = cache_type
        try:
            yield
        finally:
            self._mask_info = prev_mask
            self._cache = prev_cache
            self._cache_type = prev_type

    @property
    def target_weight(self) -> Float[Tensor, "d_out d_in"]:
        """Always (d_out, d_in) regardless of underlying base type."""
        match self.linear:
            case nn.Linear():
                return self.linear.weight
            case RadfordConv1D():
                return self.linear.weight.T

    # Aliases so callers that used to introspect `target_model.<path>.in_features` /
    # `.out_features` (back when that path was a plain `nn.Linear`) keep working.
    # NOTE: deliberately NOT exposing `.weight` — under the legacy `Components` API
    # `components[name].weight` meant `V @ U`, but `target_model.<path>.weight`
    # meant the target weight. Disambiguate by using `target_weight` or
    # `component_weight` explicitly.
    @property
    def in_features(self) -> int:
        return self.d_in

    @property
    def out_features(self) -> int:
        return self.d_out

    @property
    def bias(self) -> Tensor | None:
        # nn.Linear.bias and Conv1D.bias are typed as Tensor in torch's stubs but can be
        # None at runtime when bias=False. This property normalizes the access.
        return getattr(self.linear, "bias", None)

    @property
    def component_weight(self) -> Float[Tensor, "d_out d_in"]:
        """V @ U transposed to nn.Linear convention (d_out, d_in)."""
        return einops.einsum(self.V, self.U, "d_in C, C d_out -> d_out d_in")

    def faithfulness_terms(self) -> tuple[Float[Tensor, ""], int]:
        """Squared-error sum and parameter count for `W_target - V@U`."""
        delta = self.target_weight - self.component_weight
        return delta.pow(2).sum(), delta.numel()

    def get_component_acts(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... C"]:
        """x @ V — the pre-mask component activations. Used by MLP-type CI fns."""
        return einops.einsum(x.to(self.V.dtype), self.V, "... d_in, d_in C -> ... C")

    @override
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        if self._cache is not None and self._cache_type == "input":
            self._cache[self.site_name] = x

        if self._mask_info is None:
            out = self.linear(x)
        else:
            out = self._decomposed_forward(x, self._mask_info)

        if self._cache is not None and self._cache_type == "output":
            self._cache[self.site_name] = out

        return out

    def apply_decomposed(
        self,
        x: Float[Tensor, "... d_in"],
        mask: Float[Tensor, "... C"] | None = None,
        delta_mask: Float[Tensor, "..."] | None = None,
    ) -> Float[Tensor, "... d_out"]:
        """Run the decomposed path directly with explicit mask args.

        Used by metric/loss code that wants component output without going through
        the site's bound `_mask_info` slot (e.g. computing target vs. masked outputs
        side-by-side inside a single forward).
        """
        info = ComponentsMaskInfo(
            component_mask=mask
            if mask is not None
            else torch.ones((), device=self.V.device, dtype=self.V.dtype),
            routing_mask="all",
            delta_mask=delta_mask,
        )
        return self._decomposed_forward(x, info)

    def _decomposed_forward(
        self,
        x: Float[Tensor, "... d_in"],
        info: ComponentsMaskInfo,
    ) -> Float[Tensor, "... d_out"]:
        component_acts = einops.einsum(x.to(self.V.dtype), self.V, "... d_in, d_in C -> ... C")

        if self._cache is not None and self._cache_type == "component_acts":
            self._cache[f"{self.site_name}_pre_detach"] = component_acts
            component_acts = component_acts.detach().requires_grad_(True)
            self._cache[f"{self.site_name}_post_detach"] = component_acts

        masked_component_acts = component_acts * info.component_mask
        components_out: Tensor = einops.einsum(
            masked_component_acts, self.U, "... C, C d_out -> ... d_out"
        )

        target_out: Tensor | None = None
        if info.delta_mask is not None:
            full_components_out = einops.einsum(
                component_acts, self.U, "... C, C d_out -> ... d_out"
            )
            site_target_out: Tensor = self.linear(x)
            target_out = site_target_out
            if self.bias is not None:
                full_components_out = full_components_out + self.bias
            delta_out = site_target_out - full_components_out
            assert delta_out.shape[:-1] == info.delta_mask.shape
            components_out = components_out + einops.einsum(
                info.delta_mask, delta_out, "..., ... d_out -> ... d_out"
            )

        if self.bias is not None:
            components_out = components_out + self.bias

        if info.routing_mask == "all":
            return components_out
        else:
            routing_target_out: Tensor = target_out if target_out is not None else self.linear(x)
            return torch.where(info.routing_mask[..., None], components_out, routing_target_out)


class DecomposedEmbedding(nn.Module):
    """Fused embedding decomposition site.

    Mirrors `EmbeddingComponents` semantics: V is `(vocab, C)`, U is `(C, d_embed)`,
    and the decomposed forward indexes V by the input ids (no weight materialization)
    instead of doing an einsum like `DecomposedLinear`.
    """

    def __init__(self, base: nn.Embedding, C: int, site_name: str):
        super().__init__()
        self.embedding = base
        self.vocab_size = base.num_embeddings
        self.d_embed = base.embedding_dim
        self.C = C
        self.site_name = site_name

        self.V = nn.Parameter(torch.empty(self.vocab_size, C))
        self.U = nn.Parameter(torch.empty(C, self.d_embed))
        init_param_(self.V, fan_val=self.vocab_size, nonlinearity="linear")
        init_param_(self.U, fan_val=C, nonlinearity="linear")

        self._mask_info: ComponentsMaskInfo | None = None
        self._cache: dict[str, Tensor] | None = None
        self._cache_type: CacheType | None = None

    # Aliases so callers that used to access `target_model.embed.num_embeddings` etc. (back
    # when `target_model.embed` was a plain `nn.Embedding`) keep working — we are now a fused
    # site sitting at that path.
    @property
    def num_embeddings(self) -> int:
        return self.vocab_size

    @property
    def embedding_dim(self) -> int:
        return self.d_embed

    @contextmanager
    def bind(
        self,
        mask_info: ComponentsMaskInfo | None,
        cache: dict[str, Tensor] | None,
        cache_type: CacheType | None,
    ):
        prev_mask, prev_cache, prev_type = (
            self._mask_info,
            self._cache,
            self._cache_type,
        )
        self._mask_info = mask_info
        self._cache = cache
        self._cache_type = cache_type
        try:
            yield
        finally:
            self._mask_info = prev_mask
            self._cache = prev_cache
            self._cache_type = prev_type

    @property
    def target_weight(self) -> Float[Tensor, "vocab d_embed"]:
        return self.embedding.weight

    @property
    def component_weight(self) -> Float[Tensor, "vocab d_embed"]:
        return einops.einsum(self.V, self.U, "vocab C, C d_embed -> vocab d_embed")

    def faithfulness_terms(self) -> tuple[Float[Tensor, ""], int]:
        """Squared-error sum and parameter count for `W_target - V@U`."""
        delta = self.target_weight - self.component_weight
        return delta.pow(2).sum(), delta.numel()

    def get_component_acts(self, idx: Tensor) -> Float[Tensor, "... C"]:
        """V[idx] — the pre-mask component activations for embedding sites."""
        return self.V[idx]

    @override
    def forward(self, idx: Tensor) -> Float[Tensor, "... d_embed"]:
        if self._cache is not None and self._cache_type == "input":
            self._cache[self.site_name] = idx

        if self._mask_info is None:
            out = self.embedding(idx)
        else:
            out = self._decomposed_forward(idx, self._mask_info)

        if self._cache is not None and self._cache_type == "output":
            self._cache[self.site_name] = out

        return out

    def apply_decomposed(
        self,
        idx: Tensor,
        mask: Float[Tensor, "... C"] | None = None,
        delta_mask: Float[Tensor, "..."] | None = None,
    ) -> Float[Tensor, "... d_embed"]:
        """Run the decomposed path directly with explicit mask args.

        Used by metric/loss code that wants component output without going through
        the site's bound `_mask_info` slot.
        """
        info = ComponentsMaskInfo(
            component_mask=mask
            if mask is not None
            else torch.ones((), device=self.V.device, dtype=self.V.dtype),
            routing_mask="all",
            delta_mask=delta_mask,
        )
        return self._decomposed_forward(idx, info)

    def _decomposed_forward(
        self, idx: Tensor, info: ComponentsMaskInfo
    ) -> Float[Tensor, "... d_embed"]:
        component_acts: Tensor = self.V[idx]  # (..., C)

        if self._cache is not None and self._cache_type == "component_acts":
            self._cache[f"{self.site_name}_pre_detach"] = component_acts
            component_acts = component_acts.detach().requires_grad_(True)
            self._cache[f"{self.site_name}_post_detach"] = component_acts

        masked_component_acts = component_acts * info.component_mask
        components_out: Tensor = einops.einsum(
            masked_component_acts, self.U, "... C, C d_embed -> ... d_embed"
        )

        target_out: Tensor | None = None
        if info.delta_mask is not None:
            full_components_out = einops.einsum(
                component_acts, self.U, "... C, C d_embed -> ... d_embed"
            )
            site_target_out: Tensor = self.embedding(idx)
            target_out = site_target_out
            delta_out = site_target_out - full_components_out
            assert delta_out.shape[:-1] == info.delta_mask.shape
            components_out = components_out + einops.einsum(
                info.delta_mask, delta_out, "..., ... d_embed -> ... d_embed"
            )

        if info.routing_mask == "all":
            return components_out
        else:
            routing_target_out: Tensor = (
                target_out if target_out is not None else self.embedding(idx)
            )
            return torch.where(info.routing_mask[..., None], components_out, routing_target_out)


DecomposedSite = DecomposedLinear | DecomposedEmbedding


def _wrap_target_module(base: nn.Module, C: int, site_name: str) -> DecomposedSite:
    """Build the right `Decomposed*` site for a target submodule."""
    match base:
        case nn.Linear() | RadfordConv1D():
            return DecomposedLinear(base, C=C, site_name=site_name)
        case nn.Embedding():
            return DecomposedEmbedding(base, C=C, site_name=site_name)
        case _:
            raise ValueError(
                f"_wrap_target_module: unsupported base {type(base)}. "
                f"The legacy `identity_module_info` / Identity shim path is not supported "
                f"under the fused-decomposition-sites architecture."
            )


def install_decomposed_sites(
    target_model: nn.Module, module_to_c: dict[str, int]
) -> dict[str, DecomposedSite]:
    """Replace each named submodule of `target_model` in place with a `DecomposedSite`.

    Returns a dict of `{site_name: DecomposedSite}` for callers that want direct
    handles (loss code, harvest, etc.); the canonical reference is the site's
    position in the target_model tree.

    Lookups happen up front so that nested decomposition paths
    (e.g. `linear1` and `linear1.pre_identity` both decomposed) resolve against
    the *original* tree shape — replacing `linear1` first would hide its
    `pre_identity` child.
    """
    resolved: list[tuple[str, str, str, nn.Module, int]] = []
    for site_name, C in module_to_c.items():
        base = target_model.get_submodule(site_name)
        parent_path, _, child_name = site_name.rpartition(".")
        parent = target_model.get_submodule(parent_path) if parent_path else target_model
        resolved.append((site_name, parent_path, child_name, base, C))

    # Deepest path first so the parent's `setattr` doesn't clobber an already-wrapped child
    # (e.g. `linear1.pre_identity` must be installed before `linear1`).
    resolved.sort(key=lambda entry: entry[0].count("."), reverse=True)

    sites: dict[str, DecomposedSite] = {}
    for site_name, _parent_path, child_name, base, c in resolved:
        # Re-resolve the parent against the current tree state — a deeper sibling may
        # have rewritten an ancestor's attribute to a Decomposed* site, but we still
        # want to attach our new site at the same name on the same parent.
        parent_path, _, _ = site_name.rpartition(".")
        parent = target_model.get_submodule(parent_path) if parent_path else target_model
        site = _wrap_target_module(base, C=c, site_name=site_name)
        setattr(parent, child_name, site)
        sites[site_name] = site

    # Restore the dict in the original key order so iteration matches user-provided order.
    return {name: sites[name] for name in module_to_c}


def get_site(target_model: nn.Module, site_name: str) -> DecomposedSite:
    """Walk the target tree and assert the named submodule is a decomposed site."""
    mod = target_model.get_submodule(site_name)
    assert isinstance(mod, DecomposedLinear | DecomposedEmbedding), (
        f"Module at {site_name!r} is {type(mod).__name__}, not a decomposed site"
    )
    return mod


def iter_sites(target_model: nn.Module) -> list[tuple[str, DecomposedSite]]:
    """Iterate `(site_name, site)` pairs in target-model tree order."""
    out: list[tuple[str, DecomposedSite]] = []
    for name, mod in target_model.named_modules():
        if isinstance(mod, DecomposedLinear | DecomposedEmbedding):
            out.append((name, mod))
    return out
