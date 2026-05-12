"""FSDP2-based wrapping for ComponentModel.

Uses `torch.distributed.fsdp.fully_shard` (FSDP2 — the per-module composable API)
rather than `FullyShardedDataParallel` (FSDP1). FSDP1's bookkeeping doesn't survive
SPD's multi-forward training step (we call `wrapped_model(...)` 5-7 times per step
across the target-only forward, PPGD warmups, and each loss flavor's forward),
producing `setStorage out of bounds` failures at backward. FSDP2 is built for that
pattern and keeps the module tree intact (no `_fsdp_wrapped_module` indirection).

Wrap layout: each `DecomposedSite`, each `TransformerBlock` (target + CI fn), and
`GlobalSharedTransformerCiFn` itself become their own FSDP units. The root unit owns
the remaining params (target.wte, target.ln_f, target.lm_head, etc.). Frozen target
submodules that aren't decomposed stay in the root unit.

Mixed precision: bf16 reduce + bf16 buffers, params stay fp32. Subsumes the legacy
`autocast_bf16` context manager.
"""

from __future__ import annotations

from torch import Tensor, nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from param_decomp.models.component_model import ComponentModel
from param_decomp.models.components import GlobalSharedTransformerCiFn, TransformerBlock
from param_decomp.models.decomposed_module import DecomposedEmbedding, DecomposedLinear


def calc_weight_deltas_full(component_model: ComponentModel) -> dict[str, Tensor]:
    """Compute weight deltas as regular Tensors under FSDP2.

    Mirrors what the training loop does outside any forward: gather each wrapped
    site's V/U via `.unshard()`, read deltas, materialize to a regular Tensor via
    `.full_tensor()`, then `.reshard()`. Downstream code (loss / eval metrics)
    feeds these into `bmm` against regular Tensors, which DTensor's dispatcher
    refuses — so the full_tensor() conversion is the load-bearing step.

    Use this only outside an FSDP-wrapped forward. Inside a forward each site's
    params are already unsharded by FSDP's pre-forward hook.
    """
    sites_to_unshard = [m for m in component_model.modules() if hasattr(m, "unshard")]
    for s in sites_to_unshard:
        s.unshard()  # pyright: ignore[reportCallIssue]
    try:
        weight_deltas: dict[str, Tensor] = {}
        for k, v in component_model.calc_weight_deltas().items():
            v = v.detach()
            v = v.full_tensor() if hasattr(v, "full_tensor") else v.clone()
            weight_deltas[k] = v
    finally:
        for s in sites_to_unshard:
            s.reshard()  # pyright: ignore[reportCallIssue]
    return weight_deltas


def _untie_target_weights_(target_model: nn.Module) -> None:
    """Break parameter sharing across target modules so FSDP doesn't see two paths to
    the same storage.

    LlamaSimpleMLP ties `wte.weight = lm_head.weight`. FSDP1 and FSDP2 both get
    confused by tied params and end up freeing the underlying storage from one
    reference while the other still uses it. Since the target is frozen, the two
    copies stay identical for the rest of training, so we can safely clone-detach
    to break the alias.
    """
    seen: dict[int, list[tuple[nn.Module, str]]] = {}
    for module in target_model.modules():
        for name, param in module.named_parameters(recurse=False):
            seen.setdefault(id(param), []).append((module, name))

    for refs in seen.values():
        if len(refs) <= 1:
            continue
        for parent, attr in refs[1:]:
            old = getattr(parent, attr)
            setattr(
                parent,
                attr,
                nn.Parameter(old.detach().clone(), requires_grad=old.requires_grad),
            )


def fsdp_wrap(
    component_model: ComponentModel,
    device_id: int,
    autocast_bf16: bool,
) -> ComponentModel:
    """Apply FSDP2 sharding in place to a ComponentModel and return it.

    Walks the model tree bottom-up calling `fully_shard` on each unit we want to
    treat as its own FSDP shard group. The returned object is the same Python
    object as the input — `fully_shard` mutates module __class__ metadata to weave
    in the FSDP hooks but doesn't replace anything in the tree.

    Args:
        component_model: ComponentModel built with fused-decomposition sites, on the
            right device, in eval mode for the target.
        device_id: Local rank's device index.
        autocast_bf16: If True, params are stored as bf16 shards and reductions use bf16;
            outputs remain fp32. Halves sharded-param memory and allreduce bandwidth.
    """
    del device_id  # FSDP2 picks up the device from the module + current torch.cuda.device

    _untie_target_weights_(component_model)

    # output_dtype=fp32 is critical (not just for precision): it forces FSDP2 to cast each
    # wrapped unit's output, which materializes a regular `Tensor` from the unsharded DTensor.
    # Without this, the wrapped forward returns a DTensor and downstream layers (e.g. attn
    # bmm in the target's CausalSelfAttention) get mixed DTensor+Tensor inputs that DTensor's
    # dispatcher refuses.
    #
    # When autocast_bf16=True: store and reduce params as bf16 (halves sharded param memory
    # and allreduce bandwidth) while keeping outputs fp32. This avoids the dtype-mismatch
    # that afflicts output_dtype=bfloat16 (frozen target layers expect fp32 inputs). Compute
    # inside each FSDP unit runs in mixed precision (bf16 param × fp32 input → fp32 accum).
    import torch as _torch

    param_dtype = _torch.bfloat16 if autocast_bf16 else _torch.float32
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=param_dtype,
        output_dtype=_torch.float32,
    )

    # Wrap only the inner units that hold the bulk of trainable params: DecomposedSites
    # (V/U + the wrapped frozen target submodule) and CI-fn TransformerBlocks. Wrap the
    # CI fn itself so its small input/output projectors get gathered when ci_fn(...) is
    # called outside any forward.
    #
    # We deliberately do NOT wrap `component_model` at the root level. With the root wrapped,
    # the remaining target submodules (wte / ln_f / lm_head) become DTensor-owned and their
    # outputs (or downstream values) leak DTensor into the target's forward — which then
    # mixes with regular Tensor outputs from FSDP-wrapped DecomposedSites and trips the
    # DTensor dispatcher in attention's bmm. Leaving the root un-wrapped keeps those small
    # frozen params as regular nn.Parameters; their memory cost is bounded (a few hundred MB
    # at the 4B-target scale) and they don't update.
    for module in component_model.modules():
        if isinstance(module, DecomposedLinear | DecomposedEmbedding | TransformerBlock):
            fully_shard(module, mp_policy=mp_policy)

    for submodule in component_model.ci_fn.modules():
        if isinstance(submodule, GlobalSharedTransformerCiFn):
            fully_shard(submodule, mp_policy=mp_policy)

    return component_model
