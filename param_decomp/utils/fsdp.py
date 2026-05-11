"""FSDP wrapping for ComponentModel.

Wraps each `DecomposedSite` and each `TransformerBlock` in a separate FSDP unit.
The auto-wrap policy is what makes this work: by targeting the smallest units that
hold meaningful parameter counts, we get cross-rank parameter sharding without
ever crossing FSDP-unit boundaries inside a forward (which was the original
hook-related problem — see §5b of `fsdp_scaling_report.html`).

Use `use_orig_params=True` so that the optimizer can still be constructed from a
plain list of `nn.Parameter`s (the existing flow in `run_param_decomp.py`).

`MixedPrecision` here subsumes both the legacy `autocast_bf16` context manager and
the report's §7c "bf16 master weights" lever — activations are bf16, gradients are
all-reduced in bf16, and (by default) parameters stay in fp32. If we later need
bf16 master weights too for memory, that's `param_dtype=torch.bfloat16`.
"""

from collections.abc import Callable

import torch
from torch import nn
from torch.distributed.fsdp import (
    BackwardPrefetch,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)

from param_decomp.models.component_model import ComponentModel
from param_decomp.models.components import TransformerBlock
from param_decomp.models.decomposed_module import DecomposedEmbedding, DecomposedLinear


def _make_auto_wrap_policy() -> Callable[[nn.Module, bool, int], bool]:
    """Wrap each DecomposedSite and CI-fn TransformerBlock as its own FSDP unit.

    Frozen target submodules that aren't decomposed (ln_f, rms norms, the
    embedding layer if not decomposed, etc.) are left at the outermost wrap. Their
    parameter counts are small enough that the per-step all-gather cost of
    sharding them would exceed any saving.
    """

    def policy(module: nn.Module, recurse: bool, nonwrapped_numel: int) -> bool:
        del nonwrapped_numel  # FSDP passes this for size-based policies; we use type-based.
        # Keep recursing through composite containers so we find the targetable units.
        if recurse:
            return True
        return isinstance(module, DecomposedLinear | DecomposedEmbedding | TransformerBlock)

    return policy


def fsdp_wrap(
    component_model: ComponentModel,
    device_id: int,
    autocast_bf16: bool,
) -> FSDP:
    """FSDP-wrap a ComponentModel.

    Args:
        component_model: The ComponentModel to wrap. Must already be on the right device.
        device_id: Local rank's device index (passed straight to FSDP).
        autocast_bf16: If True, activations are bf16 and gradient reductions happen in bf16.
            Parameters stay fp32 by default — flip `param_dtype` here if we want bf16 masters.

    Returns:
        FSDP-wrapped model. The optimizer should be constructed from `wrapped.parameters()`
        (or, since we use `use_orig_params=True`, from any of the underlying parameter
        references that existed pre-wrap).
    """
    mixed_precision = (
        MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        if autocast_bf16
        else None
    )

    return FSDP(
        component_model,
        auto_wrap_policy=_make_auto_wrap_policy(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        device_id=device_id,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        use_orig_params=True,
    )
