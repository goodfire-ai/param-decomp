"""fp8 (e4m3) matmul for the FROZEN target-model weights in the masked forward.

The 2-/3-pool masked forward spends most of its time in `F.linear` over the frozen
suffix weights (attention q/k/v/o and the non-decomposed MLPs). Those weights never train,
so they can be pre-quantized once to e4m3 and matmul'd on the B200 fp8 tensor cores via
`torch._scaled_mm`. Activations are dynamically per-tensor quantized each call.

Scope is deliberately narrow: ONLY frozen target weights. Trained V/U components, the CI
fn, and the weight-delta path stay bf16 — fp8 here would corrupt gradients.

Tensorwise scaling (one f32 scale per tensor) is used: `_scaled_mm` rowwise/blockwise
measured slower on these GPT-2-XL matmul shapes (see jax_spike microbench). Enabled by
`componentize_*` only when `PD_FP8_FROZEN=1`.
"""

import os
from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.autograd.function import FunctionCtx

# Buffers are registered via register_buffer in __init__ (pyright doesn't model that as an
# init), matching the annotation pattern used across the vendored component modules.
# pyright: reportUninitializedInstanceVariable=false

_E4M3_MAX = 448.0
_FP8 = torch.float8_e4m3fn


def fp8_frozen_enabled() -> bool:
    return os.environ.get("PD_FP8_FROZEN", "").strip() in ("1", "true", "yes")


def _quantize_tensorwise(t: Float[Tensor, "rows cols"]) -> tuple[Tensor, Tensor]:
    """Cast `t` to e4m3 with a single f32 amax/448 scale. Returns (fp8 tensor, f32 scale [1,1])."""
    amax = t.abs().amax().clamp(min=1e-12).float()
    scale = (amax / _E4M3_MAX).reshape(1, 1)
    return (t.float() / scale).to(_FP8), scale


def _fp8_matmul_forward(
    x2d: Float[Tensor, "tokens d_in"],
    weight_fp8: Float[Tensor, "d_out d_in"],
    weight_scale: Float[Tensor, "1 1"],
    bias: Float[Tensor, "... d_out"] | None,
) -> Float[Tensor, "tokens d_out"]:
    """fp8 e4m3 `_scaled_mm`: dynamic per-tensor activation scale, frozen pre-quantized weight.
    `weight_fp8` is `[d_out, d_in]` contiguous; `.t()` is the col-major `mat2` `_scaled_mm` wants."""
    amax = x2d.abs().amax().clamp(min=1e-12).float()
    x_scale = (amax / _E4M3_MAX).reshape(1, 1)
    xq = (x2d.float() / x_scale).to(_FP8)
    return torch._scaled_mm(
        xq,
        weight_fp8.t(),
        scale_a=x_scale,
        scale_b=weight_scale,
        bias=bias,
        out_dtype=torch.bfloat16,
    )


class _Fp8FrozenMatmul(torch.autograd.Function):
    """fp8 forward over a FROZEN weight, with a bf16 input-gradient backward.

    `_scaled_mm` has no autograd backward, but the masked recon backprops through the
    activation `x` (the weight is frozen, so no weight grad is needed). Forward runs the fp8
    matmul; backward dequantizes the frozen e4m3 weight to bf16 and computes
    `grad_x = grad_out @ W` — exact for a frozen weight, and the only grad the graph needs.
    """

    @staticmethod
    def forward(  # pyright: ignore[reportImplicitOverride]
        ctx: FunctionCtx,
        x2d: Tensor,
        weight_fp8: Tensor,
        weight_scale: Tensor,
        bias: Tensor | None,
    ) -> Tensor:
        ctx.save_for_backward(weight_fp8, weight_scale)
        return _fp8_matmul_forward(x2d, weight_fp8, weight_scale, bias)

    @staticmethod
    def backward(  # pyright: ignore[reportImplicitOverride, reportIncompatibleMethodOverride]
        ctx: FunctionCtx,
        grad_out: Tensor,
    ) -> tuple[Tensor | None, None, None, None]:
        weight_fp8, weight_scale = ctx.saved_tensors  # pyright: ignore[reportAttributeAccessIssue]
        # Dequant frozen e4m3 weight to bf16: grad_x = grad_out @ W, W is [d_out, d_in].
        weight = (weight_fp8.to(grad_out.dtype)) * weight_scale.to(grad_out.dtype)
        grad_x = grad_out @ weight
        return grad_x, None, None, None


def _fp8_linear(
    x: Float[Tensor, "... d_in"],
    weight_fp8: Float[Tensor, "d_out d_in"],
    weight_scale: Float[Tensor, "1 1"],
    bias: Float[Tensor, "... d_out"] | None,
) -> Float[Tensor, "... d_out"]:
    """Differentiable fp8 frozen linear: `x @ weight.T` in e4m3, input-gradient backward only."""
    lead = x.shape[:-1]
    x2d = x.reshape(-1, x.shape[-1])
    out = _Fp8FrozenMatmul.apply(x2d, weight_fp8, weight_scale, bias)
    return out.reshape(*lead, out.shape[-1])


class Fp8FrozenLinear(nn.Module):
    """Drop-in for a frozen `nn.Linear`: pre-quantized e4m3 weight + fp8 `_scaled_mm` forward.

    Holds the weight pre-quantized contiguous in `[d_out, d_in]` plus its f32 scale, both as
    buffers so `.to(device)` and `.compile()` carry them. Bias stays bf16/f32 and is fused
    into `_scaled_mm`.
    """

    weight_fp8: Float[Tensor, "d_out d_in"]
    weight_scale: Float[Tensor, "1 1"]
    bias: Float[Tensor, "... d_out"] | None

    def __init__(self, weight: Float[Tensor, "d_out d_in"], bias: Tensor | None):
        super().__init__()
        wq, scale = _quantize_tensorwise(weight)
        self.register_buffer("weight_fp8", wq.contiguous())
        self.register_buffer("weight_scale", scale)
        self.register_buffer("bias", bias)

    @override
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return _fp8_linear(x, self.weight_fp8, self.weight_scale, self.bias)


def fp8_frozen_target_forward(
    x: Float[Tensor, "... d_in"],
    target_weight_fp8: Float[Tensor, "d_out d_in"],
    target_weight_scale: Float[Tensor, "1 1"],
    bias: Float[Tensor, "... d_out"] | None,
) -> Float[Tensor, "... d_out"]:
    """fp8 frozen-target matmul for a `ComponentLinear`, from its pre-quantized buffers."""
    return _fp8_linear(x, target_weight_fp8, target_weight_scale, bias)


def convert_frozen_linears_to_fp8(module: nn.Module) -> int:
    """Swap every frozen `nn.Linear` leaf under `module` for an `Fp8FrozenLinear`.

    The decomposed sites are `_ComponentModule`s (not `nn.Linear` subclasses), so the
    `nn.Linear` filter already excludes them — they own their own frozen-weight path.
    Returns the count swapped. Call this on the SUFFIX blocks the masked forward runs (blocks
    `[decomposition_start_layer:]`), not the whole model — fp8'ing the frozen prefix would
    shift the cached clean residual that the clean target also consumes."""
    swapped = 0
    for parent in module.modules():
        for attr, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                bias = child.bias.data if child.bias is not None else None  # pyright: ignore[reportUnnecessaryComparison]
                setattr(parent, attr, Fp8FrozenLinear(child.weight.data, bias))
                swapped += 1
    return swapped
