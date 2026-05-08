"""Tests for parameter-component primitives in `param_decomp/models/components.py`.

Currently focused on the weight-delta rewrite from the scaling investigation
(`scaling_investigation_plan.md` Phase 4): `forward_with_target_weight` should be
algebraically identical to `forward(weight_delta_and_mask=...)` while avoiding
materialization of the (d_out, d_in) delta.
"""

import pytest
import torch

from param_decomp.models.components import LinearComponents


@pytest.mark.parametrize("with_mask", [True, False])
@pytest.mark.parametrize("with_bias", [True, False])
def test_weight_delta_rewrite_equivalence(with_mask: bool, with_bias: bool) -> None:
    torch.manual_seed(0)
    d_in, d_out, C = 32, 48, 16
    batch_dims = (4, 8)

    x = torch.randn(*batch_dims, d_in)
    W_target = torch.randn(d_out, d_in)
    delta_mask = torch.rand(*batch_dims)
    mask = torch.rand(*batch_dims, C) if with_mask else None
    bias = torch.randn(d_out) if with_bias else None

    comp = LinearComponents(C=C, d_in=d_in, d_out=d_out, bias=bias)

    weight_delta = W_target - comp.weight

    out_old = comp.forward(
        x,
        mask=mask,
        weight_delta_and_mask=(weight_delta, delta_mask),
    )
    out_new = comp.forward_with_target_weight(
        x,
        target_weight=W_target,
        mask=mask,
        weight_delta_mask=delta_mask,
    )

    torch.testing.assert_close(out_old, out_new, atol=1e-5, rtol=1e-5)


def test_weight_delta_rewrite_no_delta_path() -> None:
    """With weight_delta_mask=None the rewrite should match a no-delta forward."""
    torch.manual_seed(1)
    d_in, d_out, C = 16, 24, 8
    x = torch.randn(2, 5, d_in)
    W_target = torch.randn(d_out, d_in)
    mask = torch.rand(2, 5, C)

    comp = LinearComponents(C=C, d_in=d_in, d_out=d_out, bias=None)

    out_no_delta = comp.forward(x, mask=mask, weight_delta_and_mask=None)
    out_rewrite = comp.forward_with_target_weight(
        x, target_weight=W_target, mask=mask, weight_delta_mask=None
    )

    torch.testing.assert_close(out_no_delta, out_rewrite, atol=1e-6, rtol=1e-6)


def test_weight_delta_rewrite_grads_match() -> None:
    """Backward through the rewrite should give the same gradients on V/U as the
    materialized-delta path. This is what guarantees training behavior is preserved.
    """
    torch.manual_seed(2)
    d_in, d_out, C = 16, 24, 8
    batch_dims = (3, 4)
    x = torch.randn(*batch_dims, d_in)
    W_target = torch.randn(d_out, d_in)
    delta_mask = torch.rand(*batch_dims)
    mask = torch.rand(*batch_dims, C)

    comp_a = LinearComponents(C=C, d_in=d_in, d_out=d_out, bias=None)
    comp_b = LinearComponents(C=C, d_in=d_in, d_out=d_out, bias=None)
    comp_b.V.data = comp_a.V.data.clone()
    comp_b.U.data = comp_a.U.data.clone()

    out_old = comp_a.forward(
        x,
        mask=mask,
        weight_delta_and_mask=(W_target - comp_a.weight, delta_mask),
    )
    out_old.sum().backward()

    out_new = comp_b.forward_with_target_weight(
        x, target_weight=W_target, mask=mask, weight_delta_mask=delta_mask
    )
    out_new.sum().backward()

    assert comp_a.V.grad is not None and comp_b.V.grad is not None
    assert comp_a.U.grad is not None and comp_b.U.grad is not None
    torch.testing.assert_close(comp_a.V.grad, comp_b.V.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(comp_a.U.grad, comp_b.U.grad, atol=1e-5, rtol=1e-5)
