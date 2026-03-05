"""Verify that ∂|y|/∂x = sign(y) · ∂y/∂x for a scalar y, even through nonlinearities.

The chain rule: ∂|y|/∂x = (d|y|/dy) · (∂y/∂x) = sign(y) · ∂y/∂x

This holds regardless of what nonlinear computation sits between x and y,
because ∂y/∂x already accounts for all intermediate nonlinearities.
The sign(y) factor is just the outermost link in the chain.
"""

import torch
from torch import nn


def test_simple_linear():
    """Linear: y = Wx, trivial case."""
    x = torch.randn(5, requires_grad=True)
    W = torch.randn(3, 5)
    y_vec = W @ x
    y = y_vec[1]  # pick one scalar

    grad = torch.autograd.grad(y, x, retain_graph=True)[0]
    grad_abs = torch.autograd.grad(y.abs(), x, retain_graph=True)[0]
    grad_trick = y.sign() * grad

    assert torch.allclose(grad_abs, grad_trick, atol=1e-7), (
        f"FAIL: {(grad_abs - grad_trick).abs().max()}"
    )
    print(f"  linear: max diff = {(grad_abs - grad_trick).abs().max():.2e} ✓")


def test_deep_nonlinear():
    """Deep net with ReLU, tanh, and GELU — representative of a transformer."""
    torch.manual_seed(42)
    net = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.Tanh(),
        nn.Linear(16, 16),
        nn.GELU(),
        nn.Linear(16, 4),
    )
    x = torch.randn(8, requires_grad=True)
    y_vec = net(x)
    y = y_vec[2]  # scalar output

    grad = torch.autograd.grad(y, x, retain_graph=True)[0]
    grad_abs = torch.autograd.grad(y.abs(), x, retain_graph=True)[0]
    grad_trick = y.sign() * grad

    assert torch.allclose(grad_abs, grad_trick, atol=1e-6), (
        f"FAIL: {(grad_abs - grad_trick).abs().max()}"
    )
    print(
        f"  deep nonlinear (y={y.item():.4f}): max diff = {(grad_abs - grad_trick).abs().max():.2e} ✓"
    )


def test_negative_target():
    """Ensure it works when y < 0 (sign flips the gradient)."""
    torch.manual_seed(99)
    net = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1))
    # Find an input that gives negative output
    for _seed in range(200):
        x = torch.randn(4, requires_grad=True)
        y = net(x).squeeze()
        if y.item() < -0.1:
            break
    assert y.item() < 0, "Couldn't find negative output"

    grad = torch.autograd.grad(y, x, retain_graph=True)[0]
    grad_abs = torch.autograd.grad(y.abs(), x, retain_graph=True)[0]
    grad_trick = y.sign() * grad

    assert torch.allclose(grad_abs, grad_trick, atol=1e-6), (
        f"FAIL: {(grad_abs - grad_trick).abs().max()}"
    )
    print(
        f"  negative target (y={y.item():.4f}): max diff = {(grad_abs - grad_trick).abs().max():.2e} ✓"
    )


def test_multiple_inputs():
    """Multiple input tensors (mirrors the app's in_post_detaches list)."""
    torch.manual_seed(7)
    x1 = torch.randn(3, 4, requires_grad=True)
    x2 = torch.randn(3, 4, requires_grad=True)

    # Nonlinear function of both inputs
    h = torch.relu(x1) + torch.tanh(x2)
    y = (h @ torch.randn(4, 1)).sum()  # scalar

    grads = torch.autograd.grad(y, [x1, x2], retain_graph=True)
    grads_abs = torch.autograd.grad(y.abs(), [x1, x2], retain_graph=True)

    for i, (g, g_abs) in enumerate(zip(grads, grads_abs, strict=True)):
        g_trick = y.sign() * g
        assert torch.allclose(g_abs, g_trick, atol=1e-6), (
            f"FAIL input {i}: {(g_abs - g_trick).abs().max()}"
        )

    print(f"  multiple inputs (y={y.item():.4f}): all match ✓")


def test_sum_of_abs_DOES_NOT_work():
    """Show that the trick FAILS for sum-of-abs (dataset attributions case).

    ∂(Σ|y_i|)/∂x ≠ sign(Σy_i) · ∂(Σy_i)/∂x
    because each y_i has a different sign.
    """
    torch.manual_seed(42)
    x = torch.randn(4, requires_grad=True)
    W = torch.randn(3, 4)
    y_vec = W @ x  # [3]

    target_signed = y_vec.sum()
    target_abs = y_vec.abs().sum()

    grad_signed = torch.autograd.grad(target_signed, x, retain_graph=True)[0]
    grad_abs = torch.autograd.grad(target_abs, x, retain_graph=True)[0]

    # The WRONG trick: use sign of the sum
    grad_wrong = target_signed.sign() * grad_signed

    # The correct per-element version
    grad_correct = sum(
        y_vec[i].sign() * torch.autograd.grad(y_vec[i], x, retain_graph=True)[0]
        for i in range(len(y_vec))
    )

    wrong_diff = (grad_abs - grad_wrong).abs().max()
    correct_diff = (grad_abs - grad_correct).abs().max()
    print(
        f"  sum-of-abs: wrong trick diff = {wrong_diff:.4f}, correct per-element diff = {correct_diff:.2e}"
    )
    assert wrong_diff > 0.01, "Expected the wrong trick to fail for sum-of-abs"
    assert correct_diff < 1e-6, "Per-element version should match"
    print("  → confirms: trick works for scalar y, NOT for sum-of-abs ✓")


if __name__ == "__main__":
    print("Testing ∂|y|/∂x = sign(y) · ∂y/∂x for scalar y:\n")
    test_simple_linear()
    test_deep_nonlinear()
    test_negative_target()
    test_multiple_inputs()
    print()
    print("Testing that the trick does NOT work for sum-of-abs:\n")
    test_sum_of_abs_DOES_NOT_work()
    print("\nAll tests passed.")
