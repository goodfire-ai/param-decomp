from typing import Any, Literal, override

import torch
from torch import Tensor
from torch.autograd import Function

SigmoidType = Literal[
    "normal",
    "hard",
    "leaky_hard",
    "upper_leaky_hard",
    "lower_leaky_hard",
    "swish_hard",
]


class LowerLeakyHardSigmoidFunction(Function):
    @override
    @staticmethod
    def forward(ctx: Any, x: Tensor, alpha: float = 0.01) -> Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return torch.clamp(x, min=0, max=1)

    @override
    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor, None]:
        grad_output = grad_outputs[0]  # Since we only have a single input to the forward method
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha

        # Gradient as if forward pass was alpha * x for x<=0 when the gradient is negative
        grad_input = torch.where(
            x <= 0,
            torch.where(grad_output < 0, alpha * grad_output, torch.zeros_like(grad_output)),
            torch.where(x <= 1, grad_output, torch.zeros_like(grad_output)),
        )

        return grad_input, None  # None for alpha gradient since it's not a tensor


def normal_sigmoid(x: Tensor) -> Tensor:
    return torch.sigmoid(x)


def hard_sigmoid(x: Tensor) -> Tensor:
    return torch.clamp(x, min=0, max=1)


def leaky_hard_sigmoid(x: Tensor, alpha: float = 0.01) -> Tensor:
    return torch.where(x > 0, torch.clamp(x, max=1), alpha * x)


def upper_leaky_hard_sigmoid(x: Tensor, alpha: float = 0.01) -> Tensor:
    return torch.where(x > 1, 1 + alpha * (x - 1), torch.clamp(x, min=0, max=1))


def lower_leaky_hard_sigmoid(x: Tensor, alpha: float = 0.01) -> Tensor:
    return LowerLeakyHardSigmoidFunction.apply(x, alpha)  # pyright: ignore[reportReturnType]


def swish(x: Tensor, beta: float = 1.0) -> Tensor:
    return x * torch.sigmoid(beta * x)


def upside_down_swish(x: Tensor, beta: float = 1.0) -> Tensor:
    return x * torch.sigmoid(beta * -x)


def swish_hard_sigmoid(
    x: Tensor, beta: float = 10.0, scale: float = 0.5, xshift: float = 0.5, yshift: float = 0.5
) -> Tensor:
    """A sigmoid function that uses swish functions at the boundaries.

    As the `beta' parameter increases, the function approximates a hard sigmoid.
    """
    x = x - xshift
    return (
        yshift
        + (upside_down_swish(x - scale, beta) - swish(x, beta))
        + (swish(x + scale, beta) - upside_down_swish(x, beta))
    )


SIGMOID_TYPES = {
    "normal": normal_sigmoid,
    "hard": hard_sigmoid,
    "leaky_hard": leaky_hard_sigmoid,
    "upper_leaky_hard": upper_leaky_hard_sigmoid,
    "lower_leaky_hard": lower_leaky_hard_sigmoid,
    "swish_hard": swish_hard_sigmoid,
}
