"""Write-vector editing for SPD components.

Each SPD component is a rank-1 adapter: V[:, c] @ U[c, :]. The write vector (U row)
determines what the component contributes to the residual stream.

u_replaced: context manager that computes a weight-space delta from a U-vector
replacement and applies it as a forward hook on the target linear layer.

Usage:
    model, tok, config = load_model("wandb:goodfire/spd/s-55ea3f9b")

    unembed = model.target_model.lm_head.weight[token_id].detach()
    new_u = -3.0 * unembed / unembed.norm()
    with u_replaced(model, "h.2.mlp.down_proj", 2359, new_u) as forward_fn:
        logits = forward_fn(tokens.unsqueeze(0))
"""

from contextlib import contextmanager
from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor

from spd.models.component_model import ComponentModel


def _resolve_linear(root: torch.nn.Module, path: str) -> torch.nn.Linear:
    mod: Any = root
    for part in path.split("."):
        assert hasattr(mod, part), (
            f"{type(mod).__name__} has no attribute {part!r} (resolving {path!r})"
        )
        mod = getattr(mod, part)
    assert isinstance(mod, torch.nn.Linear), (
        f"{path!r} resolved to {type(mod).__name__}, expected Linear"
    )
    return mod


@contextmanager
def u_replaced(
    model: ComponentModel,
    module_name: str,
    u_idx: int,
    new_u: Float[Tensor, " d_out"],
):
    """Replace U[u_idx] with new_u via a weight-space delta hook on the target layer.

    The weight delta is: delta_W = outer(new_u - old_u, V[:, u_idx]).
    Applied as a forward hook so the edit goes through the same code path as the
    unmodified model (and the LoRA baseline).
    """
    comp = model.components[module_name]
    old_u = comp.U.data[u_idx]
    assert old_u.shape == new_u.shape

    delta_u = (new_u - old_u).float()
    v = comp.V.data[:, u_idx].float()
    delta_W = torch.outer(delta_u, v)  # [d_out, d_in]

    linear = _resolve_linear(model.target_model, module_name)

    def hook(_mod: torch.nn.Module, _inp: tuple[Any, ...], out: Tensor) -> Tensor:
        return out + (_inp[0].float() @ delta_W.T).to(out.dtype)

    handle = linear.register_forward_hook(hook)
    try:
        yield model
    finally:
        handle.remove()
