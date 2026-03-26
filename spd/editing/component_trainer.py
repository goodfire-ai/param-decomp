"""Write-vector editing for SPD components.

Each SPD component is a rank-1 adapter: V[:, c] @ U[c, :]. The write vector (U row)
determines what the component contributes to the residual stream.

The delta formulation: a forward hook on the target linear adds
    (x @ V_col) * U_delta
where V_col is the component's read vector (fixed) and U_delta is the learned or
analytical perturbation.

Usage:
    model, tok, config = load_model("wandb:goodfire/spd/s-55ea3f9b")

    # Analytical
    unembed = model.target_model.lm_head.weight[token_id].detach()
    u_delta = -3.0 * unembed / unembed.norm()
    with write_edit(model, "h.2.mlp.down_proj:2359", u_delta) as forward_fn:
        logits = forward_fn(tokens.unsqueeze(0))

    # Trained
    u_delta = train_write_delta(model, "h.2.mlp.down_proj:2359", train_seqs, lr=1e-3, n_steps=100)
    with write_edit(model, "h.2.mlp.down_proj:2359", u_delta) as forward_fn:
        logits = forward_fn(tokens.unsqueeze(0))
"""

from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from spd.editing.utils import parse_component_key
from spd.models.component_model import ComponentModel


def _resolve_hook_args(
    model: ComponentModel, comp_key: str
) -> tuple[torch.nn.Linear, Float[Tensor, " d_in"]]:
    module_path, cidx = parse_component_key(comp_key)
    v_col = model.components[module_path].V[:, cidx].detach()
    mod: Any = model.target_model
    for part in module_path.split("."):
        mod = getattr(mod, part)
    assert isinstance(mod, torch.nn.Linear)
    return mod, v_col


@contextmanager
def write_edit(
    model: ComponentModel,
    comp_key: str,
    u_delta: Float[Tensor, " d_out"],
):
    """Context manager applying a write-vector delta. Yields forward_fn(tokens) -> logits."""
    linear, v_col = _resolve_hook_args(model, comp_key)

    def hook(_mod: torch.nn.Module, _inp: tuple[Any, ...], out: Tensor) -> Tensor:
        activation = _inp[0] @ v_col
        return out + activation.unsqueeze(-1) * u_delta.unsqueeze(0)

    handle = linear.register_forward_hook(hook)

    def forward_fn(tokens: Tensor) -> Tensor:
        return model(tokens)

    try:
        yield forward_fn
    finally:
        handle.remove()


def train_write_delta(
    model: ComponentModel,
    comp_key: str,
    train_seqs: list[tuple[Int[Tensor, " seq"], list[int]]],
    lr: float,
    n_steps: int,
) -> Float[Tensor, " d_out"]:
    """Train a write-vector delta. Returns the learned U delta tensor."""
    linear, v_col = _resolve_hook_args(model, comp_key)

    d_out = int(linear.weight.shape[0])
    u_delta = torch.zeros(d_out, device=v_col.device, requires_grad=True)

    def hook(_mod: torch.nn.Module, _inp: tuple[Any, ...], out: Tensor) -> Tensor:
        activation = _inp[0] @ v_col
        return out + activation.unsqueeze(-1) * u_delta.unsqueeze(0)

    handle = linear.register_forward_hook(hook)
    optimizer = torch.optim.AdamW([u_delta], lr=lr)

    try:
        for _ in range(n_steps):
            for tokens_mut, positions in train_seqs:
                logits = model(tokens_mut.unsqueeze(0))
                pos_t = torch.tensor(positions, device=tokens_mut.device)
                loss = F.cross_entropy(logits[0, positions], tokens_mut[pos_t + 1])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    finally:
        handle.remove()

    return u_delta.detach()
