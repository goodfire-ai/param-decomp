"""Write-vector editing for SPD components.

Each SPD component is a rank-1 adapter: V[:, c] @ U[c, :]. The write vector (U row)
determines what the component contributes to the residual stream. This module provides
functions to train or analytically set write vectors, producing a U delta tensor that
can be applied via a forward hook.

The delta formulation: with the hook installed, forward pass computes
    target_model(x) + (x @ V_col) * U_delta
where V_col is the component's read vector (fixed) and U_delta is the learned or
analytical perturbation. No weight snapshots needed.

Usage:
    em, tok = EditableModel.from_wandb("wandb:goodfire/spd/s-55ea3f9b")

    # Analytical: set U to negated unembed direction
    unembed = em.model.target_model.lm_head.weight[token_id].detach()
    u_delta = -3.0 * unembed / unembed.norm()
    with write_edit(em.model, "h.2.mlp.down_proj:2359", u_delta) as forward_fn:
        logits = forward_fn(tokens.unsqueeze(0))

    # Trained:
    u_delta = train_write_delta(em.model, "h.2.mlp.down_proj:2359", train_seqs, lr=1e-3)
    with write_edit(em.model, "h.2.mlp.down_proj:2359", u_delta) as forward_fn:
        logits = forward_fn(tokens.unsqueeze(0))
"""

from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from spd.editing._editing import parse_component_key
from spd.models.component_model import ComponentModel


def _get_linear(model: ComponentModel, module_path: str) -> torch.nn.Linear:
    mod: Any = model.target_model
    for part in module_path.split("."):
        mod = getattr(mod, part)
    assert isinstance(mod, torch.nn.Linear)
    return mod


@contextmanager
def write_edit(
    model: ComponentModel,
    comp_key: str,
    u_delta: Float[Tensor, " d_out"],
):
    """Context manager that applies a write-vector delta to a component.

    Yields a forward_fn(tokens) -> logits that runs the target model with the
    rank-1 perturbation: output += (x @ V_col) * U_delta.
    """
    module_path, cidx = parse_component_key(comp_key)
    v_col = model.components[module_path].V[:, cidx].detach()
    linear = _get_linear(model, module_path)

    def hook(_mod: torch.nn.Module, _inp: tuple[Any, ...], out: Tensor) -> Tensor:
        x = _inp[0]  # [..., d_in]
        activation = x @ v_col  # [...] — scalar per position
        return out + activation.unsqueeze(-1) * u_delta.unsqueeze(0)

    handle = linear.register_forward_hook(hook)

    def forward_fn(tokens: Tensor) -> Tensor:
        return model._extract_output(model.target_model(tokens))

    try:
        yield forward_fn
    finally:
        handle.remove()


def train_write_delta(
    model: ComponentModel,
    comp_key: str,
    train_seqs: list[tuple[Int[Tensor, " seq"], list[int]]],
    lr: float = 1e-3,
    n_steps: int = 100,
) -> Float[Tensor, " d_out"]:
    """Train a write-vector delta for a component. Returns the learned U delta.

    Does not mutate the model. The delta can be applied with write_edit().
    """
    module_path, cidx = parse_component_key(comp_key)
    v_col = model.components[module_path].V[:, cidx].detach()
    linear = _get_linear(model, module_path)

    d_out = int(linear.weight.shape[0])
    u_delta = torch.zeros(d_out, device=v_col.device)
    u_delta.requires_grad = True

    def hook(_mod: torch.nn.Module, _inp: tuple[Any, ...], out: Tensor) -> Tensor:
        x = _inp[0]
        activation = x @ v_col
        return out + activation.unsqueeze(-1) * u_delta.unsqueeze(0)

    handle = linear.register_forward_hook(hook)
    optimizer = torch.optim.AdamW([u_delta], lr=lr)

    def forward(tokens: Tensor) -> Tensor:
        return model._extract_output(model.target_model(tokens))

    try:
        for _ in range(n_steps):
            for tokens_mut, positions in train_seqs:
                logits = forward(tokens_mut.unsqueeze(0))
                pos_t = torch.tensor(positions, device=tokens_mut.device)
                loss = F.cross_entropy(logits[0, positions], tokens_mut[pos_t + 1])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    finally:
        handle.remove()

    return u_delta.detach()
