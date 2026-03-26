"""Write-vector editing for SPD components.

Each SPD component is a rank-1 adapter: V[:, c] @ U[c, :]. The write vector (U row)
determines what the component contributes to the residual stream.

write_edit: directly replaces the U row, restores on exit.
train_write_delta: optimizes U[c] via gradient descent on CE + optional KL reg.

Usage:
    model, tok, config = load_model("wandb:goodfire/spd/s-55ea3f9b")

    # Analytical: replace U with scaled negated unembed
    unembed = model.target_model.lm_head.weight[token_id].detach()
    new_u = -3.0 * unembed / unembed.norm()
    with write_edit(model, "h.2.mlp.down_proj", 2359, new_u) as forward_fn:
        logits = forward_fn(tokens.unsqueeze(0))

    # Trained
    new_u = train_write_vector(model, "h.2.mlp.down_proj", 2359, train_seqs, lr=1e-3, n_steps=100)
    with write_edit(model, "h.2.mlp.down_proj", 2359, new_u) as forward_fn:
        logits = forward_fn(tokens.unsqueeze(0))
"""

from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from spd.models.component_model import ComponentModel


@contextmanager
def write_edit(
    model: ComponentModel,
    module_name: str,
    u_idx: int,
    new_u: Float[Tensor, " d_out"],
):
    """Replace U[u_idx] with new_u by patching the target weight matrix.

    The target linear's weight includes all components: W = Σ V[:,c] @ U[c,:] + delta.
    Changing U[c] by (new_u - old_u) changes W by V[:,c] ⊗ (new_u - old_u).
    """
    comp = model.components[module_name]
    old_u = comp.U.data[u_idx].clone()
    assert old_u.shape == new_u.shape

    # Find the target linear
    mod: Any = model.target_model
    for part in module_name.split("."):
        mod = getattr(mod, part)
    assert isinstance(mod, torch.nn.Linear)

    # ΔW = V[:,c] ⊗ (new_u - old_u)  — outer product, shape [d_in, d_out]
    # Weight is [d_out, d_in], so we need (new_u - old_u) ⊗ V[:,c]^T = [d_out] x [d_in]
    v_col = comp.V[:, u_idx].detach()  # [d_in]
    u_diff = (new_u - old_u).detach()  # [d_out]
    delta_w = u_diff.unsqueeze(1) * v_col.unsqueeze(0)  # [d_out, d_in]

    mod.weight.data.add_(delta_w)
    comp.U.data[u_idx] = new_u

    def forward_fn(tokens: Tensor) -> Tensor:
        return model(tokens)

    try:
        yield forward_fn
    finally:
        mod.weight.data.sub_(delta_w)
        comp.U.data[u_idx] = old_u


def train_write_vector(
    model: ComponentModel,
    module_name: str,
    u_idx: int,
    train_seqs: list[tuple[Int[Tensor, " seq"], list[int]]],
    lr: float,
    n_steps: int,
    kl_weight: float = 0.0,
) -> Float[Tensor, " d_out"]:
    """Optimize U[u_idx] via gradient descent. Returns the trained U row.

    Uses a forward hook to add V_col ⊗ (u_param - original_u) to the layer output,
    so gradients flow through u_param. The target weight matrix is not modified.
    """
    comp = model.components[module_name]
    original_u = comp.U[u_idx].detach().clone()
    v_col = comp.V[:, u_idx].detach()

    # Find target linear
    mod: Any = model.target_model
    for part in module_name.split("."):
        mod = getattr(mod, part)
    assert isinstance(mod, torch.nn.Linear)

    # Cache baselines before installing hook
    base_probs: list[Tensor] = []
    if kl_weight > 0:
        with torch.no_grad():
            for tokens, _ in train_seqs:
                base_probs.append(model(tokens.unsqueeze(0))[0].softmax(-1))

    u_param = original_u.clone().requires_grad_(True)
    optimizer = torch.optim.AdamW([u_param], lr=lr)

    def hook(_mod: torch.nn.Module, _inp: tuple[Any, ...], out: Tensor) -> Tensor:
        u_diff = u_param - original_u
        activation = _inp[0] @ v_col
        return out + activation.unsqueeze(-1) * u_diff.unsqueeze(0)

    handle = mod.register_forward_hook(hook)

    try:
        for _ in range(n_steps):
            for i, (tokens_mut, positions) in enumerate(train_seqs):
                logits = model(tokens_mut.unsqueeze(0))[0]
                pos_t = torch.tensor(positions, device=tokens_mut.device)
                ce = F.cross_entropy(logits[positions], tokens_mut[pos_t + 1])

                kl_loss = torch.tensor(0.0, device=tokens_mut.device)
                if kl_weight > 0:
                    probs = logits.softmax(-1)
                    kl = (probs * ((probs + 1e-10).log() - (base_probs[i] + 1e-10).log())).sum(-1)
                    fire_mask = torch.ones(
                        len(tokens_mut), dtype=torch.bool, device=tokens_mut.device
                    )
                    fire_mask[positions] = False
                    kl_loss = kl[fire_mask].mean()

                loss = ce + kl_weight * kl_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    finally:
        handle.remove()

    return u_param.detach()
