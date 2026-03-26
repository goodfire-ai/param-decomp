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
    """Replace U[u_idx] with new_u. Restores on exit. Yields forward_fn(tokens) -> logits."""
    comp = model.components[module_name]
    existing_u = comp.U[u_idx].clone()
    assert existing_u.shape == new_u.shape

    comp.U.data[u_idx] = new_u

    def forward_fn(tokens: Tensor) -> Tensor:
        return model(tokens)

    try:
        yield forward_fn
    finally:
        comp.U.data[u_idx] = existing_u


def train_write_vector(
    model: ComponentModel,
    module_name: str,
    u_idx: int,
    train_seqs: list[tuple[Int[Tensor, " seq"], list[int]]],
    lr: float,
    n_steps: int,
    kl_weight: float = 0.0,
) -> Float[Tensor, " d_out"]:
    """Optimize U[u_idx] via gradient descent. Returns the trained U row."""
    comp = model.components[module_name]
    original_u = comp.U[u_idx].detach().clone()
    original_requires_grad = comp.U.requires_grad

    # Cache baselines before any modification
    base_probs: list[Tensor] = []
    if kl_weight > 0:
        with torch.no_grad():
            for tokens, _ in train_seqs:
                base_probs.append(model(tokens.unsqueeze(0))[0].softmax(-1))

    # u_param is the trainable copy; we poke it into comp.U.data each step
    comp.U.requires_grad_(False)
    u_param = original_u.clone().requires_grad_(True)
    optimizer = torch.optim.AdamW([u_param], lr=lr)

    try:
        for _ in range(n_steps):
            for i, (tokens_mut, positions) in enumerate(train_seqs):
                comp.U.data[u_idx] = u_param
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
        comp.U.data[u_idx] = original_u
        comp.U.requires_grad_(original_requires_grad)

    return u_param.detach()
