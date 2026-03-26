"""Rank-1 LoRA baseline for comparison with SPD component editing.

CE loss at fire positions (teach it to predict target token) + KL loss at all
other positions (preserve original predictions). Single forward pass per step.

Usage:
    with LoRATrainer(model.target_model, "h.2.mlp.down_proj", kl_weight=10.0, lr=1e-3) as lora:
        for _ in range(300):
            lora.train_step(tokens, baselines, fire_mask, pad_mask)
        result = eval_edit(lora.forward)
"""

from typing import Any

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor

from spd.editing.component_trainer import _resolve_linear


class LoRATrainer:
    """Rank-1 LoRA on a single linear layer. Context manager — hook removed on exit."""

    def __init__(
        self,
        target_model: torch.nn.Module,
        layer_path: str,
        lr: float,
        kl_weight: float,
    ):
        self.target_model = target_model
        self.target_model.eval()
        self.linear = _resolve_linear(target_model, layer_path)

        d_out, d_in = int(self.linear.weight.shape[0]), int(self.linear.weight.shape[1])
        self.A = torch.randn(1, d_in, device="cuda") * 0.01
        self.B = torch.zeros(d_out, 1, device="cuda")
        self.A.requires_grad = True
        self.B.requires_grad = True

        self._hook = self.linear.register_forward_hook(self._fwd_hook)
        self.optimizer = torch.optim.AdamW([self.A, self.B], lr=lr)
        self.kl_weight = kl_weight

    def __enter__(self) -> "LoRATrainer":
        return self

    def __exit__(self, *_: object) -> None:
        self._hook.remove()

    def _fwd_hook(self, _mod: torch.nn.Module, _inp: tuple[Any, ...], out: Tensor) -> Tensor:
        return out + (_inp[0] @ self.A.T) @ self.B.T

    def forward(self, tokens: Tensor) -> Tensor:
        return self.target_model(tokens)[0]

    def train_step(
        self,
        tokens: Int[Tensor, "B S"],
        baselines: Float[Tensor, "B S V"],
        fire_mask: Bool[Tensor, "B S"],
        pad_mask: Bool[Tensor, "B S"],
    ) -> float:
        """One training step. All args are pre-padded tensors."""
        logits = self.forward(tokens)

        # CE at fire positions: predict next token
        fire_idx = fire_mask.nonzero(as_tuple=False)
        ce = F.cross_entropy(
            logits[fire_idx[:, 0], fire_idx[:, 1]],
            tokens[fire_idx[:, 0], fire_idx[:, 1] + 1],
        )

        # KL at non-fire, non-pad positions
        kl_loss = torch.tensor(0.0, device=tokens.device)
        if self.kl_weight > 0:
            kl_mask = pad_mask & ~fire_mask
            probs = logits.softmax(-1)
            kl_per_pos = (probs * ((probs + 1e-10).log() - (baselines + 1e-10).log())).sum(-1)
            kl_loss = kl_per_pos[kl_mask].mean()

        loss = ce + self.kl_weight * kl_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
