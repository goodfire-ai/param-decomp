"""Rank-1 LoRA baseline for comparison with SPD component editing.

CE loss at fire positions (teach it to predict target token) + KL loss at all
other positions (preserve original predictions). Single forward pass per step.

Usage:
    with LoRATrainer(model.target_model, "h.2.mlp.down_proj", train_seqs, lr=1e-3) as lora:
        for _ in range(300):
            lora.train_step(kl_weight=10.0)
        result = eval_edit(lora.forward)
"""

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


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

        mod: Any = target_model
        for part in layer_path.split("."):
            mod = getattr(mod, part)
        assert isinstance(mod, torch.nn.Linear)
        self.linear = mod

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
        batch: list[tuple[tuple[Tensor, list[int]], Tensor]],
    ) -> float:
        """One step on a batch of (example, baseline) pairs. Accumulates loss, single optimizer step."""
        ce_total = torch.tensor(0.0, device="cuda")
        kl_total = torch.tensor(0.0, device="cuda")

        for (tokens, positions), baseline in batch:
            logits = self.forward(tokens.unsqueeze(0))[0]
            pos_t = torch.tensor(positions, device="cuda")
            ce_total = ce_total + F.cross_entropy(logits[positions], tokens[pos_t + 1])

            if self.kl_weight > 0:
                probs = logits.softmax(-1)
                kl = (probs * ((probs + 1e-10).log() - (baseline + 1e-10).log())).sum(-1)
                fire_mask = torch.ones(len(tokens), dtype=torch.bool, device="cuda")
                fire_mask[positions] = False
                kl_total = kl_total + kl[fire_mask].mean()

        n = len(batch)
        loss = ce_total / n + self.kl_weight * kl_total / n
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
