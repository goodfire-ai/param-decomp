"""Rank-1 LoRA baseline for comparison with SPD component editing.

CE loss at fire positions (teach it to predict target token) + KL loss at all
other positions (preserve original predictions). Single forward pass per step.
"""

import random
from typing import Any

import torch
import torch.nn.functional as F
from jaxtyping import Int
from torch import Tensor


class LoRATrainer:
    """Rank-1 LoRA on a single linear layer. Reusable across runs without model reload."""

    def __init__(
        self,
        target_model: torch.nn.Module,
        layer_path: str,
        train_seqs: list[tuple[Int[Tensor, " seq"], list[int]]],
        lr: float = 1e-3,
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
        self.lr = lr

        self._hook = self.linear.register_forward_hook(self._fwd_hook)
        self.optimizer = torch.optim.AdamW([self.A, self.B], lr=lr)

        self.train_seqs = train_seqs
        self._cache_baselines()

    def _cache_baselines(self) -> None:
        """Cache per-token baseline probs for all training sequences."""
        self.base_probs: dict[int, Tensor] = {}
        with torch.no_grad():
            for tokens, _ in self.train_seqs:
                self.base_probs[id(tokens)] = self.forward(tokens.unsqueeze(0))[0].softmax(-1)

    def _fwd_hook(self, _mod: torch.nn.Module, _inp: tuple[Any, ...], out: Tensor) -> Tensor:
        return out + (_inp[0] @ self.A.T) @ self.B.T

    def forward(self, tokens: Tensor) -> Tensor:
        return self.target_model(tokens)[0]

    def train_step(self, kl_weight: float, batch_size: int = 8) -> float:
        """One step: CE at fire positions + KL at all other positions. Returns loss."""
        idxs = random.sample(range(len(self.train_seqs)), min(batch_size, len(self.train_seqs)))

        ce_total = torch.tensor(0.0, device="cuda")
        kl_total = torch.tensor(0.0, device="cuda")
        n_ce = 0
        n_kl = 0

        for idx in idxs:
            tokens, positions = self.train_seqs[idx]
            logits = self.forward(tokens.unsqueeze(0))[0]  # [seq, vocab]

            # CE at fire positions
            pos_t = torch.tensor(positions, device="cuda")
            ce_total = ce_total + F.cross_entropy(logits[positions], tokens[pos_t + 1])
            n_ce += 1

            # KL at all other positions
            if kl_weight > 0:
                probs = logits.softmax(-1)
                base = self.base_probs[id(tokens)]
                kl = (probs * ((probs + 1e-10).log() - (base + 1e-10).log())).sum(-1)
                fire_mask = torch.ones(len(tokens), dtype=torch.bool, device="cuda")
                fire_mask[positions] = False
                kl_total = kl_total + kl[fire_mask].mean()
                n_kl += 1

        loss = ce_total / n_ce + kl_weight * (kl_total / max(n_kl, 1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def reset(self) -> None:
        """Reset LoRA params to init. No model reload needed."""
        d_in = int(self.linear.weight.shape[1])
        with torch.no_grad():
            self.A.copy_(torch.randn(1, d_in, device="cuda") * 0.01)
            self.B.zero_()
        self.optimizer = torch.optim.AdamW([self.A, self.B], lr=self.lr)
        self._cache_baselines()

    def cleanup(self) -> None:
        """Remove forward hook."""
        self._hook.remove()
