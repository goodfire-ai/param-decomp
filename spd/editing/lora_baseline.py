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
        """One batched step. Pads sequences, single forward pass, masked CE + KL."""
        B = len(batch)
        max_len = max(tokens.shape[0] for (tokens, _), _ in batch)
        vocab = int(self.linear.weight.shape[0])

        # Pad tokens and baselines to max_len
        tokens_padded = torch.zeros(B, max_len, dtype=torch.long, device="cuda")
        baselines_padded = torch.zeros(B, max_len, vocab, device="cuda")
        seq_lens = []
        all_positions: list[list[int]] = []
        for i, ((tokens, positions), baseline) in enumerate(batch):
            seq_lens.append(tokens.shape[0])
            tokens_padded[i, : tokens.shape[0]] = tokens
            baselines_padded[i, : baseline.shape[0]] = baseline
            all_positions.append(positions)

        logits = self.forward(tokens_padded)  # [B, max_len, vocab]

        # CE at fire positions
        ce_total = torch.tensor(0.0, device="cuda")
        for i, positions in enumerate(all_positions):
            pos_t = torch.tensor(positions, device="cuda")
            ce_total = ce_total + F.cross_entropy(logits[i, positions], tokens_padded[i, pos_t + 1])
        ce = ce_total / B

        # KL at non-fire, non-pad positions
        kl_loss = torch.tensor(0.0, device="cuda")
        if self.kl_weight > 0:
            probs = logits.softmax(-1)
            kl = (probs * ((probs + 1e-10).log() - (baselines_padded + 1e-10).log())).sum(-1)
            for i, positions in enumerate(all_positions):
                mask = torch.zeros(max_len, dtype=torch.bool, device="cuda")
                mask[: seq_lens[i]] = True
                mask[positions] = False
                kl_loss = kl_loss + kl[i, mask].mean()
            kl_loss = kl_loss / B

        loss = ce + self.kl_weight * kl_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
