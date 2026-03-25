"""Rank-1 LoRA baseline for comparison with SPD component editing.

Applies a rank-1 low-rank adapter (B @ A) to a single linear layer via a forward hook.
Supports batched KL regularization to preserve the original distribution.

Usage:
    em, tok = EditableModel.from_wandb("wandb:goodfire/spd/s-55ea3f9b")
    lora = LoRATrainer(em.model.target_model, layer_path="h.2.mlp.down_proj", lr=1e-3)
    lora.prepare_reg(global_seqs)

    for step in range(300):
        lora.train_step(train_seqs, kl_weight=10.0)

    # Eval...
    lora.reset()   # reset params, reuse model
    lora.cleanup()  # remove hook when done
"""

from __future__ import annotations

import random
from typing import Any

import torch
import torch.nn.functional as F
from jaxtyping import Int
from torch import Tensor
from torch.utils.hooks import RemovableHandle


class LoRATrainer:
    """Rank-1 LoRA on a single linear layer. Reusable across runs without model reload."""

    def __init__(self, target_model: torch.nn.Module, layer_path: str, lr: float = 1e-3):
        self.target_model = target_model
        self.target_model.eval()
        self.layer_path = layer_path

        mod: Any = target_model
        for part in layer_path.split("."):
            mod = getattr(mod, part)
        self.linear: torch.nn.Linear = mod
        assert isinstance(self.linear, torch.nn.Linear)

        d_out, d_in = int(self.linear.weight.shape[0]), int(self.linear.weight.shape[1])
        self.A = torch.randn(1, d_in, device="cuda") * 0.01
        self.B = torch.zeros(d_out, 1, device="cuda")
        self.A.requires_grad = True
        self.B.requires_grad = True
        self.lr = lr

        self._hook: RemovableHandle = self.linear.register_forward_hook(self._fwd_hook)
        self.optimizer = torch.optim.AdamW([self.A, self.B], lr=lr)

        self._reg_batch: Tensor | None = None
        self._reg_lens: list[int] = []
        self._reg_base_batch: Tensor | None = None

    def _fwd_hook(self, _mod: torch.nn.Module, _inp: tuple[Any, ...], out: Tensor) -> Tensor:
        return out + (_inp[0] @ self.A.T) @ self.B.T

    def forward(self, tokens: Tensor) -> Tensor:
        return self.target_model(tokens)[0]

    def __call__(self, tokens: Tensor) -> Tensor:
        return self.forward(tokens)

    def prepare_reg(self, reg_seqs: list[Int[Tensor, " seq"]]) -> None:
        """Pre-batch and cache baselines for KL regularization sequences."""
        max_len = max(t.shape[0] for t in reg_seqs)
        self._reg_batch = torch.zeros(len(reg_seqs), max_len, dtype=torch.long, device="cuda")
        self._reg_lens = []
        base_list = []
        with torch.no_grad():
            for i, t in enumerate(reg_seqs):
                self._reg_batch[i, : t.shape[0]] = t
                self._reg_lens.append(t.shape[0])
                base_list.append(self.forward(t.unsqueeze(0))[0].softmax(-1))
        vocab = base_list[0].shape[-1]
        self._reg_base_batch = torch.zeros(len(reg_seqs), max_len, vocab, device="cuda")
        for i, (bl, seq_len) in enumerate(zip(base_list, self._reg_lens, strict=True)):
            self._reg_base_batch[i, :seq_len] = bl[:seq_len]

    def train_step(
        self,
        train_seqs: list[tuple[Int[Tensor, " seq"], list[int]]],
        kl_weight: float = 0.0,
        batch_size: int = 8,
    ) -> float:
        """One training step with optional KL regularization. Returns total loss."""
        idxs = random.sample(range(len(train_seqs)), min(batch_size, len(train_seqs)))
        batch = [train_seqs[j] for j in idxs]
        max_len = max(t.shape[0] for t, _ in batch)
        batch_t = torch.zeros(len(batch), max_len, dtype=torch.long, device="cuda")
        batch_pos = []
        for i, (t, pos) in enumerate(batch):
            batch_t[i, : t.shape[0]] = t
            batch_pos.append(pos)

        logits = self.forward(batch_t)
        ce = torch.tensor(0.0, device="cuda")
        for i, pos in enumerate(batch_pos):
            ce = ce + F.cross_entropy(
                logits[i, pos], batch_t[i, torch.tensor(pos, device="cuda") + 1]
            )
        ce = ce / len(batch)

        kl_reg = torch.tensor(0.0, device="cuda")
        if kl_weight > 0 and self._reg_batch is not None:
            assert self._reg_base_batch is not None
            logits_reg = self.forward(self._reg_batch)
            probs_reg = logits_reg.softmax(-1)
            kl_all = (
                probs_reg * ((probs_reg + 1e-10).log() - (self._reg_base_batch + 1e-10).log())
            ).sum(-1)
            for i, seq_len in enumerate(self._reg_lens):
                kl_reg = kl_reg + kl_all[i, :seq_len].mean()
            kl_reg = kl_reg / len(self._reg_lens)

        total = ce + kl_weight * kl_reg
        self.optimizer.zero_grad()
        total.backward()
        self.optimizer.step()
        return total.item()

    def reset(self) -> None:
        """Reset LoRA params to init. No model reload needed."""
        d_in = int(self.linear.weight.shape[1])
        with torch.no_grad():
            self.A.copy_(torch.randn(1, d_in, device="cuda") * 0.01)
            self.B.zero_()
        self.optimizer = torch.optim.AdamW([self.A, self.B], lr=self.lr)
        if self._reg_batch is not None:
            assert self._reg_base_batch is not None
            base_list = []
            with torch.no_grad():
                for i in range(self._reg_batch.shape[0]):
                    t = self._reg_batch[i, : self._reg_lens[i]]
                    base_list.append(self.forward(t.unsqueeze(0))[0].softmax(-1))
            for i, (bl, seq_len) in enumerate(zip(base_list, self._reg_lens, strict=True)):
                self._reg_base_batch[i, :seq_len] = bl[:seq_len]

    def cleanup(self) -> None:
        """Remove forward hook."""
        self._hook.remove()
