"""Pile 4-layer LlamaSimpleMLP entry point — reproduces the VPD paper's pile-4L
decomposition using `nano_param_decomp/run.py`.

`C_PER_MODULE_4L` pins the per-module component counts. `load_paper_target_model`
fetches the pretrained 4-layer LlamaSimpleMLP from W&B.

Launch from the repo root (relative imports require `-m`):

    # 8-GPU single-node
    torchrun --standalone --nproc_per_node=8 -m nano_param_decomp.pile_4L

    # Single-GPU smoke test
    python -m nano_param_decomp.pile_4L
"""

import os
import types
from collections.abc import Iterator

import datasets
import torch
import torch.nn as nn
from torch import Tensor

from param_decomp.pretrain.models.llama_simple_mlp import LlamaSimpleMLP

from .run import Config, decompose

C_PER_MODULE_4L: dict[str, int] = {
    "h.0.attn.q_proj": 512,
    "h.0.attn.k_proj": 512,
    "h.0.attn.v_proj": 1024,
    "h.0.attn.o_proj": 1024,
    "h.0.mlp.c_fc": 3072,
    "h.0.mlp.down_proj": 3584,
    "h.1.attn.q_proj": 512,
    "h.1.attn.k_proj": 512,
    "h.1.attn.v_proj": 1024,
    "h.1.attn.o_proj": 1024,
    "h.1.mlp.c_fc": 3072,
    "h.1.mlp.down_proj": 3584,
    "h.2.attn.q_proj": 512,
    "h.2.attn.k_proj": 512,
    "h.2.attn.v_proj": 1024,
    "h.2.attn.o_proj": 1024,
    "h.2.mlp.c_fc": 3072,
    "h.2.mlp.down_proj": 3584,
    "h.3.attn.q_proj": 512,
    "h.3.attn.k_proj": 512,
    "h.3.attn.v_proj": 1024,
    "h.3.attn.o_proj": 1024,
    "h.3.mlp.c_fc": 3072,
    "h.3.mlp.down_proj": 3584,
}


def load_paper_target_model(
    run_path: str = "goodfire/spd/runs/t-9d2b8f02",
) -> nn.Module:
    """Load the 4-layer pretrained LlamaSimpleMLP used in the VPD paper. Requires a `.env`
    with WandB credentials."""
    model = LlamaSimpleMLP.from_pretrained(run_path)
    # LlamaSimpleMLP.forward returns (logits, loss); our training loop expects bare logits.
    # Monkey-patch the bound forward, leaving the submodule structure intact so component
    # paths like `h.0.mlp.c_fc` still resolve via `get_submodule`.
    original_forward = model.forward

    def forward_logits_only(_self: nn.Module, idx: Tensor) -> Tensor:
        logits, _loss = original_forward(idx)
        assert logits is not None
        return logits

    model.forward = types.MethodType(forward_logits_only, model)
    return model


def make_loader(
    batch_size: int, seq_len: int, rank: int, world_size: int, split: str, seed: int
) -> Iterator[Tensor]:
    """Stream pre-tokenized Pile shards. The dataset is sharded by rank, then a per-rank
    buffered shuffle is layered on top of the on-disk pre-shuffle. Each example's
    `input_ids` is already at least `seq_len` long, so we just truncate and stack."""
    ds = datasets.load_dataset(
        "danbraunai/pile-uncopyrighted-tok-shuffled", split=split, streaming=True
    )
    if world_size > 1:
        ds = ds.shard(num_shards=world_size, index=rank)
    ds = ds.shuffle(seed=seed, buffer_size=1000)
    ds = ds.map(lambda ex: {"input_ids": ex["input_ids"][:seq_len]}).with_format("torch")
    local_B = batch_size // world_size
    while True:
        batch: list[Tensor] = []
        for ex in ds:
            batch.append(ex["input_ids"])
            if len(batch) == local_B:
                yield torch.stack(batch, dim=0)
                batch = []


if __name__ == "__main__":
    cfg = Config(
        C_per_module=C_PER_MODULE_4L,
        use_wandb=True,
        wandb_run_name="nano_param_decomp_pile_4L",
    )
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    decompose(
        load_paper_target_model(),
        cfg,
        make_loader(cfg.batch_size, cfg.seq_len, rank, world_size, "train", cfg.seed),
        make_loader(cfg.eval_batch_size, cfg.seq_len, rank, world_size, "val", cfg.seed + 1),
    )
