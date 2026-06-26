"""From-scratch decomposable SimpleStories 2-layer entry point.

Builds a *randomly initialised* LlamaSimpleMLP with the SimpleStories-2L architecture
(copied from a prior pretrained run's `model_config.yaml`; the checkpoint is ignored —
we only want the shape), replaces its decomposed `nn.Linear`s with `ComponentLinear`, and
trains everything from scratch with `from_scratch.decompose`.

    torchrun --standalone --nproc_per_node=8 -m nano_param_decomp.from_scratch_simplestories
    python -m nano_param_decomp.from_scratch_simplestories   # single-GPU smoke
"""

# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false

import os
import types
from collections.abc import Iterator

import datasets
import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoTokenizer

from param_decomp_lab.experiments.lm.pretrain.models.llama_simple_mlp import (
    LlamaSimpleMLP,
    LlamaSimpleMLPConfig,
)
from param_decomp_lab.experiments.lm.pretrain.run_info import PretrainRunInfo

from .from_scratch import Config, decompose

C_PER_MODULE_SS_2L: dict[str, int] = {
    "h.0.attn.q_proj": 288,
    "h.0.attn.k_proj": 288,
    "h.0.attn.v_proj": 384,
    "h.0.attn.o_proj": 480,
    "h.0.mlp.c_fc": 1152,
    "h.0.mlp.down_proj": 960,
    "h.1.attn.q_proj": 288,
    "h.1.attn.k_proj": 288,
    "h.1.attn.v_proj": 384,
    "h.1.attn.o_proj": 480,
    "h.1.mlp.c_fc": 1152,
    "h.1.mlp.down_proj": 960,
}


def build_random_model(arch_run_path: str = "goodfire/spd/runs/gf6rbga0") -> nn.Module:
    """Instantiate a fresh random LlamaSimpleMLP with the SimpleStories-2L architecture. Only
    the run's `model_config.yaml` is read; its trained weights are discarded."""
    run_info = PretrainRunInfo.from_path(arch_run_path)
    run_info.model_config_dict.setdefault("model_type", "LlamaSimpleMLP")
    model = LlamaSimpleMLP(LlamaSimpleMLPConfig(**run_info.model_config_dict))
    # LlamaSimpleMLP.forward returns (logits, loss); the training loop expects bare logits.
    original_forward = model.forward

    def forward_logits_only(_self: nn.Module, idx: Tensor) -> Tensor:
        logits, _loss = original_forward(idx)
        assert logits is not None
        return logits

    model.forward = types.MethodType(forward_logits_only, model)
    return model


def make_loader(
    batch_size: int, seq_len: int, rank: int, world_size: int, seed: int
) -> Iterator[Tensor]:
    """Tokenize `SimpleStories/SimpleStories` on the fly (lowercased) and EOS-pack into fixed
    `seq_len` chunks. Sharded by rank, then per-rank shuffled."""
    ds = datasets.load_dataset("SimpleStories/SimpleStories", split="train", streaming=False)
    if world_size > 1:
        ds = ds.shard(num_shards=world_size, index=rank)
    ds = ds.shuffle(seed=seed)
    tok = AutoTokenizer.from_pretrained("SimpleStories/test-SimpleStories-gpt2-1.25M")
    eos = tok.eos_token_id
    local_B = batch_size // world_size
    while True:
        buf: list[int] = []
        batch: list[Tensor] = []
        for ex in ds:
            buf.extend(tok.encode(ex["story"].lower(), add_special_tokens=False))
            buf.append(eos)
            while len(buf) >= seq_len:
                batch.append(torch.tensor(buf[:seq_len], dtype=torch.long))
                buf = buf[seq_len:]
                if len(batch) == local_B:
                    yield torch.stack(batch, dim=0)
                    batch = []


if __name__ == "__main__":
    cfg = Config(
        C_per_module=C_PER_MODULE_SS_2L,
        seq_len=512,
        ci_d_model=512,
        ci_n_blocks=4,
        ci_n_heads=8,
        ci_mlp_hidden=2048,
        use_wandb=True,
        wandb_run_name="from_scratch_simplestories_2L",
    )
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    decompose(
        build_random_model(),
        cfg,
        make_loader(cfg.batch_size, cfg.seq_len, rank, world_size, cfg.seed),
        make_loader(cfg.eval_batch_size, cfg.seq_len, rank, world_size, cfg.seed + 1),
    )
