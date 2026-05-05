"""SimpleStories 2-layer LlamaSimpleMLP entry point — reproduces
`param_decomp/experiments/lm/ss_llama_simple_mlp-2L.yaml` using
`nano_param_decomp/run.py`.

`C_PER_MODULE_SS_2L` is copied verbatim from the YAML's `module_info` (6 module
types × 2 layers). `load_simplestories_target_model` is the only place that
touches the `param_decomp` package — it fetches the pretrained 2-layer
LlamaSimpleMLP from W&B.

Tokenization: the YAML sets `is_tokenized: false` and tokenizes
`SimpleStories/SimpleStories` on the fly with `SimpleStories/test-SimpleStories-gpt2-1.25M`.
The local `make_loader` below does the same: tokenize each story (lowercased to match the main
path's SimpleStories handling) and EOS-pack into fixed-length `seq_len` chunks.

Launch from the repo root (relative imports require `-m`):

    # 8-GPU single-node
    torchrun --standalone --nproc_per_node=8 -m nano_param_decomp.simplestories_4L

    # Single-GPU smoke test
    python -m nano_param_decomp.simplestories_4L
"""

# HF tokenizer / dataset stubs are loose; suppress the resulting noise.
# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false

import os
import types
from collections.abc import Iterator

import datasets
import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoTokenizer

from param_decomp.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from param_decomp.pretrain.run_info import PretrainRunInfo

from .run import Config, decompose

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


def load_simplestories_target_model(
    run_path: str = "goodfire/spd/runs/gf6rbga0",
) -> nn.Module:
    """Load the pretrained 2-layer SimpleStories LlamaSimpleMLP referenced by
    `pretrained_model_name` in the YAML.

    The cached `model_config.yaml` for this run predates the `model_type` field, so we
    inject it before instantiating — same patch the main `lm_decomposition.py` applies."""
    run_info = PretrainRunInfo.from_path(run_path)
    run_info.model_config_dict.setdefault("model_type", "LlamaSimpleMLP")
    model = LlamaSimpleMLP.from_run_info(run_info)
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
    """Tokenize the raw `SimpleStories/SimpleStories` dataset on the fly and EOS-pack into
    fixed `seq_len` chunks. Mirrors `param_decomp/data.py::tokenize_and_concatenate` (simplified)
    with `to_lower=True` to match the main path's SimpleStories handling."""
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
        # CI transformer (smaller than the pile-4L defaults)
        ci_d_model=512,
        ci_n_blocks=4,
        ci_n_heads=8,
        ci_mlp_hidden=2048,
        # Loss coefficients
        coeff_faith=1e6,
        coeff_imp=0.003,
        # Importance minimality (linear p-anneal 2.0 -> 0.5)
        p_end=0.5,
        imp_beta=0.1,
        # Persistent PGD
        ppgd_beta1=0.8,
        # Main LR schedule
        main_lr=3e-4,
        # Faithfulness warmup
        faithfulness_warmup_steps=200,
        # Batch
        batch_size=24,
        # Logging
        log_every=50,
        use_wandb=True,
        wandb_run_name="nano_param_decomp_simplestories_2L",
    )
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    decompose(
        load_simplestories_target_model(),
        cfg,
        make_loader(cfg.batch_size, cfg.seq_len, rank, world_size, cfg.seed),
        make_loader(cfg.eval_batch_size, cfg.seq_len, rank, world_size, cfg.seed + 1),
    )
