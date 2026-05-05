"""SimpleStories 2-layer LlamaSimpleMLP entry point — reproduces
`param_decomp/experiments/lm/ss_llama_simple_mlp-2L.yaml` using
`nano_param_decomp/run.py`.

`C_PER_MODULE_SS_2L` is copied verbatim from the YAML's `module_info` (6 module
types × 2 layers). `load_simplestories_target_model` is the only place that
touches the `param_decomp` package — it fetches the pretrained 2-layer
LlamaSimpleMLP from W&B.

Note on tokenization: the YAML sets `is_tokenized: false` and tokenizes
`SimpleStories/SimpleStories` on the fly with `SimpleStories/test-SimpleStories-gpt2-1.25M`.
The nano dataloader assumes the dataset column already contains token id lists
(matching pile-4L's `danbraunai/pile-uncopyrighted-tok-shuffled`), so a
pre-tokenized SimpleStories variant is required for this entry point.

Launch (8-GPU single-node):
    torchrun --nproc_per_node=8 nano_param_decomp/simplestories_4L.py

Single-GPU smoke test:
    python nano_param_decomp/simplestories_4L.py
"""

import types

import torch.nn as nn
from torch import Tensor

from param_decomp.pretrain.models.llama_simple_mlp import LlamaSimpleMLP

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
    `pretrained_model_name` in the YAML."""
    model = LlamaSimpleMLP.from_pretrained(run_path)
    original_forward = model.forward

    def forward_logits_only(_self: nn.Module, idx: Tensor) -> Tensor:
        logits, _loss = original_forward(idx)
        assert logits is not None
        return logits

    model.forward = types.MethodType(forward_logits_only, model)
    return model


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
        # Dataset (must be pre-tokenized — see module docstring)
        dataset_name="SimpleStories/SimpleStories",
        dataset_column="story",
        use_wandb=True,
        wandb_run_name="nano_param_decomp_simplestories_2L",
    )
    decompose(load_simplestories_target_model(), cfg)
