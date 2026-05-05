"""Pile 4-layer LlamaSimpleMLP entry point — reproduces the SPD paper's
`pile_llama_simple_mlp-4L.yaml` decomposition using `nano_param_decomp/run.py`.

`C_PER_MODULE_4L` is copied verbatim from the 4L YAML's `module_info` and pins
the per-module component counts to the paper's choice. `load_paper_target_model`
is the only place that touches the `param_decomp` package — it fetches the
specific 4-layer pretrained LlamaSimpleMLP from W&B.

Launch (8-GPU single-node):
    torchrun --nproc_per_node=8 nano_param_decomp/pile_4L.py

Single-GPU smoke test:
    python nano_param_decomp/pile_4L.py
"""

import types

import torch.nn as nn
from torch import Tensor

from param_decomp.pretrain.models.llama_simple_mlp import LlamaSimpleMLP

from .run import Config, decompose

# Per-module component count, copied verbatim from the 4L YAML `module_info`.
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
    """Load the specific 4-layer pretrained LlamaSimpleMLP used in the SPD paper.

    Requires a `.env` with WandB credentials; the model is cached at
    `PARAM_DECOMP_OUT_DIR/pretrain_cache/<project>-<run_id>/` on first download.
    """
    model = LlamaSimpleMLP.from_pretrained(run_path)
    # LlamaSimpleMLP.forward returns (logits, loss); our training loop expects bare logits.
    # Monkey-patch the bound forward — submodule structure is untouched so C_PER_MODULE_4L
    # paths like `h.0.mlp.c_fc` still resolve via `get_submodule`.
    original_forward = model.forward

    def forward_logits_only(_self: nn.Module, idx: Tensor) -> Tensor:
        logits, _loss = original_forward(idx)
        assert logits is not None
        return logits

    model.forward = types.MethodType(forward_logits_only, model)
    return model


if __name__ == "__main__":
    cfg = Config(
        C_per_module=C_PER_MODULE_4L,
        use_wandb=True,
        wandb_run_name="nano_param_decomp_pile_4L",
    )
    decompose(load_paper_target_model(), cfg)
