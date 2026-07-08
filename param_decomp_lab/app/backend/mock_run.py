"""Mock data interface for the circuit builder.

A tiny randomly-initialized LlamaSimpleMLP wrapped in a REAL ComponentModel with random
V/U — the full circuit-builder pipeline (j-vectors, LoRA assembly, comparison) runs the
exact code path of the real run; only the data sources are fake:

- target weights + V/U: random (seeded)
- tokenizer: byte-level (vocab 256), no network
- token batches for j-vector averaging: seeded random bytes
- labels + activating examples: deterministic placeholders

Swap `load_mock_context` for a SavedLMRun-backed loader once p-55ea3f9b is synced.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from jaxtyping import Int
from torch import Tensor

from param_decomp.component_model import ComponentModel
from param_decomp_config.ci_fn import LayerwiseCiConfig
from param_decomp.decomposition_targets import resolve_decomposition_targets
from param_decomp_config.decomposition_target import DecompositionTargetConfig
from param_decomp_lab.app.backend.circuit_builder import (
    SubcomponentInfoProvider,
    TokenBatchProvider,
    TokenizerProtocol,
)
from param_decomp_lab.batch_and_loss_fns import make_run_batch
from param_decomp_lab.experiments.lm.pretrain.models.llama_simple_mlp import (
    LlamaSimpleMLP,
    LlamaSimpleMLPConfig,
)

MOCK_VOCAB = 256  # byte-level


class ByteTokenizer:
    """Byte-level tokenizer: token id = byte value. No network, fully deterministic."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode_tokens(self, token_ids: list[int]) -> list[str]:
        return [bytes([t]).decode("utf-8", errors="replace") for t in token_ids]


@dataclass
class RandomTokenProvider:
    seed: int = 0

    def batches(self, batch_size: int, seq_len: int) -> Iterator[Int[Tensor, "B T"]]:
        generator = torch.Generator().manual_seed(self.seed)
        while True:
            yield torch.randint(0, MOCK_VOCAB, (batch_size, seq_len), generator=generator)


class MockInfoProvider:
    """Deterministic placeholder labels + activating examples."""

    def label(self, site: str, idx: int) -> str | None:
        return f"[mock] {site} component {idx}"

    def activating_examples(self, site: str, idx: int, limit: int) -> list[dict]:
        return [
            {
                "tokens": [f"tok{(idx + k + j) % 97}" for j in range(8)],
                "active_position": 4,
                "activation": round(1.0 / (k + 1), 3),
            }
            for k in range(min(limit, 3))
        ]


# Same site layout as p-55ea3f9b (scaled down): 4 blocks, separate-attn + gelu MLP.
def mock_target_config(
    n_layer: int = 4, n_embd: int = 32, n_head: int = 2, block_size: int = 64
) -> LlamaSimpleMLPConfig:
    return LlamaSimpleMLPConfig(
        model_type="LlamaSimpleMLP",
        block_size=block_size,
        vocab_size=MOCK_VOCAB,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        n_intermediate=n_embd * 4,
        rotary_dim=n_embd // n_head,
        n_ctx=block_size,
        n_key_value_heads=n_head,  # full MHA, like the real target
        use_grouped_query_attention=True,
        flash_attention=False,  # CPU-friendly
    )


def mock_component_model(cfg: LlamaSimpleMLPConfig | None = None, seed: int = 0) -> ComponentModel:
    """Random tiny ComponentModel with the same decomposed sites as p-55ea3f9b."""
    cfg = cfg or mock_target_config()
    torch.manual_seed(seed)
    target = LlamaSimpleMLP(cfg)
    target.eval()
    target.requires_grad_(False)

    c_per_site = {"attn.q_proj": 12, "attn.k_proj": 12, "attn.v_proj": 16,
                  "attn.o_proj": 16, "mlp.c_fc": 24, "mlp.down_proj": 24}
    targets = resolve_decomposition_targets(
        target,
        [
            DecompositionTargetConfig(module_pattern=f"h.*.{within}", C=c)
            for within, c in c_per_site.items()
        ],
    )
    return ComponentModel(
        target_model=target,
        run_batch=make_run_batch(0),  # model(idx) -> (logits, loss); extract logits
        decomposition_targets=targets,
        ci_config=LayerwiseCiConfig(mode="layerwise", fn_type="mlp", hidden_dims=[8]),
        sigmoid_type="leaky_hard",
    )


@dataclass
class CircuitBuilderContext:
    """Everything the circuit-builder endpoints need, real or mock."""

    run_id: str
    model: ComponentModel
    tokenizer: TokenizerProtocol
    token_provider: TokenBatchProvider
    info: SubcomponentInfoProvider
    seq_len: int  # seq len for j-vector batches
    batch_size: int


def load_mock_context(seed: int = 0) -> CircuitBuilderContext:
    return CircuitBuilderContext(
        run_id="mock",
        model=mock_component_model(seed=seed),
        tokenizer=ByteTokenizer(),
        token_provider=RandomTokenProvider(seed=seed),
        info=MockInfoProvider(),
        seq_len=32,
        batch_size=4,
    )
