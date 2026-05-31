"""One-time conversion of the canonical synthetic OthelloGPT into an in-repo target model.

Downloads the TransformerLens-format weights Neel Nanda mirrored on the HF Hub, remaps
them into the `nn.Linear`-based `OthelloGPT` layout (so the decomposition target system
can hook the matrices), and writes a pretrain-style run directory that
`PretrainRunInfo.from_path` / the `kind: pretrained` LM target loads.

The source checkpoint has LayerNorm folded into the following linear weights, so it
matches `OthelloGPT`'s parameter-free `LayerNormPre`. Correctness is checked two ways:
(1) the converted `nn.Linear` model must match an independent einsum forward over the raw
TransformerLens tensors, and (2) next-move cross-entropy on the real synthetic dataset
must be far below the uniform-over-61 baseline.

Run once:  python -m param_decomp_lab.experiments.lm.convert_othello_gpt
"""

import math
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from einops import rearrange
from huggingface_hub import hf_hub_download
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import functional as F
from transformers import PreTrainedTokenizerFast  # pyright: ignore[reportAttributeAccessIssue]

from param_decomp_lab.experiments.lm.pretrain.models.othello_gpt import OthelloGPT, OthelloGPTConfig
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR

HF_REPO = "NeelNanda/Othello-GPT-Transformer-Lens"
HF_FILE = "synthetic_model.pth"
DATASET = "taufeeque/othellogpt"

RUN_DIR = PARAM_DECOMP_OUT_DIR / "target_models" / "othello-gpt-synthetic"
TOKENIZER_DIR = Path(__file__).parent / "othello_tokenizer"


def convert_state_dict(tl_sd: dict[str, Tensor], config: OthelloGPTConfig) -> dict[str, Tensor]:
    """Map TransformerLens parameter names/shapes onto `OthelloGPT` state-dict keys."""
    out: dict[str, Tensor] = {
        "wte.weight": tl_sd["embed.W_E"],
        "wpe.weight": tl_sd["pos_embed.W_pos"],
        "lm_head.weight": tl_sd["unembed.W_U"].T.contiguous(),
        "lm_head.bias": tl_sd["unembed.b_U"],
    }
    for i in range(config.n_layer):
        p = f"blocks.{i}"
        h = f"h.{i}"
        for tl_w, tl_b, proj in [
            ("W_Q", "b_Q", "q_proj"),
            ("W_K", "b_K", "k_proj"),
            ("W_V", "b_V", "v_proj"),
        ]:
            out[f"{h}.attn.{proj}.weight"] = rearrange(
                tl_sd[f"{p}.attn.{tl_w}"], "head d_model d_head -> (head d_head) d_model"
            ).contiguous()
            out[f"{h}.attn.{proj}.bias"] = rearrange(
                tl_sd[f"{p}.attn.{tl_b}"], "head d_head -> (head d_head)"
            ).contiguous()
        out[f"{h}.attn.o_proj.weight"] = rearrange(
            tl_sd[f"{p}.attn.W_O"], "head d_head d_model -> d_model (head d_head)"
        ).contiguous()
        out[f"{h}.attn.o_proj.bias"] = tl_sd[f"{p}.attn.b_O"]
        out[f"{h}.mlp.c_fc.weight"] = tl_sd[f"{p}.mlp.W_in"].T.contiguous()
        out[f"{h}.mlp.c_fc.bias"] = tl_sd[f"{p}.mlp.b_in"]
        out[f"{h}.mlp.down_proj.weight"] = tl_sd[f"{p}.mlp.W_out"].T.contiguous()
        out[f"{h}.mlp.down_proj.bias"] = tl_sd[f"{p}.mlp.b_out"]
    return out


def reference_logits(
    tl_sd: dict[str, Tensor], idx: Int[Tensor, "batch pos"], config: OthelloGPTConfig
) -> Float[Tensor, "batch pos vocab"]:
    """Independent TransformerLens-style forward over the raw tensors, to cross-check the mapping."""
    eps = config.layer_norm_eps

    def lnpre(x: Tensor) -> Tensor:
        x = x - x.mean(-1, keepdim=True)
        return x / (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()

    _b, t = idx.shape
    x = tl_sd["embed.W_E"][idx] + tl_sd["pos_embed.W_pos"][:t]
    causal = torch.triu(torch.full((t, t), float("-inf")), diagonal=1)
    for i in range(config.n_layer):
        p = f"blocks.{i}"
        h = lnpre(x)
        q = (
            torch.einsum("btd,hde->bhte", h, tl_sd[f"{p}.attn.W_Q"])
            + tl_sd[f"{p}.attn.b_Q"][:, None, :]
        )
        k = (
            torch.einsum("btd,hde->bhte", h, tl_sd[f"{p}.attn.W_K"])
            + tl_sd[f"{p}.attn.b_K"][:, None, :]
        )
        v = (
            torch.einsum("btd,hde->bhte", h, tl_sd[f"{p}.attn.W_V"])
            + tl_sd[f"{p}.attn.b_V"][:, None, :]
        )
        scores = torch.einsum("bhte,bhse->bhts", q, k) / math.sqrt(q.shape[-1]) + causal
        z = torch.einsum("bhts,bhse->bhte", scores.softmax(-1), v)
        x = x + torch.einsum("bhte,hed->btd", z, tl_sd[f"{p}.attn.W_O"]) + tl_sd[f"{p}.attn.b_O"]
        h = lnpre(x)
        m = F.gelu(h @ tl_sd[f"{p}.mlp.W_in"] + tl_sd[f"{p}.mlp.b_in"])
        x = x + m @ tl_sd[f"{p}.mlp.W_out"] + tl_sd[f"{p}.mlp.b_out"]
    x = lnpre(x)
    return x @ tl_sd["unembed.W_U"] + tl_sd["unembed.b_U"]


def build_tokenizer() -> None:
    """Write a minimal 61-token WordLevel tokenizer (ids 0..60) the pre-tokenized loader can load."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {str(i): i for i in range(61)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="0"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()  # pyright: ignore[reportAttributeAccessIssue]
    fast = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="0", eos_token="0", pad_token="0"
    )
    fast.save_pretrained(TOKENIZER_DIR)


def main() -> None:
    config = OthelloGPTConfig(model_type="OthelloGPT")

    ckpt_path = hf_hub_download(HF_REPO, HF_FILE)
    tl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    tl_sd = {k: v for k, v in tl_sd.items() if not k.endswith((".mask", ".IGNORE"))}

    model = OthelloGPT(config)
    model.load_state_dict(convert_state_dict(tl_sd, config), strict=True)
    model.eval()

    # 1. Conversion correctness: nn.Linear model == independent einsum forward over raw tensors.
    idx = torch.randint(0, config.vocab_size, (4, config.block_size))
    with torch.no_grad():
        got, _ = model(idx)
        assert got is not None
        ref = reference_logits(tl_sd, idx, config)
    max_diff = (got - ref).abs().max().item()
    assert max_diff < 1e-4, f"converted model diverges from reference forward: {max_diff}"
    print(f"[ok] conversion matches reference forward (max logit diff {max_diff:.2e})")

    # 2. The model actually works on the real synthetic data: CE far below uniform (log 61 = 4.11).
    ds = load_dataset(DATASET, split="validation", streaming=True)
    rows = [r["tokens"][: config.block_size + 1] for r in ds.take(64)]
    batch = torch.tensor([r for r in rows if len(r) == config.block_size + 1])
    inputs, targets = batch[:, :-1], batch[:, 1:]
    with torch.no_grad():
        logits, _ = model(inputs)
        assert logits is not None
        ce = F.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1)).item()
        acc = (logits.argmax(-1) == targets).float().mean().item()
    print(
        f"[check] next-move CE={ce:.3f} nats (uniform={math.log(config.vocab_size):.3f}), top1 acc={acc:.3f}"
    )
    assert ce < 3.0, f"CE {ce:.3f} too high — weights or dataset tokenization mismatch"

    # 3. Package as a pretrain-style run dir consumed by PretrainRunInfo.from_path / kind: pretrained.
    build_tokenizer()
    ckpt_dir = RUN_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "model_step_0.pt")
    (RUN_DIR / "model_config.yaml").write_text(yaml.dump(config.model_dump(mode="json")))
    (RUN_DIR / "final_config.yaml").write_text(
        yaml.dump(
            {
                "seed": 0,
                "data": {
                    "dataset_name": DATASET,
                    "tokenizer_name": str(TOKENIZER_DIR.resolve()),
                    "max_seq_len": config.block_size,
                    "is_tokenized": True,
                    "streaming": True,
                    "column_name": "tokens",
                    "train_split": "train",
                    "eval_split": "validation",
                },
            }
        )
    )

    print(f"[done] wrote target model to {RUN_DIR}")
    print("       target.run_path:", ckpt_dir / "model_step_0.pt")
    print("       data.tokenizer_name:", TOKENIZER_DIR.resolve())


if __name__ == "__main__":
    main()
