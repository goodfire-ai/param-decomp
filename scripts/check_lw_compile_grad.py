"""Correctness check for torch.compile on the LW step: compiled vs eager grads, RNG-pinned.

A 2.6× speedup is worthless if the grads are wrong. compile fuses/reorders float ops so it
won't be bit-exact, but V/U grads + loss must match to bf16 precision (ratio≈1, cosine≈1, no
systematic scale shift) — same fixed inputs, same params, eager vs compiled.

Run: srun --gres=gpu:1 python scripts/check_lw_compile_grad.py
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from param_decomp.components import make_components  # noqa: E402
from param_decomp.fused_linear_kl import fused_linear_kl_div  # noqa: E402
from param_decomp.masks import ComponentsMaskInfo  # noqa: E402
from param_decomp_lab.experiments.lm.pretrain.models.gpt2_simple import (  # noqa: E402
    GPT2Simple,
    GPT2SimpleConfig,
)
from param_decomp_lab.experiments.lm.vendored.gpt2 import componentize_gpt2  # noqa: E402

N_LAYERS, D_MODEL, N_HEADS, VOCAB, SEQ, C, BL = 48, 1600, 25, 50257, 1024, 1024, 4
DEV = "cuda"
SITE = "h.0.attn.q_proj"


def build():
    cfg = GPT2SimpleConfig(
        model_type="GPT2Simple",
        n_layer=N_LAYERS,
        n_head=N_HEADS,
        n_embd=D_MODEL,
        vocab_size=VOCAB,
        block_size=SEQ,
    )
    cg = componentize_gpt2(GPT2Simple(cfg), make_components(GPT2Simple(cfg), {SITE: C})).to(DEV)
    cg.enable_activation_checkpointing()
    return cg


def run_grads(model, cg, idx, ci, u) -> dict[str, torch.Tensor]:
    comp = cg.get_submodule(SITE).components
    comp.V.grad = None
    comp.U.grad = None
    with cg.bypass_lm_head() as lm_w, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            target_h = model(idx, None)
        mi = {SITE: ComponentsMaskInfo(component_mask=ci + (1 - ci) * u, routing_mask="all")}
        pred_h = model(idx, mi)
        loss, n = fused_linear_kl_div(
            pred_h.reshape(-1, D_MODEL), target_h.reshape(-1, D_MODEL), lm_w
        )
        (loss / n).backward()
    return {
        "loss": (loss / n).detach().float(),
        "g_V": comp.V.grad.detach().float(),
        "g_U": comp.U.grad.detach().float(),
    }


def main() -> None:
    assert torch.cuda.is_available()
    cg = build()
    g = torch.Generator(device=DEV).manual_seed(7)
    idx = torch.randint(0, VOCAB, (BL, SEQ), device=DEV, generator=g)
    ci = torch.rand(BL, SEQ, C, device=DEV, generator=g)
    u = torch.rand(BL, SEQ, C, device=DEV, generator=g)

    eager = run_grads(cg, cg, idx, ci, u)
    compiled = run_grads(torch.compile(cg), cg, idx, ci, u)

    print("compiled vs eager (RNG-pinned, bl=4, XL, ckpt on):")
    for k in eager:
        a, b = eager[k].flatten(), compiled[k].flatten()
        if a.numel() == 1:
            print(
                f"  {k:6s} eager={a.item():.5e} compiled={b.item():.5e} rel={abs(b.item() - a.item()) / (abs(a.item()) + 1e-12):.2e}"
            )
            continue
        ratio = (b.abs().mean() / a.abs().mean()).item()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        rel = ((b - a).abs() / (a.abs() + 1e-8)).mean().item()
        flag = "OK" if (0.98 < ratio < 1.02 and cos > 0.999) else "  <-- CHECK"
        print(f"  {k:6s} ratio={ratio:.4f} cos={cos:.5f} mean_rel_err={rel:.4f} {flag}")


if __name__ == "__main__":
    main()
