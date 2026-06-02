"""Does torch.compile speed up the LW step? (the vendored rewrite should make it traceable.)

Eager vs torch.compile(model) on the full LW step (target_fwd + masked recon fwd + fused-KL +
backward, bl=256, ckpt on). Reports step time, speedup, and the Dynamo graph-break count — the
rewrite threads masks as forward args (pure forward) so it should compile; the target attention
uses F.scaled_dot_product_attention directly (no sdpa_kernel context that broke the CI-fn compile).

Run: srun --gres=gpu:1 python scripts/bench_lw_compile.py [BL]
"""

import sys
import time
from pathlib import Path

import torch
import torch._dynamo
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from param_decomp.components import make_components  # noqa: E402
from param_decomp.fused_linear_kl import fused_linear_kl_div  # noqa: E402
from param_decomp.masks import ComponentsMaskInfo  # noqa: E402
from param_decomp_lab.experiments.lm.pretrain.models.gpt2_simple import (  # noqa: E402
    GPT2Simple,
    GPT2SimpleConfig,
)
from param_decomp_lab.experiments.lm.vendored.gpt2 import (  # noqa: E402
    ComponentGPT2,
    componentize_gpt2,
)

N_LAYERS, D_MODEL, N_HEADS, VOCAB, SEQ, C = 48, 1600, 25, 50257, 1024, 1024
DEV = "cuda"
SITE = "h.0.attn.q_proj"
BL = int(sys.argv[1]) if len(sys.argv) > 1 else 256


def build() -> ComponentGPT2:
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


def lw_step(model, cg: ComponentGPT2, opt: torch.optim.Optimizer, idx: Tensor) -> None:
    """One LW step; `model` is the (maybe-compiled) callable, `cg` the raw module for bypass ctx."""
    opt.zero_grad(set_to_none=True)
    with cg.bypass_lm_head() as lm_w, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            target_h = model(idx, None)
        ci = torch.rand(BL, SEQ, C, device=DEV)
        u = torch.rand(BL, SEQ, C, device=DEV)
        mi = {SITE: ComponentsMaskInfo(component_mask=ci + (1 - ci) * u, routing_mask="all")}
        pred_h = model(idx, mi)
        loss, n = fused_linear_kl_div(
            pred_h.reshape(-1, D_MODEL), target_h.reshape(-1, D_MODEL), lm_w
        )
        (loss / n).backward()
    opt.step()


def timed(model, cg, opt, idx, n_warm: int, n: int) -> float:
    for _ in range(n_warm):
        lw_step(model, cg, opt, idx)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        lw_step(model, cg, opt, idx)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def main() -> None:
    assert torch.cuda.is_available()
    print(f"LW compile bench, bl={BL}, ckpt on, {torch.cuda.get_device_name(0)}\n")
    cg = build()
    opt = torch.optim.Adam([p for p in cg.parameters() if p.requires_grad], lr=1e-4)
    idx = torch.randint(0, VOCAB, (BL, SEQ), device=DEV)

    eager = timed(cg, cg, opt, idx, n_warm=3, n=5)
    print(f"eager     : {eager:.2f}s / step")

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    import os as _os

    if _os.environ.get("COMPILE_BLOCKS", "").strip() in ("1", "true", "yes"):
        # compile each block, leave the model's checkpoint loop eager (the distributed fix)
        for _blk in cg._h:  # type: ignore[attr-defined]
            _blk.compile()
        compiled = cg
        print("(compile-blocks: per-block compile, eager checkpoint)")
    else:
        compiled = torch.compile(cg)
    comp = timed(compiled, cg, opt, idx, n_warm=5, n=5)  # warmup includes compilation
    breaks = sum(v for k, v in torch._dynamo.utils.counters["graph_break"].items())
    n_uniq = len(torch._dynamo.utils.counters["graph_break"])
    print(f"compiled  : {comp:.2f}s / step   (graph breaks: {breaks} total, {n_uniq} unique)")
    print(f"\n>>> compile speedup: {eager / comp:.2f}×  ({eager:.2f}s → {comp:.2f}s)")
    if breaks:
        print("graph-break reasons:")
        for k, v in sorted(
            torch._dynamo.utils.counters["graph_break"].items(), key=lambda x: -x[1]
        )[:8]:
            print(f"  {v:4d}× {k}")


if __name__ == "__main__":
    main()
