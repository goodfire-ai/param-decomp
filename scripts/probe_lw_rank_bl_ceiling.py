"""Realistic single-LW-rank bl_lw ceiling with activation checkpointing (GPT2-XL).

Unlike probe_lw_target_bl_ceiling.py (conservative: all 96 sites masked at once, full
unsharded V/U), this models what ONE layerwise rank actually holds and runs:
  - only `SITES_PER_BLOCK` sites are decomposed (the rank's owned, sharded V/U); every other
    q/k is a plain frozen Linear (target);
  - the per-site reconstruction STREAMS — for each owned site, one masked full-model forward
    (LM head bypassed → hidden state) + fused linear-KL vs the clean target hidden + backward,
    freed before the next site;
  - Adam over the owned V/U only.
Block checkpointing on; bf16 autocast. Sweeps bl until OOM to find the TRUE per-rank ceiling.

Run: srun --gres=gpu:1 python scripts/probe_lw_rank_bl_ceiling.py
"""

import gc
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
from param_decomp_lab.experiments.lm.vendored.gpt2 import (  # noqa: E402
    ComponentGPT2,
    componentize_gpt2,
)

N_LAYERS, D_MODEL, N_HEADS, VOCAB, SEQ, C = 48, 1600, 25, 50257, 1024, 1024
SITES_PER_BLOCK = 2
DEV = "cuda"
OWNED = [f"h.{i // 2}.attn.{'q' if i % 2 == 0 else 'k'}_proj" for i in range(SITES_PER_BLOCK)]
BLS = [16, 32, 48, 64, 96, 128, 192, 256]


def build(ckpt: bool) -> ComponentGPT2:
    cfg = GPT2SimpleConfig(
        model_type="GPT2Simple",
        n_layer=N_LAYERS,
        n_head=N_HEADS,
        n_embd=D_MODEL,
        vocab_size=VOCAB,
        block_size=SEQ,
    )
    model = GPT2Simple(cfg)
    cg = componentize_gpt2(model, make_components(model, {s: C for s in OWNED})).to(DEV)
    if ckpt:
        cg.enable_activation_checkpointing()
    return cg


def run(bl: int, ckpt: bool) -> tuple[str, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    try:
        cg = build(ckpt)
        opt = torch.optim.Adam([p for p in cg.parameters() if p.requires_grad], lr=1e-3)
        opt.zero_grad(set_to_none=True)
        idx = torch.randint(0, VOCAB, (bl, SEQ), device=DEV)
        with cg.bypass_lm_head() as lm_w, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                target_h = cg(idx, None)
            for site in OWNED:
                mi = {
                    site: ComponentsMaskInfo(
                        component_mask=torch.rand(bl, SEQ, C, device=DEV), routing_mask="all"
                    )
                }
                pred_h = cg(idx, mi)
                loss, n = fused_linear_kl_div(
                    pred_h.reshape(-1, D_MODEL), target_h.reshape(-1, D_MODEL), lm_w
                )
                (loss / n).backward()
        opt.step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
        del cg, opt, idx, target_h
        return "OK", peak
    except torch.cuda.OutOfMemoryError:
        return "OOM", torch.cuda.max_memory_allocated() / 1e9
    finally:
        torch.cuda.empty_cache()
        gc.collect()


def main() -> None:
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    cap = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(
        f"GPU {torch.cuda.get_device_name(0)} ~{cap:.0f}GB | realistic LW rank: "
        f"{SITES_PER_BLOCK} owned sites decomposed (C={C}), streaming per-site recon "
        f"(bypass+fused-KL), block ckpt ON, seq {SEQ}"
    )
    print(f"\n{'bl_lw':>6} | {'plain peak':>14} | {'ckpt peak':>14}")
    print("-" * 40)

    def fmt(r: tuple[str, float]) -> str:
        return "OOM" if r[0] == "OOM" else "skip" if r[0] == "skip" else f"{r[1]:.1f}GB"

    plain_oom = ckpt_oom = False
    for bl in BLS:
        rp = ("skip", 0.0) if plain_oom else run(bl, False)
        rc = ("skip", 0.0) if ckpt_oom else run(bl, True)
        plain_oom = plain_oom or rp[0] == "OOM"
        ckpt_oom = ckpt_oom or rc[0] == "OOM"
        print(f"{bl:>6} | {fmt(rp):>14} | {fmt(rc):>14}", flush=True)
    print("\n=> per-rank bl_lw ceiling (plain vs ckpt) = last OK below first OOM in each column")


if __name__ == "__main__":
    main()
