"""Decompose the serial LW block: phase times, backward recompute-vs-grad, FLOP by module.

At fixed GPUs the LW per-rank work is irreducible by topology (sites×batch is fixed), so the
only fixed-GPU wins are *reducing the per-rank serial work*. This quantifies what comprises it:

  [A] phase times @ bl=256 (cuda-event): target_fwd / masked_fwd / KL / backward.
  [B] backward split @ bl=16 (fits un-checkpointed): time backward with ckpt vs without →
      recompute fraction of the backward (the rest is the true gradient compute).
  [C] one forward's FLOPs by module bucket (FlopCounterMode @ bl=8): attention / MLP / lm_head /
      embed / component-path, and per-block (to size the prefix-reuse lever: masking site at
      layer L leaves blocks 0..L-1 identical to the clean target forward).

Run: srun --gres=gpu:1 python scripts/bench_lw_block_breakdown.py
"""

import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

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


def ev():
    return torch.cuda.Event(enable_timing=True)


def phase_step(cg, opt, idx, totals):
    bl = idx.shape[0]
    opt.zero_grad(set_to_none=True)
    marks = {}
    with cg.bypass_lm_head() as lm_w, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        s = ev()
        s.record()
        with torch.no_grad():
            target_h = cg(idx, None)
        e = ev()
        e.record()
        torch.cuda.synchronize()
        totals["1_target_fwd"] += s.elapsed_time(e)
        ci = torch.rand(bl, SEQ, C, device=DEV)
        u = torch.rand(bl, SEQ, C, device=DEV)
        mi = {SITE: ComponentsMaskInfo(component_mask=ci + (1 - ci) * u, routing_mask="all")}
        s = ev()
        s.record()
        pred_h = cg(idx, mask_infos=mi)
        e = ev()
        e.record()
        torch.cuda.synchronize()
        totals["2_masked_fwd"] += s.elapsed_time(e)
        s = ev()
        s.record()
        loss, n = fused_linear_kl_div(
            pred_h.reshape(-1, D_MODEL), target_h.reshape(-1, D_MODEL), lm_w
        )
        e = ev()
        e.record()
        torch.cuda.synchronize()
        totals["3_kl"] += s.elapsed_time(e)
        s = ev()
        s.record()
        (loss / n).backward()
        e = ev()
        e.record()
        torch.cuda.synchronize()
        totals["4_backward"] += s.elapsed_time(e)
    opt.step()
    del marks


def backward_only_ms(cg, opt, idx, ckpt: bool) -> float:
    cg._use_activation_checkpointing = ckpt  # type: ignore[attr-defined]
    bl = idx.shape[0]
    opt.zero_grad(set_to_none=True)
    with cg.bypass_lm_head() as lm_w, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            target_h = cg(idx, None)
        ci = torch.rand(bl, SEQ, C, device=DEV)
        u = torch.rand(bl, SEQ, C, device=DEV)
        mi = {SITE: ComponentsMaskInfo(component_mask=ci + (1 - ci) * u, routing_mask="all")}
        pred_h = cg(idx, mask_infos=mi)
        loss, n = fused_linear_kl_div(
            pred_h.reshape(-1, D_MODEL), target_h.reshape(-1, D_MODEL), lm_w
        )
        torch.cuda.synchronize()
        s = ev()
        s.record()
        (loss / n).backward()
        e = ev()
        e.record()
        torch.cuda.synchronize()
    return s.elapsed_time(e)


def module_flop_buckets(cg, idx):
    with (
        FlopCounterMode(display=False) as fc,
        cg.bypass_lm_head(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        torch.no_grad(),
    ):
        cg(idx, None)
    counts = fc.get_flop_counts()
    buckets: dict[str, float] = defaultdict(float)
    per_block: dict[str, float] = defaultdict(float)
    for mod, ops in counts.items():
        if mod == "Global":
            continue
        f = sum(ops.values())
        low = mod.lower()
        if ".attn" in low:
            buckets["attention"] += 0  # counted via leaf modules; bucket below by leaf
        # bucket by leaf-ish substring
    # simpler: re-bucket from Global op totals + name scan on leaf modules
    leaf: dict[str, float] = {}
    for mod, ops in counts.items():
        if mod in ("Global",):
            continue
        leaf[mod] = sum(ops.values())
    # keep only deepest (leaf) modules: a module whose name is a prefix of another is a parent
    names = sorted(leaf)
    is_parent = {n: any(m != n and m.startswith(n + ".") for m in names) for n in names}
    for n, f in leaf.items():
        if is_parent[n]:
            continue
        low = n.lower()
        if "lm_head" in low:
            buckets["lm_head"] += f
        elif "wte" in low or "wpe" in low or "embed" in low:
            buckets["embed"] += f
        elif "_proj" in low and ("attn.q" in low or "attn.k" in low):
            buckets["component_qk"] += f
        elif ".attn" in low:
            buckets["attention"] += f
        elif ".mlp" in low:
            buckets["mlp"] += f
        else:
            buckets["other"] += f
        # per-block
        if ".h." in low:
            blk = low.split(".h.")[1].split(".")[0]
            per_block[blk] += f
    return fc.get_total_flops(), dict(buckets), dict(per_block)


def main() -> None:
    assert torch.cuda.is_available()
    cg = build()
    opt = torch.optim.Adam([p for p in cg.parameters() if p.requires_grad], lr=1e-4)

    print("[A] phase times @ bl=256 (ckpt on):")
    idx = torch.randint(0, VOCAB, (256, SEQ), device=DEV)
    for _ in range(2):
        phase_step(cg, opt, idx, defaultdict(float))
    torch.cuda.synchronize()
    totals: dict[str, float] = defaultdict(float)
    nt = 4
    for _ in range(nt):
        phase_step(cg, opt, idx, totals)
    tot = sum(totals.values()) / nt / 1000
    for k in sorted(totals):
        v = totals[k] / nt / 1000
        print(f"  {k:14s} {v:5.2f}s ({100 * v / tot:4.1f}%)")
    print(f"  {'TOTAL':14s} {tot:5.2f}s")
    del idx
    torch.cuda.empty_cache()

    print("\n[B] backward recompute split @ bl=16:")
    idx16 = torch.randint(0, VOCAB, (16, SEQ), device=DEV)
    for ck in (True, False):
        backward_only_ms(cg, opt, idx16, ck)  # warm
    bwd_ck = min(backward_only_ms(cg, opt, idx16, True) for _ in range(3))
    bwd_nock = min(backward_only_ms(cg, opt, idx16, False) for _ in range(3))
    recompute = max(bwd_ck - bwd_nock, 0.0)
    print(
        f"  backward w/ ckpt : {bwd_ck / 1000:.3f}s   w/o ckpt (grad only): {bwd_nock / 1000:.3f}s"
    )
    print(
        f"  => recompute ≈ {recompute / 1000:.3f}s = {100 * recompute / bwd_ck:.0f}% of backward; "
        f"grad ≈ {100 * bwd_nock / bwd_ck:.0f}%"
    )
    cg._use_activation_checkpointing = True  # type: ignore[attr-defined]
    del idx16
    torch.cuda.empty_cache()

    print("\n[C] one forward's FLOPs by module @ bl=8:")
    idx8 = torch.randint(0, VOCAB, (8, SEQ), device=DEV)
    total, buckets, per_block = module_flop_buckets(cg, idx8)
    for k, v in sorted(buckets.items(), key=lambda x: -x[1]):
        print(f"  {k:14s} {v / 1e12:7.1f} TFLOP ({100 * v / total:4.1f}%)")
    print(f"  {'TOTAL':14s} {total / 1e12:7.1f} TFLOP")
    if per_block:
        blk_vals = sorted(per_block.values())
        print(
            f"  per-block: {len(per_block)} blocks, ~{blk_vals[len(blk_vals) // 2] / 1e12:.2f} TFLOP each "
            f"(uniform → masking site at layer L leaves blocks 0..L-1 == clean target)"
        )


if __name__ == "__main__":
    main()
