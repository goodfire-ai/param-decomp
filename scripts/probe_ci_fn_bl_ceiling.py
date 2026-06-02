"""Single-GPU bl-ceiling + activation-checkpointing gain for the GPT2-XL Q/K CI fn.

Builds `GlobalSharedTransformerCiFn` at production scale (96 sites, d_model 4096,
8 blocks, ~2B params) and runs fwd → lower-leaky → split → bwd → Adam.step at a
sweep of per-rank batch sizes (`bl`), with and without `torch.utils.checkpoint`
on the transformer blocks. Reports peak GPU memory and the OOM boundary for each.

This is the single-node CI-pool scaffolding: it answers "what bl_ci fits, and how
much does checkpointing buy" on ONE GPU, instead of multi-node 3-pool OOM probes.

Run (memory sweep): srun --gres=gpu:1 python scripts/probe_ci_fn_bl_ceiling.py
Run (per-kernel profile): srun --gres=gpu:1 python scripts/probe_ci_fn_bl_ceiling.py --profile
"""

import gc
import sys
from collections.abc import Callable
from pathlib import Path

import fire
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from param_decomp.ci_fns import GlobalSharedTransformerCiFn, TargetLayerConfig  # noqa: E402
from param_decomp.ci_sigmoids import SIGMOID_TYPES  # noqa: E402

N_SITES, IN_DIM, C, D_MODEL, N_LAYERS, N_HEADS, MAX_LEN, MLP, SEQ = (
    96,
    1600,
    1024,
    4096,
    8,
    32,
    1024,
    [16384],
    1024,
)
DEV = "cuda"
SITES = [f"h.{i // 2}.attn.{'q' if i % 2 == 0 else 'k'}_proj" for i in range(N_SITES)]
BLS = [4, 8, 16, 24, 32, 48, 64]
PROFILE_BL = 8


class Ckpt(torch.nn.Module):
    def __init__(self, b: torch.nn.Module):
        super().__init__()
        self.b = b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.utils.checkpoint.checkpoint(self.b, x, use_reentrant=False)


def build(ckpt: bool) -> GlobalSharedTransformerCiFn:
    cfgs = {s: TargetLayerConfig(input_dim=IN_DIM, C=C) for s in SITES}
    m = GlobalSharedTransformerCiFn(
        target_model_layer_configs=cfgs,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        max_len=MAX_LEN,
        mlp_hidden_dims=MLP,
    ).to(DEV)
    if ckpt:
        m._blocks = torch.nn.ModuleList([Ckpt(b) for b in m._blocks])
    return m


def run(bl: int, ckpt: bool) -> tuple[str, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    lower = SIGMOID_TYPES["lower_leaky_hard"]
    try:
        m = build(ckpt)
        opt = torch.optim.Adam(m.parameters(), lr=1e-4)
        inputs = {s: torch.randn(bl, SEQ, IN_DIM, device=DEV) for s in SITES}
        seeds = [torch.randn(bl, SEQ, C, device=DEV) for _ in SITES]
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = m(inputs)
        splits = list(torch.split(lower(out), m.split_sizes, dim=-1))
        torch.autograd.backward(splits, seeds)
        opt.step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
        del m, opt, inputs, seeds, out, splits
        return "OK", peak
    except torch.cuda.OutOfMemoryError:
        return "OOM", torch.cuda.max_memory_allocated() / 1e9
    finally:
        torch.cuda.empty_cache()
        gc.collect()


def ci_fn_step(
    m: GlobalSharedTransformerCiFn,
    opt: torch.optim.Optimizer,
    inputs: dict[str, torch.Tensor],
    seeds: list[torch.Tensor],
    lower: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """One CI-fn step: fwd → lower-leaky → split → bwd → Adam."""
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = m(inputs)
    splits = list(torch.split(lower(out), m.split_sizes, dim=-1))
    torch.autograd.backward(splits, seeds)
    opt.step()


def profile_step() -> None:
    """Per-kernel torch.profiler breakdown of the CI-fn fwd+bwd at a fixed bl (single-GPU)."""
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    bl = PROFILE_BL
    lower = SIGMOID_TYPES["lower_leaky_hard"]
    print(
        f"GPU {torch.cuda.get_device_name(0)} | profiling GPT2-XL Q/K CI-fn phase "
        f"(fwd→lower-leaky→split→bwd, {N_SITES} sites, d{D_MODEL}, {N_LAYERS} blocks) at "
        f"bl={bl}, seq={SEQ}, ckpt ON, single-GPU no-comm\n"
    )
    m = build(ckpt=True)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    inputs = {s: torch.randn(bl, SEQ, IN_DIM, device=DEV) for s in SITES}
    seeds = [torch.randn(bl, SEQ, C, device=DEV) for _ in SITES]

    for _ in range(3):  # warmup (alloc/autotune)
        ci_fn_step(m, opt, inputs, seeds, lower)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        ci_fn_step(m, opt, inputs, seeds, lower)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))


def main(profile: bool = False) -> None:
    if profile:
        profile_step()
        return
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    cap = torch.cuda.get_device_properties(0).total_memory / 1e9
    np = sum(p.numel() for p in build(False).parameters())
    torch.cuda.empty_cache()
    print(
        f"GPU {torch.cuda.get_device_name(0)} ~{cap:.0f}GB | CI fn {np / 1e9:.2f}B params "
        f"({N_SITES} sites, d{D_MODEL}, {N_LAYERS} blocks), seq {SEQ}"
    )
    print(f"\n{'bl':>4} | {'plain peak':>14} | {'ckpt peak':>14}")
    print("-" * 40)

    def fmt(r: tuple[str, float]) -> str:
        return "OOM" if r[0] == "OOM" else "skip" if r[0] == "skip" else f"{r[1]:.1f}GB"

    plain_oom = ckpt_oom = False
    for bl in BLS:
        rp = ("skip", 0.0) if plain_oom else run(bl, False)
        rc = ("skip", 0.0) if ckpt_oom else run(bl, True)
        plain_oom = plain_oom or rp[0] == "OOM"
        ckpt_oom = ckpt_oom or rc[0] == "OOM"
        print(f"{bl:>4} | {fmt(rp):>14} | {fmt(rc):>14}", flush=True)
    print("\n=> plain bl_ci ceiling: last OK below first OOM | checkpointing extends it")


if __name__ == "__main__":
    fire.Fire(main)
