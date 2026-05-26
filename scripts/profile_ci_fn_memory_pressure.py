"""Companion to profile_ci_fn_standalone.py: test whether memory pressure or
running the bwd many times in a row reproduces the 600 ms wall time observed
in production (whereas step 0 in production is ~40 ms, matching the standalone).

Hypothesis: in production, subsequent steps see 54 GB resident before 8a begins
(vs 33 GB in step 0). The CUDA caching allocator may need to do synchronous
cudaMalloc/cudaFree to find space for the 10.58 GB of gradient buffers,
serializing onto the bwd stream.
"""

import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from param_decomp.ci_fns import (  # noqa: E402
    GlobalSharedTransformerCiFn,
    TargetLayerConfig,
)
from param_decomp.ci_sigmoids import SIGMOID_TYPES  # noqa: E402


N_SITES = 96
SITE_INPUT_DIM = 1600
C_PER_SITE = 1024
D_MODEL = 4096
N_LAYERS = 8
N_HEADS = 32
MAX_LEN = 1024
MLP_HIDDEN_DIMS = [16384]
BATCH = 2
SEQ = 1024
DEVICE = "cuda"


def build_ci_fn() -> GlobalSharedTransformerCiFn:
    target_cfgs = {
        f"h.{i // 2}.attn.{'q' if i % 2 == 0 else 'k'}_proj": TargetLayerConfig(
            input_dim=SITE_INPUT_DIM, C=C_PER_SITE
        )
        for i in range(N_SITES)
    }
    return GlobalSharedTransformerCiFn(
        target_model_layer_configs=target_cfgs,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        max_len=MAX_LEN,
        mlp_hidden_dims=MLP_HIDDEN_DIMS,
    ).to(DEVICE)


def build_inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        f"h.{i // 2}.attn.{'q' if i % 2 == 0 else 'k'}_proj": torch.randn(
            BATCH, SEQ, SITE_INPUT_DIM, device=DEVICE, dtype=torch.float32
        )
        for i in range(N_SITES)
    }


def measure_bwd_ms(ci_fn, inputs, lower_leaky_fn, grad_seeds) -> tuple[float, float]:
    """Return (wall_ms, gpu_event_ms)."""
    for p in ci_fn.parameters():
        p.grad = None
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        ci_out = ci_fn(inputs)
    lower = lower_leaky_fn(ci_out)
    splits = list(torch.split(lower, ci_fn.split_sizes, dim=-1))
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    start.record()
    torch.autograd.backward(splits, grad_seeds, retain_graph=True)
    end.record()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000, start.elapsed_time(end)


def main():
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    print(f"device: {torch.cuda.get_device_name(0)}")

    ci_fn = build_ci_fn()
    lower_leaky_fn = SIGMOID_TYPES["lower_leaky_hard"]
    inputs = build_inputs()
    grad_seeds = [
        torch.randn(BATCH, SEQ, C_PER_SITE, device=DEVICE, dtype=torch.float32)
        for _ in ci_fn.layer_order
    ]

    def mem_gb() -> tuple[float, float]:
        return (
            torch.cuda.memory_allocated() / 1e9,
            torch.cuda.memory_reserved() / 1e9,
        )

    print(f"\nbaseline mem (alloc / reserved): {mem_gb()}")

    print("\n=== 10 iters bwd (no pressure) ===")
    for i in range(10):
        wall, gpu = measure_bwd_ms(ci_fn, inputs, lower_leaky_fn, grad_seeds)
        a, r = mem_gb()
        print(f"  iter {i}: wall={wall:.1f}ms gpu_event={gpu:.1f}ms alloc={a:.2f}gb reserved={r:.2f}gb")

    # Apply 21 GB of resident "background" memory to mimic production state
    # (33 → 54 GB jump observed between step 0 and step 1+ in slurm log).
    print("\n=== Allocating 21 GB of background tensors to mimic production memory pressure ===")
    n_floats = int(21 * 1024**3 / 4)  # 21 GB as fp32
    bg = torch.empty(n_floats, device=DEVICE, dtype=torch.float32)
    bg.fill_(0)
    a, r = mem_gb()
    print(f"  after bg alloc: alloc={a:.2f}gb reserved={r:.2f}gb")

    print("\n=== 10 iters bwd (with 21 GB background) ===")
    for i in range(10):
        wall, gpu = measure_bwd_ms(ci_fn, inputs, lower_leaky_fn, grad_seeds)
        a, r = mem_gb()
        print(f"  iter {i}: wall={wall:.1f}ms gpu_event={gpu:.1f}ms alloc={a:.2f}gb reserved={r:.2f}gb")

    del bg
    torch.cuda.empty_cache()
    print(f"\n  after bg dealloc + empty_cache: {mem_gb()}")

    # Fragmentation test: allocate and free many small tensors to fragment
    print("\n=== Fragmenting with many small tensors then bwd ===")
    junk = [torch.empty(int(1e7), device=DEVICE) for _ in range(200)]
    a, r = mem_gb()
    print(f"  after junk alloc: alloc={a:.2f}gb reserved={r:.2f}gb")
    # Free every other one to fragment
    for i in range(0, 200, 2):
        junk[i] = torch.empty(1, device=DEVICE)
    a, r = mem_gb()
    print(f"  after fragmentation: alloc={a:.2f}gb reserved={r:.2f}gb")

    for i in range(5):
        wall, gpu = measure_bwd_ms(ci_fn, inputs, lower_leaky_fn, grad_seeds)
        a, r = mem_gb()
        print(f"  iter {i}: wall={wall:.1f}ms gpu_event={gpu:.1f}ms alloc={a:.2f}gb reserved={r:.2f}gb")


if __name__ == "__main__":
    main()
