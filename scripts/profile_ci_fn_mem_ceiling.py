"""Memory-ceiling probe for the CI pool at GPT-2 XL Q/K scale.

Faithfully reproduces what one CI rank holds during a steady-state 3-pool step,
so we can read peak `torch.cuda.max_memory_allocated()` as a function of the
per-rank batch `batch_local_ci`. Unlike profile_ci_fn_standalone.py, this
includes the AdamW optimizer state for the CI fn (2 fp32 moments per param) and
runs a full fwd -> sigmoid/split -> bwd (both grad seeds) -> optimizer.step()
loop, which is the real peak-memory cycle on a CI rank.

It does NOT model the target model's forward (the CI rank caches per-site
pre-weight acts of shape [B, S, 1600] under no_grad; modeled here as plain input
tensors) nor cross-pool comm buffers (g_ci recv buffers, modeled as the grad
seeds). These are minor relative to the CI fn graph + optimizer state.

Usage: python scripts/profile_ci_fn_mem_ceiling.py [bl1 bl2 ...]
Default sweep: 4 6 8 10 12.
"""

import sys
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
N_BLOCKS = 8
N_HEADS = 32
MAX_LEN = 1024
MLP_HIDDEN_DIMS = [16384]
SEQ = 1024
DEVICE = "cuda"

CI_FN_LR = 1e-4


def site_name(i: int) -> str:
    return f"h.{i // 2}.attn.{'q' if i % 2 == 0 else 'k'}_proj"


def build_ci_fn() -> GlobalSharedTransformerCiFn:
    target_cfgs = {
        site_name(i): TargetLayerConfig(input_dim=SITE_INPUT_DIM, C=C_PER_SITE)
        for i in range(N_SITES)
    }
    return GlobalSharedTransformerCiFn(
        target_model_layer_configs=target_cfgs,
        d_model=D_MODEL,
        n_layers=N_BLOCKS,
        n_heads=N_HEADS,
        max_len=MAX_LEN,
        mlp_hidden_dims=MLP_HIDDEN_DIMS,
    ).to(DEVICE)


def build_inputs(bl: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        site_name(i): torch.randn(
            bl, SEQ, SITE_INPUT_DIM, device=DEVICE, dtype=torch.float32, requires_grad=False
        )
        for i in range(N_SITES)
    }


def build_grad_seeds(layer_order: list[str], bl: int) -> list[torch.Tensor]:
    return [
        torch.randn(bl, SEQ, C_PER_SITE, device=DEVICE, dtype=torch.float32) for _ in layer_order
    ]


def run_step(ci_fn, inputs, grad_seeds, lower_leaky_fn, optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        ci_out = ci_fn(inputs)
    lower = lower_leaky_fn(ci_out)
    splits = list(torch.split(lower, ci_fn.split_sizes, dim=-1))
    # mirror step_ci: downstream grads injected on lower, plus an imp-min seed
    # entering via the same graph. We approximate the imp-min path's extra graph
    # cost by summing |ci_out| as a scalar and backward'ing it too (retain_graph).
    torch.autograd.backward(splits, grad_seeds, retain_graph=True)
    imp_proxy = ci_out.abs().sum()
    torch.autograd.backward([imp_proxy], [None])
    optimizer.step()


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required"
    torch.cuda.set_device(0)
    bls = [int(x) for x in sys.argv[1:]] or [4, 6, 8, 10, 12]

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}")
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"device total mem: {total_gb:.1f} GB")

    ci_fn = build_ci_fn()
    n_params = sum(p.numel() for p in ci_fn.parameters())
    print(f"ci_fn n_params: {n_params:,}  ({n_params * 4 / 1e9:.2f} GB fp32 weights)")
    lower_leaky_fn = SIGMOID_TYPES["lower_leaky_hard"]
    optimizer = torch.optim.AdamW(ci_fn.parameters(), lr=CI_FN_LR)

    after_model_gb = torch.cuda.memory_allocated() / 1e9
    print(f"after model build (weights only): {after_model_gb:.2f} GB allocated")
    print(f"{'=' * 70}")
    print(f"{'bl':>4} {'peak_alloc_GB':>14} {'peak_reserved_GB':>17} {'fits_72GB':>10}")
    print(f"{'=' * 70}")

    for bl in bls:
        inputs = build_inputs(bl)
        grad_seeds = build_grad_seeds(ci_fn.layer_order, bl)
        # warmup: build adam state + cache allocator pools
        for _ in range(2):
            run_step(ci_fn, inputs, grad_seeds, lower_leaky_fn, optimizer)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            run_step(ci_fn, inputs, grad_seeds, lower_leaky_fn, optimizer)
        torch.cuda.synchronize()
        peak_alloc = torch.cuda.max_memory_allocated() / 1e9
        peak_reserved = torch.cuda.max_memory_reserved() / 1e9
        fits = "yes" if peak_alloc <= 72.0 else "NO"
        print(f"{bl:>4} {peak_alloc:>14.2f} {peak_reserved:>17.2f} {fits:>10}")
        del inputs, grad_seeds
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
