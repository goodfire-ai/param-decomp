"""Microbenchmark: materialize-delta vs forward_with_target_weight.

Runs both paths through every decomposed module at Jose-realistic shapes, measures
peak CUDA memory and wall time per step, and prints the delta. Single-process,
no DDP, no PPGD — strips the SPD machinery down to just the math we changed in
Phase 4.

Use to validate the report's §7d ~320 MB / step prediction (Jose, 10 modules).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch

from param_decomp.models.components import LinearComponents


@dataclass(frozen=True)
class JoseModule:
    name: str
    d_in: int
    d_out: int
    C: int


# Mirrors the actual decomposed modules in Jose (`pile_llama_simple_mlp-4L.yaml`)
# at the measured target dims (n_embd=768, n_intermediate=3072), 4 layers.
JOSE_MODULES_PER_LAYER = [
    JoseModule("c_fc", 768, 3072, 3072),
    JoseModule("down_proj", 3072, 768, 3584),
    JoseModule("q_proj", 768, 768, 512),
    JoseModule("k_proj", 768, 768, 512),
    JoseModule("v_proj", 768, 768, 1024),
    JoseModule("o_proj", 768, 768, 1024),
]


def build_modules(
    n_layers: int, device: torch.device
) -> list[tuple[JoseModule, LinearComponents, torch.Tensor]]:
    out: list[tuple[JoseModule, LinearComponents, torch.Tensor]] = []
    for _ in range(n_layers):
        for spec in JOSE_MODULES_PER_LAYER:
            comp = LinearComponents(C=spec.C, d_in=spec.d_in, d_out=spec.d_out, bias=None).to(
                device
            )
            target_weight = torch.randn(spec.d_out, spec.d_in, device=device)
            out.append((spec, comp, target_weight))
    return out


def run_materialized(
    modules: list[tuple[JoseModule, LinearComponents, torch.Tensor]],
    batch: int,
    seq: int,
    device: torch.device,
) -> tuple[float, float]:
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for spec, comp, W in modules:
        x = torch.randn(batch, seq, spec.d_in, device=device, requires_grad=True)
        mask = torch.rand(batch, seq, spec.C, device=device)
        delta_mask = torch.rand(batch, seq, device=device)
        weight_delta = W - comp.weight  # materializes (d_out, d_in)
        out = comp.forward(x, mask=mask, weight_delta_and_mask=(weight_delta, delta_mask))
        out.sum().backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    return peak_gb, elapsed


def run_rewrite(
    modules: list[tuple[JoseModule, LinearComponents, torch.Tensor]],
    batch: int,
    seq: int,
    device: torch.device,
) -> tuple[float, float]:
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for spec, comp, W in modules:
        x = torch.randn(batch, seq, spec.d_in, device=device, requires_grad=True)
        mask = torch.rand(batch, seq, spec.C, device=device)
        delta_mask = torch.rand(batch, seq, device=device)
        out = comp.forward_with_target_weight(
            x, target_weight=W, mask=mask, weight_delta_mask=delta_mask
        )
        out.sum().backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    return peak_gb, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_iters", type=int, default=3)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "needs CUDA"
    device = torch.device("cuda")

    print(f"Jose-realistic shapes: n_layers={args.n_layers}, batch={args.batch}, seq={args.seq}")
    print(
        f"Modules per layer: {len(JOSE_MODULES_PER_LAYER)}, total: {args.n_layers * len(JOSE_MODULES_PER_LAYER)}"
    )
    torch.manual_seed(0)
    modules = build_modules(args.n_layers, device)

    # Warm up
    run_materialized(modules, args.batch, args.seq, device)
    torch.cuda.empty_cache()

    mat_peaks, mat_times = [], []
    rew_peaks, rew_times = [], []
    for i in range(args.n_iters):
        peak, t = run_materialized(modules, args.batch, args.seq, device)
        mat_peaks.append(peak)
        mat_times.append(t)
        torch.cuda.empty_cache()
        peak, t = run_rewrite(modules, args.batch, args.seq, device)
        rew_peaks.append(peak)
        rew_times.append(t)
        torch.cuda.empty_cache()
        print(
            f"  iter {i}: mat peak {mat_peaks[-1]:.3f} GB / {mat_times[-1] * 1000:.0f} ms;  "
            f"rewrite peak {rew_peaks[-1]:.3f} GB / {rew_times[-1] * 1000:.0f} ms"
        )

    avg_mat_peak = sum(mat_peaks) / len(mat_peaks)
    avg_rew_peak = sum(rew_peaks) / len(rew_peaks)
    avg_mat_t = sum(mat_times) / len(mat_times)
    avg_rew_t = sum(rew_times) / len(rew_times)

    print()
    print(f"Avg materialize-delta peak:  {avg_mat_peak:.3f} GB   ({avg_mat_t * 1000:.0f} ms)")
    print(f"Avg rewrite path peak:       {avg_rew_peak:.3f} GB   ({avg_rew_t * 1000:.0f} ms)")
    print(
        f"Memory delta:                {(avg_mat_peak - avg_rew_peak) * 1000:.0f} MB ({100 * (avg_mat_peak - avg_rew_peak) / avg_mat_peak:.1f}% saved)"
    )
    print(
        f"Time delta:                  {(avg_rew_t - avg_mat_t) * 1000:.0f} ms ({100 * (avg_rew_t - avg_mat_t) / avg_mat_t:.1f}% slower)"
    )


if __name__ == "__main__":
    main()
