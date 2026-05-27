"""Standalone CPU/single-GPU timing of each 3-pool slow eval metric on production-sized
CI tensors. Skips the cross-pool collectives — just measures the per-rank compute/plot
cost, to identify which metric (if any) is the wall-time offender.

Shapes match GPT2-XL Q/K production after gather_all_tensors on the 4-rank PPGD pool:
  96 sites, batch=128, seq=1024, C=1024
i.e. each site's CI tensor is [128, 1024, 1024] fp32 = 512 MB.

Usage:
    PD_SLOW_BENCH_DEVICE=cpu python -m scripts.bench_slow_eval_metrics
    PD_SLOW_BENCH_DEVICE=cuda:0 python -m scripts.bench_slow_eval_metrics
"""

import gc
import os
import time

import torch

from param_decomp_lab.eval_metrics.plotting import (
    plot_ci_values_histograms,
    plot_component_activation_density,
    plot_mean_component_cis_both_scales,
)

N_SITES = 96
BATCH = 128
SEQ = 1024
C = 1024
DEVICE = os.environ.get("PD_SLOW_BENCH_DEVICE", "cpu")


def _site_names() -> list[str]:
    out = []
    for layer in range(48):
        out.append(f"h.{layer}.attn.q_proj")
        out.append(f"h.{layer}.attn.k_proj")
    assert len(out) == N_SITES
    return out


def _make_ci_dict(small: bool = False) -> dict[str, torch.Tensor]:
    b, s, c = (4, 16, C) if small else (BATCH, SEQ, C)
    return {name: torch.rand((b, s, c), device=DEVICE) for name in _site_names()}


def _make_per_component_dict() -> dict[str, torch.Tensor]:
    return {name: torch.rand((C,), device=DEVICE) for name in _site_names()}


def time_block(label: str, fn) -> float:
    gc.collect()
    t0 = time.time()
    result = fn()
    dt = time.time() - t0
    print(f"  [{label}] {dt:.2f}s")
    del result
    gc.collect()
    return dt


def bench_plot_ci_histograms(ci: dict[str, torch.Tensor]) -> None:
    print("plot_ci_values_histograms (CIHistograms.compute() figure gen):")
    time_block("96 sites × [128,1024,1024]", lambda: plot_ci_values_histograms(ci))


def bench_plot_component_activation_density(per_component: dict[str, torch.Tensor]) -> None:
    print("plot_component_activation_density:")
    time_block("96 sites × [1024]", lambda: plot_component_activation_density(per_component))


def bench_plot_mean_component_cis_both_scales(per_component: dict[str, torch.Tensor]) -> None:
    print("plot_mean_component_cis_both_scales (CIMeanPerComponent.compute()):")
    time_block("96 sites × [1024]", lambda: plot_mean_component_cis_both_scales(per_component))


def bench_cihistograms_data_prep(ci: dict[str, torch.Tensor]) -> None:
    """Just the .flatten().cpu().numpy() per site — no matplotlib."""
    print("CIHistograms data prep only (flatten→cpu→numpy per site):")

    def _go():
        out = {}
        for k, v in ci.items():
            out[k] = v.flatten().cpu().numpy()
        return out

    time_block("96 sites × [128,1024,1024] flatten+cpu+numpy", _go)


def main() -> None:
    print(f"Device: {DEVICE}")
    print(
        f"Per-site CI tensor: [{BATCH}, {SEQ}, {C}] fp32 = "
        f"{BATCH * SEQ * C * 4 / 1e9:.2f} GB / site"
    )
    print(f"Total CI memory: {N_SITES * BATCH * SEQ * C * 4 / 1e9:.1f} GB")
    print()

    print("=== Allocating dummy CI tensors ===")
    t0 = time.time()
    ci = _make_ci_dict()
    print(f"  alloc took {time.time() - t0:.2f}s")
    print()

    print("=== Bench: data-prep only (no matplotlib) ===")
    bench_cihistograms_data_prep(ci)
    print()

    print("=== Bench: CIHistograms full plot ===")
    bench_plot_ci_histograms(ci)
    print()

    del ci
    gc.collect()

    print("=== Bench: per-component plot fns (small input) ===")
    per_comp = _make_per_component_dict()
    bench_plot_component_activation_density(per_comp)
    bench_plot_mean_component_cis_both_scales(per_comp)


if __name__ == "__main__":
    main()
