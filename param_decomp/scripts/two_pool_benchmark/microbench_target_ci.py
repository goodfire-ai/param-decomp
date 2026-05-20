"""Minimal microbench: time each piece of `a/1_target_and_ci_fwd` in isolation.

Single-GPU, no distributed, no comm. Just runs the same shapes as a maxA pool-A
rank (batch=66, seq=1024, 1 owned site) and reports cuda-sync'd timings for:

  - target forward only (no hooks)
  - target forward with cache_type="input" hooks
  - ci_fn forward alone (given cached pre-weight-acts)
  - sigmoid + assertion step alone
  - full calc_causal_importances
  - full step_pool_a's a/1 (target + cache + CI)

Run all 42 sites in sequence to see per-site variation (mlp.down_proj has
d_in=3072, attn projections have d_in=768).

Run via:
    sbatch param_decomp/scripts/two_pool_benchmark/microbench_target_ci.sbatch
"""

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownLambdaType=false

import statistics
import time

import torch

from param_decomp.configs import AttnConfig, GlobalCiConfig, GlobalSharedTransformerCiConfig
from param_decomp.models.batch_and_loss_fns import run_batch_passthrough
from param_decomp.models.component_model import ComponentModel
from param_decomp.scripts.two_pool_benchmark._tiny_model import TinyTransformer, sites_for_block
from param_decomp.utils.module_utils import ModulePathInfo

VOCAB = 8192
D_MODEL = 768
N_HEADS = 12
D_MLP = 3072
N_BLOCKS = 6
BATCH = 66
SEQ = 1024
C = 32
CI_D_MODEL = 128
CI_N_BLOCKS = 2
CI_N_HEADS = 4


def make_ci_config() -> GlobalCiConfig:
    return GlobalCiConfig(
        fn_type="global_shared_transformer",
        simple_transformer_ci_cfg=GlobalSharedTransformerCiConfig(
            d_model=CI_D_MODEL,
            n_blocks=CI_N_BLOCKS,
            attn_config=AttnConfig(n_heads=CI_N_HEADS),
        ),
    )


N_WARMUP = 3
N_PROFILE = 8


def time_phase(fn, name: str, n: int = N_PROFILE, warmup: int = N_WARMUP) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        _ = fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    avg = statistics.mean(ts)
    print(f"  {name:42s} avg={avg:7.2f}ms  min={min(ts):7.2f}ms  max={max(ts):7.2f}ms  (n={n})")
    return avg


def main() -> None:
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")

    target = TinyTransformer(VOCAB, D_MODEL, N_BLOCKS, N_HEADS, D_MLP).to(device)
    target.requires_grad_(False)

    all_sites = [s for b in range(N_BLOCKS) for s in sites_for_block(b)]
    print(
        f"batch={BATCH} seq={SEQ} d={D_MODEL} d_mlp={D_MLP} n_blocks={N_BLOCKS} C={C} "
        f"ci_d_model={CI_D_MODEL} ci_n_blocks={CI_N_BLOCKS} ci_n_heads={CI_N_HEADS}"
    )
    print(f"sites total: {len(all_sites)}\n")

    # 1. Target forward only (no hooks, no CI).
    target_only_model = ComponentModel(
        target_model=target,
        run_batch=run_batch_passthrough,
        module_path_info=[ModulePathInfo(module_path=all_sites[0], C=C)],
        ci_config=make_ci_config(),
        sigmoid_type="leaky_hard",
    ).to(device)
    batch = torch.randint(0, VOCAB, (BATCH, SEQ), device=device)

    print("=== target only (no hooks) ===")
    target_only_time = time_phase(lambda: target(batch), "target_fwd_raw")

    print("\n=== ComponentModel forward, cache_type=none ===")
    time_phase(lambda: target_only_model(batch, cache_type="none"), "component_model_no_cache")

    print("\n=== ComponentModel forward, cache_type=input ===")
    out = target_only_model(batch, cache_type="input")
    time_phase(lambda: target_only_model(batch, cache_type="input"), "component_model_input_cache")

    print("\n=== ci_fn forward only (1 site, owned=blocks.0.attn.q_proj) ===")
    time_phase(lambda: target_only_model.ci_fn(out.cache), "ci_fn_alone_1site")

    print("\n=== sigmoid + assertions (1 site) ===")
    raw = target_only_model.ci_fn(out.cache)
    time_phase(
        lambda: target_only_model._apply_sigmoid_to_ci_outputs(raw, "continuous"),
        "sigmoid_asserts_1site",
    )

    print("\n=== full calc_causal_importances (1 site) ===")
    time_phase(
        lambda: target_only_model.calc_causal_importances(
            pre_weight_acts=out.cache,
            sampling="continuous",
            detach_inputs=False,
        ),
        "calc_ci_full_1site",
    )

    print("\n=== full a/1 (target+cache+CI, 1 site) ===")

    def full_a1_1site():
        o = target_only_model(batch, cache_type="input")
        return target_only_model.calc_causal_importances(
            pre_weight_acts=o.cache,
            sampling="continuous",
            detach_inputs=False,
        )

    a1_1site_time = time_phase(full_a1_1site, "full_a1_1site")

    # 2. Now do per-site CI fn cost. Build a ComponentModel for each site
    #    and time JUST its ci_fn. This breaks down where the cost varies.
    print("\n=== per-site CI fn cost (one site at a time) ===")
    per_site_ms: dict[str, float] = {}
    for site in all_sites[:7]:  # one full block to see variation
        cm = ComponentModel(
            target_model=target,
            run_batch=run_batch_passthrough,
            module_path_info=[ModulePathInfo(module_path=site, C=C)],
            ci_config=make_ci_config(),
            sigmoid_type="leaky_hard",
        ).to(device)
        o = cm(batch, cache_type="input")
        d_in = o.cache[site].shape[-1]
        avg = time_phase(
            lambda cm=cm, o=o: cm.ci_fn(o.cache),
            f"  {site} (d_in={d_in})",
            n=4,
            warmup=2,
        )
        per_site_ms[site] = avg

    # 3. Wider-equivalent: 3 sites per rank.
    print("\n=== full a/1 (3 sites, as wider) ===")
    cm3 = ComponentModel(
        target_model=target,
        run_batch=run_batch_passthrough,
        module_path_info=[ModulePathInfo(module_path=s, C=C) for s in all_sites[:3]],
        ci_config=make_ci_config(),
        sigmoid_type="leaky_hard",
    ).to(device)

    def full_a1_3sites():
        o = cm3(batch, cache_type="input")
        return cm3.calc_causal_importances(
            pre_weight_acts=o.cache,
            sampling="continuous",
            detach_inputs=False,
        )

    a1_3sites_time = time_phase(full_a1_3sites, "full_a1_3sites")

    print(
        f"\nsummary:\n"
        f"  raw target fwd:      {target_only_time:.2f}ms\n"
        f"  full a/1 with 1 site: {a1_1site_time:.2f}ms  (expected ≈ maxA's 78ms)\n"
        f"  full a/1 with 3 sites: {a1_3sites_time:.2f}ms (expected ≈ wider's 121ms)\n"
    )


if __name__ == "__main__":
    main()
