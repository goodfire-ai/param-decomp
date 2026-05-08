"""Decompose the ~3 GB/B activation finding from Phase 1.

Phase 1 measured ~3 GB peak activation per per-rank batch element on Jose, vs the
report's ~1 GB/B prediction. This benchmark exercises a stripped-down forward+
backward through the actual ComponentModel at Jose dims, varying which pieces are
in play, and reports peak/B for each. Single GPU.

Configurations:
  bare_target            target-only forward (no components, no CI)
  + components fwd       one component forward, no backward, no PPGD
  + components fwd+bwd   one forward+backward through component
  + ci_fn                additional forward through GlobalSharedTransformerCiFn
  full_step (no PPGD)    same as Phase 1 setup minus PPGD warmup

This isolates the contribution of each piece to peak/B.
"""

from __future__ import annotations

import argparse

import torch

from param_decomp.configs import (
    AttnConfig,
    GlobalCiConfig,
    GlobalSharedTransformerCiConfig,
    ModulePatternInfoConfig,
)
from param_decomp.models.batch_and_loss_fns import make_run_batch
from param_decomp.models.component_model import ComponentModel
from param_decomp.pretrain.models.llama_simple_mlp import (
    LlamaSimpleMLP,
    LlamaSimpleMLPConfig,
)
from param_decomp.utils.module_utils import expand_module_patterns


def build_target() -> LlamaSimpleMLP:
    """Match Jose's actual t-9d2b8f02 target dims."""
    cfg = LlamaSimpleMLPConfig(
        model_type="LlamaSimpleMLP",
        n_layer=4,
        n_embd=768,
        n_head=6,
        n_intermediate=3072,
        vocab_size=50277,
        n_ctx=512,
        block_size=512,
        n_key_value_heads=6,
        rotary_dim=128,
        use_grouped_query_attention=True,
    )
    return LlamaSimpleMLP(cfg)


def build_ci_config() -> GlobalCiConfig:
    """Match Jose's CI fn config from pile_llama_simple_mlp-4L.yaml."""
    return GlobalCiConfig(
        mode="global",
        fn_type="global_shared_transformer",
        simple_transformer_ci_cfg=GlobalSharedTransformerCiConfig(
            d_model=2048,
            n_blocks=8,
            mlp_hidden_dim=[8192],
            attn_config=AttnConfig(n_heads=16, max_len=512, rope_base=10000.0),
        ),
    )


JOSE_MODULE_INFO = [
    ModulePatternInfoConfig(module_pattern="h.*.mlp.c_fc", C=3072),
    ModulePatternInfoConfig(module_pattern="h.*.mlp.down_proj", C=3584),
    ModulePatternInfoConfig(module_pattern="h.*.attn.q_proj", C=512),
    ModulePatternInfoConfig(module_pattern="h.*.attn.k_proj", C=512),
    ModulePatternInfoConfig(module_pattern="h.*.attn.v_proj", C=1024),
    ModulePatternInfoConfig(module_pattern="h.*.attn.o_proj", C=1024),
]


def measure(label: str, fn, device: torch.device) -> tuple[str, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"  {label:40s}  peak {peak:6.2f} GB")
    return label, peak


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=512)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    torch.manual_seed(0)

    print(f"Batch={args.batch}, seq={args.seq}")
    target = build_target().to(device)
    target.requires_grad_(False)

    fixed_params_gb = sum(p.numel() for p in target.parameters()) * 4 / 1e9
    print(f"Target params: {fixed_params_gb:.3f} GB fp32 (frozen)")

    module_path_info = expand_module_patterns(target, JOSE_MODULE_INFO)
    cm = ComponentModel(
        target_model=target,
        run_batch=make_run_batch(output_extract=0),
        module_path_info=module_path_info,
        ci_config=build_ci_config(),
        sigmoid_type="leaky_hard",
    ).to(device)

    component_params = []
    for name in cm.target_module_paths:
        component_params.extend(cm.components[name].parameters())
    ci_fn_params = list(cm.ci_fn.parameters())
    cm_fixed_gb = (
        (sum(p.numel() for p in component_params) + sum(p.numel() for p in ci_fn_params)) * 4 / 1e9
    )
    print(f"Trainable params: {cm_fixed_gb:.3f} GB fp32")

    idx = torch.randint(0, target.config.vocab_size, (args.batch, args.seq), device=device)

    results = []

    def f1():
        with torch.no_grad():
            target(idx)

    results.append(measure("[fwd] target only (no_grad)", f1, device))

    # (target-only fwd+bwd skipped — target is frozen, no grads to backprop)

    # Target-only forward via ComponentModel (no mask_infos, no component dispatch).
    # Forward only — target is frozen, no grad path.
    def f3():
        cm(idx, cache_type="input")

    results.append(measure("[fwd] CM target-only with input cache", f3, device))

    # Compute CI on the cached input acts. This forward through GlobalSharedTransformerCiFn
    # is the same shape as inside training.
    def f4():
        out = cm(idx, cache_type="input")
        ci = cm.calc_causal_importances(
            pre_weight_acts=out.cache, detach_inputs=False, sampling="continuous"
        )
        # Reduce to a scalar so we can backward through CI fn + target-only logits
        loss = out.output.sum() + sum(v.sum() for v in ci.lower_leaky.values())
        loss.backward()

    results.append(measure("[fwd+bwd] target + CI fn", f4, device))

    # Full component forward through one of the modules' components (no PPGD warmup).
    from param_decomp.models.components import make_mask_infos

    def f5():
        out_cache = cm(idx, cache_type="input")
        ci = cm.calc_causal_importances(
            pre_weight_acts=out_cache.cache, detach_inputs=False, sampling="continuous"
        )
        weight_deltas = cm.calc_weight_deltas()
        delta_masks = {
            name: torch.ones(idx.shape, device=device) for name in cm.target_module_paths
        }
        mask_infos = make_mask_infos(
            component_masks=ci.lower_leaky,
            weight_deltas_and_masks={
                name: (weight_deltas[name], delta_masks[name]) for name in cm.target_module_paths
            },
        )
        out_components = cm(idx, mask_infos=mask_infos)
        loss = out_components.sum() + sum(v.sum() for v in ci.lower_leaky.values())
        loss.backward()

    results.append(measure("[fwd+bwd] full step (no PPGD)", f5, device))

    print()
    print("Summary (peak GB above target params):")
    base_peak = results[0][1]
    for label, peak in results:
        delta = peak - base_peak
        per_b = delta / args.batch
        print(f"  {label:40s}  peak {peak:6.2f} GB  Δvs[0] {delta:+.2f} GB  /B {per_b:+.3f} GB")


if __name__ == "__main__":
    main()
