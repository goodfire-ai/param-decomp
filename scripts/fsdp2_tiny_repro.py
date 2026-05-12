"""Tiny single-GPU FSDP2 repro for fast iteration on the fused-decomposition flow.

Goal: build a 2-layer LlamaSimpleMLP-shape ComponentModel with d_model=32, wrap with
FSDP2, run target + decomposed forwards, find where it breaks. Single process so we
can iterate in seconds without SLURM.

Run with:
    torchrun --standalone --nproc_per_node=1 scripts/fsdp2_tiny_repro.py
"""

from __future__ import annotations

import os
import sys
import traceback

import torch
import torch.distributed as dist

from param_decomp.configs import (
    AttnConfig,
    GlobalCiConfig,
    GlobalSharedTransformerCiConfig,
    ModulePatternInfoConfig,
)
from param_decomp.models.batch_and_loss_fns import make_run_batch
from param_decomp.models.component_model import ComponentModel
from param_decomp.models.mask_info import make_mask_infos
from param_decomp.pretrain.models.llama_simple_mlp import LlamaSimpleMLP, LlamaSimpleMLPConfig
from param_decomp.utils.fsdp import fsdp_wrap
from param_decomp.utils.module_utils import expand_module_patterns


def build_tiny_target() -> LlamaSimpleMLP:
    # Use Jose-scale dims so FSDP2 actually shards params (small dims fall back to Replicate,
    # which hides the DTensor mixing issue).
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


JOSE_MODULE_INFO = [
    ModulePatternInfoConfig(module_pattern="h.*.mlp.c_fc", C=3072),
    ModulePatternInfoConfig(module_pattern="h.*.mlp.down_proj", C=3584),
    ModulePatternInfoConfig(module_pattern="h.*.attn.q_proj", C=512),
    ModulePatternInfoConfig(module_pattern="h.*.attn.k_proj", C=512),
    ModulePatternInfoConfig(module_pattern="h.*.attn.v_proj", C=1024),
    ModulePatternInfoConfig(module_pattern="h.*.attn.o_proj", C=1024),
]


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl")

    def step(label: str, fn) -> None:
        print(f"\n--- {label} ---", flush=True)
        try:
            fn()
            print(f"OK: {label}", flush=True)
        except Exception:
            print(f"FAIL: {label}", flush=True)
            traceback.print_exc()

    target = build_tiny_target().to(device).eval()
    target.requires_grad_(False)

    module_path_info = expand_module_patterns(target, JOSE_MODULE_INFO)
    cm = ComponentModel(
        target_model=target,
        run_batch=make_run_batch(output_extract=0),
        module_path_info=module_path_info,
        ci_config=GlobalCiConfig(
            mode="global",
            fn_type="global_shared_transformer",
            simple_transformer_ci_cfg=GlobalSharedTransformerCiConfig(
                d_model=128,
                n_blocks=2,
                mlp_hidden_dim=[256],
                attn_config=AttnConfig(n_heads=4, max_len=512, rope_base=10000.0),
            ),
        ),
        sigmoid_type="leaky_hard",
    ).to(device)

    print(f"trainable: {sum(p.numel() for p in cm.parameters() if p.requires_grad)}", flush=True)

    wrapped = fsdp_wrap(cm, device_id=local_rank, autocast_bf16=False)

    idx = torch.randint(0, 50277, (2, 64), device=device)

    step("target-only forward", lambda: wrapped(idx))

    def decomposed_forward():
        masks = {
            site_name: torch.ones(2, 64, site.C, device=device)
            for site_name, site in cm.components.items()
        }
        out = wrapped(idx, mask_infos=make_mask_infos(masks))
        loss = out.float().pow(2).mean()
        loss.backward()

    step("decomposed forward + backward", decomposed_forward)

    def decomposed_with_weight_deltas():
        # Mirror the training loop: gather V/U via unshard, materialize deltas as regular
        # Tensors via full_tensor, then run a forward+backward that exercises the
        # weight_delta path in _decomposed_forward (the failure site in Jose runs).
        sites_to_unshard = [s for s in cm.modules() if hasattr(s, "unshard")]
        for s in sites_to_unshard:
            s.unshard()
        try:
            weight_deltas = {}
            for k, v in cm.calc_weight_deltas().items():
                v = v.detach()
                v = v.full_tensor() if hasattr(v, "full_tensor") else v.clone()
                weight_deltas[k] = v
        finally:
            for s in sites_to_unshard:
                s.reshard()

        masks = {
            site_name: torch.ones(2, 64, site.C, device=device)
            for site_name, site in cm.components.items()
        }
        delta_mask = {name: torch.ones(2, 64, device=device) for name in cm.components}
        weight_deltas_and_masks = {
            name: (weight_deltas[name], delta_mask[name]) for name in cm.components
        }
        infos = make_mask_infos(masks, weight_deltas_and_masks=weight_deltas_and_masks)
        out = wrapped(idx, mask_infos=infos)
        loss = out.float().pow(2).mean()
        loss.backward()

    step("decomposed forward + backward (with weight_deltas)", decomposed_with_weight_deltas)

    def jose_like_step():
        """Mimic Jose's pattern: target-only forward, then multiple decomposed forwards
        for different loss flavors, then ONE backward at the end."""
        # 1. compute weight_deltas
        sites_to_unshard = [s for s in cm.modules() if hasattr(s, "unshard")]
        for s in sites_to_unshard:
            s.unshard()
        try:
            weight_deltas = {}
            for k, v in cm.calc_weight_deltas().items():
                v = v.detach()
                weight_deltas[k] = v.full_tensor() if hasattr(v, "full_tensor") else v.clone()
        finally:
            for s in sites_to_unshard:
                s.reshard()

        # 2. target-only forward with cache (like the start of Jose's step)
        target_out = wrapped(idx, cache_type="input")
        target_logits = target_out.output
        cache = target_out.cache

        # 3. CI fn forward (uses cache)
        ci = cm.calc_causal_importances(
            pre_weight_acts=cache, detach_inputs=False, sampling="continuous"
        )

        # 4. multiple decomposed forwards with different masks (mimicking compute_losses)
        delta_masks = {name: torch.ones(2, 64, device=device) for name in cm.components}
        masks_a = ci.lower_leaky
        infos_a = make_mask_infos(
            masks_a,
            weight_deltas_and_masks={n: (weight_deltas[n], delta_masks[n]) for n in cm.components},
        )
        out_a = wrapped(idx, mask_infos=infos_a)

        masks_b = {n: torch.ones_like(v) for n, v in ci.lower_leaky.items()}
        infos_b = make_mask_infos(
            masks_b,
            weight_deltas_and_masks={n: (weight_deltas[n], delta_masks[n]) for n in cm.components},
        )
        out_b = wrapped(idx, mask_infos=infos_b)

        # 5. combined loss
        loss = (
            (out_a - target_logits).pow(2).mean()
            + (out_b - target_logits).pow(2).mean()
            + sum(v.pow(2).sum() for v in weight_deltas.values()) * 1e-3
        )
        loss.backward()

    step("jose-like multi-forward step", jose_like_step)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
