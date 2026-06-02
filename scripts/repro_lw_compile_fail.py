"""Isolate the distributed LW-compile failure: BackendCompilerFailed KeyError
'_scaled_dot_product_flash_attention' in the AOTAutograd partitioner (compile + ckpt + flash SDPA).

bench_lw_compile passed (torch.compile, NO weight_delta). The distributed step uses
make_mask_infos WITH weight_deltas_and_masks (_layerwise_one_site) and Module.compile(). This
tests the combinations on a small model (head_dim=64 → flash SDPA selected) to find the trigger.

Run: srun --gres=gpu:1 python scripts/repro_lw_compile_fail.py
"""

import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from param_decomp.components import make_components  # noqa: E402
from param_decomp.fused_linear_kl import fused_linear_kl_div  # noqa: E402
from param_decomp.masks import make_mask_infos  # noqa: E402
from param_decomp_lab.experiments.lm.pretrain.models.gpt2_simple import (  # noqa: E402
    GPT2Simple,
    GPT2SimpleConfig,
)
from param_decomp_lab.experiments.lm.vendored.gpt2 import componentize_gpt2  # noqa: E402

N_LAYER, D_MODEL, N_HEAD, VOCAB, SEQ, C, BL = 48, 1600, 25, 50257, 1024, 1024, 8  # XL, small bl
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))  # set by torchrun; 0 single-process
_NGPU = torch.cuda.device_count() or 1
GPU = LOCAL_RANK % _NGPU  # allow oversubscription (nproc > nGPU) to crank concurrent compilers
DEV = f"cuda:{GPU}"
SITE = "h.0.attn.q_proj"


def build():
    cfg = GPT2SimpleConfig(
        model_type="GPT2Simple",
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=D_MODEL,
        vocab_size=VOCAB,
        block_size=SEQ,
    )
    cg = componentize_gpt2(GPT2Simple(cfg), make_components(GPT2Simple(cfg), {SITE: C})).to(DEV)
    cg.enable_activation_checkpointing()
    return cg


def run_once(*, use_module_compile: bool, weight_delta: bool) -> str:
    torch._dynamo.reset()
    cg = build()
    comp = cg.get_submodule(SITE).components
    model = cg
    if use_module_compile:
        cg.compile()
    else:
        model = torch.compile(cg)
    idx = torch.randint(0, VOCAB, (BL, SEQ), device=DEV)
    # the real _layerwise_one_site CI source is a requires_grad leaf (grad flows back to it),
    # adding a backward path through SDPA → can flip the min-cut toward recomputing SDPA.
    ci = torch.rand(BL, SEQ, C, device=DEV, requires_grad=True)
    u = torch.rand(BL, SEQ, C, device=DEV)
    try:
        with cg.bypass_lm_head() as lm_w, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                target_h = model(idx, None)
            mask = ci + (1 - ci) * u
            if weight_delta:
                delta = cg.target_weight(SITE) - comp.weight
                dm = torch.rand(BL, SEQ, device=DEV)
                mi = make_mask_infos(
                    {SITE: mask}, weight_deltas_and_masks={SITE: (delta, dm)}, routing_masks="all"
                )
            else:
                mi = make_mask_infos({SITE: mask}, routing_masks="all")
            pred_h = model(idx, mask_infos=mi)
            loss, n = fused_linear_kl_div(
                pred_h.reshape(-1, D_MODEL), target_h.reshape(-1, D_MODEL), lm_w
            )
            (loss / n).backward()
        torch.cuda.synchronize()
        return "PASS"
    except Exception as e:  # noqa: BLE001 — diagnostic: report which combo fails
        msg = str(e).splitlines()[-1] if str(e) else type(e).__name__
        return f"FAIL: {type(e).__name__}: {msg[:120]}"


def main() -> None:
    assert torch.cuda.is_available()
    torch.cuda.set_device(GPU)
    torch.set_float32_matmul_precision("high")  # the trainer sets this (optimize.py:221)
    if os.environ.get("FORCE_RECOMPUTE", "").strip() in ("1", "true", "yes"):
        import torch._functorch.config as _fc

        _fc.activation_memory_budget = 0.0  # force the min-cut to recompute everything (incl SDPA)
    if os.environ.get("DIST_INIT", "").strip() in ("1", "true", "yes"):
        # init NCCL before compile (as the real trainer does) — inductor then forks its
        # parallel-compile workers from a process with live NCCL comm threads (fork hazard).
        import torch.distributed as dist

        dist.init_process_group("nccl")
        x = torch.ones(1, device=DEV)
        dist.all_reduce(x)  # force NCCL comm init (threads + state) before compile
        torch.cuda.synchronize()
    if os.environ.get("PERRANK_CACHE", "").strip() in ("1", "true", "yes"):
        # the candidate fix: isolate each rank's inductor + triton cache so concurrent
        # compilers don't race on the shared (username-keyed) default dir.
        user = os.environ.get("USER", "u")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/torchinductor_{user}_r{LOCAL_RANK}"
        os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_{user}_r{LOCAL_RANK}"
    cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR", "<default /tmp/torchinductor_USER (shared)>")
    # Under `torchrun --nproc_per_node=N` this runs N processes compiling the same graph
    # concurrently into the (default) shared inductor cache — the suspected distributed trigger.
    r = run_once(use_module_compile=True, weight_delta=True)
    print(f"[rank{LOCAL_RANK}] Module.compile weight_delta=True: {r}  (cache={cache})", flush=True)


if __name__ == "__main__":
    main()
