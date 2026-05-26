"""Standalone profile of the LW pool's D3 per-site loop at GPT-2 XL Q/K scale.

Reproduces the ``lw/D3_layerwise`` phase from the 3-pool trainer in a single
process so ``torch.profiler`` can actually run.

What D3 does per step (from ``param_decomp/three_pool/step_layerwise.py``):

    for site in my_owned_sites:  # 8 sites in production
        mask = ci_s + (1 - ci_s) * u
        delta = target_weight(site) - components[site].weight
        delta_mask = torch.rand(...)
        mask_infos = make_mask_infos({site: mask}, weight_deltas_and_masks={...})
        pred = component_model(batch_local, mask_infos=mask_infos)   # full GPT-2 XL fwd
        loss, n = strategy.recon_loss(pred=pred, target=target_local)  # fused_linear_kl
        loss.backward()                                                 # bwd to V/U + CI leaf

So each LW step does 8 full GPT-2 XL forwards + recon backwards, with the
component-replacement hook firing at one site per iter.

Config matches ``param_decomp_lab/experiments/lm/_xl_production/gpt2_xl_qk_smoke.yaml``:
  - target: GPT2Simple at gpt2-xl scale (n_layer=48, n_embd=1600, vocab=50257)
  - 96 sites total (q_proj + k_proj on all 48 layers), each C=1024
  - LW block in production: 8 owned sites
  - per-rank LW batch B=2, S=1024
  - fused KL recon under bf16 autocast

Random weights — we're measuring kernels, not learning. No HF download.

Production observed: lw/D3_layerwise ~748 ms/step (8 sites).
"""

import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, schedule

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from param_decomp.ci_fns import LayerwiseCiConfig  # noqa: E402
from param_decomp.component_model import ComponentModel  # noqa: E402
from param_decomp.decomposition_targets import DecompositionTarget  # noqa: E402
from param_decomp.masks import make_mask_infos  # noqa: E402
from param_decomp.two_pool.loss_strategy import LayerwiseLossStrategy  # noqa: E402
from param_decomp_lab.experiments.lm.pretrain.models.gpt2_simple import (  # noqa: E402
    GPT2Simple,
    GPT2SimpleConfig,
)

# --- Config matching gpt2_xl_qk_smoke.yaml -----------------------------------
N_LAYERS = 48  # gpt2-xl
N_HEADS = 25
N_EMBD = 1600
VOCAB = 50257
BLOCK_SIZE = 1024
C_PER_SITE = 1024
BATCH = 2
SEQ = 1024
N_OWNED_SITES = 8  # one LW block in production
DEVICE = "cuda"


def build_target() -> GPT2Simple:
    cfg = GPT2SimpleConfig(
        model_type="GPT2Simple",
        block_size=BLOCK_SIZE,
        vocab_size=VOCAB,
        n_layer=N_LAYERS,
        n_head=N_HEADS,
        n_embd=N_EMBD,
        flash_attention=True,
    )
    model = GPT2Simple(cfg).to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def build_component_model(target: GPT2Simple) -> tuple[ComponentModel, list[str]]:
    # All 96 q_proj + k_proj sites; pick first 8 (h.0..h.3) as "owned".
    all_sites = [f"h.{i // 2}.attn.{'q' if i % 2 == 0 else 'k'}_proj" for i in range(2 * N_LAYERS)]
    decomposition_targets = [DecompositionTarget(module_path=s, C=C_PER_SITE) for s in all_sites]
    # GPT2Simple.forward returns (logits, loss); the yaml uses output_extract=0.
    run_batch = lambda model, batch: model(batch)[0]

    cm = ComponentModel(
        target_model=target,
        run_batch=run_batch,
        decomposition_targets=decomposition_targets,
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[8]),  # CI fn unused for LW D3
        sigmoid_type="leaky_hard",
    )
    # LW pool drops its CI fn in production — mirror that to save memory.
    cm.drop_ci_fn()
    cm = cm.to(DEVICE)

    # Owned sites: first 8 alternating q/k on layers 0..3 (one LW block).
    owned = all_sites[:N_OWNED_SITES]
    return cm, owned


def lw_d3_once(
    cm: ComponentModel,
    strategy: LayerwiseLossStrategy,
    batch: torch.Tensor,
    target_local: torch.Tensor,
    owned_sites: list[str],
) -> None:
    """Replicate the lw/D3_layerwise body: per-site fwd+bwd through GPT-2 XL."""
    # Fresh CI leaves per step — same shape as production releaved fp32 CI recvs.
    ci_leaves = {
        s: torch.rand(
            BATCH, SEQ, C_PER_SITE, device=DEVICE, dtype=torch.float32, requires_grad=True
        )
        for s in owned_sites
    }
    for s in owned_sites:
        ci_s = ci_leaves[s]
        u = torch.rand_like(ci_s)
        mask = ci_s + (1 - ci_s) * u
        delta = cm.target_weight(s) - cm.components[s].weight
        delta_mask = torch.rand(ci_s.shape[:-1], device=ci_s.device, dtype=ci_s.dtype)
        mask_infos = make_mask_infos(
            {s: mask},
            weight_deltas_and_masks={s: (delta, delta_mask)},
            routing_masks="all",
        )
        pred = cm(batch, mask_infos=mask_infos)
        loss, n_positions = strategy.recon_loss(pred=pred, target=target_local)
        (loss / (n_positions * len(owned_sites))).backward()


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required"
    torch.cuda.set_device(0)
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}")

    target = build_target()
    cm, owned = build_component_model(target)

    n_params_components = sum(p.numel() for s in owned for p in cm.components[s].parameters())
    n_params_target = sum(p.numel() for p in target.parameters())
    print(f"target n_params: {n_params_target:,}")
    print(f"owned components n_params: {n_params_components:,}")
    print(f"owned sites ({len(owned)}): {owned}")

    # Fused KL strategy — bypasses lm_head and uses fused_linear_kl_div.
    strategy = LayerwiseLossStrategy.fused(target.lm_head.weight)

    # Inputs.
    torch.manual_seed(0)
    batch = torch.randint(0, VOCAB, (BATCH, SEQ), device=DEVICE, dtype=torch.long)
    # Target hidden under bypass: GPT2Simple under lm_head=Identity returns hidden state.
    with torch.no_grad(), strategy.context(cm.target_model):
        target_local = cm(batch).detach()

    # ---- Warmup ---------------------------------------------------------
    print("\n=== Warmup (3 iters) ===")
    for i in range(3):
        for p in cm.parameters():
            p.grad = None
        with (
            strategy.context(cm.target_model),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        ):
            lw_d3_once(cm, strategy, batch, target_local, owned)
        torch.cuda.synchronize()
        print(f"  warmup {i} done")

    # ---- Wall-clock + GPU-event timing (10 iters) ----------------------
    print("\n=== Timed (10 iters) ===")
    wall_ms: list[float] = []
    gpu_ms: list[float] = []
    for _ in range(10):
        for p in cm.parameters():
            p.grad = None
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        start.record()
        with (
            strategy.context(cm.target_model),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        ):
            lw_d3_once(cm, strategy, batch, target_local, owned)
        end.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        wall_ms.append((t1 - t0) * 1000)
        gpu_ms.append(start.elapsed_time(end))

    wall_mean = sum(wall_ms) / len(wall_ms)
    gpu_mean = sum(gpu_ms) / len(gpu_ms)
    print(f"  wall ms : mean={wall_mean:.1f}  min={min(wall_ms):.1f}  max={max(wall_ms):.1f}")
    print(f"  gpu  ms : mean={gpu_mean:.1f}  min={min(gpu_ms):.1f}  max={max(gpu_ms):.1f}")

    # ---- torch.profiler -----------------------------------------------
    print("\n=== torch.profiler ===")
    out_dir = Path(__file__).resolve().parent
    trace_path = out_dir / "lw_d3_trace.json"
    text_path = out_dir / "lw_d3_output.txt"

    prof_schedule = schedule(wait=1, warmup=1, active=3, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule,
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        for _ in range(5):
            for p in cm.parameters():
                p.grad = None
            with (
                strategy.context(cm.target_model),
                torch.autocast(device_type="cuda", dtype=torch.bfloat16),
            ):
                lw_d3_once(cm, strategy, batch, target_local, owned)
            torch.cuda.synchronize()
            prof.step()

    prof.export_chrome_trace(str(trace_path))
    print(f"  trace: {trace_path}")

    sections: list[str] = []
    sections.append(f"device: {torch.cuda.get_device_name(0)}")
    sections.append(f"torch: {torch.__version__}")
    sections.append(f"target n_params: {n_params_target:,}")
    sections.append(f"owned components n_params: {n_params_components:,}")
    sections.append(f"wall ms (mean over 10): {wall_mean:.1f}")
    sections.append(f"gpu ms (mean over 10): {gpu_mean:.1f}")
    sections.append("\nproduction observed: lw/D3_layerwise ~748 ms/step (8 sites)")
    sections.append("")
    sections.append("=" * 80)
    sections.append("TOP 25 by self_cuda_time_total")
    sections.append("=" * 80)
    sections.append(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
    sections.append("\n" + "=" * 80)
    sections.append("TOP 25 by self_cpu_time_total")
    sections.append("=" * 80)
    sections.append(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25))

    out_text = "\n".join(sections)
    print(out_text)
    text_path.write_text(out_text)
    print(f"\nfull text saved to: {text_path}")


if __name__ == "__main__":
    main()
