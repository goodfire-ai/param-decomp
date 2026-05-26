"""Standalone profile of the PPGD pool's D3 warmup loop at GPT-2 XL Q/K scale.

Reproduces the ``pgd/D3_warmup`` phase from the 3-pool trainer in a single
process so ``torch.profiler`` can actually run.

What D3 warmup does per outer step (from ``step_ppgd.py`` →
``PersistentPGDState.warmup``):

    for _ in range(n_warmup_steps):  # 2 in production
        sum_loss, n = compute_recon_sum_and_n(model, batch, target_out, ci, weight_deltas)
            # builds per-site masks from sources+ci, runs full target fwd (with all
            # 96 sites' components replacing their target modules), fused KL recon
        grads = torch.autograd.grad(sum_loss/n, sources, retain_graph=False)
        self.optimizer.step(sources, grads)  # in-place
        # + clamp sources to [0, 1]

So each warmup iter does 1 full GPT-2 XL fwd with all 96 sites masked, 1 bwd
back to source leaves, 1 source opt step.

Config matches ``param_decomp_lab/experiments/lm/_xl_production/gpt2_xl_qk_smoke.yaml``:
  - target: GPT2Simple at gpt2-xl scale (n_layer=48, n_embd=1600, vocab=50257)
  - 96 sites total (q_proj + k_proj on all 48 layers), each C=1024
  - per-rank PPGD batch B=2, S=1024
  - scope: per_batch_per_position (no cross-rank sync)
  - n_warmup_steps: 2, optimizer: adam (beta1=0.5, beta2=0.99, lr=0.01)
  - use_delta_component: true (source_c = C + 1)

Production observed: pgd/D3_warmup ~400 ms/step.

Random weights — measuring kernels, not learning. No HF download.
"""

import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile
from torch.profiler import schedule as prof_schedule

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from param_decomp.ci_fns import LayerwiseCiConfig  # noqa: E402
from param_decomp.component_model import ComponentModel  # noqa: E402
from param_decomp.decomposition_targets import DecompositionTarget  # noqa: E402
from param_decomp.masks import AllLayersRouter  # noqa: E402
from param_decomp.metrics.persistent_pgd_state import (  # noqa: E402
    AdamPGDConfig,
    PerBatchPerPositionScope,
    PersistentPGDState,
)
from param_decomp.schedule import ScheduleConfig  # noqa: E402
from param_decomp.two_pool.loss_strategy import LayerwiseLossStrategy  # noqa: E402
from param_decomp_lab.experiments.lm.pretrain.models.gpt2_simple import (  # noqa: E402
    GPT2Simple,
    GPT2SimpleConfig,
)

# --- Config matching gpt2_xl_qk_smoke.yaml -----------------------------------
N_LAYERS = 48
N_HEADS = 25
N_EMBD = 1600
VOCAB = 50257
BLOCK_SIZE = 1024
C_PER_SITE = 1024
BATCH = 2
SEQ = 1024
N_WARMUP_STEPS = 2
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
    all_sites = [f"h.{i // 2}.attn.{'q' if i % 2 == 0 else 'k'}_proj" for i in range(2 * N_LAYERS)]
    decomposition_targets = [DecompositionTarget(module_path=s, C=C_PER_SITE) for s in all_sites]
    run_batch = lambda model, batch: model(batch)[0]

    cm = ComponentModel(
        target_model=target,
        run_batch=run_batch,
        decomposition_targets=decomposition_targets,
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[8]),  # CI fn unused on PPGD pool
        sigmoid_type="leaky_hard",
    )
    cm.drop_ci_fn()  # PPGD pool drops the CI fn in production
    cm = cm.to(DEVICE)
    return cm, all_sites


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required"
    torch.cuda.set_device(0)
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}")

    target = build_target()
    cm, all_sites = build_component_model(target)
    n_params_target = sum(p.numel() for p in target.parameters())
    n_params_components = sum(p.numel() for s in all_sites for p in cm.components[s].parameters())
    print(f"target n_params: {n_params_target:,}")
    print(f"all components n_params: {n_params_components:,}  ({len(all_sites)} sites)")

    strategy = LayerwiseLossStrategy.fused(target.lm_head.weight)

    ppgd_state = PersistentPGDState(
        module_to_c={s: C_PER_SITE for s in all_sites},
        batch_dims=(BATCH, SEQ),
        device=DEVICE,
        use_delta_component=True,
        optimizer_cfg=AdamPGDConfig(
            beta1=0.5,
            beta2=0.99,
            eps=1e-8,
            lr_schedule=ScheduleConfig(
                start_val=0.01, warmup_pct=0.025, final_val_frac=1.0, fn_type="constant"
            ),
        ),
        scope=PerBatchPerPositionScope(),
        use_sigmoid_parameterization=False,
        n_warmup_steps=N_WARMUP_STEPS,
        n_samples=1,
        router=AllLayersRouter(),
        reconstruction_loss=strategy.recon_loss,
    )
    ppgd_state.update_lr(step=0, total_steps=50)

    # Inputs.
    torch.manual_seed(0)
    batch = torch.randint(0, VOCAB, (BATCH, SEQ), device=DEVICE, dtype=torch.long)
    # Pre-compute target hidden state once (production gets it from prev phase).
    with torch.no_grad(), strategy.context(cm.target_model):
        target_out = cm(batch).detach()

    def build_fresh_inputs() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # CI scratch — re-leafed fp32 with requires_grad in production (from CI pool recv).
        ci = {
            s: torch.rand(
                BATCH, SEQ, C_PER_SITE, device=DEVICE, dtype=torch.float32, requires_grad=True
            )
            for s in all_sites
        }
        # weight_deltas = target_weight - sum_components, fp32.
        wd = cm.calc_weight_deltas()
        return ci, wd

    def warmup_once() -> None:
        ci, wd = build_fresh_inputs()
        with (
            strategy.context(cm.target_model),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        ):
            ppgd_state.warmup(model=cm, batch=batch, target_out=target_out, ci=ci, weight_deltas=wd)

    # ---- Warmup -----------------------------------------------------------
    print("\n=== Warmup (3 iters) ===")
    for i in range(3):
        warmup_once()
        torch.cuda.synchronize()
        print(f"  warmup {i} done")

    # ---- Timed (10 iters) ------------------------------------------------
    print("\n=== Timed (10 iters) ===")
    wall_ms: list[float] = []
    gpu_ms: list[float] = []
    for _ in range(10):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        start.record()
        warmup_once()
        end.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        wall_ms.append((t1 - t0) * 1000)
        gpu_ms.append(start.elapsed_time(end))

    wall_mean = sum(wall_ms) / len(wall_ms)
    gpu_mean = sum(gpu_ms) / len(gpu_ms)
    print(f"  wall ms : mean={wall_mean:.1f}  min={min(wall_ms):.1f}  max={max(wall_ms):.1f}")
    print(f"  gpu  ms : mean={gpu_mean:.1f}  min={min(gpu_ms):.1f}  max={max(gpu_ms):.1f}")

    # ---- torch.profiler --------------------------------------------------
    print("\n=== torch.profiler ===")
    out_dir = Path(__file__).resolve().parent
    trace_path = out_dir / "pgd_d3_warmup_trace.json"
    text_path = out_dir / "pgd_d3_warmup_output.txt"

    sched = prof_schedule(wait=1, warmup=1, active=3, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        for _ in range(5):
            warmup_once()
            torch.cuda.synchronize()
            prof.step()

    prof.export_chrome_trace(str(trace_path))
    print(f"  trace: {trace_path}")

    sections: list[str] = []
    sections.append(f"device: {torch.cuda.get_device_name(0)}")
    sections.append(f"torch: {torch.__version__}")
    sections.append(f"target n_params: {n_params_target:,}")
    sections.append(f"all components n_params: {n_params_components:,}")
    sections.append(f"n_warmup_steps per outer step: {N_WARMUP_STEPS}")
    sections.append(f"wall ms (mean over 10): {wall_mean:.1f}")
    sections.append(f"gpu ms (mean over 10): {gpu_mean:.1f}")
    sections.append("\nproduction observed: pgd/D3_warmup ~400 ms/step")
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
