"""Full-PPGD compile probe: the REAL PersistentPGDState path (warmup inner loop + sources),
not just an isolated autograd.grad. Mirrors step_ppgd's D3/D4/D5 on a small vendored Llama.

Tests the two things the isolated probe skipped:
  - the n_warmup PGD inner loop (multiple compiled forwards w/ sources updated each iter) ->
    recompile / graph-break risk
  - the source leaves in the fused autograd.grad

Run with TORCH_LOGS=recompiles to surface any recompilation.
"""

import os
from contextlib import nullcontext
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

if os.environ.get("PD_PROBE_DETERMINISTIC", "0") == "1":
    torch.use_deterministic_algorithms(True, warn_only=True)
    from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: F401

from param_decomp.ci_fns import LayerwiseCiConfig
from param_decomp.decomposition_targets import DecompositionTarget
from param_decomp.masks import AllLayersRouter
from param_decomp.metrics.persistent_pgd_state import (
    AdamPGDConfig,
    PerBatchPerPositionScope,
    PersistentPGDState,
)
from param_decomp.schedule import ScheduleConfig
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl
from param_decomp_lab.experiments.lm.vendored.component_model import LMComponentModel
from param_decomp_lab.experiments.lm.vendored.llama_3_1.config import VendoredLlamaConfig
from param_decomp_lab.experiments.lm.vendored.llama_3_1.model import VendoredLlama

DEV = torch.device("cuda")
B, T, C = 2, 128, 32
SITES = ["layers.1.mlp.gate_proj", "layers.1.mlp.up_proj", "layers.1.mlp.down_proj"]
N_WARMUP = 2


def build_model() -> LMComponentModel:
    cfg = VendoredLlamaConfig(
        model_type="VendoredLlama", max_position_embeddings=T, vocab_size=64,
        n_layer=4, n_head=4, n_key_value_heads=2, n_embd=128, n_intermediate=256,
    )
    torch.manual_seed(0)
    base = VendoredLlama(cfg)
    for p in base.parameters():
        p.requires_grad_(False)
    targets = [DecompositionTarget(module_path=s, C=C) for s in SITES]
    ci_config = LayerwiseCiConfig(fn_type="mlp", hidden_dims=[32])
    torch.manual_seed(1)
    lm = LMComponentModel.build(base, targets, ci_config, sigmoid_type="leaky_hard").to(DEV)
    lm.model.enable_activation_checkpointing()
    return lm


def build_ppgd(lm: LMComponentModel) -> PersistentPGDState:
    return PersistentPGDState(
        module_to_c=lm.module_to_c,
        batch_dims=(B, T),
        device=DEV,
        use_delta_component=True,
        optimizer_cfg=AdamPGDConfig(
            beta1=0.5, beta2=0.99, eps=1e-8,
            lr_schedule=ScheduleConfig(start_val=0.01, warmup_pct=0.025, final_val_frac=1.0, fn_type="constant"),
        ),
        scope=PerBatchPerPositionScope(),
        use_sigmoid_parameterization=False,
        n_warmup_steps=N_WARMUP,
        n_samples=1,
        router=AllLayersRouter(),
        reconstruction_loss=recon_loss_kl,
    )


def make_inputs(lm: LMComponentModel):
    torch.manual_seed(2)
    idx = torch.randint(0, 64, (B, T), device=DEV)
    with torch.no_grad():
        target_out = lm(idx).detach()
    ci_scratch = {
        s: torch.rand(B, T, lm.module_to_c[s], device=DEV, dtype=torch.float32, requires_grad=True)
        for s in SITES
    }
    return idx, target_out, ci_scratch


def full_ppgd_step(lm, ppgd, idx, target_out, ci_scratch, *, autocast: bool) -> dict[str, torch.Tensor]:
    """step_ppgd D3 (warmup) + D4 (recon sum) + D5 (one autograd.grad over V/U/ci/sources)."""
    weight_deltas = lm.calc_weight_deltas()
    det = os.environ.get("PD_PROBE_DETERMINISTIC", "0") == "1"
    sdpa_ctx = sdpa_kernel(SDPBackend.MATH) if det else nullcontext()
    with sdpa_ctx, torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        ppgd.warmup(model=lm, batch=idx, target_out=target_out, ci=ci_scratch, weight_deltas=weight_deltas)
        sum_loss, _n = ppgd.compute_recon_sum_and_n(
            model=lm, batch=idx, target_out=target_out, ci=ci_scratch, weight_deltas=weight_deltas
        )
    v = [lm.components[s].V for s in SITES]
    u = [lm.components[s].U for s in SITES]
    leaves = [ci_scratch[s] for s in SITES]
    src = [ppgd.sources[s] for s in SITES]
    flat = torch.autograd.grad(sum_loss, v + u + leaves + src, retain_graph=False)
    n = len(SITES)
    out = {"loss": sum_loss.detach()}
    for i, s in enumerate(SITES):
        out[f"V/{s}"] = flat[i]
        out[f"U/{s}"] = flat[n + i]
        out[f"ci/{s}"] = flat[2 * n + i]
        out[f"src/{s}"] = flat[3 * n + i]
    return out


def rel_err(a, b) -> float:
    return ((a - b).norm() / (b.norm() + 1e-12)).item()


def time_it(fn, iters=8) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000


def compare(autocast: bool) -> tuple[float, float, float]:
    lm = build_model()
    idx, target_out, ci_scratch = make_inputs(lm)
    ppgd = build_ppgd(lm)
    init_state = ppgd.state_dict()  # warmup mutates sources+optim; restore for a fair eager/compiled match

    torch.manual_seed(1234)
    ppgd.load_state_dict(init_state)
    eager = full_ppgd_step(lm, ppgd, idx, target_out, ci_scratch, autocast=autocast)

    # CONTROL: a second eager run, identical seed + state. If this != eager, the harness is
    # nondeterministic (RNG in the forward) and any eager-vs-compiled gap is an artifact.
    torch.manual_seed(1234)
    ppgd.load_state_dict(init_state)
    eager2 = full_ppgd_step(lm, ppgd, idx, target_out, ci_scratch, autocast=autocast)
    ctrl = max(rel_err(eager2[k], eager[k]) for k in eager if k != "loss")
    print(f"    [control] eager-vs-eager worst rel_err = {ctrl:.2e}  ({'DETERMINISTIC' if ctrl < 1e-5 else 'NONDETERMINISTIC -> gap is an artifact'})")

    lm.model.compile()
    torch.manual_seed(1234)
    ppgd.load_state_dict(init_state)
    compiled = full_ppgd_step(lm, ppgd, idx, target_out, ci_scratch, autocast=autocast)

    worst = 0.0
    for k in eager:
        if k == "loss":
            continue
        e = rel_err(compiled[k], eager[k])
        worst = max(worst, e)
        print(f"    {k:30s} rel_err = {e:.2e}")
    print(f"    {'loss':30s} rel_err = {abs(compiled['loss'].item()-eager['loss'].item())/(abs(eager['loss'].item())+1e-12):.2e}")

    t_compiled = time_it(lambda: full_ppgd_step(lm, ppgd, idx, target_out, ci_scratch, autocast=autocast))
    lm2 = build_model()
    idx2, tgt2, ci2 = make_inputs(lm2)
    ppgd2 = build_ppgd(lm2)
    t_eager = time_it(lambda: full_ppgd_step(lm2, ppgd2, idx2, tgt2, ci2, autocast=autocast))
    return worst, t_eager, t_compiled


def main() -> None:
    print(f"torch {torch.__version__}, {torch.cuda.get_device_name()}  (n_warmup={N_WARMUP})")
    print("\n=== fp32 (no autocast) — CORRECTNESS ===")
    w32, te32, tc32 = compare(autocast=False)
    print(f"  WORST rel_err (fp32) = {w32:.2e}   eager {te32:.2f}ms -> compiled {tc32:.2f}ms ({te32/tc32:.2f}x)")
    print("\n=== bf16 autocast — PRODUCTION numerics + speed ===")
    wbf, tebf, tcbf = compare(autocast=True)
    print(f"  WORST rel_err (bf16) = {wbf:.2e}   eager {tebf:.2f}ms -> compiled {tcbf:.2f}ms ({tebf/tcbf:.2f}x)")
    ok = w32 < 1e-3
    print(f"\nVERDICT: full-PPGD compile CORRECT = {'PASS' if ok else 'FAIL'} (fp32 {w32:.2e}); speedup ~{tebf/tcbf:.1f}x")


if __name__ == "__main__":
    main()
