"""Stage 2.5: single-process baseline equivalent to stage 2.

Same TinyLM, same per-module CI fns, same seeds, same data, same loss coefficients.
The only difference vs stage 2 is that everything runs in one process — no dist.send/recv,
no cross-pool autograd split. Just a regular total.backward() per step.

The goal is to compare per-step loss numbers against stage 2's output. They should match
within fp32 noise. If they diverge, the stage 2 protocol implementation has a bug
(stale weights, wrong send/recv order, etc).

Run:
    .venv/bin/python -m nano_param_decomp.two_pool_stage2_5_baseline
"""

# pyright: reportIndexIssue=false, reportArgumentType=false

import json
import math

import torch
from torch import Tensor

from nano_param_decomp.run import (
    ComponentLinear,
    Config,
    PersistentPGD,
    anneal_p,
    clear_wrapper_masks,
    faithfulness_loss,
    importance_minimality_loss,
    install_components,
    stochastic_recon_loss,
)
from nano_param_decomp.two_pool_stage1 import TinyLM
from nano_param_decomp.two_pool_stage2 import ModuleCIFn, build_ci_fns, ci_forward


def baseline_step_per_module(
    target_model: torch.nn.Module,
    ci_fns: dict[str, ModuleCIFn],
    wrappers: dict[str, ComponentLinear],
    ppgd: PersistentPGD,
    optimizer: torch.optim.Optimizer,
    input_ids: Tensor,
    cfg: Config,
    imp_p: float,
) -> dict[str, float]:
    clear_wrapper_masks(wrappers)
    target_logits = target_model(input_ids)
    acts = {n: wrappers[n].last_input for n in ci_fns}
    ci_lower, ci_upper = ci_forward(ci_fns, acts)

    # Match stage 2 ordering: home losses (which consume sampling RNG) come BEFORE
    # PPGD warmup. In stage 2 these run on different ranks but the same RNG seed; in
    # single-process we need to mirror the order to keep RNG consumption identical.
    loss_faith = faithfulness_loss(wrappers)
    loss_imp = importance_minimality_loss(ci_upper, imp_p, cfg.imp_eps, cfg.imp_beta, world_size=1)
    loss_stoch = stochastic_recon_loss(target_model, wrappers, input_ids, target_logits, ci_lower)

    ppgd.warmup(target_model, wrappers, input_ids, target_logits, ci_lower, lr=cfg.ppgd_lr)
    loss_ppgd = ppgd.recon_loss(target_model, wrappers, input_ids, target_logits, ci_lower)

    total = (
        cfg.coeff_faith * loss_faith
        + cfg.coeff_imp * loss_imp
        + cfg.coeff_stoch * loss_stoch
        + cfg.coeff_ppgd * loss_ppgd
    )

    optimizer.zero_grad(set_to_none=True)
    total.backward()
    optimizer.step()

    return {
        "loss/faith": loss_faith.item(),
        "loss/imp": loss_imp.item(),
        "loss/stoch": loss_stoch.item(),
        "loss/ppgd": loss_ppgd.item(),
    }


def main() -> None:
    device = torch.device("cuda:0")

    # Mirror stage 2 setup exactly.
    vocab, d, n_layers, batch_size, seq_len, C = 32, 16, 2, 2, 8, 4
    cfg = Config(
        C_per_module={},
        batch_size=batch_size,
        seq_len=seq_len,
        n_steps=50,
        ppgd_inner_steps=2,
    )

    torch.manual_seed(0)
    target = TinyLM(vocab, d, n_layers)
    C_per_module = {f"layers.{i}.fc{j}": C for i in range(n_layers) for j in (1, 2)} | {
        "unembed": C
    }
    cfg.C_per_module = C_per_module
    site_names = sorted(C_per_module)

    wrappers = install_components(target, C_per_module)
    target = target.to(device)
    for w in wrappers.values():
        w.to(device)

    d_in_per_module = {n: int(wrappers[n].W_target.shape[1]) for n in site_names}
    ci_fns = build_ci_fns(d_in_per_module, C_per_module, hidden=32, leaky_alpha=cfg.leaky_alpha)
    for n in site_names:
        ci_fns[n].to(device)

    # PPGD init — must use the same seed sequence as stage 2 rank 1. Stage 2 rank 0
    # does the CI fn init between TinyLM init and PPGD init; rank 1 skips CI fn init.
    # For single-process to match rank 0's RNG path for the *home* losses, we build CI
    # fns. To match rank 1's RNG path for PPGD init, we seed(42) before PPGD init —
    # same as rank 1.
    torch.manual_seed(42)
    ppgd = PersistentPGD(wrappers, batch_size, seq_len, device, cfg)

    params: list[torch.nn.Parameter] = []
    for w in wrappers.values():
        params.extend([w.V, w.U])
    for f in ci_fns.values():
        params.extend(list(f.parameters()))
    optimizer = torch.optim.AdamW(params, lr=cfg.main_lr, weight_decay=0.0)

    data_rng = torch.Generator(device=device).manual_seed(0)

    def make_batch(step: int) -> Tensor:
        data_rng.manual_seed(step * 7919 + 17)
        return torch.randint(0, vocab, (batch_size, seq_len), device=device, generator=data_rng)

    sampling_seed_base = 100
    imp_p = anneal_p(0, cfg.n_steps, cfg.p_start, cfg.p_end)

    losses_log: list[dict[str, float]] = []
    print(f"[baseline] starting ({cfg.n_steps} steps)", flush=True)
    for step in range(cfg.n_steps):
        input_ids = make_batch(step)
        torch.manual_seed(sampling_seed_base + step)
        metrics = baseline_step_per_module(
            target, ci_fns, wrappers, ppgd, optimizer, input_ids, cfg, imp_p
        )
        losses_log.append({"step": step, **metrics})
        if step % 5 == 0:
            msg = " ".join(f"{k}={v:.4g}" for k, v in metrics.items())
            print(f"[baseline] step={step} {msg}", flush=True)
        for v in metrics.values():
            if not math.isfinite(v):
                print(f"[baseline] NaN at step {step}", flush=True)
                raise SystemExit(1)

    out_path = "/tmp/two_pool_stage2_5_baseline_losses.json"
    with open(out_path, "w") as f:
        json.dump(losses_log, f)
    print(f"[baseline] done. losses → {out_path}", flush=True)


if __name__ == "__main__":
    main()
