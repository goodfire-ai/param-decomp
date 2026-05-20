"""Stage 2 of the 2-pool MVP: two real processes on 2 GPUs.

Rank 0 = pool A (home): owns target, per-module CI fns, V/U params, optimizer.
Rank 1 = pool B (scratchpad): owns target, V/U replica, persistent PGD sources.

Backward strategy: serial-stitch. Pool A waits for pool B's grads, then does one
combined backward — bit-exact match to single-pool baseline. Forward of home losses
overlaps with pool B's full PPGD pipeline.

Run:
    .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \\
        -m nano_param_decomp.two_pool_stage2

The test is "loss curves resemble single-pool baseline over ~50 steps with no NaNs".
"""

# pyright: reportIndexIssue=false, reportArgumentType=false, reportOperatorIssue=false, reportUnnecessaryComparison=false

import math
import os
from typing import Literal, override

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
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
    lower_leaky,
    stochastic_recon_loss,
    upper_leaky,
)
from nano_param_decomp.two_pool_stage1 import TinyLM

# --- Hardcoded rank/role assignment ---

RANK_POOLS: dict[int, Literal["a", "b"]] = {
    0: "a",
    1: "b",
}
# Tensors flow rank 0 <-> rank 1 directly.
RANK_A = 0
RANK_B = 1


# --- Per-module CI fn (tiny) ---


class ModuleCIFn(nn.Module):
    """Per-module CI function: takes one module's pre-weight acts [B, S, d_in] and
    emits CI values [B, S, C].

    Tiny — just a 2-layer MLP for stage 2. The "real" version (in the planning docs)
    is a small transformer with optional cross-site inputs; we mock the per-module
    structure here without yet wiring the cross-site part.

    Returns (ci_lower, ci_upper) — both shape [B, S, C+1] (last is the delta-mask
    slot). The +1 mirrors what PersistentPGD expects (it stores B,S,C+1 sources).
    """

    def __init__(self, d_in: int, C: int, hidden: int, leaky_alpha: float) -> None:
        super().__init__()
        self.proj_in = nn.Linear(d_in, hidden)
        self.proj_out = nn.Linear(hidden, C)
        self.alpha = leaky_alpha

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = F.gelu(self.proj_in(F.rms_norm(x, (x.shape[-1],))))
        logits = self.proj_out(h)
        return lower_leaky(logits, self.alpha), upper_leaky(logits, self.alpha)


def build_ci_fns(
    d_in_per_module: dict[str, int], c_per_module: dict[str, int], hidden: int, leaky_alpha: float
) -> dict[str, ModuleCIFn]:
    return {
        n: ModuleCIFn(d_in_per_module[n], c_per_module[n], hidden, leaky_alpha)
        for n in d_in_per_module
    }


def ci_forward(
    ci_fns: dict[str, ModuleCIFn], acts: dict[str, Tensor]
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    ci_lower: dict[str, Tensor] = {}
    ci_upper: dict[str, Tensor] = {}
    for name, fn in ci_fns.items():
        l, u = fn(acts[name])
        ci_lower[name] = l
        ci_upper[name] = u
    return ci_lower, ci_upper


# --- Cross-pool comm ---


def _send_dict(tensors: dict[str, Tensor], dst: int, names: list[str]) -> None:
    """Send a dict of tensors in `names` order."""
    for n in names:
        dist.send(tensors[n].contiguous(), dst=dst)


def _recv_dict(
    template: dict[str, Tensor], src: int, names: list[str], device: torch.device
) -> dict[str, Tensor]:
    """Recv tensors with shapes matching `template[name]`, in `names` order. Returns new dict."""
    out: dict[str, Tensor] = {}
    for n in names:
        buf = torch.empty_like(template[n], device=device)
        dist.recv(buf, src=src)
        out[n] = buf
    return out


# --- Pool A (home) step ---


def pool_a_step(
    target_model: nn.Module,
    wrappers: dict[str, ComponentLinear],
    ci_fns: dict[str, ModuleCIFn],
    optimizer: torch.optim.Optimizer,
    input_ids: Tensor,
    cfg: Config,
    imp_p: float,
    site_names: list[str],
    device: torch.device,
) -> dict[str, float]:
    # Forward
    clear_wrapper_masks(wrappers)
    target_logits = target_model(input_ids)
    acts = {n: wrappers[n].last_input for n in site_names}
    ci_lower, ci_upper = ci_forward(ci_fns, acts)

    # Send ci_lower (detached values — pool B doesn't need our graph) to pool B.
    ci_lower_detached = {n: v.detach() for n, v in ci_lower.items()}
    _send_dict(ci_lower_detached, dst=RANK_B, names=site_names)

    # Home loss forwards (no backward yet — happens after receiving B's grads).
    loss_faith = faithfulness_loss(wrappers)
    loss_imp = importance_minimality_loss(ci_upper, imp_p, cfg.imp_eps, cfg.imp_beta, world_size=1)
    loss_stoch = stochastic_recon_loss(target_model, wrappers, input_ids, target_logits, ci_lower)
    total_home = (
        cfg.coeff_faith * loss_faith + cfg.coeff_imp * loss_imp + cfg.coeff_stoch * loss_stoch
    )

    # Receive grads from pool B.
    v_grads_template = {n: wrappers[n].V for n in site_names}
    u_grads_template = {n: wrappers[n].U for n in site_names}
    ci_grads_template = ci_lower_detached
    v_grads_recv = _recv_dict(v_grads_template, src=RANK_B, names=site_names, device=device)
    u_grads_recv = _recv_dict(u_grads_template, src=RANK_B, names=site_names, device=device)
    ci_grads_recv = _recv_dict(ci_grads_template, src=RANK_B, names=site_names, device=device)

    # Seed V/U .grad with B's contribution; home backward accumulates onto these.
    optimizer.zero_grad(set_to_none=True)
    for n in site_names:
        wrappers[n].V.grad = v_grads_recv[n]
        wrappers[n].U.grad = u_grads_recv[n]

    # Single home backward: total_home as scalar root + per-site ci_lower as extra roots
    # seeded by ci_grads_recv. Merges at ci_lower before lower_leaky.backward.
    torch.autograd.backward(
        tensors=[total_home, *(ci_lower[n] for n in site_names)],
        grad_tensors=[None, *(ci_grads_recv[n] for n in site_names)],
    )

    # We also need to compute and log ppgd's contribution to the total — pool B has it
    # but we don't get a scalar back. For logging only, skip; could send it as one
    # extra float if we want to track. Punt for stage 2.

    optimizer.step()

    # Send updated V/U back to pool B.
    v_updated = {n: wrappers[n].V.detach() for n in site_names}
    u_updated = {n: wrappers[n].U.detach() for n in site_names}
    _send_dict(v_updated, dst=RANK_B, names=site_names)
    _send_dict(u_updated, dst=RANK_B, names=site_names)

    return {
        "loss/faith": loss_faith.item(),
        "loss/imp": loss_imp.item(),
        "loss/stoch": loss_stoch.item(),
    }


# --- Pool B (scratchpad) step ---


def pool_b_step(
    target_model: nn.Module,
    wrappers: dict[str, ComponentLinear],
    ppgd: PersistentPGD,
    input_ids: Tensor,
    cfg: Config,
    site_names: list[str],
    device: torch.device,
) -> dict[str, float]:
    # Forward target (frozen, replicated — both pools get identical target_logits).
    clear_wrapper_masks(wrappers)
    with torch.no_grad():
        target_logits = target_model(input_ids)

    # Recv ci_lower values from A.
    ci_template: dict[str, Tensor] = {}
    B, S = input_ids.shape
    for n in site_names:
        ci_template[n] = torch.empty(B, S, wrappers[n].C, device=device)
    ci_recv = _recv_dict(ci_template, src=RANK_A, names=site_names, device=device)

    # Re-leaf as ci_scratch (requires_grad for the seed grad we'll send back).
    ci_scratch = {n: v.detach().clone().requires_grad_(True) for n, v in ci_recv.items()}

    # PPGD warmup + recon loss against ci_scratch.
    ppgd.warmup(target_model, wrappers, input_ids, target_logits, ci_scratch, lr=cfg.ppgd_lr)
    loss_ppgd = ppgd.recon_loss(target_model, wrappers, input_ids, target_logits, ci_scratch)
    total_ppgd = cfg.coeff_ppgd * loss_ppgd

    # Extract V/U grads and ci_scratch grads via autograd.grad (no .grad pollution).
    params: list[Tensor] = []
    for n in site_names:
        params.extend([wrappers[n].V, wrappers[n].U])
    ci_scratch_list = [ci_scratch[n] for n in site_names]
    grads = torch.autograd.grad(total_ppgd, params + ci_scratch_list)

    v_grads = {n: grads[2 * i] for i, n in enumerate(site_names)}
    u_grads = {n: grads[2 * i + 1] for i, n in enumerate(site_names)}
    ci_grads = {n: grads[2 * len(site_names) + i] for i, n in enumerate(site_names)}

    # Send to A.
    _send_dict(v_grads, dst=RANK_A, names=site_names)
    _send_dict(u_grads, dst=RANK_A, names=site_names)
    _send_dict(ci_grads, dst=RANK_A, names=site_names)

    # Recv updated V/U from A and write into wrappers.
    v_template = {n: wrappers[n].V for n in site_names}
    u_template = {n: wrappers[n].U for n in site_names}
    v_new = _recv_dict(v_template, src=RANK_A, names=site_names, device=device)
    u_new = _recv_dict(u_template, src=RANK_A, names=site_names, device=device)
    with torch.no_grad():
        for n in site_names:
            wrappers[n].V.copy_(v_new[n])
            wrappers[n].U.copy_(u_new[n])

    return {"loss/ppgd": loss_ppgd.item()}


# --- Main ---


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, "stage 2 needs exactly 2 ranks (one per pool)"
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    pool = RANK_POOLS[rank]
    print(f"[rank{rank}] pool={pool} device={device}", flush=True)

    # Tiny model + decomp config — both ranks build identically.
    vocab, d, n_layers, batch_size, seq_len, C = 32, 16, 2, 2, 8, 4

    cfg = Config(
        C_per_module={},
        batch_size=batch_size,
        seq_len=seq_len,
        n_steps=50,
        # PPGD lr (fixed for stage 2; no schedule)
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

    # Per-module CI fns — only built on pool A. Pool B doesn't need them.
    ci_fns: dict[str, ModuleCIFn] = {}
    if pool == "a":
        d_in_per_module = {n: int(wrappers[n].W_target.shape[1]) for n in site_names}
        ci_fns_dict = build_ci_fns(
            d_in_per_module, C_per_module, hidden=32, leaky_alpha=cfg.leaky_alpha
        )
        for n in site_names:
            ci_fns_dict[n].to(device)
        ci_fns = ci_fns_dict
        print(
            f"[rank{rank}] built per-module CI fns: total params = "
            f"{sum(p.numel() for f in ci_fns.values() for p in f.parameters()):,}",
            flush=True,
        )

    # Optimizer only on pool A.
    optimizer: torch.optim.Optimizer | None = None
    if pool == "a":
        params: list[nn.Parameter] = []
        for w in wrappers.values():
            params.extend([w.V, w.U])
        for f in ci_fns.values():
            params.extend(list(f.parameters()))
        optimizer = torch.optim.AdamW(params, lr=cfg.main_lr, weight_decay=0.0)

    # PPGD only on pool B.
    ppgd: PersistentPGD | None = None
    if pool == "b":
        torch.manual_seed(42)
        ppgd = PersistentPGD(wrappers, batch_size, seq_len, device, cfg)

    # Deterministic input batch generator (both ranks identical).
    data_rng = torch.Generator(device=device).manual_seed(0)

    def make_batch(step: int) -> Tensor:
        data_rng.manual_seed(step * 7919 + 17)
        return torch.randint(0, vocab, (batch_size, seq_len), device=device, generator=data_rng)

    # Sampling RNG must also be seeded identically each step for the stoch loss mask
    # sampling (sample_continuous_masks / sample_uniform_k_subset_routing read the
    # default torch RNG). Both ranks do this in lockstep.
    sampling_seed_base = 100

    print(f"[rank{rank}] starting training loop ({cfg.n_steps} steps)", flush=True)
    imp_p = anneal_p(0, cfg.n_steps, cfg.p_start, cfg.p_end)

    for step in range(cfg.n_steps):
        input_ids = make_batch(step)
        torch.manual_seed(sampling_seed_base + step)

        if pool == "a":
            metrics = pool_a_step(
                target, wrappers, ci_fns, optimizer, input_ids, cfg, imp_p, site_names, device
            )
        else:
            metrics = pool_b_step(target, wrappers, ppgd, input_ids, cfg, site_names, device)

        if step % 5 == 0:
            msg = " ".join(f"{k}={v:.4g}" for k, v in metrics.items())
            print(f"[rank{rank}] step={step} {msg}", flush=True)

        # Check for NaNs.
        for k, v in metrics.items():
            if not math.isfinite(v):
                print(f"[rank{rank}] NaN/Inf at step {step}: {k}={v}", flush=True)
                dist.destroy_process_group()
                raise SystemExit(1)

    print(f"[rank{rank}] done", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
