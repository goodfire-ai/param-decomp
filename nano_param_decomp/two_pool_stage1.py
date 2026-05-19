"""Stage 1 of the 2-pool MVP: single-process numerical sanity check.

Goal: verify that splitting the backward pass into

  (a) home losses backward with retain_graph=True  (faith + imp + stoch)
  (b) PPGD loss backward against a detached ci_scratch leaf
  (c) stitch: torch.autograd.backward(ci_lower, grad_tensors=ci_scratch.grad)
              to flow PPGD's CI-gradient back through the CI fn body

produces *identical* gradients to the baseline single-shot total.backward()
on every trained parameter (V, U, every CI-fn weight).

If this matches bit-for-bit, the autograd-splitting math is right and stage 2
(actual two processes with dist.send/recv) is just plumbing.

Run:
    .venv/bin/python -m nano_param_decomp.two_pool_stage1
"""

import copy
from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nano_param_decomp.run import (
    CITransformer,
    Config,
    PersistentPGD,
    anneal_p,
    clear_wrapper_masks,
    faithfulness_loss,
    importance_minimality_loss,
    install_components,
    stochastic_recon_loss,
)


class TinyLM(nn.Module):
    """Embedding -> n_layers of (linear -> gelu -> linear) residual blocks -> unembed.

    Decomposable nn.Linears live at layers.{i}.fc1 and layers.{i}.fc2.
    """

    def __init__(self, vocab: int, d: int, n_layers: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([nn.Module() for _ in range(n_layers)])
        for layer in self.layers:
            layer.fc1 = nn.Linear(d, d, bias=False)
            layer.fc2 = nn.Linear(d, d, bias=False)
        self.unembed = nn.Linear(d, vocab, bias=False)

    @override
    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = x + layer.fc2(F.gelu(layer.fc1(x)))
        return self.unembed(x)


def _forward_get_ci(
    target_model: nn.Module,
    ci_fn: CITransformer,
    wrappers: dict,
    input_ids: Tensor,
) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
    clear_wrapper_masks(wrappers)
    target_logits = target_model(input_ids)
    acts = {n: w.last_input for n, w in wrappers.items()}
    ci_lower, ci_upper, _ = ci_fn(acts)
    return target_logits, ci_lower, ci_upper


def baseline_step(
    target_model: nn.Module,
    ci_fn: CITransformer,
    wrappers: dict,
    ppgd: PersistentPGD,
    input_ids: Tensor,
    cfg: Config,
    imp_p: float,
) -> None:
    """The standard single-shot nano step: one total.backward() over everything."""
    target_logits, ci_lower, ci_upper = _forward_get_ci(target_model, ci_fn, wrappers, input_ids)

    ppgd.warmup(target_model, wrappers, input_ids, target_logits, ci_lower, lr=cfg.ppgd_lr)

    loss_faith = faithfulness_loss(wrappers)
    loss_imp = importance_minimality_loss(
        ci_upper, imp_p, cfg.imp_eps, cfg.imp_beta, world_size=1
    )
    loss_stoch = stochastic_recon_loss(
        target_model, wrappers, input_ids, target_logits, ci_lower
    )
    loss_ppgd = ppgd.recon_loss(target_model, wrappers, input_ids, target_logits, ci_lower)

    total = (
        cfg.coeff_faith * loss_faith
        + cfg.coeff_imp * loss_imp
        + cfg.coeff_stoch * loss_stoch
        + cfg.coeff_ppgd * loss_ppgd
    )
    total.backward()


def split_step(
    target_model: nn.Module,
    ci_fn: CITransformer,
    wrappers: dict,
    ppgd: PersistentPGD,
    input_ids: Tensor,
    cfg: Config,
    imp_p: float,
) -> None:
    """Two-pool-shaped split with *equivalent* gradients to baseline.

    Trick: do PPGD's forward+backward first (against a detached ci_scratch leaf) to extract
    ci_scratch.grad — this is dL_ppgd/dci_lower treating ci_lower as a leaf. THEN run home
    losses' backward as a single call with ci_lower added as an extra root seeded by
    ci_scratch.grad. This way ppgd's contribution and stoch's contribution merge at ci_lower
    *before* lower_leaky.backward fires, so any piecewise nonlinearity downstream of ci_lower
    sees a single combined signal — same as baseline.
    """
    target_logits, ci_lower, ci_upper = _forward_get_ci(target_model, ci_fn, wrappers, input_ids)

    # The "cross-pool cut": ci_scratch is a leaf copy that the PPGD half will use.
    ci_scratch = {n: v.detach().clone().requires_grad_(True) for n, v in ci_lower.items()}

    # --- Scratchpad pool work ---
    ppgd.warmup(target_model, wrappers, input_ids, target_logits, ci_scratch, lr=cfg.ppgd_lr)
    loss_ppgd = ppgd.recon_loss(target_model, wrappers, input_ids, target_logits, ci_scratch)
    total_ppgd = cfg.coeff_ppgd * loss_ppgd
    # Backward through scratchpad — stops at ci_scratch (a leaf). Populates V.grad, U.grad
    # additively, and ci_scratch.grad. Does NOT touch CI fn (ci_scratch detached from it).
    total_ppgd.backward()

    # --- Home pool work ---
    loss_faith = faithfulness_loss(wrappers)
    loss_imp = importance_minimality_loss(
        ci_upper, imp_p, cfg.imp_eps, cfg.imp_beta, world_size=1
    )
    loss_stoch = stochastic_recon_loss(
        target_model, wrappers, input_ids, target_logits, ci_lower
    )
    total_home = (
        cfg.coeff_faith * loss_faith
        + cfg.coeff_imp * loss_imp
        + cfg.coeff_stoch * loss_stoch
    )

    # Single backward call: total_home as scalar root + ci_lower tensors as extra roots
    # seeded by ci_scratch.grad. Autograd accumulates the seeds at ci_lower before walking
    # through lower_leaky's backward, so the piecewise non-linearity sees the combined
    # (stoch + ppgd) gradient — identical to what baseline's total.backward() sees.
    ci_names = list(ci_lower)
    torch.autograd.backward(
        tensors=[total_home, *(ci_lower[n] for n in ci_names)],
        grad_tensors=[None, *(ci_scratch[n].grad for n in ci_names)],
    )


def collect_grads(target_model: nn.Module, ci_fn: CITransformer, wrappers: dict) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, w in wrappers.items():
        assert w.V.grad is not None, f"{name}.V has no grad"
        assert w.U.grad is not None, f"{name}.U has no grad"
        out[f"wrap.{name}.V"] = w.V.grad.detach().clone()
        out[f"wrap.{name}.U"] = w.U.grad.detach().clone()
    for name, p in ci_fn.named_parameters():
        assert p.grad is not None, f"ci_fn.{name} has no grad"
        out[f"ci_fn.{name}"] = p.grad.detach().clone()
    return out


def zero_all_grads(target_model: nn.Module, ci_fn: CITransformer, wrappers: dict) -> None:
    for w in wrappers.values():
        if w.V.grad is not None:
            w.V.grad = None
        if w.U.grad is not None:
            w.U.grad = None
    for p in ci_fn.parameters():
        p.grad = None


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    # Tiny everything
    vocab = 32
    d = 16
    n_layers = 2
    batch_size = 2
    seq_len = 8
    C = 4

    cfg = Config(
        C_per_module={},  # filled below
        batch_size=batch_size,
        seq_len=seq_len,
        # Tiny CI transformer
        ci_d_model=32,
        ci_n_blocks=1,
        ci_n_heads=2,
        ci_mlp_hidden=64,
        ppgd_inner_steps=2,
    )

    # Build model once; we'll snapshot state and reset between baseline / split.
    torch.manual_seed(0)
    target = TinyLM(vocab, d, n_layers)

    C_per_module = {f"layers.{i}.fc{j}": C for i in range(n_layers) for j in (1, 2)} | {
        "unembed": C
    }
    cfg.C_per_module = C_per_module

    wrappers = install_components(target, C_per_module)
    d_in_per_module = {n: int(w.W_target.shape[1]) for n, w in wrappers.items()}
    ci_fn = CITransformer(d_in_per_module, C_per_module, cfg)
    target = target.to(device)
    ci_fn = ci_fn.to(device)
    print(f"target params={sum(p.numel() for p in target.parameters()):,}")
    print(f"ci_fn  params={sum(p.numel() for p in ci_fn.parameters()):,}")
    print(f"wrappers: {list(wrappers)}")

    # Fixed input batch — random tokens, no HF, no streaming.
    torch.manual_seed(1)
    input_ids = torch.randint(0, vocab, (batch_size, seq_len), device=device)

    # Snapshot state so we can reset cleanly between runs.
    initial_target_state = copy.deepcopy(target.state_dict())
    initial_ci_fn_state = copy.deepcopy(ci_fn.state_dict())

    imp_p = anneal_p(0, cfg.n_steps, cfg.p_start, cfg.p_end)

    def run_once(step_fn) -> dict[str, Tensor]:
        # Reset params, RNG, PPGD state to the same starting point.
        target.load_state_dict(initial_target_state)
        ci_fn.load_state_dict(initial_ci_fn_state)
        zero_all_grads(target, ci_fn, wrappers)
        torch.manual_seed(42)
        ppgd = PersistentPGD(wrappers, batch_size, seq_len, device, cfg)
        # Re-seed *after* PPGD init so step-internal randomness (stoch loss masks, routing)
        # has identical state between runs.
        torch.manual_seed(123)
        step_fn(target, ci_fn, wrappers, ppgd, input_ids, cfg, imp_p)
        return collect_grads(target, ci_fn, wrappers)

    print("\n--- running baseline ---")
    baseline_grads = run_once(baseline_step)
    print(f"collected {len(baseline_grads)} grad tensors")

    print("\n--- running split ---")
    split_grads = run_once(split_step)
    print(f"collected {len(split_grads)} grad tensors")

    print("\n--- comparing ---")
    assert baseline_grads.keys() == split_grads.keys()
    # Tolerance is on *relative* diff. The split path does 3 separate `.grad` accumulations
    # vs the baseline's single fused one, so absolute diffs reflect fp32 summation-order
    # noise (multiples of one ULP of the gradient's magnitude). 1e-5 relative is generous;
    # in practice fp32 ULP ~6e-8 so we'd expect ~1e-7.
    rel_tol = 1e-5
    worst_rel_key = ""
    max_rel = 0.0
    max_abs = 0.0
    for key in baseline_grads:
        b = baseline_grads[key]
        s = split_grads[key]
        abs_diff = (b - s).abs().max().item()
        denom = b.abs().max().item()
        rel = abs_diff / denom if denom > 0 else 0.0
        if rel > max_rel:
            max_rel = rel
            max_abs = abs_diff
            worst_rel_key = key

    print(f"worst (by relative): {worst_rel_key}")
    print(f"  max_rel = {max_rel:.4e}")
    print(f"  max_abs = {max_abs:.4e}  (baseline magnitude {baseline_grads[worst_rel_key].abs().max().item():.4e})")
    if max_rel < rel_tol:
        print(f"\nPASS — split step matches baseline within relative tol {rel_tol:.0e}.")
    else:
        print(f"\nFAIL — relative diff exceeds tol {rel_tol:.0e}.")
        diffs = []
        for key in baseline_grads:
            b = baseline_grads[key]
            s = split_grads[key]
            denom = b.abs().max().item()
            rel = ((b - s).abs().max().item()) / denom if denom > 0 else 0.0
            diffs.append((key, rel))
        diffs.sort(key=lambda kv: kv[1], reverse=True)
        for k, v in diffs[:10]:
            print(f"  {k}: rel={v:.4e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
