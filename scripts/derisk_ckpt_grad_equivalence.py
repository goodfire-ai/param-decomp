"""Derisk 0a: ckpt-recompute grad-equivalence for mask delivery.

Validates the core mechanism of the GPT2-vendoring plan: with each component's mask
threaded as a forward ARG into an activation-checkpointed block, the grads w.r.t. the
component params (V, U), the CI values, and the PPGD source are identical to the
non-checkpointed run. The reference is the non-checkpointed, arg-threaded run.

Three delivery mechanisms are compared under checkpointing:
  - arg        : mask passed as a checkpoint() argument            -> expected MATCH
  - attr_keep  : mask set on the module, kept alive through backward -> expected MATCH
  - attr_clear : mask set on the module, CLEARED after the forward
                 (simulating a context-manager exit before backward) -> expected BREAK

The attr_clear break is the empirical justification for arg-threading: ckpt recomputes
the forward DURING backward, so a mask read from a side attribute that has been cleared
is gone on recompute -> source falls out of the graph (source.grad is None) and V/U grads
are wrong. arg-threading needs no such lifecycle discipline because checkpoint() saves
and replays its arguments.

Run: srun --gres=gpu:1 python scripts/derisk_ckpt_grad_equivalence.py
"""

import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from param_decomp.components import LinearComponents  # noqa: E402

D, C, B, T, N_BLOCKS = 256, 64, 4, 32, 4
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class CompBlock(nn.Module):
    """A pre-LN block whose q-projection is a LinearComponents taking a per-position mask.

    `forward` accepts the mask as an arg (arg delivery) or reads `self._mask` (attribute
    delivery) when no arg is given. Has internal activations (relu, residual) so that
    activation checkpointing actually recomputes something.
    """

    def __init__(self) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(D)
        self.q = LinearComponents(C=C, d_in=D, d_out=D, bias=None)
        self.mlp = nn.Linear(D, D)
        self._mask: Tensor | None = None

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        m = mask if mask is not None else self._mask
        q = self.q(self.ln(x), mask=m)
        return x + self.mlp(torch.relu(q))


def build_model() -> nn.ModuleList:
    torch.manual_seed(0)
    return nn.ModuleList([CompBlock() for _ in range(N_BLOCKS)]).to(DEV)


def run_variant(
    model: nn.ModuleList,
    x0: Tensor,
    cis0: list[Tensor],
    srcs0: list[Tensor],
    *,
    delivery: str,
    use_ckpt: bool,
    bf16: bool,
) -> tuple[dict[str, Tensor | None], float]:
    x = x0.clone().requires_grad_(True)
    cis = [c.clone().requires_grad_(True) for c in cis0]
    srcs = [s.clone().requires_grad_(True) for s in srcs0]
    for p in model.parameters():
        p.grad = None

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (bf16 and DEV == "cuda")
        else nullcontext()
    )
    h = x
    with autocast:
        for i, block in enumerate(model):
            mask = cis[i] + (1 - cis[i]) * srcs[i]
            match delivery:
                case "arg":
                    h = checkpoint(block, h, mask, use_reentrant=False) if use_ckpt else block(h, mask)
                case "arg_dict":
                    # mask nested in a dict arg (the Phase-2 threading design: mask_infos keyed by path)
                    md = {"m": mask}

                    def _run(xx: Tensor, dd: dict[str, Tensor], _b: nn.Module = block) -> Tensor:
                        return _b(xx, dd["m"])

                    h = checkpoint(_run, h, md, use_reentrant=False) if use_ckpt else _run(h, md)
                case "attr_keep" | "attr_clear":
                    block._mask = mask
                    h = checkpoint(block, h, use_reentrant=False) if use_ckpt else block(h)
                    if delivery == "attr_clear":
                        block._mask = None
                case _:
                    raise ValueError(delivery)
        loss = h.float().pow(2).mean()
    loss.backward()

    grads: dict[str, Tensor | None] = {}
    for i, block in enumerate(model):
        grads[f"b{i}.V"] = None if block.q.V.grad is None else block.q.V.grad.clone()
        grads[f"b{i}.U"] = None if block.q.U.grad is None else block.q.U.grad.clone()
        grads[f"b{i}.mlp"] = None if block.mlp.weight.grad is None else block.mlp.weight.grad.clone()
        grads[f"b{i}.ci"] = None if cis[i].grad is None else cis[i].grad.clone()
        grads[f"b{i}.src"] = None if srcs[i].grad is None else srcs[i].grad.clone()
    grads["x"] = None if x.grad is None else x.grad.clone()
    return grads, loss.item()


def max_grad_diff(ref: dict[str, Tensor | None], got: dict[str, Tensor | None]) -> tuple[float, str]:
    worst, where = 0.0, "none-mismatch"
    for k, rg in ref.items():
        gg = got[k]
        if rg is None or gg is None:
            if (rg is None) != (gg is None):
                return float("inf"), f"{k}: ref={'None' if rg is None else 'T'} got={'None' if gg is None else 'T'}"
            continue
        d = (rg - gg).abs().max().item()
        if d > worst:
            worst, where = d, k
    return worst, where


def main() -> None:
    print(f"device={DEV}  d={D} C={C} B={B} T={T} blocks={N_BLOCKS}\n")
    model = build_model()
    torch.manual_seed(1)
    x0 = torch.randn(B, T, D, device=DEV)
    cis0 = [torch.rand(B, T, C, device=DEV) for _ in range(N_BLOCKS)]  # CI in [0,1)
    srcs0 = [torch.rand(B, T, C, device=DEV) for _ in range(N_BLOCKS)]  # continuous source

    for bf16 in (False, True):
        if bf16 and DEV != "cuda":
            continue
        tag = "bf16-autocast" if bf16 else "fp32"
        ref, ref_loss = run_variant(
            model, x0, cis0, srcs0, delivery="arg", use_ckpt=False, bf16=bf16
        )
        print(f"=== {tag} === (reference: no-ckpt arg-threaded, loss={ref_loss:.6f})")
        print(f"{'variant':>26} | {'loss':>10} | {'max grad diff':>14} | worst")
        print("-" * 78)
        variants = [
            ("arg", True, "EXPECT MATCH"),
            ("arg_dict", True, "EXPECT MATCH"),
            ("attr_keep", True, "EXPECT MATCH"),
            ("attr_clear", True, "EXPECT BREAK"),
        ]
        for delivery, use_ckpt, expect in variants:
            try:
                got, loss = run_variant(
                    model, x0, cis0, srcs0, delivery=delivery, use_ckpt=use_ckpt, bf16=bf16
                )
            except Exception as e:  # noqa: BLE001 — negative controls are expected to raise
                msg = type(e).__name__ + ": " + str(e).splitlines()[0]
                print(f"{delivery + ' ckpt':>26} | {'RAISED':>10} | {'broke':>14} | {msg[:30]}  [{expect}]")
                continue
            diff, where = max_grad_diff(ref, got)
            ds = "inf/None" if diff == float("inf") else f"{diff:.2e}"
            print(f"{delivery + ' ckpt':>26} | {loss:>10.6f} | {ds:>14} | {where}  [{expect}]")
        print()


if __name__ == "__main__":
    main()
