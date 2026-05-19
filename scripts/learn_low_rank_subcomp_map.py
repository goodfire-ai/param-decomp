"""Learn a low-rank c_fc-side mapping for a single mlp-output subcomponent.

For a chosen output subcomponent `c` in `h.<L>.mlp.down_proj` of a `LlamaSimpleMLP`
PD run (e.g. the `pile_llama_simple_mlp-4L` decomposition), learn a rank-k
replacement of the c_fc-side U matrix that reproduces c's activation:

    a^{out}_{b,t,c}_pred = sum_i V_down_proj[i, c] * NewGELU( (Q @ R @ a^{in}_{b,t})_i )

where
    a^{in}_{b,t,c'} = sum_j V_c_fc[j, c'] * h_{b,t,j}   (h is the c_fc input)

Targets are the actual a^{out}_{b,t,c} computed by running the target model on
input data — equivalent (for this run) to a ComponentModel forward with the
c_fc subcomponent masks and the c_fc delta-component mask all set to 1.

Only adjacent (c_fc, down_proj) MLP pairs in a `LlamaSimpleMLP` target model
are supported.
"""

import argparse
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch import Tensor

from param_decomp.configs import ScheduleConfig
from param_decomp.data import loop_dataloader, train_loader_and_tokenizer
from param_decomp.models.component_model import ComponentModel, ParamDecompRunInfo
from param_decomp.utils.general_utils import bf16_autocast, get_scheduled_value


def newgelu(x: Tensor) -> Tensor:
    """NewGELU, matching `LlamaSimpleMLP`'s activation."""
    c = float(np.sqrt(2.0 / np.pi))
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x.pow(3))))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--wandb-path",
        required=True,
        help="PD run path, e.g. 'goodfire/spd/runs/s-55ea3f9b' or 'wandb:...'.",
    )
    p.add_argument(
        "--output-module",
        required=True,
        help="Target down_proj module, e.g. 'h.3.mlp.down_proj'.",
    )
    p.add_argument(
        "--subcomp-id",
        type=int,
        required=True,
        help="Output subcomponent index c within the down_proj module.",
    )
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate (start_val).")
    p.add_argument(
        "--lr-fn-type",
        choices=["constant", "cosine", "linear"],
        default="cosine",
    )
    p.add_argument("--warmup-pct", type=float, default=0.05)
    p.add_argument(
        "--final-val-frac",
        type=float,
        default=0.1,
        help="Final LR as fraction of peak. Forced to 1.0 when --lr-fn-type=constant.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--out-dir",
        default=str(Path(__file__).parent / "low_rank_subcomp_map_out"),
    )
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--ema-window", type=int, default=100)
    p.add_argument("--wandb-project", default="spd-low-rank-subcomp-map")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def init_low_rank(
    d_intermediate: int, c_in: int, rank: int, device: str
) -> tuple[nn.Parameter, nn.Parameter]:
    """Init Q (d_intermediate, rank), R (rank, C_in) so Q@R has fan_in≈1/C_in entry-scale."""
    target_std = (1.0 / c_in) ** 0.5
    per_factor_std = (target_std / (rank**0.5)) ** 0.5
    Q = nn.Parameter(torch.randn(d_intermediate, rank, device=device) * per_factor_std)
    R = nn.Parameter(torch.randn(rank, c_in, device=device) * per_factor_std)
    return Q, R


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    assert args.output_module.startswith("h.") and args.output_module.endswith(".mlp.down_proj"), (
        f"--output-module must be 'h.<L>.mlp.down_proj', got {args.output_module!r}"
    )
    layer = int(args.output_module.split(".")[1])
    in_module = f"h.{layer}.mlp.c_fc"
    out_module = args.output_module
    gelu_path = f"h.{layer}.mlp.gelu"

    run_tag = f"L{layer}_c{args.subcomp_id}_r{args.rank}"
    out_dir = Path(args.out_dir) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    wandb_path = (
        args.wandb_path if args.wandb_path.startswith("wandb:") else f"wandb:{args.wandb_path}"
    )
    print(f"Loading PD run {wandb_path}...")
    run_info = ParamDecompRunInfo.from_path(wandb_path)
    pd_config = run_info.config
    model = ComponentModel.from_run_info(run_info).to(args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    assert in_module in model.components, (
        f"Module {in_module} not decomposed in this run. Available: {sorted(model.components.keys())[:5]}..."
    )
    assert out_module in model.components, f"Module {out_module} not decomposed in this run."

    V_cfc = model.components[in_module].V.detach().float()  # (d_model, C_in)
    V_dp_c = model.components[out_module].V.detach()[:, args.subcomp_id].contiguous().float()
    c_in = V_cfc.shape[1]
    d_intermediate = V_dp_c.shape[0]
    n_total_out = model.components[out_module].V.shape[1]
    assert 0 <= args.subcomp_id < n_total_out, (
        f"subcomp_id {args.subcomp_id} out of range [0, {n_total_out})"
    )
    print(
        f"Layer {layer} | c_fc C_in={c_in} | d_intermediate={d_intermediate} "
        f"| rank={args.rank} | target c={args.subcomp_id}"
    )

    Q, R = init_low_rank(d_intermediate, c_in, args.rank, args.device)
    opt = torch.optim.Adam([Q, R], lr=args.lr)

    final_val_frac = 1.0 if args.lr_fn_type == "constant" else args.final_val_frac
    schedule = ScheduleConfig(
        start_val=args.lr,
        warmup_pct=args.warmup_pct,
        final_val_frac=final_val_frac,
        fn_type=args.lr_fn_type,
    )

    train_loader, _ = train_loader_and_tokenizer(pd_config, args.batch_size)
    data_iter = loop_dataloader(train_loader)

    captured: dict[str, Tensor] = {}
    cfc_mod = model.target_model.get_submodule(in_module)
    gelu_mod = model.target_model.get_submodule(gelu_path)

    def _cfc_pre_hook(_mod: nn.Module, inputs: tuple[Tensor, ...]) -> None:
        captured["cfc_in"] = inputs[0].detach()

    def _gelu_pre_hook(_mod: nn.Module, inputs: tuple[Tensor, ...]) -> None:
        captured["h_pre_gelu"] = inputs[0].detach()

    h1 = cfc_mod.register_forward_pre_hook(_cfc_pre_hook)
    h2 = gelu_mod.register_forward_pre_hook(_gelu_pre_hook)

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or run_tag,
            config={
                **vars(args),
                "C_in": c_in,
                "d_intermediate": d_intermediate,
                "layer": layer,
                "in_module": in_module,
                "out_module": out_module,
                "n_total_out_subcomps": n_total_out,
                "pd_wandb_path": wandb_path,
            },
        )

    loss_hist: list[float] = []
    r2_hist: list[float] = []

    try:
        for step in range(args.steps):
            tokens = next(data_iter).to(args.device)

            captured.clear()
            with torch.no_grad(), bf16_autocast():
                _ = model.target_model(tokens)
            h = captured["cfc_in"].float()  # (B, S, d_model)
            h_pre_gelu = captured["h_pre_gelu"].float()  # (B, S, d_intermediate)

            with torch.no_grad():
                a_in = einops.einsum(h, V_cfc, "b s d, d c -> b s c")  # (B, S, C_in)
                label = einops.einsum(newgelu(h_pre_gelu), V_dp_c, "b s d, d -> b s")  # (B, S)

            low_rank_acts = einops.einsum(a_in, R, "b s c, r c -> b s r")
            pre_gelu_pred = einops.einsum(low_rank_acts, Q, "b s r, d r -> b s d")
            pred = einops.einsum(newgelu(pre_gelu_pred), V_dp_c, "b s d, d -> b s")

            diff = pred - label
            loss = diff.pow(2).mean()
            with torch.no_grad():
                var = label.var(unbiased=False)
                r2 = 1.0 - loss.detach() / (var + 1e-12)

            lr_now = get_scheduled_value(step, args.steps, schedule)
            for g in opt.param_groups:
                g["lr"] = lr_now

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            loss_hist.append(loss.item())
            r2_hist.append(float(r2.item()))

            log_now = (step + 1) % args.log_every == 0 or step == 0
            if log_now:
                w = min(len(loss_hist), args.ema_window)
                loss_ema = float(np.mean(loss_hist[-w:]))
                r2_ema = float(np.mean(r2_hist[-w:]))
                print(
                    f"step {step + 1:>5d}/{args.steps}  lr={lr_now:.3e}  "
                    f"loss={loss.item():.6f}  R^2={r2.item():+.4f}  "
                    f"(EMA{w} loss={loss_ema:.6f}  R^2={r2_ema:+.4f})  "
                    f"|label|_var={float(var):.3e}"
                )
                if use_wandb:
                    wandb.log(
                        {
                            "loss": loss.item(),
                            "r2": float(r2.item()),
                            "loss_ema": loss_ema,
                            "r2_ema": r2_ema,
                            "lr": lr_now,
                            "label_var": float(var.item()),
                        },
                        step=step + 1,
                    )
    finally:
        h1.remove()
        h2.remove()

    ckpt_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "Q": Q.detach().cpu(),
            "R": R.detach().cpu(),
            "V_dp_c": V_dp_c.cpu(),
            "args": vars(args),
            "layer": layer,
            "in_module": in_module,
            "out_module": out_module,
            "C_in": c_in,
            "d_intermediate": d_intermediate,
            "loss_history": loss_hist,
            "r2_history": r2_hist,
        },
        ckpt_path,
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(loss_hist, lw=0.8)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("step")
    ax[0].set_title("MSE loss")
    ax[1].plot(r2_hist, lw=0.8)
    ax[1].set_xlabel("step")
    ax[1].set_title("R² (variance recovered)")
    ax[1].axhline(0.0, color="k", lw=0.4, alpha=0.4)
    ax[1].axhline(1.0, color="k", lw=0.4, alpha=0.4)
    fig.suptitle(f"low-rank subcomp map | layer {layer} | c={args.subcomp_id} | rank={args.rank}")
    fig.tight_layout()
    plot_path = out_dir / "training_curves.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    print(f"\nSaved checkpoint to {ckpt_path}")
    print(f"Saved training curves to {plot_path}")

    if use_wandb:
        wandb.log({"training_curves": wandb.Image(str(plot_path))})
        wandb.finish()


if __name__ == "__main__":
    main()
