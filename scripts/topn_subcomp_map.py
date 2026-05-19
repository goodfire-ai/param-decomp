"""Top-N column-selection baseline for low-rank subcomponent mapping.

For a chosen output subcomponent c in `h.<L>.mlp.down_proj` of a `LlamaSimpleMLP`
PD run, this script:

  1. Computes (grad * input) attributions of c's activation w.r.t. each c_fc
     input subcomponent c' over `--n-attr-batches` batches:

         attr(c'; b, t) = ( sum_d V_dp[d, c] * NewGELU'(h_pre_gelu[b, t, d]) * U_cfc[c', d] )
                          * a^{in}_{b,t,c'}

     The closed form is used (analytically equivalent to autograd `d a_out_c /
     d a_in_c' * a_in_c'`) for efficiency. Subcomponents are scored by
     `mean_{b,t} |attr|`.

  2. For each N in `--n-keep`, keeps only the top-N rows of `U_cfc` (zeros the
     rest) and evaluates the approximation

         pred(b, t) = V_dp[:, c] . NewGELU( sum_{c' in top-N} U_cfc[c', :] * a^{in}_{b,t,c'} )

     against the true subcomp-c activation V_dp[:, c] . NewGELU(c_fc(h)) on
     `--n-eval-batches` batches. Reports per-N MSE and R² (variance recovered),
     plus a `full` baseline that uses every row of U_cfc (note: this is not
     R²=1 in general, because the c_fc decomposition `U_cfc @ V_cfc.T` only
     approximates the target c_fc weight up to the trained delta component;
     `full` is the ceiling reachable by any subset of U_cfc rows).

Intended as a comparison baseline for the learned low-rank script
`scripts/learn_low_rank_subcomp_map.py`.
"""

import argparse
import json
from pathlib import Path
from typing import TypedDict

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch import Tensor, nn

from param_decomp.data import loop_dataloader, train_loader_and_tokenizer
from param_decomp.models.component_model import ComponentModel, ParamDecompRunInfo
from param_decomp.utils.general_utils import bf16_autocast


class _Row(TypedDict):
    condition: str
    n_keep: int | None
    mse: float
    r2: float


def newgelu(x: Tensor) -> Tensor:
    """NewGELU, matching `LlamaSimpleMLP`'s activation."""
    c = float(np.sqrt(2.0 / np.pi))
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x.pow(3))))


def newgelu_prime(h: Tensor) -> Tensor:
    """Element-wise derivative of NewGELU."""
    c = float(np.sqrt(2.0 / np.pi))
    inner = c * (h + 0.044715 * h.pow(3))
    tanh = torch.tanh(inner)
    sech2 = 1.0 - tanh.pow(2)
    d_inner = c * (1.0 + 3.0 * 0.044715 * h.pow(2))
    return 0.5 * (1.0 + tanh) + 0.5 * h * sech2 * d_inner


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
    p.add_argument("--subcomp-id", type=int, required=True)
    p.add_argument(
        "--n-keep",
        type=str,
        required=True,
        help="Comma-separated list of N values, e.g. '2,4,8,16,32', or just '8'.",
    )
    p.add_argument(
        "--n-attr-batches",
        type=int,
        default=10,
        help="Number of batches used to compute attribution scores.",
    )
    p.add_argument(
        "--n-eval-batches",
        type=int,
        default=10,
        help="Number of batches used to evaluate MSE/R² of each top-N approximation.",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--out-dir",
        default=str(Path(__file__).parent / "topn_subcomp_map_out"),
    )
    p.add_argument("--wandb-project", default="spd-low-rank-subcomp-map")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


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

    n_keep_list = sorted({int(x) for x in args.n_keep.split(",") if x.strip()})
    assert len(n_keep_list) > 0, "--n-keep must contain at least one positive int"
    assert all(n > 0 for n in n_keep_list), "--n-keep values must be positive"

    run_tag = f"L{layer}_c{args.subcomp_id}_topN_{'-'.join(str(n) for n in n_keep_list)}"
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

    assert in_module in model.components, f"{in_module} not decomposed in this run"
    assert out_module in model.components, f"{out_module} not decomposed in this run"

    V_cfc = model.components[in_module].V.detach().float()  # (d_model, C_in)
    U_cfc = model.components[in_module].U.detach().float()  # (C_in, d_intermediate)
    V_dp_c = model.components[out_module].V.detach()[:, args.subcomp_id].contiguous().float()
    c_in = V_cfc.shape[1]
    d_intermediate = V_dp_c.shape[0]
    n_total_out = model.components[out_module].V.shape[1]
    assert 0 <= args.subcomp_id < n_total_out, (
        f"subcomp_id {args.subcomp_id} out of range [0, {n_total_out})"
    )
    max_n = max(n_keep_list)
    assert max_n <= c_in, f"max N={max_n} exceeds C_in={c_in}"

    print(
        f"Layer {layer} | c_fc C_in={c_in} | d_intermediate={d_intermediate} "
        f"| target c={args.subcomp_id} | N sweep={n_keep_list}"
    )

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
                "n_keep_list": n_keep_list,
                "method": "topn_attribution",
            },
        )

    train_loader, _ = train_loader_and_tokenizer(pd_config, args.batch_size)
    data_iter = loop_dataloader(train_loader)

    try:
        print(f"\nPhase 1: attribution over {args.n_attr_batches} batches")
        abs_attr_sum = torch.zeros(c_in, device=args.device)
        n_pos_attr = 0
        for b_idx in range(args.n_attr_batches):
            tokens = next(data_iter).to(args.device)
            captured.clear()
            with torch.no_grad(), bf16_autocast():
                _ = model.target_model(tokens)
            h = captured["cfc_in"].float()
            h_pre_gelu = captured["h_pre_gelu"].float()
            with torch.no_grad():
                a_in = einops.einsum(h, V_cfc, "b s d, d c -> b s c")
                gp = newgelu_prime(h_pre_gelu)
                # J[b,t,c'] = sum_d V_dp_c[d] * gp[b,t,d] * U_cfc[c',d]
                vg = einops.einsum(gp, V_dp_c, "b s d, d -> b s d")
                jac = einops.einsum(vg, U_cfc, "b s d, c d -> b s c")
                attribution = jac * a_in
                abs_attr_sum += attribution.abs().sum(dim=(0, 1))
            n_pos_attr += a_in.shape[0] * a_in.shape[1]
            print(f"  attr batch {b_idx + 1:>3d}/{args.n_attr_batches}")

        attr_score = (abs_attr_sum / n_pos_attr).cpu()
        ranked = torch.argsort(attr_score, descending=True)

        print(f"\nPhase 2: top-N evaluation over {args.n_eval_batches} batches")
        eval_keys = ["full"] + [f"top{n}" for n in n_keep_list]
        sum_sq_err = {k: 0.0 for k in eval_keys}
        sum_label = 0.0
        sum_sq_label = 0.0
        n_pos_eval = 0

        # Pre-slice rows of U_cfc per N for efficiency.
        top_idx_per_n = {n: ranked[:n].to(args.device) for n in n_keep_list}
        u_cfc_top_per_n = {n: U_cfc[top_idx_per_n[n]] for n in n_keep_list}

        for b_idx in range(args.n_eval_batches):
            tokens = next(data_iter).to(args.device)
            captured.clear()
            with torch.no_grad(), bf16_autocast():
                _ = model.target_model(tokens)
            h = captured["cfc_in"].float()
            h_pre_gelu = captured["h_pre_gelu"].float()
            with torch.no_grad():
                a_in = einops.einsum(h, V_cfc, "b s d, d c -> b s c")
                label = einops.einsum(newgelu(h_pre_gelu), V_dp_c, "b s d, d -> b s")

                bsz, seq = label.shape
                n_pos_eval += bsz * seq
                sum_label += label.sum().item()
                sum_sq_label += label.pow(2).sum().item()

                pre_gelu_full = einops.einsum(a_in, U_cfc, "b s c, c d -> b s d")
                pred_full = einops.einsum(newgelu(pre_gelu_full), V_dp_c, "b s d, d -> b s")
                sum_sq_err["full"] += (pred_full - label).pow(2).sum().item()

                for n in n_keep_list:
                    a_in_top = a_in[..., top_idx_per_n[n]]
                    pre_gelu_pred = einops.einsum(
                        a_in_top, u_cfc_top_per_n[n], "b s n, n d -> b s d"
                    )
                    pred = einops.einsum(newgelu(pre_gelu_pred), V_dp_c, "b s d, d -> b s")
                    sum_sq_err[f"top{n}"] += (pred - label).pow(2).sum().item()
            print(f"  eval batch {b_idx + 1:>3d}/{args.n_eval_batches}")
    finally:
        h1.remove()
        h2.remove()

    mean_label = sum_label / n_pos_eval
    var_label = sum_sq_label / n_pos_eval - mean_label**2
    var_label = max(var_label, 1e-12)

    results: list[_Row] = []
    print(f"\n{'condition':<10} {'N':>6} {'MSE':>14} {'R²':>10}")
    print("-" * 44)
    for key in eval_keys:
        mse = sum_sq_err[key] / n_pos_eval
        r2 = 1.0 - mse / var_label
        n_val: int | None = None if key == "full" else int(key.removeprefix("top"))
        results.append({"condition": key, "n_keep": n_val, "mse": mse, "r2": r2})
        n_str = "all" if n_val is None else str(n_val)
        print(f"{key:<10} {n_str:>6} {mse:>14.6e} {r2:>+10.4f}")

    summary = {
        "args": vars(args),
        "layer": layer,
        "in_module": in_module,
        "out_module": out_module,
        "subcomp_id": args.subcomp_id,
        "C_in": c_in,
        "d_intermediate": d_intermediate,
        "n_pos_attr": n_pos_attr,
        "n_pos_eval": n_pos_eval,
        "mean_label": mean_label,
        "var_label": var_label,
        "n_keep_list": n_keep_list,
        "results": results,
    }

    torch.save(
        {
            **summary,
            "attr_score": attr_score,
            "ranked_indices": ranked,
            "V_dp_c": V_dp_c.cpu(),
            "top_idx_per_n": {n: top_idx_per_n[n].cpu() for n in n_keep_list},
        },
        out_dir / "results.pt",
    )
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if len(n_keep_list) > 1:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ns = [r["n_keep"] for r in results if r["n_keep"] is not None]
        r2s = [r["r2"] for r in results if r["n_keep"] is not None]
        full_r2 = next(r["r2"] for r in results if r["condition"] == "full")
        ax.semilogx(ns, r2s, "o-", label="top-N column selection")
        ax.axhline(
            full_r2, color="k", ls="--", lw=0.8, label=f"all {c_in} subcomps (R²={full_r2:.4f})"
        )
        ax.axhline(0, color="k", lw=0.4, alpha=0.4)
        ax.set_xlabel("N (kept c_fc subcomponents)")
        ax.set_ylabel("R²")
        ax.set_title(f"Top-N approximation | layer {layer} | c={args.subcomp_id}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        r2_plot = out_dir / "r2_vs_n.png"
        fig.savefig(r2_plot, dpi=140)
        plt.close(fig)
    else:
        r2_plot = None

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    sorted_scores = attr_score[ranked].numpy()
    ax.semilogy(np.clip(sorted_scores, 1e-20, None))
    ax.set_xlabel("Rank")
    ax.set_ylabel("Mean |attribution| per c'")
    ax.set_title(f"Attribution scores (sorted) | layer {layer} | c={args.subcomp_id}")
    ax.grid(alpha=0.3)
    for n in n_keep_list:
        ax.axvline(n, color="r", lw=0.5, alpha=0.5)
    attr_plot = out_dir / "attribution_scores.png"
    fig.tight_layout()
    fig.savefig(attr_plot, dpi=140)
    plt.close(fig)

    print(f"\nSaved checkpoint to {out_dir / 'results.pt'}")
    print(f"Saved summary to {out_dir / 'summary.json'}")
    print(f"Saved attribution-score plot to {attr_plot}")
    if r2_plot is not None:
        print(f"Saved R²-vs-N plot to {r2_plot}")

    if use_wandb:
        for r in results:
            cond = str(r["condition"])
            wandb.summary[f"{cond}/mse"] = r["mse"]
            wandb.summary[f"{cond}/r2"] = r["r2"]
        table = wandb.Table(columns=["condition", "n_keep", "mse", "r2"])
        for r in results:
            table.add_data(r["condition"], r["n_keep"], r["mse"], r["r2"])
        log_payload: dict[str, object] = {
            "results": table,
            "attribution_scores_plot": wandb.Image(str(attr_plot)),
        }
        if r2_plot is not None:
            log_payload["r2_vs_n_plot"] = wandb.Image(str(r2_plot))
        wandb.log(log_payload)
        wandb.finish()


if __name__ == "__main__":
    main()
