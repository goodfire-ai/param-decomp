"""Replace all blocks' attention with fewer heads AND all MLPs with a small bottleneck, jointly, on CI-masked forwards.

Same component-interface trick as `run.py`/`run_all.py`, applied to attention too: keep the
frozen component readers (`V_q`, `V_k`, `V_v`) and the frozen output writer (`U_o`) so the CI
masks still gate the same component spaces, and learn a fresh `n_heads`-head attention in
between (standard MHA, `head_dim` chosen freely). RoPE tables are rebuilt for the chosen
`head_dim` with the target's `rope_theta`/`rope_scaling` (via the vendored `_rotary_cos_sin`),
so a swept `head_dim` gets exactly the rotary a real model of that head_dim would use. The
target itself has 6 heads of dim 128, so this sweeps n_heads <= 6. MLPs use the bottleneck
from `run.py`. Everything is trained jointly against KL(teacher ‖ student) on output logits;
CI masks for all components are reused from the teacher pass; all decomposition components
stay frozen.

Run: python -m param_decomp_lab.experiments.mlp_compress.run_attn --n_heads H --head_dim D [--n_compressed 16] [--steps N] ...
"""

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import override

import einops
import fire
import torch
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv
from torch import Tensor, nn

from param_decomp.component_model import ComponentModel
from param_decomp.components import LinearComponents, init_param_
from param_decomp.masks import ComponentsMaskInfo
from param_decomp_lab.batch_and_loss_fns import calc_kl_divergence_lm
from param_decomp_lab.experiments.lm.run import SavedLMRun, build_lm_loader
from param_decomp_lab.experiments.mlp_compress.run import (
    RUN_DIR,
    CompressedMaskedMLP,
    compute_ci_and_teacher,
)
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR, REPO_ROOT

OUT_BASE = PARAM_DECOMP_OUT_DIR / "runs/s-55ea3f9b/attn_mlp_compress"
ORIG_N_HEADS = 6
ORIG_HEAD_DIM = 128


def mlp_module_names(block: int) -> tuple[str, str]:
    return f"h.{block}.mlp.c_fc", f"h.{block}.mlp.down_proj"


def attn_module_names(block: int) -> tuple[str, str, str, str]:
    base = f"h.{block}.attn"
    return f"{base}.q_proj", f"{base}.k_proj", f"{base}.v_proj", f"{base}.o_proj"


def rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class CompressedMaskedAttention(nn.Module):
    """Causal self-attention with `n_heads` heads operating in CI-mask component space.

    Reads input through the frozen q/k/v component readers (`V_*`, masked by CI), runs a fresh
    `n_heads`-head attention via learned projections, then writes back through the frozen o
    component writer (`U_o`, masked by CI). `rotary_cos`/`rotary_sin` must be sized for the
    chosen `head_dim` (rebuilt by the caller via `_rotary_cos_sin`).
    """

    def __init__(
        self,
        V_q: Tensor,
        V_k: Tensor,
        V_v: Tensor,
        U_o: Tensor,
        n_heads: int,
        head_dim: int,
        rotary_cos: Tensor,
        rotary_sin: Tensor,
    ):
        super().__init__()
        assert rotary_cos.shape[1] == head_dim and rotary_sin.shape[1] == head_dim
        self.register_buffer("V_q", V_q.clone())
        self.register_buffer("V_k", V_k.clone())
        self.register_buffer("V_v", V_v.clone())
        self.register_buffer("U_o", U_o.clone())
        self.register_buffer("rotary_cos", rotary_cos.clone())
        self.register_buffer("rotary_sin", rotary_sin.clone())
        self.n_heads = n_heads
        self.head_dim = head_dim
        inner = n_heads * head_dim
        C_q, C_k, C_v = V_q.shape[1], V_k.shape[1], V_v.shape[1]
        C_o = U_o.shape[0]
        self.W_q = nn.Parameter(torch.empty(C_q, inner))
        self.W_k = nn.Parameter(torch.empty(C_k, inner))
        self.W_v = nn.Parameter(torch.empty(C_v, inner))
        self.W_o = nn.Parameter(torch.empty(inner, C_o))
        if inner > 0:
            init_param_(self.W_q, fan_val=C_q, nonlinearity="linear")
            init_param_(self.W_k, fan_val=C_k, nonlinearity="linear")
            init_param_(self.W_v, fan_val=C_v, nonlinearity="linear")
            init_param_(self.W_o, fan_val=inner, nonlinearity="linear")

    def _to_heads(self, t: Tensor) -> Tensor:
        return einops.rearrange(t, "b s (h d) -> b h s d", h=self.n_heads)

    def _apply_rotary(self, x: Tensor, seq_len: int) -> Tensor:
        assert isinstance(self.rotary_cos, Tensor) and isinstance(self.rotary_sin, Tensor)
        cos = self.rotary_cos[:seq_len].to(x.dtype)
        sin = self.rotary_sin[:seq_len].to(x.dtype)
        return x * cos + rotate_half(x) * sin

    @override
    def forward(self, x: Tensor, ci_q: Tensor, ci_k: Tensor, ci_v: Tensor, ci_o: Tensor) -> Tensor:
        assert isinstance(self.V_q, Tensor) and isinstance(self.V_k, Tensor)
        assert isinstance(self.V_v, Tensor) and isinstance(self.U_o, Tensor)
        if self.n_heads == 0:
            return x.new_zeros(*x.shape[:-1], self.U_o.shape[1])
        seq_len = x.shape[1]
        a_q = einops.einsum(x, self.V_q, "... d, d C -> ... C") * ci_q
        a_k = einops.einsum(x, self.V_k, "... d, d C -> ... C") * ci_k
        a_v = einops.einsum(x, self.V_v, "... d, d C -> ... C") * ci_v

        q = self._to_heads(einops.einsum(a_q, self.W_q, "... C, C i -> ... i"))
        k = self._to_heads(einops.einsum(a_k, self.W_k, "... C, C i -> ... i"))
        v = self._to_heads(einops.einsum(a_v, self.W_v, "... C, C i -> ... i"))

        q = self._apply_rotary(q, seq_len)
        k = self._apply_rotary(k, seq_len)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = einops.rearrange(y, "b h s d -> b s (h d)")

        a_o = einops.einsum(y, self.W_o, "... i, i C -> ... C") * ci_o
        return einops.einsum(a_o, self.U_o, "... C, C d -> ... d")


@contextmanager
def blocks_replaced(
    mlp_modules: dict[int, nn.Module],
    attn_modules: dict[int, nn.Module],
    compressed_mlp: dict[int, CompressedMaskedMLP],
    compressed_attn: dict[int, CompressedMaskedAttention],
    ci: dict[str, Tensor],
) -> Iterator[None]:
    handles = []
    for block, mlp in mlp_modules.items():
        cfc_name, down_name = mlp_module_names(block)
        compressed = compressed_mlp[block]
        ci_cfc, ci_down = ci[cfc_name], ci[down_name]

        def make_mlp_hook(compressed: CompressedMaskedMLP, ci_cfc: Tensor, ci_down: Tensor):
            def hook(_module: nn.Module, args: tuple[Tensor, ...], _output: Tensor) -> Tensor:
                return compressed(args[0], ci_cfc, ci_down)

            return hook

        handles.append(mlp.register_forward_hook(make_mlp_hook(compressed, ci_cfc, ci_down)))

    for block, attn in attn_modules.items():
        q_name, k_name, v_name, o_name = attn_module_names(block)
        compressed = compressed_attn[block]
        ci_q, ci_k, ci_v, ci_o = ci[q_name], ci[k_name], ci[v_name], ci[o_name]

        def make_attn_hook(
            compressed: CompressedMaskedAttention,
            ci_q: Tensor,
            ci_k: Tensor,
            ci_v: Tensor,
            ci_o: Tensor,
        ):
            def hook(_module: nn.Module, args: tuple[Tensor, ...], _output: Tensor) -> Tensor:
                return compressed(args[0], ci_q, ci_k, ci_v, ci_o)

            return hook

        handles.append(
            attn.register_forward_hook(make_attn_hook(compressed, ci_q, ci_k, ci_v, ci_o))
        )
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def student_forward(
    comp_model: ComponentModel,
    batch: Tensor,
    ci: dict[str, Tensor],
    mask_infos: dict[str, ComponentsMaskInfo],
    mlp_modules: dict[int, nn.Module],
    attn_modules: dict[int, nn.Module],
    compressed_mlp: dict[int, CompressedMaskedMLP],
    compressed_attn: dict[int, CompressedMaskedAttention],
) -> Tensor:
    drop: set[str] = set()
    for block in mlp_modules:
        drop.update(mlp_module_names(block))
        drop.update(attn_module_names(block))
    student_mask_infos = {k: v for k, v in mask_infos.items() if k not in drop}
    with blocks_replaced(mlp_modules, attn_modules, compressed_mlp, compressed_attn, ci):
        return comp_model(batch, mask_infos=student_mask_infos)


def main(
    n_heads: int,
    head_dim: int,
    n_compressed: int = 16,
    steps: int = 20_000,
    batch_size: int = 32,
    lr: float = 1e-3,
    warmup_steps: int = 200,
    final_lr_frac: float = 0.1,
    eval_every: int = 250,
    n_eval_batches: int = 4,
    save_every: int = 2_500,
    seed: int = 0,
    use_wandb: bool = True,
) -> None:
    load_dotenv(REPO_ROOT / ".env")
    assert torch.cuda.is_available(), "needs a GPU"
    device = "cuda"
    torch.manual_seed(seed)
    autocast = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)  # noqa: E731

    out_dir = OUT_BASE / f"h{n_heads}_d{head_dim}_n{n_compressed}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=False)

    pd_run = SavedLMRun.from_path(RUN_DIR)
    comp_model = pd_run.load_model().to(device)
    comp_model.eval()
    comp_model.requires_grad_(False)

    attn0 = comp_model.target_model.get_submodule("h.0.attn")
    assert attn0.rotary_adjacent_pairs is False, "compressed attn uses half-split rotate_half"
    rotary_sin, rotary_cos = attn0.calculate_sin_cos_rotary(  # pyright: ignore[reportCallIssue]
        head_dim, attn0.n_ctx, base=attn0.rotary_base
    )
    rotary_cos, rotary_sin = rotary_cos.to(device), rotary_sin.to(device)

    n_blocks = 0
    while mlp_module_names(n_blocks)[0] in comp_model.components:
        n_blocks += 1
    assert n_blocks == 4, f"expected 4 blocks, found {n_blocks}"

    mlp_modules: dict[int, nn.Module] = {}
    attn_modules: dict[int, nn.Module] = {}
    compressed_mlp: dict[int, CompressedMaskedMLP] = {}
    compressed_attn: dict[int, CompressedMaskedAttention] = {}
    for block in range(n_blocks):
        cfc_name, down_name = mlp_module_names(block)
        cfc = comp_model.components[cfc_name]
        down = comp_model.components[down_name]
        assert isinstance(cfc, LinearComponents) and isinstance(down, LinearComponents)
        assert cfc.bias is None and down.bias is None, "target MLP is bias-free"
        assert cfc.d_out == down.d_in == 3072
        mlp = comp_model.target_model.get_submodule(f"h.{block}.mlp")
        mlp_modules[block] = mlp
        gelu = mlp.gelu
        assert isinstance(gelu, nn.Module)
        compressed_mlp[block] = CompressedMaskedMLP(
            V_cfc=cfc.V.data,
            U_down=down.U.data,
            n_compressed=n_compressed,
            activation=gelu,
            bypass=False,
        ).to(device)

        q_name, k_name, v_name, o_name = attn_module_names(block)
        q, k, v, o = (comp_model.components[n] for n in (q_name, k_name, v_name, o_name))
        assert all(isinstance(c, LinearComponents) for c in (q, k, v, o))
        assert q.bias is None and o.bias is None, "target attn is bias-free"
        assert q.d_in == k.d_in == v.d_in == 768 and o.d_out == 768
        attn = comp_model.target_model.get_submodule(f"h.{block}.attn")
        assert attn.head_dim == ORIG_HEAD_DIM
        compressed_attn[block] = CompressedMaskedAttention(
            V_q=q.V.data,
            V_k=k.V.data,
            V_v=v.V.data,
            U_o=o.U.data,
            n_heads=n_heads,
            head_dim=head_dim,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
        ).to(device)
        attn_modules[block] = attn

    params = [
        p for c in (*compressed_mlp.values(), *compressed_attn.values()) for p in c.parameters()
    ]
    n_trainable = sum(p.numel() for p in params if p.requires_grad)
    config = {
        "run": "s-55ea3f9b (via p-55ea3f9b)",
        "n_blocks": n_blocks,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "n_compressed_mlp": n_compressed,
        "orig_n_heads": ORIG_N_HEADS,
        "orig_head_dim": ORIG_HEAD_DIM,
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "warmup_steps": warmup_steps,
        "final_lr_frac": final_lr_frac,
        "seed": seed,
        "n_trainable_params": n_trainable,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"out_dir: {out_dir}")
    print(f"config: {json.dumps(config, indent=2)}")

    wb = None
    if use_wandb:
        wb = wandb.init(
            project="spd",
            name=f"compress-attn{n_heads}h-d{head_dim}-mlp{n_compressed}-s-55ea3f9b",
            group="attn_mlp_headdim_sweep",
            tags=["mlp_compress", "attn_compress"],
            config=config,
        )
        print(f"wandb: {wb.url}")

    train_loader = build_lm_loader(
        pd_run.cfg.target,
        pd_run.cfg.data,
        split="train",
        device=device,
        batch_size=batch_size,
        seed=seed,
    )
    eval_loader = build_lm_loader(
        pd_run.cfg.target,
        pd_run.cfg.data,
        split="eval",
        device=device,
        batch_size=batch_size,
        seed=seed,
    )
    eval_iter = iter(eval_loader)
    eval_batches = [next(eval_iter).to(device) for _ in range(n_eval_batches)]

    opt = torch.optim.Adam(params, lr=lr)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        cos = 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)).item())
        return final_lr_frac + (1 - final_lr_frac) * cos

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    eval_ctx = []
    with torch.no_grad(), autocast():
        for b in eval_batches:
            ci, mask_infos, target_logits, teacher_logits = compute_ci_and_teacher(comp_model, b)
            eval_ctx.append((b, ci, mask_infos, target_logits, teacher_logits))

    with torch.no_grad(), autocast():
        kl_teacher_vs_target = 0.0
        for _b, _ci, _mask_infos, target_logits, teacher_logits in eval_ctx:
            kl_teacher_vs_target += calc_kl_divergence_lm(
                pred=teacher_logits.float(), target=target_logits.float()
            ).item()
        kl_teacher_vs_target /= len(eval_ctx)
    references = {"kl_ci_masked_vs_target": kl_teacher_vs_target}
    print(f"references: {references}")
    (out_dir / "references.json").write_text(json.dumps(references, indent=2))
    if wb is not None:
        wb.summary.update(references)

    def run_eval() -> dict[str, float]:
        kl_vs_teacher = 0.0
        kl_vs_target = 0.0
        with torch.no_grad(), autocast():
            for b, ci, mask_infos, target_logits, teacher_logits in eval_ctx:
                student_logits = student_forward(
                    comp_model,
                    b,
                    ci,
                    mask_infos,
                    mlp_modules,
                    attn_modules,
                    compressed_mlp,
                    compressed_attn,
                ).float()
                kl_vs_teacher += calc_kl_divergence_lm(
                    pred=student_logits, target=teacher_logits.float()
                ).item()
                kl_vs_target += calc_kl_divergence_lm(
                    pred=student_logits, target=target_logits.float()
                ).item()
        return {
            "eval/kl_student_vs_teacher": kl_vs_teacher / len(eval_ctx),
            "eval/kl_student_vs_target": kl_vs_target / len(eval_ctx),
        }

    metrics_path = out_dir / "metrics.jsonl"
    train_iter = iter(train_loader)
    last_log_time = time.time()
    for step in range(steps):
        batch = next(train_iter).to(device)

        with torch.no_grad(), autocast():
            ci, mask_infos, _, teacher_logits = compute_ci_and_teacher(comp_model, batch)
        with autocast():
            student_logits = student_forward(
                comp_model,
                batch,
                ci,
                mask_infos,
                mlp_modules,
                attn_modules,
                compressed_mlp,
                compressed_attn,
            )
        loss = calc_kl_divergence_lm(pred=student_logits.float(), target=teacher_logits.float())

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        record: dict[str, float] = {
            "step": step,
            "train/kl": loss.item(),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        if step % eval_every == 0 or step == steps - 1:
            record |= run_eval()
            now = time.time()
            record["steps_per_s"] = eval_every / (now - last_log_time)
            last_log_time = now
            print(json.dumps(record))
            with metrics_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        if wb is not None:
            wb.log(record, step=step)

        if (step > 0 and step % save_every == 0) or step == steps - 1:
            torch.save(
                {
                    "mlp": {b: c.state_dict() for b, c in compressed_mlp.items()},
                    "attn": {b: c.state_dict() for b, c in compressed_attn.items()},
                },
                out_dir / f"compressed_{step}.pt",
            )

    final = run_eval()
    print(f"final: {final} | references: {references}")
    (out_dir / "final.json").write_text(json.dumps(final | references, indent=2))
    if wb is not None:
        wb.summary.update(final)
        wb.finish()


if __name__ == "__main__":
    fire.Fire(main)
