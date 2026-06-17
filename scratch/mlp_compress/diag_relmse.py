"""One-off: why does relative MSE at post_attn_0 start well below 1.0 while the other 7 points start ~1.0?"""

import torch
from dotenv import load_dotenv
from torch import Tensor, nn

from param_decomp.components import LinearComponents
from param_decomp_lab.experiments.lm.run import SavedLMRun, build_lm_loader
from param_decomp_lab.infra.settings import REPO_ROOT
from run import CompressedMaskedMLP, RUN_DIR, compute_ci_and_teacher
from run_attn import HEAD_DIM, CompressedMaskedAttention, attn_module_names, blocks_replaced, mlp_module_names, student_forward
from run_mse import capture_residuals, relative_mse, resid_point_names


def main(n_heads: int = 2, n_compressed: int = 64, seed: int = 0) -> None:
    load_dotenv(REPO_ROOT / ".env")
    device = "cuda"
    torch.manual_seed(seed)
    pd_run = SavedLMRun.from_path(RUN_DIR)
    comp_model = pd_run.load_model().to(device)
    comp_model.eval()
    comp_model.requires_grad_(False)
    n_blocks = 4
    point_names = resid_point_names(n_blocks)

    mlp_modules: dict[int, nn.Module] = {}
    attn_modules: dict[int, nn.Module] = {}
    compressed_mlp: dict[int, CompressedMaskedMLP] = {}
    compressed_attn: dict[int, CompressedMaskedAttention] = {}
    for b in range(n_blocks):
        cfc_name, down_name = mlp_module_names(b)
        cfc = comp_model.components[cfc_name]
        down = comp_model.components[down_name]
        assert isinstance(cfc, LinearComponents) and isinstance(down, LinearComponents)
        mlp = comp_model.target_model.get_submodule(f"h.{b}.mlp")
        mlp_modules[b] = mlp
        compressed_mlp[b] = CompressedMaskedMLP(
            V_cfc=cfc.V.data, U_down=down.U.data, n_compressed=n_compressed, gelu=mlp.gelu
        ).to(device)
        q_name, k_name, v_name, o_name = attn_module_names(b)
        q, k, v, o = (comp_model.components[n] for n in (q_name, k_name, v_name, o_name))
        attn = comp_model.target_model.get_submodule(f"h.{b}.attn")
        compressed_attn[b] = CompressedMaskedAttention(
            V_q=q.V.data, V_k=k.V.data, V_v=v.V.data, U_o=o.U.data,
            n_heads=n_heads, rotary_cos=attn.rotary_cos, rotary_sin=attn.rotary_sin,
        ).to(device)
        attn_modules[b] = attn

    loader = build_lm_loader(pd_run.cfg.target, pd_run.cfg.data, split="eval", device=device, batch_size=8, seed=seed)
    batch = next(iter(loader)).to(device)

    autocast = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)  # noqa: E731
    emb_holder: dict[str, Tensor] = {}
    h0 = comp_model.target_model.get_submodule("h.0")

    def emb_pre(_m: nn.Module, args: tuple[Tensor, ...]) -> None:
        emb_holder["emb"] = args[0]

    with torch.no_grad(), autocast():
        handle = h0.register_forward_pre_hook(emb_pre)
        with capture_residuals(comp_model, n_blocks) as tr:
            ci, mask_infos, _, _ = compute_ci_and_teacher(comp_model, batch)
        teacher_resids = {k: v.detach().clone() for k, v in tr.items()}
        emb = emb_holder["emb"].detach().clone()
        with capture_residuals(comp_model, n_blocks) as sr:
            student_forward(comp_model, batch, ci, mask_infos, mlp_modules, attn_modules, compressed_mlp, compressed_attn)
        student_resids = {k: v.detach().clone() for k, v in sr.items()}
        handle.remove()

    def ms(x: Tensor) -> float:
        return x.float().pow(2).mean().item()

    print(f"n_heads={n_heads} n_compressed={n_compressed}")
    print(f"{'point':<14} {'relmse':>8} {'ms(t)':>10} {'ms(s)':>10} {'ms(s-t)':>10} {'ms(teacher_contrib)':>20} {'ms(student_contrib)':>20}")
    prev_t = emb
    prev_s = emb
    print(f"{'emb(block0_in)':<14} {'-':>8} {ms(emb):>10.3f} {ms(emb):>10.3f} {'-':>10} {'-':>20} {'-':>20}")
    for k in point_names:
        t = teacher_resids[k]
        s = student_resids[k]
        relmse = relative_mse(s.float(), t.float()).item()
        t_contrib = t - prev_t
        s_contrib = s - prev_s
        print(f"{k:<14} {relmse:>8.3f} {ms(t):>10.3f} {ms(s):>10.3f} {ms(s - t):>10.3f} {ms(t_contrib):>20.3f} {ms(s_contrib):>20.3f}")
        prev_t = t
        prev_s = s


if __name__ == "__main__":
    import fire

    fire.Fire(main)
