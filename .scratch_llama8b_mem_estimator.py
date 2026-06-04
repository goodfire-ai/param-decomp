"""Per-pool resident + activation memory estimator for 3-pool decomposition.

Calibrated against the live GPT2-XL q/k run (p-b6505e9c, "big512"): known peaks
LW 37.6 / CI 131.9 / PPGD 103.1 GB. Then projected to Llama-3.1-8B decomposing the
3 SwiGLU matrices (gate/up/down) at one layer, with a CI fn sized to 10x the target params.

Run:  uv run python .scratch_llama8b_mem_estimator.py

Resident is exact (param counts x dtype/optimizer bytes). Activation is dominant-term
only -- the rough part; we sanity-check it against the GPT2-XL peaks before trusting the
Llama projection. All numbers GB = 1024^3 bytes.
"""

from dataclasses import dataclass

GB = 1024**3
KL_CHUNK = 4  # fused-KL processes the vocab/logits in chunks; only a fraction resident


# ----------------------------------------------------------------------------- model specs
@dataclass
class TargetSpec:
    name: str
    d_model: int
    n_layers: int
    vocab: int
    # decomposed sites: (path, d_in, d_out) for each matrix
    sites: list[tuple[str, int, int]]
    decomposed_layer: int  # which block index the sites live in (for autograd-graph depth)
    # per-block forward activation footprint knobs
    intermediate: int  # MLP hidden width (GELU 1x; SwiGLU we double-count gate+up below)
    swiglu: bool
    params_total: float  # full model param count (for the frozen-target resident floor)


GPT2_XL = TargetSpec(
    name="GPT2-XL",
    d_model=1600,
    n_layers=48,
    vocab=50257,
    sites=[(f"h.{i}.attn.{p}_proj", 1600, 1600) for i in range(48) for p in ("q", "k")],
    decomposed_layer=0,  # q/k span all layers; forward is whole-model regardless
    intermediate=6400,
    swiglu=False,
    params_total=1.556e9,
)

LLAMA_8B = TargetSpec(
    name="Llama-3.1-8B (gate/up/down @ L18)",
    d_model=4096,
    n_layers=32,
    vocab=128256,
    sites=[
        ("model.layers.18.mlp.gate_proj", 4096, 14336),
        ("model.layers.18.mlp.up_proj", 4096, 14336),
        ("model.layers.18.mlp.down_proj", 14336, 4096),
    ],
    decomposed_layer=18,
    intermediate=14336,
    swiglu=True,
    params_total=8.03e9,
)


# ----------------------------------------------------------------------------- topology / run
@dataclass
class Run:
    target: TargetSpec
    C: int  # components per site
    ci_params: float  # CI-fn param count
    ci_d_model: int
    ci_n_blocks: int
    ci_mlp_hidden: int
    bl_lw: int  # per-rank batch on each pool
    bl_ci: int
    bl_ppgd: int
    n_lw_sites_per_rank: int  # sites an LW rank owns (joint recon, 1 fwd)
    S: int = 1024
    target_dtype_bytes: int = 4  # frozen target: 4=fp32, 2=bf16
    ckpt: bool = True  # activation checkpointing of the recon forward
    # calibration overhead multipliers (allocator frag + un-modelled buffers), fit on GPT2-XL
    lw_overhead: float = 1.0
    ci_overhead: float = 1.0
    ppgd_overhead: float = 1.0


# ----------------------------------------------------------------------------- param counts
def component_params(run: Run) -> float:
    """V[d_in,C] + U[C,d_out] per site, summed over all sites."""
    return sum(C_in_out(run, d_in, d_out) for _, d_in, d_out in run.target.sites)


def C_in_out(run: Run, d_in: int, d_out: int) -> float:
    return run.C * (d_in + d_out)


def ci_fn_param_estimate(d_model, n_blocks, mlp_hidden, total_input_dim, total_c) -> float:
    """GlobalSharedTransformerCiFn: input proj + n transformer blocks + output head."""
    input_proj = total_input_dim * d_model + d_model
    output_head = d_model * total_c + total_c
    # per block: attn qkv+out (4 d^2) + MLP (2 d*hidden) + 2 layernorms (2d)
    per_block = 4 * d_model * d_model + 2 * d_model * mlp_hidden + 2 * d_model
    return input_proj + output_head + n_blocks * per_block


# ----------------------------------------------------------------------------- resident GB
def resident_gb(run: Run) -> dict[str, dict[str, float]]:
    t = run.target
    n_sites = len(t.sites)
    comp_total = component_params(run)
    comp_per_lw_rank = comp_total * run.n_lw_sites_per_rank / n_sites

    target_gb = t.params_total * run.target_dtype_bytes / GB

    out = {}
    # CI pool: frozen target + CI fn (AdamW: 4 param +4 grad +4 m +4 v = 16 B/param)
    out["CI"] = {
        "target(frozen)": target_gb,
        "ci_fn(AdamW)": run.ci_params * 16 / GB,
    }
    # LW pool: frozen target + owned V/U (AdamW 16 B/param). CI fn dropped.
    out["LW"] = {
        "target(frozen)": target_gb,
        "components(AdamW)": comp_per_lw_rank * 16 / GB,
    }
    # PPGD pool: frozen target + FULL V/U replica (param+grad = 8 B/param, no optim) + sources
    sources = n_sites * run.bl_ppgd * run.S * (run.C + 1) * 4 / GB  # fp32 PGD sources
    out["PPGD"] = {
        "target(frozen)": target_gb,
        "components(replica)": comp_total * 8 / GB,
        "pgd_sources": sources,
    }
    return out


# ----------------------------------------------------------------------------- activation GB
def activation_gb(run: Run) -> dict[str, dict[str, float]]:
    t = run.target
    n_sites = len(t.sites)
    total_c = n_sites * run.C
    total_input_dim = sum(d_in for _, d_in, _ in t.sites)
    S = run.S

    out = {}

    # ---- CI pool: harvest H (fp32 upcast), concat, output head, CI value/grad buffers
    H_cache = total_input_dim * run.bl_ci * S * 4 / GB  # 96x[bl,S,d_in] fp32
    concat = total_input_dim * run.bl_ci * S * 4 / GB  # [bl,S,total_input] fp32
    ci_out = total_c * run.bl_ci * S * 2 / GB  # output head bf16
    ci_values = 3 * total_c * run.bl_ci * S * 2 / GB  # lower/upper/pre bf16
    g_ci = 2 * total_c * run.bl_ci * S * 4 / GB  # grad dest from LW+PPGD fp32
    # transformer block activations (ckpt+compile saves most); residual stream retained
    ci_blocks = run.ci_d_model * run.bl_ci * S * 2 * (run.ci_n_blocks + 1) / GB
    out["CI"] = {
        "H_cache(fp32,x2 prefetch)": 2 * H_cache,
        "concat(fp32)": concat,
        "ci_output_head": ci_out,
        "ci_values(low/up/pre)": ci_values,
        "g_ci_recv(fp32)": g_ci,
        "ci_transformer_blocks": ci_blocks,
    }

    # ---- LW pool: full target forward (masked recon) + fused-KL logits + CI leaf
    # autograd graph spans layers >= decomposed_layer + head; clean prefix is no_grad.
    graph_layers = t.n_layers - t.decomposed_layer
    resid = t.d_model * run.bl_lw * S * 2 / GB  # one residual-stream tensor bf16
    if run.ckpt:
        # ckpt: ~1 block's intermediates resident at a time + retained block inputs
        mlp_mult = 2 if t.swiglu else 1
        fwd_act = (
            resid * graph_layers  # retained block boundaries
            + t.intermediate * mlp_mult * run.bl_lw * S * 2 / GB
        )  # 1 block recompute
    else:
        mlp_mult = 2 if t.swiglu else 1
        fwd_act = (resid + t.intermediate * mlp_mult * run.bl_lw * S * 2 / GB) * graph_layers
    target_local = t.d_model * run.bl_lw * S * 2 / GB  # cached clean hidden bf16
    kl_logits = 2 * t.vocab * run.bl_lw * S * 2 / GB / KL_CHUNK  # fused/chunked KL, pred+target
    ci_leaf = run.n_lw_sites_per_rank * run.C * run.bl_lw * S * 4 / GB  # fp32 re-leaf (F-DTYPE-2)
    out["LW"] = {
        "recon_fwd_act": fwd_act,
        "target_local(clean)": target_local,
        "kl_logits(chunked)": kl_logits,
        "ci_leaf(fp32)": ci_leaf,
    }

    # ---- PPGD pool: like LW but full-model CI, 3 forwards (2 warmup+1), NOT checkpointed
    graph_layers_p = t.n_layers - t.decomposed_layer
    mlp_mult = 2 if t.swiglu else 1
    p_fwd_one = (t.d_model + t.intermediate * mlp_mult) * run.bl_ppgd * S * 2 / GB * graph_layers_p
    p_kl = 2 * t.vocab * run.bl_ppgd * S * 2 / GB / KL_CHUNK
    p_ci_full = n_sites * run.C * run.bl_ppgd * S * 4 / GB  # full-model CI fp32 releaf
    out["PPGD"] = {
        "recon_fwd_act(no ckpt)": p_fwd_one,  # one graph live at a time (retain=False)
        "kl_logits(chunked)": p_kl,
        "ci_full_releaf(fp32)": p_ci_full,
    }
    return out


# ----------------------------------------------------------------------------- report
def total(d: dict[str, float]) -> float:
    return sum(d.values())


def report(run: Run, known_peaks: dict[str, float] | None = None):
    res = resident_gb(run)
    act = activation_gb(run)
    overhead = {"LW": run.lw_overhead, "CI": run.ci_overhead, "PPGD": run.ppgd_overhead}
    print(
        f"\n{'=' * 78}\n{run.target.name}   C={run.C}  CI={run.ci_params / 1e9:.2f}B  "
        f"bl(lw/ci/ppgd)={run.bl_lw}/{run.bl_ci}/{run.bl_ppgd}  "
        f"target={run.target_dtype_bytes * 8}b  ckpt={run.ckpt}\n{'=' * 78}"
    )
    comp = component_params(run)
    print(
        f"  target params {run.target.params_total / 1e9:.2f}B | "
        f"V/U total {comp / 1e6:.1f}M | CI fn {run.ci_params / 1e9:.2f}B "
        f"({run.ci_params / sum(run.C * (a + b) for _, a, b in run.target.sites):.0f}x V/U; "
        f"{run.ci_params / sum((a * b) for _, a, b in run.target.sites):.1f}x target-mat)"
    )
    for pool in ("CI", "LW", "PPGD"):
        r, a = res[pool], act[pool]
        rt, at = total(r), total(a)
        peak = (rt + at) * overhead[pool]
        print(
            f"\n  --- {pool} pool ---  resident {rt:6.1f}  + activation {at:6.1f}  "
            f"= {rt + at:6.1f} GB  (x{overhead[pool]:.2f} -> {peak:6.1f})"
        )
        for k, v in r.items():
            print(f"      R  {k:28s} {v:7.2f}")
        for k, v in a.items():
            print(f"      A  {k:28s} {v:7.2f}")
        if known_peaks and pool in known_peaks:
            kp = known_peaks[pool]
            print(
                f"      ** predicted {peak:.1f} vs KNOWN {kp:.1f} GB  "
                f"(err {100 * (peak - kp) / kp:+.0f}%)"
            )
        if peak > 191:
            print("      !! EXCEEDS 191 GB B200")
    return res, act


# ============================================================================= GPT2-XL calib
gpt2_run = Run(
    target=GPT2_XL,
    C=1024,
    ci_params=ci_fn_param_estimate(4096, 8, 16384, 96 * 1600, 96 * 1024),
    ci_d_model=4096,
    ci_n_blocks=8,
    ci_mlp_hidden=16384,
    bl_lw=64,
    bl_ci=16,
    bl_ppgd=8,
    n_lw_sites_per_rank=6,  # 16 block-groups x 6 sites x DDP8
)
# fit overheads to the three known peaks
KNOWN = {"LW": 37.6, "CI": 131.9, "PPGD": 103.1}
res, act = resident_gb(gpt2_run), activation_gb(gpt2_run)
gpt2_run.lw_overhead = KNOWN["LW"] / (total(res["LW"]) + total(act["LW"]))
gpt2_run.ci_overhead = KNOWN["CI"] / (total(res["CI"]) + total(act["CI"]))
gpt2_run.ppgd_overhead = KNOWN["PPGD"] / (total(res["PPGD"]) + total(act["PPGD"]))
print(
    "# Calibration overheads fit on GPT2-XL big512 peaks:",
    {
        k: round(v, 2)
        for k, v in {
            "LW": gpt2_run.lw_overhead,
            "CI": gpt2_run.ci_overhead,
            "PPGD": gpt2_run.ppgd_overhead,
        }.items()
    },
)
report(gpt2_run, KNOWN)


# ============================================================================= Llama 8B
# CI fn sized to ~10x target params (1.76B). Search d_model/n_blocks to hit it.
def size_ci_for_budget(budget, total_input_dim, total_c, d_model=4096, mlp_hidden=16384):
    base = ci_fn_param_estimate(d_model, 0, mlp_hidden, total_input_dim, total_c)
    per_block = 4 * d_model * d_model + 2 * d_model * mlp_hidden + 2 * d_model
    n = max(1, round((budget - base) / per_block))
    return n, ci_fn_param_estimate(d_model, n, mlp_hidden, total_input_dim, total_c)


# carry GPT2-XL's calibrated overheads to Llama (same code paths)
OV = dict(
    lw_overhead=gpt2_run.lw_overhead,
    ci_overhead=gpt2_run.ci_overhead,
    ppgd_overhead=gpt2_run.ppgd_overhead,
)

for C in (1024, 4096):
    tin = sum(d_in for _, d_in, _ in LLAMA_8B.sites)  # 4096+4096+14336 = 22528
    tc = 3 * C
    n_blk, ci_p = size_ci_for_budget(1.7616e9, tin, tc)
    for dtype_b, bl_lw, bl_ppgd, bl_ci in [(4, 8, 4, 8), (2, 16, 8, 16)]:
        run = Run(
            target=LLAMA_8B,
            C=C,
            ci_params=ci_p,
            ci_d_model=4096,
            ci_n_blocks=n_blk,
            ci_mlp_hidden=16384,
            bl_lw=bl_lw,
            bl_ci=bl_ci,
            bl_ppgd=bl_ppgd,
            n_lw_sites_per_rank=3,  # only 3 sites total -> one block owns all 3
            target_dtype_bytes=dtype_b,
            **OV,
        )
        report(run)
