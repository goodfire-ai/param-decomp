"""Theoretical-minimum per-GPU peak memory for one full32L VPD training step.

A FLOOR, not a prediction: it sums the IRREDUCIBLE resident terms (ZeRO-1 optimizer
state ÷N, the ÷fsdp bf16 compute weights, the frozen target gathered ÷fsdp, and ONE
forward's activation working set) under the assumption of perfect remat + minimal weight
residency. The gap between this floor and the MEASURED compiled peak (`memreport`) is the
"reducible transient" budget — what sensible compilation/scheduling could recover.

Every term is computed from the run config (C-values, CI arch) + the public Llama-3.1-8B
dims, and labelled EXACT vs ASSUMED. Run:

    python -m param_decomp.tools.theoretical_min_memory param_decomp/configs/<cfg>.yaml [--dp 32 --fsdp 8]

Claim under validation (see lore `state--full32l-mfu`): for the production config
(llama8b_full32L_HSDP_b32_dp32, dp=32, fsdp=8) the floor is ~33 GB vs a measured 96 GB
peak ⇒ ~63 GB is reducible transient. Validate by re-running against the cited commit.
"""

import sys
from pathlib import Path
from typing import Any

import yaml

# Llama-3.1-8B dims (HF `meta-llama/Llama-3.1-8B` config.json — public, verifiable).
D_MODEL = 4096
N_LAYERS = 32
INTERMEDIATE = 14336
VOCAB = 128256
Q_OUT = 4096  # 32 heads x 128
KV_OUT = 1024  # 8 kv heads x 128
SEQ = 512

# W[d_out, d_in] per kind -> (d_in, d_out) for V[d_in,C] @ U[C,d_out].
KIND_DIMS = {
    "q_proj": (D_MODEL, Q_OUT),
    "k_proj": (D_MODEL, KV_OUT),
    "v_proj": (D_MODEL, KV_OUT),
    "o_proj": (Q_OUT, D_MODEL),
    "gate_proj": (D_MODEL, INTERMEDIATE),
    "up_proj": (D_MODEL, INTERMEDIATE),
    "down_proj": (INTERMEDIATE, D_MODEL),
}


def _tiled_spec(cfg: dict[str, Any]) -> tuple[int, dict[str, int]]:
    """(number of selected layers, per-matrix C) from the tiled glu_transformer c-spec."""
    sites = cfg["decomposition"]["sites"]
    assert sites["kind"] == "glu_transformer", f"llama8b-only tool, got {sites['kind']!r}"
    selection = sites["layers"]
    match selection["kind"]:
        case "all":
            n_selected = N_LAYERS
        case "range":
            n_selected = selection["end"] - selection["start"]
        case "list":
            n_selected = len(selection["indices"])
        case unknown:
            raise ValueError(f"unknown layer selection {unknown!r}")
    return n_selected, sites["cs"]


def vu_params(cfg: dict[str, Any]) -> int:
    """Exact V/U leaf-param count: sum_site C*(d_in + d_out)."""
    n_selected, cs = _tiled_spec(cfg)
    per_layer = sum(c * sum(KIND_DIMS[f"{matrix}_proj"]) for matrix, c in cs.items())
    return n_selected * per_layer


def ci_fn_params(cfg: dict[str, Any]) -> int:
    """Exact CI-fn leaf-param count for the chunkwise transformer (blocks_per_chunk=1 →
    one chunk per decomposed layer; each chunk = in_proj[d_resid,d] + n_blocks CIBlocks +
    glued out-head[d, ΣC_layer]). Mirrors ci_fn._init_chunk_transformer shapes."""
    ci = cfg["decomposition"]["ci"]
    d, mlp, n_blocks = ci["d_model"], ci["ffn"]["hidden"], ci["n_blocks"]
    bpc = ci["blocks_per_chunk"]
    d_resid = D_MODEL  # _resolve_d_resid -> n_embd

    n_selected, cs = _tiled_spec(cfg)
    assert n_selected % bpc == 0
    n_chunks = n_selected // bpc
    c_per_chunk = bpc * sum(cs.values())  # tiled ⇒ homogeneous per chunk by construction

    in_proj = d_resid * d + d
    block = 4 * (d * d) + (d * mlp + mlp) + (mlp * d + d)  # wq,wk,wv,wo + w1,b1 + w2,b2
    out_head = d * c_per_chunk + c_per_chunk
    return n_chunks * (in_proj + n_blocks * block + out_head)


def main() -> None:
    cfg_path = Path(sys.argv[1])
    dp = int(sys.argv[sys.argv.index("--dp") + 1]) if "--dp" in sys.argv else 32
    fsdp = int(sys.argv[sys.argv.index("--fsdp") + 1]) if "--fsdp" in sys.argv else 8
    cfg = yaml.safe_load(cfg_path.read_text())
    batch = cfg["pd"]["batch_size"]
    seq_per_gpu = batch // dp  # data shards over the full mesh (replicate x dp = dp here)

    vu = vu_params(cfg)
    ci = ci_fn_params(cfg)
    trainable = vu + ci
    target = N_LAYERS * sum(d_in * d_out for d_in, d_out in KIND_DIMS.values())
    # embed + lm_head are NOT decomposed and are REPLICATED (not ÷fsdp) on the frozen target
    # (targets/glu_transformer.py) — full-resident per GPU. Llama-8B does not tie them.
    embed_lmhead = 2 * VOCAB * D_MODEL

    GB = 1024**3
    # ZeRO-1 ÷N: fp32 master + Adam m + Adam v = 12 B/param, sharded over the full mesh.
    opt = trainable * 12 / dp / GB  # EXACT (sharding.py: master+m+v ÷N)
    vu_bf16 = vu * 2 / fsdp / GB  # ÷fsdp resident compute weight (EXACT layout)
    ci_bf16 = ci * 2 / fsdp / GB  # ÷fsdp resident compute weight (EXACT layout)
    tgt_bf16 = target * 2 / fsdp / GB  # EXACT: layer weights ÷fsdp (targets/glu_transformer.py)
    embed_bf16 = embed_lmhead * 2 / GB  # EXACT: embed+lm_head REPLICATED (full-resident)
    # ONE forward's activations (per-layer remat => residual carry stack + logits), per GPU.
    resid_stack = N_LAYERS * seq_per_gpu * SEQ * D_MODEL * 2 / GB  # bf16 [L,b,t,d] carry
    logits = 2 * seq_per_gpu * SEQ * VOCAB * 4 / GB  # f32 clean+masked [b,t,V]
    acts = resid_stack + logits  # ASSUMED floor (perfect remat, 1 live forward)

    floor = opt + vu_bf16 + ci_bf16 + tgt_bf16 + embed_bf16 + acts
    print(f"config: {cfg_path.name}   dp={dp} fsdp={fsdp}  batch={batch} ({seq_per_gpu} seq/GPU)")
    print(f"trainable params: V/U={vu / 1e9:.2f}B + CI-fn={ci / 1e9:.2f}B = {trainable / 1e9:.2f}B")
    print(
        f"frozen target:    {target / 1e9:.2f}B (layers) + {embed_lmhead / 1e9:.2f}B (embed+lm_head)\n"
    )
    print("per-GPU theoretical-minimum peak (GB):")
    print(f"  optimizer ÷N (master+m+v, fp32)   {opt:6.2f}   EXACT")
    print(f"  V/U bf16 compute ÷fsdp            {vu_bf16:6.2f}   EXACT")
    print(f"  CI-fn bf16 compute ÷fsdp          {ci_bf16:6.2f}   EXACT")
    print(f"  frozen layers bf16 ÷fsdp          {tgt_bf16:6.2f}   EXACT")
    print(f"  frozen embed+lm_head bf16 (repl)  {embed_bf16:6.2f}   EXACT (replicated)")
    print(f"  activations (1 fwd: carry+logits) {acts:6.2f}   ASSUMED (perfect remat)")
    print(f"  {'-' * 44}")
    print(f"  FLOOR                             {floor:6.2f}")


if __name__ == "__main__":
    main()
