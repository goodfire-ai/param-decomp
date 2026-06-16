"""JAX equivalence check: the JAX single-pool PD loss terms vs the torch reference.

Run in the JAX env AFTER `torch_reference.py`. Loads the SAME fixtures, builds the
Llama `DecomposedLM` with the identical (zeroed-attn) suffix weights, and computes each
loss term through the generic trainer's OWN helpers (`train.py`), feeding the FIXED
masks / sources / routing from the fixtures (no RNG). Compares to
`torch_reference.json` at fp32 tolerance.

Term wiring (all `jax_single_pool.train` + the `DecomposedLM` boundary):
  * faith — `faithfulness_loss(lm.weight_deltas(frozen, vu))`
  * imp   — `importance_minimality_terms(ci_upper, p, beta, eps)` (per-site dicts)
  * stoch — per chunk: `mask = ci+(1-ci)*u`, fixed delta mask, fixed route over the
            chunk's 3 sites; `lm.masked_logits(..., live=chunk)`; `kl_per_position`
            vs `lm.clean_logits` (the frozen path, SPEC S3). Mean over chunks.
  * ppgd  — `source_masks` + `lm.masked_logits(..., live=all)`.

Bit-identical is impossible across RNG/FP backends; we assert each term within
`RTOL`/`ATOL` of the torch value.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", False)

from vendored_jax.llama import LlamaConfig  # noqa: E402

from jax_single_pool.adversary import source_masks  # noqa: E402
from jax_single_pool.llama8b import (  # noqa: E402
    MLP_KINDS,
    DecompVU,
    FrozenAttn,
    SuffixLayer,
    Target,
    llama_decomposed_lm,
    llama_site_specs,
    mlp_family_site_cs,
    site_name,
)
from jax_single_pool.losses import (  # noqa: E402
    faithfulness_loss,
    importance_minimality_terms,
    kl_per_position,
)

HERE = Path(__file__).resolve().parent
RTOL = 2e-4
ATOL = 1e-5


# fp32 throughout the harness so the cross-framework comparison is fp-tight (the torch
# reference is fp32). The production step runs bf16 compute; here we isolate the loss
# MATH from bf16 rounding (a bf16 forward agrees with torch only to ~1e-3).
FP = jnp.float32


def _zero_attn(d: int, di: int) -> FrozenAttn:
    """Attn with zeroed projections (contributes 0); head dims are arbitrary since the
    output is 0. RoPE never affects a zero output."""
    n_head, n_kv_head, head_dim = 2, 1, d // 2
    qd = n_head * head_dim
    kvd = n_kv_head * head_dim
    z = lambda r, c: jnp.zeros((r, c), FP)  # noqa: E731
    return FrozenAttn(
        wq=z(qd, d), wk=z(kvd, d), wv=z(kvd, d), wo=z(d, qd),
        n_head=n_head, n_kv_head=n_kv_head, head_dim=head_dim, n_rep=n_head // n_kv_head,
    )  # fmt: skip


def _build(f: dict[str, np.ndarray]):
    a = lambda key: jnp.asarray(f[key], dtype=FP)  # noqa: E731
    d = int(f["_scalar_N_EMBD"])
    di = int(f["_scalar_N_INTERMEDIATE"])
    n_layers = int(f["_scalar_N_DECOMP_LAYERS"])
    n_tail = int(f["_scalar_N_TAIL"])
    eps = float(f["_scalar_EPS"])
    C = int(f[f"Vg_0"].shape[-1])  # noqa: F541

    decomp_layers = [
        SuffixLayer(
            ln1=a(f"ln1_{i}"),
            ln2=a(f"ln2_{i}"),
            attn=_zero_attn(d, di),
            Wg=a(f"Wg_{i}"),
            Wu=a(f"Wu_{i}"),
            Wd=a(f"Wd_{i}"),
        )  # fmt: skip
        for i in range(n_layers)
    ]
    tail = [
        SuffixLayer(
            ln1=a(f"tail_ln1_{j}"),
            ln2=a(f"tail_ln2_{j}"),
            attn=_zero_attn(d, di),
            Wg=a(f"tail_Wg_{j}"),
            Wu=a(f"tail_Wu_{j}"),
            Wd=a(f"tail_Wd_{j}"),
        )  # fmt: skip
        for j in range(n_tail)
    ]
    # inv_freq unused (attn zeroed); a dummy valid-shaped array.
    inv_freq = jnp.ones((d // 4,), jnp.float32)
    tgt = Target(
        layers=decomp_layers + tail, norm=a("norm"), lm_head=a("lm_head"),
        inv_freq=inv_freq, eps=eps,
    )  # fmt: skip
    vu = DecompVU(
        vu={
            site_name(i, kind): (a(f"V{kind[0]}_{i}"), a(f"U{kind[0]}_{i}"))
            for i in range(n_layers)
            for kind in MLP_KINDS
        }
    )
    cfg = LlamaConfig(
        vocab_size=int(f["lm_head"].shape[0]),
        n_layer=n_layers + n_tail,
        n_head=2,
        n_kv_head=1,
        n_embd=d,
        n_intermediate=di,
        rope_theta=10000.0,
        rms_norm_eps=eps,
        max_position_embeddings=512,
        rope_factor=8.0,
        rope_low_freq_factor=1.0,
        rope_high_freq_factor=4.0,
        rope_original_max_position_embeddings=128,
    )
    lm = llama_decomposed_lm(cfg, llama_site_specs(cfg, mlp_family_site_cs(0, n_layers - 1, C)))
    return lm, tgt, vu, n_layers


def compute_jax_terms(f: dict[str, np.ndarray]) -> dict[str, float]:
    """The four JAX loss-term values on the fixtures `f` (fp32). Shared by `main` and
    the pytest so there is one term-computation path."""
    lm, tgt, vu, n_layers = _build(f)
    resid = jnp.asarray(f["resid"], dtype=FP)

    clean = jax.lax.stop_gradient(lm.clean_logits(tgt, resid))

    # fixtures key CI per kind as (B, T, L, C); the trainer keys per site.
    def per_site(prefix: str) -> dict[str, jnp.ndarray]:
        by_kind = {k: jnp.asarray(f[f"{prefix}_{k}"], dtype=FP) for k in MLP_KINDS}
        return {site_name(i, k): by_kind[k][:, :, i] for i in range(n_layers) for k in MLP_KINDS}

    ci_lower = per_site("ci_lower")
    ci_upper = per_site("ci_upper")

    # ---- faith ----
    faith = float(faithfulness_loss(lm.weight_deltas(tgt, vu)))

    # ---- imp ----
    imp_lp, imp_entropy = importance_minimality_terms(
        ci_upper, jnp.asarray(float(f["_scalar_IMP_P"])), float(f["_scalar_IMP_EPS"])
    )
    imp = float(imp_lp + float(f["_scalar_IMP_BETA"]) * imp_entropy)

    # ---- stoch (per-chunk, FIXED masks) ----
    stoch_u = per_site("stoch_u")
    stoch_delta = per_site("stoch_delta")
    stoch_total = 0.0
    for i in range(n_layers):
        chunk = tuple(site_name(i, k) for k in MLP_KINDS)
        masks = {s: ci_lower[s] + (1.0 - ci_lower[s]) * stoch_u[s] for s in chunk}
        delta_masks = {s: stoch_delta[s] for s in chunk}
        routes = {site_name(i, k): jnp.asarray(f[f"route_chunk{i}_{k}"]) for k in MLP_KINDS}
        pred = lm.masked_logits(tgt, vu, resid, masks, delta_masks, routes, chunk)
        stoch_total += float(kl_per_position(pred, clean))
    stoch = stoch_total / n_layers

    # ---- ppgd (FIXED sources) ----
    source = per_site("ppgd_source")  # {site: (1, T, C+1)}
    masks, delta_masks = source_masks(ci_lower, source, lm.site_names, "clamp")
    pred = lm.masked_logits(tgt, vu, resid, masks, delta_masks, None, lm.site_names)
    ppgd = float(kl_per_position(pred, clean))

    return {"faith": faith, "imp": imp, "stoch": stoch, "ppgd": ppgd}


def main() -> None:
    f = dict(np.load(HERE / "fixtures.npz"))
    ref = json.loads((HERE / "torch_reference.json").read_text())
    jaxv = compute_jax_terms(f)
    print(f"{'term':6} {'jax':>16} {'torch':>16} {'rel_err':>12}  ok")
    all_ok = True
    for term in ("faith", "imp", "stoch", "ppgd"):
        jv, tv = jaxv[term], ref[term]
        rel = abs(jv - tv) / (abs(tv) + 1e-30)
        ok = abs(jv - tv) <= ATOL + RTOL * abs(tv)
        all_ok = all_ok and ok
        print(f"{term:6} {jv:16.8e} {tv:16.8e} {rel:12.3e}  {'PASS' if ok else 'FAIL'}")
    assert all_ok, "JAX term(s) diverge from torch reference beyond tolerance"
    print("\nALL TERMS NUMERICALLY EQUIVALENT (fp32) to the torch 2-pool reference.")


if __name__ == "__main__":
    main()
