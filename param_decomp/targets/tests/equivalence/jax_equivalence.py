"""JAX equivalence check: the JAX single-pool PD loss terms vs the torch reference.

Loads the SAME fixtures behind the frozen `torch_reference.json` golden, builds the
Llama `DecomposedModel` with the identical (zeroed-attn) suffix weights, and computes each
loss term through the generic trainer's OWN helpers (`train.py`), feeding the FIXED
masks / sources / routing from the fixtures (no RNG). Compares to
`torch_reference.json` at fp32 tolerance.

Term wiring (all `param_decomp.core.train` + the `DecomposedModel` boundary):
  * imp   — `importance_minimality_terms(ci_upper, p, beta, eps)` (per-site dicts)
  * ppgd  — `masks_from_sources` + `run_masked` over every site; `kl_per_position`
            vs `model.clean_output` (the frozen path, SPEC S3).

The torch golden's `stoch` term drove partial per-chunk masked forwards — a capability
the masked forward no longer has (masks must cover every site) — so it is not compared.

Bit-identical is impossible across RNG/FP backends; we assert each term within
`RTOL`/`ATOL` of the torch value.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", False)

from param_decomp.core.adversary import split_source_channels  # noqa: E402
from param_decomp.core.components import component_stacks_from_sites  # noqa: E402
from param_decomp.core.losses import importance_minimality_terms  # noqa: E402
from param_decomp.core.masking import masks_from_sources  # noqa: E402
from param_decomp.targets.glu_transformer import (  # noqa: E402
    MLP_KINDS,
    FrozenAttn,
    GatedMLP,
    GLULayer,
    build_decomposed_lm,
    glu_site_specs,
    mlp_family_site_cs,
    site_name,
)
from param_decomp.targets.losses import kl_per_position  # noqa: E402
from param_decomp.targets.testing import run_clean, run_masked  # noqa: E402
from param_decomp.vendored_jax.llama import LlamaConfig  # noqa: E402

HERE = Path(__file__).resolve().parent
RTOL = 2e-4
ATOL = 1e-5


# fp32 throughout the harness so the cross-framework comparison is fp-tight (the torch
# reference is fp32). The production step runs bf16 compute; here we isolate the loss
# MATH from bf16 rounding (a bf16 forward agrees with torch only to ~1e-3).
FP = jnp.float32


def _zero_attn(d: int, _di: int) -> FrozenAttn:
    """Attn with zeroed projections (contributes 0); head dims are arbitrary since the
    output is 0. RoPE never affects a zero output."""
    n_head, n_kv_head, head_dim = 2, 1, d // 2
    qd = n_head * head_dim
    kvd = n_kv_head * head_dim
    z = lambda r, c: jnp.zeros((r, c), FP)  # noqa: E731
    return FrozenAttn(
        wq=z(qd, d), wk=z(kvd, d), wv=z(kvd, d), wo=z(d, qd),
        n_head=n_head, n_kv_head=n_kv_head, head_dim=head_dim, n_rep=n_head // n_kv_head,
        implementation="auto",
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
        GLULayer(
            ln1=a(f"ln1_{i}"),
            ln2=a(f"ln2_{i}"),
            attn=_zero_attn(d, di),
            mlp=GatedMLP(Wg=a(f"Wg_{i}"), Wu=a(f"Wu_{i}"), Wd=a(f"Wd_{i}")),
        )
        for i in range(n_layers)
    ]
    tail = [
        GLULayer(
            ln1=a(f"tail_ln1_{j}"),
            ln2=a(f"tail_ln2_{j}"),
            attn=_zero_attn(d, di),
            mlp=GatedMLP(Wg=a(f"tail_Wg_{j}"), Wu=a(f"tail_Wu_{j}"), Wd=a(f"tail_Wd_{j}")),
        )
        for j in range(n_tail)
    ]
    # inv_freq unused (attn zeroed); a dummy valid-shaped array.
    inv_freq = jnp.ones((d // 4,), jnp.float32)
    vu = component_stacks_from_sites(
        {
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
    model = build_decomposed_lm(
        embed=jnp.zeros((cfg.vocab_size, cfg.n_embd), jnp.float32),
        layers=decomp_layers + tail,
        norm=a("norm"),
        lm_head=a("lm_head"),
        inv_freq=inv_freq,
        cfg=cfg,
        sites=glu_site_specs(cfg, mlp_family_site_cs(0, n_layers - 1, C)),
    )
    return model, vu, n_layers


def compute_jax_terms(f: dict[str, np.ndarray]) -> dict[str, float]:
    """The retained JAX oracle-comparable loss terms on fixtures `f` (fp32). Shared by `main` and
    the pytest so there is one term-computation path."""
    model, vu, n_layers = _build(f)
    resid = jnp.asarray(f["resid"], dtype=FP)

    clean = jax.lax.stop_gradient(run_clean(model, resid))

    # fixtures key CI per kind as (B, T, L, C); the trainer keys per site.
    def per_site(prefix: str) -> dict[str, jnp.ndarray]:
        by_kind = {k: jnp.asarray(f[f"{prefix}_{k}"], dtype=FP) for k in MLP_KINDS}
        return {site_name(i, k): by_kind[k][:, :, i] for i in range(n_layers) for k in MLP_KINDS}

    ci_lower = per_site("ci_lower")
    ci_upper = per_site("ci_upper")

    # ---- imp ----
    # a' = B·T reproduces the old rolled `log2(1 + B·T·f_c)`, so `imp_lp + beta·freq`
    # equals the old `imp_lp + beta·entropy` the golden was generated against.
    n_positions = int(np.prod(next(iter(ci_upper.values())).shape[:-1]))
    imp_lp, imp_freq = importance_minimality_terms(
        ci_upper,
        jnp.asarray(float(f["_scalar_IMP_P"])),
        float(f["_scalar_IMP_EPS"]),
        reference_datapoint_count=n_positions,
    )
    imp = float(imp_lp + float(f["_scalar_IMP_BETA"]) * imp_freq)

    # ---- ppgd (FIXED sources) ----
    source = {
        site: split_source_channels(jnp.asarray(packed))
        for site, packed in per_site("ppgd_source").items()
    }
    masks, delta_masks = masks_from_sources(ci_lower, source)
    pred = run_masked(
        model,
        model.prepare_compute_weights(vu, None),
        resid,
        masks,
        delta_masks,
        None,
        True,
        remat=False,
    )
    ppgd = float(kl_per_position(pred, clean))

    return {"imp": imp, "ppgd": ppgd}


def main() -> None:
    f = dict(np.load(HERE / "fixtures.npz"))
    ref = json.loads((HERE / "torch_reference.json").read_text())
    jaxv = compute_jax_terms(f)
    print(f"{'term':6} {'jax':>16} {'torch':>16} {'rel_err':>12}  ok")
    all_ok = True
    for term in ("imp", "ppgd"):
        jv, tv = jaxv[term], ref[term]
        rel = abs(jv - tv) / (abs(tv) + 1e-30)
        ok = abs(jv - tv) <= ATOL + RTOL * abs(tv)
        all_ok = all_ok and ok
        print(f"{term:6} {jv:16.8e} {tv:16.8e} {rel:12.3e}  {'PASS' if ok else 'FAIL'}")
    assert all_ok, "JAX term(s) diverge from torch reference beyond tolerance"
    print("\nALL RETAINED TERMS NUMERICALLY EQUIVALENT (fp32) to the torch 2-pool reference.")


if __name__ == "__main__":
    main()
