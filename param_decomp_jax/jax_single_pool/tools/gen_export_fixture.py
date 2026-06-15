"""JAX side of the export round-trip verification (run in THIS repo's venv).

Builds a tiny `CIFn` + `DecompVU` for three shapes — single-layer (L18-like, 3 MLP
sites), two-layer (6 sites, exercising the multi-layer site permutation), and a mixed
attention+MLP layer with heterogeneous per-site C (q/k/v/o + gate, exercising the
attention-site keys and the per-site-C permutation blocks) — exports them through
the REAL `export.py` mapping functions, and records fixture inputs plus the JAX-side
outputs: per-site component forwards `((x@V)*m)@U` and the full CI-fn lower/upper.

`verify_export_torch.py` (torch venv) then rebuilds the torch modules from the
safetensors and must reproduce these outputs — the proof that the key names, layouts,
and the site-order permutation are right.

    # regenerate (JAX venv):
    python -m jax_single_pool.tools.gen_export_fixture
    # verify (torch param-decomp venv):
    python jax_single_pool/tools/verify_export_torch.py
"""

from pathlib import Path

import equinox as eqx
import jax
import numpy as np
from jax import random
from safetensors.numpy import save_file
from vendored_jax.llama import LlamaConfig

from jax_single_pool.ci_fn import CIArch, CIFn, init_ci_fn
from jax_single_pool.export import (
    ci_fn_state,
    components_state,
    frozen_target_keys,
)
from jax_single_pool.llama8b import (
    canonical_site_cs,
    init_decomp_vu,
    llama_site_specs,
    mlp_family_site_cs,
    site_name,
)
from jax_single_pool.lm import SiteC

OUT_DIR = Path(__file__).resolve().parent / "export_fixtures"

B, T = 2, 8
N_EMBD, N_INTERMEDIATE, VOCAB = 16, 32, 48
C = 6
ARCH = CIArch(d_model=16, n_blocks=2, n_heads=2, mlp_hidden=24)

CASES: dict[str, tuple[SiteC, ...]] = {
    "l18": mlp_family_site_cs(18, 18, C),
    "l20_21": mlp_family_site_cs(20, 21, C),
    "l18_attn": canonical_site_cs(
        (
            SiteC(site_name(18, "q"), 4),
            SiteC(site_name(18, "k"), 6),
            SiteC(site_name(18, "v"), 6),
            SiteC(site_name(18, "o"), 8),
            SiteC(site_name(18, "gate"), 4),
        )
    ),
}


def _tiny_llama_cfg(n_layer: int) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=VOCAB,
        n_layer=n_layer,
        n_head=2,
        n_kv_head=1,
        n_embd=N_EMBD,
        n_intermediate=N_INTERMEDIATE,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
        max_position_embeddings=512,
    )


def _randomize_biases(ci_fn: CIFn, key: jax.Array) -> CIFn:
    """Zero-init biases would hide bias-mapping bugs; out_b spread wide so the squashed
    CI covers all three leaky-hard regimes (<0, (0,1), >1)."""

    def bias_leaves(m: CIFn) -> tuple[jax.Array, ...]:
        leaves: list[jax.Array] = [m.in_proj_b, m.out_b]
        for block in m.blocks:
            leaves += [block.b1, block.b2]
        return tuple(leaves)

    keys = iter(random.split(key, len(bias_leaves(ci_fn))))
    new_values = tuple(
        random.uniform(next(keys), leaf.shape, leaf.dtype, -1.2, 1.2) for leaf in bias_leaves(ci_fn)
    )
    return eqx.tree_at(bias_leaves, ci_fn, new_values)


def _gen_case(case: str, site_cs: tuple[SiteC, ...], key: jax.Array) -> None:
    n_layer = max(int(sc.name.split(".")[1]) for sc in site_cs) + 1
    cfg = _tiny_llama_cfg(n_layer)
    sites = llama_site_specs(cfg, site_cs)
    vu_key, ci_key, bias_key, data_key = random.split(key, 4)

    vu = init_decomp_vu(sites, vu_key)
    ci_fn = _randomize_biases(init_ci_fn(ARCH, sites, ci_key), bias_key)

    arrays: dict[str, np.ndarray] = {}
    site_inputs: dict[str, jax.Array] = {}
    for spec in sites:
        x_key, m_key = random.split(random.fold_in(data_key, len(site_inputs)))
        x = random.normal(x_key, (B, T, spec.d_in))
        m = random.uniform(m_key, (B, T, spec.C))
        site_inputs[spec.name] = x
        V, U = vu.site(spec.name)
        arrays[f"x::{spec.name}"] = np.asarray(x)
        arrays[f"mask::{spec.name}"] = np.asarray(m)
        arrays[f"component_out::{spec.name}"] = np.asarray(((x @ V) * m) @ U)

    ci = ci_fn(site_inputs)
    for name in ci_fn.site_names:
        arrays[f"ci_lower::{name}"] = np.asarray(ci.lower[name])
        arrays[f"ci_upper::{name}"] = np.asarray(ci.upper[name])
    lower_all = np.concatenate([arrays[f"ci_lower::{n}"].ravel() for n in ci_fn.site_names])
    interior_frac = float(((lower_all > 0) & (lower_all < 1)).mean())
    assert interior_frac > 0.1, "CI fixture saturated — all-0/1 lower-leaky values prove nothing"

    arrays["_site_names"] = np.array([s.name for s in sites])
    arrays["_d_in"] = np.array([s.d_in for s in sites])
    arrays["_d_out"] = np.array([s.d_out for s in sites])
    arrays["_C"] = np.array([s.C for s in sites])
    arrays["_arch"] = np.array([ARCH.d_model, ARCH.n_blocks, ARCH.n_heads, ARCH.mlp_hidden])
    arrays["_dims"] = np.array([B, T, n_layer, N_EMBD, N_INTERMEDIATE, VOCAB])
    decomposed_sites = frozenset(spec.name for spec in sites)
    arrays["_frozen_keys"] = np.array(sorted(frozen_target_keys(n_layer, decomposed_sites)))

    tensors = components_state(vu, sites) | ci_fn_state(ci_fn, sites)
    OUT_DIR.mkdir(exist_ok=True)
    save_file(tensors, str(OUT_DIR / f"{case}.safetensors"))
    np.savez(OUT_DIR / f"{case}.npz", **arrays)  # pyright: ignore[reportArgumentType] (numpy savez **kwds stub is strict)
    print(f"wrote {case}: {len(tensors)} exported tensors, {len(arrays)} fixture arrays")


def main() -> None:
    jax.config.update("jax_platforms", "cpu")
    jax.config.update("jax_enable_x64", False)
    for case_idx, (case, site_cs) in enumerate(CASES.items()):
        _gen_case(case, site_cs, random.PRNGKey(case_idx))


if __name__ == "__main__":
    main()
