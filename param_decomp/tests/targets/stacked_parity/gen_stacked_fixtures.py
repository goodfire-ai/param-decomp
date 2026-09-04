"""Pin the STACKED (pre-site-generality) Llama target's outputs as parity fixtures.

This script runs against the `feature/jax-single-pool-pd` code (the stacked `DecompVU`
with `(L, ., .)` arrays, `LayerRange`, `llama_decomposed_lm(cfg, layer_range, C)`); it
does NOT run against `feature/jax-site-generality` or later — the names it imports were
restructured away. Regenerate from a base-branch checkout, e.g.:

    <base-checkout>/.venv/bin/python \
        param_decomp/tests/targets/stacked_parity/gen_stacked_fixtures.py

`test_stacked_parity.py` then rebuilds the same model in the per-site representation
and must reproduce: clean logits BIT-IDENTICAL, masked logits / weight deltas /
site inputs to reassociation tolerance (SPEC D4, rel ~1e-5). This script once also
recorded a 2-step training trajectory; that part is deleted because its loss was the
removed L_p penalty, which only this old branch can still express — no comparable
recording can be made again (the committed npz still carries the dead `out::` arrays).

Everything the new representation cannot regenerate by re-running unchanged init code
is saved as arrays: the frozen suffix weights, the per-site V/U (the V/U init's RNG
derivation changed with the layout), the fixed masks/routes of the direct masked
calls. The CI fn and sources ARE re-initialised by the test (that code is unchanged)
and asserted leaf-identical against the copies saved here.
"""

from pathlib import Path

import equinox as eqx
import jax
import numpy as np
from jax import random

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", False)

from param_decomp.core.adversary import (  # noqa: E402
    init_persistent_sources,
)
from param_decomp.core.ci_fn import CIArch, init_ci_fn  # noqa: E402
from param_decomp.targets.llama31 import (  # noqa: E402
    KINDS,
    LayerRange,
    init_decomp_vu,
    llama_decomposed_lm,
    site_name,
)
from param_decomp.tests.targets.test_llama31 import _tiny_cfg, _tiny_target  # noqa: E402
from param_decomp.vendored_jax.llama import LlamaConfig  # noqa: E402

OUT = Path(__file__).resolve().parent / "stacked_fixtures.npz"

FIRST_LAYER, LAST_LAYER = 3, 5
C = 8
B, T = 2, 16
N_TRAIN_STEPS = 2
N_WARMUP = 2
CI_ARCH = CIArch(d_model=16, n_blocks=2, n_heads=2, ffn_hidden=32)


def _save_target_arrays(cfg: LlamaConfig, tgt, layer_range: LayerRange) -> dict[str, np.ndarray]:
    """Flatten the stacked-era `Target` to per-absolute-layer raw weight arrays."""
    arrays: dict[str, np.ndarray] = {}
    for layer_idx, frozen_layer in enumerate(tgt.decomp_layers):
        layer = layer_range.layers[layer_idx]
        attn = frozen_layer.attn
        for field, value in (
            ("ln1", frozen_layer.ln1),
            ("ln2", frozen_layer.ln2),
            ("wq", attn.wq),
            ("wk", attn.wk),
            ("wv", attn.wv),
            ("wo", attn.wo),
            ("Wg", frozen_layer.mlp.Wg),
            ("Wu", frozen_layer.mlp.Wu),
            ("Wd", frozen_layer.mlp.Wd),
        ):
            arrays[f"tgt::layers.{layer}.{field}"] = np.asarray(value)
    for tail_idx, blk in enumerate(tgt.tail):
        layer = layer_range.last + 1 + tail_idx
        for field, value in (
            ("ln1", blk.ln1),
            ("ln2", blk.ln2),
            ("wq", blk.attn.wq),
            ("wk", blk.attn.wk),
            ("wv", blk.attn.wv),
            ("wo", blk.attn.wo),
            ("Wg", blk.mlp.wg),
            ("Wu", blk.mlp.wu),
            ("Wd", blk.mlp.wd),
        ):
            arrays[f"tgt::layers.{layer}.{field}"] = np.asarray(value)
    arrays["tgt::norm"] = np.asarray(tgt.norm)
    arrays["tgt::lm_head"] = np.asarray(tgt.lm_head)
    return arrays


def main() -> None:
    cfg = _tiny_cfg()
    layer_range = LayerRange(FIRST_LAYER, LAST_LAYER)
    tgt = _tiny_target(cfg, layer_range, random.PRNGKey(0))
    model = llama_decomposed_lm(cfg, layer_range, C)
    vu = init_decomp_vu(cfg, C, layer_range.n_layers, random.PRNGKey(1))
    ci_fn = init_ci_fn(CI_ARCH, model.sites, random.PRNGKey(2))
    sources = init_persistent_sources(
        model.site_names,
        tuple(s.C for s in model.sites),
        (1, T),
        jax.numpy.float32,
        random.PRNGKey(3),
    )
    resid = random.normal(random.PRNGKey(4), (B, T, cfg.n_embd)) * 0.5

    arrays = _save_target_arrays(cfg, tgt, layer_range)
    for layer_idx, layer in enumerate(layer_range.layers):
        for kind in KINDS:
            V, U = vu.site(layer_idx, kind)
            arrays[f"vu::V::{site_name(layer, kind)}"] = np.asarray(V)
            arrays[f"vu::U::{site_name(layer, kind)}"] = np.asarray(U)
    for leaf_idx, leaf in enumerate(jax.tree.leaves(eqx.filter(ci_fn, eqx.is_array))):
        arrays[f"ci_leaf::{leaf_idx}"] = np.asarray(leaf)
    for name, source in sources.items():
        arrays[f"src::{name}"] = np.asarray(source)
    arrays["resid"] = np.asarray(resid)

    # ── direct forward pins (fp32, eager) ──
    arrays["out::clean"] = np.asarray(model.clean_output(tgt, resid))
    for name, site_input in model.site_inputs(tgt, resid).items():
        arrays[f"out::site_input::{name}"] = np.asarray(site_input)
    for name, delta in model.weight_deltas(tgt, vu).items():
        arrays[f"out::wd::{name}"] = np.asarray(delta)

    mask_rng = np.random.default_rng(99)
    masks = {
        name: mask_rng.uniform(0.0, 1.0, (B, T, C)).astype(np.float32) for name in model.site_names
    }
    delta_masks = {
        name: mask_rng.uniform(0.0, 1.0, (B, T)).astype(np.float32) for name in model.site_names
    }
    chunk0 = model.site_names[:3]
    routes0 = {name: mask_rng.random((B, T)) < 0.6 for name in chunk0}
    for name in model.site_names:
        arrays[f"mask::{name}"] = masks[name]
        arrays[f"delta_mask::{name}"] = delta_masks[name]
    for name in chunk0:
        arrays[f"route0::{name}"] = routes0[name]

    jm = {k: jax.numpy.asarray(v) for k, v in masks.items()}
    jdm = {k: jax.numpy.asarray(v) for k, v in delta_masks.items()}
    arrays["out::masked_all"] = np.asarray(
        model.masked_output(tgt, vu, resid, jm, jdm, None, model.site_names, True)
    )
    arrays["out::masked_subset"] = np.asarray(
        model.masked_output(
            tgt,
            vu,
            resid,
            {s: jm[s] for s in chunk0},
            {s: jdm[s] for s in chunk0},
            {s: jax.numpy.asarray(routes0[s]) for s in chunk0},
            chunk0,
            True,
        )
    )

    scalars = dict(
        FIRST_LAYER=FIRST_LAYER,
        LAST_LAYER=LAST_LAYER,
        C=C,
        B=B,
        T=T,
        N_TRAIN_STEPS=N_TRAIN_STEPS,
        N_WARMUP=N_WARMUP,
    )
    for name, value in scalars.items():
        arrays[f"_scalar_{name}"] = np.array(value)

    np.savez(OUT, **arrays)  # pyright: ignore[reportArgumentType] (numpy savez **kwds stub is strict)
    print(f"wrote {OUT} ({len(arrays)} arrays)")


if __name__ == "__main__":
    main()
