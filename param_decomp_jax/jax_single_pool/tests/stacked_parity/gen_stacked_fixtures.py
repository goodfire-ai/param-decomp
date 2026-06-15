"""Pin the STACKED (pre-site-generality) Llama target's outputs as parity fixtures.

This script runs against the `feature/jax-single-pool-pd` code (the stacked `DecompVU`
with `(L, ., .)` arrays, `LayerRange`, `llama_decomposed_lm(cfg, layer_range, C)`); it
does NOT run against `feature/jax-site-generality` or later — the names it imports were
restructured away. Regenerate from a base-branch checkout, e.g.:

    <base-checkout>/param_decomp_jax/.venv/bin/python \
        jax_single_pool/tests/stacked_parity/gen_stacked_fixtures.py

`test_stacked_parity.py` then rebuilds the same model in the per-site representation
and must reproduce: clean logits BIT-IDENTICAL, masked logits / weight deltas /
site inputs and a 2-step training trajectory to reassociation tolerance (SPEC D4,
rel ~1e-5).

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
import optax
from jax import random

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", False)

from vendored_jax.llama import LlamaConfig  # noqa: E402

from jax_single_pool.adversary import (  # noqa: E402
    init_persistent_sources,
    init_sources_adam_state,
)
from jax_single_pool.ci_fn import CIArch, init_ci_fn  # noqa: E402
from jax_single_pool.llama8b import (  # noqa: E402
    KINDS,
    LayerRange,
    init_decomp_vu,
    llama_decomposed_lm,
    site_name,
)
from jax_single_pool.recon import subset_chunk_plan  # noqa: E402
from jax_single_pool.tests.test_llama8b import _tiny_cfg, _tiny_target  # noqa: E402
from jax_single_pool.train import TrainState, make_train_step  # noqa: E402
from param_decomp_config.losses import (  # noqa: E402
    AdamPGDConfig,
    ImportanceMinimalityLossConfig,
    PersistentPGDReconLossConfig,
    SCScope,
)
from param_decomp_config.schedule import ScheduleConfig  # noqa: E402

OUT = Path(__file__).resolve().parent / "stacked_fixtures.npz"

FIRST_LAYER, LAST_LAYER = 3, 5
C = 8
B, T = 2, 16
N_TRAIN_STEPS = 2
N_WARMUP = 2
CI_ARCH = CIArch(d_model=16, n_blocks=2, n_heads=2, mlp_hidden=32)
STABLE_METRIC_KEYS = (
    "total", "faith", "imp", "stoch", "ppgd", "p_imp", "src_lr",
    "grad_norms/summary/components", "grad_norms/summary/ci_fns", "grad_norms/summary/total",
)  # fmt: skip


def _save_target_arrays(cfg: LlamaConfig, tgt, layer_range: LayerRange) -> dict[str, np.ndarray]:
    """Flatten the stacked-era `Target` to per-absolute-layer raw weight arrays."""
    arrays: dict[str, np.ndarray] = {}
    for layer_idx, frozen_layer in enumerate(tgt.decomp_layers):
        layer = layer_range.layers[layer_idx]
        attn = frozen_layer.attn
        for field, value in (
            ("ln1", frozen_layer.ln1), ("ln2", frozen_layer.ln2),
            ("wq", attn.wq), ("wk", attn.wk), ("wv", attn.wv), ("wo", attn.wo),
            ("Wg", frozen_layer.Wg), ("Wu", frozen_layer.Wu), ("Wd", frozen_layer.Wd),
        ):  # fmt: skip
            arrays[f"tgt::layers.{layer}.{field}"] = np.asarray(value)
    for tail_idx, blk in enumerate(tgt.tail):
        layer = layer_range.last + 1 + tail_idx
        for field, value in (
            ("ln1", blk.ln1), ("ln2", blk.ln2),
            ("wq", blk.attn.wq), ("wk", blk.attn.wk), ("wv", blk.attn.wv), ("wo", blk.attn.wo),
            ("Wg", blk.mlp.wg), ("Wu", blk.mlp.wu), ("Wd", blk.mlp.wd),
        ):  # fmt: skip
            arrays[f"tgt::layers.{layer}.{field}"] = np.asarray(value)
    arrays["tgt::norm"] = np.asarray(tgt.norm)
    arrays["tgt::lm_head"] = np.asarray(tgt.lm_head)
    return arrays


def main() -> None:
    cfg = _tiny_cfg()
    layer_range = LayerRange(FIRST_LAYER, LAST_LAYER)
    tgt = _tiny_target(cfg, layer_range, random.PRNGKey(0))
    lm = llama_decomposed_lm(cfg, layer_range, C)
    vu = init_decomp_vu(cfg, C, layer_range.n_layers, random.PRNGKey(1))
    ci_fn = init_ci_fn(CI_ARCH, lm.sites, random.PRNGKey(2))
    sources = init_persistent_sources(
        lm.site_names, tuple(s.C for s in lm.sites), T, random.PRNGKey(3)
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
    arrays["out::clean"] = np.asarray(lm.clean_logits(tgt, resid))
    for name, site_input in lm.site_inputs(tgt, resid).items():
        arrays[f"out::site_input::{name}"] = np.asarray(site_input)
    for name, delta in lm.weight_deltas(tgt, vu).items():
        arrays[f"out::wd::{name}"] = np.asarray(delta)

    mask_rng = np.random.default_rng(99)
    masks = {
        name: mask_rng.uniform(0.0, 1.0, (B, T, C)).astype(np.float32) for name in lm.site_names
    }
    delta_masks = {
        name: mask_rng.uniform(0.0, 1.0, (B, T)).astype(np.float32) for name in lm.site_names
    }
    chunk0 = lm.site_names[:3]
    routes0 = {name: mask_rng.random((B, T)) < 0.6 for name in chunk0}
    for name in lm.site_names:
        arrays[f"mask::{name}"] = masks[name]
        arrays[f"delta_mask::{name}"] = delta_masks[name]
    for name in chunk0:
        arrays[f"route0::{name}"] = routes0[name]

    jm = {k: jax.numpy.asarray(v) for k, v in masks.items()}
    jdm = {k: jax.numpy.asarray(v) for k, v in delta_masks.items()}
    arrays["out::masked_all"] = np.asarray(
        lm.masked_logits(tgt, vu, resid, jm, jdm, None, lm.site_names)
    )
    arrays["out::masked_subset"] = np.asarray(
        lm.masked_logits(
            tgt,
            vu,
            resid,
            {s: jm[s] for s in chunk0},
            {s: jdm[s] for s in chunk0},
            {s: jax.numpy.asarray(routes0[s]) for s in chunk0},
            chunk0,
        )  # fmt: skip
    )

    # ── 2-step training trajectory ──
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)
    state = TrainState(
        components=vu, ci_fn=ci_fn,
        components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
        ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
        sources=sources, sources_adam_state=init_sources_adam_state(sources),
        step=jax.numpy.zeros((), jax.numpy.int32),
    )  # fmt: skip
    step_fn = make_train_step(
        lm=lm,
        faith_coeff=1e5,
        stoch_coeff=0.5,
        imp_min=ImportanceMinimalityLossConfig(
            coeff=5e-6,
            pnorm=2.0,
            beta=0.2,
            p_anneal_start_frac=0.0,
            p_anneal_final_p=0.4,
            p_anneal_end_frac=1.0,
        ),
        adversary=PersistentPGDReconLossConfig(
            coeff=0.5,
            scope=SCScope(),
            optimizer=AdamPGDConfig(
                beta1=0.5,
                beta2=0.99,
                lr_schedule=ScheduleConfig(start_val=0.01, warmup_pct=0.025),
            ),
            n_warmup_steps=N_WARMUP,
        ),
        components_optimizer=opt_vu,
        ci_fn_optimizer=opt_ci,
        total_steps=100,
        recon_plan=subset_chunk_plan(lm.site_names, 3, 1),
        remat_recon_forwards=False,
        mesh=None,
    )
    run_key = random.PRNGKey(7)
    for step_idx in range(N_TRAIN_STEPS):
        state, metrics = step_fn(state, tgt, resid, random.fold_in(run_key, step_idx))
        for key in STABLE_METRIC_KEYS:
            arrays[f"out::step{step_idx}::{key}"] = np.asarray(metrics[key])

    for layer_idx, layer in enumerate(layer_range.layers):
        for kind in KINDS:
            V, U = state.components.site(layer_idx, kind)
            arrays[f"out::final_V::{site_name(layer, kind)}"] = np.asarray(V)
            arrays[f"out::final_U::{site_name(layer, kind)}"] = np.asarray(U)
    for name, source in state.sources.items():
        arrays[f"out::final_src::{name}"] = np.asarray(source)

    scalars = dict(
        FIRST_LAYER=FIRST_LAYER, LAST_LAYER=LAST_LAYER, C=C, B=B, T=T,
        N_TRAIN_STEPS=N_TRAIN_STEPS, N_WARMUP=N_WARMUP,
    )  # fmt: skip
    for name, value in scalars.items():
        arrays[f"_scalar_{name}"] = np.array(value)

    np.savez(OUT, **arrays)  # pyright: ignore[reportArgumentType] (numpy savez **kwds stub is strict)
    print(f"wrote {OUT} ({len(arrays)} arrays)")


if __name__ == "__main__":
    main()
