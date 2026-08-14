"""SPIKE (not a collected test): derisk the compositional prefix/suffix split.

Claim under test: a depth-suffix of a GLU transformer is expressible as a
`DecomposedModel` whose batch is the prefix's residual activations, with the prefix run
once in the data path — no engine change. Run directly:

    uv run python param_decomp/targets/tests/spike_prefix_suffix.py

Known-accepted spike shortcuts (each becomes real work if we build this):
- jaxtyping disabled process-wide: the GLU forwards annotate `inputs: Int[Array, "b t"]`;
  the real build widens the target's input edge per the #828 opaque-batch contract.
- `SuffixGLU.embed_tokens` is an identity override: the real build gives the suffix
  target an explicit resid input arm instead of a subclass trick.
"""

import os

os.environ["JAXTYPING_DISABLE"] = "1"

import dataclasses
from typing import override

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array

from param_decomp.core.components import SiteC, init_component_stacks
from param_decomp.core.configs import (
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    StochasticReconLossConfig,
)
from param_decomp.core.model import MaterializedMasking, prepare_compute_weights
from param_decomp.core.objective import build_objective
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.core.train import (
    Decomposition,
    TrainingItem,
    TrainState,
    make_train_step,
)
from param_decomp.targets.glu_transformer import (
    KIND_ORDER,
    GLUDecomposedModel,
    glu_site_specs,
    site_name,
)
from param_decomp.targets.testing import (
    tiny_glu_cfg,
    tiny_glu_chunkwise_ci_fn,
    tiny_glu_decomposed_lm,
)

SPLIT = 5  # blocks [0..4] = prefix, [5..7] = suffix; decomposed sites live in block 6
BATCH, SEQ = 4, 16


class SuffixGLU(GLUDecomposedModel):
    """The depth-suffix target: its 'embedding' is the identity on resid activations."""

    @override
    def embed_tokens(self, tokens: Array) -> Array:  # spike: real build widens the input edge
        return tokens


def main() -> None:
    cfg = tiny_glu_cfg()
    n_suffix = cfg.n_layer - SPLIT
    suffix_cfg = dataclasses.replace(cfg, n_layer=n_suffix)

    # Full model decomposes original block 6; the suffix names the same physical sites
    # in its own coordinates (block 6 - SPLIT = 1).
    full_sites = glu_site_specs(cfg, tuple(SiteC(site_name(6, k), 2) for k in KIND_ORDER))
    suffix_sites = glu_site_specs(
        suffix_cfg, tuple(SiteC(site_name(6 - SPLIT, k), 2) for k in KIND_ORDER)
    )
    full = tiny_glu_decomposed_lm(cfg, full_sites, jax.random.PRNGKey(0))
    suffix = SuffixGLU(
        embed=full.embed,  # unused: embed_tokens is identity
        stacked=jax.tree.map(lambda a: a[SPLIT:], full.stacked),
        n_layer=n_suffix,
        norm=full.norm,
        lm_head=full.lm_head,
        inv_freq=full.inv_freq,
        sites=suffix_sites,
        has_position_axis=True,
        eps=full.eps,
    )

    tokens = jax.random.randint(jax.random.PRNGKey(1), (BATCH, SEQ), 0, cfg.vocab_size)

    # ── the prefix as a data mapper: one full-model forward capturing resid.SPLIT ──
    resid = full.clean_forward(tokens, frozenset({f"resid.{SPLIT}"})).captures[f"resid.{SPLIT}"]

    # ── 1. clean equivalence ──
    full_logits = full.clean_forward(tokens).output
    suffix_logits = suffix.clean_forward(resid).output
    assert jnp.array_equal(full_logits, suffix_logits), (
        f"clean logits diverge: max |Δ| = {jnp.max(jnp.abs(full_logits - suffix_logits))}"
    )
    print(f"1. clean equivalence: BIT-EXACT (max|Δ|=0, shape {full_logits.shape})")

    # ── 2. masked equivalence on the shared physical sites ──
    vu_full = init_component_stacks(full_sites, jax.random.PRNGKey(2))
    vu_suffix = init_component_stacks(suffix_sites, jax.random.PRNGKey(2))
    for (fa, fb), (sa, sb) in zip(vu_full.stacks.values(), vu_suffix.stacks.values(), strict=True):
        assert jnp.array_equal(fa, sa) and jnp.array_equal(fb, sb), "vu init mismatch"

    mask_key = jax.random.PRNGKey(3)
    masks_full = {
        s.name: jax.random.uniform(jax.random.fold_in(mask_key, i), (BATCH, SEQ, s.C))
        for i, s in enumerate(full_sites)
    }
    masks_suffix = {
        s.name: masks_full[f.name] for s, f in zip(suffix_sites, full_sites, strict=True)
    }
    out_full = full.masked_forward(
        prepare_compute_weights(full, vu_full),
        tokens,
        masking=MaterializedMasking(
            component_masks=masks_full, weight_delta_masks=None, routes=None
        ),
        remat=False,
    ).output
    out_suffix = suffix.masked_forward(
        prepare_compute_weights(suffix, vu_suffix),
        resid,
        masking=MaterializedMasking(
            component_masks=masks_suffix, weight_delta_masks=None, routes=None
        ),
        remat=False,
    ).output
    max_delta = float(jnp.max(jnp.abs(out_full - out_suffix)))
    # Two distinct jit graphs fuse differently; fp32 ulp-scale reassociation is the
    # equality floor between them. Anything beyond that would be a semantic gap.
    assert jnp.allclose(out_full, out_suffix, atol=1e-6, rtol=1e-6), max_delta
    print(f"2. masked equivalence: to fp32 reassociation (max|Δ|={max_delta:.2e})")

    # ── 3. engine smoke: train the suffix on resid batches (the fixed-pool recipe) ──
    ci_fn = tiny_glu_chunkwise_ci_fn(suffix, jax.random.PRNGKey(4), n_blocks=1)
    objective = build_objective(
        (
            FaithfulnessLossConfig(coeff=1.0),
            ImportanceMinimalityLossConfig(coeff=3e-3, pnorm=ScheduleConfig.constant(1.0)),
            StochasticReconLossConfig(coeff=1.0),
        ),
        suffix.site_names,
    )
    opt_vu = optax.adamw(1e-3, weight_decay=0.0)
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)
    state = TrainState(
        decomposition=Decomposition(components=vu_suffix, ci_fn=ci_fn),
        training=TrainingItem(
            components_opt_state=opt_vu.init(eqx.filter(vu_suffix, eqx.is_array)),
            ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
            adversaries={},
            step=jnp.zeros((), jnp.int32),
        ),
    )
    step = make_train_step(
        model_static=suffix,
        losses=objective,
        components_optimizer=opt_vu,
        ci_fn_optimizer=opt_ci,
        total_steps=10,
        remat_recon_forwards=False,
        remat_ci_fn=False,
        ci_capture_keys=ci_fn.capture_keys,
    )
    # The fixed-pool recipe: the prefix ran ONCE above; every step reuses its output.
    for i in range(3):
        state, metrics = step(suffix, state, resid, jax.random.PRNGKey(100 + i))
        assert all(bool(jnp.isfinite(jnp.asarray(v)).all()) for v in metrics.values()), i
        print(
            f"3. engine step {i}: total={float(metrics['total']):.6f} "
            f"recon={float(metrics['loss/StochasticReconLoss']):.6f} "
            f"faith={float(metrics['faith']):.6f}"
        )
    print("SPIKE PASS: suffix target trains on prefix-mapped resid batches, zero engine changes")


if __name__ == "__main__":
    main()
