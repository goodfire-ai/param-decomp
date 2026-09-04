"""Every diagnostic eval tier runs against an OWNER-placed run.

Regression seam: the tiers used to call the unplaced `prepare_compute_weights` (and the
raw persistence-layout CI evaluate) while threading real `PlacementRules` into their
forwards — under the Explicit mesh, owner-layout masters reaching per-site slicing die
at trace. Each tier must take the placed prepare + materialize lifecycle the trainer
uses (`recon_eval` was the reference dispatch).

Second regression seam (the reno 1375752 eval OOM): any eval-tier value built WITHOUT
the batch's sharding in its type — a bare `random.uniform` mask source, a binning
spelling with no sharded lowering — lowers REPLICATED under the Explicit mesh, and the
per-kind `[n_layers, B, T, C]` scan stacks then hold the FULL eval batch on every rank
(112 GiB per big kind at the 32L production shape). The compiled-HLO pin and the
placed-vs-unplaced equality tests below hold that seam closed."""

import re
from typing import Any

import jax
import jax.numpy as jnp
import numpy as _np
import numpy as np
import pytest
from jax import random
from jax.sharding import AxisType, Mesh

from param_decomp.core.ci_fn import (
    Chunk,
    ChunkwiseTransformerCIArch,
    MHACIAttention,
    PlacedCIFn,
    build_ci_fn,
    resolve_ci_placement,
)
from param_decomp.core.components import SiteC, init_component_stacks
from param_decomp.core.configs import WellTemperednessConfig
from param_decomp.core.init_placed import init_ci_fn_placed, init_component_stacks_placed
from param_decomp.core.model import PlacedModel
from param_decomp.core.placement import from_config
from param_decomp.core.sharding import place_target, shard_batch
from param_decomp.core.slow_eval import make_ci_reduction_step, make_position_ci_step
from param_decomp.core.well_temperedness import make_well_temperedness_step
from param_decomp.experiments.lm.arithmetic_eval import make_arithmetic_grid_step
from param_decomp.experiments.lm.attn_patterns_eval import (
    attn_output_key_by_site,
    make_ci_attn_patterns_step,
    make_stochastic_attn_patterns_step,
)
from param_decomp.experiments.lm.eval import make_ce_kl_step, make_ci_l0_step
from param_decomp.experiments.lm.eval_context import make_lm_batch_context_step
from param_decomp.targets.glu_transformer import glu_site_specs, site_name
from param_decomp.targets.testing import tiny_glu_cfg, tiny_glu_decomposed_lm
from param_decomp.targets.transformer_taps import resid_tap_key
from param_decomp.vendored_jax.llama import LlamaConfig

pytestmark = [
    pytest.mark.multidevice,
    pytest.mark.skipif(
        jax.default_backend() != "cpu" or jax.device_count() < 4,
        reason="requires a four-device CPU topology from make test-multidevice",
    ),
]


def _ci_arch(cfg: LlamaConfig, site_names: tuple[str, ...]) -> ChunkwiseTransformerCIArch:
    return ChunkwiseTransformerCIArch(
        chunks=(
            Chunk(input_taps=(resid_tap_key(3),), output_sites=site_names[:7]),
            Chunk(input_taps=(resid_tap_key(4),), output_sites=site_names[7:]),
        ),
        input_dim=cfg.n_embd,
        d_model=16,
        n_blocks=1,
        attention=MHACIAttention(n_heads=2),
        ffn_hidden=32,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )


def _placed_setup(seq: int, gbatch: int):
    cfg = tiny_glu_cfg()
    C = 8
    site_cs = tuple(
        SiteC(site_name(layer, kind), C)
        for layer in (3, 4)
        for kind in ("q", "k", "v", "o", "gate", "up", "down")
    )
    sites = glu_site_specs(cfg, site_cs)
    model = tiny_glu_decomposed_lm(cfg, sites, random.PRNGKey(0))
    mesh = Mesh(
        _np.asarray(jax.devices()[:4]).reshape(2, 2, 1),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    rules = from_config("owner", mesh, model.sites)
    model = place_target(model, rules)
    vu = init_component_stacks_placed(sites, random.PRNGKey(1), rules)
    arch = _ci_arch(cfg, model.site_names)
    ci_fn = init_ci_fn_placed(arch, model.sites, random.PRNGKey(2), mesh, rules)
    tokens = random.randint(random.PRNGKey(4), (gbatch, seq), 0, cfg.vocab_size)
    tokens = shard_batch(tokens, mesh, batch_axis=0)
    return model, vu, PlacedCIFn(fn=ci_fn, placement=rules.ci_fn), tokens, mesh, rules


def test_every_placed_eval_tier_traces_and_returns_finite_values():
    model, vu, ci_fn, tokens, mesh, _ = _placed_setup(seq=16, gbatch=8)
    key = random.PRNGKey(9)

    with jax.set_mesh(mesh):
        ce_kl = make_ce_kl_step(model, ci_fn.fn.capture_keys, 0.5, mesh)(
            model, vu, ci_fn, tokens, key
        )
        assert all(np.isfinite(np.asarray(v)).all() for v in ce_kl.values())

        l0 = make_ci_l0_step(model, ci_fn.fn.capture_keys, 0.1, None, mesh)(
            model, vu, ci_fn, tokens, key
        )
        assert all(np.isfinite(np.asarray(v)).all() for v in l0.values())

        output_key_by_site = attn_output_key_by_site(model)
        context_step = make_lm_batch_context_step(
            model, ci_fn.fn.capture_keys, frozenset(output_key_by_site.values()), mesh
        )
        ctx_tokens, _clean_output, captures, ci, prepared_weights = context_step(
            model, vu, ci_fn, tokens
        )
        clean_qk = {site: captures[key] for site, key in output_key_by_site.items()}

        density, ci_sums, n_positions, binned_lower, binned_pre, density_hist = (
            make_ci_reduction_step(0.1, 8, 16)(ci.preactivations)
        )
        assert int(n_positions) == tokens.shape[0] * tokens.shape[1]
        for tree in (density, ci_sums, binned_lower, binned_pre, density_hist):
            assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(tree))

        lower_sum, upper_sum, n_batch = make_position_ci_step()(ci.preactivations)
        assert int(n_batch) == tokens.shape[0]
        for tree in (lower_sum, upper_sum):
            assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(tree))
        wt = make_well_temperedness_step(
            model,
            ci_fn.fn.capture_keys,
            WellTemperednessConfig(
                groups=None, n_locations=2, n_components_per_region=4, ablations_per_forward=4
            ),
            mesh,
        )(model, vu, ci_fn, tokens, key)
        assert np.isfinite(np.asarray(wt.damage)).all()

        kl, _ = make_ci_attn_patterns_step(model)(
            model, prepared_weights, ctx_tokens, ci.lower, clean_qk, key
        )
        assert all(np.isfinite(np.asarray(v)).all() for v in kl.values())

        kl, _ = make_stochastic_attn_patterns_step(model, 2)(
            model, prepared_weights, ctx_tokens, ci.lower, clean_qk, key
        )
        assert all(np.isfinite(np.asarray(v)).all() for v in kl.values())

        ci_grid, activation_grid, max_ci = make_arithmetic_grid_step(
            model,
            ci_fn.fn.capture_keys,
            answer_position=8,
            n_valid_rows=tokens.shape[0],
        )(model, vu, ci_fn, tokens)
        for grid in (ci_grid, activation_grid, max_ci):
            assert all(np.isfinite(np.asarray(v)).all() for v in grid.values())

        # The trainer-side dtype policy holds through the placed prepare: compute
        # residents are bf16 and carry the chained-reduced typing.
        from param_decomp.core.model import prepare_compute_weights

        prepared = prepare_compute_weights(model, vu)
        v_leaf = prepared["gate"]["V"]
        assert v_leaf.dtype == jnp.bfloat16
        assert set(jax.typeof(v_leaf).sharding.spec.reduced) == {"replicate"}


_SHAPE_TOKEN = re.compile(r"\b(?:pred|s8|u8|bf16|f16|s16|u16|f32|s32|u32|f64|s64|u64)\[([\d,]+)\]")


def _compiled_shape_dims(step: Any, args: tuple[Any, ...]) -> set[int]:
    """Every dimension extent appearing in a buffer shape of the step's compiled (i.e.
    SPMD-partitioned, per-device) module. Shape tokens are `dtype[d0,d1,...]`; the dtype
    prefix keeps metadata brackets (equinox arg tags, source annotations) out of the
    census. `step` is an `eqx.filter_jit` wrapper (its Compiled wraps the jax one)."""
    hlo = step.lower(*args).compile().compiled.as_text()
    dims: set[int] = set()
    for group in _SHAPE_TOKEN.findall(hlo):
        dims.update(int(d) for d in group.split(","))
    return dims


def test_eval_steps_keep_the_batch_axis_sharded_in_compiled_hlo():
    """The reno 1375752 eval-OOM pin: no buffer in the compiled per-device eval modules
    may carry the GLOBAL batch extent. Bare mask-source draws lowered replicated and the
    masked forward stacked them into full-batch `[n_layers, B, T, C]` scan inputs
    (`uniform_like` is the required spelling); the slow tier's `bincount` binning could
    not lower sharded at all. The batch extent (40) collides with no other dimension of
    this setup, so its absence IS the sharding proof; the local extent (40/dp = 10) must
    appear — the positive control that the census reads real shapes."""
    model, vu, ci_fn, tokens, mesh, _ = _placed_setup(seq=24, gbatch=40)
    key = random.PRNGKey(9)
    keys = ci_fn.fn.capture_keys
    output_key_by_site = attn_output_key_by_site(model)
    with jax.set_mesh(mesh):
        context_step = make_lm_batch_context_step(
            model, keys, frozenset(output_key_by_site.values()), mesh
        )
        ctx_tokens, _clean_output, captures, ci, prepared_weights = context_step(
            model, vu, ci_fn, tokens
        )
        clean_qk = {site: captures[key_] for site, key_ in output_key_by_site.items()}
        step_args = {
            "ce_kl": (make_ce_kl_step(model, keys, 0.5, mesh), (model, vu, ci_fn, tokens, key)),
            "ci_l0": (
                make_ci_l0_step(model, keys, 0.1, None, mesh),
                (model, vu, ci_fn, tokens, key),
            ),
            "batch_context": (context_step, (model, vu, ci_fn, tokens)),
            "ci_reduction": (make_ci_reduction_step(0.1, 8, 16), (ci.preactivations,)),
            "stochastic_attn": (
                make_stochastic_attn_patterns_step(model, 2),
                (model, prepared_weights, ctx_tokens, ci.lower, clean_qk, key),
            ),
        }
        for name, (step, args) in step_args.items():
            dims = _compiled_shape_dims(step, args)
            assert 40 not in dims, f"{name}: a compiled buffer holds the FULL eval batch"
            assert 10 in dims, f"{name}: no per-device batch shard found — census is blind"


def test_sharded_binning_and_uniform_draws_are_value_identical():
    """The two sharded eval spellings, pinned EXACTLY on identical inputs.

    `_binned_values` / `_per_component_ci_hist` are pure reductions — the batch-sharded
    array must produce byte-identical counts/edges to the unsharded one (integer counts
    are reorder-proof). `uniform_like` must draw the same values sharded as unsharded:
    threefry is counter-based, so partitioning the draw never changes it (SPEC D4)."""
    from param_decomp.core.linear_plan import uniform_like
    from param_decomp.core.slow_eval import _binned_values, _per_component_ci_hist

    mesh = Mesh(
        _np.asarray(jax.devices()[:4]).reshape(2, 2, 1),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    values = random.uniform(random.PRNGKey(0), (8, 16, 8), jnp.float32, -0.5, 1.5)
    ci = jnp.clip(values, 0.0, 1.0).astype(jnp.bfloat16)
    key = random.PRNGKey(7)

    counts_u, lo_u, hi_u = jax.jit(_binned_values, static_argnums=1)(values, 16)
    hist_u = jax.jit(_per_component_ci_hist, static_argnums=1)(ci, 8)
    draw_u = jax.jit(uniform_like)(key, ci)

    with jax.set_mesh(mesh):
        sharded = shard_batch(values, mesh, batch_axis=0)
        ci_sharded = shard_batch(ci, mesh, batch_axis=0)
        counts_p, lo_p, hi_p = jax.jit(_binned_values, static_argnums=1)(sharded, 16)
        hist_p = jax.jit(_per_component_ci_hist, static_argnums=1)(ci_sharded, 8)
        draw_p = jax.jit(uniform_like)(key, ci_sharded)

    assert float(lo_u) == float(lo_p) and float(hi_u) == float(hi_p)
    assert np.array_equal(np.asarray(counts_p), np.asarray(counts_u))
    assert np.array_equal(np.asarray(hist_p), np.asarray(hist_u))
    assert np.array_equal(np.asarray(draw_p), np.asarray(draw_u))


def test_scalar_eval_values_match_unplaced():
    """Placed (ddp, batch-sharded, Explicit mesh) CE/KL scalars equal the unplaced run's
    within reassociation. ddp replicates the weights, so the arms differ only in batch
    layout; a wrong sharded draw (different mask VALUES rather than a different layout)
    shifts the stochastic/random variants at the 1e-2+ scale, far past this tolerance.
    Per-element bf16 kernel differences across batch layouts make bit-exactness
    unavailable — the spelling-level exactness lives in
    `test_sharded_binning_and_uniform_draws_are_value_identical`."""
    cfg = tiny_glu_cfg()
    C, seq, gbatch = 8, 16, 8
    site_cs = tuple(
        SiteC(site_name(layer, kind), C)
        for layer in (3, 4)
        for kind in ("q", "k", "v", "o", "gate", "up", "down")
    )
    sites = glu_site_specs(cfg, site_cs)
    raw_model = tiny_glu_decomposed_lm(cfg, sites, random.PRNGKey(0))
    vu = init_component_stacks(sites, random.PRNGKey(1))
    arch = _ci_arch(cfg, tuple(s.name for s in sites))
    ci_fn = build_ci_fn(arch, sites, random.PRNGKey(2))
    tokens = random.randint(random.PRNGKey(4), (gbatch, seq), 0, cfg.vocab_size)
    key = random.PRNGKey(9)

    unplaced_model = PlacedModel(model=raw_model, placement=None)
    unplaced_ci = PlacedCIFn(fn=ci_fn, placement=None)
    keys = unplaced_ci.fn.capture_keys
    ce_kl_single = make_ce_kl_step(unplaced_model, keys, 0.5)(
        unplaced_model, vu, unplaced_ci, tokens, key
    )

    mesh = Mesh(
        _np.asarray(jax.devices()[:4]).reshape(2, 2, 1),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    rules = from_config("ddp", mesh, sites)
    sharded_model = PlacedModel(model=raw_model, placement=rules)
    sharded_ci = PlacedCIFn(fn=ci_fn, placement=resolve_ci_placement(arch, rules))
    sharded_tokens = shard_batch(tokens, mesh, batch_axis=0)
    with jax.set_mesh(mesh):
        ce_kl_sharded = make_ce_kl_step(sharded_model, keys, 0.5, mesh)(
            sharded_model, vu, sharded_ci, sharded_tokens, key
        )

    assert ce_kl_single.keys() == ce_kl_sharded.keys()
    for name in ce_kl_single:
        single, sharded = float(ce_kl_single[name]), float(ce_kl_sharded[name])
        # atol sized to the O(1) CE/KL means the keys reduce from (the `ce_difference_*`
        # keys are differences of two such means, and `rounded_masked` amplifies a
        # per-element bf16 flip at the 0.5 threshold into one changed forward).
        assert abs(single - sharded) <= 1e-4 * abs(single) + 5e-4, (name, single, sharded)
