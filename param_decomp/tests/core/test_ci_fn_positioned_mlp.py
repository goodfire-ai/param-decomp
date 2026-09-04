"""The per-site MLP CI fn over a POSITIONED target.

`has_position_axis` is the target's tensor shape, not a property of the CI fn: `SiteMLP` is
`[*leading, d_in] -> [*leading, C]`, pointwise over every leading axis, so the same weights
serve `[batch, d]` and `[batch, position, d]` alike. The arch therefore declares the axis and
the runtime checks it against the model (`run_state.init_decomposition`).

This is what a positioned target whose positions are too numerous to attend over should reach
for: position-local like `n_blocks=0`, but with a learned hidden layer and — the sharp
difference — able to read tap MAGNITUDE. The chunkwise path RMS-norms every tap before
`in_proj`, so a blockless chunk is `RMSNorm -> affine`: scale-invariant by construction.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import random

from param_decomp.core.ci_fn import (
    Chunk,
    ChunkwiseTransformerCIArch,
    CIFn,
    LayerwiseMLPCIArch,
    LayerwiseMLPCIFn,
    MHACIAttention,
    PlacedCIFn,
    build_ci_fn,
    evaluate_ci,
)
from param_decomp.core.components import SiteSpec, component_stacks_from_sites
from param_decomp.core.configs import (
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    StochasticReconLossConfig,
)
from param_decomp.core.faithfulness import faithfulness_loss_for
from param_decomp.core.model import PlacedModel
from param_decomp.core.objective import build_objective
from param_decomp.core.precision import COMPUTE_DT
from param_decomp.core.schedule import Knot, ScheduleConfig
from param_decomp.core.train import (
    Decomposition,
    ForwardSubstrate,
    TrainingItem,
    TrainState,
    make_train_step,
)
from param_decomp.tests.core.test_generic_model_io import SyntheticDecomposedModel

SITE = "block.0.proj"
B, T, D, C = 2, 5, 8, 4
SITES = (SiteSpec(name=SITE, d_in=D, d_out=D, C=C, group=SITE),)


def _positioned_mlp_ci_fn() -> LayerwiseMLPCIFn:
    ci_fn = build_ci_fn(
        LayerwiseMLPCIArch(
            hidden_dims=(16,),
            has_position_axis=True,
            input_names=tuple(s.name for s in SITES),
        ),
        SITES,
        random.PRNGKey(0),
    )
    assert isinstance(ci_fn, LayerwiseMLPCIFn)
    return ci_fn


def _blockless_chunkwise_ci_fn():
    return build_ci_fn(
        ChunkwiseTransformerCIArch(
            chunks=(Chunk(input_taps=(SITE,), output_sites=(SITE,)),),
            input_dim=D,
            d_model=16,
            n_blocks=0,
            attention=MHACIAttention(n_heads=2),
            ffn_hidden=32,
            ffn_kind="gelu",
            learned_norm_scale=False,
        ),
        SITES,
        random.PRNGKey(0),
    )


def _taps(key_idx: int = 1) -> dict[str, jax.Array]:
    return {SITE: random.normal(random.PRNGKey(key_idx), (B, T, D))}


def test_mlp_arch_serves_a_positioned_target():
    ci_fn = _positioned_mlp_ci_fn()
    assert ci_fn.has_position_axis is True

    ci = ci_fn(_taps(), remat=False, placement=None)
    assert ci.preactivations[SITE].shape == (B, T, C)
    assert bool(jnp.isfinite(ci.lower[SITE]).all())


def test_evaluate_ci_casts_fp32_taps_to_compute_precision():
    ci_fn = _positioned_mlp_ci_fn()
    taps = _taps()
    assert taps[SITE].dtype == jnp.float32
    assert ci_fn(taps, remat=False, placement=None).preactivations[SITE].dtype == jnp.float32
    assert (
        evaluate_ci(PlacedCIFn(fn=ci_fn, placement=None), taps, remat=False)
        .preactivations[SITE]
        .dtype
        == COMPUTE_DT
    )


def test_positioned_mlp_is_position_local():
    """Position t's CI reads position t's tap and nothing else — the property that makes this
    affordable when positions are a large derived set (residue PAIRS)."""
    ci_fn = _positioned_mlp_ci_fn()
    taps = _taps()
    perturbed = {SITE: taps[SITE].at[:, 2, :].add(100.0)}

    moved = jnp.abs(
        ci_fn(taps, remat=False, placement=None).preactivations[SITE]
        - ci_fn(perturbed, remat=False, placement=None).preactivations[SITE]
    ).max(axis=(0, 2))

    assert float(moved[2]) > 0.0, "the perturbed position must move at all"
    untouched = jnp.delete(moved, 2, axis=0)
    assert jnp.array_equal(untouched, jnp.zeros_like(untouched)), (
        f"leaked across positions: {moved}"
    )


def test_positioned_mlp_reads_tap_magnitude_where_the_blockless_chunk_cannot():
    """The reason to prefer this over `n_blocks=0`. The chunkwise path RMS-norms every tap
    before `in_proj`, so a blockless chunk is `RMSNorm -> affine`: its CI depends only on the
    tap's DIRECTION and is invariant to scaling it. The MLP reads the tap raw, so a
    subcomponent whose activation grows can be scored differently — which is usually the
    point of a causal-importance function."""

    def scale_gap(ci_fn: CIFn) -> float:
        x = _taps(1)[SITE]
        f = lambda t: ci_fn({SITE: t}, remat=False, placement=None).preactivations[SITE]  # noqa: E731
        return float(jnp.abs(f(x) - f(x * 7.0)).max())

    assert scale_gap(_positioned_mlp_ci_fn()) > 1.0, "the MLP CI fn ignored tap magnitude"
    assert scale_gap(_blockless_chunkwise_ci_fn()) < 1e-5, (
        "the blockless chunk was NOT scale-invariant — the contrast this test draws is void"
    )


def test_positioned_mlp_trains_a_positioned_target():
    """The real `make_train_step`: the MLP CI fn drives a positioned dict-in/tuple-out target."""
    key = random.PRNGKey(2)
    model = SyntheticDecomposedModel(
        feat_proj=random.normal(random.fold_in(key, 7), (D, D)),
        W=random.normal(random.fold_in(key, 0), (D, D)),
        read_coords=random.normal(random.fold_in(key, 1), (4, D)),
        read_aux=random.normal(random.fold_in(key, 2), (2, D)),
        sites=SITES,
        has_position_axis=True,
    )
    components = component_stacks_from_sites(
        {
            SITE: (
                random.normal(random.fold_in(key, 3), (D, C)) * 0.1,
                random.normal(random.fold_in(key, 4), (C, D)) * 0.1,
            )
        }
    )
    inputs = {
        "feat": random.normal(random.fold_in(key, 5), (B, T, D)),
        "gain": random.uniform(random.fold_in(key, 6), (B, T)),
    }
    ci_fn = _positioned_mlp_ci_fn()
    assert ci_fn.has_position_axis == model.has_position_axis  # the run_state agreement assert

    opt_vu = optax.adamw(1e-2, weight_decay=0.0)
    opt_ci = optax.adamw(1e-2, weight_decay=0.0)
    state = TrainState(
        decomposition=Decomposition(components=components, ci_fn=ci_fn),
        training=TrainingItem(
            components_opt_state=opt_vu.init(eqx.filter(components, eqx.is_array)),
            ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
            adversaries={},
            freq_ema=None,
            step=jnp.zeros((), jnp.int32),
        ),
    )
    loss_terms = build_objective(
        (
            FaithfulnessLossConfig(coeff=1.0),
            ImportanceMinimalityLossConfig(
                coeff=1e-4,
                gamma=ScheduleConfig(
                    max_val=1.0, points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.01))
                ),
            ),
            StochasticReconLossConfig(coeff=1.0),
        ),
        model.site_names,
    )
    placed = PlacedModel(model=model, placement=None)
    step_fn = make_train_step(
        model_static=placed,
        substrate=ForwardSubstrate.of(
            placed,
            remat_recon_forwards=False,
            remat_ci_fn=True,
            ci_capture_keys=ci_fn.capture_keys,
            ci_placement=None,
        ),
        objective=loss_terms,
        components_optimizer=opt_vu,
        ci_fn_optimizer=opt_ci,
        total_steps=10,
        faithfulness=faithfulness_loss_for(model),
    )

    first_before = jax.device_get(ci_fn.site_mlps[SITE].weights[0])  # survives step donation
    for step_idx in range(2):
        state, metrics = step_fn(placed, state, inputs, random.fold_in(random.PRNGKey(3), step_idx))
        assert jnp.isfinite(metrics["total"]), (step_idx, metrics["total"])

    trained = state.decomposition.ci_fn
    assert isinstance(trained, LayerwiseMLPCIFn)
    assert not jnp.allclose(trained.site_mlps[SITE].weights[0], first_before), "CI fn did not move"
