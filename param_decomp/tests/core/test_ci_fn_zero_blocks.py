"""`n_blocks=0` — the positioned but position-LOCAL chunkwise CI fn.

A positioned target forces `has_position_axis=True` on its CI fn (`run_state.init_decomposition`
asserts the two agree), and the chunkwise transformer is the only arch that declares it — but its
blocks self-attend OVER the position axis. For a sequence target that is exactly right. For a
target whose positions are a large derived set (residue PAIRS: a 128-residue crop is 16384
positions) the O(P²) attention is infeasible, not merely slow.

`n_blocks=0` is the supported answer: the chunk reduces to `in_proj → per-site output heads`, a
per-position MLP with no attention, still declaring `has_position_axis=True`. These pin that it
constructs, that the zero-length block axis scans, that a real train step runs, and — the
load-bearing property — that the result is exactly position-local.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import random

from param_decomp.core.ci_fn import (
    Chunk,
    ChunkwiseTransformerCIArch,
    ChunkwiseTransformerCIFn,
    MHACIAttention,
    build_ci_fn,
    init_chunkwise_transformer_ci_fn,
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
D_MODEL, FFN = 16, 32
SITES = (SiteSpec(name=SITE, d_in=D, d_out=D, C=C, group=SITE),)


def _arch(n_blocks: int) -> ChunkwiseTransformerCIArch:
    return ChunkwiseTransformerCIArch(
        chunks=(Chunk(input_taps=(SITE,), output_sites=(SITE,)),),
        input_dim=D,
        d_model=D_MODEL,
        n_blocks=n_blocks,
        attention=MHACIAttention(n_heads=2),
        ffn_hidden=FFN,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )


def _ci_fn(n_blocks: int) -> ChunkwiseTransformerCIFn:
    return init_chunkwise_transformer_ci_fn(_arch(n_blocks), SITES, random.PRNGKey(0))


def _taps() -> dict[str, jax.Array]:
    return {SITE: random.normal(random.PRNGKey(1), (B, T, D))}


def test_zero_blocks_is_in_proj_plus_heads_and_nothing_else():
    """The whole point: no attention parameters exist, so no attention can run."""
    ci_fn = _ci_fn(0)
    assert ci_fn.chunks.blocks == []
    assert ci_fn.has_position_axis is True
    n_chunks = 1
    expected = n_chunks * (D * D_MODEL + D_MODEL + D_MODEL * C + C)  # in_proj (w,b) + head (w,b)
    trainable = sum(x.size for x in jax.tree.leaves(ci_fn.chunks))
    assert trainable == expected


def test_zero_blocks_forward_shape_and_finiteness():
    ci_fn = _ci_fn(0)
    for remat in (False, True):
        ci = ci_fn(_taps(), remat=remat, placement=None)
        assert ci.preactivations[SITE].shape == (B, T, C)
        assert ci.lower[SITE].shape == (B, T, C)
        assert bool(jnp.isfinite(ci.lower[SITE]).all())


def test_zero_blocks_is_exactly_position_local():
    """The property a pair-shaped target buys `n_blocks=0` for: position t's CI reads position
    t's tap and nothing else. Contrasted against `n_blocks=1`, whose attention necessarily
    leaks the perturbation to every position — so this pins the arch, not the harness."""
    taps = _taps()
    perturbed = {SITE: taps[SITE].at[:, 2, :].add(100.0)}

    def moved(n_blocks: int) -> jax.Array:
        ci_fn = _ci_fn(n_blocks)
        base = ci_fn(taps, remat=False, placement=None).preactivations[SITE]
        pert = ci_fn(perturbed, remat=False, placement=None).preactivations[SITE]
        return jnp.abs(base - pert).max(axis=(0, 2))  # per-position

    local = moved(0)
    assert float(local[2]) > 0.0, "the perturbed position must move at all"
    untouched = jnp.delete(local, 2, axis=0)
    assert jnp.array_equal(untouched, jnp.zeros_like(untouched)), (
        f"n_blocks=0 leaked across positions: {local}"
    )
    assert float(jnp.min(jnp.delete(moved(1), 2, axis=0))) > 0.0, (
        "n_blocks=1 did NOT leak across positions — attention is not running, "
        "so the locality assertion above proves nothing"
    )


def test_zero_blocks_trains_a_positioned_target():
    """The real `make_train_step` over a positioned target, `remat_ci_fn=True` so the zero-length
    scan is exercised under `jax.checkpoint` too."""
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
    ci_fn = build_ci_fn(_arch(0), model.sites, random.PRNGKey(11))
    assert isinstance(ci_fn, ChunkwiseTransformerCIFn)
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

    in_proj_before = jax.device_get(ci_fn.chunks.in_proj_w)  # host copy survives step donation
    for step_idx in range(2):
        state, metrics = step_fn(placed, state, inputs, random.fold_in(random.PRNGKey(3), step_idx))
        assert jnp.isfinite(metrics["total"]), (step_idx, metrics["total"])

    trained = state.decomposition.ci_fn
    assert isinstance(trained, ChunkwiseTransformerCIFn)
    assert not jnp.allclose(trained.chunks.in_proj_w, in_proj_before), (
        "the CI fn did not move — with no blocks there is nothing else left to train"
    )
    final_ci = trained(
        model.clean_forward(inputs, ci_fn.capture_keys, placement=None).captures,
        remat=False,
        placement=None,
    )
    assert final_ci.lower[SITE].shape == (B, T, C)
