"""Guards the no-bake invariant (model.py module docstring): the frozen target weights must
reach the jitted train step as TRACED ARGS (`jaxpr.invars`), never closed over and baked
into `jaxpr.consts` — for a real 8B target, baking would silently push multi-GB into the HLO.
"""

import inspect

import equinox as eqx
import jax.numpy as jnp
import optax
from jax import random

from param_decomp.core.ci_fn import (
    Chunk,
    ChunkwiseTransformerCIArch,
    MHACIAttention,
    build_ci_fn,
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
from param_decomp.core.tests.test_generic_model_io import SyntheticDecomposedModel
from param_decomp.core.train import (
    Decomposition,
    ForwardSubstrate,
    TrainingItem,
    TrainState,
    make_train_step,
)

SITE = "block.0.proj"
B, T, D, C = 2, 3, 32, 5
K_COORDS, M_AUX = 4, 2
FROZEN_W_SIZE = D * D  # 1024 — comfortably above any legitimately-constant scalar


def _build_step_and_args():
    key = random.PRNGKey(0)
    model = SyntheticDecomposedModel(
        feat_proj=random.normal(random.fold_in(key, 7), (D, D)),
        W=random.normal(random.fold_in(key, 0), (D, D)),
        read_coords=random.normal(random.fold_in(key, 1), (K_COORDS, D)),
        read_aux=random.normal(random.fold_in(key, 2), (M_AUX, D)),
        sites=(SiteSpec(name=SITE, d_in=D, d_out=D, C=C, group=SITE),),
        has_position_axis=True,
    )
    assert model.W.size == FROZEN_W_SIZE
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
    ci_arch = ChunkwiseTransformerCIArch(
        chunks=(Chunk(input_taps=(SITE,), output_sites=(SITE,)),),
        input_dim=D,
        d_model=8,
        n_blocks=1,
        attention=MHACIAttention(n_heads=2),
        ffn_hidden=16,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )
    ci_fn = build_ci_fn(ci_arch, model.sites, random.PRNGKey(11))
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
    nonfaith_terms = (
        ImportanceMinimalityLossConfig(
            coeff=1e-4,
            pnorm=ScheduleConfig(
                max_val=2.0, points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.5))
            ),
        ),
        StochasticReconLossConfig(coeff=1.0),
    )
    loss_terms = build_objective(
        (FaithfulnessLossConfig(coeff=1.0), *nonfaith_terms), model.site_names
    )
    placed = PlacedModel(model=model, placement=None)
    step_fn = make_train_step(
        model_static=placed,
        substrate=ForwardSubstrate.of(
            placed,
            remat_recon_forwards=False,
            remat_ci_fn=False,
            ci_capture_keys=ci_fn.capture_keys,
            ci_placement=None,
        ),
        objective=loss_terms,
        components_optimizer=opt_vu,
        ci_fn_optimizer=opt_ci,
        total_steps=10,
        faithfulness=faithfulness_loss_for(model),
    )
    return step_fn, placed, state, inputs


def test_frozen_weights_are_jaxpr_args_not_baked_consts():
    """The real jitted step traces the model's array leaves as invars (not consts)."""
    step_fn, model, state, inputs = _build_step_and_args()
    # `make_train_step` returns the `eqx.filter_jit`-wrapped step; unwrap to the undecorated
    # inner function. `filter_make_jaxpr` partitions it exactly as `filter_jit` does (arrays
    # traced, the rest static), so this is the same arg→trace boundary production runs through.
    inner_step = inspect.unwrap(step_fn)
    closed_jaxpr = eqx.filter_make_jaxpr(inner_step)(model, state, inputs, random.PRNGKey(3))[0]

    invar_sizes = {v.aval.size for v in closed_jaxpr.jaxpr.invars}
    assert FROZEN_W_SIZE in invar_sizes, (
        "frozen W not traced as an argument — it must appear among jaxpr.invars"
    )
    assert all(const.size < FROZEN_W_SIZE for const in closed_jaxpr.consts), (
        f"a constant >= the frozen W size ({FROZEN_W_SIZE}) was baked into the HLO: "
        f"{sorted(c.size for c in closed_jaxpr.consts)}"
    )
