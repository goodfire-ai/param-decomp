"""The dtype census of the placed train step, on the tiny qwen36
cell at an 8-device simulated `(data, tp)` mesh.

Every buffer class is read off a TYPED value — the state pytree, `jax.eval_shape` of the
placement lifecycle's outputs, or the jaxpr of a loss kernel — never off a production
hook. JAX types a cotangent as its primal, so the resident compute weights' dtype IS the
resident cotangent stacks' dtype and the dtype the deferred exit reduce-scatter sums in:
one assertion pins compute weights, cotangent accumulation, and the wire together.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.core import ShapedArray
from jax.extend.core import ClosedJaxpr, Jaxpr
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.core.adversary import (
    PersistentAdversary,
    SourcesMomentumState,
    init_sources_opt_state,
)
from param_decomp.core.ci_fn import (
    MoEChunkwiseTransformerCIFn,
    PlacedCIFn,
    evaluate_compute_ci,
    materialize_ci_compute_weights,
    resolve_ci_placement,
)
from param_decomp.core.configs import (
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    MomentumSgdPGDConfig,
    PersistentPGDReconLossConfig,
    StochasticReconLossConfig,
)
from param_decomp.core.faithfulness import faithfulness_loss_for
from param_decomp.core.init_placed import (
    init_ci_fn_placed,
    init_component_stacks_placed,
    init_sources_sharded,
)
from param_decomp.core.losses import activity_sum_from_ci
from param_decomp.core.model import (
    Positioned,
    faithfulness_weight_deltas,
    prepare_compute_weights,
)
from param_decomp.core.objective import build_objective
from param_decomp.core.placement import from_config
from param_decomp.core.schedule import Knot, ScheduleConfig
from param_decomp.core.sharding import place_target
from param_decomp.core.train import (
    Decomposition,
    ForwardSubstrate,
    TrainingItem,
    TrainState,
    make_train_step,
)
from param_decomp.targets.losses import lm_output_kl_per_position
from param_decomp.tests.targets.test_qwen36_placement import (
    BATCH,
    CENSUS_CS,
    DATA,
    SEQ,
    TP,
    _batch_placed,
    _model_and_components,
    _moe_ci_arch,
)

multidevice = pytest.mark.skipif(len(jax.devices()) < 8, reason="requires eight local devices")

RESIDENT_COTANGENT_DTYPE = jnp.bfloat16
"""Sharded spelling as it stands: masters are cast to the compute dtype BEFORE
the entry gather, so the resident stacks — and with them their cotangents and the exit
reduce-scatter's wire — are bf16. This test pins that implemented boundary rather than
a hypothetical per-term pullback or fp32 resident-cotangent design."""

_TRANSCENDENTAL = frozenset({"exp", "exp2", "log", "log1p", "logistic", "tanh", "erf", "rsqrt"})


def _dtypes(tree: Any) -> set[jnp.dtype]:
    return {jnp.dtype(leaf.dtype) for leaf in jax.tree.leaves(tree)}


def _inexact_dtypes(tree: Any) -> set[jnp.dtype]:
    return {dtype for dtype in _dtypes(tree) if jnp.issubdtype(dtype, jnp.inexact)}


def _transcendental_input_dtypes(jaxpr: Jaxpr) -> set[jnp.dtype]:
    """The dtypes every transcendental primitive in the (recursively opened) jaxpr consumes."""
    found: set[jnp.dtype] = set()
    for eqn in jaxpr.eqns:
        if eqn.primitive.name in _TRANSCENDENTAL:
            for var in eqn.invars:
                assert isinstance(var.aval, ShapedArray), var.aval
                found.add(jnp.dtype(var.aval.dtype))
        for value in eqn.params.values():
            match value:
                case ClosedJaxpr():
                    found |= _transcendental_input_dtypes(value.jaxpr)
                case Jaxpr():
                    found |= _transcendental_input_dtypes(value)
                case _:
                    pass
    return found


def _mesh() -> Mesh:
    devices = np.asarray(jax.devices()[: DATA * TP]).reshape(DATA, TP)
    return Mesh(devices, ("data", "tp"), axis_types=(AxisType.Explicit,) * 2)


@multidevice
@pytest.mark.multidevice
def test_placed_train_step_dtype_census():
    ppgd = PersistentPGDReconLossConfig(
        coeff=0.5,
        n_warmup_steps=1,
        source_shape="bsc",
        source_dtype="uint16",
        optimizer=MomentumSgdPGDConfig(momentum=0.9, lr_schedule=ScheduleConfig.constant(0.05)),
    )
    model, _components = _model_and_components(CENSUS_CS)
    sites = model.sites
    arch = _moe_ci_arch(model.cfg)
    objective = build_objective(
        (
            FaithfulnessLossConfig(coeff=1.0),
            ImportanceMinimalityLossConfig(
                coeff=1e-4,
                gamma=ScheduleConfig(
                    max_val=1.0, points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.01))
                ),
            ),
            StochasticReconLossConfig(coeff=1.0),
            ppgd,
        ),
        model.site_names,
    )
    mesh = _mesh()
    rules = from_config("owner-replicated-resident-moe", mesh, sites)
    placed_model = place_target(model, rules)
    ci_placement = resolve_ci_placement(arch, rules)
    opt_vu = optax.adamw(1e-3, weight_decay=0.0)
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)

    with jax.set_mesh(mesh):
        components = init_component_stacks_placed(sites, jax.random.PRNGKey(1), rules)
        ci_fn = init_ci_fn_placed(arch, sites, jax.random.PRNGKey(2), mesh, rules)
        assert isinstance(ci_fn, MoEChunkwiseTransformerCIFn)
        sources = init_sources_sharded(
            sites, Positioned(SEQ), "bsc", BATCH, jnp.uint16, jax.random.PRNGKey(7), mesh
        )
        adversary = PersistentAdversary(
            sources=sources,
            opt_state=init_sources_opt_state(ppgd.optimizer, sources),
            state_key=ppgd.type,
            optimizer=ppgd.optimizer,
            n_warmup=ppgd.n_warmup_steps,
        )
        state = TrainState(
            decomposition=Decomposition(components=components, ci_fn=ci_fn),
            training=TrainingItem(
                components_opt_state=opt_vu.init(eqx.filter(components, eqx.is_array)),
                ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
                adversaries={ppgd.type: adversary},
                freq_ema=None,
                step=jnp.zeros((), jnp.int32),
            ),
        )
        step_fn = make_train_step(
            model_static=placed_model,
            substrate=ForwardSubstrate.of(
                placed_model,
                remat_recon_forwards=True,
                remat_ci_fn=True,
                ci_capture_keys=ci_fn.capture_keys,
                ci_placement=ci_placement,
            ),
            objective=objective,
            components_optimizer=opt_vu,
            ci_fn_optimizer=opt_ci,
            total_steps=4,
            faithfulness=faithfulness_loss_for(placed_model),
        )
        tokens = _batch_placed(
            jax.random.randint(jax.random.PRNGKey(3), (BATCH, SEQ), 0, model.cfg.vocab_size), mesh
        )
        key = jax.random.PRNGKey(4)

        # fp32 masters (V/U and CI fn) and fp32 optimizer moments, before and after a step
        new_state, metrics = jax.eval_shape(
            lambda m, s, b, k: step_fn(m, s, b, k), placed_model, state, tokens, key
        )
        for item in (state, new_state):
            assert _dtypes(item.decomposition) == {jnp.dtype(jnp.float32)}, _dtypes(
                item.decomposition
            )
            assert _inexact_dtypes(item.training.components_opt_state) == {jnp.dtype(jnp.float32)}
            assert _inexact_dtypes(item.training.ci_fn_opt_state) == {jnp.dtype(jnp.float32)}
            # uint16 fixed-point sources with the momentum_sgd bf16 velocity
            adv = item.training.adversaries[ppgd.type]
            assert _dtypes(adv.sources) == {jnp.dtype(jnp.uint16)}, _dtypes(adv.sources)
            assert isinstance(adv.opt_state, SourcesMomentumState)
            assert _dtypes(adv.opt_state.velocity) == {jnp.dtype(jnp.bfloat16)}
        # loss scalars fp32 (every inexact metric; the step counter is the one integer)
        assert _inexact_dtypes(metrics) == {jnp.dtype(jnp.float32)}, _inexact_dtypes(metrics)

        # compute weights bf16 == resident cotangent stacks == the exit wire
        prepared = jax.eval_shape(prepare_compute_weights, placed_model, components)
        assert _dtypes(prepared) == {jnp.dtype(RESIDENT_COTANGENT_DTYPE)}, _dtypes(prepared)
        _, pullback = jax.vjp(lambda c: prepare_compute_weights(placed_model, c), components)
        # the cotangent of a `reduced`-typed resident arrives `unreduced` (a per-rank partial)
        cotangents = jax.tree.map(
            lambda s: jax.ShapeDtypeStruct(
                s.shape,
                s.dtype,
                sharding=NamedSharding(
                    mesh,
                    P(*s.sharding.spec.partitions, unreduced=frozenset(s.sharding.spec.reduced)),
                ),
            ),
            prepared,
        )
        (master_grad,) = jax.eval_shape(pullback, cotangents)
        assert _dtypes(master_grad) == {jnp.dtype(jnp.float32)}, _dtypes(master_grad)
        compute_ci_fn = jax.eval_shape(
            materialize_ci_compute_weights, PlacedCIFn(fn=ci_fn, placement=ci_placement)
        )
        assert _inexact_dtypes(compute_ci_fn.fn) == {jnp.dtype(jnp.bfloat16)}

        # faithfulness deltas fp32
        deltas = jax.eval_shape(faithfulness_weight_deltas, placed_model, components)
        assert _dtypes(deltas) == {jnp.dtype(jnp.float32)}, _dtypes(deltas)

        # (uniform CI compute): CI activations bf16 (router indices ride as integers)
        taps = jax.eval_shape(
            lambda m, b: m.clean_forward(b, ci_fn.capture_keys).captures, placed_model, tokens
        )
        ci = jax.eval_shape(
            lambda cf, t: evaluate_compute_ci(materialize_ci_compute_weights(cf), t, remat=True),
            PlacedCIFn(fn=ci_fn, placement=ci_placement),
            taps,
        )
        assert _inexact_dtypes(ci) == {jnp.dtype(jnp.bfloat16)}, _inexact_dtypes(ci)

        # the imp-min reduction and the KL run their transcendentals in fp32 on bf16 inputs
        gamma = jnp.asarray(0.5, jnp.float32)
        imp_activity = lambda c: activity_sum_from_ci(c.upper, gamma, normalize_at_one=False)  # noqa: E731
        assert jax.eval_shape(imp_activity, ci).dtype == jnp.float32
        assert _transcendental_input_dtypes(jax.make_jaxpr(imp_activity)(ci).jaxpr) <= {
            jnp.dtype(jnp.float32)
        }
        logits = jax.ShapeDtypeStruct((BATCH, SEQ, model.cfg.vocab_size), jnp.bfloat16)
        assert jax.eval_shape(lm_output_kl_per_position, logits, logits).dtype == jnp.float32
        kl_jaxpr = jax.make_jaxpr(lm_output_kl_per_position)(logits, logits).jaxpr
        assert _transcendental_input_dtypes(kl_jaxpr) == {jnp.dtype(jnp.float32)}, (
            _transcendental_input_dtypes(kl_jaxpr)
        )
