"""Target-generic fast-tier eval operations (`PGDReconLoss`, `CI_L0`).

The non-LM counterpart of `experiments/lm/scalar_eval_operations.py`: it binds the core
kernels — which are generic over both waist geometries, the target's input pytree and its
output/recon metric — to the ordinary `EvalInvocation` cadence. A target that samples its
own eval batches needs nothing beyond `sample_eval_batch`, whatever shape those batches
are; positioned non-categorical targets included.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jaxtyping import Array, PRNGKeyArray

from param_decomp.core.ci_fn import PlacedCIFn
from param_decomp.core.ci_l0_eval import make_ci_l0_eval_step
from param_decomp.core.components import ComponentStacks
from param_decomp.core.configs import CI_L0Config, PGDReconLossConfig
from param_decomp.core.eval_schedule import EvalSchedule
from param_decomp.core.model import CaptureKeys, PlacedModel
from param_decomp.core.recon import resolve_reconstruction_spec
from param_decomp.core.recon_eval import FreshPGDReconEval, make_fresh_pgd_eval_step
from param_decomp.core.run import EvalInvocation, PassOperation
from param_decomp.experiments.eval_config import EvalConfig

type ScalarStep = Callable[
    [PlacedModel, ComponentStacks, PlacedCIFn, Any, PRNGKeyArray], dict[str, Array]
]


def _averaged_over_eval_batches(
    step: ScalarStep,
    eval_config: EvalConfig,
    schedule: EvalSchedule,
    seed: int,
    model: PlacedModel,
    sample_eval_batch: Callable[[int], Any],
) -> PassOperation[EvalInvocation]:
    """Run `step` over the pass's eval batches and average each scalar it emits."""
    eval_key = jax.random.PRNGKey(seed + 1)

    def run(context: EvalInvocation) -> dict[str, float]:
        pass_index = context.now_step // eval_config.every
        sums: dict[str, Array] = {}
        for batch_index in range(eval_config.n_steps):
            flat_index = pass_index * eval_config.n_steps + batch_index
            values = step(
                model,
                context.state.decomposition.components,
                context.placed_ci_fn,
                sample_eval_batch(flat_index),
                jax.random.fold_in(eval_key, flat_index),
            )
            for name, value in values.items():
                sums[name] = sums.get(name, jnp.zeros(())) + value
        return {f"eval/{name}": float(value) / eval_config.n_steps for name, value in sums.items()}

    return PassOperation(schedule, run)


def make_fresh_pgd_operation(
    metric: PGDReconLossConfig,
    eval_config: EvalConfig,
    schedule: EvalSchedule,
    seed: int,
    compiler_options: dict[str, bool | int | str],
    model: PlacedModel,
    ci_capture_keys: CaptureKeys,
    mesh: Mesh | None,
    sample_eval_batch: Callable[[int], Any],
) -> PassOperation[EvalInvocation]:
    assert metric.init == "random" and metric.source_shape == "c", metric
    probe = FreshPGDReconEval(
        name=metric.name or metric.type,
        n_steps=metric.n_steps,
        step_size=metric.step_size,
        reconstruction=resolve_reconstruction_spec(metric.hidden_acts_reconstruction),
    )
    pgd_step = make_fresh_pgd_eval_step(
        model,
        probe,
        ci_capture_keys,
        mesh,
        compiler_options=compiler_options,
    )

    def step(
        model: PlacedModel,
        components: ComponentStacks,
        placed_ci_fn: PlacedCIFn,
        inputs: Any,
        key: PRNGKeyArray,
    ) -> dict[str, Array]:
        return {f"loss/{probe.name}": pgd_step(model, components, placed_ci_fn, inputs, key)}

    return _averaged_over_eval_batches(step, eval_config, schedule, seed, model, sample_eval_batch)


def make_ci_l0_operation(
    metric: CI_L0Config,
    eval_config: EvalConfig,
    schedule: EvalSchedule,
    seed: int,
    compiler_options: dict[str, bool | int | str],
    model: PlacedModel,
    ci_capture_keys: CaptureKeys,
    mesh: Mesh | None,
    sample_eval_batch: Callable[[int], Any],
) -> PassOperation[EvalInvocation]:
    groups = (
        {name: tuple(patterns) for name, patterns in metric.groups.items()}
        if metric.groups is not None
        else None
    )
    step = make_ci_l0_eval_step(
        model,
        ci_capture_keys,
        metric.ci_alive_threshold,
        groups,
        mesh,
        compiler_options=compiler_options,
    )
    return _averaged_over_eval_batches(step, eval_config, schedule, seed, model, sample_eval_batch)
