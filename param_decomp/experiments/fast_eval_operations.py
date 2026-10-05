"""Target-generic fast-tier eval operations (`PGDReconLoss`, `CI_L0`).

The non-LM counterpart of `experiments/lm/scalar_eval_operations.py`: it binds the core
kernels — which are generic over both waist geometries, the target's input pytree and its
output/recon metric — to the ordinary `EvalInvocation` cadence. A target that samples its
own eval batches needs nothing beyond `sample_eval_batch`, whatever shape those batches
are; positioned non-categorical targets included.
"""

from collections.abc import Callable

import jax
import numpy as np
from jax.sharding import Mesh
from jaxtyping import Array, PRNGKeyArray

from param_decomp.core.ci_fn.interface import CIFn
from param_decomp.core.ci_l0_eval import make_ci_l0_eval_step
from param_decomp.core.components import ComponentStacks
from param_decomp.core.configs import (
    AnyPGDEvalConfig,
    CI_L0Config,
)
from param_decomp.core.eval_schedule import EvalSchedule
from param_decomp.core.model import CaptureKeys, ComponentActivations, PlacedModel
from param_decomp.core.recon_eval import fresh_pgd_probe, make_fresh_pgd_eval_step
from param_decomp.core.run import EvalInvocation, StandaloneOperation, StandaloneOperationPlan
from param_decomp.experiments.eval_config import EvalConfig

type ScalarStep[TargetIn, Out, PreparedT: ComponentActivations, Conditioning, PreparedMaskingT] = (
    Callable[
        [
            PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
            ComponentStacks,
            CIFn[Conditioning],
            TargetIn,
            PRNGKeyArray,
        ],
        dict[str, Array],
    ]
)


def _averaged_over_eval_batches[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    step: ScalarStep[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    eval_config: EvalConfig,
    schedule: EvalSchedule,
    seed: int,
    compiler_options: dict[str, bool | int | str],
    model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    sample_eval_batch: Callable[[np.uint32], TargetIn],
) -> StandaloneOperationPlan[EvalInvocation[Conditioning]]:
    """Run `step` over the pass's eval batches and average each scalar it emits."""
    eval_key = jax.random.PRNGKey(seed + 1)

    def score(
        model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
        components: ComponentStacks,
        ci_fn: CIFn[Conditioning],
        inputs: TargetIn,
        batch_index: np.uint32,
    ) -> dict[str, Array]:
        key = jax.random.fold_in(eval_key, batch_index)
        return step(model, components, ci_fn, inputs, key)

    def prepare(
        example: EvalInvocation[Conditioning],
    ) -> StandaloneOperation[EvalInvocation[Conditioning]]:
        compiled_step = (
            jax.jit(score, compiler_options=compiler_options)
            .lower(
                model,
                example.decomposition.components,
                example.decomposition.ci_fn,
                sample_eval_batch(np.uint32(0)),
                np.uint32(0),
            )
            .compile()
        )

        def run(context: EvalInvocation[Conditioning]) -> dict[str, float]:
            pass_index = context.now_step // eval_config.every
            sums: dict[str, float] = {}
            for batch_index in range(eval_config.n_steps):
                flat_index = np.uint32(pass_index * eval_config.n_steps + batch_index)
                values = compiled_step(
                    model,
                    context.decomposition.components,
                    context.decomposition.ci_fn,
                    sample_eval_batch(flat_index),
                    flat_index,
                )
                for name, value in values.items():
                    sums[name] = sums.get(name, 0.0) + float(value)
            return {f"eval/{name}": value / eval_config.n_steps for name, value in sums.items()}

        return StandaloneOperation(schedule, run)

    return StandaloneOperationPlan(prepare)


def make_fresh_pgd_operation[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    metric: AnyPGDEvalConfig,
    eval_config: EvalConfig,
    schedule: EvalSchedule,
    seed: int,
    compiler_options: dict[str, bool | int | str],
    model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    ci_capture_keys: CaptureKeys,
    mesh: Mesh | None,
    sample_eval_batch: Callable[[np.uint32], TargetIn],
) -> StandaloneOperationPlan[EvalInvocation[Conditioning]]:
    step = make_fresh_pgd_eval_step(model, fresh_pgd_probe(metric), ci_capture_keys, mesh)
    return _averaged_over_eval_batches(
        step, eval_config, schedule, seed, compiler_options, model, sample_eval_batch
    )


def make_ci_l0_operation[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    metric: CI_L0Config,
    eval_config: EvalConfig,
    schedule: EvalSchedule,
    seed: int,
    compiler_options: dict[str, bool | int | str],
    model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    ci_capture_keys: CaptureKeys,
    mesh: Mesh | None,
    sample_eval_batch: Callable[[np.uint32], TargetIn],
) -> StandaloneOperationPlan[EvalInvocation[Conditioning]]:
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
    )
    return _averaged_over_eval_batches(
        step, eval_config, schedule, seed, compiler_options, model, sample_eval_batch
    )
