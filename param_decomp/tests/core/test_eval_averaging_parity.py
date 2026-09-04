"""Per-metric averaging parity: JAX `sum/n_steps` vs torch position-weighted accumulate.

Issue #715. JAX averages an eval metric across `n_steps` batches as
`(Σ_j v_j) / n_steps` (`run.py` eval loop), assuming a uniform `(B,T)`. Torch
accumulates `Σ_j v_j · (B·T)` and divides by `Σ_j (B·T)` (`CEandKLLosses.update` /
`compute`, `CI_L0`). Under fixed `(B,T)` the `(B·T)` factor cancels and the two are
identical FOR A MEAN-TYPE METRIC.

This file is the recorded per-metric verdict required by the acceptance criteria. It
checks the averaging math directly (the per-batch step itself is covered by
`test_eval.py`), and pins the Jensen caveat that makes the equivalence metric-specific:

  PRODUCTION FAST SET:
    - `ce_kl/kl_<variant>`            mean per-position KL  (mean)
    - `ce_kl/ce_difference_<variant>` mean CE minus target mean CE (affine in means -> mean)
    - `l0/<thr>_<site>` / `l0/<thr>_<group>` mean L0 per example (group = per-batch
        sum of member means, then averaged) (mean)
    - `loss/PGDReconLoss[/e2e]`       mean per-position KL at the final source when the
        probe has no residual auxiliary (mean)
    - `loss/PGDReconLoss/hidden_acts_reconstruction[/<point>]` and the combined loss when configured:
        the same per-batch energy ratio/objective used by training, then averaged over eval
        batches (mean-of-batch-objectives, deliberately not a globally pooled energy ratio)

  Any other metric that is a nonlinear function of a GLOBAL accumulated sum — e.g.
    `log2(Σ_batches L0)` — is Jensen-divergent: torch's accumulate-then-compute differs
    from JAX's per-batch-mean-then-average. `test_nonmean_metric_is_jensen_divergent`
    exhibits the gap. If such a metric is ever added to the in-loop fast set it MUST
    either accumulate then compute or explicitly specify batch-objective semantics like
    hidden-activation reconstruction.
"""

import math
from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PRNGKeyArray

from param_decomp.core.ci_fn import CI, PlacedCIFn
from param_decomp.core.components import ComponentStacks
from param_decomp.core.eval_schedule import Every
from param_decomp.core.model import PlacedModel
from param_decomp.core.run import EvalInvocation
from param_decomp.experiments.eval_config import EvalConfig
from param_decomp.experiments.fast_eval_operations import _averaged_over_eval_batches
from param_decomp.experiments.lm.eval import PreparedLMBatch
from param_decomp.experiments.lm.eval_context import LMBatchContext, LMEvalPass
from param_decomp.experiments.lm.scalar_eval_operations import _make_scalar_operation


def _jax_average(per_batch_values: list[float], n_steps: int) -> float:
    """`run.py`: metric_sums accumulate then divide by n_steps."""
    return sum(per_batch_values) / n_steps


def _torch_position_weighted_average(
    per_batch_values: list[float], positions_per_batch: list[int]
) -> float:
    """`CEandKLLosses`: Σ_j v_j·n_j  /  Σ_j n_j  (== `CI_L0`'s sum/count with n_j==1)."""
    weighted_sum = sum(v * n for v, n in zip(per_batch_values, positions_per_batch, strict=True))
    return weighted_sum / sum(positions_per_batch)


def test_mean_metric_averaging_exact_under_fixed_bt():
    """Fixed (B,T): JAX sum/n_steps == torch position-weighted, exactly, for a mean
    metric (kl / ce_difference / l0 / pgd all share this accumulator shape)."""
    rng = np.random.default_rng(0)
    n_steps = 7
    per_batch = rng.normal(size=n_steps).tolist()
    fixed_positions = [B_T := 4 * 16] * n_steps

    jax_avg = _jax_average(per_batch, n_steps)
    torch_avg = _torch_position_weighted_average(per_batch, fixed_positions)
    assert math.isclose(jax_avg, torch_avg, rel_tol=1e-12, abs_tol=0.0)
    assert B_T  # silence unused; documents that the constant cancels


def test_mean_metric_averaging_diverges_under_ragged_bt():
    """The equivalence is LOAD-BEARING on uniform (B,T): if batches carried different
    position counts the two averages part ways. Production holds (B,T) fixed, so this is
    a guard documenting WHY, not a supported regime."""
    per_batch = [1.0, 3.0]
    ragged_positions = [1, 99]  # if torch ever saw ragged batches
    jax_avg = _jax_average(per_batch, n_steps=2)  # 2.0
    torch_avg = _torch_position_weighted_average(per_batch, ragged_positions)  # ~2.98
    assert not math.isclose(jax_avg, torch_avg, rel_tol=1e-3)


def test_nonmean_metric_is_jensen_divergent():
    """A metric that is a nonlinear fn of a GLOBAL accumulated sum (the S8-class caveat,
    e.g. `log2(Σ_batches L0)`) is NOT exact under `sum/n_steps`. Exhibits the gap so a
    future addition can't silently ride the mean path: torch's accumulate-then-compute
    (log2 of the global sum) differs from JAX's per-batch-mean-then-average
    (mean of per-batch log2s)."""
    per_batch_l0 = [3.0, 12.0, 48.0, 6.0]
    accumulate_then_compute = math.log2(sum(per_batch_l0))
    mean_of_per_batch_compute = sum(math.log2(x) for x in per_batch_l0) / len(per_batch_l0)
    assert not math.isclose(accumulate_then_compute, mean_of_per_batch_compute, rel_tol=1e-3)


def test_hidden_acts_reconstruction_averages_batch_objectives_instead_of_pooling_energy():
    """Hidden-activation reconstruction deliberately evaluates the training objective on each batch, then
    averages those ratios. Pooling numerator/denominator across eval batches is a different
    weighting (batches with more clean energy count more) and must not be substituted."""
    numerators = [1.0, 9.0]
    denominators = [1.0, 3.0]
    mean_of_batch_ratios = sum(n / d for n, d in zip(numerators, denominators, strict=True)) / 2
    globally_pooled_ratio = sum(numerators) / sum(denominators)
    assert mean_of_batch_ratios == 2.0
    assert globally_pooled_ratio == 2.5


def _state_stub() -> SimpleNamespace:
    return SimpleNamespace(decomposition=SimpleNamespace(components=None, ci_fn=None))


def _value_context(batch_index: int, value: float) -> LMBatchContext:
    return LMBatchContext(
        pass_index=0,
        batch_index=batch_index,
        tokens=jnp.asarray(value),
        clean_output=jnp.asarray(value),
        captures={},
        ci=CI(preactivations={}, lower={}, upper={}),
        prepared_weights=None,
    )


def test_lm_scalar_operation_averages_residual_batch_objectives():
    def scorer(
        _model: PlacedModel,
        batch: PreparedLMBatch[Any],
        _key: PRNGKeyArray,
    ) -> dict[str, Array]:
        return {"loss/probe/hidden_acts_reconstruction": batch.tokens}

    operation = _make_scalar_operation(
        Every(1),
        scorer,
        ("loss/probe/",),
        cast(Any, object()),
        jnp.array([0, 0], dtype=jnp.uint32),
        train_steps=0,
        eval_steps=2,
        compiler_options={},
    )
    state = operation.init()
    state = operation.update(state, _value_context(0, 1.0))
    state = operation.update(state, _value_context(1, 3.0))
    eval_pass = LMEvalPass(
        state=cast(Any, _state_stub()),
        now_step=0,
        placed_ci_fn=PlacedCIFn(fn=None, placement=None),  # pyright: ignore[reportArgumentType]
        pass_index=0,
        batches=(),
    )
    record = operation.finish(eval_pass, state)
    assert record["eval/loss/probe/hidden_acts_reconstruction"] == 2.0


def test_generic_scalar_operation_averages_residual_batch_objectives():
    def step(
        _model: PlacedModel,
        _components: ComponentStacks,
        _placed_ci_fn: PlacedCIFn,
        value: jax.Array,
        _key: PRNGKeyArray,
    ) -> dict[str, Array]:
        return {"loss/probe/hidden_acts_reconstruction": value}

    eval_config = EvalConfig(batch_size=1, n_steps=2, every=1, slow_every=1)
    operation = _averaged_over_eval_batches(
        step,
        eval_config,
        Every(1),
        seed=0,
        model=object(),  # pyright: ignore[reportArgumentType]
        sample_eval_batch=lambda index: jnp.asarray((1.0, 3.0)[index]),
    )
    context = EvalInvocation(
        state=_state_stub(),  # pyright: ignore[reportArgumentType]
        now_step=0,
        placed_ci_fn=PlacedCIFn(fn=None, placement=None),  # pyright: ignore[reportArgumentType]
    )
    assert operation.run(context)["eval/loss/probe/hidden_acts_reconstruction"] == 2.0
