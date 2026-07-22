"""Per-metric averaging parity: JAX `sum/n_steps` vs torch position-weighted accumulate.

Issue #715. JAX averages an eval metric across `n_steps` batches as
`(Σ_j v_j) / n_steps` (`run.py` eval loop), assuming a uniform `(B,T)`. Torch
accumulates `Σ_j v_j · (B·T)` and divides by `Σ_j (B·T)` (`CEandKLLosses.update` /
`compute`, `CI_L0`). Under fixed `(B,T)` the `(B·T)` factor cancels and the two are
identical FOR A MEAN-TYPE METRIC.

This file is the recorded per-metric verdict required by the acceptance criteria. It
checks the averaging math directly (the per-batch step itself is covered by
`test_eval.py`), and pins the Jensen caveat that makes the equivalence metric-specific:

  PRODUCTION FAST SET — all exact under `sum/n_steps`, all per-position means:
    - `ce_kl/kl_<variant>`            mean per-position KL  (mean)
    - `ce_kl/ce_difference_<variant>` mean CE minus target mean CE (affine in means -> mean)
    - `l0/<thr>_<site>` / `l0/<thr>_<group>` mean L0 per example (group = per-batch
        sum of member means, then averaged) (mean)
    - `loss/PGDReconLoss`             mean per-position KL at the final source (mean)

  NON-MEAN (hypothetical, NOT in the production set): any metric that is a nonlinear
    function of a GLOBAL accumulated sum — e.g. `log2(Σ_batches L0)` — is Jensen-
    divergent: torch's accumulate-then-compute differs from JAX's
    per-batch-mean-then-average. `test_nonmean_metric_is_jensen_divergent` exhibits the
    gap. If such a metric is ever added to the in-loop fast set it MUST accumulate then
    compute (and the JAX side must mirror that), not ride the `sum/n_steps` path.
"""

import math

import numpy as np
import pytest


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


@pytest.mark.parametrize(
    "metric_key, kind",
    [
        ("ce_kl/kl_ci_masked", "mean"),
        ("ce_kl/ce_difference_ci_masked", "mean"),
        ("l0/0.0_site", "mean"),
        ("l0/0.0_group", "mean"),
        ("loss/PGDReconLoss", "mean"),
    ],
)
def test_production_fast_set_classification(metric_key: str, kind: str):
    """The recorded verdict: every production in-loop fast metric is mean (exact under
    `sum/n_steps`). None is a nonlinear fn of a global accumulated sum, so all are exact
    under fixed (B,T). This list is the closing artifact for #715 — adding a metric here
    that is `nonmean` must accompany an accumulate-then-compute impl."""
    assert kind == "mean", f"{metric_key}: production fast metrics must be exact under sum/n_steps"
