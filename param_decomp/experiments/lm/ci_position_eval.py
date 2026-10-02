"""`CI_L0`'s active-component count at every context position, as one native W&B chart."""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from param_decomp.core.ci_l0_eval import alive_component_counts
from param_decomp.core.components import SiteCI
from param_decomp.core.eval_schedule import EvalSchedule
from param_decomp.core.metrics import LineChart, LogRecord
from param_decomp.core.model import ComponentActivations
from param_decomp.core.run import (
    SharedForwardOperation,
    SharedForwardOperationPlan,
    shared_forward_operation,
)
from param_decomp.experiments.lm.eval_context import LMBatchContext, LMEvalPass
from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments
from param_decomp.lm.inputs import input_token_ids

CI_POSITION_THRESHOLDS = (0.0, 0.01, 0.1)
CI_POSITION_CHART_KEY = "eval/l0/per_position"


def position_active_counts(lower: Mapping[str, SiteCI]) -> Array:
    """Sum active entries over sites, components and sequences, retaining position."""
    return jnp.stack(
        [
            jnp.stack([alive_component_counts(ci, t).sum(0) for t in CI_POSITION_THRESHOLDS])
            for ci in lower.values()
        ]
    ).sum(0)


def make_ci_position_counts_operation[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
](
    schedule: EvalSchedule, compiler_options: dict[str, bool | int | str]
) -> SharedForwardOperationPlan[
    LMEvalPass[TargetIn, Conditioning], LMBatchContext[TargetIn, PreparedT, Conditioning]
]:
    def prepare(
        _example_pass: LMEvalPass[TargetIn, Conditioning],
        example_context: LMBatchContext[TargetIn, PreparedT, Conditioning],
    ) -> SharedForwardOperation[
        LMEvalPass[TargetIn, Conditioning], LMBatchContext[TargetIn, PreparedT, Conditioning]
    ]:
        step = (
            jax.jit(position_active_counts, compiler_options=compiler_options)
            .lower(example_context.forward.ci.lower)
            .compile()
        )

        def update(
            total: tuple[np.ndarray, int] | None,
            context: LMBatchContext[TargetIn, PreparedT, Conditioning],
        ) -> tuple[np.ndarray, int]:
            counts = np.asarray(step(context.forward.ci.lower), np.float64)
            rows = input_token_ids(context.forward.tokens).shape[0]
            return (counts, rows) if total is None else (total[0] + counts, total[1] + rows)

        def finish(
            eval_pass: LMEvalPass[TargetIn, Conditioning], total: tuple[np.ndarray, int] | None
        ) -> LogRecord:
            del eval_pass
            assert total is not None
            sums, rows = total
            return {
                CI_POSITION_CHART_KEY: LineChart(
                    xs=np.arange(sums.shape[1]),
                    series=tuple(
                        (f"CI > {t:g}", means)
                        for t, means in zip(CI_POSITION_THRESHOLDS, sums / rows, strict=True)
                    ),
                    x_label="Sequence position",
                    title="Mean active CI entries per token (all sites)",
                )
            }

        return shared_forward_operation(schedule, lambda: None, update, finish)

    return SharedForwardOperationPlan(prepare)
