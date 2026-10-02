"""Whole-context counts: numerical meaning across batches, and the sink's native transport."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest
import wandb

from param_decomp.core.components import SelectedCI
from param_decomp.core.eval_schedule import Every
from param_decomp.core.metrics import LineChart
from param_decomp.core.run import MetricsSink
from param_decomp.experiments.lm.ci_position_eval import (
    CI_POSITION_CHART_KEY,
    make_ci_position_counts_operation,
)
from param_decomp.lm.batch import LMBatch


def test_counts_sum_sites_and_components_and_average_rows_across_batches() -> None:
    # Dense site (rows=4, T=2, C=2); selected site picks blocks 0 and 2 of 4 (c=1) at CI 0.1,
    # so its two unselected entries per token are exact zeros.
    dense = jnp.array(
        [
            [[0.0, 0.01], [0.1, 0.2]],
            [[0.11, 0.0], [0.3, 0.4]],
            [[0.2, 0.3], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]
    )
    picks = jnp.broadcast_to(jnp.array([0, 2]), (4, 2, 2))
    plan = make_ci_position_counts_operation(Every(20), {})
    contexts = []
    for rows in (slice(0, 2), slice(2, 4)):  # two fixed-size batches, as the eval pass feeds them
        lower = {
            "dense": dense[rows],
            "selected": SelectedCI(jnp.full(picks[rows].shape, 0.1), picks[rows], n_blocks=4),
        }
        tokens = LMBatch(jnp.zeros(dense[rows].shape[:2], jnp.int32))
        context = cast(
            Any,
            SimpleNamespace(
                forward=SimpleNamespace(ci=SimpleNamespace(lower=lower), tokens=tokens)
            ),
        )
        contexts.append(context)
    operation = plan.prepare(cast(Any, None), contexts[0])
    total = operation.init()
    for context in contexts:
        total = operation.update(total, context)
    chart = operation.finish(cast(Any, None), total)[CI_POSITION_CHART_KEY]
    assert isinstance(chart, LineChart)
    np.testing.assert_array_equal(chart.xs, [0, 1])
    assert [name for name, _ in chart.series] == ["CI > 0", "CI > 0.01", "CI > 0.1"]
    # dense: >0 [4,4], >0.01 [3,4], >0.1 [3,3]; selected: two 0.1 picks per token -> [8,8],[8,8],[0,0].
    np.testing.assert_allclose(
        [ys for _, ys in chart.series], np.array([[12, 12], [11, 12], [3, 3]]) / 4
    )


def test_sink_logs_a_native_line_series_with_every_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads: list[dict[str, Any]] = []
    monkeypatch.setattr(wandb, "log", lambda payload, *, step, commit: payloads.append(payload))
    positions = np.arange(4096)
    chart = LineChart(positions, (("a", positions * 2.0), ("b", positions * 3.0)), "x", "t")
    path = tmp_path / "metrics.jsonl"
    with path.open("w") as stream:
        MetricsSink(stream, wandb, "legacy").log(20, {"chart": chart, "eval/loss": 1.0})
    assert json.loads(path.read_text()) == {"step": 20, "eval/loss": 1.0}
    table = payloads[0]["chart"].table
    assert table.columns == ["step", "lineKey", "lineVal"]
    assert table.data == [
        [x, name, y]
        for name, ys in chart.series
        for x, y in zip(positions.tolist(), ys.tolist(), strict=True)
    ]
