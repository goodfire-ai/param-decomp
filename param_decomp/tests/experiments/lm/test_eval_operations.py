"""Tests for target-generic evaluation operations bound to LM runs."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from param_decomp.core.configs import WellTemperednessConfig
from param_decomp.core.model import CaptureKeys
from param_decomp.core.run import PassOperation
from param_decomp.experiments.eval_config import EvalConfig
from param_decomp.experiments.lm import eval_operations
from param_decomp.experiments.lm.eval_context import LMEvalPass
from param_decomp.experiments.lm.eval_keys import EvalKeyStream


def _eval_config() -> EvalConfig:
    return EvalConfig(
        batch_size=8,
        n_steps=3,
        every=10,
        slow_every=20,
        metrics=[
            WellTemperednessConfig(
                groups=None,
                n_locations=2,
                n_components_per_region=4,
                ablations_per_forward=4,
            )
        ],
    )


def test_well_temperedness_uses_named_rng_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def make_operation(
        metric: WellTemperednessConfig,
        schedule: Any,
        model: Any,
        ci_capture_keys: CaptureKeys,
        mesh: Any,
        compiler_options: dict[str, bool | int | str],
        *,
        inputs_for_context: Any,
        figure_rendering: Any,
    ) -> PassOperation[Any]:
        captured.update(
            metric=metric,
            model=model,
            ci_capture_keys=ci_capture_keys,
            mesh=mesh,
            compiler_options=compiler_options,
            inputs_for_context=inputs_for_context,
            figure_rendering=figure_rendering,
        )
        return PassOperation(schedule, lambda _context: {})

    renderer = object()
    monkeypatch.setattr(eval_operations, "make_well_temperedness_operation", make_operation)
    monkeypatch.setattr(eval_operations, "BackgroundRenderer", lambda _sink: renderer)
    monkeypatch.setattr(eval_operations, "scan_shards", lambda _path: ())
    monkeypatch.setattr(eval_operations, "BatchSchedule", lambda *_args: object())
    monkeypatch.setattr(
        eval_operations,
        "read_dataset_meta",
        lambda _path: SimpleNamespace(seq_len=4),
    )
    monkeypatch.setattr(
        eval_operations,
        "ShardServer",
        lambda *_args: SimpleNamespace(per_process=jax.local_device_count()),
    )
    run_key = jax.random.PRNGKey(11)
    train_steps = 100
    built = SimpleNamespace(
        pd=SimpleNamespace(steps=train_steps, seed=3),
        data=SimpleNamespace(eval_dir=Path("unused")),
        ci_fn=SimpleNamespace(capture_keys=frozenset()),
        target=cast(Any, None),
    )

    evaluation = eval_operations.make_lm_evaluation(
        cast(Any, built),
        _eval_config(),
        cast(Any, SimpleNamespace(site_names=("site",), sites=())),
        run_key,
        cast(Any, None),
        n_proc=1,
        sink=cast(Any, SimpleNamespace(accepts_deferred_media=True)),
        compiler_options={},
    )
    batch = jnp.arange(4)
    _, key = captured["inputs_for_context"](
        LMEvalPass(
            state=cast(Any, None),
            now_step=30,
            placed_ci_fn=cast(Any, None),
            pass_index=3,
            batches=(batch,),
        )
    )

    assert len(evaluation.operations) == 1
    assert captured["figure_rendering"] is renderer
    assert captured["ci_capture_keys"] == frozenset()
    np.testing.assert_array_equal(
        key,
        jax.random.fold_in(
            run_key,
            EvalKeyStream.WELL_TEMPEREDNESS * train_steps + 3,
        ),
    )
