"""Tests for the well-temperedness evaluation operation."""

from dataclasses import replace

import jax
import numpy as np
import pytest

from param_decomp.core.eval_schedule import Every
from param_decomp.core.train import Decomposition
from param_decomp.experiments.lm import well_temperedness_eval
from param_decomp.experiments.lm.eval_config import WellTemperednessConfig
from param_decomp.experiments.lm.eval_context import LMEvalPass
from param_decomp.experiments.lm.eval_keys import EvalKeyStream
from param_decomp.experiments.lm.well_temperedness import (
    Ablations,
    make_well_temperedness_step,
    well_temperedness_log_entries,
)
from param_decomp.tests.experiments.lm.test_well_temperedness import _setup_glu_transformer


def test_disabled_figure_rendering_never_builds_a_png(monkeypatch: pytest.MonkeyPatch) -> None:
    preactivations = np.array([[[-2.0, -1.0]], [[0.2, 0.8]], [[1.0, 2.0]]], dtype=np.float32)
    ablations = Ablations(
        preactivations=preactivations,
        damage=np.abs(preactivations),
        site_indices=np.zeros_like(preactivations, dtype=np.int32),
    )
    monkeypatch.setattr(
        well_temperedness_eval,
        "make_well_temperedness_step",
        lambda *_args, **_kwargs: lambda *_step_args: ablations,
    )

    def unexpected_render(_ablations: Ablations) -> bytes:
        raise AssertionError("disabled figure rendering reached matplotlib")

    monkeypatch.setattr(well_temperedness_eval, "_plot_preactivation_vs_damage", unexpected_render)
    metric = WellTemperednessConfig(
        groups=None,
        n_locations=2,
        n_components_per_region=4,
        ablations_per_forward=4,
    )
    model, components, ci_fn, batch = _setup_glu_transformer()
    operation = well_temperedness_eval.make_well_temperedness_operation(
        metric,
        Every(1),
        model,
        frozenset(),
        mesh=None,
        compiler_options={},
        run_key=jax.random.PRNGKey(0),
        train_steps=10,
        figure_rendering=None,
    )
    decomposition = Decomposition(components, ci_fn)

    invocation = LMEvalPass(
        decomposition=decomposition,
        persistent_sources={},
        now_step=1,
        pass_index=1,
        batches=(batch,),
    )
    record = operation.prepare(invocation).run(invocation)

    assert record
    assert all("figures" not in name for name in record)


def test_compiled_measurement_preserves_sampling_stream() -> None:
    model, components, ci_fn, batch = _setup_glu_transformer()
    metric = WellTemperednessConfig(
        groups=None,
        n_locations=2,
        n_components_per_region=4,
        ablations_per_forward=4,
    )
    run_key = jax.random.PRNGKey(11)
    train_steps = 100
    invocation = LMEvalPass(
        decomposition=Decomposition(components, ci_fn),
        persistent_sources={},
        now_step=0,
        pass_index=0,
        batches=(batch,),
    )
    plan = well_temperedness_eval.make_well_temperedness_operation(
        metric,
        Every(10),
        model,
        ci_fn.capture_keys,
        mesh=None,
        compiler_options={},
        run_key=run_key,
        train_steps=train_steps,
        figure_rendering=None,
    )
    measure = jax.jit(make_well_temperedness_step(model, ci_fn.capture_keys, metric, None))
    operation = plan.prepare(invocation)
    for pass_index in (0, 3):
        expected_ablations = measure(
            model,
            components,
            ci_fn,
            batch,
            jax.random.fold_in(run_key, EvalKeyStream.WELL_TEMPEREDNESS * train_steps + pass_index),
        )
        expected = well_temperedness_log_entries(jax.device_get(expected_ablations), {})
        with jax.no_tracing():
            actual = operation.run(
                replace(invocation, now_step=pass_index * 10, pass_index=pass_index)
            )
        assert actual == {
            f"eval/slow/well_temperedness/{name}": value for name, value in expected.items()
        }
