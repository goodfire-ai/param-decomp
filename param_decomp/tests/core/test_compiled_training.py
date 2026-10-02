"""Training executes prepared native JAX executables as its state advances."""

from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from param_decomp.tests.core.test_no_bake_invariant import _build_step_and_args
from param_decomp.tests.experiments.tms.test_targeted_tms import (
    _loss_metrics,
    _stochastic_nontarget,
    _tiny_setup,
)


@pytest.mark.parametrize("kind", ["ordinary", "targeted"])
def test_compiled_training_advances_without_compilation(kind: Literal["ordinary", "targeted"]):
    match kind:
        case "ordinary":
            step, model, state, inputs = _build_step_and_args()
            batches = (inputs,)
        case "targeted":
            config, model, state, step = _tiny_setup(_loss_metrics(), _stochastic_nontarget())
            batches = (
                jnp.ones((16, config.n_features)),
                jnp.ones((32, config.n_features)),
            )
    keys = tuple(jax.random.PRNGKey(index) for index in range(3))
    step_batches = tuple(jax.tree.map(lambda value: value.copy(), batches) for _ in keys)
    arguments = (model, state, *step_batches[0], keys[0])
    compiled = (
        jax.jit(step, donate_argnums=tuple(range(1, len(arguments)))).lower(*arguments).compile()
    )
    events: list[str] = []
    metrics: dict[str, jax.Array] = {}

    def compilation_listener(
        event: str, start_time: float, end_time: float, **_metadata: Any
    ) -> None:
        del start_time, end_time
        if event == "/jax/core/compile/backend_compile_duration":
            events.append(event)

    jax.monitoring.register_event_time_span_listener(compilation_listener)
    try:
        for batch, key in zip(step_batches, keys, strict=True):
            state, metrics = compiled(model, state, *batch, key)
            jax.block_until_ready((state, metrics))
    finally:
        jax.monitoring.unregister_event_time_span_listener(compilation_listener)

    assert int(state.training.step) == len(keys)
    assert all(np.isfinite(np.asarray(value)).all() for value in metrics.values())
    assert not events
