"""Tests for target-generic evaluation operations bound to toy runs."""

from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from param_decomp.core.configs import WellTemperednessConfig
from param_decomp.core.model import CaptureKeys
from param_decomp.core.run import EvalInvocation, EvalOperation
from param_decomp.experiments import toy_eval
from param_decomp.experiments.eval_config import EvalConfig


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


def test_well_temperedness_uses_its_own_rng_domain_and_skips_untransported_figure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    ) -> EvalOperation[Any]:
        captured.update(
            metric=metric,
            model=model,
            ci_capture_keys=ci_capture_keys,
            mesh=mesh,
            compiler_options=compiler_options,
            inputs_for_context=inputs_for_context,
            figure_rendering=figure_rendering,
        )
        return EvalOperation(schedule, lambda _context: {})

    monkeypatch.setattr(toy_eval, "make_well_temperedness_operation", make_operation)
    sampled_indices: list[int] = []
    config = _eval_config()
    seed = 7

    operations = toy_eval.make_toy_evaluation_operations(
        config,
        seed,
        compiler_options={},
        model=cast(Any, SimpleNamespace(site_names=("site",))),
        ci_capture_keys=frozenset(),
        mesh=cast(Any, None),
        sample_eval_batch=lambda index: sampled_indices.append(index) or jnp.array([index]),
        probe_ci=cast(Any, None),
        wandb_configured=False,
    )
    inputs, key = captured["inputs_for_context"](
        EvalInvocation(state=cast(Any, None), now_step=20, placed_ci_fn=cast(Any, None))
    )

    assert len(operations) == 1
    assert captured["figure_rendering"] is None
    assert sampled_indices == [config.n_steps * 2]
    np.testing.assert_array_equal(inputs, jnp.array([config.n_steps * 2]))
    np.testing.assert_array_equal(key, jax.random.fold_in(jax.random.PRNGKey(seed + 2), 2))
    assert not np.array_equal(
        key,
        jax.random.fold_in(jax.random.PRNGKey(seed + 1), 2),
    )
