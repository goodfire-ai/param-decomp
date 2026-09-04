"""Tests for the well-temperedness evaluation operation."""

from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from param_decomp.core import well_temperedness_eval
from param_decomp.core.ci_fn import PlacedCIFn
from param_decomp.core.configs import WellTemperednessConfig
from param_decomp.core.eval_schedule import Every
from param_decomp.core.run import EvalInvocation
from param_decomp.core.well_temperedness import Ablations


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
    operation = well_temperedness_eval.make_well_temperedness_operation(
        metric,
        Every(1),
        cast(Any, SimpleNamespace(site_names=("site",))),
        frozenset(),
        mesh=None,
        compiler_options={},
        inputs_for_context=lambda _context: (jnp.zeros((1,)), jax.random.PRNGKey(0)),
        figure_rendering=None,
    )
    state = SimpleNamespace(decomposition=SimpleNamespace(components=object(), ci_fn=object()))

    record = operation.run(
        EvalInvocation(
            cast(Any, state),
            now_step=1,
            placed_ci_fn=PlacedCIFn(fn=cast(Any, None), placement=None),
        )
    )

    assert record
    assert all("figures" not in name for name in record)
