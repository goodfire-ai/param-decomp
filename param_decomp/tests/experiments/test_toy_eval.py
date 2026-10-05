"""Tests for the toy binding of authored evaluation operations."""

import io
from types import SimpleNamespace
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from PIL import Image

from param_decomp.core.ci_fn.implementations.layerwise_mlp import (
    LayerwiseMLPCIFnArch,
    init_layerwise_mlp_ci_fn,
)
from param_decomp.core.ci_fn.runtime import evaluate_ci_from_captures
from param_decomp.core.components import (
    ComponentStacks,
    SiteC,
    init_component_stacks,
    require_full_emission,
)
from param_decomp.core.configs import (
    CIHistogramsConfig,
    CIMeanPerComponentConfig,
    ComponentActivationDensityConfig,
)
from param_decomp.core.eval_schedule import eval_due
from param_decomp.core.metrics import PNGImage
from param_decomp.core.model import MaterializedMasking, PlacedModel
from param_decomp.core.run import EvalInvocation
from param_decomp.core.sharding import single_device_mesh
from param_decomp.core.slow_eval import plot_mean_component_cis_both_scales
from param_decomp.core.train import Decomposition
from param_decomp.experiments import toy_eval
from param_decomp.experiments.eval_config import EvalConfig
from param_decomp.experiments.lm.eval_config import CEandKLLossesConfig, WellTemperednessConfig
from param_decomp.targets import tms


@pytest.mark.parametrize(
    ("metric", "reason"),
    [
        (CEandKLLossesConfig(rounding_threshold=0.5), "neither tokens nor logits"),
        (
            WellTemperednessConfig(
                groups=None, n_locations=2, n_components_per_region=4, ablations_per_forward=4
            ),
            "positionless toy target has no positions",
        ),
        (CIHistogramsConfig(n_batches_accum=1), "has no toy binding"),
        (ComponentActivationDensityConfig(), "has no toy binding"),
    ],
)
def test_unsupported_metrics_refuse_at_toy_binding(metric: Any, reason: str) -> None:
    eval_config = EvalConfig(batch_size=8, n_steps=3, every=10, slow_every=20, metrics=[metric])
    with pytest.raises(AssertionError, match=reason):
        toy_eval.make_toy_evaluation_operations(
            eval_config,
            7,
            compiler_options={},
            model=cast(Any, SimpleNamespace(site_names=("site",))),
            ci_capture_keys=frozenset(),
            mesh=cast(Any, None),
            sample_eval_batch=lambda index: jnp.array([index]),
            probe_ci=cast(Any, None),
            wandb_configured=False,
        )


def test_mean_ci_requires_transport_before_sampling() -> None:
    config = EvalConfig(
        batch_size=8, n_steps=3, every=10, slow_every=20, metrics=[CIMeanPerComponentConfig()]
    )
    with pytest.raises(AssertionError, match="CIMeanPerComponent requires a configured wandb"):
        toy_eval.make_toy_evaluation_operations(
            config,
            7,
            compiler_options={},
            model=cast(Any, None),
            ci_capture_keys=frozenset(),
            mesh=cast(Any, None),
            sample_eval_batch=cast(Any, None),
            probe_ci=cast(Any, None),
            wandb_configured=False,
        )


def _toy_setup():
    mesh = single_device_mesh()
    cfg = tms.TMSConfig(n_features=5, n_hidden=2)
    sites = tms.site_specs(cfg, (SiteC("linear1", 8), SiteC("linear2", 6)))
    target = tms.init_tms_target(cfg, jax.random.PRNGKey(3))
    model = PlacedModel(model=tms.tms_decomposed_model(cfg, target, sites), placement=None)
    arch = LayerwiseMLPCIFnArch(
        hidden_dims=(16,),
        has_position_axis=False,
        input_names=tms.site_input_tap_keys(model.model.site_names),
    )
    decomposition: Decomposition[jax.Array] = Decomposition(
        components=init_component_stacks(sites, jax.random.PRNGKey(1)),
        ci_fn=init_layerwise_mlp_ci_fn(arch, sites, jax.random.PRNGKey(0)),
    )
    return mesh, cfg, model, decomposition


@pytest.mark.parametrize("on_first_step", [False, True])
def test_mean_ci_uses_slow_schedule(on_first_step: bool) -> None:
    mesh, cfg, model, decomposition = _toy_setup()
    config = EvalConfig(
        batch_size=2,
        n_steps=3,
        every=10,
        slow_every=20,
        slow_on_first_step=on_first_step,
        metrics=[CIMeanPerComponentConfig()],
    )
    (plan,) = toy_eval.make_toy_evaluation_operations(
        config,
        7,
        compiler_options={},
        model=model,
        ci_capture_keys=decomposition.ci_fn.capture_keys,
        mesh=mesh,
        sample_eval_batch=lambda _index: jnp.zeros((2, cfg.n_features)),
        probe_ci=cast(Any, None),
        wandb_configured=True,
    )
    with jax.set_mesh(mesh):
        operation = plan.prepare(EvalInvocation(decomposition, {}, 0))
    assert eval_due(operation.schedule, 0) is on_first_step
    assert not eval_due(operation.schedule, 10)
    assert eval_due(operation.schedule, 20)
    assert not eval_due(operation.schedule, 21)


def test_mean_ci_averages_examples_resets_each_pass_and_renders(monkeypatch: pytest.MonkeyPatch):
    """Exercise the prepared toy forward/CI path; derive means independently in numpy.

    The plan compiles at the fixed eval-batch shape. A second all-zero pass catches
    retained accumulators. The single-feature probe must not run.
    """
    mesh, cfg, model, decomposition = _toy_setup()
    batches = [
        jnp.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0, 0.0]]),
        jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]),
        jnp.array([[0.0, 0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.75, 0.0, 0.0]]),
    ]
    sampled: list[int] = []

    def sample(index: np.uint32) -> jax.Array:
        sampled.append(int(index))
        if index == 0:
            return batches[0]
        return batches[int(index) - 6] if index < 12 else jnp.zeros((2, cfg.n_features))

    def forbidden_probe(_state: Any) -> dict[str, jax.Array]:
        raise AssertionError("mean-CI must sample held-out data, not the geometry probe")

    rendered: list[dict[str, np.ndarray]] = []

    def capture_means(means: dict[str, np.ndarray]) -> tuple[bytes, bytes]:
        rendered.append(means)
        return plot_mean_component_cis_both_scales(means)

    monkeypatch.setattr(toy_eval, "plot_mean_component_cis_both_scales", capture_means)
    config = EvalConfig(
        batch_size=2, n_steps=3, every=10, slow_every=20, metrics=[CIMeanPerComponentConfig()]
    )
    (plan,) = toy_eval.make_toy_evaluation_operations(
        config,
        7,
        compiler_options={},
        model=model,
        ci_capture_keys=decomposition.ci_fn.capture_keys,
        mesh=mesh,
        sample_eval_batch=sample,
        probe_ci=forbidden_probe,
        wandb_configured=True,
    )

    @eqx.filter_jit
    def reference_preactivations(
        model: PlacedModel[jax.Array, jax.Array, ComponentStacks, jax.Array, MaterializedMasking],
        decomposition: Decomposition[jax.Array],
        x: jax.Array,
    ):
        clean = model.clean_forward(x, decomposition.ci_fn.capture_keys)
        return evaluate_ci_from_captures(
            decomposition.ci_fn.prepare(),
            clean.captures,
            clean.conditioning,
            model.prepare_compute_weights(decomposition.components),
            sequence=clean.sequence,
            remat=False,
        ).preactivations

    with jax.set_mesh(mesh):
        expected_batches = [reference_preactivations(model, decomposition, x) for x in batches]
        operation = plan.prepare(EvalInvocation(decomposition, {}, 0))
        assert sampled == [0]
        record = operation.run(EvalInvocation(decomposition, {}, 20))
        assert sampled == [0, 6, 7, 8]
        for site in model.model.site_names:
            lowers = [
                np.clip(np.asarray(require_full_emission(batch[site]), dtype=np.float32), 0, 1)
                for batch in expected_batches
            ]
            expected = np.concatenate(lowers).mean(0)
            np.testing.assert_allclose(rendered[0][site], expected, rtol=1e-6, atol=1e-7)
            assert np.any(expected > 0)

        operation.run(EvalInvocation(decomposition, {}, 40))

    assert sampled == [0, 6, 7, 8, 12, 13, 14]
    assert all(np.array_equal(means, np.zeros_like(means)) for means in rendered[1].values())
    assert set(record) == {
        "slow_eval/figures/ci_mean_per_component",
        "slow_eval/figures/ci_mean_per_component_log",
    }
    for image in record.values():
        assert isinstance(image, PNGImage)
        with Image.open(io.BytesIO(image.encoded)) as png:
            assert png.format == "PNG"
            assert png.width > 0 and png.height > 0
            png.verify()
