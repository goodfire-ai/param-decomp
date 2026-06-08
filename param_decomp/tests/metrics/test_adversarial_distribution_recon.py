"""Sanity checks for the adversarial-distribution reconstruction loss."""

from typing import cast

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from param_decomp.ci_fns import (
    AttnConfig,
    GlobalCiConfig,
    GlobalSharedTransformerCiConfig,
    LayerwiseCiConfig,
)
from param_decomp.component_model import ComponentModel
from param_decomp.configs import Cadence, OptimizerConfig, PDConfig, RuntimeConfig
from param_decomp.decomposition_targets import DecompositionTarget, DecompositionTargetConfig
from param_decomp.metrics.adversarial_distribution_recon import (
    AdversarialDistributionReconLoss,
    AdversarialDistributionReconLossConfig,
    AdversaryHeadOptimizerConfig,
    AdversaryHeadState,
)
from param_decomp.metrics.context import MetricContext
from param_decomp.optimize import Trainer
from param_decomp.schedule import ScheduleConfig
from param_decomp.tests.metrics.fixtures import TwoLayerLinearModel
from param_decomp_lab.batch_and_loss_fns import recon_loss_mse, run_batch_passthrough

C = 2


def _transformer_ci_config(d_model: int = 16) -> GlobalCiConfig:
    return GlobalCiConfig(
        mode="global",
        fn_type="global_shared_transformer",
        simple_transformer_ci_cfg=GlobalSharedTransformerCiConfig(
            d_model=d_model,
            n_blocks=2,
            mlp_hidden_dim=[2 * d_model],
            attn_config=AttnConfig(n_heads=4, max_len=8, rope_base=10000.0),
        ),
    )


def _make_model() -> ComponentModel:
    torch.manual_seed(0)
    target = TwoLayerLinearModel(d_in=4, d_hidden=3, d_out=4)
    target.requires_grad_(False)
    return ComponentModel(
        target_model=target,
        run_batch=run_batch_passthrough,
        decomposition_targets=[
            DecompositionTarget(module_path="fc1", C=C),
            DecompositionTarget(module_path="fc2", C=C),
        ],
        ci_config=_transformer_ci_config(),
        sigmoid_type="leaky_hard",
    )


def _cfg(
    distribution: str = "gaussian_sigmoid", start_frac: float = 0.0
) -> AdversarialDistributionReconLossConfig:
    return AdversarialDistributionReconLossConfig(
        coeff=1.0,
        optimizer=AdversaryHeadOptimizerConfig(lr_schedule=ScheduleConfig(start_val=0.1)),
        distribution=distribution,  # pyright: ignore[reportArgumentType]
        start_frac=start_frac,
    )


def _ctx(
    model: ComponentModel,
    batch: Tensor,
    *,
    is_eval: bool,
    use_delta_component: bool,
    step: int = 0,
    total_steps: int = 100,
) -> MetricContext:
    cached = model(batch, cache_type="input")
    ci = model.calc_causal_importances(pre_weight_acts=cached.cache, sampling="continuous")
    return MetricContext(
        model=model,
        batch=batch,
        target_out=cached.output,
        pre_weight_acts=cached.cache,
        ci=ci,
        weight_deltas=model.calc_weight_deltas() if use_delta_component else {},
        step=step,
        total_steps=total_steps,
        use_delta_component=use_delta_component,
        sampling="continuous",
        n_mask_samples=1,
        reconstruction_loss=recon_loss_mse,
        is_eval=is_eval,
    )


def _make_state(
    model: ComponentModel, *, distribution: str, use_delta_component: bool
) -> AdversaryHeadState:
    return AdversaryHeadState(
        model=model,
        device="cpu",
        use_delta_component=use_delta_component,
        distribution=distribution,  # pyright: ignore[reportArgumentType]
        optimizer_cfg=AdversaryHeadOptimizerConfig(lr_schedule=ScheduleConfig(start_val=0.1)),
        n_samples=1,
        reconstruction_loss=recon_loss_mse,
    )


@pytest.mark.parametrize("use_delta_component", [True, False])
@pytest.mark.parametrize("distribution", ["gaussian_sigmoid", "beta"])
def test_source_shapes_and_unit_interval(distribution: str, use_delta_component: bool) -> None:
    model = _make_model()
    batch = torch.randn(5, 4)
    state = _make_state(model, distribution=distribution, use_delta_component=use_delta_component)
    pre_weight_acts = model(batch, cache_type="input").cache
    _, sources = state.distribution_params_and_sources(pre_weight_acts)
    expected_c = C + (1 if use_delta_component else 0)
    for name in ("fc1", "fc2"):
        assert sources[name].shape == (5, expected_c)
        assert (sources[name] >= 0).all() and (sources[name] <= 1).all()


@pytest.mark.parametrize("distribution", ["gaussian_sigmoid", "beta"])
def test_reparam_grads_reach_head(distribution: str) -> None:
    """The reparameterized sample must carry gradient back to the head params."""
    model = _make_model()
    batch = torch.randn(3, 4)
    state = _make_state(model, distribution=distribution, use_delta_component=True)
    pre_weight_acts = model(batch, cache_type="input").cache
    _, sources = state.distribution_params_and_sources(pre_weight_acts)
    loss = sum((s**2).sum() for s in sources.values())
    assert isinstance(loss, Tensor) and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in state.head.parameters() if p.grad is not None]
    assert grads, "no grad reached the adversary head"
    assert any((g != 0).any() for g in grads), "adversary head grads are all zero"


def test_unsupported_layerwise_ci_fn_is_rejected() -> None:
    torch.manual_seed(0)
    target = TwoLayerLinearModel(d_in=4, d_hidden=3, d_out=4)
    target.requires_grad_(False)
    model = ComponentModel(
        target_model=target,
        run_batch=run_batch_passthrough,
        decomposition_targets=[DecompositionTarget(module_path="fc1", C=C)],
        ci_config=LayerwiseCiConfig(fn_type="shared_mlp", hidden_dims=[4]),
        sigmoid_type="leaky_hard",
    )
    with pytest.raises(AssertionError, match="global_shared_transformer"):
        _make_state(model, distribution="beta", use_delta_component=True)


def test_eval_metric_keys() -> None:
    torch.manual_seed(1)
    model = _make_model()
    batch = torch.randn(3, 4)

    metric = AdversarialDistributionReconLoss(_cfg("gaussian_sigmoid"))
    metric.bind(model=model, device="cpu")
    metric.update(_ctx(model, batch, is_eval=True, use_delta_component=True))

    result = cast(dict[str, Tensor], metric.compute())
    cls = type(metric).__name__
    expected = {
        f"{cls}/hidden_acts",
        f"{cls}/hidden_acts/fc1",
        f"{cls}/hidden_acts/fc2",
        f"{cls}/output_recon",
        f"{cls}/adv_params/source_frac_saturated",
    }
    for param in ("mu", "sigma", "source"):
        for stat in ("mean", "std", "min", "max"):
            expected.add(f"{cls}/adv_params/{param}/{stat}")
    assert set(result) == expected
    for v in result.values():
        assert torch.isfinite(v)


def test_ascend_moves_params_up_the_loss() -> None:
    """`ascend()` steps each head param in the +dL/dparam (ascent) direction.

    AdamW's first step is `param -= lr * sign(param.grad)`; with the grad negated for ascent
    that is `param += lr * sign(descent_grad)`, so the per-param delta matches the sign of the
    descent gradient the outer backward left behind.
    """
    torch.manual_seed(2)
    model = _make_model()
    batch = torch.randn(3, 4)

    metric = AdversarialDistributionReconLoss(_cfg("gaussian_sigmoid"))
    metric.bind(model=model, device="cpu")
    loss = metric.update(_ctx(model, batch, is_eval=False, use_delta_component=True))
    assert loss is not None and torch.isfinite(loss)
    assert metric.state is not None

    params = list(metric.state.head.parameters())
    before = [p.detach().clone() for p in params]
    loss.backward()
    descent_grads = [p.grad.detach().clone() for p in params if p.grad is not None]
    assert len(descent_grads) == len(params)
    assert any((g != 0).any() for g in descent_grads)

    metric.after_backward()  # negate + AdamW step

    for p, before_p, grad in zip(params, before, descent_grads, strict=True):
        delta = p.detach() - before_p
        moved = grad != 0
        assert torch.equal(torch.sign(delta[moved]), torch.sign(grad[moved]))


def test_ascend_increases_recon_loss_on_pinned_noise() -> None:
    """One adversary step raises the recon loss it is trained on (components/CI fn held fixed)."""
    torch.manual_seed(0)
    model = _make_model()
    batch = torch.randn(3, 4)
    ctx = _ctx(model, batch, is_eval=False, use_delta_component=True)

    metric = AdversarialDistributionReconLoss(_cfg("gaussian_sigmoid"))
    metric.bind(model=model, device="cpu")
    metric.update(ctx)  # build the head so the seeded RNG below only drives the source draw
    assert metric.state is not None
    metric.state.optimizer.zero_grad(set_to_none=True)

    seed = 1234
    torch.manual_seed(seed)
    loss_before = metric.update(ctx)
    assert loss_before is not None
    before = loss_before.item()
    loss_before.backward()
    metric.after_backward()

    torch.manual_seed(seed)  # identical source draw
    with torch.no_grad():
        loss_after = metric.update(ctx)
    assert loss_after is not None
    assert loss_after.item() > before


def test_state_dict_roundtrip() -> None:
    torch.manual_seed(3)
    model = _make_model()
    batch = torch.randn(3, 4)

    metric = AdversarialDistributionReconLoss(_cfg("beta"))
    metric.bind(model=model, device="cpu")
    loss = metric.update(_ctx(model, batch, is_eval=False, use_delta_component=False))
    assert loss is not None
    loss.backward()
    metric.after_backward()
    saved = metric.state_dict()

    fresh = AdversarialDistributionReconLoss(_cfg("beta"))
    fresh.bind(model=_make_model(), device="cpu")
    fresh.load_state_dict(saved)  # deferred until first update builds the head
    fresh.update(_ctx(fresh.model, batch, is_eval=True, use_delta_component=False))
    assert fresh.state is not None and metric.state is not None
    for a, b in zip(
        metric.state.head.parameters(), fresh.state.head.parameters(), strict=True
    ):
        assert torch.equal(a, b)


class _NoOpSink:
    def log(self, metrics: dict[str, object], step: int) -> None:
        del metrics, step

    def console(self, *lines: str) -> None:
        del lines

    def checkpoint(self, snapshot: object) -> None:
        del snapshot

    def finish(self) -> None:
        pass


def test_trains_through_the_trainer_and_steps_the_adversary() -> None:
    """End-to-end: the trainer descends components/CI fn while the adversary head ascends."""
    pd_config = PDConfig(
        seed=0,
        n_mask_samples=1,
        ci_config=_transformer_ci_config(d_model=8),
        decomposition_targets=[DecompositionTargetConfig(module_pattern="fc1", C=C)],
        components_optimizer=OptimizerConfig(lr_schedule=ScheduleConfig(start_val=1e-3)),
        ci_fn_optimizer=OptimizerConfig(lr_schedule=ScheduleConfig(start_val=1e-3)),
        steps=3,
        batch_size=2,
        use_delta_component=True,
        loss_metrics=[
            AdversarialDistributionReconLossConfig(
                coeff=1.0,
                optimizer=AdversaryHeadOptimizerConfig(
                    lr_schedule=ScheduleConfig(start_val=1e-2)
                ),
                distribution="gaussian_sigmoid",
            )
        ],
    )

    def run_batch_unpacking(model: object, batch: object) -> Tensor:
        if isinstance(batch, list | tuple):
            batch = batch[0]
        assert isinstance(batch, Tensor)
        out = model(batch)  # pyright: ignore[reportCallIssue]
        assert isinstance(out, Tensor)
        return out

    trainer = Trainer(
        target_model=TwoLayerLinearModel(d_in=2, d_hidden=2, d_out=2),
        run_batch=run_batch_unpacking,
        reconstruction_loss=recon_loss_mse,
        pd_config=pd_config,
        runtime_config=RuntimeConfig(device="cpu", autocast_bf16=False),
    )
    loader = DataLoader(TensorDataset(torch.randn(8, 2)), batch_size=2)

    metric = trainer.loss_metrics["AdversarialDistributionReconLoss"]
    assert isinstance(metric, AdversarialDistributionReconLoss)

    trainer.run(loader, _NoOpSink(), Cadence(train_log_every=1), eval_loop=None)

    assert metric.state is not None
    # The adversary head is owned by neither trainer optimizer.
    head_ids = {id(p) for p in metric.state.head.parameters()}
    assert not head_ids & {id(p) for p in trainer._ci_fn_params}  # pyright: ignore[reportPrivateUsage]
    assert not head_ids & {id(p) for p in trainer._component_params}  # pyright: ignore[reportPrivateUsage]

    # Snapshot must carry the adversary head state for resumption.
    snapshot = trainer.snapshot()
    assert snapshot.loss_metrics["AdversarialDistributionReconLoss"]["head"]


def test_dormant_before_start_frac() -> None:
    model = _make_model()
    batch = torch.randn(3, 4)

    metric = AdversarialDistributionReconLoss(_cfg("gaussian_sigmoid", start_frac=0.5))
    metric.bind(model=model, device="cpu")
    out = metric.update(_ctx(model, batch, is_eval=False, use_delta_component=False, step=10))
    assert out is None
    assert metric.state is None
    metric.after_backward()  # no-op, must not raise
