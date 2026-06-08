"""Sanity checks for the head-initialized PGD reconstruction loss."""

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
from param_decomp.metrics.adversarial_distribution_recon import AdversaryHeadOptimizerConfig
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.head_init_pgd_recon import (
    HeadInitPGDReconLoss,
    HeadInitPGDReconLossConfig,
)
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


def _cfg(**kw) -> HeadInitPGDReconLossConfig:
    base = dict(
        coeff=1.0,
        optimizer=AdversaryHeadOptimizerConfig(lr_schedule=ScheduleConfig(start_val=1e-2)),
        pgd_step_size=0.1,
        pgd_steps_min=2,
        pgd_steps_max=4,
        random_restart=True,
        head_hidden_dims=[16],
    )
    base.update(kw)
    return HeadInitPGDReconLossConfig(**base)  # pyright: ignore[reportArgumentType]


def _ctx(model: ComponentModel, batch: Tensor, *, is_eval: bool, use_delta_component: bool) -> MetricContext:
    cached = model(batch, cache_type="input")
    ci = model.calc_causal_importances(pre_weight_acts=cached.cache, sampling="continuous")
    return MetricContext(
        model=model,
        batch=batch,
        target_out=cached.output,
        pre_weight_acts=cached.cache,
        ci=ci,
        weight_deltas=model.calc_weight_deltas() if use_delta_component else {},
        step=0,
        total_steps=100,
        use_delta_component=use_delta_component,
        sampling="continuous",
        n_mask_samples=1,
        reconstruction_loss=recon_loss_mse,
        is_eval=is_eval,
    )


@pytest.mark.parametrize("use_delta_component", [True, False])
def test_attack_sources_in_unit_interval(use_delta_component: bool) -> None:
    model = _make_model()
    batch = torch.randn(5, 4)
    metric = HeadInitPGDReconLoss(_cfg())
    metric.bind(model=model, device="cpu")
    metric._ensure_state(_ctx(model, batch, is_eval=False, use_delta_component=use_delta_component))  # pyright: ignore[reportPrivateUsage]
    state = metric.state
    assert state is not None
    pre = model(batch, cache_type="input").cache
    ci = model.calc_causal_importances(pre_weight_acts=pre, sampling="continuous")
    wd = model.calc_weight_deltas() if use_delta_component else None
    s0 = state.predict_sources(pre)
    s_k, n_steps, _ = state.run_attack(
        model=model, batch=batch, ci=ci.lower_leaky, weight_deltas=wd,
        target_out=model.target_model(batch), head_sources=s0,
    )
    assert 2 <= n_steps <= 4
    for name in ("fc1", "fc2"):
        exp_c = C + (1 if use_delta_component else 0)
        assert s_k[name].shape == (5, exp_c)
        assert (s_k[name] >= 0).all() and (s_k[name] <= 1).all()


def test_defender_loss_trains_ci_not_head() -> None:
    """The returned loss backprops into CI fn/components; the head is trained separately."""
    model = _make_model()
    batch = torch.randn(3, 4)
    metric = HeadInitPGDReconLoss(_cfg())
    metric.bind(model=model, device="cpu")
    loss = metric.update(_ctx(model, batch, is_eval=False, use_delta_component=True))
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    ci_grads = [p.grad for p in model.ci_fn.parameters() if p.grad is not None]
    assert ci_grads and any((g != 0).any() for g in ci_grads), "defender loss should train the CI fn"
    # The head is NOT in the defender graph → no grad from the defender backward.
    assert metric.state is not None
    assert all(p.grad is None for p in metric.state.head.parameters())


def test_after_backward_steps_head_via_distillation() -> None:
    """after_backward distills the head toward the PGD endpoint (head params move)."""
    model = _make_model()
    batch = torch.randn(3, 4)
    metric = HeadInitPGDReconLoss(_cfg())
    metric.bind(model=model, device="cpu")
    loss = metric.update(_ctx(model, batch, is_eval=False, use_delta_component=True))
    assert loss is not None and metric.state is not None
    before = [p.detach().clone() for p in metric.state.head.parameters()]
    loss.backward()  # defender backward (must not free the head's distill graph)
    metric.after_backward()
    after = list(metric.state.head.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True)), "head should move"


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
    metric = HeadInitPGDReconLoss(_cfg())
    metric.bind(model=model, device="cpu")
    with pytest.raises(AssertionError, match="global_shared_transformer"):
        metric.update(_ctx(model, torch.randn(3, 4), is_eval=False, use_delta_component=True))


def test_eval_metric_keys() -> None:
    model = _make_model()
    batch = torch.randn(3, 4)
    metric = HeadInitPGDReconLoss(_cfg())
    metric.bind(model=model, device="cpu")
    metric.update(_ctx(model, batch, is_eval=True, use_delta_component=True))
    result = cast(dict[str, Tensor], metric.compute())
    cls = type(metric).__name__
    expected = {
        f"{cls}/hidden_acts", f"{cls}/hidden_acts/fc1", f"{cls}/hidden_acts/fc2",
        f"{cls}/output_recon", f"{cls}/head_distill_mse", f"{cls}/pgd_n_steps_mean",
        f"{cls}/random_restart_win_frac", f"{cls}/source_frac_saturated",
    }
    assert set(result) == expected
    for v in result.values():
        assert torch.isfinite(v)


def test_state_dict_roundtrip() -> None:
    model = _make_model()
    batch = torch.randn(3, 4)
    metric = HeadInitPGDReconLoss(_cfg())
    metric.bind(model=model, device="cpu")
    loss = metric.update(_ctx(model, batch, is_eval=False, use_delta_component=False))
    assert loss is not None
    loss.backward()
    metric.after_backward()
    saved = metric.state_dict()

    fresh = HeadInitPGDReconLoss(_cfg())
    fresh.bind(model=_make_model(), device="cpu")
    fresh.load_state_dict(saved)
    fresh.update(_ctx(fresh.model, batch, is_eval=True, use_delta_component=False))
    assert fresh.state is not None and metric.state is not None
    for a, b in zip(metric.state.head.parameters(), fresh.state.head.parameters(), strict=True):
        assert torch.equal(a, b)


class _NoOpSink:
    def log(self, metrics: dict[str, object], step: int) -> None: ...
    def console(self, *lines: str) -> None: ...
    def checkpoint(self, snapshot: object) -> None: ...
    def finish(self) -> None: ...


def test_trains_through_the_trainer() -> None:
    """End-to-end: defender descends via total_loss.backward(); head distills in after_backward."""
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
        loss_metrics=[_cfg()],
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
    metric = trainer.loss_metrics["HeadInitPGDReconLoss"]
    assert isinstance(metric, HeadInitPGDReconLoss)
    trainer.run(loader, _NoOpSink(), Cadence(train_log_every=1), eval_loop=None)

    assert metric.state is not None
    head_ids = {id(p) for p in metric.state.head.parameters()}
    assert not head_ids & {id(p) for p in trainer._ci_fn_params}  # pyright: ignore[reportPrivateUsage]
    assert not head_ids & {id(p) for p in trainer._component_params}  # pyright: ignore[reportPrivateUsage]
    assert trainer.snapshot().loss_metrics["HeadInitPGDReconLoss"]["head"]
