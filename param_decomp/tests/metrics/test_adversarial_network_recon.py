"""Sanity checks for the adversarial-network reconstruction loss."""

from typing import cast

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from param_decomp.ci_fns import (
    AttnConfig,
    GlobalCiConfig,
    GlobalCiFnWrapper,
    GlobalSharedTransformerCiConfig,
    LayerwiseCiConfig,
)
from param_decomp.component_model import CIOutputs, ComponentModel
from param_decomp.configs import Cadence, OptimizerConfig, PDConfig, RuntimeConfig
from param_decomp.decomposition_targets import DecompositionTarget, DecompositionTargetConfig
from param_decomp.metrics.adversarial_network_recon import (
    AdversarialNetworkReconLoss,
    AdversarialNetworkReconLossConfig,
    AdversaryNetworkState,
    AdversaryOptimizerConfig,
)
from param_decomp.metrics.context import MetricContext
from param_decomp.optimize import Trainer
from param_decomp.schedule import ScheduleConfig
from param_decomp.tests.metrics.fixtures import TwoLayerLinearModel
from param_decomp_lab.batch_and_loss_fns import recon_loss_mse, run_batch_passthrough

C = 2


def _make_model(fn_type: str = "shared_mlp") -> ComponentModel:
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
        ci_config=LayerwiseCiConfig(fn_type=fn_type, hidden_dims=[4]),  # pyright: ignore[reportArgumentType]
        sigmoid_type="leaky_hard",
    )


def _ci_outputs(ci: dict[str, Tensor]) -> CIOutputs:
    return CIOutputs(lower_leaky=ci, upper_leaky=ci, pre_sigmoid=dict(ci))


def _ctx(
    model: ComponentModel,
    batch: Tensor,
    ci: dict[str, Tensor],
    *,
    is_eval: bool,
    use_delta_component: bool,
    step: int = 0,
    total_steps: int = 100,
) -> MetricContext:
    return MetricContext(
        model=model,
        batch=batch,
        target_out=model.target_model(batch),
        pre_weight_acts={},
        ci=_ci_outputs(ci),
        weight_deltas=model.calc_weight_deltas() if use_delta_component else {},
        step=step,
        total_steps=total_steps,
        use_delta_component=use_delta_component,
        sampling="continuous",
        n_mask_samples=1,
        reconstruction_loss=recon_loss_mse,
        is_eval=is_eval,
    )


def _cfg(start_frac: float = 0.0) -> AdversarialNetworkReconLossConfig:
    return AdversarialNetworkReconLossConfig(
        coeff=1.0,
        optimizer=AdversaryOptimizerConfig(lr_schedule=ScheduleConfig(start_val=0.1)),
        start_frac=start_frac,
    )


@pytest.mark.parametrize("use_delta_component", [True, False])
def test_source_shapes_match_components_plus_delta(use_delta_component: bool) -> None:
    model = _make_model()
    state = AdversaryNetworkState(
        model=model,
        ci_config=model.ci_config,
        device="cpu",
        use_delta_component=use_delta_component,
        optimizer_cfg=AdversaryOptimizerConfig(lr_schedule=ScheduleConfig(start_val=0.1)),
        source_sigmoid="normal",
        input_source="noise",
        n_samples=1,
        reconstruction_loss=recon_loss_mse,
    )
    sources = state.generate_sources(pre_weight_acts={}, batch_dims=(5,))
    expected_c = C + (1 if use_delta_component else 0)
    for name in ("fc1", "fc2"):
        assert sources[name].shape == (5, expected_c)
        assert (sources[name] > 0).all() and (sources[name] < 1).all()


@pytest.mark.parametrize("source_sigmoid", ["normal", "lower_leaky_hard", "half_sin"])
def test_source_sigmoid_keeps_sources_in_unit_interval(source_sigmoid: str) -> None:
    model = _make_model()
    state = AdversaryNetworkState(
        model=model,
        ci_config=model.ci_config,
        device="cpu",
        use_delta_component=True,
        optimizer_cfg=AdversaryOptimizerConfig(lr_schedule=ScheduleConfig(start_val=0.1)),
        source_sigmoid=source_sigmoid,  # pyright: ignore[reportArgumentType]
        input_source="noise",
        n_samples=1,
        reconstruction_loss=recon_loss_mse,
    )
    # Push the network's pre-activations far positive/negative so a hard-clamp squashing
    # actually saturates, then check the sources are still valid mask sources in [0, 1].
    for p in state.network.parameters():
        torch.nn.init.normal_(p, std=50.0)
    sources = state.generate_sources(pre_weight_acts={}, batch_dims=(4,))
    for name in ("fc1", "fc2"):
        assert (sources[name] >= 0).all() and (sources[name] <= 1).all()


def test_hidden_acts_input_source_uses_activations_deterministically() -> None:
    """With `input_source="hidden_acts"` the adversary consumes the supplied acts (no RNG)."""
    model = _make_model(fn_type="shared_mlp")
    state = AdversaryNetworkState(
        model=model,
        ci_config=model.ci_config,
        device="cpu",
        use_delta_component=True,
        optimizer_cfg=AdversaryOptimizerConfig(lr_schedule=ScheduleConfig(start_val=0.1)),
        source_sigmoid="normal",
        input_source="hidden_acts",
        n_samples=1,
        reconstruction_loss=recon_loss_mse,
    )
    # fc1 input dim = 4, fc2 input dim = 3 (the target modules' d_in).
    acts = {"fc1": torch.randn(5, 4), "fc2": torch.randn(5, 3)}
    first = state.generate_sources(acts, batch_dims=(5,))
    second = state.generate_sources(acts, batch_dims=(5,))
    other_acts = {"fc1": torch.randn(5, 4), "fc2": torch.randn(5, 3)}
    on_other = state.generate_sources(other_acts, batch_dims=(5,))
    for name in ("fc1", "fc2"):
        assert first[name].shape == (5, C + 1)
        assert torch.equal(first[name], second[name])  # deterministic given fixed acts
        assert not torch.equal(first[name], on_other[name])  # but responds to the acts


def test_hidden_acts_input_source_runs_end_to_end() -> None:
    torch.manual_seed(0)
    model = _make_model(fn_type="shared_mlp")
    batch = torch.randn(3, 4)
    ci = {"fc1": torch.full((3, C), 0.5), "fc2": torch.full((3, C), 0.5)}
    pre_weight_acts = {"fc1": torch.randn(3, 4), "fc2": torch.randn(3, 3)}
    ctx = MetricContext(
        model=model,
        batch=batch,
        target_out=model.target_model(batch),
        pre_weight_acts=pre_weight_acts,
        ci=_ci_outputs(ci),
        weight_deltas=model.calc_weight_deltas(),
        step=0,
        total_steps=100,
        use_delta_component=True,
        sampling="continuous",
        n_mask_samples=1,
        reconstruction_loss=recon_loss_mse,
        is_eval=False,
    )
    metric = AdversarialNetworkReconLoss(
        AdversarialNetworkReconLossConfig(
            coeff=1.0,
            optimizer=AdversaryOptimizerConfig(lr_schedule=ScheduleConfig(start_val=0.1)),
            input_source="hidden_acts",
        )
    )
    metric.bind(model=model, device="cpu")
    loss = metric.update(ctx)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    metric.after_backward()


def test_mlp_ci_fn_type_is_rejected() -> None:
    model = _make_model(fn_type="mlp")
    with pytest.raises(AssertionError, match="per-component-scalar"):
        AdversaryNetworkState(
            model=model,
            ci_config=model.ci_config,
            device="cpu",
            use_delta_component=True,
            optimizer_cfg=AdversaryOptimizerConfig(lr_schedule=ScheduleConfig(start_val=0.1)),
            source_sigmoid="normal",
            input_source="noise",
            n_samples=1,
            reconstruction_loss=recon_loss_mse,
        )


def _transformer_override(d_model: int = 16) -> GlobalCiConfig:
    return GlobalCiConfig(
        fn_type="global_shared_transformer",
        simple_transformer_ci_cfg=GlobalSharedTransformerCiConfig(
            d_model=d_model,
            n_blocks=2,
            mlp_hidden_dim=[32],
            attn_config=AttnConfig(n_heads=4, max_len=8),
        ),
    )


def test_architecture_override_builds_independent_network() -> None:
    """A transformer override gives the adversary a different architecture than the CI fn."""
    model = _make_model(fn_type="shared_mlp")  # layerwise CI fn
    state = AdversaryNetworkState(
        model=model,
        ci_config=_transformer_override(),
        device="cpu",
        use_delta_component=True,
        optimizer_cfg=AdversaryOptimizerConfig(lr_schedule=ScheduleConfig(start_val=0.1)),
        source_sigmoid="normal",
        input_source="noise",
        n_samples=1,
        reconstruction_loss=recon_loss_mse,
    )
    assert isinstance(state.network, GlobalCiFnWrapper)
    sources = state.generate_sources(pre_weight_acts={}, batch_dims=(3,))
    for name in ("fc1", "fc2"):
        assert sources[name].shape == (3, C + 1)


def test_architecture_override_validates_against_unsupported_types() -> None:
    """The mlp/embedding checks key off the effective (override) config, not the model's."""
    model = _make_model(fn_type="shared_mlp")  # a supported model CI fn
    with pytest.raises(AssertionError, match="per-component-scalar"):
        AdversaryNetworkState(
            model=model,
            ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[2]),
            device="cpu",
            use_delta_component=True,
            optimizer_cfg=AdversaryOptimizerConfig(lr_schedule=ScheduleConfig(start_val=0.1)),
            source_sigmoid="normal",
            input_source="noise",
            n_samples=1,
            reconstruction_loss=recon_loss_mse,
        )


def test_architecture_override_round_trips_through_config() -> None:
    cfg = AdversarialNetworkReconLossConfig(
        coeff=1.0,
        optimizer=AdversaryOptimizerConfig(lr_schedule=ScheduleConfig(start_val=0.1)),
        architecture=_transformer_override(d_model=32),
    )
    reloaded = AdversarialNetworkReconLossConfig.model_validate(cfg.model_dump())
    assert isinstance(reloaded.architecture, GlobalCiConfig)
    assert reloaded.architecture.simple_transformer_ci_cfg is not None
    assert reloaded.architecture.simple_transformer_ci_cfg.d_model == 32


def test_eval_metric_keys() -> None:
    torch.manual_seed(1)
    model = _make_model()
    batch = torch.randn(3, 4)
    ci = {"fc1": torch.full((3, C), 0.6), "fc2": torch.full((3, C), 0.7)}

    metric = AdversarialNetworkReconLoss(_cfg())
    metric.bind(model=model, device="cpu")
    metric.update(_ctx(model, batch, ci, is_eval=True, use_delta_component=False))

    result = cast(dict[str, Tensor], metric.compute())
    cls = type(metric).__name__
    assert set(result) == {
        f"{cls}/hidden_acts",
        f"{cls}/hidden_acts/fc1",
        f"{cls}/hidden_acts/fc2",
        f"{cls}/output_recon",
    }
    for v in result.values():
        assert torch.isfinite(v) and v.item() >= 0


def test_ascend_moves_params_up_the_loss() -> None:
    """`ascend()` should step each adversary param in the +dL/dparam (ascent) direction.

    AdamW's first step is `param -= lr * sign(param.grad)`; with the grad negated for ascent
    that is `param += lr * sign(descent_grad)`, so the per-param delta matches the sign of the
    descent gradient the outer backward left behind.
    """
    torch.manual_seed(2)
    model = _make_model()
    batch = torch.randn(3, 4)
    ci = {"fc1": torch.full((3, C), 0.5), "fc2": torch.full((3, C), 0.5)}

    metric = AdversarialNetworkReconLoss(_cfg())
    metric.bind(model=model, device="cpu")

    loss = metric.update(_ctx(model, batch, ci, is_eval=False, use_delta_component=True))
    assert loss is not None and torch.isfinite(loss)
    assert metric.state is not None

    params = list(metric.state.network.parameters())
    before = [p.detach().clone() for p in params]
    loss.backward()
    descent_grads: list[Tensor] = []
    for p in params:
        assert p.grad is not None
        descent_grads.append(p.grad.detach().clone())
    assert any((g != 0).any() for g in descent_grads), "expected non-zero adversary grads"

    metric.after_backward()  # negates + AdamW step

    for p, before_p, grad in zip(params, before, descent_grads, strict=True):
        delta = p.detach() - before_p
        moved = grad != 0
        assert torch.equal(torch.sign(delta[moved]), torch.sign(grad[moved]))


def test_ascend_increases_recon_loss_on_pinned_noise() -> None:
    """End-to-end sign check: one adversary step raises the recon loss it's trained on.

    With the noise draw pinned (same seed) and components/CI fn held fixed, recomputing the
    loss after a single `after_backward()` must yield a *higher* value — the adversary
    ascends. A regression here would mean the adversary is helping reconstruction.
    """
    torch.manual_seed(0)
    model = _make_model()
    batch = torch.randn(3, 4)
    ci = {"fc1": torch.full((3, C), 0.5), "fc2": torch.full((3, C), 0.5)}
    ctx = _ctx(model, batch, ci, is_eval=False, use_delta_component=True)

    metric = AdversarialNetworkReconLoss(_cfg())
    metric.bind(model=model, device="cpu")
    metric.update(ctx)  # build the network so the seeded RNG below only drives the noise draw
    assert metric.state is not None
    metric.state.optimizer.zero_grad(set_to_none=True)

    seed = 1234
    torch.manual_seed(seed)
    loss_before = metric.update(ctx)
    assert loss_before is not None
    before = loss_before.item()
    loss_before.backward()
    metric.after_backward()

    torch.manual_seed(seed)  # identical noise draw
    with torch.no_grad():
        loss_after = metric.update(ctx)
    assert loss_after is not None
    assert loss_after.item() > before


def test_state_dict_roundtrip() -> None:
    torch.manual_seed(3)
    model = _make_model()
    batch = torch.randn(3, 4)
    ci = {"fc1": torch.full((3, C), 0.5), "fc2": torch.full((3, C), 0.5)}

    metric = AdversarialNetworkReconLoss(_cfg())
    metric.bind(model=model, device="cpu")
    loss = metric.update(_ctx(model, batch, ci, is_eval=False, use_delta_component=False))
    assert loss is not None
    loss.backward()
    metric.after_backward()
    saved = metric.state_dict()

    fresh = AdversarialNetworkReconLoss(_cfg())
    fresh.bind(model=_make_model(), device="cpu")
    fresh.load_state_dict(saved)  # deferred until first update builds the network
    fresh.update(_ctx(fresh.model, batch, ci, is_eval=True, use_delta_component=False))
    assert fresh.state is not None
    assert metric.state is not None

    for a, b in zip(
        metric.state.network.parameters(), fresh.state.network.parameters(), strict=True
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
    """End-to-end: the trainer descends components/CI fn while the adversary ascends."""
    pd_config = PDConfig(
        seed=0,
        n_mask_samples=1,
        ci_config=LayerwiseCiConfig(fn_type="shared_mlp", hidden_dims=[4]),
        decomposition_targets=[DecompositionTargetConfig(module_pattern="fc1", C=C)],
        components_optimizer=OptimizerConfig(lr_schedule=ScheduleConfig(start_val=1e-3)),
        ci_fn_optimizer=OptimizerConfig(lr_schedule=ScheduleConfig(start_val=1e-3)),
        steps=3,
        batch_size=2,
        use_delta_component=True,
        loss_metrics=[
            AdversarialNetworkReconLossConfig(
                coeff=1.0,
                optimizer=AdversaryOptimizerConfig(lr_schedule=ScheduleConfig(start_val=1e-2)),
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

    metric = trainer.loss_metrics["AdversarialNetworkReconLoss"]
    assert isinstance(metric, AdversarialNetworkReconLoss)

    trainer.run(loader, _NoOpSink(), Cadence(train_log_every=1), eval_loop=None)

    assert metric.state is not None
    # The adversary network is not owned by either trainer optimizer.
    adv_ids = {id(p) for p in metric.state.network.parameters()}
    assert not adv_ids & {id(p) for p in trainer._ci_fn_params}
    assert not adv_ids & {id(p) for p in trainer._component_params}

    # Snapshot must carry the adversary state for resumption.
    snapshot = trainer.snapshot()
    assert snapshot.loss_metrics["AdversarialNetworkReconLoss"]["network"]


def test_dormant_before_start_frac() -> None:
    model = _make_model()
    batch = torch.randn(3, 4)
    ci = {"fc1": torch.full((3, C), 0.5), "fc2": torch.full((3, C), 0.5)}

    metric = AdversarialNetworkReconLoss(_cfg(start_frac=0.5))
    metric.bind(model=model, device="cpu")
    out = metric.update(_ctx(model, batch, ci, is_eval=False, use_delta_component=False, step=10))
    assert out is None
    assert metric.state is None
    metric.after_backward()  # no-op, must not raise
