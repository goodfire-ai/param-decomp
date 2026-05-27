import math
from typing import Any, override

import pytest
import torch
from pydantic import ValidationError
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from param_decomp.base_config import BaseConfig
from param_decomp.ci_fns import LayerwiseCiConfig
from param_decomp.configs import (
    AnyLossMetricConfig,
    Cadence,
    OptimizerConfig,
    PDConfig,
    RuntimeConfig,
)
from param_decomp.decomposition_targets import DecompositionTargetConfig
from param_decomp.metrics.base import Metric, MetricResult
from param_decomp.metrics.faithfulness import FaithfulnessLossConfig
from param_decomp.optimize import EvalLoop, Trainer
from param_decomp.schedule import ScheduleConfig


class TinyLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


def run_batch_passthrough(model: nn.Module, batch: Any) -> Tensor:
    if isinstance(batch, list | tuple):
        batch = batch[0]
    assert isinstance(batch, Tensor)
    out = model(batch)
    assert isinstance(out, Tensor)
    return out


def recon_loss_mse(pred: Tensor, target: Tensor) -> tuple[Tensor, int]:
    assert pred.shape == target.shape
    return ((pred - target) ** 2).sum(), pred.numel()


class CaptureSink:
    def __init__(self) -> None:
        self.logged: list[tuple[int, dict[str, Any]]] = []
        self.checkpoints: list[dict[str, Tensor]] = []

    def log(self, metrics: dict[str, Any], step: int) -> None:
        self.logged.append((step, dict(metrics)))

    def console(self, *lines: str) -> None:
        del lines

    def checkpoint(self, snapshot: Any) -> None:
        model_state = snapshot.component_model
        checkpoint: dict[str, Tensor] = {}
        for key, value in model_state.items():
            assert isinstance(value, Tensor)
            checkpoint[key] = value.detach().cpu().clone()
        self.checkpoints.append(checkpoint)

    def finish(self) -> None:
        pass


def make_cadence(*, train_log_every: int = 10**9) -> Cadence:
    """Default cadence for tests: nothing fires unless we explicitly set the freq."""
    return Cadence(train_log_every=train_log_every, save_every=None)


def make_eval_loop(
    loader: DataLoader[Any],
    *,
    metrics: list[Metric[Any]] | None = None,
    every: int = 10**9,
) -> EvalLoop:
    return EvalLoop(
        loader=loader,
        metrics=metrics if metrics is not None else [],
        n_steps=1,
        every=every,
        slow_every=every,
        slow_on_first_step=False,
    )


def make_pd_config(
    *, steps: int = 1, loss_metrics: list[AnyLossMetricConfig] | None = None
) -> PDConfig:
    if loss_metrics is None:
        loss_metrics = [FaithfulnessLossConfig(coeff=1.0)]
    return PDConfig(
        seed=123,
        n_mask_samples=1,
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[2]),
        decomposition_targets=[DecompositionTargetConfig(module_pattern="fc", C=2)],
        components_optimizer=OptimizerConfig(lr_schedule=ScheduleConfig(start_val=1e-3)),
        ci_fn_optimizer=OptimizerConfig(lr_schedule=ScheduleConfig(start_val=1e-3)),
        steps=steps,
        batch_size=2,
        loss_metrics=loss_metrics,
    )


def make_loader() -> DataLoader[Any]:
    return DataLoader(TensorDataset(torch.ones(4, 2)), batch_size=2)


def test_pd_config_requires_at_least_one_loss() -> None:
    with pytest.raises(
        ValidationError, match="loss_metrics must contain at least one training loss"
    ):
        make_pd_config(loss_metrics=[])


def test_pd_config_requires_positive_steps() -> None:
    with pytest.raises(ValidationError):
        make_pd_config(steps=0)


def test_optimize_logs_missing_grad_norms_as_nan() -> None:
    sink = CaptureSink()
    loader = make_loader()
    trainer = Trainer(
        target_model=TinyLinear(),
        run_batch=run_batch_passthrough,
        reconstruction_loss=recon_loss_mse,
        pd_config=make_pd_config(),
        runtime_config=RuntimeConfig(device="cpu", autocast_bf16=False),
    )
    trainer.run(loader, sink, make_cadence(train_log_every=1), eval_loop=None)

    train_logs = [
        metrics for _, metrics in sink.logged if any(k.startswith("train/") for k in metrics)
    ]
    assert len(train_logs) >= 1
    first = train_logs[0]
    assert any(
        key.startswith("train/grad_norms/ci_fns/") and math.isnan(value)
        for key, value in first.items()
    )
    assert math.isnan(first["train/grad_norms/summary/ci_fns"])
    assert math.isnan(first["train/grad_norms/summary/total"])


class DummyEvalConfig(BaseConfig):
    pass


class DummyEvalMetric(Metric[DummyEvalConfig]):
    log_namespace = "dummy"

    @override
    def reset(self) -> None:
        pass

    @override
    def update(self, ctx: Any) -> Tensor | None:
        del ctx
        return None

    @override
    def compute(self) -> MetricResult:
        return torch.tensor(0.0)


def test_optimize_rejects_duplicate_eval_metric_names() -> None:
    loader = make_loader()
    with pytest.raises(AssertionError, match="duplicate eval metric 'DummyEvalMetric'"):
        trainer = Trainer(
            target_model=TinyLinear(),
            run_batch=run_batch_passthrough,
            reconstruction_loss=recon_loss_mse,
            pd_config=make_pd_config(),
            runtime_config=RuntimeConfig(device="cpu", autocast_bf16=False),
        )
        trainer.run(
            loader,
            CaptureSink(),
            make_cadence(),
            eval_loop=make_eval_loop(
                loader,
                metrics=[DummyEvalMetric(DummyEvalConfig()), DummyEvalMetric(DummyEvalConfig())],
            ),
        )


def test_optimize_runs_without_eval_loop() -> None:
    sink = CaptureSink()
    loader = make_loader()
    trainer = Trainer(
        target_model=TinyLinear(),
        run_batch=run_batch_passthrough,
        reconstruction_loss=recon_loss_mse,
        pd_config=make_pd_config(steps=2),
        runtime_config=RuntimeConfig(device="cpu", autocast_bf16=False),
    )
    trainer.run(loader, sink, make_cadence(train_log_every=1), eval_loop=None)
    assert not any(any(k.startswith("eval/") for k in metrics) for _, metrics in sink.logged)


def test_optimize_seeds_component_model_construction() -> None:
    first = run_with_external_seed(0)
    second = run_with_external_seed(1)
    assert first.keys() == second.keys()
    for key in first:
        torch.testing.assert_close(first[key], second[key])


def run_with_external_seed(seed: int) -> dict[str, Tensor]:
    torch.manual_seed(seed)
    sink = CaptureSink()
    loader = make_loader()
    trainer = Trainer(
        target_model=TinyLinear(),
        run_batch=run_batch_passthrough,
        reconstruction_loss=recon_loss_mse,
        pd_config=make_pd_config(),
        runtime_config=RuntimeConfig(device="cpu", autocast_bf16=False),
    )
    trainer.run(loader, sink, make_cadence(), eval_loop=None)
    assert len(sink.checkpoints) == 1
    return sink.checkpoints[0]


def test_trainer_snapshot_round_trips() -> None:
    """A trainer reconstructed via ``Trainer.from_snapshot`` produces matching state."""
    pd_config = make_pd_config(steps=3)
    runtime_config = RuntimeConfig(device="cpu", autocast_bf16=False)

    trainer_a = Trainer(
        target_model=TinyLinear(),
        run_batch=run_batch_passthrough,
        reconstruction_loss=recon_loss_mse,
        pd_config=pd_config,
        runtime_config=runtime_config,
    )
    trainer_a.run(make_loader(), CaptureSink(), make_cadence(), eval_loop=None)
    snap = trainer_a.snapshot()

    trainer_b = Trainer.from_snapshot(
        snap,
        target_model=TinyLinear(),
        run_batch=run_batch_passthrough,
        reconstruction_loss=recon_loss_mse,
    )

    assert trainer_b.step == trainer_a.step
    sd_a = trainer_a.snapshot().component_model
    sd_b = trainer_b.snapshot().component_model
    assert sd_a.keys() == sd_b.keys()
    for k in sd_a:
        torch.testing.assert_close(sd_a[k], sd_b[k])

    opt_a = trainer_a.components_optimizer.state_dict()
    opt_b = trainer_b.components_optimizer.state_dict()
    assert opt_a["param_groups"] == opt_b["param_groups"]
    assert set(opt_a["state"].keys()) == set(opt_b["state"].keys())
    for pid in opt_a["state"]:
        for k, v in opt_a["state"][pid].items():
            if isinstance(v, Tensor):
                torch.testing.assert_close(v, opt_b["state"][pid][k])
            else:
                assert v == opt_b["state"][pid][k]


def test_trainer_resumes_from_snapshot_and_matches_uninterrupted_run() -> None:
    """Train K steps in one shot vs train K/2 → save → resume → train K/2;
    the final model weights should match up to RNG drift (we accept some, but
    on CPU with deterministic Adam the trajectory is bit-exact)."""
    pd_config = make_pd_config(steps=4)
    runtime_config = RuntimeConfig(device="cpu", autocast_bf16=False)

    torch.manual_seed(7)
    trainer_full = Trainer(
        target_model=TinyLinear(),
        run_batch=run_batch_passthrough,
        reconstruction_loss=recon_loss_mse,
        pd_config=pd_config,
        runtime_config=runtime_config,
    )
    trainer_full.run(make_loader(), CaptureSink(), make_cadence(), eval_loop=None)
    full_model = trainer_full.snapshot().component_model
    final_full = {k: v.clone() for k, v in full_model.items()}

    # Same fresh start, but save after step 2 and resume.
    torch.manual_seed(7)
    pd_config_half = make_pd_config(steps=2)
    trainer_half = Trainer(
        target_model=TinyLinear(),
        run_batch=run_batch_passthrough,
        reconstruction_loss=recon_loss_mse,
        pd_config=pd_config_half,
        runtime_config=runtime_config,
    )
    trainer_half.run(make_loader(), CaptureSink(), make_cadence(), eval_loop=None)
    snap = trainer_half.snapshot()

    # Resume — extend ``steps`` to 4 by mutating the saved pd_config dict.
    snap.pd_config["steps"] = 4
    trainer_resumed = Trainer.from_snapshot(
        snap,
        target_model=TinyLinear(),
        run_batch=run_batch_passthrough,
        reconstruction_loss=recon_loss_mse,
    )
    assert trainer_resumed.step == 2
    trainer_resumed.run(make_loader(), CaptureSink(), make_cadence(), eval_loop=None)

    resumed_model = trainer_resumed.snapshot().component_model
    assert final_full.keys() == resumed_model.keys()
    for k in final_full:
        torch.testing.assert_close(final_full[k], resumed_model[k])
