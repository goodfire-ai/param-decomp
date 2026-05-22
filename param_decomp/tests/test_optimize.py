import math
from typing import Any, override

import pytest
import torch
from pydantic import ValidationError
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from param_decomp.base_config import BaseConfig
from param_decomp.ci_fns import LayerwiseCiConfig
from param_decomp.configs import AnyLossMetricConfig, OptimizerConfig, PDConfig, RuntimeConfig
from param_decomp.decomposition_targets import DecompositionTargetConfig
from param_decomp.metrics.base import Metric, MetricResult
from param_decomp.metrics.faithfulness import FaithfulnessLossConfig
from param_decomp.optimize import optimize
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
    n_eval_steps = 1

    def __init__(self, *, log_train: bool = False) -> None:
        self.log_train = log_train
        self.logged: list[tuple[str | None, int, dict[str, Any]]] = []
        self.checkpoints: list[dict[str, Tensor]] = []

    def should_log_train(self, step: int) -> bool:
        return self.log_train and step == 0

    def should_eval(self, step: int) -> bool:
        del step
        return False

    def should_run_slow_eval(self, step: int) -> bool:
        del step
        return False

    def should_save(self, step: int, *, total_steps: int) -> bool:
        return step == total_steps

    def log(self, metrics: dict[str, Any], *, step: int, section: str | None = None) -> None:
        self.logged.append((section, step, dict(metrics)))

    def console(self, *lines: str) -> None:
        del lines

    def checkpoint(self, state_dict: dict[str, Any], *, step: int) -> None:
        del step
        checkpoint: dict[str, Tensor] = {}
        for key, value in state_dict.items():
            assert isinstance(value, Tensor)
            checkpoint[key] = value.detach().cpu().clone()
        self.checkpoints.append(checkpoint)


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
    sink = CaptureSink(log_train=True)
    loader = make_loader()
    optimize(
        target_model=TinyLinear(),
        train_loader=loader,
        eval_loader=loader,
        run_batch=run_batch_passthrough,
        reconstruction_loss=recon_loss_mse,
        pd_config=make_pd_config(),
        runtime_config=RuntimeConfig(device="cpu", autocast_bf16=False),
        sink=sink,
        eval_metrics=[],
    )

    train_logs = [metrics for section, _, metrics in sink.logged if section == "train"]
    assert len(train_logs) == 1
    assert any(
        key.startswith("grad_norms/ci_fns/") and math.isnan(value)
        for key, value in train_logs[0].items()
    )
    assert math.isnan(train_logs[0]["grad_norms/summary/ci_fns"])
    assert math.isnan(train_logs[0]["grad_norms/summary/total"])


class DummyEvalConfig(BaseConfig):
    pass


class DummyEvalMetric(Metric[DummyEvalConfig]):
    section = "dummy"
    config_type = DummyEvalConfig

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
        optimize(
            target_model=TinyLinear(),
            train_loader=loader,
            eval_loader=loader,
            run_batch=run_batch_passthrough,
            reconstruction_loss=recon_loss_mse,
            pd_config=make_pd_config(),
            runtime_config=RuntimeConfig(device="cpu", autocast_bf16=False),
            sink=CaptureSink(),
            eval_metrics=[DummyEvalMetric(DummyEvalConfig()), DummyEvalMetric(DummyEvalConfig())],
        )


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
    optimize(
        target_model=TinyLinear(),
        train_loader=loader,
        eval_loader=loader,
        run_batch=run_batch_passthrough,
        reconstruction_loss=recon_loss_mse,
        pd_config=make_pd_config(),
        runtime_config=RuntimeConfig(device="cpu", autocast_bf16=False),
        sink=sink,
        eval_metrics=[],
    )
    assert len(sink.checkpoints) == 1
    return sink.checkpoints[0]
