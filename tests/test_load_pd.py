"""End-to-end test for the `run_pd` / `load_pd` round-trip with a minimal custom model."""

import tempfile
from pathlib import Path
from typing import Any, override

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from param_decomp import PDTarget, load_pd, run_pd
from param_decomp.configs import (
    CIMaskedReconLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    LayerwiseCiConfig,
    ModulePatternInfoConfig,
    PDConfig,
    ScheduleConfig,
)
from param_decomp.models.batch_and_loss_fns import recon_loss_mse


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _make_loader(*, batch_size: int = 4, n_batches: int = 4) -> DataLoader[Any]:
    inputs = torch.randn(batch_size * n_batches, 8)
    return DataLoader(TensorDataset(inputs), batch_size=batch_size, shuffle=False)


def _run_batch(model: nn.Module, batch: tuple[Tensor]) -> Tensor:
    return model(batch[0])


def test_run_and_load_round_trip(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "param_decomp.run_param_decomp.PARAM_DECOMP_OUT_DIR",
        Path(tempfile.mkdtemp()),
    )

    model = TinyModel().eval()
    for p in model.parameters():
        p.requires_grad_(False)

    config = PDConfig(
        seed=0,
        autocast_bf16=False,
        n_mask_samples=1,
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[4]),
        sigmoid_type="leaky_hard",
        module_info=[
            ModulePatternInfoConfig(module_pattern="fc1", C=4),
            ModulePatternInfoConfig(module_pattern="fc2", C=4),
        ],
        use_delta_component=True,
        loss_metric_configs=[
            FaithfulnessLossConfig(coeff=1.0),
            CIMaskedReconLossConfig(coeff=1.0),
            ImportanceMinimalityLossConfig(coeff=0.001, pnorm=2.0, beta=0.0),
        ],
        lr_schedule=ScheduleConfig(start_val=1e-3),
        steps=2,
        batch_size=4,
        eval_batch_size=4,
        train_log_freq=1,
        eval_freq=1,
        slow_eval_freq=1,
        n_eval_steps=1,
        slow_eval_on_first_step=False,
        ci_alive_threshold=0.0,
    )

    target = PDTarget(
        model=model,
        run_batch=_run_batch,
        reconstruction_loss=recon_loss_mse,
        name="custom",
    )

    out_dir = run_pd(
        config=config,
        target=target,
        train_loader=_make_loader(),
        eval_loader=_make_loader(),
        device="cpu",
    )

    assert out_dir is not None
    assert (out_dir / "pd_config.yaml").exists()
    # No experiment_config was passed, so experiment_config.yaml should NOT be saved.
    assert not (out_dir / "experiment_config.yaml").exists()

    # Reload via load_pd with a fresh target.
    fresh_model = TinyModel().eval()
    for p in fresh_model.parameters():
        p.requires_grad_(False)
    fresh_target = PDTarget(
        model=fresh_model,
        run_batch=_run_batch,
        reconstruction_loss=recon_loss_mse,
        name="custom",
    )
    reloaded = load_pd(out_dir, target=fresh_target)
    assert set(reloaded.target_module_paths) == {"fc1", "fc2"}
