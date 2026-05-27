"""Lab-side resumption integration: canonical training state round-trips through a
real `RunSink` into a tmp run_dir, then back through `read_training_snapshot` and
`Trainer.from_snapshot`.

Single-process / 1-pool only — exercises the wiring around the lab's resumption
module without the cost of spinning up DDP.
"""

from pathlib import Path
from typing import Any, override

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from param_decomp.ci_fns import LayerwiseCiConfig
from param_decomp.configs import (
    Cadence,
    OptimizerConfig,
    PDConfig,
    RuntimeConfig,
)
from param_decomp.decomposition_targets import DecompositionTargetConfig
from param_decomp.metrics.faithfulness import FaithfulnessLossConfig
from param_decomp.optimize import Trainer
from param_decomp.schedule import ScheduleConfig
from param_decomp_lab.resumption import (
    ResumeConfig,
    read_training_snapshot,
    resolve_step,
)
from param_decomp_lab.run_sink import RunSink


class TinyLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


def _run_batch(model: nn.Module, batch: Any) -> Tensor:
    if isinstance(batch, list | tuple):
        batch = batch[0]
    assert isinstance(batch, Tensor)
    out = model(batch)
    assert isinstance(out, Tensor)
    return out


def _recon_loss(pred: Tensor, target: Tensor) -> tuple[Tensor, int]:
    assert pred.shape == target.shape
    return ((pred - target) ** 2).sum(), pred.numel()


def _pd_config(steps: int) -> PDConfig:
    return PDConfig(
        seed=123,
        n_mask_samples=1,
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[2]),
        decomposition_targets=[DecompositionTargetConfig(module_pattern="fc", C=2)],
        components_optimizer=OptimizerConfig(lr_schedule=ScheduleConfig(start_val=1e-3)),
        ci_fn_optimizer=OptimizerConfig(lr_schedule=ScheduleConfig(start_val=1e-3)),
        steps=steps,
        batch_size=2,
        loss_metrics=[FaithfulnessLossConfig(coeff=1.0)],
    )


def _loader() -> DataLoader[Any]:
    return DataLoader(TensorDataset(torch.ones(4, 2)), batch_size=2)


def _runtime() -> RuntimeConfig:
    return RuntimeConfig(device="cpu", autocast_bf16=False)


def _cadence() -> Cadence:
    return Cadence(train_log_every=10**9, save_every=2)


def test_run_sink_writes_model_and_training_files(tmp_path: Path) -> None:
    """A fresh run writes both `model_<step>.pth` and `training_<step>.pth` per save."""
    run_dir = tmp_path / "run"
    sink = RunSink.local(run_dir)

    trainer = Trainer(
        target_model=TinyLinear(),
        run_batch=_run_batch,
        reconstruction_loss=_recon_loss,
        pd_config=_pd_config(steps=4),
        runtime_config=_runtime(),
    )
    trainer.run(_loader(), sink, _cadence())

    # Cadence.should_save skips step 0; final step always saves. So we expect 2 and 4.
    expected_steps = [2, 4]
    for step in expected_steps:
        assert (run_dir / f"model_{step}.pth").is_file()
        assert (run_dir / f"training_{step}.pth").is_file()


def test_resume_round_trip_matches_uninterrupted_run(tmp_path: Path) -> None:
    """Train K steps in one shot vs train K/2 -> write training_<step>.pth -> read it ->
    Trainer.from_snapshot -> train K/2 more. Final weights match bit-for-bit on CPU.
    """
    # Reference: uninterrupted run.
    torch.manual_seed(7)
    trainer_full = Trainer(
        target_model=TinyLinear(),
        run_batch=_run_batch,
        reconstruction_loss=_recon_loss,
        pd_config=_pd_config(steps=4),
        runtime_config=_runtime(),
    )
    full_sink_dir = tmp_path / "full"
    trainer_full.run(_loader(), RunSink.local(full_sink_dir), _cadence())
    final_full = {k: v.clone() for k, v in trainer_full.component_model.state_dict().items()}

    # Phase 1: train 2 steps, the sink writes training_2.pth.
    torch.manual_seed(7)
    parent_dir = tmp_path / "parent"
    trainer_half = Trainer(
        target_model=TinyLinear(),
        run_batch=_run_batch,
        reconstruction_loss=_recon_loss,
        pd_config=_pd_config(steps=2),
        runtime_config=_runtime(),
    )
    trainer_half.run(_loader(), RunSink.local(parent_dir), _cadence())
    assert (parent_dir / "training_2.pth").is_file()

    # Phase 2: resume from parent's training_2.pth, train to step 4.
    resume_cfg = ResumeConfig(from_run=parent_dir, step=2)
    snapshot = read_training_snapshot(
        resume_cfg.from_run, resolve_step(parent_dir, resume_cfg.step)
    )
    snapshot.pd_config["steps"] = 4
    trainer_resumed = Trainer.from_snapshot(
        snapshot,
        target_model=TinyLinear(),
        run_batch=_run_batch,
        reconstruction_loss=_recon_loss,
    )
    assert trainer_resumed.step == 2
    resumed_dir = tmp_path / "resumed"
    trainer_resumed.run(_loader(), RunSink.local(resumed_dir), _cadence())

    resumed_final = trainer_resumed.component_model.state_dict()
    assert final_full.keys() == resumed_final.keys()
    for k in final_full:
        torch.testing.assert_close(final_full[k], resumed_final[k])


def test_resolve_step_finds_latest(tmp_path: Path) -> None:
    """`resolve_step('latest', ...)` returns the highest-numbered training file."""
    parent_dir = tmp_path / "parent"
    trainer = Trainer(
        target_model=TinyLinear(),
        run_batch=_run_batch,
        reconstruction_loss=_recon_loss,
        pd_config=_pd_config(steps=4),
        runtime_config=_runtime(),
    )
    trainer.run(_loader(), RunSink.local(parent_dir), _cadence())

    assert resolve_step(parent_dir, "latest") == 4
    assert resolve_step(parent_dir, 2) == 2


def test_keep_last_n_checkpoints_prunes_older_pairs(tmp_path: Path) -> None:
    """With ``keep_last_n_checkpoints=1``, only the most recent (model, training)
    pair survives. Earlier saves get deleted right after the next write.

    Default (``None``) keeps everything — covered by every other test in this file.
    """
    run_dir = tmp_path / "run"
    sink = RunSink.local(run_dir, keep_last_n_checkpoints=1)

    # save_every=2 + steps=4 → saves at step 2 and step 4 (final).
    trainer = Trainer(
        target_model=TinyLinear(),
        run_batch=_run_batch,
        reconstruction_loss=_recon_loss,
        pd_config=_pd_config(steps=4),
        runtime_config=_runtime(),
    )
    trainer.run(_loader(), sink, Cadence(train_log_every=1, save_every=2))

    # Step-2 pair must be gone; step-4 pair must remain.
    assert not (run_dir / "model_2.pth").exists()
    assert not (run_dir / "training_2.pth").exists()
    assert (run_dir / "model_4.pth").is_file()
    assert (run_dir / "training_4.pth").is_file()
