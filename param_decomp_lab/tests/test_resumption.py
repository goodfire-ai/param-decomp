"""Lab-side resumption integration: shards round-trip through a real
:class:`ResumableRunSink` into a tmp run_dir, then back through
:func:`read_resume_snapshot` and :meth:`Trainer.from_snapshot`.

Single-process / 1-pool only — exercises the wiring around the lab's
resumption module without the cost of spinning up DDP. The distributed
multi-rank path for 2-pool / 3-pool is intentionally not covered here;
those code paths share the same shard format and Snapshot shape, so this
test catches the bulk of integration risk.
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
    ResumableRunSink,
    ResumeConfig,
    list_resume_steps,
    read_resume_snapshot,
    shard_path,
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


def test_resumable_sink_writes_per_rank_shards(tmp_path: Path) -> None:
    """Fresh run with ``ResumableRunSink`` writes a shard per checkpoint step."""
    run_dir = tmp_path / "run"
    sink = ResumableRunSink(RunSink.local(run_dir), run_dir=run_dir, rank=0)

    trainer = Trainer(
        target_model=TinyLinear(),
        run_batch=_run_batch,
        reconstruction_loss=_recon_loss,
        pd_config=_pd_config(steps=4),
        runtime_config=_runtime(),
    )
    trainer.run(_loader(), sink, _cadence())

    # Cadence.should_save skips step 0; final step always saves. So we expect 2 and 4.
    steps = list_resume_steps(run_dir)
    assert steps == [2, 4]
    for step in steps:
        assert shard_path(run_dir, step, 0).is_file()
        # Consumable model on rank 0 is also written by the base sink.
        assert (run_dir / f"model_{step}.pth").is_file()


def test_resume_round_trip_matches_uninterrupted_run(tmp_path: Path) -> None:
    """Train K steps in one shot vs train K/2 → write shards → read shard →
    Trainer.from_snapshot → train K/2 more. Final weights match bit-for-bit on CPU.
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
    trainer_full.run(
        _loader(),
        ResumableRunSink(RunSink.local(full_sink_dir), run_dir=full_sink_dir, rank=0),
        _cadence(),
    )
    full_consumable = trainer_full.snapshot().consumable
    assert full_consumable is not None
    final_full = {k: v.clone() for k, v in full_consumable.items()}

    # Phase 1: train 2 steps, write shards.
    torch.manual_seed(7)
    parent_dir = tmp_path / "parent"
    trainer_half = Trainer(
        target_model=TinyLinear(),
        run_batch=_run_batch,
        reconstruction_loss=_recon_loss,
        pd_config=_pd_config(steps=2),
        runtime_config=_runtime(),
    )
    trainer_half.run(
        _loader(),
        ResumableRunSink(RunSink.local(parent_dir), run_dir=parent_dir, rank=0),
        _cadence(),
    )
    assert (parent_dir / "resume" / "step_2").is_dir()

    # Phase 2: resume from parent's step-2 shard, train to step 4.
    resume_cfg = ResumeConfig(from_run=parent_dir, step=2, overrides=None)
    snapshot = read_resume_snapshot(resume_cfg, rank=0, current_device="cpu")
    trainer_resumed = Trainer.from_snapshot(
        snapshot,
        target_model=TinyLinear(),
        run_batch=_run_batch,
        reconstruction_loss=_recon_loss,
        cfg_overrides={"steps": 4},
    )
    assert trainer_resumed.step == 2
    resumed_dir = tmp_path / "resumed"
    trainer_resumed.run(
        _loader(),
        ResumableRunSink(RunSink.local(resumed_dir), run_dir=resumed_dir, rank=0),
        _cadence(),
    )

    resumed_consumable = trainer_resumed.snapshot().consumable
    assert resumed_consumable is not None
    assert final_full.keys() == resumed_consumable.keys()
    for k in final_full:
        torch.testing.assert_close(final_full[k], resumed_consumable[k])


def test_read_resume_snapshot_patches_device(tmp_path: Path) -> None:
    """The saved runtime_config.device is overridden by ``current_device``."""
    parent_dir = tmp_path / "parent"
    trainer = Trainer(
        target_model=TinyLinear(),
        run_batch=_run_batch,
        reconstruction_loss=_recon_loss,
        pd_config=_pd_config(steps=2),
        runtime_config=_runtime(),  # cpu
    )
    trainer.run(
        _loader(),
        ResumableRunSink(RunSink.local(parent_dir), run_dir=parent_dir, rank=0),
        _cadence(),
    )

    resume_cfg = ResumeConfig(from_run=parent_dir, step="latest", overrides=None)
    snapshot = read_resume_snapshot(resume_cfg, rank=0, current_device="cpu:1")
    # The saved device was "cpu"; current_device override sets it on the
    # resume blob so reconstructed runtime_config picks it up.
    assert snapshot.resume["runtime_config"]["device"] == "cpu:1"
