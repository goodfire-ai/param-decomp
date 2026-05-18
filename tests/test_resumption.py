"""Tests for resumable training.

Covers:

- ``StatefulLoop`` state round-trips for both map-style and infinite-generated loaders.
- ``TrainingState`` save/load round-trip and atomic write.
- ``PersistentPGDState`` + ``AdamPGDOptimizer`` state-dict round-trip.
"""

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from param_decomp.training_state import (
    TRAINING_STATE_FILENAME,
    TrainingState,
    capture_rng_state,
    restore_rng_state,
)
from param_decomp.utils.data_utils import StatefulLoop


def _make_map_loader(n: int, batch_size: int, seed: int) -> DataLoader[Any]:
    data = TensorDataset(torch.arange(n))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return DataLoader(data, batch_size=batch_size, shuffle=True, generator=gen, drop_last=True)


def _collect(loop: StatefulLoop[Any], n: int) -> list[list[int]]:
    out: list[list[int]] = []
    for _ in range(n):
        batch = next(loop)
        out.append(batch[0].tolist())
    return out


def test_stateful_loop_data_determinism_across_resume() -> None:
    """The fresh-vs-resume batch sequence past the cut must match for a shuffle+generator loader."""
    # Reference: 20 batches in one go.
    ref_loader = _make_map_loader(n=128, batch_size=4, seed=42)
    ref_loop = StatefulLoop(ref_loader, seed=42)
    ref = _collect(ref_loop, 20)

    # Cut at 7: take 7, snapshot, restore into a fresh loop, take 13 more.
    cut_loader = _make_map_loader(n=128, batch_size=4, seed=42)
    cut_loop = StatefulLoop(cut_loader, seed=42)
    first = _collect(cut_loop, 7)
    snapshot = cut_loop.state_dict()

    resumed_loader = _make_map_loader(n=128, batch_size=4, seed=42)
    resumed_loop = StatefulLoop(resumed_loader, seed=42)
    resumed_loop.load_state_dict(snapshot)
    rest = _collect(resumed_loop, 13)

    assert first + rest == ref, "post-resume batches must match the non-interrupted sequence"


def test_stateful_loop_advances_epoch_on_exhaustion() -> None:
    """Epoch counter increments and within-epoch counter resets when the loader runs out."""
    loader = _make_map_loader(n=8, batch_size=4, seed=0)  # 2 batches per epoch
    loop = StatefulLoop(loader, seed=0)
    # 5 batches → epoch 0: 2 batches, epoch 1: 2 batches, epoch 2: 1 batch
    for _ in range(5):
        next(loop)
    state = loop.state_dict()
    assert state == {"epoch": 2, "batches_in_epoch": 1}


def test_training_state_round_trip(tmp_path: Path) -> None:
    """save -> load preserves all fields including nested tensors."""
    state = TrainingState(
        step=42,
        model_sd={"weight": torch.randn(3, 4)},
        components_opt_sd={"state": {"step": 41}},
        ci_fn_opt_sd={"state": {"step": 41}},
        ppgd_sd=[{"sources": {"m": torch.randn(2, 5)}, "optimizer": {"lr": 0.01}}],
        train_loop_sd={"epoch": 1, "batches_in_epoch": 7},
        eval_loop_sd={"epoch": 0, "batches_in_epoch": 3},
        rng_sd=capture_rng_state(),
        wandb_run_id="s-abcd1234",
    )
    path = tmp_path / TRAINING_STATE_FILENAME
    state.save(path)
    loaded = TrainingState.load(path)

    assert loaded.step == 42
    assert loaded.wandb_run_id == "s-abcd1234"
    assert loaded.train_loop_sd == {"epoch": 1, "batches_in_epoch": 7}
    assert torch.equal(loaded.model_sd["weight"], state.model_sd["weight"])
    assert torch.equal(loaded.ppgd_sd[0]["sources"]["m"], state.ppgd_sd[0]["sources"]["m"])


def test_training_state_atomic_write_leaves_no_tmp(tmp_path: Path) -> None:
    state = TrainingState(
        step=0,
        model_sd={},
        components_opt_sd={},
        ci_fn_opt_sd={},
        ppgd_sd=[],
        train_loop_sd={"epoch": 0, "batches_in_epoch": 0},
        eval_loop_sd={"epoch": 0, "batches_in_epoch": 0},
        rng_sd=capture_rng_state(),
        wandb_run_id=None,
    )
    path = tmp_path / TRAINING_STATE_FILENAME
    state.save(path)
    tmps = list(tmp_path.glob("*.tmp"))
    assert tmps == [], f"unexpected tmp files left behind: {tmps}"
    assert path.exists()


def _tms_config(steps: int, save_freq: int | None) -> tuple[object, object, object]:
    """Build (PDConfig, target_model, datasets) for a tiny TMS PD run.

    Seeds the global RNG before constructing the target model so every scenario in this
    test gets the exact same target weights -- otherwise the saved ``target_model.*``
    tensors (which are frozen and not learned) would differ run-to-run and dominate the
    comparison.
    """
    from param_decomp.configs import (
        FaithfulnessLossConfig,
        ImportanceMinimalityLossConfig,
        LayerwiseCiConfig,
        LossMetricsConfig,
        ModulePatternInfoConfig,
        OptimizerConfig,
        PDConfig,
        ScheduleConfig,
        StochasticReconLossConfig,
    )
    from param_decomp.experiments.tms.models import TMSModel, TMSModelConfig
    from param_decomp.utils.general_utils import set_seed

    set_seed(123)
    tms_cfg = TMSModelConfig(
        n_features=5,
        n_hidden=2,
        n_hidden_layers=0,
        tied_weights=False,
        init_bias_to_zero=False,
        device="cpu",
    )
    config = PDConfig(
        wandb_project=None,
        wandb_run_name=None,
        seed=0,
        n_mask_samples=1,
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[8]),
        module_info=[
            ModulePatternInfoConfig(module_pattern="linear1", C=8),
            ModulePatternInfoConfig(module_pattern="linear2", C=8),
        ],
        loss_metrics=LossMetricsConfig(
            importance_minimality=ImportanceMinimalityLossConfig(
                coeff=1e-3, pnorm=2.0, beta=0.5, eps=1e-12
            ),
            stochastic_recon=StochasticReconLossConfig(coeff=1.0),
            faithfulness=FaithfulnessLossConfig(coeff=1.0),
        ),
        components_optimizer=OptimizerConfig(
            lr_schedule=ScheduleConfig(
                start_val=1e-3, fn_type="cosine", warmup_pct=0.0, final_val_frac=0.0
            ),
        ),
        ci_fn_optimizer=OptimizerConfig(
            lr_schedule=ScheduleConfig(
                start_val=1e-3, fn_type="cosine", warmup_pct=0.0, final_val_frac=0.0
            ),
        ),
        batch_size=4,
        steps=steps,
        n_eval_steps=1,
        faithfulness_warmup_steps=0,
        faithfulness_warmup_lr=0.001,
        faithfulness_warmup_weight_decay=0.0,
        train_log_freq=10_000,
        save_freq=save_freq,
        ci_alive_threshold=0.1,
        eval_batch_size=4,
        eval_freq=10_000,
        slow_eval_freq=10_000,
    )
    target_model = TMSModel(config=tms_cfg).to("cpu")
    target_model.eval()
    return config, target_model, tms_cfg


def _run_tms(
    config: object, target_model: object, out_dir: Path, *, resume_from: Path | None
) -> dict[str, torch.Tensor]:
    """Run optimize() to completion; return the final ComponentModel state_dict (CPU copy)."""
    from param_decomp.models.batch_and_loss_fns import (
        recon_loss_mse,
        run_batch_first_element,
    )
    from param_decomp.run_pd import optimize
    from param_decomp.training_state import TrainingState
    from param_decomp.utils.data_utils import DatasetGeneratedDataLoader, SparseFeatureDataset
    from param_decomp.utils.general_utils import set_seed

    set_seed(0)
    dataset = SparseFeatureDataset(
        n_features=5,
        feature_probability=0.1,
        device="cpu",
        data_generation_type="at_least_zero_active",
        value_range=(0.0, 1.0),
        synced_inputs=None,
    )
    train_loader = DatasetGeneratedDataLoader(dataset, batch_size=4, shuffle=False)
    eval_loader = DatasetGeneratedDataLoader(dataset, batch_size=4, shuffle=False)

    resume_state = TrainingState.load(resume_from) if resume_from is not None else None
    optimize(
        target_model=target_model,  # type: ignore[arg-type]
        config=config,  # type: ignore[arg-type]
        device="cpu",
        train_loader=train_loader,
        eval_loader=eval_loader,
        run_batch=run_batch_first_element,
        reconstruction_loss=recon_loss_mse,
        out_dir=out_dir,
        tied_weights=None,
        resume_state=resume_state,
    )
    # Final checkpoint is model_<steps>.pth on every run.
    ckpt = torch.load(out_dir / f"model_{config.steps}.pth", map_location="cpu", weights_only=True)  # type: ignore[attr-defined]
    return ckpt


def test_e2e_resume_matches_uninterrupted_tms(tmp_path: Path) -> None:
    """Train 30 steps in one go vs train 15 steps -> resume -> 30 steps. Final model should match.

    TMS uses ``DatasetGeneratedDataLoader`` which draws from global torch RNG; coupled with
    stochastic mask sampling this would normally drift across resume. ``TrainingState`` saves
    the RNG state at the exact step boundary, so the resumed half re-enters with byte-equal
    RNG and produces the same trajectory.
    """
    # Reference: train 30 steps end-to-end with no save.
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    ref_cfg, ref_tgt, _ = _tms_config(steps=30, save_freq=None)
    ref_sd = _run_tms(ref_cfg, ref_tgt, ref_dir, resume_from=None)

    # Cut: train 30 steps with save_freq=15. ``training_state.pt`` is written at step 15 and
    # is *not* overwritten at step 30 (final-step saves only write the inference checkpoint).
    # Total steps must match the reference so the cosine LR schedule produces the same values.
    cut_dir = tmp_path / "cut"
    cut_dir.mkdir()
    cut_cfg, cut_tgt, _ = _tms_config(steps=30, save_freq=15)
    _ = _run_tms(cut_cfg, cut_tgt, cut_dir, resume_from=None)
    resume_ckpt = cut_dir / TRAINING_STATE_FILENAME
    assert resume_ckpt.exists()

    # Resume from step 15 into a fresh dir and train through step 30.
    res_dir = tmp_path / "res"
    res_dir.mkdir()
    res_cfg, res_tgt, _ = _tms_config(steps=30, save_freq=None)
    res_sd = _run_tms(res_cfg, res_tgt, res_dir, resume_from=resume_ckpt)

    # All parameter tensors should match.
    assert set(ref_sd.keys()) == set(res_sd.keys())
    for name in ref_sd:
        assert torch.allclose(ref_sd[name], res_sd[name], atol=1e-6, rtol=1e-5), (
            f"resumed final state diverged from uninterrupted run on '{name}': "
            f"max_abs_diff={(ref_sd[name] - res_sd[name]).abs().max().item():.3e}"
        )


def test_rng_capture_restore_makes_torch_deterministic() -> None:
    snap = capture_rng_state()
    a1 = torch.randn(5)
    np_a1 = np.random.randn(5)

    restore_rng_state(snap)
    a2 = torch.randn(5)
    np_a2 = np.random.randn(5)

    assert torch.equal(a1, a2)
    assert np.array_equal(np_a1, np_a2)
