"""Eval-only entry point: load a saved LM PD run, run one eval pass, log to wandb.

Builds a tiny saved-run on disk by running a few training steps with `Trainer`, then
calls `_eval_only_main` against it. wandb is mocked end-to-end so we don't depend on
network or credentials.
"""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from transformers import GPT2LMHeadModel

from param_decomp.ci_fns import LayerwiseCiConfig
from param_decomp.configs import Cadence, OptimizerConfig, PDConfig, RuntimeConfig
from param_decomp.decomposition_targets import DecompositionTargetConfig
from param_decomp.metrics.faithfulness import FaithfulnessLossConfig
from param_decomp.optimize import Trainer
from param_decomp.schedule import ScheduleConfig
from param_decomp_lab.batch_and_loss_fns import make_run_batch, recon_loss_kl
from param_decomp_lab.eval_metrics.ci_l0 import CI_L0Config
from param_decomp_lab.experiments.lm.data import (
    LMDataConfig,
    collate_fn_for,
    create_lm_data_loader,
)
from param_decomp_lab.experiments.lm.run import (
    HFTarget,
    LMExperimentConfig,
    LMTargetConfig,
    SavedLMRun,
    _eval_only_main,
    _resolve_train_run_id,
    _step_from_checkpoint_name,
)
from param_decomp_lab.experiments.utils import EvalConfig, WandbConfig
from param_decomp_lab.run_sink import RunSink

_MODEL_NAME = "SimpleStories/test-SimpleStories-gpt2-1.25M"


def _make_cfg(steps: int) -> LMExperimentConfig:
    pd = PDConfig(
        seed=0,
        n_mask_samples=1,
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[16]),
        decomposition_targets=[
            DecompositionTargetConfig(module_pattern="transformer.h.0.mlp.c_fc", C=4),
        ],
        loss_metrics=[FaithfulnessLossConfig(coeff=1.0)],
        components_optimizer=OptimizerConfig(lr_schedule=ScheduleConfig(start_val=1e-3)),
        ci_fn_optimizer=OptimizerConfig(lr_schedule=ScheduleConfig(start_val=1e-3)),
        batch_size=2,
        steps=steps,
    )
    target = LMTargetConfig(
        spec=HFTarget(model_class="transformers.GPT2LMHeadModel", model_name=_MODEL_NAME),
        output_extract="logits",
    )
    data = LMDataConfig(
        dataset_name="SimpleStories/SimpleStories",
        tokenizer_name=_MODEL_NAME,
        max_seq_len=16,
        train_split="train[:32]",
        eval_split="test[100:132]",
        is_tokenized=False,
        streaming=False,
        column_name="story",
    )
    return LMExperimentConfig(
        pd=pd,
        runtime=RuntimeConfig(device="cpu", autocast_bf16=False),
        cadence=Cadence(train_log_every=10**9, save_every=None),
        target=target,
        data=data,
        eval=EvalConfig(
            batch_size=2,
            n_steps=1,
            every=1,
            slow_every=1,
            slow_on_first_step=True,
            metrics=[CI_L0Config(groups=None, ci_alive_threshold=0.0)],
        ),
        wandb=WandbConfig(project="param-decomp-test", entity="goodfire"),
    )


def _write_tiny_saved_run(out_dir: Path) -> int:
    """Train a tiny LM PD run for a single step and persist it to `out_dir`.

    Returns the step number written. Mirrors what `pd-lm` does end-to-end but skips
    SLURM / wandb so the test stays single-process and offline.
    """
    cfg = _make_cfg(steps=1)
    cfg.to_file(out_dir / "experiment_config.yaml")

    target_model = GPT2LMHeadModel.from_pretrained(_MODEL_NAME)
    target_model.eval()

    train_loader, _ = create_lm_data_loader(
        cfg.data,
        split=cfg.data.train_split,
        batch_size=cfg.pd.batch_size,
        seed=cfg.pd.seed,
        collate_fn=collate_fn_for(cfg.data),
    )
    trainer = Trainer(
        target_model=target_model,
        run_batch=make_run_batch(cfg.target.output_extract),
        reconstruction_loss=recon_loss_kl,
        pd_config=cfg.pd,
        runtime_config=cfg.runtime,
    )
    trainer.run(train_loader, RunSink.local(out_dir), cfg.cadence)
    assert (out_dir / f"model_{cfg.pd.steps}.pth").is_file()
    return cfg.pd.steps


def test_step_from_checkpoint_name() -> None:
    assert _step_from_checkpoint_name("model_5000.pth") == 5000
    assert _step_from_checkpoint_name("model_0.pth") == 0


def test_resolve_train_run_id_local_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "decompositions" / "p-deadbeef"
    run_dir.mkdir(parents=True)
    assert _resolve_train_run_id(run_dir) == "p-deadbeef"


def test_resolve_train_run_id_wandb_ref() -> None:
    assert _resolve_train_run_id("p-deadbeef") == "p-deadbeef"
    assert _resolve_train_run_id("goodfire/param-decomp/p-deadbeef") == "p-deadbeef"
    assert (
        _resolve_train_run_id("https://wandb.ai/goodfire/param-decomp/runs/p-deadbeef")
        == "p-deadbeef"
    )


@pytest.mark.slow
def test_eval_only_main_against_tiny_saved_run(tmp_path: Path) -> None:
    """End-to-end: train 1 step, then `_eval_only_main` reloads and evals.

    wandb is mocked: `wandb.init` is a no-op and `wandb.log` records its arguments so
    the assertion can inspect the payload keys / step.
    """
    run_dir = tmp_path / "p-feedface"
    run_dir.mkdir()
    step_written = _write_tiny_saved_run(run_dir)

    # Sanity: the run is reloadable via the public SavedLMRun API.
    pd_run = SavedLMRun.from_path(run_dir)
    assert pd_run.checkpoint_path.name == f"model_{step_written}.pth"
    component_model = pd_run.load_model()
    assert component_model is not None

    init_calls: list[dict[str, Any]] = []
    logged: list[dict[str, Any]] = []

    def fake_init(**kwargs: Any) -> None:
        init_calls.append(kwargs)
        return None

    def fake_log(payload: dict[str, Any], step: int | None = None) -> None:
        logged.append({"payload": payload, "step": step})

    def fake_finish() -> None:
        return None

    with (
        patch("param_decomp_lab.experiments.lm.run.wandb.init", side_effect=fake_init),
        patch("param_decomp_lab.experiments.lm.run.wandb.log", side_effect=fake_log),
        patch("param_decomp_lab.experiments.lm.run.wandb.finish", side_effect=fake_finish),
        patch(
            "param_decomp_lab.experiments.lm.run.get_wandb_entity",
            return_value="goodfire",
        ),
        # `try_wandb` wraps `wandb.log` — patch it to call through directly so the
        # fake_log above actually gets the call (try_wandb imports CommError).
        patch(
            "param_decomp_lab.experiments.lm.run.try_wandb",
            side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs),
        ),
    ):
        _eval_only_main(run_dir, step=step_written, group=None, tags=None)

    assert len(init_calls) == 1, f"expected one wandb.init call, got {len(init_calls)}"
    init_kwargs = init_calls[0]
    assert init_kwargs["id"] == "p-feedface", init_kwargs
    assert init_kwargs["resume"] == "must", init_kwargs
    assert init_kwargs["project"] == "param-decomp-test", init_kwargs

    assert len(logged) == 1, f"expected one wandb.log call, got {len(logged)}"
    entry = logged[0]
    assert entry["step"] == step_written
    payload = entry["payload"]
    assert payload, "wandb.log payload is empty"
    assert all(k.startswith("eval/") for k in payload), payload
    # CI_L0 emits keys under its `log_namespace="l0"` namespace; the eval prefix adds
    # `eval/` on top.
    assert any("l0/" in k for k in payload), list(payload)


@pytest.mark.slow
def test_eval_only_main_logs_skipped_when_no_wandb(tmp_path: Path) -> None:
    """When the parent cfg has no `wandb:` block, eval still runs but no wandb calls happen."""
    run_dir = tmp_path / "p-feedface"
    run_dir.mkdir()
    step_written = _write_tiny_saved_run(run_dir)

    # Drop the wandb block from the saved config and re-write it.
    saved_cfg_path = run_dir / "experiment_config.yaml"
    cfg = LMExperimentConfig.from_file(saved_cfg_path)
    cfg_no_wandb = cfg.model_copy(update={"wandb": None})
    cfg_no_wandb.to_file(saved_cfg_path)

    init_calls: list[dict[str, Any]] = []

    def fake_init(**kwargs: Any) -> None:
        init_calls.append(kwargs)
        return None

    with (
        patch("param_decomp_lab.experiments.lm.run.wandb.init", side_effect=fake_init),
        patch("param_decomp_lab.experiments.lm.run.wandb.log"),
        patch("param_decomp_lab.experiments.lm.run.wandb.finish"),
    ):
        _eval_only_main(run_dir, step=step_written, group=None, tags=None)

    assert init_calls == [], "wandb.init should not be called when cfg.wandb is None"
