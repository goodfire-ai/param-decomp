"""The shared-config (wrapper) route — the trainer's only config surface.

Committed wrapper yamls deliberately carry NO `run_id` (`pd-jax-lm` mints one and
stamps the workspace copy at submit time), so tests inject one the same way."""

from pathlib import Path

import pytest
import yaml

from jax_single_pool.config import (
    WRAPPER_KEYS,
    WRAPPER_OPTIONAL_KEYS,
    assert_supported_weights_dtype,
    build_experiment_config,
    load_run_dir_config,
    load_wrapper,
)
from jax_single_pool.llama8b import mlp_family_site_cs
from jax_single_pool.lm import SiteC
from jax_single_pool.recon import build_recon_terms
from param_decomp_config.jax_wrapper import (
    RUN_ID_KEY,
    SUBMIT_MINTED_KEYS,
    WRAPPER_KEYS_BEFORE_SUBMIT,
)
from param_decomp_config.jax_wrapper import WRAPPER_KEYS as SHARED_WRAPPER_KEYS
from param_decomp_config.lm import LMExperimentConfig
from param_decomp_config.losses import PersistentPGDReconLossConfig

CONFIGS = Path(__file__).parent.parent / "configs"
RUN_ID = "p-0123abcd"


def _stamped_wrapper(tmp_path: Path, wrapper: Path) -> Path:
    """A tmp copy of `wrapper` with `run_id` stamped and the torch path absolutized —
    what the pd-jax-lm workspace copy looks like."""
    raw = yaml.safe_load(wrapper.read_text())
    raw["torch_config"] = str((wrapper.parent / raw["torch_config"]).resolve())
    raw["run_id"] = RUN_ID
    stamped = tmp_path / wrapper.name
    stamped.write_text(yaml.safe_dump(raw))
    return stamped


def test_b128_wrapper_converts(tmp_path: Path):
    converted, torch_yaml_path, torch_raw = load_wrapper(
        _stamped_wrapper(tmp_path, CONFIGS / "llama8b_l18_b128_cmp32_from_torch.yaml")
    )
    assert torch_yaml_path == (CONFIGS / "torch" / "llama8b_l18_b128_cmp32_1pool.yaml").resolve()
    assert torch_raw["pd"]["batch_size"] == 128
    assert converted.run_name == "jax-l18-b128-cmp32-from-torch"
    assert converted.data.global_batch == 128
    assert converted.target.sites == mlp_family_site_cs(18, 18, 24576)
    spec = build_recon_terms(
        converted.loss_metrics, tuple(sc.name for sc in converted.target.sites),
        converted.n_mask_samples, converted.sampling,
    )  # fmt: skip
    assert spec.faith_coeff == 1e5 and spec.imp_min.pnorm == 2.0
    (ppgd,) = spec.persistent.values()
    assert isinstance(ppgd, PersistentPGDReconLossConfig)
    assert ppgd.n_warmup_steps == 2 and converted.vu_optimizer.grad_clip_norm == 0.01
    assert [t.name for t in spec.recon_terms] == [
        "StochasticReconSubsetLoss",
        "PersistentPGDReconLoss",
    ]


def _reference_torch_cfg():
    from param_decomp_config.lm import LMExperimentConfig

    raw = yaml.safe_load((CONFIGS / "torch" / "llama8b_l18_b128_cmp32_1pool.yaml").read_text())
    return LMExperimentConfig(**raw), raw


def test_eval_block_maps_and_defers_offline_metrics(capsys: pytest.CaptureFixture[str]):
    torch_cfg, raw = _reference_torch_cfg()
    raw["eval"] = {
        "batch_size": 128,
        "every": 1000,
        "n_steps": 1,
        "slow_every": 10000,
        "metrics": [
            {"type": "CEandKLLosses", "rounding_threshold": 0.0},
            {"type": "CI_L0", "groups": None, "ci_alive_threshold": 0.0},
            {
                "type": "PGDReconLoss",
                "coeff": None,
                "init": "random",
                "mask_scope": "c",
                "n_steps": 20,
                "step_size": 0.1,
            },
            {"type": "CIHistograms", "n_batches_accum": 1},
            {"type": "ComponentActivationDensity", "ci_alive_threshold": 0.0},
        ],
    }
    torch_cfg = type(torch_cfg)(**raw)
    cfg = build_experiment_config(
        torch_cfg, run_name="t", run_id=RUN_ID, out_dir=Path("/tmp"), remat_recon_forwards=True
    )
    assert cfg.eval is not None
    assert (cfg.eval.batch_size, cfg.eval.every, cfg.eval.n_steps) == (128, 1000, 1)
    assert cfg.eval.rounding_threshold == 0.0 and cfg.eval.ci_alive_threshold == 0.0
    assert cfg.eval.pgd is not None and (cfg.eval.pgd.n_steps, cfg.eval.pgd.step_size) == (20, 0.1)
    assert "deferred to the offline path" in capsys.readouterr().out


def test_unsupported_settings_refuse():
    torch_cfg, raw = _reference_torch_cfg()

    hidden_acts_training_loss = dict(
        raw,
        pd=dict(
            raw["pd"],
            loss_metrics=raw["pd"]["loss_metrics"]
            + [{"type": "StochasticHiddenActsReconLoss", "coeff": 1.0}],
        ),
    )
    with pytest.raises(AssertionError, match="unsupported training loss"):
        build_experiment_config(
            type(torch_cfg)(**hidden_acts_training_loss), run_name="t", run_id=RUN_ID,
            out_dir=Path("/tmp"), remat_recon_forwards=True,
        )  # fmt: skip

    sigmoid_ppgd = dict(
        raw,
        pd=dict(
            raw["pd"],
            loss_metrics=[
                dict(m, use_sigmoid_parameterization=True)
                if m["type"] == "PersistentPGDReconLoss"
                else m
                for m in raw["pd"]["loss_metrics"]
            ],
        ),
    )
    with pytest.raises(AssertionError):
        build_experiment_config(
            type(torch_cfg)(**sigmoid_ppgd), run_name="t", run_id=RUN_ID, out_dir=Path("/tmp"),
            remat_recon_forwards=True,
        )  # fmt: skip

    non_site_target = dict(
        raw,
        pd=dict(
            raw["pd"],
            decomposition_targets=[{"module_pattern": "layers.18.input_layernorm", "C": 512}],
        ),
    )
    with pytest.raises(AssertionError, match="unsupported decomposition target"):
        build_experiment_config(
            type(torch_cfg)(**non_site_target), run_name="t", run_id=RUN_ID, out_dir=Path("/tmp"),
            remat_recon_forwards=True,
        )  # fmt: skip

    embedding_target = dict(
        raw,
        pd=dict(
            raw["pd"],
            decomposition_targets=[{"module_pattern": "embed_tokens", "C": 512}],
        ),
    )
    with pytest.raises(AssertionError, match="unsupported decomposition target"):
        build_experiment_config(
            type(torch_cfg)(**embedding_target), run_name="t", run_id=RUN_ID, out_dir=Path("/tmp"),
            remat_recon_forwards=True,
        )  # fmt: skip


def test_unsupported_model_family_refuses_and_supported_families_dispatch():
    """E23 (PARITY_MATRIX §11 row 2): only Llama-3.1-8B (`hf`/`hf_weights_in_vendored`
    → `TargetConfig`) and `LlamaSimpleMLP` (`pretrained` →
    `LlamaSimpleMLPTargetConfig`) convert; every other family is refused at convert
    time. The torch schema's `LMTargetSpec` discriminated union still validates a
    GPT-2 spec (it's a well-formed `kind`), so the refusal must come from
    `_resolve_target`'s per-family asserts, not pydantic."""
    from jax_single_pool.config import LlamaSimpleMLPTargetConfig, TargetConfig

    torch_cfg, raw = _reference_torch_cfg()

    def _converted_target(spec: dict[str, str]):
        cfg = build_experiment_config(
            type(torch_cfg)(**dict(raw, target=dict(raw["target"], spec=spec))),
            run_name="t", run_id=RUN_ID, out_dir=Path("/tmp"), remat_recon_forwards=True,
        )  # fmt: skip
        return cfg.target

    vendored_llama = _converted_target(
        {
            "kind": "hf_weights_in_vendored",
            "model_class": "param_decomp_lab.experiments.lm.vendored.llama_3_1.model.VendoredLlama",
            "model_name": "meta-llama/Llama-3.1-8B",
        }
    )
    assert isinstance(vendored_llama, TargetConfig)

    raw_hf_llama = _converted_target(
        {
            "kind": "hf",
            "model_class": "transformers.LlamaForCausalLM",
            "model_name": "meta-llama/Llama-3.1-8B",
        }
    )
    assert isinstance(raw_hf_llama, TargetConfig)

    gpt2_hf = {
        "kind": "hf",
        "model_class": "transformers.GPT2LMHeadModel",
        "model_name": "gpt2",
    }
    with pytest.raises(AssertionError, match="transformers.GPT2LMHeadModel"):
        _converted_target(gpt2_hf)

    gpt2_vendored = {
        "kind": "hf_weights_in_vendored",
        "model_class": "param_decomp_lab.experiments.lm.pretrain.models.gpt2.GPT2Simple",
        "model_name": "gpt2",
    }
    with pytest.raises(AssertionError, match="GPT2Simple"):
        _converted_target(gpt2_vendored)

    other_hf_llama = {
        "kind": "hf",
        "model_class": "transformers.LlamaForCausalLM",
        "model_name": "meta-llama/Llama-3.2-1B",
    }
    with pytest.raises(AssertionError, match="Llama-3.2-1B"):
        _converted_target(other_hf_llama)

    # `pretrained` dispatches into the LlamaSimpleMLP branch (proven by the
    # model-class assert firing there); a non-LlamaSimpleMLP pretrained spec refuses
    # before any disk access.
    non_simple_mlp_pretrained = {
        "kind": "pretrained",
        "model_class": "param_decomp_lab.experiments.lm.pretrain.models.gpt2.GPT2Simple",
        "run_path": "goodfire/spd/runs/t-deadbeef",
    }
    with pytest.raises(AssertionError, match="GPT2Simple"):
        _converted_target(non_simple_mlp_pretrained)

    assert LlamaSimpleMLPTargetConfig is not None  # the `pretrained` happy-path type


def test_decaying_persistent_source_schedule_refuses():
    """The JAX source schedule is `warmup_then_constant_lr` (no post-warmup decay);
    a torch PPGD source `lr_schedule` that decays would silently flatten, so the
    conversion gate must refuse it (issue #646; matrix S13/S20)."""
    torch_cfg, raw = _reference_torch_cfg()
    decaying_source = dict(
        raw,
        pd=dict(
            raw["pd"],
            loss_metrics=[
                dict(
                    m,
                    optimizer=dict(
                        m["optimizer"],
                        lr_schedule=dict(
                            m["optimizer"]["lr_schedule"],
                            fn_type="cosine",
                            final_val_frac=0.1,
                        ),
                    ),
                )
                if m["type"] == "PersistentPGDReconLoss"
                else m
                for m in raw["pd"]["loss_metrics"]
            ],
        ),
    )
    with pytest.raises(AssertionError):
        build_experiment_config(
            type(torch_cfg)(**decaying_source), run_name="t", run_id=RUN_ID,
            out_dir=Path("/tmp"), remat_recon_forwards=True,
        )  # fmt: skip


def test_arbitrary_sites_with_per_site_c_convert():
    """Attention + MLP sites across non-contiguous layers with heterogeneous C —
    the general site space this trainer now implements."""
    torch_cfg, raw = _reference_torch_cfg()
    general = dict(
        raw,
        pd=dict(
            raw["pd"],
            decomposition_targets=[
                {"module_pattern": "layers.20.mlp.up_proj", "C": 64},
                {"module_pattern": "model.layers.18.self_attn.q_proj", "C": 128},
                {"module_pattern": "layers.18.self_attn.v_proj", "C": 32},
            ],
        ),
    )
    cfg = build_experiment_config(
        type(torch_cfg)(**general), run_name="t", run_id=RUN_ID, out_dir=Path("/tmp"),
        remat_recon_forwards=True,
    )  # fmt: skip
    assert cfg.target.sites == (
        SiteC("layers.18.self_attn.q_proj", 128),
        SiteC("layers.18.self_attn.v_proj", 32),
        SiteC("layers.20.mlp.up_proj", 64),
    )


def test_c49k_yaml_converts(tmp_path: Path):
    """The C49k/200k yaml (raw-HF target spec, bf16 weights_dtype, `model.`-prefixed
    site patterns) must convert cleanly."""
    converted, _torch_path, _raw = load_wrapper(
        _stamped_wrapper(tmp_path, CONFIGS / "llama8b_l18_C49k_200k_from_torch.yaml")
    )
    assert converted.target.sites == mlp_family_site_cs(18, 18, 49152)
    assert converted.steps == 200000
    assert converted.data.global_batch == 512 and converted.data.seq_len == 2048
    assert converted.vu_optimizer.lr == 7e-05 and converted.ci_optimizer.lr == 7e-05
    assert converted.eval is not None and converted.eval.pgd is not None
    assert converted.wandb is not None and converted.wandb.entity is None


def test_fp32_frozen_target_is_refused(tmp_path: Path):
    """A config requesting an fp32 frozen target must crash at the train/submit
    boundary — the bf16-only targets have no fp32 capability, and there is no silent
    downgrade (issue #727). Consumption paths (`load_run_dir_config`) ignore the field,
    so the guard lives in the wrapper route, not the shared builder."""
    wrapper = _stamped_wrapper(tmp_path, CONFIGS / "llama8b_l18_C49k_200k_from_torch.yaml")
    schema_yaml_path = (
        wrapper.parent / yaml.safe_load(wrapper.read_text())["torch_config"]
    ).resolve()
    cfg = LMExperimentConfig(**yaml.safe_load(schema_yaml_path.read_text()))
    cfg = cfg.model_copy(
        update={"target": cfg.target.model_copy(update={"weights_dtype": "float32"})}
    )
    with pytest.raises(AssertionError, match="weights_dtype"):
        assert_supported_weights_dtype(cfg)


def test_load_run_dir_config_rebuilds_wrapper_runs(tmp_path: Path):
    """The exporter reads run dirs via `load_run_dir_config`; runs pin the wrapper as
    config.yaml + the torch yaml as experiment_config.yaml (run.py's
    `_pin_config_copy`), and the rebuilt config must equal the launch-time conversion."""
    wrapper = _stamped_wrapper(tmp_path, CONFIGS / "llama8b_l18_C49k_200k_from_torch.yaml")
    expected, torch_yaml_path, _ = load_wrapper(wrapper)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(wrapper.read_text())
    (run_dir / "experiment_config.yaml").write_text(torch_yaml_path.read_text())
    assert load_run_dir_config(run_dir) == expected


def test_offline_eval_submission_argv(tmp_path: Path):
    from jax_single_pool.run import offline_eval_submission_argv

    assert offline_eval_submission_argv(tmp_path, 0) is None  # init checkpoint
    argv = offline_eval_submission_argv(tmp_path, 5000)
    assert argv is not None and argv[0] == "sbatch"
    assert f"--job-name=jsp-oeval-{tmp_path.name}" in argv
    assert "--dependency=singleton" in argv
    assert argv[-2:] == [str(tmp_path), "5000"]
    assert Path(argv[-3]).name == "offline_eval_once.sbatch" and Path(argv[-3]).exists()


def test_wrapper_run_id_required_and_drives_identity(tmp_path: Path):
    """The run dir and wandb id are the p-id (torch runs/<id>/ convention); the human
    name stays the wandb display name. Missing or malformed run_id refuses."""
    wrapper = _stamped_wrapper(tmp_path, CONFIGS / "llama8b_l18_C49k_200k_from_torch.yaml")
    cfg, _, _ = load_wrapper(wrapper)
    assert cfg.run_id == RUN_ID
    assert cfg.run_dir.name == RUN_ID
    assert cfg.run_name == "jax-l18-C49k-200k"

    with pytest.raises(AssertionError, match="keys must be"):
        load_wrapper(CONFIGS / "llama8b_l18_C49k_200k_from_torch.yaml")  # no run_id

    bad_id = tmp_path / "bad_id.yaml"
    bad_id.write_text(wrapper.read_text().replace(RUN_ID, "run42"))
    with pytest.raises(AssertionError, match="run_id must be"):
        load_wrapper(bad_id)


def test_loader_uses_shared_wrapper_key_set():
    """The runtime loader's key sets are the shared constants (no hand-copied
    literals); a hand-authored wrapper carries the required keys minus run_id, and
    the submit-minted keys are exactly run_id + the optional wandb knobs."""
    assert WRAPPER_KEYS is SHARED_WRAPPER_KEYS
    assert WRAPPER_KEYS_BEFORE_SUBMIT | {RUN_ID_KEY} == WRAPPER_KEYS
    assert {RUN_ID_KEY} | WRAPPER_OPTIONAL_KEYS == SUBMIT_MINTED_KEYS


def test_loader_rejects_unexpected_key(tmp_path: Path):
    wrapper = _stamped_wrapper(tmp_path, CONFIGS / "llama8b_l18_C49k_200k_from_torch.yaml")
    raw = yaml.safe_load(wrapper.read_text())
    raw["bogus_key"] = "x"
    wrapper.write_text(yaml.safe_dump(raw))
    with pytest.raises(AssertionError, match="keys must be"):
        load_wrapper(wrapper)
