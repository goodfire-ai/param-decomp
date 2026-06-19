"""The single-file run-config route — the trainer's only config surface.

Committed configs deliberately carry NO `run_id` (`pd-jax-lm` mints one and stamps the
workspace copy at submit time) and some leave `out_dir` absent (minted at submit), so
tests inject both the same way `pd-jax-lm` does."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from jax_single_pool.config import (
    DataConfig,
    assert_supported_weights_dtype,
    build_experiment_config,
    load_config,
    load_run_dir_config,
)
from jax_single_pool.llama8b import mlp_family_site_cs
from jax_single_pool.lm import SiteC
from jax_single_pool.recon import build_recon_terms
from param_decomp_config.lm import LMExperimentConfig
from param_decomp_config.losses import (
    ImportanceMinimalityLossConfig,
    PersistentPGDReconLossConfig,
)

CONFIGS = Path(__file__).parent.parent / "configs"
RUN_ID = "p-0123abcd"


def _stamped_config(tmp_path: Path, config: Path) -> Path:
    """A tmp copy of `config` with `run_id` + (if absent) `out_dir` stamped — what the
    pd-jax-lm workspace copy looks like at submit time."""
    raw = yaml.safe_load(config.read_text())
    raw["run_id"] = RUN_ID
    if raw.get("out_dir") is None:
        raw["out_dir"] = "/tmp/out"
    stamped = tmp_path / config.name
    stamped.write_text(yaml.safe_dump(raw))
    return stamped


def _reference_lm_raw():
    raw = yaml.safe_load((CONFIGS / "llama8b_l18_b128_cmp32.yaml").read_text())
    raw["run_id"] = RUN_ID
    return raw


def test_b128_config_converts(tmp_path: Path):
    converted, raw = load_config(_stamped_config(tmp_path, CONFIGS / "llama8b_l18_b128_cmp32.yaml"))
    assert raw["pd"]["batch_size"] == 128
    assert converted.run_name == "jax-l18-b128-cmp32-from-torch"
    assert converted.data.global_batch == 128
    assert converted.target.sites == mlp_family_site_cs(18, 18, 24576)
    spec = build_recon_terms(
        converted.loss_metrics, tuple(sc.name for sc in converted.target.sites),
        converted.n_mask_samples, converted.sampling,
    )  # fmt: skip
    assert isinstance(spec.imp_min, ImportanceMinimalityLossConfig)
    assert spec.faith_coeff == 1e5 and spec.imp_min.pnorm == 2.0
    (ppgd,) = spec.persistent.values()
    assert isinstance(ppgd, PersistentPGDReconLossConfig)
    assert ppgd.n_warmup_steps == 2 and converted.vu_optimizer.grad_clip_norm == 0.01
    assert [t.name for t in spec.recon_terms] == [
        "StochasticReconSubsetLoss",
        "PersistentPGDReconLoss",
    ]


def test_eval_block_maps_and_defers_offline_metrics(capsys: pytest.CaptureFixture[str]):
    raw = _reference_lm_raw()
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
    cfg = build_experiment_config(LMExperimentConfig(**raw))
    assert cfg.eval is not None
    assert (cfg.eval.batch_size, cfg.eval.every, cfg.eval.n_steps) == (128, 1000, 1)
    assert cfg.eval.rounding_threshold == 0.0 and cfg.eval.ci_alive_threshold == 0.0
    assert cfg.eval.pgd is not None and (cfg.eval.pgd.n_steps, cfg.eval.pgd.step_size) == (20, 0.1)
    assert "deferred to the offline path" in capsys.readouterr().out


def test_unsupported_settings_refuse():
    raw = _reference_lm_raw()

    hidden_acts_training_loss = dict(
        raw,
        pd=dict(
            raw["pd"],
            loss_metrics=raw["pd"]["loss_metrics"]
            + [{"type": "StochasticHiddenActsReconLoss", "coeff": 1.0}],
        ),
    )
    with pytest.raises(AssertionError, match="unsupported training loss"):
        build_experiment_config(LMExperimentConfig(**hidden_acts_training_loss))

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
    # use_sigmoid_parameterization was removed (clamp-only); the strip-on-load shim
    # accepts False (stored configs) but rejects True at config construction.
    with pytest.raises(ValidationError):
        LMExperimentConfig(**sigmoid_ppgd)

    non_site_target = dict(
        raw,
        pd=dict(
            raw["pd"],
            decomposition_targets=[{"module_pattern": "layers.18.input_layernorm", "C": 512}],
        ),
    )
    with pytest.raises(AssertionError, match="unsupported decomposition target"):
        build_experiment_config(LMExperimentConfig(**non_site_target))

    embedding_target = dict(
        raw,
        pd=dict(
            raw["pd"],
            decomposition_targets=[{"module_pattern": "embed_tokens", "C": 512}],
        ),
    )
    with pytest.raises(AssertionError, match="unsupported decomposition target"):
        build_experiment_config(LMExperimentConfig(**embedding_target))


def test_unsupported_model_family_refuses_and_supported_families_dispatch():
    """E23 (PARITY_MATRIX §11 row 2): only Llama-3.1-8B (`hf`/`hf_weights_in_vendored`
    → `TargetConfig`) and `LlamaSimpleMLP` (`pretrained` →
    `LlamaSimpleMLPTargetConfig`) convert; every other family is refused at convert
    time. The schema's `LMTargetSpec` discriminated union still validates a GPT-2 spec
    (it's a well-formed `kind`), so the refusal must come from `_resolve_target`'s
    per-family asserts, not pydantic."""
    from jax_single_pool.config import LlamaSimpleMLPTargetConfig, TargetConfig

    raw = _reference_lm_raw()

    def _converted_target(spec: dict[str, str]):
        cfg = build_experiment_config(
            LMExperimentConfig(**dict(raw, target=dict(raw["target"], spec=spec)))
        )
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
    a source `lr_schedule` that decays would silently flatten, so the conversion gate
    must refuse it (issue #646; matrix S13/S20)."""
    raw = _reference_lm_raw()
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
        build_experiment_config(LMExperimentConfig(**decaying_source))


def test_arbitrary_sites_with_per_site_c_convert():
    """Attention + MLP sites across non-contiguous layers with heterogeneous C —
    the general site space this trainer now implements."""
    raw = _reference_lm_raw()
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
    cfg = build_experiment_config(LMExperimentConfig(**general))
    assert cfg.target.sites == (
        SiteC("layers.18.self_attn.q_proj", 128),
        SiteC("layers.18.self_attn.v_proj", 32),
        SiteC("layers.20.mlp.up_proj", 64),
    )


def test_c49k_config_converts(tmp_path: Path):
    """The C49k/200k config (raw-HF target spec, bf16 weights_dtype, `model.`-prefixed
    site patterns) must convert cleanly."""
    converted, _raw = load_config(_stamped_config(tmp_path, CONFIGS / "llama8b_l18_C49k_200k.yaml"))
    assert converted.target.sites == mlp_family_site_cs(18, 18, 49152)
    assert converted.steps == 200000
    assert isinstance(converted.data, DataConfig)
    assert converted.data.global_batch == 512 and converted.data.seq_len == 2048
    assert converted.vu_optimizer.lr == 7e-05 and converted.ci_optimizer.lr == 7e-05
    assert converted.eval is not None and converted.eval.pgd is not None
    assert converted.wandb is not None and converted.wandb.entity is None


def test_nine_layer_config_converts(tmp_path: Path):
    """The launch-critical 9-layer chunkwise config: 27 MLP sites (layers 18-26), seq
    512, B=128, 40k steps, eps 1e-6, comp 1.5e-4 / ci_fn 5e-5, remat on."""
    converted, _raw = load_config(
        _stamped_config(tmp_path, CONFIGS / "llama8b_l18-26_9layer_chunkwise.yaml")
    )
    assert converted.run_name == "jax-l18-26-9L-seq512-b128-40k"
    assert len(converted.target.sites) == 27
    assert isinstance(converted.data, DataConfig)
    assert converted.data.seq_len == 512 and converted.data.global_batch == 128
    assert converted.steps == 40000
    assert converted.vu_optimizer.lr == 1.5e-4 and converted.ci_optimizer.lr == 5e-5
    assert converted.remat_recon_forwards is True
    imp = next(m for m in converted.loss_metrics if m.type == "ImportanceMinimalityLoss")
    assert imp.eps == 1e-6 and imp.coeff == 5e-6


def test_tms_config_converts(tmp_path: Path):
    """The TMS config dispatches to the TMS schema (structural `n_hidden` marker),
    builds the vendored TMS target (untied linear1/linear2 sites), the layerwise-MLP CI
    arch, and the positionless synthetic data config."""
    from jax_single_pool.ci_fn_mlp import MLPCIArch
    from jax_single_pool.config import TMSDataConfig, TMSTargetConfig

    converted, _raw = load_config(_stamped_config(tmp_path, CONFIGS / "tms_5-2.yaml"))
    assert isinstance(converted.target, TMSTargetConfig)
    assert converted.target.n_features == 5 and converted.target.n_hidden == 2
    assert converted.target.sites == (SiteC("linear1", 20), SiteC("linear2", 20))
    assert converted.target.pretrain_steps == 5000
    assert isinstance(converted.data, TMSDataConfig)
    assert converted.data.n_features == 5 and converted.data.global_batch == 4096
    assert converted.data.feature_probability == 0.05
    assert isinstance(converted.ci_fn, MLPCIArch) and converted.ci_fn.hidden_dims == (16,)
    assert converted.eval is None  # TMS validates via the in-loop target-CI metric
    # the shared recon-term builder accepts the TMS loss list (Stochastic + Layerwise + faith)
    build_recon_terms(
        converted.loss_metrics,
        tuple(sc.name for sc in converted.target.sites),
        converted.n_mask_samples,
        converted.sampling,
    )


def test_fp32_frozen_target_is_refused(tmp_path: Path):
    """A config requesting an fp32 frozen target must crash at the train/submit
    boundary — the bf16-only targets have no fp32 capability, and there is no silent
    downgrade (issue #727). Consumption paths (`load_run_dir_config`) ignore the field,
    so the guard lives in the build route, not a reload path."""
    raw = yaml.safe_load((CONFIGS / "llama8b_l18_C49k_200k.yaml").read_text())
    raw["run_id"] = RUN_ID
    cfg = LMExperimentConfig(**raw)
    cfg = cfg.model_copy(
        update={"target": cfg.target.model_copy(update={"weights_dtype": "float32"})}
    )
    with pytest.raises(AssertionError, match="weights_dtype"):
        assert_supported_weights_dtype(cfg)


def test_load_run_dir_config_rebuilds_runs(tmp_path: Path):
    """Tools read run dirs via `load_run_dir_config`; runs pin the single self-contained
    config as `config.yaml` (run.py's `_pin_config_copy`), and the rebuilt config must
    equal the launch-time conversion."""
    stamped = _stamped_config(tmp_path, CONFIGS / "llama8b_l18_C49k_200k.yaml")
    expected, _ = load_config(stamped)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(stamped.read_text())
    assert load_run_dir_config(run_dir) == expected


def test_run_id_required_and_drives_identity(tmp_path: Path):
    """The run dir and wandb id are the p-id (runs/<id>/ convention); the human name
    stays the wandb display name. Missing or malformed run_id refuses at build time."""
    cfg, _ = load_config(_stamped_config(tmp_path, CONFIGS / "llama8b_l18_C49k_200k.yaml"))
    assert cfg.run_id == RUN_ID
    assert cfg.run_dir.name == RUN_ID
    assert cfg.run_name == "jax-l18-C49k-200k"

    # the committed config carries no run_id (minted at submit) → build refuses
    with pytest.raises(AssertionError, match="run_id must be"):
        load_config(CONFIGS / "llama8b_l18_C49k_200k.yaml")

    bad_id = _stamped_config(tmp_path, CONFIGS / "llama8b_l18_C49k_200k.yaml")
    bad_id.write_text(bad_id.read_text().replace(RUN_ID, "run42"))
    with pytest.raises(AssertionError, match="run_id must be"):
        load_config(bad_id)
