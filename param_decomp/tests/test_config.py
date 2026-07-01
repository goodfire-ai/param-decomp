"""The single-file run-config route — the trainer's only config surface.

The run id is NOT a config field: the launcher mints one and passes it to the build
helpers as an explicit arg (`RUN_ID` here), and the run dir derives from it
(`PARAM_DECOMP_OUT_DIR/runs/<run_id>`)."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from param_decomp.built_run import DataConfig
from param_decomp.components import SiteC
from param_decomp.configs import (
    ImportanceMinimalityLossConfig,
    PDConfig,
    PersistentPGDReconLossConfig,
)
from param_decomp.recon import (
    build_loss_terms,
    persistent_configs,
)
from param_decomp.targets.llama8b import mlp_family_site_cs
from param_decomp_lab.experiments.lm.config import (
    LMExperimentConfig,
    assert_supported_weights_dtype,
    build_experiment_config,
    load_config,
    load_run_dir_config,
)

CONFIGS = Path(__file__).parent.parent / "configs"
RUN_ID = "p-0123abcd"


def _reference_lm_raw():
    return yaml.safe_load((CONFIGS / "llama8b_l18_b128_cmp32.yaml").read_text())


def test_removed_pdconfig_fields_strip_from_stored_configs_but_reject_bad_values():
    # Provenance shim: stored run config.yamls carry sigmoid_type / use_delta_component /
    # tied_weights / identity_decomposition_targets (now removed from PDConfig). They strip on
    # load when carrying their only-ever-supported value; a non-supported value is rejected.
    pd = yaml.safe_load((CONFIGS / "llama8b_l18_b128_cmp32.yaml").read_text())["pd"]
    PDConfig.model_validate(pd)  # clean (no dead keys)
    PDConfig.model_validate(
        {
            **pd,
            "sigmoid_type": "leaky_hard",
            "use_delta_component": True,
            "tied_weights": None,
            "identity_decomposition_targets": None,
        }
    )  # stored config carrying the dead keys -> stripped + loads
    for bad in (
        {"sigmoid_type": "swish_hard"},
        {"use_delta_component": False},
        {"tied_weights": [["a", "b"]]},
        {"identity_decomposition_targets": [{"module_pattern": "x", "C": 1}]},
    ):
        with pytest.raises((ValidationError, AssertionError)):
            PDConfig.model_validate({**pd, **bad})


def test_legacy_top_level_n_mask_samples_pushes_onto_stochastic_terms():
    # Provenance shim: `n_mask_samples` moved from a `pd`-level knob onto the stochastic
    # loss configs. A stored config carrying it at `pd` level loads and its value lands on
    # each stochastic recon term that doesn't set its own; an explicit per-term value wins.
    pd = yaml.safe_load((CONFIGS / "llama8b_l18_b128_cmp32.yaml").read_text())["pd"]
    pd = {
        **pd,
        "n_mask_samples": 4,
        "loss_metrics": [
            {"type": "FaithfulnessLoss", "coeff": 1.0},
            {"type": "ImportanceMinimalityLoss", "coeff": 1.0, "pnorm": 2.0},
            {"type": "StochasticReconLoss", "coeff": 1.0},
            {"type": "StochasticReconSubsetLoss", "coeff": 1.0, "n_mask_samples": 7},
        ],
    }
    validated = PDConfig.model_validate(pd)
    assert not hasattr(validated, "n_mask_samples")
    by_type = {m.type: m for m in validated.loss_metrics}
    assert by_type["StochasticReconLoss"].n_mask_samples == 4  # pyright: ignore[reportAttributeAccessIssue]
    assert by_type["StochasticReconSubsetLoss"].n_mask_samples == 7  # pyright: ignore[reportAttributeAccessIssue]


def test_b128_config_converts():
    converted, raw = load_config(CONFIGS / "llama8b_l18_b128_cmp32.yaml", RUN_ID)
    assert raw["pd"]["batch_size"] == 128
    assert converted.run.run_name == "jax-l18-b128-cmp32-from-torch"
    assert converted.data is not None and converted.data.global_batch == 128
    assert converted.target.sites == mlp_family_site_cs(18, 18, 24576)
    losses = build_loss_terms(
        converted.pd.loss_metrics, tuple(sc.name for sc in converted.target.sites)
    )
    faith, imp = losses.faith, losses.imp
    assert isinstance(imp.cfg, ImportanceMinimalityLossConfig)
    assert faith.coeff == 1e5 and imp.cfg.pnorm == 2.0
    (ppgd,) = persistent_configs(losses.recon).values()
    assert isinstance(ppgd, PersistentPGDReconLossConfig)
    assert ppgd.n_warmup_steps == 2
    assert converted.pd.components_optimizer.grad_clip_norm == 0.01
    assert [t.name for t in losses.recon] == [
        "StochasticReconSubsetLoss",
        "PersistentPGDReconLoss",
    ]


def test_eval_block_maps_slow_tier_and_defers_offline_only_metrics(
    capsys: pytest.CaptureFixture[str],
):
    raw = _reference_lm_raw()
    raw["eval"] = {
        "batch_size": 128,
        "every": 1000,
        "n_steps": 1,
        "slow_every": 10000,
        "slow_on_first_step": True,
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
            {"type": "CIHistograms", "n_batches_accum": 7, "density_heatmap_n_bins": 40},
            # distinct cutoff: pins that density reads its OWN ci_alive_threshold, not CI_L0's
            {"type": "ComponentActivationDensity", "ci_alive_threshold": 0.05},  # slow tier
            {"type": "IdentityCIError", "identity_ci": None, "dense_ci": None},  # in-loop slow
            {"type": "UVPlots", "identity_patterns": None, "dense_patterns": None},  # in-loop slow
        ],
    }
    cfg = build_experiment_config(LMExperimentConfig(**raw), RUN_ID)
    assert cfg.eval is not None
    assert (cfg.eval.batch_size, cfg.eval.every, cfg.eval.n_steps) == (128, 1000, 1)
    assert (cfg.eval.slow_every, cfg.eval.slow_on_first_step) == (10000, True)
    assert cfg.eval.slow_n_batches_accum == 7  # read off the CIHistograms metric
    assert cfg.eval.density_heatmap_n_bins == 40  # opt-in per-token CI density heatmap
    assert cfg.eval.rounding_threshold == 0.0
    assert cfg.eval.l0_ci_alive_threshold == 0.0 and cfg.eval.density_ci_alive_threshold == 0.05
    assert cfg.eval.pgd is not None and (cfg.eval.pgd.n_steps, cfg.eval.pgd.step_size) == (20, 0.1)
    # the plot / permutation / UV / identity metrics all run in-loop — `_eval` accepts them
    # without raising, and nothing is deferred (no offline path)
    assert "deferred" not in capsys.readouterr().out


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
        build_experiment_config(LMExperimentConfig(**hidden_acts_training_loss), RUN_ID)

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
        build_experiment_config(LMExperimentConfig(**non_site_target), RUN_ID)

    embedding_target = dict(
        raw,
        pd=dict(
            raw["pd"],
            decomposition_targets=[{"module_pattern": "embed_tokens", "C": 512}],
        ),
    )
    with pytest.raises(AssertionError, match="unsupported decomposition target"):
        build_experiment_config(LMExperimentConfig(**embedding_target), RUN_ID)


def test_unsupported_model_family_refuses_and_supported_families_dispatch():
    """E23: only Llama-3.1-8B (`hf`/`hf_weights_in_vendored`
    → `TargetConfig`) and `LlamaSimpleMLP` (`pretrained` →
    `LlamaSimpleMLPTargetConfig`) convert; every other family is refused at convert
    time. The schema's `LMTargetSpec` discriminated union still validates a GPT-2 spec
    (it's a well-formed `kind`), so the refusal must come from `_resolve_target`'s
    per-family asserts, not pydantic."""
    from param_decomp_lab.experiments.lm.config import LlamaSimpleMLPTargetConfig, TargetConfig

    raw = _reference_lm_raw()

    def _converted_target(spec: dict[str, str]):
        cfg = build_experiment_config(
            LMExperimentConfig(**dict(raw, target=dict(raw["target"], spec=spec))), RUN_ID
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
        build_experiment_config(LMExperimentConfig(**decaying_source), RUN_ID)


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
    cfg = build_experiment_config(LMExperimentConfig(**general), RUN_ID)
    assert cfg.target.sites == (
        SiteC("layers.18.self_attn.q_proj", 128),
        SiteC("layers.18.self_attn.v_proj", 32),
        SiteC("layers.20.mlp.up_proj", 64),
    )


def test_c49k_config_converts():
    """The C49k/200k config (raw-HF target spec, bf16 weights_dtype, `model.`-prefixed
    site patterns) must convert cleanly."""
    converted, _raw = load_config(CONFIGS / "llama8b_l18_C49k_200k.yaml", RUN_ID)
    assert converted.target.sites == mlp_family_site_cs(18, 18, 49152)
    assert converted.pd.steps == 200000
    assert isinstance(converted.data, DataConfig)
    assert converted.data.global_batch == 512 and converted.data.seq_len == 2048
    assert converted.pd.components_optimizer.lr_schedule.start_val == 7e-05
    assert converted.pd.ci_fn_optimizer.lr_schedule.start_val == 7e-05
    assert converted.eval is not None and converted.eval.pgd is not None
    assert converted.run.wandb is not None and converted.run.wandb.entity is None


def test_nine_layer_config_converts():
    """The launch-critical 9-layer chunkwise config: 27 MLP sites (layers 18-26), seq
    512, B=128, 40k steps, eps 1e-6, comp 1.5e-4 / ci_fn 5e-5, remat on."""
    converted, _raw = load_config(CONFIGS / "llama8b_l18-26_9layer_chunkwise.yaml", RUN_ID)
    assert converted.run.run_name == "jax-l18-26-9L-seq512-b128-40k"
    assert len(converted.target.sites) == 27
    assert isinstance(converted.data, DataConfig)
    assert converted.data.seq_len == 512 and converted.data.global_batch == 128
    assert converted.pd.steps == 40000
    assert converted.pd.components_optimizer.lr_schedule.start_val == 1.5e-4
    assert converted.pd.ci_fn_optimizer.lr_schedule.start_val == 5e-5
    assert converted.runtime.remat_recon_forwards is True
    imp = next(m for m in converted.pd.loss_metrics if m.type == "ImportanceMinimalityLoss")
    assert imp.eps == 1e-6 and imp.coeff == 5e-6


def test_fp32_frozen_target_is_refused():
    """A config requesting an fp32 frozen target must crash at the train/submit
    boundary — the bf16-only targets have no fp32 capability, and there is no silent
    downgrade (issue #727). Consumption paths (`load_run_dir_config`) ignore the field,
    so the guard lives in the build route, not a reload path."""
    raw = yaml.safe_load((CONFIGS / "llama8b_l18_C49k_200k.yaml").read_text())
    cfg = LMExperimentConfig(**raw)
    cfg = cfg.model_copy(
        update={"target": cfg.target.model_copy(update={"weights_dtype": "float32"})}
    )
    with pytest.raises(AssertionError, match="weights_dtype"):
        assert_supported_weights_dtype(cfg)


def test_load_run_dir_config_rebuilds_runs(tmp_path: Path):
    """Tools read run dirs via `load_run_dir_config`; runs pin the single self-contained
    config as `config.yaml` (run.py's `_pin_config_copy`), and the rebuilt config must
    equal the launch-time conversion. The run id is the run-dir name."""
    config = CONFIGS / "llama8b_l18_C49k_200k.yaml"
    expected, _ = load_config(config, RUN_ID)
    run_dir = tmp_path / RUN_ID
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(config.read_text())
    assert load_run_dir_config(run_dir) == expected


def test_run_id_drives_identity_and_rejects_malformed():
    """The run dir and wandb id are the p-id (runs/<id>/ convention); the human name
    stays the wandb display name. The run id is the build helper's arg; a malformed id
    refuses at build time."""
    config = CONFIGS / "llama8b_l18_C49k_200k.yaml"
    cfg, _ = load_config(config, RUN_ID)
    assert cfg.run.run_id == RUN_ID
    assert cfg.run.run_dir.name == RUN_ID
    assert cfg.run.run_name == "jax-l18-C49k-200k"

    with pytest.raises(AssertionError, match="run_id must be"):
        load_config(config, "run42")
