"""The single-file run-config route — the trainer's only config surface.

The run id is NOT a config field: the launcher mints one and passes it to the build
helpers as an explicit arg (`RUN_ID` here), and the run dir derives from it
(`<data_root>/runs/<run_id>`)."""

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import pytest
import yaml
from pydantic import ValidationError

from param_decomp.core.components import SiteC
from param_decomp.core.configs import (
    CI_L0Config,
    CIHistogramsConfig,
    ComponentActivationDensityConfig,
    ImportanceMinimalityLossConfig,
    PersistentPGDReconLossConfig,
    PGDReconLossConfig,
)
from param_decomp.core.losses import scheduled_value_traced
from param_decomp.core.objective import build_objective
from param_decomp.core.recon import (
    persistent_configs,
)
from param_decomp.experiments.lm.config import (
    LMExperimentConfig,
    build_experiment_config,
    load_config,
)
from param_decomp.experiments.lm.eval_config import (
    ArithmeticCIGridConfig,
    CEandKLLossesConfig,
)
from param_decomp.experiments.lm.resolved import ResolvedLMData
from param_decomp.targets.glu_transformer import mlp_family_site_cs

CONFIGS = Path(__file__).parents[2] / "experiments" / "lm" / "configs"
RUN_ID = "p-0123abcd"
DATA_ROOT = Path("out")


def _reference_lm_raw():
    return yaml.safe_load((CONFIGS / "llama8b_l18_b128_cmp32.yaml").read_text())


def test_b128_config_converts():
    converted, authored = load_config(CONFIGS / "llama8b_l18_b128_cmp32.yaml", RUN_ID, DATA_ROOT)
    # The substrate rides on the authored config, not the engine's bundle.
    assert authored.runtime.world_size == 1 and authored.runtime.sharding == "zero1"
    assert converted.run.run_name == "jax-l18-b128-cmp32-from-torch"
    assert converted.pd.batch_size == 128 and converted.data is not None
    assert converted.target.sites == mlp_family_site_cs(18, 18, 24576)
    losses = build_objective(
        converted.pd.loss_metrics, tuple(sc.name for sc in converted.target.sites)
    )
    faith, imp = losses.faith, losses.imp
    assert isinstance(imp.cfg, ImportanceMinimalityLossConfig)
    assert faith.coeff == 1.0e3 and imp.cfg.pnorm.max_val == 2.0
    (ppgd,) = persistent_configs(losses.recon).values()
    assert isinstance(ppgd, PersistentPGDReconLossConfig)
    assert ppgd.n_warmup_steps == 2
    assert converted.pd.components_optimizer.grad_clip_norm == 0.01
    assert [t.name for t in losses.recon] == [
        "StochasticReconSubsetLoss",
        "PersistentPGDReconLoss",
    ]


def test_implicit_faithful_config_still_requires_exactly_one_faithfulness_term():
    raw = _reference_lm_raw()
    raw["pd"]["loss_metrics"] = [
        metric for metric in raw["pd"]["loss_metrics"] if metric["type"] != "FaithfulnessLoss"
    ]
    with pytest.raises(ValidationError, match="need exactly one FaithfulnessLoss"):
        LMExperimentConfig.model_validate(raw)


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
                "name": "fresh_probe",
                "init": "random",
                "source_shape": "c",
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
    cfg = LMExperimentConfig(**raw)
    assert cfg.eval is not None
    assert (cfg.eval.batch_size, cfg.eval.every, cfg.eval.n_steps) == (128, 1000, 1)
    assert (cfg.eval.slow_every, cfg.eval.slow_on_first_step) == (10000, True)
    assert any(
        isinstance(metric, CIHistogramsConfig)
        and metric.n_batches_accum == 7
        and metric.density_heatmap_n_bins == 40
        for metric in cfg.eval.metrics
    )
    assert any(
        isinstance(metric, CEandKLLossesConfig) and metric.rounding_threshold == 0.0
        for metric in cfg.eval.metrics
    )
    assert any(
        isinstance(metric, CI_L0Config) and metric.ci_alive_threshold == 0.0
        for metric in cfg.eval.metrics
    )
    assert any(
        isinstance(metric, ComponentActivationDensityConfig) and metric.ci_alive_threshold == 0.05
        for metric in cfg.eval.metrics
    )
    assert any(
        isinstance(metric, PGDReconLossConfig)
        and metric.name == "fresh_probe"
        and metric.n_steps == 20
        and metric.step_size == 0.1
        for metric in cfg.eval.metrics
    )
    assert "deferred" not in capsys.readouterr().out


def test_eval_data_resolves_to_a_separate_holdout():
    """`eval_data` is required and resolves somewhere other than the training shards."""
    raw = _reference_lm_raw()

    built = build_experiment_config(LMExperimentConfig(**raw), RUN_ID, DATA_ROOT)
    assert built.data.eval_dir != built.data.dir

    with pytest.raises(ValidationError):
        LMExperimentConfig(**dict(raw, data={"train": raw["data"]["train"]}))

    with pytest.raises(AssertionError, match="not a holdout"):
        same_both = {"train": raw["data"]["train"], "eval": raw["data"]["train"]}
        build_experiment_config(LMExperimentConfig(**dict(raw, data=same_both)), RUN_ID, DATA_ROOT)


def test_eval_pgd_threads_hidden_acts_reconstruction_into_built_probe():
    raw = _reference_lm_raw()
    raw["eval"] = {
        "batch_size": 1,
        "n_steps": 1,
        "every": 1,
        "slow_every": 1,
        "metrics": [
            {"type": "CEandKLLosses", "rounding_threshold": 0.0},
            {"type": "CI_L0", "groups": None, "ci_alive_threshold": 0.0},
            {
                "type": "PGDReconLoss",
                "init": "random",
                "source_shape": "c",
                "n_steps": 1,
                "step_size": 0.1,
                "hidden_acts_reconstruction": {"coeff": 1.0, "points": ["resid.19"]},
            },
        ],
    }
    authored = LMExperimentConfig(**raw)
    assert authored.eval is not None
    [metric] = [m for m in authored.eval.metrics if isinstance(m, PGDReconLossConfig)]
    assert metric.hidden_acts_reconstruction is not None
    assert metric.hidden_acts_reconstruction.coeff == 1.0
    assert metric.hidden_acts_reconstruction.points == ("resid.19",)
    build_experiment_config(authored, RUN_ID, DATA_ROOT)


def test_unsupported_settings_refuse():
    raw = _reference_lm_raw()

    # Non-matrix / cross-family site names are unrepresentable in the tiled spec: the cs
    # keys are the family's Literal matrix vocabulary, so these are rejected at PARSE, not
    # deferred to a convert-time assert.
    def _with_cs(cs: dict[str, int]):
        sites = dict(raw["decomposition"]["sites"], cs=cs)
        return dict(raw, decomposition=dict(raw["decomposition"], sites=sites))

    with pytest.raises(ValidationError):
        LMExperimentConfig(**_with_cs({"input_layernorm": 512}))

    with pytest.raises(ValidationError):
        LMExperimentConfig(**_with_cs({"embed_tokens": 512}))

    # cross-family matrix name (simple-MLP's c_fc in a GLU spec)
    with pytest.raises(ValidationError):
        LMExperimentConfig(**_with_cs({"c_fc": 512}))

    # family <-> target mismatch survives parse (both are well-formed) but is refused at
    # resolve: a simple_mlp c-spec against the GLU Llama-3.1 target.
    simple_mlp_sites = dict(
        raw,
        decomposition=dict(
            raw["decomposition"],
            sites={"kind": "simple_mlp", "layers": {"kind": "all"}, "cs": {"c_fc": 512}},
        ),
    )
    with pytest.raises(AssertionError, match="c-spec family"):
        build_experiment_config(LMExperimentConfig(**simple_mlp_sites), RUN_ID, DATA_ROOT)


def test_unsupported_model_variant_refuses_and_supported_variants_dispatch():
    """E23: only the `HF_MODEL_VARIANTS` models (`hf`/`hf_weights_in_vendored`
    → `TargetConfig`; Llama-3.1-8B and the registered Qwen3 checkpoints) and
    `LlamaSimpleMLP` (`pretrained` → `LlamaSimpleMLPTargetConfig`) convert; every other
    variant is refused at convert time. The schema's `LMTargetSpec` discriminated union
    still validates a GPT-2 spec (it's a well-formed `kind`), so the refusal must come
    from `_resolve_target`'s
    per-family asserts, not pydantic."""
    from param_decomp.experiments.lm.resolved import LlamaSimpleMLPTargetConfig, TargetConfig

    raw = _reference_lm_raw()

    def _converted_target(spec: dict[str, str]):
        cfg = build_experiment_config(
            LMExperimentConfig(**dict(raw, target=dict(raw["target"], spec=spec))),
            RUN_ID,
            DATA_ROOT,
        )
        return cfg.target

    vendored_llama = _converted_target(
        {
            "kind": "hf_weights_in_vendored",
            "model_class": "param_decomp.experiments.lm.vendored.llama_3_1.model.VendoredLlama",
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

    for model_name in (
        "Qwen/Qwen3-0.6B-Base",
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B-Base",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B-Base",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B-Base",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B-Base",
        "Qwen/Qwen3-14B",
    ):
        raw_hf_qwen3 = _converted_target(
            {
                "kind": "hf",
                "model_class": "transformers.Qwen3ForCausalLM",
                "model_name": model_name,
            }
        )
        assert isinstance(raw_hf_qwen3, TargetConfig)

    gpt2_hf = {
        "kind": "hf",
        "model_class": "transformers.GPT2LMHeadModel",
        "model_name": "gpt2",
    }
    with pytest.raises(AssertionError, match="transformers.GPT2LMHeadModel"):
        _converted_target(gpt2_hf)

    gpt2_vendored = {
        "kind": "hf_weights_in_vendored",
        "model_class": "param_decomp.experiments.lm.pretrain.models.gpt2.GPT2Simple",
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
        "model_class": "param_decomp.experiments.lm.pretrain.models.gpt2.GPT2Simple",
        "run_path": "goodfire/spd/runs/t-deadbeef",
    }
    with pytest.raises(AssertionError, match="GPT2Simple"):
        _converted_target(non_simple_mlp_pretrained)

    assert LlamaSimpleMLPTargetConfig is not None  # the `pretrained` happy-path type


def test_decaying_persistent_source_schedule_accepted_and_decays():
    """Issue #646 refused a decaying persistent-source `lr_schedule` because the JAX
    source LR was computed by a specialized `warmup_then_constant_lr` with no decay
    branch at all — a configured decay would have silently flattened. `adversary.py`'s
    `source_lr` now goes through the same generic `scheduled_value_traced` every other
    optimizer uses, so that gap is gone: the conversion accepts a decaying source
    schedule, and the schedule actually decays (not silently flattened)."""
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
                        lr_schedule={
                            "max_val": 0.01,
                            "points": [
                                {"at": 0.0, "frac": 1.0},
                                {"at": 1.0, "frac": 0.1, "interp": "cosine"},
                            ],
                        },
                    ),
                )
                if m["type"] == "PersistentPGDReconLoss"
                else m
                for m in raw["pd"]["loss_metrics"]
            ],
        ),
    )
    built = build_experiment_config(LMExperimentConfig(**decaying_source), RUN_ID, DATA_ROOT)
    losses = build_objective(built.pd.loss_metrics, tuple(sc.name for sc in built.target.sites))
    (cfg,) = persistent_configs(losses.recon).values()
    schedule = cfg.optimizer.lr_schedule
    assert not schedule.is_constant and schedule.max_val == 0.01

    total_steps = built.pd.steps
    start = scheduled_value_traced(jnp.float32(0), total_steps, schedule)
    end = scheduled_value_traced(jnp.float32(total_steps - 1), total_steps, schedule)
    assert float(start) == pytest.approx(schedule.max_val, rel=1e-3)
    assert float(end) == pytest.approx(schedule.max_val * 0.1, rel=1e-3)


def test_tiled_sites_with_per_matrix_c_convert():
    """Attention + MLP matrices with heterogeneous per-matrix C, tiled over a
    non-contiguous layer list — the general site space the tiled spec expresses. (Per-LAYER
    heterogeneous C is deliberately unrepresentable now: tiling is what makes the chunkwise
    CI fn's chunks homogeneous by construction.) Sites resolve in canonical order:
    layer-ascending, KIND_ORDER within a layer."""
    raw = _reference_lm_raw()
    general = dict(
        raw,
        decomposition=dict(
            raw["decomposition"],
            sites={
                "kind": "glu_transformer",
                "layers": {"kind": "list", "indices": [18, 20]},
                "cs": {"up": 64, "q": 128, "v": 32},
            },
        ),
    )
    cfg = build_experiment_config(LMExperimentConfig(**general), RUN_ID, DATA_ROOT)
    assert cfg.target.sites == (
        SiteC("layers.18.self_attn.q_proj", 128),
        SiteC("layers.18.self_attn.v_proj", 32),
        SiteC("layers.18.mlp.up_proj", 64),
        SiteC("layers.20.self_attn.q_proj", 128),
        SiteC("layers.20.self_attn.v_proj", 32),
        SiteC("layers.20.mlp.up_proj", 64),
    )


def test_all_block_resids_concatenates_one_tap_per_block():
    """`input_tap` default (`first_block_resid`): a multi-block chunk reads ONE tap — the
    residual entering its first block. `all_block_resids` concatenates one tap per block in
    the chunk, widening `ci_fn.input_dim` `blocks_per_chunk`x."""
    from param_decomp.core.ci_fn import Chunk, ChunkwiseTransformerCIArch

    raw = _reference_lm_raw()

    def _two_block_cfg(input_tap: str) -> dict[str, Any]:
        return dict(
            raw,
            decomposition=dict(
                sites={
                    "kind": "glu_transformer",
                    "layers": {"kind": "range", "start": 18, "end": 20},
                    "cs": {"down": 64},
                },
                ci=dict(raw["decomposition"]["ci"], blocks_per_chunk=2, input_tap=input_tap),
            ),
        )

    output_sites = ("layers.18.mlp.down_proj", "layers.19.mlp.down_proj")

    default_cfg = build_experiment_config(
        LMExperimentConfig(**_two_block_cfg("first_block_resid")), RUN_ID, DATA_ROOT
    )
    assert isinstance(default_cfg.ci_fn, ChunkwiseTransformerCIArch)
    assert default_cfg.ci_fn.chunks == (Chunk(input_taps=("resid.18",), output_sites=output_sites),)
    assert default_cfg.ci_fn.input_dim == 4096

    all_taps_cfg = build_experiment_config(
        LMExperimentConfig(**_two_block_cfg("all_block_resids")), RUN_ID, DATA_ROOT
    )
    assert isinstance(all_taps_cfg.ci_fn, ChunkwiseTransformerCIArch)
    assert all_taps_cfg.ci_fn.chunks == (
        Chunk(input_taps=("resid.18", "resid.19"), output_sites=output_sites),
    )
    assert all_taps_cfg.ci_fn.input_dim == 4096 * 2

    taps_cfg = build_experiment_config(
        LMExperimentConfig(**_two_block_cfg("all_block_taps")), RUN_ID, DATA_ROOT
    )
    assert isinstance(taps_cfg.ci_fn, ChunkwiseTransformerCIArch)
    assert taps_cfg.ci_fn.chunks == (
        Chunk(
            input_taps=(
                "attn_in.18",
                "attn_out.18",
                "mlp_in.18",
                "mlp_hidden.18",
                "attn_in.19",
                "attn_out.19",
                "mlp_in.19",
                "mlp_hidden.19",
            ),
            output_sites=output_sites,
        ),
    )
    assert taps_cfg.ci_fn.input_dim == (4096 + 4096 + 4096 + 14336) * 2


def test_c49k_config_converts():
    """The C49k/200k config (raw-HF target spec, bf16 weights_dtype, `model.`-prefixed
    site patterns) must convert cleanly."""
    converted, _raw = load_config(CONFIGS / "llama8b_l18_C49k_200k.yaml", RUN_ID, DATA_ROOT)
    assert converted.target.sites == mlp_family_site_cs(18, 18, 49152)
    assert converted.pd.steps == 200000
    assert isinstance(converted.data, ResolvedLMData)
    assert converted.pd.batch_size == 512 and converted.data.dir.name == "fineweb_llama_tok_2048"
    assert converted.pd.components_optimizer.lr_schedule.max_val == 7e-05
    assert converted.pd.ci_fn_optimizer.lr_schedule.max_val == 7e-05
    authored = LMExperimentConfig.model_validate(_raw)
    assert authored.eval is not None
    assert any(isinstance(metric, PGDReconLossConfig) for metric in authored.eval.metrics)
    assert converted.run.wandb is not None and converted.run.wandb.entity is None


def test_nine_layer_config_converts():
    """The launch-critical 9-layer chunkwise config: 27 MLP sites (layers 18-26), seq
    512, B=128, 40k steps, eps 1e-6, comp 1.5e-4 / ci_fn 5e-5, remat on."""
    converted, authored = load_config(
        CONFIGS / "llama8b_l18-26_9layer_chunkwise.yaml", RUN_ID, DATA_ROOT
    )
    assert converted.run.run_name == "jax-l18-26-9L-seq512-b128-40k"
    assert len(converted.target.sites) == 27
    assert isinstance(converted.data, ResolvedLMData)
    assert converted.data.dir.name == "fineweb_llama_tok_512" and converted.pd.batch_size == 128
    assert converted.pd.steps == 40000
    assert converted.pd.components_optimizer.lr_schedule.max_val == 1.5e-4
    assert converted.pd.ci_fn_optimizer.lr_schedule.max_val == 5e-5
    assert authored.runtime.remat_recon_forwards is True
    imp = next(m for m in converted.pd.loss_metrics if m.type == "ImportanceMinimalityLoss")
    assert imp.eps == 1e-6 and imp.coeff == 5e-6


def test_pinned_run_uses_current_schema(tmp_path: Path):
    """Pinned configs have the same strict contract as authored configs."""
    raw = yaml.safe_load((CONFIGS / "llama8b_l18_C49k_200k.yaml").read_text())
    raw["runtime"]["launch"] = "slurm"
    config = tmp_path / "launch_config.yaml"
    config.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValidationError):
        load_config(config, RUN_ID, DATA_ROOT)


def test_run_id_drives_identity_and_rejects_malformed():
    """The run dir and wandb id are the p-id (runs/<id>/ convention); the human name
    stays the wandb display name. The run id is the build helper's arg; a malformed id
    refuses at build time."""
    config = CONFIGS / "llama8b_l18_C49k_200k.yaml"
    cfg, _ = load_config(config, RUN_ID, DATA_ROOT)
    assert cfg.run.run_id == RUN_ID
    assert cfg.run.run_dir.name == RUN_ID
    assert cfg.run.run_name == "jax-l18-C49k-200k"

    with pytest.raises(AssertionError, match="run_id must be"):
        load_config(config, "run42", DATA_ROOT)


def test_arithmetic_ci_grid_metric_builds_to_arithmetic_eval_config():
    # In-tree coverage of the ArithmeticCIGrid authored-operation path (C49k enables
    # it by default; drop that entry and inject a known one so the assert is config-independent).
    raw = yaml.safe_load((CONFIGS / "llama8b_l18_C49k_200k.yaml").read_text())
    arithmetic_raw = next(
        metric for metric in raw["eval"]["metrics"] if metric["type"] == "ArithmeticCIGrid"
    )
    raw["eval"]["metrics"] = [
        metric for metric in raw["eval"]["metrics"] if metric["type"] != "ArithmeticCIGrid"
    ]
    raw["eval"]["metrics"].append(arithmetic_raw | {"a_range": [1, 50]})
    authored = LMExperimentConfig(**raw)
    assert authored.eval is not None
    arithmetic = next(
        metric for metric in authored.eval.metrics if isinstance(metric, ArithmeticCIGridConfig)
    )
    assert arithmetic.operation == "add"
    assert arithmetic.a_range == (1, 50)
    assert arithmetic.b_range == (1, 100)
    assert arithmetic.thresholds == [0.1]
    assert arithmetic.top_k == 24
    assert arithmetic.probe_metrics.ce_kl.rounding_threshold == 0.0
    assert arithmetic.probe_metrics.ci_l0.ci_alive_threshold == 0.0
    assert arithmetic.probe_metrics.fresh_pgd is not None
    assert arithmetic.probe_metrics.fresh_pgd.n_steps == 20


@pytest.mark.parametrize(
    ("field", "value"),
    (("coeff", None), ("init", "random"), ("source_shape", "c"), ("type", "PGDReconLoss")),
)
def test_arithmetic_probe_rejects_unexecuted_fresh_pgd_fields(field: str, value: object):
    raw = yaml.safe_load((CONFIGS / "llama8b_l18_C49k_200k.yaml").read_text())
    arithmetic = next(
        metric for metric in raw["eval"]["metrics"] if metric["type"] == "ArithmeticCIGrid"
    )
    arithmetic["probe_metrics"]["fresh_pgd"][field] = value

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        LMExperimentConfig.model_validate(raw)


def test_placement_table_parses_typed_and_fails_closed():
    from param_decomp.core.configs import PlacementTableConfig

    table: dict[str, Any] = {
        "components": {
            "optimizer_state": {"stack": "replicate", "d_in": "fsdp", "d_out": "fsdp", "C": "tp"},
            "compute_weights": {"d_in": "fsdp", "d_out": "fsdp", "C": "tp"},
            "faithfulness_weights": {
                "stack": "replicate",
                "d_in": "fsdp",
                "d_out": "fsdp",
                "C": "tp",
            },
            "faithfulness_deltas": {"stack": "replicate", "d_out": "fsdp"},
            "operands": {"C": "tp"},
            "ns_compute": {"stack": "replicate"},
        },
        "ci_fn": {
            "attention": {
                "optimizer_state": {},
                "compute_weights": {},
                "operands": {},
                "ns_compute": {},
            },
            "ffn": {
                "optimizer_state": {},
                "compute_weights": {},
                "operands": {},
                "ns_compute": {},
            },
            "input": {
                "optimizer_state": {},
                "compute_weights": {},
                "operands": {},
                "ns_compute": {},
            },
            "output": {
                "optimizer_state": {},
                "compute_weights": {},
                "operands": {},
                "ns_compute": {},
            },
            "vectors": {},
            "activations": {},
        },
        "activations": {
            "external": {"batch": ["replicate", "fsdp"]},
            "component": {"batch": ["replicate", "fsdp"], "C": "tp"},
        },
        "target": {
            "embedding": {"persist": {"d_model": "fsdp"}, "operand": {}},
            "normalization": {},
            "position_encoding": {},
            "column": {
                "persist": {"d_in": "fsdp", "d_out": "tp"},
                "operand": {"d_out": "tp"},
                "input": "external",
                "output": "intermediate",
            },
            "row": {
                "persist": {"d_out": "fsdp", "d_in": "tp"},
                "operand": {"d_in": "tp"},
                "input": "intermediate",
                "output": "external",
            },
            "output": {"persist": {"d_model": "fsdp"}, "operand": {}},
            "intermediate": {
                "batch": ["replicate", "fsdp"],
                "feature": "tp",
                "q_head": "tp",
                "kv_head": "tp",
            },
            "component": {"input": "external", "output": "external"},
        },
    }
    full = PlacementTableConfig.model_validate(table)
    assert full.components.optimizer_state == {
        "stack": "replicate",
        "d_in": "fsdp",
        "d_out": "fsdp",
        "C": "tp",
    }

    # per-group fallback rows are unrepresentable: the closed schema refuses them at parse
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PlacementTableConfig.model_validate(
            {
                **table,
                "components": {
                    **table["components"],
                    "optimizer_state_fallback": {"d_in": "fsdp", "C": ["tp", "replicate"]},
                },
            }
        )

    strict = PlacementTableConfig.model_validate(
        {
            "components": {
                "optimizer_state": {},
                "compute_weights": {},
                "faithfulness_weights": {},
                "faithfulness_deltas": {},
                "operands": {},
                "ns_compute": {},
            },
            "ci_fn": {
                "attention": {
                    "optimizer_state": {},
                    "compute_weights": {},
                    "operands": {},
                    "ns_compute": {},
                },
                "ffn": {
                    "optimizer_state": {},
                    "compute_weights": {},
                    "operands": {},
                    "ns_compute": {},
                },
                "input": {
                    "optimizer_state": {},
                    "compute_weights": {},
                    "operands": {},
                    "ns_compute": {},
                },
                "output": {
                    "optimizer_state": {},
                    "compute_weights": {},
                    "operands": {},
                    "ns_compute": {},
                },
                "vectors": {},
                "activations": {},
            },
            "activations": {"external": {}, "component": {}},
            "target": {
                "embedding": {"persist": {}, "operand": {}},
                "normalization": {},
                "position_encoding": {},
                "column": {
                    "persist": {},
                    "operand": {},
                    "input": "external",
                    "output": "intermediate",
                },
                "row": {
                    "persist": {},
                    "operand": {},
                    "input": "intermediate",
                    "output": "external",
                },
                "output": {"persist": {}, "operand": {}},
                "intermediate": {},
                "component": {"input": "external", "output": "external"},
            },
        }
    )
    assert strict.components.optimizer_state == {}

    # the row vocabulary is CLOSED: unknown rows die at parse, at either level
    with pytest.raises(ValidationError):
        PlacementTableConfig.model_validate({**table, "optim/muon.ns": {}})
    with pytest.raises(ValidationError):
        PlacementTableConfig.model_validate(
            {
                "components": {**table["components"], "persist.zero1": {}},
                "ci_fn": table["ci_fn"],
                "activations": table["activations"],
                "target": table["target"],
            }
        )
    # required rows are required fields, not a runtime manifest check
    with pytest.raises(ValidationError):
        PlacementTableConfig.model_validate(
            {
                "components": {"optimizer_state": {}},
                "activations": {"external": {}, "component": {}},
            }
        )
    # a malformed rule value (axis -> non-mesh-axes) dies at parse too
    with pytest.raises(ValidationError):
        PlacementTableConfig.model_validate(
            {
                "components": {
                    "optimizer_state": {"d_in": 3},
                    "compute_weights": {},
                    "faithfulness_weights": {},
                    "faithfulness_deltas": {},
                    "operands": {},
                    "ns_compute": {},
                },
                "ci_fn": table["ci_fn"],
                "activations": {"external": {}, "component": {}},
                "target": {
                    "embedding": {"persist": {}, "operand": {}},
                    "normalization": {},
                    "position_encoding": {},
                    "column": {
                        "persist": {},
                        "operand": {},
                        "input": "external",
                        "output": "intermediate",
                    },
                    "row": {
                        "persist": {},
                        "operand": {},
                        "input": "intermediate",
                        "output": "external",
                    },
                    "output": {"persist": {}, "operand": {}},
                    "intermediate": {},
                    "component": {"input": "external", "output": "external"},
                },
            }
        )


def test_attention_eval_geometry_is_target_owned_not_configurable() -> None:
    raw = yaml.safe_load((CONFIGS / "llama8b_l18_C49k_200k.yaml").read_text())
    raw["eval"]["metrics"].append(
        {
            "type": "CIMaskedAttnPatternsReconLoss",
            "n_heads": 8,
            "q_proj_path": "q_proj",
            "k_proj_path": "k_proj",
        }
    )

    with pytest.raises(ValidationError, match="extra_forbidden"):
        LMExperimentConfig.model_validate(raw)
