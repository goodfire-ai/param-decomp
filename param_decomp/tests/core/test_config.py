"""The single-file run-config route — the trainer's only config surface.

The run id is NOT a config field: the entry point mints one and passes it to the build
helpers as an explicit arg (`RUN_ID` here), and the run dir derives from it
(`<data_root>/runs/<run_id>`)."""

from pathlib import Path
from typing import Any, Literal

import jax.numpy as jnp
import pytest
import yaml
from pydantic import TypeAdapter, ValidationError

from param_decomp.core.ci_fn.implementations.chunkwise.arch import ChunkwiseTransformerCIFnArch
from param_decomp.core.components import DenseFactorization, SiteC, SiteSpec
from param_decomp.core.configs import (
    AdamPGDConfig,
    AnyLossMetricConfig,
    BatchSourceShape,
    CI_L0Config,
    CIHistogramsConfig,
    ComponentActivationDensityConfig,
    EvalPGDReconLossConfig,
    ImportanceMinimalityLossConfig,
    MergedStochasticSubsetPPGDReconLossConfig,
    PersistentPGDReconLossConfig,
    PGDInitStrategy,
    PGDReconLossConfig,
    PGDReconSubsetLossConfig,
    SlowPGDReconLossConfig,
    TargetedLossMetricConfig,
)
from param_decomp.core.objective import build_objective, build_recon_terms
from param_decomp.core.recon import (
    FreshPGDSources,
    persistent_configs,
)
from param_decomp.core.recon_eval import FreshPGDAttack, fresh_pgd_probe
from param_decomp.core.runtime_schedule import scheduled_value_traced
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.core.world_size import SingleNode
from param_decomp.experiments.eval_config import AnyEvalMetricConfig
from param_decomp.experiments.lm.config import (
    HFTarget,
    HFWeightsInVendored,
    LMDecompositionConfig,
    LMExperimentConfig,
    LMTargetConfig,
    build_experiment_config,
    load_config,
    resolve_decomposition,
)
from param_decomp.experiments.lm.eval_config import (
    ArithmeticCIGridConfig,
    CEandKLLossesConfig,
)
from param_decomp.experiments.lm.resolved import ResolvedLMData
from param_decomp.targets.transformer import mlp_family_site_cs
from param_decomp.tests.lm_configs import arithmetic_metric, l18_mlp_raw, llama_raw

CONFIGS = Path(__file__).parents[2] / "experiments" / "lm" / "configs"
RUN_ID = "p-0123abcd"
DATA_ROOT = Path("out")


def _reference_lm_raw():
    return l18_mlp_raw()


def test_l18_mlp_variant_converts():
    authored = LMExperimentConfig.model_validate(l18_mlp_raw())
    converted = build_experiment_config(authored, RUN_ID, DATA_ROOT)
    # The substrate rides on the authored config, not the engine's bundle.
    assert (
        authored.runtime.world_size == SingleNode(n_gpus=1) and authored.runtime.sharding == "owner"
    )
    assert converted.run.run_name == "test-l18-mlp"
    assert converted.pd.batch_size == 128 and converted.data is not None
    assert converted.target.sites == mlp_family_site_cs(18, 18, 24576)
    losses = build_objective(
        converted.pd.loss_metrics,
        resolve_decomposition(authored.target, authored.decomposition, DATA_ROOT).site_specs,
    )
    faith, imp = losses.faith, losses.minimality
    assert float(faith.coeff.at(jnp.float32(0))) == 1.0e3
    assert imp.gamma.max_val == 1.0
    (ppgd,) = persistent_configs(losses.recon).values()
    assert isinstance(ppgd, PersistentPGDReconLossConfig)
    assert ppgd.n_warmup_steps == 2
    assert converted.pd.components_optimizer.grad_clip_norm == 1.0
    assert [t.name for t in losses.recon] == [
        "StochasticReconSubsetLoss",
        "PersistentPGDReconLoss",
    ]


def test_normalize_at_one_parses_from_run_config():
    raw = _reference_lm_raw()
    imp = next(m for m in raw["pd"]["loss_metrics"] if m["type"] == "ImportanceMinimalityLoss")
    imp["normalize_at_one"] = True
    authored = LMExperimentConfig.model_validate(raw)
    parsed = next(m for m in authored.pd.loss_metrics if m.type == "ImportanceMinimalityLoss")
    assert isinstance(parsed, ImportanceMinimalityLossConfig)
    assert parsed.normalize_at_one is True


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
        isinstance(metric, EvalPGDReconLossConfig)
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


@pytest.mark.parametrize(
    "kind,steps,attack,read_out_names",
    [
        ("PGDReconLoss", {"n_steps": 1}, FreshPGDAttack(0.1, (1,)), ("PGDReconLoss_1step",)),
        (
            "SlowPGDReconLoss",
            {"read_out_steps": [1, 4]},
            FreshPGDAttack(0.1, (1, 4)),
            ("SlowPGDReconLoss_1step", "SlowPGDReconLoss_4step"),
        ),
    ],
)
def test_eval_pgd_threads_hidden_acts_reconstruction_into_built_probe(
    kind: str,
    steps: dict[str, int | list[int]],
    attack: FreshPGDAttack,
    read_out_names: tuple[str, ...],
):
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
                "type": kind,
                "init": "random",
                "source_shape": "c",
                **steps,
                "step_size": 0.1,
                "auxiliaries": [
                    {
                        "name": "hidden_acts_reconstruction",
                        "coeff": 0.0,
                        "comparisons": [
                            {"capture": "resid.19", "distance": "relative_squared_error"}
                        ],
                    }
                ],
            },
        ],
    }
    authored = LMExperimentConfig(**raw)
    assert authored.eval is not None
    [metric] = [
        m
        for m in authored.eval.metrics
        if isinstance(m, (EvalPGDReconLossConfig, SlowPGDReconLossConfig))
    ]
    (auxiliary,) = metric.auxiliaries
    assert auxiliary.coeff == 0.0
    assert tuple(comparison.capture for comparison in auxiliary.comparisons) == ("resid.19",)
    probe = fresh_pgd_probe(metric)
    assert probe.attack == attack and probe.read_out_names == read_out_names
    assert probe.reconstruction_capture_keys == frozenset({"resid.19"})
    build_experiment_config(authored, RUN_ID, DATA_ROOT)


def test_training_hidden_acts_reconstruction_refuses_measurement_only_coefficient():
    raw = _reference_lm_raw()
    recon = next(metric for metric in raw["pd"]["loss_metrics"] if "Recon" in metric["type"])
    recon["auxiliaries"] = [
        {
            "name": "hidden_acts_reconstruction",
            "coeff": 0.0,
            "comparisons": [{"capture": "resid.19", "distance": "relative_squared_error"}],
        }
    ]

    with pytest.raises(ValidationError, match="zero is reserved for eval-only measurement"):
        LMExperimentConfig.model_validate(raw)


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


TargetKind = Literal[
    "HFTarget", "HFWeightsInVendored", "PretrainedTarget", "PretrainedQwen35MoeTarget"
]
SitesKind = Literal["GluTransformerCSpec", "SimpleMlpCSpec", "Qwen36MoeCSpec"]


def _repo_config(name: str) -> LMExperimentConfig:
    return LMExperimentConfig.model_validate(yaml.safe_load((CONFIGS / name).read_text()))


def _repo_target(kind: TargetKind) -> LMTargetConfig:
    match kind:
        case "HFTarget":
            return _repo_config("llama3_1_8b.yaml").target
        case "HFWeightsInVendored":
            hf = _repo_config("llama3_1_8b.yaml").target
            assert isinstance(hf.spec, HFTarget)
            vendored = HFWeightsInVendored(
                model_class="param_decomp.experiments.lm.vendored.llama_3_1.model.VendoredLlama",
                model_name=hf.spec.model_name,
            )
            return hf.model_copy(update={"spec": vendored})
        case "PretrainedTarget":
            return _repo_config("pile_llama_simple_mlp-4L.yaml").target
        case "PretrainedQwen35MoeTarget":
            return _repo_config("pile_qwen3_5_moe-4L.yaml").target


def _repo_decomposition(kind: SitesKind) -> LMDecompositionConfig:
    match kind:
        case "GluTransformerCSpec":
            return _repo_config("llama3_1_8b.yaml").decomposition
        case "SimpleMlpCSpec":
            return _repo_config("pile_llama_simple_mlp-4L.yaml").decomposition
        case "Qwen36MoeCSpec":
            return _repo_config("pile_qwen3_5_moe-4L.yaml").decomposition


@pytest.mark.parametrize(
    ("target_kind", "sites_kind"),
    [
        ("PretrainedTarget", "GluTransformerCSpec"),
        ("PretrainedQwen35MoeTarget", "GluTransformerCSpec"),
        ("HFTarget", "SimpleMlpCSpec"),
        ("HFWeightsInVendored", "SimpleMlpCSpec"),
        ("PretrainedQwen35MoeTarget", "SimpleMlpCSpec"),
        ("HFWeightsInVendored", "Qwen36MoeCSpec"),
        ("PretrainedTarget", "Qwen36MoeCSpec"),
    ],
)
def test_resolution_refuses_every_illegal_target_sites_pair(
    target_kind: TargetKind, sites_kind: SitesKind
):
    """Target spec and c-spec parse independently, so the schema admits their full
    product; resolution refuses every pair outside the legal combinations, by name."""
    with pytest.raises(ValueError, match=f"^{target_kind} can't decompose {sites_kind} sites$"):
        resolve_decomposition(_repo_target(target_kind), _repo_decomposition(sites_kind), DATA_ROOT)


def test_unsupported_model_variant_refuses_and_supported_variants_dispatch():
    """E23: only the `HF_MODEL_VARIANTS` models (`hf`/`hf_weights_in_vendored`
    → `TargetConfig`; Llama-3.1-8B and the registered Qwen3 checkpoints),
    `LlamaSimpleMLP` (`pretrained` → `LlamaSimpleMLPTargetConfig`) and the qwen36_moe
    sources (`hf` 35B / `pretrained_qwen35_moe` → `Qwen36MoeTargetConfig`, pinned by
    `test_qwen36_wiring`) convert; every other variant is refused at convert time. The
    schema's `LMTargetSpec` discriminated union still validates a GPT-2 spec (it's a
    well-formed `kind`), so the refusal must come from `resolve_decomposition`'s
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

    # `pretrained` with a simple_mlp c-spec dispatches into the LlamaSimpleMLP branch
    # (proven by the model-class assert firing there); a non-LlamaSimpleMLP pretrained spec
    # refuses before any disk access.
    non_simple_mlp_pretrained = dict(
        raw,
        target=dict(
            raw["target"],
            spec={
                "kind": "pretrained",
                "model_class": "param_decomp.experiments.lm.pretrain.models.gpt2.GPT2Simple",
                "run_path": "goodfire/spd/runs/t-deadbeef",
            },
        ),
        decomposition=dict(
            raw["decomposition"],
            sites={"kind": "simple_mlp", "layers": {"kind": "all"}, "cs": {"c_fc": 512}},
        ),
    )
    with pytest.raises(AssertionError, match="GPT2Simple"):
        build_experiment_config(LMExperimentConfig(**non_simple_mlp_pretrained), RUN_ID, DATA_ROOT)

    assert LlamaSimpleMLPTargetConfig is not None  # the `pretrained` happy-path type


def test_decaying_persistent_source_schedule_accepted_and_decays():
    """A decaying persistent-source `lr_schedule` must not be flattened: the JAX
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
    authored = LMExperimentConfig(**decaying_source)
    built = build_experiment_config(authored, RUN_ID, DATA_ROOT)
    losses = build_objective(
        built.pd.loss_metrics,
        resolve_decomposition(authored.target, authored.decomposition, DATA_ROOT).site_specs,
    )
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
    from param_decomp.core.ci_fn.implementations.chunkwise.arch import (
        Chunk,
        ChunkwiseTransformerCIFnArch,
    )

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
    assert isinstance(default_cfg.ci_fn, ChunkwiseTransformerCIFnArch)
    assert default_cfg.ci_fn.chunks == (Chunk(input_taps=("resid.18",), output_sites=output_sites),)
    assert default_cfg.ci_fn.input_dim == 4096

    all_taps_cfg = build_experiment_config(
        LMExperimentConfig(**_two_block_cfg("all_block_resids")), RUN_ID, DATA_ROOT
    )
    assert isinstance(all_taps_cfg.ci_fn, ChunkwiseTransformerCIFnArch)
    assert all_taps_cfg.ci_fn.chunks == (
        Chunk(input_taps=("resid.18", "resid.19"), output_sites=output_sites),
    )
    assert all_taps_cfg.ci_fn.input_dim == 4096 * 2

    taps_cfg = build_experiment_config(
        LMExperimentConfig(**_two_block_cfg("all_block_taps")), RUN_ID, DATA_ROOT
    )
    assert isinstance(taps_cfg.ci_fn, ChunkwiseTransformerCIFnArch)
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


def test_canonical_llama_config_converts():
    converted, authored = load_config(CONFIGS / "llama3_1_8b.yaml", RUN_ID, DATA_ROOT)
    assert len(converted.target.sites) == 32 * 7
    assert converted.pd.steps == 50000
    assert converted.pd.batch_size == 1024
    assert isinstance(converted.data, ResolvedLMData)
    assert converted.data.dir.name == "fineweb350bt_llama3_docs512_r9bb295dd_seed0_train48_v2"
    assert authored.eval is not None
    assert any(isinstance(metric, EvalPGDReconLossConfig) for metric in authored.eval.metrics)


def test_nine_layer_variant_converts():
    raw = l18_mlp_raw()
    raw["decomposition"]["sites"]["layers"] = {"kind": "range", "start": 18, "end": 27}
    authored = LMExperimentConfig.model_validate(raw)
    converted = build_experiment_config(authored, RUN_ID, DATA_ROOT)
    assert converted.target.sites == mlp_family_site_cs(18, 26, 24576)
    assert isinstance(converted.ci_fn, ChunkwiseTransformerCIFnArch)
    assert len(converted.ci_fn.chunks) == 9
    assert authored.runtime.remat_recon_forwards is True


def test_pinned_run_uses_current_schema(tmp_path: Path):
    """Pinned configs have the same strict contract as authored configs."""
    raw = llama_raw()
    raw["runtime"]["launch"] = "external"
    config = tmp_path / "launch_config.yaml"
    config.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValidationError):
        load_config(config, RUN_ID, DATA_ROOT)


def test_run_id_drives_identity_and_rejects_malformed():
    """The run dir and wandb id are the p-id (runs/<id>/ convention); the human name
    stays the wandb display name. The run id is the build helper's arg; a malformed id
    refuses at build time."""
    config = CONFIGS / "llama3_1_8b.yaml"
    cfg, _ = load_config(config, RUN_ID, DATA_ROOT)
    assert cfg.run.run_id == RUN_ID
    assert cfg.run.run_dir.name == RUN_ID
    assert cfg.run.run_name == "llama3-1-8b"

    with pytest.raises(AssertionError, match="run_id must be"):
        load_config(config, "run42", DATA_ROOT)


def test_arithmetic_ci_grid_metric_builds_to_arithmetic_eval_config():
    raw = llama_raw()
    arithmetic_raw = arithmetic_metric()
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
    raw = llama_raw()
    arithmetic = arithmetic_metric()
    raw["eval"]["metrics"].append(arithmetic)
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
        "ci_fn": "owner",
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

    # CI rows belong to the CI architecture: a table names a preset for them, and
    # hand-written CI rows refuse at parse
    assert full.ci_fn == "owner"
    with pytest.raises(ValidationError, match="ci_fn"):
        PlacementTableConfig.model_validate(
            {**table, "ci_fn": {"vectors": {}, "activations": {"batch": "replicate"}}}
        )

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
            "ci_fn": "ddp",
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
    raw = llama_raw()
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


@pytest.mark.parametrize("kind", ["PGDReconLoss", "PGDReconSubsetLoss"])
@pytest.mark.parametrize("shape", ["c", "sc"])
def test_shared_fresh_sources_are_refused_when_training_configs_are_constructed(
    kind: str, shape: str
):
    term = {
        "type": kind,
        "coeff": 1.0,
        "init": "random",
        "source_shape": shape,
        "n_steps": 1,
        "step_size": 0.1,
    }
    raw = _reference_lm_raw()
    raw["pd"]["loss_metrics"].append(term)
    with pytest.raises(ValidationError, match="source_shape"):
        LMExperimentConfig.model_validate(raw)
    cls = PGDReconLossConfig if kind == "PGDReconLoss" else PGDReconSubsetLossConfig
    with pytest.raises(ValidationError, match="source_shape"):
        cls.model_validate(term)
    with pytest.raises(ValidationError, match="source_shape"):
        TypeAdapter(TargetedLossMetricConfig).validate_python(term)


@pytest.mark.parametrize("shape", ["bc", "bsc"])
@pytest.mark.parametrize("init", ["random", "ones", "zeroes"])
@pytest.mark.parametrize("cls", [PGDReconLossConfig, PGDReconSubsetLossConfig])
def test_fresh_training_strategy_retains_its_batch_shape(
    shape: BatchSourceShape,
    init: PGDInitStrategy,
    cls: type[PGDReconLossConfig] | type[PGDReconSubsetLossConfig],
):
    cfg = cls(coeff=1.0, init=init, source_shape=shape, n_steps=1, step_size=0.1)
    (term,) = build_recon_terms(
        [cfg],
        (SiteSpec(name="site", factorization=DenseFactorization(d_in=4, d_out=4, C=4), group="g"),),
    )
    assert isinstance(term.sources, FreshPGDSources)
    assert term.sources.source_shape == shape
    assert term.sources.init == init


@pytest.mark.parametrize(
    "cls,steps",
    [(EvalPGDReconLossConfig, {"n_steps": 2}), (SlowPGDReconLossConfig, {"read_out_steps": (2,)})],
)
@pytest.mark.parametrize(
    "field,value",
    [
        ("source_shape", "bc"),
        ("source_shape", "bsc"),
        ("source_shape", "sc"),
        ("init", "ones"),
        ("init", "zeroes"),
    ],
)
def test_eval_pgd_rejects_unsupported_attacks_at_construction(
    cls: type[EvalPGDReconLossConfig] | type[SlowPGDReconLossConfig],
    steps: dict[str, int | tuple[int, ...]],
    field: str,
    value: str,
):
    raw = cls.model_validate(
        {"init": "random", "source_shape": "c", **steps, "step_size": 0.1}
    ).model_dump()
    raw[field] = value
    with pytest.raises(ValidationError, match=field):
        cls.model_validate(raw)
    with pytest.raises(ValidationError, match=field):
        TypeAdapter(AnyEvalMetricConfig).validate_python(raw)


def test_training_and_eval_parse_the_same_pgd_tag_into_distinct_types():
    training = PGDReconLossConfig(
        coeff=1.0, init="random", source_shape="bc", n_steps=2, step_size=0.1
    )
    evaluation = EvalPGDReconLossConfig(init="random", source_shape="c", n_steps=2, step_size=0.1)
    assert training.type == evaluation.type == "PGDReconLoss"
    train_adapter = TypeAdapter(AnyLossMetricConfig)
    eval_adapter = TypeAdapter(AnyEvalMetricConfig)
    assert train_adapter.validate_json(training.model_dump_json()) == training
    assert eval_adapter.validate_json(evaluation.model_dump_json()) == evaluation
    with pytest.raises(ValidationError):
        train_adapter.validate_python(evaluation)
    with pytest.raises(ValidationError):
        eval_adapter.validate_python(training)


@pytest.mark.parametrize("shape", ["c", "sc"])
@pytest.mark.parametrize(
    "cls", [PersistentPGDReconLossConfig, MergedStochasticSubsetPPGDReconLossConfig]
)
def test_persistent_training_configuration_requires_a_batch_axis(
    shape: str,
    cls: type[PersistentPGDReconLossConfig] | type[MergedStochasticSubsetPPGDReconLossConfig],
):
    fields: dict[str, object] = {
        "coeff": 1.0,
        "optimizer": AdamPGDConfig(lr_schedule=ScheduleConfig.constant(0.1)),
        "source_shape": shape,
    }
    if cls is MergedStochasticSubsetPPGDReconLossConfig:
        fields["adv_fraction"] = ScheduleConfig.constant(0.5)
    with pytest.raises(ValidationError, match="source_shape"):
        cls.model_validate(fields)
