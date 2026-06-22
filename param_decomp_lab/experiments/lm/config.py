"""LM experiment config schema (target spec, data settings, full YAML tree) PLUS the
LM YAML→`BuiltRun` conversion.

This module reads the canonical `LMExperimentConfig` schema directly and builds the engine's
`BuiltRun` bundle (`param_decomp.config`) — the pydantic `pd` / `cadence` / `runtime`
verbatim plus the resolved target / data / CI-fn arch / eval — asserting loudly on anything
the JAX trainer doesn't implement. The composition entry (`run.py`) calls `load_config` /
`build_from_schema`; consumers that read a finished run dir call `load_run_dir_config`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import Discriminator, Field, PositiveInt

from param_decomp import llama_simple_mlp
from param_decomp.base_config import BaseConfig
from param_decomp.ci_fn import CIArch
from param_decomp.config import (
    AttnPatternsEvalConfig,
    BuiltRun,
    DataConfig,
    EvalConfig,
    EvalPGDConfig,
    WeightsDtype,
)
from param_decomp.configs import (
    CEandKLLossesConfig,
    CI_L0Config,
    CIHistogramsConfig,
    CIMaskedAttnPatternsReconLossConfig,
    GlobalSharedTransformerCiFnConfig,
    PGDReconLossConfig,
    StochasticAttnPatternsReconLossConfig,
)
from param_decomp.llama8b import SITE_NAME_PATTERN, canonical_site_cs
from param_decomp.lm import SiteC
from param_decomp.recon import build_recon_terms
from param_decomp_lab.experiments.config import (
    ExperimentConfig,
    assert_canonical_algorithm_config,
    run_instance,
)


class HFTarget(BaseConfig):
    """Load a HuggingFace model via `<model_class>.from_pretrained(<model_name>)`."""

    kind: Literal["hf"] = "hf"
    model_class: str
    model_name: str


class PretrainedTarget(BaseConfig):
    """Load an in-repo lab-pretrained model.

    `run_path` accepts any form `PretrainRunInfo.from_path` does — compact W&B
    (`entity/project/runId`), full W&B (`entity/project/runs/runId`), or a local
    checkpoint path (repo-relative paths are resolved at load time by `build_target`).
    """

    kind: Literal["pretrained"] = "pretrained"
    model_class: str
    run_path: str


class HFWeightsInVendored(BaseConfig):
    """Load HF pretrained weights into a vendored `param_decomp_lab.experiments.lm.pretrain.models.*`
    architecture via `<class>.from_hf_pretrained(<hub_id>)`.

    Useful when the decomposition target needs structural changes vs HF — e.g.
    `GPT2Simple`'s separate q/k/v projections vs HF's fused `c_attn`.
    """

    kind: Literal["hf_weights_in_vendored"] = "hf_weights_in_vendored"
    model_class: str  # must expose `from_hf_pretrained`
    model_name: str  # HF hub id


LMTargetSpec = Annotated[
    HFTarget | PretrainedTarget | HFWeightsInVendored,
    Discriminator("kind"),
]


class LMTargetConfig(BaseConfig):
    """Config for the LM target model and how to extract the prediction tensor.

    `output_extract` (passed to `make_run_batch`) pulls the prediction tensor out of the
    model's forward output (default `"logits"`).
    """

    spec: LMTargetSpec
    output_extract: int | str | None = "logits"
    activation_checkpointing: bool = False
    """If True and the target exposes `enable_activation_checkpointing()`, turn on
    per-block gradient checkpointing on the frozen target forward. Trades ~33% extra
    compute for ~10–15x less stored activation memory under 3-pool — the main lever for
    raising `b_per_rank` on deep targets."""
    weights_dtype: Literal["float32", "bfloat16"] = "float32"
    """dtype for the FROZEN target weights. `bfloat16` halves the target's resident footprint
    on every pool (the dominant resident term for an 8B target) — for natively-bf16 models the
    matmuls already run bf16 under autocast, so this only changes residual/norm accumulation
    precision (measured ~5e-4 nats KL on Llama-3.1-8B clean logits, negligible vs recon KLs).
    Only the frozen target is cast; trained V/U components stay fp32 (their AdamW master)."""


class LMDataConfig(BaseConfig):
    """LM experiment dataset / dataloader settings."""

    dataset_name: str = Field(..., description="HuggingFace dataset id")
    data_files: str | None = Field(
        default=None,
        description=(
            "Explicit file glob passed to load_dataset (e.g. 'sample/350BT/*.parquet'). "
            "Resolves directly against that path instead of enumerating the whole repo "
            "tree, which slashes Hub API calls vs. selecting a config by name."
        ),
    )
    revision: str | None = Field(
        default=None,
        description="Dataset git revision (commit SHA/tag) to pin layout and data for reproducibility",
    )
    tokenizer_name: str = Field(..., description="HF tokenizer id or path")
    column_name: str = Field(default="text", description="Dataset column with the text/tokens")
    max_seq_len: PositiveInt = Field(default=512, description="Max sequence length")
    train_split: str = Field(default="train")
    eval_split: str = Field(default="test")
    is_tokenized: bool = Field(default=False)
    streaming: bool = Field(default=False)
    buffer_size: PositiveInt = Field(default=1000)
    shuffle_each_epoch: bool = Field(default=True)


class LMExperimentConfig(ExperimentConfig[LMTargetConfig, LMDataConfig]):
    pass


@dataclass(frozen=True)
class TargetConfig:
    """The Llama-3.1-8B HF target (`param_decomp.llama8b`)."""

    model_name: str
    sites: tuple[SiteC, ...]
    """Decomposed sites with per-site C, in canonical order (`canonical_site_cs`)."""

    supported_weights_dtypes: frozenset[WeightsDtype] = frozenset({"bfloat16"})
    """Frozen-target weight dtypes the loader supports (`llama8b.py` is bf16-only:
    `DT = jnp.bfloat16`). A config requesting a dtype outside this set is refused at
    convert time — no silent downgrade (issue #727)."""


@dataclass(frozen=True)
class LlamaSimpleMLPTargetConfig:
    """The `LlamaSimpleMLP` pile-pretrained target (`param_decomp.llama_simple_mlp`);
    weights from the torch pretrain cache resolved from `pretrain_run_path`."""

    pretrain_run_path: str
    sites: tuple[SiteC, ...]
    """Decomposed sites with per-site C, in canonical order
    (`llama_simple_mlp.canonical_site_cs`)."""

    supported_weights_dtypes: frozenset[WeightsDtype] = frozenset({"bfloat16"})
    """Frozen-target weight dtypes the loader supports (`llama_simple_mlp.py` loads bf16:
    `jnp.bfloat16` hardcoded at the call site). See `TargetConfig.supported_weights_dtypes`."""


AnyLMTargetConfig = TargetConfig | LlamaSimpleMLPTargetConfig
"""The LM target configs the LM composition builds. Non-LM targets (the toys) live in the
lab and satisfy `param_decomp.config.TargetSites` — the core `BuiltRun.target` is typed by
that protocol, never by a closed union, so it accepts a lab target config too."""


# Plot/heavy eval metrics the FAST in-loop scalar pass (`eval.py`) does NOT compute. They
# run in the IN-LOOP SLOW TIER (SPEC S28/S29, in-loop only — no offline CLI), on cadence
# `eval.slow_every`. The base plot metrics drive the shared `render_slow_eval_figures`
# figure set; the two hidden-acts metrics drive `compute_hidden_acts_metrics`. The
# permutation/UV/identity metrics (`UVPlots` / `PermutedCIPlots` / `IdentityCIError`) are
# ALSO in-loop but are read by the composition straight off the raw config
# (`slow_eval.eval_metrics_from_run_dir`), since the trainer's typed `EvalConfig` keeps only
# scalar-tier fields — so `_eval` just accepts them here without populating `EvalConfig`.
SLOW_TIER_EVAL_METRIC_TYPES = frozenset(
    {
        "CIHistograms",
        "ComponentActivationDensity",
        "CIMeanPerComponent",
        "StochasticHiddenActsReconLoss",
        "CIHiddenActsReconLoss",
        "UVPlots",
        "PermutedCIPlots",
        "IdentityCIError",
    }
)


def _site_cs(cfg: LMExperimentConfig) -> tuple[SiteC, ...]:
    """Decomposition targets -> canonical per-site (name, C) pairs. Any per-layer
    matrix site (q/k/v/o/gate/up/down) with its own C is supported; non-site module
    patterns refuse. Raw-HF specs name modules `model.layers.*`; the vendored class drops
    the prefix — same matrices either way."""
    site_cs = []
    for target in cfg.pd.decomposition_targets:
        name = target.module_pattern.removeprefix("model.")
        assert SITE_NAME_PATTERN.match(name), (
            f"unsupported decomposition target {target.module_pattern!r}"
        )
        site_cs.append(SiteC(name, target.C))
    return canonical_site_cs(tuple(site_cs))


def _resolve_target(cfg: LMExperimentConfig) -> AnyLMTargetConfig:
    """Target spec + decomposition patterns -> the JAX target config.

    Vendored and raw-HF Llama specs load the SAME meta-llama weights (the export
    bridge round-trip verified vendored == HF numerics); both map to the HF loader.
    `kind: pretrained` LlamaSimpleMLP specs map to the pretrain-cache loader, with
    `h.*` wildcard patterns expanded over the checkpoint's n_layer."""
    spec = cfg.target.spec
    match spec:
        case HFWeightsInVendored() | HFTarget():
            match spec:
                case HFWeightsInVendored():
                    assert spec.model_class.rsplit(".", 1)[-1] == "VendoredLlama", spec.model_class
                case HFTarget():
                    assert spec.model_class == "transformers.LlamaForCausalLM", spec.model_class
            assert "Llama-3.1-8B" in spec.model_name, spec.model_name
            return TargetConfig(model_name=spec.model_name, sites=_site_cs(cfg))
        case PretrainedTarget():
            assert spec.model_class.rsplit(".", 1)[-1] == "LlamaSimpleMLP", spec.model_class
            cache_dir = llama_simple_mlp.pretrain_cache_dir(spec.run_path)
            arch = llama_simple_mlp.load_model_config(cache_dir)
            assert cfg.data.max_seq_len <= arch.n_ctx, (cfg.data.max_seq_len, arch.n_ctx)
            sites = llama_simple_mlp.expand_wildcard_site_cs(
                tuple(SiteC(t.module_pattern, t.C) for t in cfg.pd.decomposition_targets),
                arch.n_layer,
            )
            return LlamaSimpleMLPTargetConfig(pretrain_run_path=spec.run_path, sites=sites)


def _ci_arch(cfg: LMExperimentConfig, seq_len: int) -> CIArch:
    ci = cfg.pd.ci_config
    assert isinstance(ci, GlobalSharedTransformerCiFnConfig), ci
    transformer = ci.simple_transformer_ci_cfg
    assert transformer.mlp_hidden_dim is not None and len(transformer.mlp_hidden_dim) == 1, (
        f"CI MLP must be single-hidden-layer, got {transformer.mlp_hidden_dim}"
    )
    assert transformer.attn_config.rope_base == 10000.0, transformer.attn_config
    assert transformer.attn_config.max_len >= seq_len, (transformer.attn_config.max_len, seq_len)
    return CIArch(
        d_model=transformer.d_model,
        n_blocks=transformer.n_blocks,
        n_heads=transformer.attn_config.n_heads,
        mlp_hidden=transformer.mlp_hidden_dim[0],
    )


def _assert_losses_supported(cfg: LMExperimentConfig, site_names: tuple[str, ...]) -> None:
    """Run the schema's loss configs through `build_recon_terms` so unsupported metrics
    refuse at convert time rather than on the GPUs. The engine reads `pd.loss_metrics`
    verbatim (yaml order is RNG-load-bearing), so nothing is returned."""
    build_recon_terms(cfg.pd.loss_metrics, site_names, cfg.pd.n_mask_samples, cfg.pd.sampling)


def _data(cfg: LMExperimentConfig) -> DataConfig:
    data = cfg.data
    assert data.is_tokenized and not data.streaming, (
        "JAX trainer reads pre-tokenized parquet shards; tokenize offline first"
    )
    assert data.dataset_name == "parquet" and data.column_name == "input_ids", data
    assert data.data_files is not None
    shard_glob = Path(data.data_files)
    assert shard_glob.name == "*.parquet", f"expected a *.parquet glob, got {data.data_files}"
    return DataConfig(
        dir=shard_glob.parent, seq_len=data.max_seq_len, global_batch=cfg.pd.batch_size
    )


def _assert_separate_qk_attn_paths(
    metric: CIMaskedAttnPatternsReconLossConfig | StochasticAttnPatternsReconLossConfig,
) -> None:
    """The JAX targets decompose attention as separate `*q_proj`/`*k_proj` sites; no JAX
    target produces a combined-QKV (`c_attn`) site to split. Refuse the combined config
    loudly (the attn-patterns step reads Q/K from the q/k_proj sites by name)."""
    assert metric.c_attn_path is None, (
        f"{metric.type}: combined c_attn is unsupported (no JAX target produces a merged-QKV "
        f"site); decompose separate q_proj/k_proj sites instead"
    )


def _eval(cfg: LMExperimentConfig) -> EvalConfig | None:
    if cfg.eval is None:
        return None
    ce_kl = ci_l0 = pgd = None
    attn_ci = attn_stoch = False
    slow_n_batches_accum: int | None = None
    for metric in cfg.eval.metrics:
        match metric:
            case CEandKLLossesConfig():
                ce_kl = metric
            case CI_L0Config():
                ci_l0 = metric
            case PGDReconLossConfig():
                assert metric.init == "random" and metric.mask_scope == "c", metric
                pgd = EvalPGDConfig(n_steps=metric.n_steps, step_size=metric.step_size)
            case CIMaskedAttnPatternsReconLossConfig():
                _assert_separate_qk_attn_paths(metric)
                attn_ci = True
            case StochasticAttnPatternsReconLossConfig():
                _assert_separate_qk_attn_paths(metric)
                attn_stoch = True
            case CIHistogramsConfig():
                slow_n_batches_accum = metric.n_batches_accum
            case _ if metric.type in SLOW_TIER_EVAL_METRIC_TYPES:
                pass  # rendered by the in-loop slow tier (run.py reads them off the raw cfg)
            case _:
                raise AssertionError(f"unsupported eval metric {metric.type!r}")
    assert ce_kl is not None and ci_l0 is not None, (
        "in-loop eval needs CEandKLLosses + CI_L0 in eval.metrics"
    )
    return EvalConfig(
        batch_size=cfg.eval.batch_size,
        every=cfg.eval.every,
        n_steps=cfg.eval.n_steps,
        slow_every=cfg.eval.slow_every,
        slow_on_first_step=cfg.eval.slow_on_first_step,
        slow_n_batches_accum=slow_n_batches_accum,
        rounding_threshold=ce_kl.rounding_threshold,
        ci_alive_threshold=ci_l0.ci_alive_threshold,
        l0_thresholds=ci_l0.l0_thresholds,
        l0_groups=(
            {group: tuple(patterns) for group, patterns in ci_l0.groups.items()}
            if ci_l0.groups is not None
            else None
        ),
        pgd=pgd,
        attn_patterns=(
            AttnPatternsEvalConfig(ci_masked=attn_ci, stochastic=attn_stoch)
            if attn_ci or attn_stoch
            else None
        ),
    )


def assert_supported_weights_dtype(cfg: LMExperimentConfig) -> None:
    """Refuse a frozen-target weights_dtype the loader can't honour (issue #727: no
    silent downgrade). Enforced at the train/submit boundary only — the bf16-only
    loaders ignore `weights_dtype` when *consuming* a finished run, so opening an
    already-trained bf16 run whose stored config predates the explicit-bf16
    convention must not be blocked here."""
    target = _resolve_target(cfg)
    assert cfg.target.weights_dtype in target.supported_weights_dtypes, (
        f"target {type(target).__name__} supports frozen-target weights_dtype "
        f"{sorted(target.supported_weights_dtypes)}, config asks for "
        f"{cfg.target.weights_dtype!r}. No silent downgrade (issue #727): declare a "
        f"supported dtype in the yaml."
    )


def build_experiment_config(cfg: LMExperimentConfig) -> BuiltRun:
    target = _resolve_target(cfg)
    assert_canonical_algorithm_config(cfg)
    _assert_losses_supported(cfg, tuple(sc.name for sc in target.sites))
    data = _data(cfg)

    return BuiltRun(
        pd=cfg.pd,
        runtime=cfg.runtime,
        cadence=cfg.cadence,
        run=run_instance(cfg),
        target=target,
        data=data,
        ci_fn=_ci_arch(cfg, data.seq_len),
        eval=_eval(cfg),
    )


def build_from_schema(schema_raw: dict[str, Any]) -> BuiltRun:
    """Validate a single self-contained LM run config (the canonical `LMExperimentConfig`
    schema + run-instance fields) and convert it to the engine's `BuiltRun` bundle.

    The LM composition entry (`run.py`) is LM-only. The toy domains (TMS, ResidMLP) build
    their `BuiltRun` in their own `run.py` via the public shared helpers
    (`assert_canonical_algorithm_config`, `run_instance`, `layerwise_mlp_ci_arch`)."""
    cfg = LMExperimentConfig(**schema_raw)
    assert_supported_weights_dtype(cfg)
    return build_experiment_config(cfg)


def load_config(config_path: Path) -> tuple[BuiltRun, dict[str, Any]]:
    """Parse a single self-contained LM run YAML (the canonical schema + top-level
    `run_name`/`run_id`/`out_dir`, `runtime.remat_recon_forwards`, `wandb.group`/`tags`)
    -> (built run, raw dict for wandb logging)."""
    schema_raw = yaml.safe_load(config_path.read_text())
    return build_from_schema(schema_raw), schema_raw


def load_run_dir_config(run_dir: Path) -> BuiltRun:
    """Rebuild a run's `BuiltRun` bundle from its single pinned `config.yaml`
    (for tools that read finished/live run dirs, e.g. harvest / fine-tune compat)."""
    schema_raw = yaml.safe_load((run_dir / "config.yaml").read_text())
    return build_from_schema(schema_raw)
