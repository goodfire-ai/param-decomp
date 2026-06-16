"""The trainer's internal experiment config, built DIRECTLY from the canonical
`param_decomp_config` schema.

The yaml surface is the shared torch-free schema (`LMExperimentConfig`, reached via a
small wrapper yaml carrying the run identity + jax-runtime knobs the schema cannot
express); this module reads that schema directly and maps the subspace this trainer
implements onto the dataclasses below, ASSERTING loudly on anything else — a config
either converts exactly or refuses to run, never silently approximates. The loss list
is the SHARED pydantic configs passed through verbatim (`build_recon_terms` maps them
onto recon terms); the dataclasses here carry only the jax-runtime knobs that have
no canonical-schema home (remat, checkpoint cadence, the CI-fn architecture extraction).

Wrapper entry (see `load_wrapper`): a small yaml carrying what the canonical schema
cannot express —

    torch_config: <path, relative to the wrapper>   # the LMExperimentConfig yaml
    run_id: p-1a2b3c4d        # canonical id (generate: secrets.token_hex(4)); run dir
                              # name + wandb id — the runs/<id>/ convention
    run_name: my-run          # human-readable wandb display name
    out_dir: /mnt/data/.../param-decomp/runs
    remat_recon_forwards: false                     # jax-runtime memory/compute trade
    wandb_group: my-sweep     # optional; wandb UI group (pd-jax-lm --group)
    wandb_tags: [a, b]        # optional; wandb tags (pd-jax-lm --tags a,b)

`jsp-train` detects the `torch_config` key and routes here (`load_wrapper`).

Knowingly ignored canonical-schema fields (runtime details with no JAX analog, or
JAX-side equivalents derived elsewhere): `runtime.device/dp` (GSPMD owns placement),
`target.activation_checkpointing` (the wrapper's `remat_recon_forwards` is the explicit
analog), `target.output_extract`, `data.buffer_size/shuffle_each_epoch/train_split/
eval_split` (the JAX data schedule is deterministic by construction), `eval.slow_every/
slow_on_first_step` (no slow in-loop metrics; plot/slow metrics run offline via
`jsp-export` + `pd-offline-eval`), `use_fused_kl` (a torch impl detail).
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from jax_single_pool import llama_simple_mlp
from jax_single_pool.ci_fn import CIArch
from jax_single_pool.llama8b import SITE_NAME_PATTERN, canonical_site_cs
from jax_single_pool.lm import SiteC
from jax_single_pool.recon import build_recon_terms
from param_decomp_config.eval_metrics import CEandKLLossesConfig, CI_L0Config
from param_decomp_config.experiment import WandbConfig
from param_decomp_config.jax_wrapper import WRAPPER_KEYS, WRAPPER_OPTIONAL_KEYS
from param_decomp_config.lm import (
    HFTarget,
    HFWeightsInVendored,
    LMExperimentConfig,
    PretrainedTarget,
)
from param_decomp_config.losses import PGDReconLossConfig
from param_decomp_config.pd import AnyLossMetricConfig, OptimizerConfig
from param_decomp_config.schedule import ScheduleConfig

WeightsDtype = Literal["float32", "bfloat16"]


@dataclass(frozen=True)
class TargetConfig:
    """The Llama-3.1-8B HF target (`llama8b.py`)."""

    model_name: str
    sites: tuple[SiteC, ...]
    """Decomposed sites with per-site C, in canonical order (`canonical_site_cs`)."""

    supported_weights_dtypes: frozenset[WeightsDtype] = frozenset({"bfloat16"})
    """Frozen-target weight dtypes the loader supports (`llama8b.py` is bf16-only:
    `DT = jnp.bfloat16`). A config requesting a dtype outside this set is refused at
    convert time — no silent downgrade (issue #727)."""


@dataclass(frozen=True)
class LlamaSimpleMLPTargetConfig:
    """The `LlamaSimpleMLP` pile-pretrained target (`llama_simple_mlp.py`); weights
    from the torch pretrain cache resolved from `pretrain_run_path`."""

    pretrain_run_path: str
    sites: tuple[SiteC, ...]
    """Decomposed sites with per-site C, in canonical order
    (`llama_simple_mlp.canonical_site_cs`)."""

    supported_weights_dtypes: frozenset[WeightsDtype] = frozenset({"bfloat16"})
    """Frozen-target weight dtypes the loader supports (`llama_simple_mlp.py` loads bf16:
    `jnp.bfloat16` hardcoded at the call site). See `TargetConfig.supported_weights_dtypes`."""


AnyTargetConfig = TargetConfig | LlamaSimpleMLPTargetConfig


@dataclass(frozen=True)
class DataConfig:
    dir: Path
    seq_len: int
    global_batch: int


@dataclass(frozen=True)
class VUOptimizerConfig:
    lr: float
    grad_clip_norm: float


@dataclass(frozen=True)
class CIOptimizerConfig:
    lr: float


@dataclass(frozen=True)
class FaithWarmupConfig:
    steps: int
    lr: float


@dataclass(frozen=True)
class DenseLogPhase:
    every: int
    until_step: int


@dataclass(frozen=True)
class CadenceConfig:
    log_every: int
    save_every: int
    keep_last: int
    dense_log_phase: DenseLogPhase | None


@dataclass(frozen=True)
class EvalPGDConfig:
    """Fresh sign-PGD recon probe (torch eval `PGDReconLoss`: init random, source
    shared across batch and positions)."""

    n_steps: int
    step_size: float


@dataclass(frozen=True)
class EvalConfig:
    """In-loop eval pass (torch `EvalLoop` analog, scalar metrics only — plots ride the
    offline export path). `rounding_threshold` binarises CI for the CE/KL
    `rounded_masked` variant; `ci_alive_threshold` is the CI-L0 aliveness cutoff."""

    batch_size: int
    every: int
    n_steps: int
    rounding_threshold: float
    ci_alive_threshold: float
    l0_groups: dict[str, tuple[str, ...]] | None
    """torch CI_L0 `groups`: fnmatch site patterns whose member L0s sum into a
    group-named key. None = per-site keys only."""
    pgd: EvalPGDConfig | None


@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str
    """Human-readable display name (the wandb run NAME)."""
    run_id: str
    """Canonical `p-<8hex>` id (wandb run ID + run-dir name) — the torch
    `generate_run_id` convention, making the run a first-class citizen of the
    `runs/<id>/` postprocess world. Minted at submit time by `pd-jax-lm`."""
    out_dir: Path
    seed: int
    steps: int
    target: AnyTargetConfig
    data: DataConfig
    loss_metrics: tuple[AnyLossMetricConfig, ...]
    """The shared `pd.loss_metrics` configs, verbatim and in yaml order (order is
    RNG-load-bearing — see `build_recon_terms`)."""
    n_mask_samples: int
    sampling: Literal["continuous", "binomial"]
    remat_recon_forwards: bool
    vu_optimizer: VUOptimizerConfig
    ci_optimizer: CIOptimizerConfig
    ci_fn: CIArch
    faith_warmup: FaithWarmupConfig
    cadence: CadenceConfig
    eval: EvalConfig | None
    wandb: WandbConfig | None
    wandb_group: str | None
    """wandb UI group (`pd-jax-lm --group`); None = ungrouped. torch threads the
    same CLI flag to `wandb.init(group=...)`."""
    wandb_tags: tuple[str, ...]
    """wandb tags (`pd-jax-lm --tags a,b,c`, comma-split); empty = untagged."""

    @property
    def run_dir(self) -> Path:
        return self.out_dir / self.run_id


OFFLINE_EVAL_METRIC_TYPES = frozenset(
    {
        "CIHistograms",
        "ComponentActivationDensity",
        "CIMeanPerComponent",
        "StochasticHiddenActsReconLoss",
        "CIHiddenActsReconLoss",
        "UVPlots",
        "PermutedCIPlots",
        "IdentityCIError",
        "CIMaskedAttnPatternsReconLoss",
        "StochasticAttnPatternsReconLoss",
        "AutointerpLabels",
    }
)


def _site_cs(cfg: LMExperimentConfig) -> tuple[SiteC, ...]:
    """Decomposition targets -> canonical per-site (name, C) pairs. Any per-layer
    matrix site (q/k/v/o/gate/up/down) with its own C is supported; identity targets
    and non-site module patterns refuse. Raw-HF specs name modules `model.layers.*`;
    the vendored class drops the prefix — same matrices either way."""
    assert cfg.pd.identity_decomposition_targets is None, "identity targets unsupported"
    site_cs = []
    for target in cfg.pd.decomposition_targets:
        name = target.module_pattern.removeprefix("model.")
        assert SITE_NAME_PATTERN.match(name), (
            f"unsupported decomposition target {target.module_pattern!r}"
        )
        site_cs.append(SiteC(name, target.C))
    return canonical_site_cs(tuple(site_cs))


def _resolve_target(cfg: LMExperimentConfig) -> "TargetConfig | LlamaSimpleMLPTargetConfig":
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
            assert cfg.pd.identity_decomposition_targets is None, "identity targets unsupported"
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
    assert ci.mode == "global" and ci.fn_type == "global_shared_transformer", ci
    transformer = ci.simple_transformer_ci_cfg
    assert transformer is not None
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


def _assert_cosine_to_tenth(schedule: ScheduleConfig, who: str) -> None:
    """The trainer hardcodes optax cosine decay to 0.1x with no warmup (SPEC S19/S20)."""
    assert schedule.fn_type == "cosine", f"{who}: only cosine lr supported, got {schedule}"
    assert schedule.warmup_pct == 0.0, f"{who}: lr warmup unsupported, got {schedule}"
    assert schedule.final_val_frac == 0.1, f"{who}: final_val_frac must be 0.1, got {schedule}"


def _assert_plain_adamw(optimizer: OptimizerConfig, who: str) -> None:
    assert optimizer.betas == (0.9, 0.999), f"{who}: betas must be (0.9, 0.999)"
    assert optimizer.weight_decay == 0.0, f"{who}: weight_decay must be 0"


def _losses(
    cfg: LMExperimentConfig, site_names: tuple[str, ...]
) -> tuple[AnyLossMetricConfig, ...]:
    """Pass the shared loss configs through VERBATIM (yaml order — RNG-load-bearing),
    after running them through `build_recon_terms` so unsupported metrics refuse at
    convert time rather than on the GPUs."""
    loss_metrics = tuple(cfg.pd.loss_metrics)
    build_recon_terms(loss_metrics, site_names, cfg.pd.n_mask_samples, cfg.pd.sampling)
    return loss_metrics


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


def _eval(cfg: LMExperimentConfig) -> EvalConfig | None:
    if cfg.eval is None:
        return None
    ce_kl = ci_l0 = pgd = None
    skipped_offline: list[str] = []
    for metric in cfg.eval.metrics:
        match metric:
            case CEandKLLossesConfig():
                ce_kl = metric
            case CI_L0Config():
                ci_l0 = metric
            case PGDReconLossConfig():
                assert metric.init == "random" and metric.mask_scope == "c", metric
                pgd = EvalPGDConfig(n_steps=metric.n_steps, step_size=metric.step_size)
            case _ if metric.type in OFFLINE_EVAL_METRIC_TYPES:
                skipped_offline.append(metric.type)
            case _:
                raise AssertionError(f"unsupported eval metric {metric.type!r}")
    if skipped_offline:
        print(f"eval metrics deferred to the offline path: {sorted(skipped_offline)}", flush=True)
    assert ce_kl is not None and ci_l0 is not None, (
        "in-loop eval needs CEandKLLosses + CI_L0 in eval.metrics"
    )
    return EvalConfig(
        batch_size=cfg.eval.batch_size,
        every=cfg.eval.every,
        n_steps=cfg.eval.n_steps,
        rounding_threshold=ce_kl.rounding_threshold,
        ci_alive_threshold=ci_l0.ci_alive_threshold,
        l0_groups=(
            {group: tuple(patterns) for group, patterns in ci_l0.groups.items()}
            if ci_l0.groups is not None
            else None
        ),
        pgd=pgd,
    )


def build_experiment_config(
    cfg: LMExperimentConfig,
    run_name: str,
    run_id: str,
    out_dir: Path,
    remat_recon_forwards: bool,
    wandb_group: str | None = None,
    wandb_tags: tuple[str, ...] = (),
) -> ExperimentConfig:
    target = _resolve_target(cfg)

    assert cfg.pd.sigmoid_type == "leaky_hard", cfg.pd.sigmoid_type
    assert cfg.pd.use_delta_component and cfg.pd.tied_weights is None
    assert cfg.runtime.autocast_bf16, "JAX trainer computes in bf16 (autocast analog)"
    assert cfg.pd.faithfulness_warmup_weight_decay == 0.0

    assert cfg.target.weights_dtype in target.supported_weights_dtypes, (
        f"target {type(target).__name__} supports frozen-target weights_dtype "
        f"{sorted(target.supported_weights_dtypes)}, config asks for "
        f"{cfg.target.weights_dtype!r}. No silent downgrade (issue #727): declare a "
        f"supported dtype in the yaml."
    )

    vu_opt = cfg.pd.components_optimizer
    ci_opt = cfg.pd.ci_fn_optimizer
    _assert_cosine_to_tenth(vu_opt.lr_schedule, "components_optimizer")
    _assert_cosine_to_tenth(ci_opt.lr_schedule, "ci_fn_optimizer")
    _assert_plain_adamw(vu_opt, "components_optimizer")
    _assert_plain_adamw(ci_opt, "ci_fn_optimizer")
    assert vu_opt.grad_clip_norm is not None, "components grad clip is part of the method"
    assert ci_opt.grad_clip_norm is None, "CI-fn grad clip unsupported"

    loss_metrics = _losses(cfg, tuple(sc.name for sc in target.sites))
    data = _data(cfg)

    cadence = cfg.cadence
    assert cadence.save_every is not None and cadence.keep_last_n_checkpoints is not None, cadence

    return ExperimentConfig(
        run_name=run_name,
        run_id=run_id,
        out_dir=out_dir,
        seed=cfg.pd.seed,
        steps=cfg.pd.steps,
        target=target,
        data=data,
        loss_metrics=loss_metrics,
        n_mask_samples=cfg.pd.n_mask_samples,
        sampling=cfg.pd.sampling,
        remat_recon_forwards=remat_recon_forwards,
        vu_optimizer=VUOptimizerConfig(
            lr=vu_opt.lr_schedule.start_val, grad_clip_norm=vu_opt.grad_clip_norm
        ),
        ci_optimizer=CIOptimizerConfig(lr=ci_opt.lr_schedule.start_val),
        ci_fn=_ci_arch(cfg, data.seq_len),
        faith_warmup=FaithWarmupConfig(
            steps=cfg.pd.faithfulness_warmup_steps, lr=cfg.pd.faithfulness_warmup_lr
        ),
        cadence=CadenceConfig(
            log_every=cadence.train_log_every,
            save_every=cadence.save_every,
            keep_last=cadence.keep_last_n_checkpoints,
            dense_log_phase=(
                DenseLogPhase(
                    every=cadence.dense_log_phase.every,
                    until_step=cadence.dense_log_phase.until_step,
                )
                if cadence.dense_log_phase is not None
                else None
            ),
        ),
        eval=_eval(cfg),
        wandb=cfg.wandb,
        wandb_group=wandb_group,
        wandb_tags=wandb_tags,
    )


_RUN_ID_PATTERN = re.compile(r"^p-[0-9a-f]{8}$")


def _wandb_group_tags(raw: dict[str, Any]) -> tuple[str | None, tuple[str, ...]]:
    """The wandb UI knobs (`pd-jax-lm --group`/`--tags`) are stamped into the wrapper
    at submit time, like `run_id`; both default to absent for hand-written wrappers."""
    group = raw.get("wandb_group")
    assert group is None or isinstance(group, str), group
    tags = raw.get("wandb_tags", [])
    assert isinstance(tags, list) and all(isinstance(t, str) for t in tags), tags
    return group, tuple(tags)


def load_wrapper(wrapper_path: Path) -> tuple[ExperimentConfig, Path, dict[str, Any]]:
    """Parse a wrapper YAML (see module docstring) -> (config, schema yaml path, raw
    schema dict for wandb). The schema path is resolved relative to the wrapper file.

    `run_id` is the canonical `p-<8hex>` identity (torch `generate_run_id` format):
    run dir name + wandb run id, stamped into the workspace's wrapper copy by
    `pd-jax-lm` at submit time, so resumes derive the same identity and the
    byte-compare pins it."""
    raw = yaml.safe_load(wrapper_path.read_text())
    assert WRAPPER_KEYS <= set(raw) <= WRAPPER_KEYS | WRAPPER_OPTIONAL_KEYS, (
        f"{wrapper_path}: keys must be {sorted(WRAPPER_KEYS)} "
        f"(optional: {sorted(WRAPPER_OPTIONAL_KEYS)}), got {sorted(raw)}"
    )
    run_id = raw["run_id"]
    assert _RUN_ID_PATTERN.match(run_id), f"run_id must be p-<8hex>, got {run_id!r}"
    schema_yaml_path = (wrapper_path.parent / raw["torch_config"]).resolve()
    assert schema_yaml_path.exists(), f"config not found: {schema_yaml_path}"
    schema_raw = yaml.safe_load(schema_yaml_path.read_text())
    cfg = LMExperimentConfig(**schema_raw)
    wandb_group, wandb_tags = _wandb_group_tags(raw)
    experiment_config = build_experiment_config(
        cfg,
        run_name=raw["run_name"],
        run_id=run_id,
        out_dir=Path(raw["out_dir"]),
        remat_recon_forwards=raw["remat_recon_forwards"],
        wandb_group=wandb_group,
        wandb_tags=wandb_tags,
    )
    return experiment_config, schema_yaml_path, schema_raw


def load_run_dir_config(run_dir: Path) -> ExperimentConfig:
    """Rebuild a run's `ExperimentConfig` from its pinned config copies (for tools
    that read finished/live run dirs, e.g. the exporter).

    A run dir pins the wrapper as `config.yaml` and the referenced schema yaml beside
    it as `experiment_config.yaml` (the torch `SavedLMRun` contract name); the
    wrapper's own (launch-relative) path field is ignored — the pinned copy is the
    source of truth."""
    raw = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert WRAPPER_KEYS <= set(raw) <= WRAPPER_KEYS | WRAPPER_OPTIONAL_KEYS, (
        f"{run_dir}/config.yaml: keys must be {sorted(WRAPPER_KEYS)} "
        f"(optional: {sorted(WRAPPER_OPTIONAL_KEYS)}), got {sorted(raw)}"
    )
    schema_raw = yaml.safe_load((run_dir / "experiment_config.yaml").read_text())
    wandb_group, wandb_tags = _wandb_group_tags(raw)
    return build_experiment_config(
        LMExperimentConfig(**schema_raw),
        run_name=raw["run_name"],
        run_id=raw["run_id"],
        out_dir=Path(raw["out_dir"]),
        remat_recon_forwards=raw["remat_recon_forwards"],
        wandb_group=wandb_group,
        wandb_tags=wandb_tags,
    )
