"""torch `LMExperimentConfig` YAML → the JAX trainer's `ExperimentConfig`.

The shared `param-decomp-config` package (torch-free pydantic schema, same repo, branch
`refactor/shared-config-package`) validates the torch run YAML; this module maps the
subspace this trainer implements onto its knobs and ASSERTS loudly on anything else —
a torch config either converts exactly or refuses to run, never silently approximates.

Entry: a small wrapper YAML carrying what the torch schema cannot express —

    torch_config: <path, relative to the wrapper>   # the torch LMExperimentConfig yaml
    run_id: p-1a2b3c4d        # canonical id (generate: secrets.token_hex(4)); run dir
                              # name + wandb id — the torch runs/<id>/ convention
    run_name: my-run          # human-readable wandb display name
    out_dir: /mnt/data/.../param-decomp/runs
    remat_recon_forwards: false                     # jax-runtime memory/compute trade

`jsp-train` detects the `torch_config` key and routes here (`load_torch_wrapper`).

Knowingly ignored torch fields (runtime details with no JAX analog, or JAX-side
equivalents derived elsewhere): `runtime.device/dp` (GSPMD owns placement),
`target.activation_checkpointing` (the wrapper's `remat_recon_forwards` is the
explicit analog), `target.output_extract`, `data.buffer_size/shuffle_each_epoch/
train_split/eval_split` (the JAX data schedule is deterministic by construction),
`eval.slow_every/slow_on_first_step` (no slow in-loop metrics; plot/slow metrics run
offline via `jsp-export` + `pd-offline-eval`), `use_fused_kl` (torch impl detail).
"""

import re
from pathlib import Path
from typing import Any

import yaml

from jax_single_pool import llama_simple_mlp
from jax_single_pool.ci_fn import CIArch
from jax_single_pool.config import (
    CadenceConfig,
    CIOptimizerConfig,
    DataConfig,
    DenseLogPhase,
    EvalConfig,
    EvalPGDConfig,
    ExperimentConfig,
    FaithWarmupConfig,
    LlamaSimpleMLPTargetConfig,
    TargetConfig,
    VUOptimizerConfig,
)
from jax_single_pool.llama8b import SITE_NAME_PATTERN, canonical_site_cs
from jax_single_pool.lm import SiteC
from jax_single_pool.recon import build_recon_terms
from param_decomp_config.eval_metrics import CEandKLLossesConfig, CI_L0Config
from param_decomp_config.lm import (
    HFTarget,
    HFWeightsInVendored,
    LMExperimentConfig,
    PretrainedTarget,
)
from param_decomp_config.losses import PGDReconLossConfig
from param_decomp_config.pd import AnyLossMetricConfig, OptimizerConfig
from param_decomp_config.schedule import ScheduleConfig

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


def _resolve_target(torch_cfg: LMExperimentConfig) -> TargetConfig | LlamaSimpleMLPTargetConfig:
    """Target spec + decomposition patterns -> the JAX target config.

    Vendored and raw-HF Llama specs load the SAME meta-llama weights (the export
    bridge round-trip verified vendored == HF numerics); both map to the HF loader.
    `kind: pretrained` LlamaSimpleMLP specs map to the pretrain-cache loader, with
    `h.*` wildcard patterns expanded over the checkpoint's n_layer."""
    spec = torch_cfg.target.spec
    match spec:
        case HFWeightsInVendored() | HFTarget():
            match spec:
                case HFWeightsInVendored():
                    assert spec.model_class.rsplit(".", 1)[-1] == "VendoredLlama", spec.model_class
                case HFTarget():
                    assert spec.model_class == "transformers.LlamaForCausalLM", spec.model_class
            assert "Llama-3.1-8B" in spec.model_name, spec.model_name
            return TargetConfig(model_name=spec.model_name, sites=_site_cs(torch_cfg))
        case PretrainedTarget():
            assert spec.model_class.rsplit(".", 1)[-1] == "LlamaSimpleMLP", spec.model_class
            assert torch_cfg.pd.identity_decomposition_targets is None, (
                "identity targets unsupported"
            )
            cache_dir = llama_simple_mlp.pretrain_cache_dir(spec.run_path)
            arch = llama_simple_mlp.load_model_config(cache_dir)
            assert torch_cfg.data.max_seq_len <= arch.n_ctx, (
                torch_cfg.data.max_seq_len, arch.n_ctx,
            )  # fmt: skip
            sites = llama_simple_mlp.expand_wildcard_site_cs(
                tuple(SiteC(t.module_pattern, t.C) for t in torch_cfg.pd.decomposition_targets),
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
        sigmoid_type=cfg.pd.sigmoid_type,
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


def convert_torch_lm_config(
    torch_cfg: LMExperimentConfig,
    run_name: str,
    run_id: str,
    out_dir: Path,
    remat_recon_forwards: bool,
) -> ExperimentConfig:
    target = _resolve_target(torch_cfg)

    assert torch_cfg.pd.use_delta_component and torch_cfg.pd.tied_weights is None
    assert torch_cfg.runtime.autocast_bf16, "JAX trainer computes in bf16 (autocast analog)"
    assert torch_cfg.pd.faithfulness_warmup_weight_decay == 0.0

    if torch_cfg.target.weights_dtype == "float32":
        print(
            "DIVERGENCE: torch config asks for an fp32 frozen target; the JAX trainer keeps "
            "the frozen target in bf16 (measured ~5e-4 nats KL on clean logits — negligible "
            "vs recon KLs, but not bit-parity).",
            flush=True,
        )
    else:
        assert torch_cfg.target.weights_dtype == "bfloat16", torch_cfg.target.weights_dtype

    vu_opt = torch_cfg.pd.components_optimizer
    ci_opt = torch_cfg.pd.ci_fn_optimizer
    _assert_cosine_to_tenth(vu_opt.lr_schedule, "components_optimizer")
    _assert_cosine_to_tenth(ci_opt.lr_schedule, "ci_fn_optimizer")
    _assert_plain_adamw(vu_opt, "components_optimizer")
    _assert_plain_adamw(ci_opt, "ci_fn_optimizer")
    assert vu_opt.grad_clip_norm is not None, "components grad clip is part of the method"
    assert ci_opt.grad_clip_norm is None, "CI-fn grad clip unsupported"

    loss_metrics = _losses(torch_cfg, tuple(sc.name for sc in target.sites))
    data = _data(torch_cfg)

    cadence = torch_cfg.cadence
    assert cadence.save_every is not None and cadence.keep_last_n_checkpoints is not None, cadence

    return ExperimentConfig(
        run_name=run_name,
        run_id=run_id,
        out_dir=out_dir,
        seed=torch_cfg.pd.seed,
        steps=torch_cfg.pd.steps,
        target=target,
        data=data,
        loss_metrics=loss_metrics,
        n_mask_samples=torch_cfg.pd.n_mask_samples,
        sampling=torch_cfg.pd.sampling,
        remat_recon_forwards=remat_recon_forwards,
        vu_optimizer=VUOptimizerConfig(
            lr=vu_opt.lr_schedule.start_val, grad_clip_norm=vu_opt.grad_clip_norm
        ),
        ci_optimizer=CIOptimizerConfig(lr=ci_opt.lr_schedule.start_val),
        ci_fn=_ci_arch(torch_cfg, data.seq_len),
        faith_warmup=FaithWarmupConfig(
            steps=torch_cfg.pd.faithfulness_warmup_steps, lr=torch_cfg.pd.faithfulness_warmup_lr
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
        eval=_eval(torch_cfg),
        wandb=torch_cfg.wandb,
    )


WRAPPER_KEYS = {"torch_config", "run_id", "run_name", "out_dir", "remat_recon_forwards"}
_RUN_ID_PATTERN = re.compile(r"^p-[0-9a-f]{8}$")


def load_torch_wrapper(wrapper_path: Path) -> tuple[ExperimentConfig, Path, dict[str, Any]]:
    """Parse a wrapper YAML (see module docstring) -> (config, torch yaml path, raw torch
    dict for wandb). The torch path is resolved relative to the wrapper file.

    `run_id` is the canonical `p-<8hex>` identity (torch `generate_run_id` format):
    run dir name + wandb run id, stamped into the workspace's wrapper copy by
    `pd-jax-lm` at submit time, so resumes derive the same identity and the
    byte-compare pins it."""
    raw = yaml.safe_load(wrapper_path.read_text())
    assert set(raw) == WRAPPER_KEYS, f"{wrapper_path}: keys must be {sorted(WRAPPER_KEYS)}"
    run_id = raw["run_id"]
    assert _RUN_ID_PATTERN.match(run_id), f"run_id must be p-<8hex>, got {run_id!r}"
    torch_yaml_path = (wrapper_path.parent / raw["torch_config"]).resolve()
    assert torch_yaml_path.exists(), f"torch config not found: {torch_yaml_path}"
    torch_raw = yaml.safe_load(torch_yaml_path.read_text())
    torch_cfg = LMExperimentConfig(**torch_raw)
    cfg = convert_torch_lm_config(
        torch_cfg,
        run_name=raw["run_name"],
        run_id=run_id,
        out_dir=Path(raw["out_dir"]),
        remat_recon_forwards=raw["remat_recon_forwards"],
    )
    return cfg, torch_yaml_path, torch_raw


def load_run_dir_config(run_dir: Path) -> ExperimentConfig:
    """Rebuild a run's `ExperimentConfig` from its pinned config copies (for tools
    that read finished/live run dirs, e.g. the exporter).

    A run dir pins the wrapper as `config.yaml` and the referenced torch yaml beside
    it as `experiment_config.yaml` (the torch `SavedLMRun` contract name); the
    wrapper's own (launch-relative) path field is ignored — the pinned copy is the
    source of truth."""
    raw = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert set(raw) == WRAPPER_KEYS, f"{run_dir}/config.yaml: keys must be {sorted(WRAPPER_KEYS)}"
    torch_raw = yaml.safe_load((run_dir / "experiment_config.yaml").read_text())
    return convert_torch_lm_config(
        LMExperimentConfig(**torch_raw),
        run_name=raw["run_name"],
        run_id=raw["run_id"],
        out_dir=Path(raw["out_dir"]),
        remat_recon_forwards=raw["remat_recon_forwards"],
    )
