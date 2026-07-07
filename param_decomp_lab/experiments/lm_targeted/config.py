"""Targeted-PD (tPD) LM experiment config schema + the tPD YAML->`BuiltRun` conversion.

The LM analog of the toy targeted runs (`experiments/{tms,resid_mlp}`): the TARGET stream is
a fixed prompt pool (`data.prompts_file`), the NON-TARGET stream is the normal parquet path
(`nontarget.data.dataset_name == "parquet"`), and the delta component is forced on over the
non-target pass (SPEC §11). Reuses the plain-LM target / eval / ci-fn resolution
(`experiments.lm.config`) verbatim; the only tPD-specific build is the `data` seam
(`_targeted_data`), which points the engine's `data` at the non-target parquet stream.
"""

from pathlib import Path

import yaml
from pydantic import model_validator

from param_decomp.built_run import BuiltRun, DataConfig
from param_decomp_lab.experiments.config import (
    NontargetConfig,
    assert_canonical_algorithm_config,
    assert_targeted_faithfulness_off,
    ci_arch,
    run_instance,
)
from param_decomp_lab.experiments.lm.config import (
    LMDataConfig,
    LMExperimentConfig,
    _assert_losses_supported,
    _eval,
    _resolve_chunkwise_ci_arch,
    _resolve_target,
    assert_supported_weights_dtype,
)


class LMTargetedExperimentConfig(LMExperimentConfig):
    """Targeted-PD LM run: `data` is the fixed prompt-pool TARGET stream, `nontarget.data` is
    the broad parquet NON-TARGET stream."""

    nontarget: NontargetConfig[LMDataConfig]

    @model_validator(mode="after")
    def _assert_targeted_invariants(self) -> "LMTargetedExperimentConfig":
        assert_targeted_faithfulness_off(self.pd)
        assert self.data.prompts_file is not None, (
            "targeted LM target stream must be a prompt pool (set data.prompts_file)"
        )
        assert self.nontarget.data.dataset_name is not None, (
            "targeted LM non-target stream must be parquet (set nontarget.data.dataset_name)"
        )
        return self


def nontarget_parquet_dir(cfg: LMTargetedExperimentConfig) -> Path:
    """The non-target parquet shard dir, resolved from `nontarget.data` the same way
    `experiments.lm.config._data` resolves the plain-LM parquet dir."""
    data = cfg.nontarget.data
    assert data.is_tokenized and not data.streaming, (
        "JAX trainer reads pre-tokenized parquet shards; tokenize offline first"
    )
    assert data.dataset_name == "parquet" and data.column_name == "input_ids", data
    assert data.data_files is not None
    shard_glob = Path(data.data_files)
    assert shard_glob.name == "*.parquet", f"expected a *.parquet glob, got {data.data_files}"
    return shard_glob.parent


def _targeted_data(cfg: LMTargetedExperimentConfig) -> DataConfig:
    """The engine's `data` seam for tPD = the NON-target parquet stream (the broad
    distribution the in-loop eval + perf read). The target stream is the fixed prompts, fed
    via the `sample_batch` seam, so it doesn't ride on `data` — persistent-PGD sources size
    off `TargetPromptGeometry.seq_len` instead of `data.seq_len` (SPEC S39). `global_batch`
    doubles as the TARGET batch (`make_prompt_sample_batch` reads it), which is what a `bsc`
    persistent scope sizes against."""
    return DataConfig(
        dir=nontarget_parquet_dir(cfg),
        seq_len=cfg.nontarget.data.max_seq_len,
        global_batch=cfg.nontarget.batch_size,
    )


def build_targeted(cfg: LMTargetedExperimentConfig, run_id: str) -> BuiltRun:
    """Convert the tPD schema to the engine's `BuiltRun`. Reuses the plain-LM target / eval /
    ci-fn resolution (`build_experiment_config` verbatim minus its `_data` call — the prompt
    `data` config asserts parquet); only `data` is tPD-specific (`_targeted_data`)."""
    target = _resolve_target(cfg)
    assert_canonical_algorithm_config(cfg)
    _assert_losses_supported(cfg, tuple(sc.name for sc in target.sites))
    return BuiltRun(
        pd=cfg.pd,
        runtime=cfg.runtime,
        cadence=cfg.cadence,
        run=run_instance(cfg, run_id),
        target=target,
        data=_targeted_data(cfg),
        ci_fn=ci_arch(cfg.pd.ci_config, lambda ci: _resolve_chunkwise_ci_arch(target, ci)),
        eval=_eval(cfg),
    )


def load_targeted_config(
    config_path: Path, run_id: str
) -> tuple[BuiltRun, LMTargetedExperimentConfig]:
    """Parse a tPD LM run YAML -> (built run, validated config). The composition root needs the
    config too (for the tPD-specific `nontarget` / `data.prompts_file` fields absent from
    `BuiltRun`), so it is validated once here and returned rather than re-parsed."""
    cfg = LMTargetedExperimentConfig(**yaml.safe_load(config_path.read_text()))
    assert_supported_weights_dtype(cfg)
    return build_targeted(cfg, run_id), cfg
