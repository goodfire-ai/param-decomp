"""Translate pre-refactor LM `final_config.yaml` files to `LMExperimentConfig`.

LM runs saved before "The sexy refactor" (PR #486) stored their config under
`final_config.yaml` with a flat schema (top-level `task_config`, `module_info`,
`loss_metric_configs`, `pretrained_model_*`, etc.). The new code reads
`run_meta.yaml` with the nested `ExperimentConfig` schema. This module bridges the
gap so old runs (and their cached checkpoints) remain loadable via `SavedLMRun`.
"""

from pathlib import Path
from typing import Any

import yaml

LEGACY_RUN_META_FILENAME = "final_config.yaml"

# spd.pretrain.* was renamed to param_decomp_lab.experiments.lm.pretrain.* in the
# package split. Pretrained-model classes in legacy configs still reference the old path.
_PRETRAIN_MODULE_RENAMES = {
    "spd.pretrain.": "param_decomp_lab.experiments.lm.pretrain.",
}


def _rewrite_classname(fqn: str) -> str:
    for old_prefix, new_prefix in _PRETRAIN_MODULE_RENAMES.items():
        if fqn.startswith(old_prefix):
            return new_prefix + fqn[len(old_prefix) :]
    return fqn


def _rename_classname_to_type(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Loss/eval metric entries used `classname` as the discriminator; new schema uses `type`."""
    out: list[dict[str, Any]] = []
    for entry in entries:
        renamed = {k: v for k, v in entry.items() if k != "classname"}
        renamed["type"] = entry["classname"]
        out.append(renamed)
    return out


# Old `ci_config` carried these optional fields that were removed in the refactor.
# When the legacy YAML records them as `None`, strip them so pydantic doesn't complain
# under `extra="forbid"`.
_LEGACY_CI_CONFIG_FIELDS_TO_DROP = (
    "reader_hidden_dims",
    "d_resid_ci_fn",
    "block_groups",
    "transition_attn_config",
    "transition_hidden_dim",
)


def _strip_legacy_ci_fields(ci_config: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(ci_config)
    for field in _LEGACY_CI_CONFIG_FIELDS_TO_DROP:
        if cleaned.get(field) is None:
            cleaned.pop(field, None)
    return cleaned


def _output_extract(legacy_attr: Any) -> int | str | None:
    """Translate `pretrained_model_output_attr` to the new `output_extract` field."""
    if isinstance(legacy_attr, str) and legacy_attr.startswith("idx_"):
        return int(legacy_attr.removeprefix("idx_"))
    return legacy_attr


def translate_legacy_lm_config(legacy_path: Path) -> dict[str, Any]:
    """Read a pre-refactor `final_config.yaml` and return a dict shaped like `LMExperimentConfig`.

    The result is intended to be passed to `LMExperimentConfig.model_validate(...)`. We
    return a dict (rather than instantiating the pydantic model here) so callers control
    the import and validation surface — and to keep this module free of import cycles
    against `experiments.lm.run`.
    """
    with open(legacy_path) as f:
        old = yaml.safe_load(f)

    task_cfg = old["task_config"]
    assert task_cfg.get("task_name") == "lm", (
        f"Legacy config at {legacy_path} is not an LM run (task_name={task_cfg.get('task_name')!r})"
    )

    components_optimizer: dict[str, Any] = {"lr_schedule": old["lr_schedule"]}
    if old.get("grad_clip_norm_components") is not None:
        components_optimizer["grad_clip_norm"] = old["grad_clip_norm_components"]
    ci_fn_optimizer: dict[str, Any] = {"lr_schedule": old["lr_schedule"]}
    if old.get("grad_clip_norm_ci_fns") is not None:
        ci_fn_optimizer["grad_clip_norm"] = old["grad_clip_norm_ci_fns"]

    pd: dict[str, Any] = {
        "seed": old.get("seed", 0),
        "n_mask_samples": old["n_mask_samples"],
        "ci_config": _strip_legacy_ci_fields(old["ci_config"]),
        "sampling": old.get("sampling", "continuous"),
        "sigmoid_type": old.get("sigmoid_type", "leaky_hard"),
        "decomposition_targets": old["module_info"],
        "identity_decomposition_targets": old.get("identity_module_info"),
        "use_delta_component": old.get("use_delta_component", True),
        "loss_metrics": _rename_classname_to_type(old.get("loss_metric_configs", [])),
        "components_optimizer": components_optimizer,
        "ci_fn_optimizer": ci_fn_optimizer,
        "steps": old["steps"],
        "batch_size": old["batch_size"],
        "faithfulness_warmup_steps": old.get("faithfulness_warmup_steps", 0),
        "faithfulness_warmup_lr": old.get("faithfulness_warmup_lr", 0.001),
        "faithfulness_warmup_weight_decay": old.get("faithfulness_warmup_weight_decay", 0.0),
    }

    runtime: dict[str, Any] = {
        "autocast_bf16": old.get("autocast_bf16", True),
        "device": "cuda",
        "dp": None,
    }

    cadence: dict[str, Any] = {"train_log_every": old["train_log_freq"]}
    if old.get("save_freq") is not None:
        cadence["save_every"] = old["save_freq"]

    eval_metrics = _rename_classname_to_type(old.get("eval_metric_configs", []))
    eval_cfg: dict[str, Any] | None = None
    if eval_metrics or old.get("eval_freq") is not None:
        eval_cfg = {
            "batch_size": old["eval_batch_size"],
            "n_steps": old["n_eval_steps"],
            "every": old["eval_freq"],
            "slow_every": old["slow_eval_freq"],
            "slow_on_first_step": old.get("slow_eval_on_first_step", True),
            "metrics": eval_metrics,
        }

    pretrained_model_name = old.get("pretrained_model_name")
    assert pretrained_model_name, (
        f"Legacy LM configs without `pretrained_model_name` (HF-target runs) are not "
        f"supported by the migrator yet; config at {legacy_path}"
    )
    target: dict[str, Any] = {
        "spec": {
            "kind": "pretrained",
            "model_class": _rewrite_classname(old["pretrained_model_class"]),
            "run_path": pretrained_model_name,
        },
        "output_extract": _output_extract(old.get("pretrained_model_output_attr")),
    }

    data: dict[str, Any] = {
        "tokenizer_name": old["tokenizer_name"],
        "max_seq_len": task_cfg["max_seq_len"],
        "buffer_size": task_cfg.get("buffer_size", 1000),
        "dataset_name": task_cfg["dataset_name"],
        "column_name": task_cfg.get("column_name", "text"),
        "train_split": task_cfg.get("train_data_split", "train"),
        "eval_split": task_cfg.get("eval_data_split", "test"),
        "shuffle_each_epoch": task_cfg.get("shuffle_each_epoch", True),
        "is_tokenized": task_cfg.get("is_tokenized", False),
        "streaming": task_cfg.get("streaming", False),
    }

    wandb_cfg: dict[str, Any] | None = None
    if old.get("wandb_project"):
        wandb_cfg = {"project": old["wandb_project"]}

    return {
        "pd": pd,
        "runtime": runtime,
        "cadence": cadence,
        "target": target,
        "data": data,
        "eval": eval_cfg,
        "wandb": wandb_cfg,
    }
