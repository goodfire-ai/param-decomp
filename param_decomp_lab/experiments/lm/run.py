"""LM PD experiment: the `build_target` consumer bridge.

The `LMExperimentConfig` schema (target spec, data config) lives in
`param_decomp_config.lm`.

The torch training driver has been retired (the JAX single-pool trainer is
production; the torch oracle lives at git tag `torch-oracle`). Torch-run loading was
dropped too (the `SavedLMRun` reload + `component_model_io` + vendored archs); a
JAX-native run loader returns as the #10 torch->jax adapter. What survives is
`build_target` — `JaxPDAdapter` builds the target *architecture* from config to derive
layer topology (no checkpoint restore).
"""

import importlib

import torch
import torch.nn as nn

from param_decomp_config.lm import (
    HFTarget,
    HFWeightsInVendored,
    LMTargetConfig,
    PretrainedTarget,
)
from param_decomp_lab.distributed import ensure_cached_and_call
from param_decomp_lab.infra.paths import validate_path


def _resolve_class(fqn: str) -> type:
    """Load a class from a fully-qualified name, e.g. 'transformers.LlamaForCausalLM'."""
    module_path, _, class_name = fqn.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def build_target(target_cfg: LMTargetConfig) -> nn.Module:
    """Load the LM target model in eval mode, dispatching on `target_cfg.spec.kind`."""
    spec = target_cfg.spec
    cls = _resolve_class(spec.model_class)
    match spec:
        case HFTarget():
            target_model = ensure_cached_and_call(cls.from_pretrained, spec.model_name)
        case PretrainedTarget():
            from param_decomp_lab.experiments.lm.pretrain.run_info import PretrainRunInfo

            run_info = ensure_cached_and_call(
                PretrainRunInfo.from_path, validate_path(spec.run_path)
            )
            # Older PretrainRunInfo objects predate model_type; default it from the model class.
            if "model_type" not in run_info.model_config_dict:
                run_info.model_config_dict["model_type"] = spec.model_class.rsplit(".", 1)[-1]
            target_model = cls.from_run_info(run_info)
        case HFWeightsInVendored():
            assert hasattr(cls, "from_hf_pretrained"), (
                f"HFWeightsInVendored target requires {spec.model_class!r} to expose a "
                "`from_hf_pretrained` classmethod"
            )
            target_model = ensure_cached_and_call(cls.from_hf_pretrained, spec.model_name)
    if target_cfg.activation_checkpointing:
        assert hasattr(target_model, "enable_activation_checkpointing"), (
            f"activation_checkpointing=True but {type(target_model).__name__} has no "
            "`enable_activation_checkpointing()` method"
        )
        target_model.enable_activation_checkpointing()
    if target_cfg.weights_dtype == "bfloat16":
        target_model = target_model.to(torch.bfloat16)
    target_model.eval()
    return target_model
