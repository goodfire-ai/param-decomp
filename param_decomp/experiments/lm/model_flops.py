"""Calculate analytical MFU from an LM decomposition config, without loading weights.

Sequence lengths are explicit because dataset references and prompt pools do not
encode their token counts in the training config. Counts assume uninterrupted
sequences; document packing can reduce the useful attention work.
"""

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import fire

from param_decomp.core.ci_fn.architecture import CIFnArchitectureFootprint
from param_decomp.core.components import SiteSpec
from param_decomp.core.configs import NontargetConfig, PDConfig, TargetedPDConfig
from param_decomp.core.flops.model import (
    TrainingFlops,
    decomposition_step_flops,
    nontarget_reconstruction_plans,
    reconstruction_plans,
    targeted_step_flops,
)
from param_decomp.core.flops.optimizer import OptimizerFlops, prepare_optimizer_flops
from param_decomp.core.flops.types import StepFlops
from param_decomp.core.hardware_utilization import (
    DeviceKind,
    ModelCost,
    peak_bf16_dense_flops_per_second,
)
from param_decomp.core.model import Positioned
from param_decomp.experiments.lm.config import (
    LMExperimentConfig,
    LMTargetedExperimentConfig,
    hf_model_variant,
    resolve_decomposition,
    resolve_lm_ci_fn_arch,
)
from param_decomp.experiments.lm.resolved import (
    AnyLMTargetConfig,
    LlamaSimpleMLPTargetConfig,
    Qwen36MoeTargetConfig,
    TargetConfig,
    dense_transformer_anatomy,
)
from param_decomp.experiments.model_flops import (
    StreamDescription,
    prepare_ordinary_step_flops,
    prepare_targeted_step_flops,
    stream_flops,
)
from param_decomp.infra.pretrain_cache import resolved_model_config_dir
from param_decomp.targets.llama_simple_mlp import load_model_config
from param_decomp.targets.qwen36_moe import qwen36_moe_flops
from param_decomp.targets.transformer import transformer_flops


@dataclass(frozen=True)
class TrainingCompute:
    model: TrainingFlops
    optimizer: OptimizerFlops
    step: int

    @property
    def total(self) -> float:
        return self.model.total + self.optimizer.total


def _stream_description(
    target: AnyLMTargetConfig,
    sites: tuple[SiteSpec, ...],
    ci_fn_arch: CIFnArchitectureFootprint,
    batch_size: int,
    sequence_length: int,
    data_root: Path,
    prefix: str,
) -> StreamDescription:
    match target:
        case TargetConfig(model_name=model_name):
            n_selected_blocks_per_token = None
            rule = partial(
                transformer_flops,
                hf_model_variant(model_name).arch,
                dense_transformer_anatomy(target),
                sequence_length=sequence_length,
            )
        case LlamaSimpleMLPTargetConfig(pretrain_run_path=run_path):
            n_selected_blocks_per_token = None
            arch = load_model_config(resolved_model_config_dir(data_root, run_path))
            rule = partial(
                transformer_flops,
                arch,
                dense_transformer_anatomy(target),
                sequence_length=sequence_length,
            )
        case Qwen36MoeTargetConfig(arch=arch):
            n_selected_blocks_per_token = arch.n_experts_per_token
            rule = partial(qwen36_moe_flops, arch, sequence_length=sequence_length)
    return StreamDescription(
        rule,
        sites,
        ci_fn_arch,
        batch_size,
        Positioned(sequence_length),
        n_selected_blocks_per_token,
        prefix,
    )


def prepare_lm_step_flops(
    pd: PDConfig,
    target: AnyLMTargetConfig,
    sites: tuple[SiteSpec, ...],
    ci_fn_arch: CIFnArchitectureFootprint,
    sequence_length: int,
    data_root: Path,
) -> StepFlops:
    """Resolve target shapes once, before entering the ordinary training loop."""
    stream = _stream_description(
        target, sites, ci_fn_arch, pd.batch_size, sequence_length, data_root, ""
    )
    return prepare_ordinary_step_flops(pd, stream)


def prepare_targeted_lm_step_flops(
    pd: TargetedPDConfig,
    nontarget: NontargetConfig,
    target: AnyLMTargetConfig,
    sites: tuple[SiteSpec, ...],
    ci_fn_arch: CIFnArchitectureFootprint,
    target_sequence_length: int,
    nontarget_sequence_length: int,
    data_root: Path,
) -> StepFlops:
    """Resolve both stream geometries once, before entering the targeted training loop."""
    target_stream = _stream_description(
        target, sites, ci_fn_arch, pd.batch_size, target_sequence_length, data_root, "target/"
    )
    nontarget_stream = _stream_description(
        target,
        sites,
        ci_fn_arch,
        nontarget.batch_size,
        nontarget_sequence_length,
        data_root,
        "nontarget/",
    )
    return prepare_targeted_step_flops(pd, nontarget, target_stream, nontarget_stream)


def decomposition_compute(
    config: LMExperimentConfig, sequence_length: int, data_root: Path, step: int
) -> TrainingCompute:
    """Count one global VPD training step; pretrained targets resolve metadata only."""
    if not 0 <= step < config.pd.steps:
        raise ValueError("Training step must be within the configured run")
    resolved = resolve_decomposition(config.target, config.decomposition, data_root)
    ci_fn_arch = resolve_lm_ci_fn_arch(resolved, config.decomposition.ci)
    stream = stream_flops(
        _stream_description(
            resolved.target,
            resolved.site_specs,
            ci_fn_arch,
            config.pd.batch_size,
            sequence_length,
            data_root,
            "",
        ),
        reconstruction_plans(config.pd, step),
    )
    return TrainingCompute(
        decomposition_step_flops(config.pd, resolved.site_specs, stream),
        prepare_optimizer_flops(
            config.pd, resolved.site_specs, ci_fn_arch, Positioned(sequence_length)
        )(step),
        step,
    )


def targeted_decomposition_compute(
    config: LMTargetedExperimentConfig,
    target_sequence_length: int,
    nontarget_sequence_length: int,
    data_root: Path,
    step: int,
) -> TrainingCompute:
    """Count both tPD streams at their independent global batch sizes and lengths."""
    if not 0 <= step < config.pd.steps:
        raise ValueError("Training step must be within the configured run")
    resolved = resolve_decomposition(config.target, config.decomposition, data_root)
    ci_fn_arch = resolve_lm_ci_fn_arch(resolved, config.decomposition.ci)
    target = stream_flops(
        _stream_description(
            resolved.target,
            resolved.site_specs,
            ci_fn_arch,
            config.pd.batch_size,
            target_sequence_length,
            data_root,
            "target/",
        ),
        reconstruction_plans(config.pd, step),
    )
    nontarget = stream_flops(
        _stream_description(
            resolved.target,
            resolved.site_specs,
            ci_fn_arch,
            config.nontarget.batch_size,
            nontarget_sequence_length,
            data_root,
            "nontarget/",
        ),
        nontarget_reconstruction_plans(config.nontarget),
    )
    return TrainingCompute(
        targeted_step_flops(target, nontarget),
        prepare_optimizer_flops(
            config.pd, resolved.site_specs, ci_fn_arch, Positioned(target_sequence_length)
        )(step),
        step,
    )


def performance_report(
    compute: TrainingCompute, device_kind: DeviceKind, n_devices: int, step_time_s: float
) -> dict[str, object]:
    """A JSON-ready breakdown; MFU is a fraction of dense BF16 fleet peak."""
    flops = compute.model
    model_cost = ModelCost(flops.total, n_devices, device_kind)
    total_cost = ModelCost(compute.total, n_devices, device_kind)
    return {
        "training_step": compute.step,
        "model_flops_per_step": flops.total,
        "forward_flops_per_step": flops.forward,
        "backward_flops_per_step": flops.backward,
        "device_kind": device_kind,
        "n_devices": n_devices,
        "peak_reference": "dense_bfloat16",
        "step_time_s": step_time_s,
        "optimizer_flops_per_step": compute.optimizer.total,
        "total_flops_per_step": compute.total,
        "ideal_step_time_without_optimizer_s": model_cost.ideal_step_time_s,
        "ideal_step_time_with_optimizer_s": total_cost.ideal_step_time_s,
        "mfu_without_optimizer": model_cost.mfu(step_time_s),
        "mfu_with_optimizer": total_cost.mfu(step_time_s),
        "optimizer_terms": [
            {
                "name": term.name,
                "n_repetitions": term.n_repetitions,
                "dtype": term.dtype,
                "contraction_flops": term.n_repetitions * term.contraction_flops,
                "elementwise_flops": term.n_repetitions * term.elementwise_flops,
                "total_flops": term.total,
            }
            for term in compute.optimizer.terms
        ],
        "terms": [
            {
                "name": term.name,
                "n_repetitions": term.n_repetitions,
                "forward_flops": term.n_repetitions * term.flops.forward,
                "backward_flops": term.n_repetitions * term.flops.backward,
                "total_flops": term.total,
            }
            for term in flops.terms
        ],
    }


def ordinary(
    config: str,
    data_root: str,
    sequence_length: int,
    device_kind: DeviceKind,
    step_time_s: float,
    step: int,
) -> None:
    """Report ordinary VPD MFU, using the global device count in runtime.mesh."""
    peak_bf16_dense_flops_per_second(device_kind)
    authored = LMExperimentConfig.from_file(config)
    flops = decomposition_compute(authored, sequence_length, Path(data_root), step)
    print(
        json.dumps(
            performance_report(
                flops, device_kind, authored.runtime.world_size.device_count, step_time_s
            ),
            indent=2,
        )
    )


def targeted(
    config: str,
    data_root: str,
    target_sequence_length: int,
    nontarget_sequence_length: int,
    device_kind: DeviceKind,
    step_time_s: float,
    step: int,
) -> None:
    """Report tPD MFU with the tokenized prompt length and broad-stream row length."""
    peak_bf16_dense_flops_per_second(device_kind)
    authored = LMTargetedExperimentConfig.from_file(config)
    flops = targeted_decomposition_compute(
        authored, target_sequence_length, nontarget_sequence_length, Path(data_root), step
    )
    print(
        json.dumps(
            performance_report(
                flops, device_kind, authored.runtime.world_size.device_count, step_time_s
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    fire.Fire({"ordinary": ordinary, "targeted": targeted})
