"""The `runtime:` section: explicit topology and the placement-preset vocabulary."""

import typing
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from param_decomp.core.configs import PlacementTableConfig
from param_decomp.core.placement import PRESET_NAMES
from param_decomp.core.run import JaxProfilerTrace, NsightCaptureWindow
from param_decomp.experiments.lm.config import LMExperimentConfig
from param_decomp.experiments.lm.runtime import (
    TUNED_V1_COMPILER_OPTIONS,
    AdHocProfiling,
    NsightSystemsProfiling,
    ProfilingDisabled,
    RuntimeConfig,
)
from param_decomp.experiments.lm.training import engine_profiling

CONFIGS = Path(__file__).parents[3] / "experiments" / "lm" / "configs"
SEAT = CONFIGS / "llama8b_l18_C49k_200k.yaml"

_MINIMAL_RUNTIME: dict[str, Any] = {
    "replicate": 1,
    "fsdp": 1,
    "tp": 1,
    "sharding": "zero1",
    "compilation_cache_dir": "~/.cache/param-decomp/xla",
    "compiler_options": "tuned-v1",
}


def test_topology_is_explicit_and_fails_closed():
    def runtime(**overrides: Any) -> RuntimeConfig:
        return RuntimeConfig.model_validate(_MINIMAL_RUNTIME | overrides)

    configured = runtime(replicate=16, fsdp=4, tp=2)
    assert configured.world_size == 128
    assert configured.data_parallel_size == 64

    for missing in ("replicate", "fsdp", "tp"):
        authored = dict(_MINIMAL_RUNTIME, replicate=16, fsdp=4, tp=2)
        del authored[missing]
        with pytest.raises(ValidationError):
            RuntimeConfig.model_validate(authored)

    # Removed launch intent is not part of the live authoring schema.
    with pytest.raises(ValidationError):
        runtime(launch="slurm")

    with pytest.raises(ValidationError):
        RuntimeConfig.model_validate({"dp": 128, "gpus_per_node": 8, "tp": 2, "sharding": "zero1"})


def test_compiler_options_is_required_and_fails_closed():
    """REQUIRED, no default, no merge: every run's flags trace to a visible authored token —
    a preset name from the closed vocabulary, or an explicit dict used verbatim."""
    absent = {key: value for key, value in _MINIMAL_RUNTIME.items() if key != "compiler_options"}
    with pytest.raises(ValidationError, match="compiler_options"):
        RuntimeConfig.model_validate(absent)

    with pytest.raises(ValidationError, match="tuned-v1"):
        RuntimeConfig.model_validate(_MINIMAL_RUNTIME | {"compiler_options": "tuned-v2"})

    with pytest.raises(ValidationError, match="gpu_enable_triton_gemm"):
        RuntimeConfig.model_validate(
            _MINIMAL_RUNTIME | {"compiler_options": {"gpu_enable_triton_gemm": True}}
        )


def test_compiler_options_resolve_with_no_merging():
    def resolved(authored: object) -> dict[str, bool | int | str]:
        runtime = RuntimeConfig.model_validate(_MINIMAL_RUNTIME | {"compiler_options": authored})
        return runtime.resolved_compiler_options

    assert resolved("tuned-v1") == dict(TUNED_V1_COMPILER_OPTIONS)
    assert resolved("bare") == {}

    # An explicit dict is the run's COMPLETE flag set, verbatim — nothing folded in.
    explicit = {"xla_disable_hlo_passes": "rematerialization"}
    assert resolved(explicit) == explicit


def test_profiling_arms_round_trip_and_lower_to_engine_data():
    """`runtime.profiling` is the ONE profiling surface: each authored arm parses, dumps
    back to the same document, and lowers to the engine's typed value — the engine reads
    no environment, so this mapping is the whole wire."""

    def runtime(profiling: dict[str, Any]) -> RuntimeConfig:
        cfg = RuntimeConfig.model_validate(_MINIMAL_RUNTIME | {"profiling": profiling})
        assert cfg.model_dump(mode="json")["profiling"] == profiling
        return cfg

    disabled = runtime({"kind": "disabled"})
    assert disabled.profiling == ProfilingDisabled()
    assert engine_profiling(disabled.profiling) is None

    ad_hoc = runtime({"kind": "ad_hoc", "steps": 3})
    assert ad_hoc.profiling == AdHocProfiling(steps=3)
    assert engine_profiling(ad_hoc.profiling) == JaxProfilerTrace(steps=3)

    nsight = runtime(
        {"kind": "nsight_systems", "version": "2026.4.1", "warmup_steps": 2, "capture_steps": 3}
    )
    assert nsight.profiling == NsightSystemsProfiling(
        version="2026.4.1", warmup_steps=2, capture_steps=3
    )
    assert engine_profiling(nsight.profiling) == NsightCaptureWindow(
        warmup_steps=2, capture_steps=3
    )

    with pytest.raises(ValidationError):
        runtime({"kind": "ad_hoc", "steps": 0})
    # The retired env-var plumbing must not resurface as an unvalidated escape hatch.
    with pytest.raises(ValidationError):
        runtime({"kind": "ad_hoc", "steps": 3, "env": {"PD_AD_HOC_PROFILE_STEPS": "3"}})


def test_preset_names_match_placement_presets():
    """The authored preset vocabulary and the engine's resolver must name the same set —
    a preset that parses but has no rules (or vice versa) is a config that dies at build."""
    ann = RuntimeConfig.model_fields["sharding"].annotation
    literals = [a for a in typing.get_args(ann) if typing.get_origin(a) is typing.Literal]
    assert literals, ann
    assert set(typing.get_args(literals[0])) == set(PRESET_NAMES)


def test_sharding_accepts_an_explicit_placement_table():
    """The non-preset arm of `sharding` is reachable through the section, not just as a
    standalone type: an authored table lands as `PlacementTableConfig`."""
    table = {
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
            "ns_compute": {},
        },
        "ci_fn": {
            "attention": {
                "optimizer_state": {"d_model": ["fsdp", "replicate"]},
                "compute_weights": {"d_model": "fsdp"},
                "operands": {},
                "ns_compute": {},
            },
            "ffn": {
                "optimizer_state": {"ffn_hidden": ["fsdp", "tp", "replicate"]},
                "compute_weights": {"ffn_hidden": ["fsdp", "tp"]},
                "operands": {},
                "ns_compute": {},
            },
            "input": {
                "optimizer_state": {"input": "tp", "d_model": ["fsdp", "replicate"]},
                "compute_weights": {"input": "tp", "d_model": "fsdp"},
                "operands": {},
                "ns_compute": {},
            },
            "output": {
                "optimizer_state": {"d_model": ["fsdp", "replicate"], "C": "tp"},
                "compute_weights": {"d_model": "fsdp", "C": "tp"},
                "operands": {},
                "ns_compute": {},
            },
            "vectors": {"ffn_hidden": "tp", "C": "tp"},
            "activations": {
                "batch": ["replicate", "fsdp"],
                "input": "tp",
                "q_head": "tp",
                "kv_head": "tp",
                "ffn_hidden": "tp",
                "C": "tp",
            },
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
    runtime = RuntimeConfig.model_validate(_MINIMAL_RUNTIME | {"sharding": table})
    assert isinstance(runtime.sharding, PlacementTableConfig)
    assert runtime.sharding.components.optimizer_state == {
        "stack": "replicate",
        "d_in": "fsdp",
        "d_out": "fsdp",
        "C": "tp",
    }


_PINNED_LAUNCH_ENV = {
    "xla_python_client_mem_fraction": 0.85,
    "xla_python_client_allocator": "platform",
    "xla_pjrt_gpu_host_memory_limit_gb": 512,
    "nccl_debug": "INFO",
    "malloc_arena_max": 4,
    "env": {"SOME_ONE_OFF_VAR": "1"},
}


def test_a_pinned_launch_config_still_round_trips(tmp_path: Path):
    """The current pinned config shape round-trips without moving `runtime.launch_env`."""
    raw = yaml.safe_load(SEAT.read_text())
    raw["runtime"]["launch_env"] = _PINNED_LAUNCH_ENV
    pinned = tmp_path / "launch_config.yaml"
    pinned.write_text(yaml.safe_dump(raw, sort_keys=False))

    cfg = LMExperimentConfig.from_file(pinned)

    assert (cfg.runtime.replicate, cfg.runtime.fsdp, cfg.runtime.tp) == (4, 8, 1)
    assert cfg.runtime.sharding == "zero1"
    assert cfg.runtime.launch_env.as_env() == {
        "NCCL_DEBUG": "INFO",
        "MALLOC_ARENA_MAX": "4",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85",
        "XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB": "512",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
        "SOME_ONE_OFF_VAR": "1",
    }
    dumped = cfg.model_dump(mode="json")
    assert dumped["runtime"]["launch_env"] == _PINNED_LAUNCH_ENV
    assert "launch_env" not in dumped and "runtime" not in dumped["pd"]


def test_the_launch_env_block_has_no_second_home(tmp_path: Path):
    """Anti-vacuity for the test above: the nesting is load-bearing, not incidental —
    hoisting the block anywhere else is refused, so a silent migration cannot pass."""
    raw = yaml.safe_load(SEAT.read_text())
    raw["launch_env"] = _PINNED_LAUNCH_ENV
    hoisted = tmp_path / "launch_config.yaml"
    hoisted.write_text(yaml.safe_dump(raw, sort_keys=False))

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        LMExperimentConfig.from_file(hoisted)
