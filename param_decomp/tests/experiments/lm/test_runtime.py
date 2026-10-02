"""The `runtime:` section: explicit topology and the placement-preset vocabulary."""

import typing
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from param_decomp.core.configs import (
    HsdpMeshShape,
    PlacementTableConfig,
    ResidentMeshShape,
)
from param_decomp.core.placement import PRESET_NAMES
from param_decomp.core.run import JaxProfilerTrace, NsightCaptureWindow
from param_decomp.core.world_size import MultiNode, SingleNode
from param_decomp.experiments.lm.config import LMExperimentConfig
from param_decomp.experiments.lm.runtime import (
    TUNED_V2_AUTOTUNE1_COMPILER_OPTIONS,
    TUNED_V2_COMPILER_OPTIONS,
    TYPED_ENV_VARS,
    AdHocProfiling,
    LaunchEnv,
    NsightSystemsProfiling,
    ProfilingDisabled,
    RuntimeConfig,
)
from param_decomp.experiments.lm.training import engine_profiling

CONFIGS = Path(__file__).parents[3] / "experiments" / "lm" / "configs"
SEAT = CONFIGS / "llama3_1_8b.yaml"

_MINIMAL_RUNTIME: dict[str, Any] = {
    "mesh": {"replicate": 1, "fsdp": 1, "tp": 1},
    "sharding": "zero1",
    "compilation_cache_dir": "~/.cache/param-decomp/xla",
    "compiler_options": "tuned-v2",
}

_EXPLICIT_TABLE: dict[str, Any] = {
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


def test_topology_is_explicit_and_fails_closed():
    def runtime(**overrides: Any) -> RuntimeConfig:
        return RuntimeConfig.model_validate(_MINIMAL_RUNTIME | overrides)

    configured = runtime(mesh={"replicate": 16, "fsdp": 4, "tp": 2})
    assert configured.mesh == HsdpMeshShape(replicate=16, fsdp=4, tp=2)
    assert configured.world_size == MultiNode(n_nodes=16)
    assert configured.data_parallel_size == 64

    for missing in ("replicate", "fsdp", "tp"):
        authored = {"replicate": 16, "fsdp": 4, "tp": 2}
        del authored[missing]
        with pytest.raises(ValidationError):
            runtime(mesh=authored)

    # Removed launch intent is not part of the live authoring schema.
    with pytest.raises(ValidationError):
        runtime(launch="external")

    with pytest.raises(ValidationError):
        RuntimeConfig.model_validate({"dp": 128, "gpus_per_node": 8, "tp": 2, "sharding": "zero1"})


def test_retired_flat_mesh_keys_refuse_with_the_nested_spelling():
    """The pre-nesting flat axis keys die at parse, and the refusal says the nested
    spelling — not a bare extra-keys error."""
    flat = {key: value for key, value in _MINIMAL_RUNTIME.items() if key != "mesh"}
    with pytest.raises(ValidationError, match=r"nested under `runtime.mesh:`"):
        RuntimeConfig.model_validate(flat | {"replicate": 4, "fsdp": 8, "tp": 1})
    with pytest.raises(ValidationError, match=r"nested under `runtime.mesh:`"):
        RuntimeConfig.model_validate(flat | {"data": 4, "tp": 2})


def test_resident_mesh_parses():
    """The two-axis `mesh: {data, tp}` spelling parses to `ResidentMeshShape`. Which
    sharding it pairs with is not a parse-time fact: the placement table's mesh
    vocabulary is checked at construction (`test_placement`)."""
    configured = RuntimeConfig.model_validate(
        _MINIMAL_RUNTIME | {"mesh": {"data": 4, "tp": 2}, "sharding": "zero1-replicated-resident"}
    )
    assert configured.mesh == ResidentMeshShape(data=4, tp=2)
    assert configured.world_size == SingleNode(n_gpus=8)
    assert configured.data_parallel_size == 4
    # A mixed mesh spelling matches neither shape (disjoint required keys, extra=forbid).
    with pytest.raises(ValidationError):
        RuntimeConfig.model_validate(_MINIMAL_RUNTIME | {"mesh": {"data": 4, "fsdp": 2, "tp": 1}})


@pytest.mark.parametrize(
    "mesh",
    [
        {"replicate": 3, "fsdp": 2, "tp": 2},
        {"replicate": 1, "fsdp": 9, "tp": 1},
        {"data": 3, "tp": 4},
        {"data": 17, "tp": 1},
    ],
)
def test_runtime_refuses_partial_multi_node_worlds_at_parse(mesh: dict[str, int]):
    with pytest.raises(ValidationError, match="World size must"):
        RuntimeConfig.model_validate(_MINIMAL_RUNTIME | {"mesh": mesh})


@pytest.mark.parametrize("devices", [1, 7, 8, 16, 24])
def test_world_size_is_derived_without_changing_the_saved_schema(devices: int):
    configured = RuntimeConfig.model_validate(
        _MINIMAL_RUNTIME | {"mesh": {"data": devices, "tp": 1}}
    )
    assert configured.world_size.device_count == devices
    saved = configured.model_dump(mode="json")
    assert "world_size" not in saved
    assert RuntimeConfig.model_validate(saved).world_size == configured.world_size


def test_dead_tuned_v1_spellings_refuse_naming_the_successor():
    """jaxlib 0.11 removed the pipelined-collective trio the tuned-v1 presets froze, so
    the retired tokens can never again mean their exact flag sets: each refuses at parse
    naming its successor, never silently resolving to different flags."""
    for dead in ("tuned-v1", "tuned-v1-resident"):
        with pytest.raises(ValidationError, match="retired.*Author 'tuned-v2'"):
            RuntimeConfig.model_validate(_MINIMAL_RUNTIME | {"compiler_options": dead})
    with pytest.raises(ValidationError, match="retired.*Author 'tuned-v2-autotune1'"):
        RuntimeConfig.model_validate(
            _MINIMAL_RUNTIME | {"compiler_options": "tuned-v1-resident-autotune1"}
        )


def test_compiler_options_is_required_and_fails_closed():
    """REQUIRED, no default, no merge: every run's flags trace to a visible authored token —
    a preset name from the closed vocabulary, or an explicit dict used verbatim."""
    absent = {key: value for key, value in _MINIMAL_RUNTIME.items() if key != "compiler_options"}
    with pytest.raises(ValidationError, match="compiler_options"):
        RuntimeConfig.model_validate(absent)

    with pytest.raises(ValidationError, match="tuned-v2"):
        RuntimeConfig.model_validate(_MINIMAL_RUNTIME | {"compiler_options": "tuned-v3"})

    with pytest.raises(ValidationError, match="gpu_enable_triton_gemm"):
        RuntimeConfig.model_validate(
            _MINIMAL_RUNTIME | {"compiler_options": {"gpu_enable_triton_gemm": True}}
        )


def test_compiler_options_resolve_with_no_merging():
    def resolved(authored: object) -> dict[str, bool | int | str]:
        runtime = RuntimeConfig.model_validate(_MINIMAL_RUNTIME | {"compiler_options": authored})
        return runtime.resolved_compiler_options

    assert resolved("tuned-v2") == dict(TUNED_V2_COMPILER_OPTIONS)
    assert resolved("bare") == {}

    # tuned-v2-autotune1 = tuned-v2 plus the autotune dial, nothing else.
    assert resolved("tuned-v2-autotune1") == dict(TUNED_V2_AUTOTUNE1_COMPILER_OPTIONS)
    assert dict(TUNED_V2_AUTOTUNE1_COMPILER_OPTIONS) == dict(TUNED_V2_COMPILER_OPTIONS) | {
        "xla_gpu_autotune_level": 1
    }

    # The one tuned set serves every placement, resident included: double-buffering off.
    assert TUNED_V2_COMPILER_OPTIONS["xla_gpu_enable_while_loop_double_buffering"] is False

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
    runtime = RuntimeConfig.model_validate(_MINIMAL_RUNTIME | {"sharding": _EXPLICIT_TABLE})
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
    "xla_flags": "--xla_gpu_nccl_termination_timeout_seconds=120",
    "nccl_debug": "INFO",
    "malloc_arena_max": 4,
    "env": {"SOME_ONE_OFF_VAR": "1"},
}


def test_a_pinned_launch_config_still_round_trips(tmp_path: Path):
    """The current pinned config shape round-trips without moving `runtime.launch_env`."""
    raw = yaml.safe_load(SEAT.read_text())
    raw["runtime"]["launch_env"] = _PINNED_LAUNCH_ENV
    conditioning = tmp_path / "launch_config.yaml"
    conditioning.write_text(yaml.safe_dump(raw, sort_keys=False))

    cfg = LMExperimentConfig.from_file(conditioning)

    assert cfg.runtime.mesh == ResidentMeshShape(data=32, tp=4)
    assert cfg.runtime.sharding == "owner-replicated-resident"
    assert cfg.runtime.launch_env.as_env(None) == {
        "NCCL_DEBUG": "INFO",
        "MALLOC_ARENA_MAX": "4",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85",
        "XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB": "512",
        "XLA_FLAGS": "--xla_gpu_nccl_termination_timeout_seconds=120",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
        "SOME_ONE_OFF_VAR": "1",
    }
    dumped = cfg.model_dump(mode="json")
    assert dumped["runtime"]["launch_env"] == _PINNED_LAUNCH_ENV
    assert "launch_env" not in dumped and "runtime" not in dumped["pd"]


def test_launch_env_composes_inherited_xla_flags():
    """`as_env` merges the config's `XLA_FLAGS` with flags the environment already
    carries, keyed by flag name, three arms: disjoint flags compose (config's first,
    the environment's extras appended — the caller's exports survive the bootstrap),
    an identical duplicate dedupes silently, and the same flag with a DIFFERENT value
    refuses — a reviewed config value conflicting with ambient env is a confused state,
    never a silent resolution."""
    launch_env = LaunchEnv(xla_flags="--xla_a=1 --xla_b=2")
    composed = "--xla_a=1 --xla_b=2 --xla_force_host_platform_device_count=8"
    assert launch_env.as_env("--xla_force_host_platform_device_count=8")["XLA_FLAGS"] == composed
    assert (
        launch_env.as_env("--xla_b=2 --xla_force_host_platform_device_count=8")["XLA_FLAGS"]
        == composed
    )
    with pytest.raises(AssertionError, match=r"conflict on --xla_b.*'--xla_b=2'.*'--xla_b=3'"):
        launch_env.as_env("--xla_b=3")
    assert launch_env.as_env(None)["XLA_FLAGS"] == "--xla_a=1 --xla_b=2"
    assert launch_env.as_env("")["XLA_FLAGS"] == "--xla_a=1 --xla_b=2"


def test_launch_env_block_refuses_keys_typed_fields_own():
    """`env: {XLA_FLAGS: ...}` would have replaced the merged typed string (dropping the
    termination-timeout default and the inherited-flag composition); every typed field's
    variable refuses at config load, naming the field to author instead. `as_env` renders
    exactly the owned set, so the refusal list and the rendering cannot drift."""
    with pytest.raises(ValidationError, match=r"XLA_FLAGS -> launch_env\.xla_flags"):
        LaunchEnv(env={"XLA_FLAGS": "--xla_gpu_nccl_termination_timeout_seconds=5"})
    with pytest.raises(ValidationError, match=r"NCCL_DEBUG -> launch_env\.nccl_debug"):
        LaunchEnv(env={"NCCL_DEBUG": "INFO", "SOME_ONE_OFF_VAR": "1"})
    rendered = LaunchEnv(
        xla_python_client_allocator="platform", env={"SOME_ONE_OFF_VAR": "1"}
    ).as_env(None)
    assert set(rendered) == set(TYPED_ENV_VARS) | {"SOME_ONE_OFF_VAR"}


def test_the_launch_env_block_has_no_second_home(tmp_path: Path):
    """Anti-vacuity for the test above: the nesting is load-bearing, not incidental —
    hoisting the block anywhere else is refused, so a silent migration cannot pass."""
    raw = yaml.safe_load(SEAT.read_text())
    raw["launch_env"] = _PINNED_LAUNCH_ENV
    hoisted = tmp_path / "launch_config.yaml"
    hoisted.write_text(yaml.safe_dump(raw, sort_keys=False))

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        LMExperimentConfig.from_file(hoisted)
