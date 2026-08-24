"""The LM run's compute substrate: `RuntimeConfig` (the `runtime:` section of
`LMExperimentConfig`) and the pre-process env surface nested inside it (`launch_env`).

Substrate, not algorithm — every value here perturbs numerics or memory without changing
what is computed, and none of it reaches the engine as an object: the composition root
(`training.py`) unpacks it into the engine's primitives (device counts, a placement spec,
remat flags, compiler options). It is the LM's alone; the single-device toys have no
substrate to author, so their schemas carry no `runtime:` section at all.

Deliberately free of jax and of the rest of the LM schema: `run.py` validates the
`runtime:` block and exports `launch_env` BEFORE importing JAX, so this module must stay
cheap to import.
"""

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal

from pydantic import (
    Discriminator,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
)

from param_decomp.core.base_config import BaseConfig
from param_decomp.core.configs import PlacementTableConfig

TUNED_V1_COMPILER_OPTIONS: Mapping[str, bool | int | str] = MappingProxyType(
    {
        "xla_gpu_enable_latency_hiding_scheduler": True,
        "xla_gpu_enable_triton_gemm": False,
        "xla_gpu_enable_command_buffer": "",
        "xla_gpu_enable_highest_priority_async_stream": True,
        "xla_gpu_all_reduce_combine_threshold_bytes": 1073741824,
        "xla_gpu_all_gather_combine_threshold_bytes": 1073741824,
        "xla_gpu_reduce_scatter_combine_threshold_bytes": 134217728,
        "xla_gpu_enable_pipelined_all_gather": True,
        "xla_gpu_enable_pipelined_reduce_scatter": True,
        "xla_gpu_enable_pipelined_all_reduce": True,
        "xla_gpu_enable_while_loop_double_buffering": True,
        "xla_gpu_enable_all_gather_combine_by_dim": False,
        "xla_gpu_enable_reduce_scatter_combine_by_dim": False,
    }
)
"""What `compiler_options: tuned-v1` resolves to — the ONE copy of the tuned set, frozen.
It is MaxText's H100 recipe: latency-hiding scheduler + 1 GiB collective-combine
thresholds + pipelined collectives + while-loop double-buffering;
`xla_gpu_enable_command_buffer: ''` disables CUDA-graph capture, a correctness guard.
A change to the tuned set is a NEW preset name (`tuned-v2`), never an edit here — pinned
configs authoring `tuned-v1` must keep meaning these exact flags."""


class LaunchEnv(BaseConfig):
    """The process-environment surface a rank runs with — the XLA *client* knobs (mem
    fraction / allocator / host-memory limit), NCCL/glibc tuning, and a free-form env escape
    hatch — lifted into the run config so a run's `launch_config.yaml` fully captures its
    environment (tracking + repro), and A/B-ing a knob is a config edit, not a launcher edit.

    XLA *compiler* flags are NOT here — they go through `RuntimeConfig.compiler_options`
    (passed natively to each jit, no env round-trip; see that field). This class is only the
    env that must exist before the process starts (read at backend/NCCL init).

    The pre-JAX bootstrap (`run.py`) exports it before importing JAX; whoever spawns the
    ranks renders the same map into their environment. `LD_LIBRARY_PATH` is NOT here (it is
    machine-specific — resolved against the local CUDA install by whoever starts the
    process — not a tracked decision). These defaults are the single source of truth: a
    submitter renders them, it does not carry its own set.
    """

    xla_python_client_mem_fraction: PositiveFloat = 0.92
    """`XLA_PYTHON_CLIENT_MEM_FRACTION` — the BFC pool cap as a fraction of HBM."""
    xla_python_client_allocator: str | None = None
    """`XLA_PYTHON_CLIENT_ALLOCATOR` — e.g. `platform` for the on-demand cudaMalloc allocator
    (avoids BFC fragmentation OOMs near the HBM cap, at some per-alloc cost). `None` leaves
    the XLA default (BFC)."""
    xla_pjrt_gpu_host_memory_limit_gb: PositiveInt = 1024
    """`XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB` — cap on XLA's pinned host-staging pool
    (allocated on demand)."""
    nccl_debug: str = "WARN"
    """`NCCL_DEBUG` — overrides the INFO + SUBSYS=ALL default some clusters set, which logs
    every collective and bloats a run's logs to tens of GB."""
    malloc_arena_max: PositiveInt = 2
    """`MALLOC_ARENA_MAX` — caps glibc malloc arenas to bound host RSS under many threads."""
    env: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Arbitrary extra exports merged into the rank env LAST (after the typed knobs), "
            "so it can override any of them. The escape hatch for a one-off var without a "
            "schema field."
        ),
    )

    def as_env(self) -> dict[str, str]:
        """Render the ordered `{VAR: value}` map a rank's environment must carry (sans
        `LD_LIBRARY_PATH`, which is machine-specific). Only the env that must exist before
        backend/NCCL init — XLA *compiler* flags are passed natively via
        `RuntimeConfig.compiler_options`, not here. Later keys override earlier, so the
        free-form `env` block wins last."""
        rendered: dict[str, str] = {
            "NCCL_DEBUG": self.nccl_debug,
            "MALLOC_ARENA_MAX": str(self.malloc_arena_max),
            "XLA_PYTHON_CLIENT_MEM_FRACTION": str(self.xla_python_client_mem_fraction),
            "XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB": str(self.xla_pjrt_gpu_host_memory_limit_gb),
        }
        if self.xla_python_client_allocator is not None:
            rendered["XLA_PYTHON_CLIENT_ALLOCATOR"] = self.xla_python_client_allocator
        rendered |= self.env
        return rendered


class ProfilingDisabled(BaseConfig):
    kind: Literal["disabled"] = "disabled"


class AdHocProfiling(BaseConfig):
    """An in-process `jax.profiler` run: after a fixed warmup the trainer traces `steps`
    steps into `<run_dir>/profile` and exits — no step-0 checkpoint, no training past the
    trace. The run is a measurement, not a trajectory."""

    kind: Literal["ad_hoc"] = "ad_hoc"
    steps: PositiveInt


class NsightSystemsProfiling(BaseConfig):
    kind: Literal["nsight_systems"] = "nsight_systems"
    version: Literal["2026.4.1"]
    warmup_steps: NonNegativeInt
    capture_steps: PositiveInt


ProfilingConfig = Annotated[
    ProfilingDisabled | AdHocProfiling | NsightSystemsProfiling,
    Discriminator("kind"),
]


class RuntimeConfig(BaseConfig):
    """Compute substrate: explicit logical mesh, placement, rematerialization, XLA compiler
    flags, and the pre-process env surface (`launch_env`).

    Perturbs numerics but doesn't change the algorithm.
    """

    replicate: PositiveInt = Field(
        description="Logical model-replication and cross-replica ownership axis.",
    )
    fsdp: PositiveInt = Field(
        description="Logical parameter-sharding axis; also shards the data batch.",
    )
    tp: PositiveInt = Field(
        description="Logical Megatron tensor-parallel axis; batch activations replicate over it.",
    )
    sharding: Literal["owner", "zero1", "ddp"] | PlacementTableConfig = Field(
        description=(
            "Placement policy for the trainable state (placement.py). REQUIRED, no "
            "default — a layout this consequential is written down per config. Presets: "
            "`zero1` = intra-matrix ZeRO-1 over the full data mesh (no row shards the "
            "component stack axis, so every semantic group is placeable; ~equivalent "
            "comms to `owner` under elementwise optimizers); `owner` = whole-matrix "
            "ownership (stack ÷replicate, d ÷fsdp, C ÷tp) — the muon-motivated layout "
            "(Newton-Schulz stays node-local); a semantic group whose stack does not "
            "tile ÷replicate refuses at config build (placement.from_config, "
            "pre-submission for a submitted run) — there is no fallback; `ddp` = fully "
            "replicated. Or an explicit `PlacementTableConfig` table (`components: "
            "{optimizer_state, compute_weights, faithfulness_weights, "
            "faithfulness_deltas, operands}`, per-CI-weight-family "
            "`{optimizer_state, compute_weights, operands}` rows, `activations: "
            "{external, component}`, and the frozen-`target` role rows, each row a "
            "semantic-axis -> mesh-axes rule; list order is "
            "semantics). Same math under every value — layouts differ only by float "
            "reassociation (SPEC D4)."
        ),
    )
    remat_recon_forwards: bool = Field(
        default=False,
        description=(
            "JAX trainer memory/compute trade for the recon-loss masked forwards: the "
            "checkpoint policy of the target's per-block scan. True = recompute each "
            "block in the backward (`nothing_saveable`; deep targets need it to fit), "
            "False = store batch-scaled activation dots and re-forward nothing "
            "(`dots_saveable`; faster when memory allows). Compute substrate knob, no "
            "algorithm effect."
        ),
    )
    remat_ci_fn: bool = Field(
        default=False,
        description=(
            "JAX trainer memory/compute trade: rematerialize the CI-fn forward "
            "(recompute it in the backward instead of storing its activations). The "
            "CI-fn activations scale with batch, so this is the main lever for larger "
            "batch on big targets. Compute substrate knob, no algorithm effect."
        ),
    )
    compiler_options: Literal["tuned-v1", "bare"] | dict[str, bool | int | str] = Field(
        description=(
            "XLA compiler flags passed NATIVELY to every jit's `compiler_options` — no "
            "`XLA_FLAGS` env round-trip, and (unlike env) they ARE in the compile-cache key, "
            "so changing one actually recompiles. REQUIRED, no default and no merge: every "
            "run's flags trace to a visible authored token. `tuned-v1` = the frozen "
            "production set (`TUNED_V1_COMPILER_OPTIONS`); `bare` = {} (true XLA defaults — "
            "the debugging baseline); or an explicit dict, used VERBATIM as the complete "
            "flag set the run compiles with. Explicit dicts: full `xla_*` flag names, typed "
            "values (True/int/str, not 'true'); keys outside `xla_*` refuse. "
            "`xla_disable_hlo_passes: rematerialization` opts into the disable-XLA-remat "
            "win (validate save/resume first). `xla_gpu_memory_limit_slop_factor` is the "
            "memory-vs-wall dial of the latency-hiding scheduler (which deliberately "
            "spends memory for overlap): a percent scaling of the scheduler's memory "
            "budget, so it moves the COMPILED arena — the fit check's DEMANDED — not the "
            "runtime BFC pool. Memory-tight cells author it per cell in an explicit "
            "dict; measured per-cell tradeoffs live in PERF_NOTES.md. On CPU "
            "(toys/tests) the GPU flags are ignored."
        ),
    )

    @field_validator("compiler_options")
    @classmethod
    def _explicit_flags_are_xla_namespaced(
        cls, options: str | dict[str, bool | int | str]
    ) -> str | dict[str, bool | int | str]:
        if isinstance(options, dict):
            foreign = sorted(key for key in options if not key.startswith("xla_"))
            if foreign:
                raise ValueError(
                    f"compiler_options keys must be full `xla_*` flag names: {foreign}"
                )
        return options

    @property
    def resolved_compiler_options(self) -> dict[str, bool | int | str]:
        """The concrete flag map every jit receives — the presets resolve here, nowhere else."""
        match self.compiler_options:
            case "tuned-v1":
                return dict(TUNED_V1_COMPILER_OPTIONS)
            case "bare":
                return {}
            case explicit:
                return explicit

    compilation_cache_dir: Path = Field(
        description=(
            "Persistent XLA compilation-cache directory; `~` expands to the running user's "
            "home. REQUIRED, no default: where the multi-minute step compile is reused "
            "across runs/requeues is an authored decision. Author a PER-USER path — the "
            "seats set `~/.cache/param-decomp/xla` — never a shared artifact root: XLA's "
            "cache keeps a temporary autotune directory whose writer-created descendants "
            "need not stay group-writable, so a cache shared by unrelated Unix users "
            "fails their autotune lookups."
        ),
    )
    launch_env: LaunchEnv = Field(default_factory=LaunchEnv)
    """The pre-process env each rank runs with (XLA *client* / NCCL / glibc knobs — the env
    that must exist before backend init; NOT compiler flags, which go via
    `compiler_options`). Applied by the bootstrap in the process it starts, and rendered into
    the rank environment by whoever spawns the ranks; everything else about that environment
    is inherited from the caller."""
    profiling: ProfilingConfig = Field(default_factory=ProfilingDisabled)
    """The run's profiler, authored — the trainer receives it as typed data, never via env.
    `ad_hoc` is the in-process `jax.profiler` trace; `nsight_systems` attaches an external
    `nsys` (machine-specific executable resolution stays in the launcher; the profiler and
    its version remain pinned here)."""

    @property
    def world_size(self) -> int:
        return self.replicate * self.fsdp * self.tp

    @property
    def data_parallel_size(self) -> int:
        """Effective data-parallel degree after carving TP groups from the device world."""
        return self.replicate * self.fsdp
