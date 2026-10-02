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
    model_validator,
)

from param_decomp.core.base_config import BaseConfig
from param_decomp.core.configs import (
    HsdpMeshShape,
    MeshShape,
    PlacementSpec,
    ResidentMeshShape,
    SequenceSharding,
)
from param_decomp.core.world_size import WorldSize, world_size_from_device_count

TUNED_V2_COMPILER_OPTIONS: Mapping[str, bool | int | str] = MappingProxyType(
    {
        "xla_gpu_enable_latency_hiding_scheduler": True,
        "xla_gpu_enable_triton_gemm": False,
        "xla_gpu_enable_command_buffer": "",
        "xla_gpu_enable_highest_priority_async_stream": True,
        "xla_gpu_all_reduce_combine_threshold_bytes": 1073741824,
        "xla_gpu_all_gather_combine_threshold_bytes": 1073741824,
        "xla_gpu_reduce_scatter_combine_threshold_bytes": 134217728,
        "xla_gpu_enable_while_loop_double_buffering": False,
        "xla_gpu_enable_all_gather_combine_by_dim": False,
        "xla_gpu_enable_reduce_scatter_combine_by_dim": False,
    }
)
"""What `compiler_options: tuned-v2` resolves to — the ONE copy of the tuned set, frozen.
The core of MaxText's H100 recipe: latency-hiding scheduler + 1 GiB collective-combine
thresholds; `xla_gpu_enable_command_buffer: ''` disables CUDA-graph capture, a
correctness guard. The recipe's pipelined-collective arm is not expressible on this
substrate — jaxlib 0.11 removed the three
`xla_gpu_enable_pipelined_{all_gather,reduce_scatter,all_reduce}` compile options (they
survive only as `XLA_FLAGS` env flags), which is what retired the tuned-v1 preset family.
While-loop double-buffering is OFF: it keeps O(1) extra copies of a while loop's operand
tuple — pennies when the xs are per-layer weight shards, fatal when they carry
whole-depth resident stacks — so the one set is safe for every placement, resident
included. A change to the tuned set is a NEW preset name (`tuned-v3`), never an edit
here — pinned configs authoring `tuned-v2` must keep meaning these exact flags."""

TUNED_V2_AUTOTUNE1_COMPILER_OPTIONS: Mapping[str, bool | int | str] = MappingProxyType(
    {**TUNED_V2_COMPILER_OPTIONS, "xla_gpu_autotune_level": 1}
)
"""What `compiler_options: tuned-v2-autotune1` resolves to — tuned-v2 with GEMM/conv
autotuning dialed to level 1: the fast-iteration preset, trading
kernel-pick quality for a shorter first compile. Autotune picks can shift fusion choices
and therefore the arena, so a fit verdict or step time measured under this preset does
not transfer to `tuned-v2` (and vice versa). Equally frozen: a change is a new preset
name."""


def _merged_xla_flags(config_flags: str, inherited: str | None) -> str:
    """Compose the config's `XLA_FLAGS` with flags the process environment already
    carries, keyed by flag name (the token before `=`): disjoint flags compose (the
    config's first, the environment's extras appended), an identical duplicate dedupes,
    and the SAME flag with a DIFFERENT value refuses — a reviewed config value
    conflicting with ambient env is a confused state, and env flags are not in the
    compile-cache key, so resolving it silently in either direction is refused."""
    merged = {token.split("=", 1)[0]: token for token in config_flags.split()}
    for token in (inherited or "").split():
        key = token.split("=", 1)[0]
        if key not in merged:
            merged[key] = token
            continue
        assert merged[key] == token, (
            f"XLA_FLAGS conflict on {key}: the config's launch_env.xla_flags carries "
            f"{merged[key]!r}, the process environment carries {token!r} — align the "
            "config or the environment; a silent resolution is refused"
        )
    return " ".join(merged.values())


TYPED_ENV_VARS: Mapping[str, str] = MappingProxyType(
    {
        "NCCL_DEBUG": "nccl_debug",
        "MALLOC_ARENA_MAX": "malloc_arena_max",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "xla_python_client_mem_fraction",
        "XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB": "xla_pjrt_gpu_host_memory_limit_gb",
        "XLA_FLAGS": "xla_flags",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "xla_python_client_allocator",
    }
)
"""Every environment variable a typed `LaunchEnv` field renders, keyed to its field: the
closed set the free-form `env` block may NOT name (`_env_block_is_disjoint_from_typed_fields`).
`as_env` renders exactly these keys (plus nothing), so ownership and rendering cannot drift."""


class LaunchEnv(BaseConfig):
    """The process-environment surface a rank runs with — the XLA *client* knobs (mem
    fraction / allocator / host-memory limit), NCCL/glibc tuning, and a free-form env block
    for variables WITHOUT a typed field — lifted into the run config so a run's
    `launch_config.yaml` fully captures its environment (tracking + repro), and A/B-ing a
    knob is a config edit, not a process-wrapper edit.

    XLA *compiler* flags are NOT here — they go through `RuntimeConfig.compiler_options`
    (passed natively to each jit, no env round-trip; see that field). This class is only the
    env that must exist before the process starts (read at backend/NCCL init).

    The pre-JAX bootstrap (`run.py`) exports it before importing JAX; whoever spawns the
    ranks renders the same map into their environment. `LD_LIBRARY_PATH` is NOT here (it is
    machine-specific — resolved against the local CUDA install by whoever starts the
    process — not a tracked decision). These defaults are the single source of truth: a
    rank spawner renders them; it does not carry its own set.
    """

    xla_python_client_mem_fraction: PositiveFloat = 0.92
    """`XLA_PYTHON_CLIENT_MEM_FRACTION` — the BFC pool cap as a fraction of HBM."""
    xla_flags: str = "--xla_gpu_nccl_termination_timeout_seconds=600"
    """`XLA_FLAGS` — XLA *runtime* env knobs; compiler flags go through
    `RuntimeConfig.compiler_options` instead. The default bounds the collective-clique
    acquisition rendezvous, which XLA otherwise waits on FOREVER (upstream terminate
    default -1): a device thread that dies before joining — e.g. a per-device OOM at
    executable launch — parks every sibling thread at 0% GPU with no error, wedging the
    whole world. Local device threads dispatch together, so 600s outlasts any legitimate
    join skew; on expiry XLA dumps all thread stacks and aborts, making the true failure
    loud and attributed. The trainer appends its HLO-dump flags to this value. Flags the
    process environment already carries compose with these rather than being replaced
    (`as_env` + `_merged_xla_flags`: disjoint flags compose, an identical duplicate
    dedupes, a conflicting value refuses)."""
    xla_python_client_allocator: str | None = None
    """`XLA_PYTHON_CLIENT_ALLOCATOR` — e.g. `platform` for the on-demand cudaMalloc allocator
    (avoids BFC fragmentation OOMs near the HBM cap, at some per-alloc cost). `None` leaves
    the XLA default (BFC)."""
    xla_pjrt_gpu_host_memory_limit_gb: PositiveInt = 1024
    """`XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB` — cap on XLA's pinned host-staging pool
    (allocated on demand)."""
    nccl_debug: str = "WARN"
    """`NCCL_DEBUG` — overrides an ambient INFO + SUBSYS=ALL default, which logs
    every collective and bloats a run's logs to tens of GB."""
    malloc_arena_max: PositiveInt = 2
    """`MALLOC_ARENA_MAX` — caps glibc malloc arenas to bound host RSS under many threads."""
    env: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Extra exports for one-off variables WITHOUT a typed field, rendered alongside "
            "the typed knobs. A key a typed field owns (`TYPED_ENV_VARS`, `XLA_FLAGS` "
            "included) refuses at config load naming that field: the typed spelling carries "
            "the field's defaults and composition rules, which a raw override would silently "
            "discard."
        ),
    )

    @field_validator("env")
    @classmethod
    def _env_block_is_disjoint_from_typed_fields(cls, env: dict[str, str]) -> dict[str, str]:
        owned = sorted(key for key in env if key in TYPED_ENV_VARS)
        if owned:
            remedies = ", ".join(f"{key} -> launch_env.{TYPED_ENV_VARS[key]}" for key in owned)
            raise ValueError(
                f"launch_env.env names variables typed fields own: {owned}. Author the "
                f"typed field instead ({remedies}); an env-block override would silently "
                "replace the field's defaults and composition rules"
            )
        return env

    def as_env(self, inherited_xla_flags: str | None) -> dict[str, str]:
        """Render the ordered `{VAR: value}` map a rank's environment must carry (sans
        `LD_LIBRARY_PATH`, which is machine-specific). Only the env that must exist before
        backend/NCCL init — XLA *compiler* flags are passed natively via
        `RuntimeConfig.compiler_options`, not here. The typed knobs render the
        `TYPED_ENV_VARS` keys; the free-form `env` block is disjoint from them by
        validation, so no key is written twice.

        `inherited_xla_flags` is the `XLA_FLAGS` the environment already carries: a
        bootstrap applying this map in-process passes `os.environ.get("XLA_FLAGS")` so the
        caller's exports compose with the config's flags — additively, with a
        conflicting value refused (`_merged_xla_flags`); a rank spawner rendering the map
        into a fresh process passes `None` — the parent machine's env is not the
        rank's."""
        rendered: dict[str, str] = {
            "NCCL_DEBUG": self.nccl_debug,
            "MALLOC_ARENA_MAX": str(self.malloc_arena_max),
            "XLA_PYTHON_CLIENT_MEM_FRACTION": str(self.xla_python_client_mem_fraction),
            "XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB": str(self.xla_pjrt_gpu_host_memory_limit_gb),
            "XLA_FLAGS": _merged_xla_flags(self.xla_flags, inherited_xla_flags),
        }
        if self.xla_python_client_allocator is not None:
            rendered["XLA_PYTHON_CLIENT_ALLOCATOR"] = self.xla_python_client_allocator
        assert rendered.keys() <= TYPED_ENV_VARS.keys(), rendered.keys() - TYPED_ENV_VARS.keys()
        return rendered | self.env


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

    mesh: MeshShape
    """The logical mesh: `{replicate: R, fsdp: F, tp: T}` or, for the
    `*-replicated-resident` placements (which have NO fsdp axis — the working copy is
    resident whole), `{data: D, tp: T}`. The union discriminates structurally (disjoint
    required keys, extra keys refuse). Which shape a `sharding` runs on is the placement
    table's own knowledge — the mesh axes its rows name — checked once at placement
    construction (`placement.from_config`, at config build) for presets and explicit
    tables alike."""

    sharding: PlacementSpec = Field(
        description=(
            "Placement policy for the trainable state (placement.py). REQUIRED, no "
            "default — a layout this consequential is written down per config. Presets: "
            "`zero1` = intra-matrix ZeRO-1 over the full data mesh (no row shards the "
            "component stack axis, so every semantic group is placeable; ~equivalent "
            "comms to `owner` under elementwise optimizers); `owner` = whole-matrix "
            "ownership (stack ÷replicate, d ÷fsdp, C ÷tp) — the muon-motivated layout "
            "(Newton-Schulz stays node-local); a semantic group whose stack does not "
            "tile ÷replicate refuses at config build (placement.from_config, "
            "during validation before execution) — there is no fallback; "
            "`zero1-replicated-resident` = `zero1` masters with the bf16 working copy "
            "RESIDENT whole (÷tp only — classic ZeRO-1: the resident rows equal the "
            "operand rows, so the once-per-step entry gather is the only weight "
            "collective and no while body gathers weights); `owner-replicated-resident` "
            "= `owner` masters (stack-cut — the faithfulness transition is the "
            "identity) with the same replicated resident; strict like `owner` (stacks "
            "must tile ÷data); `zero1-replicated-resident-moe` / "
            "`owner-replicated-resident-moe` = the resident twins plus the "
            "expert-blocked component rows a MoE family needs (V/U expert blocks "
            "co-located with their frozen experts on tp; zero1 masters `C_block` "
            "÷data place any stack length, owner masters `{stack: data, expert: tp}` "
            "rest whole blocks per device — the stacked-muon pairing, faithfulness "
            "fully rank-local; stacks must tile ÷data) — binding only site sets that "
            "hold expert-blocked groups; `zero1-replicated-resident-moe-replicated-ns` = "
            "the zero1 twin with every Newton-Schulz staging row replicated, so stacks too "
            "short to tile ÷data still stage under muon. "
            "The `*-replicated-resident*` presets run on the "
            "two-axis (data, tp) mesh — residency leaves fsdp nothing to shard, so the "
            "axis does not exist and the run spells `mesh: {data: D, tp: T}`; "
            "`ddp` = fully replicated. Or an explicit `PlacementTableConfig` table (`components: "
            "{optimizer_state, compute_weights, faithfulness_weights, "
            "faithfulness_deltas, operands, ns_compute}`, `activations: "
            "{external, component}`, and the frozen-`target` role rows, each row a "
            "semantic-axis -> mesh-axes rule, list order being semantics; plus `ci_fn: "
            "<preset>`, the preset whose rows the CI architecture takes — CI rows are "
            "architecture-owned, never authored). Same math under every value — layouts "
            "differ only by float reassociation."
        ),
    )
    sequence_sharding: SequenceSharding = Field(
        default="replicate",
        description=(
            "How masked forwards place the between-blocks residual over tp "
            "(placement.py, `activations/masked_external`). `replicate` = the standing "
            "spelling. `sequence_parallel` shards the position axis over tp between the "
            "blocks of masked forwards; block interiors run at full width, so each block "
            "boundary's tp-axis activation reduction becomes a reduce-scatter + "
            "all-gather pair and the saved between-blocks residuals rest at 1/tp. Clean "
            "forwards, CI-fn taps and the output edge keep the replicated residual. "
            "Sequence length must tile tp; only targets implementing it accept it "
            "(qwen36_moe). Same math either way — a resharding, so layouts differ only "
            "by float reassociation."
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
            "Checkpoint policy between transformer CI chunks: True recomputes chunk "
            "activations (`nothing_saveable`); False permits saving activation dots "
            "(`dots_saveable`). Transformer blocks within each chunk are always "
            "checkpointed individually to bound backward temporaries. Compute "
            "substrate knob, no algorithm effect."
        ),
    )
    compiler_options: (
        Literal["tuned-v2", "tuned-v2-autotune1", "bare"] | dict[str, bool | int | str]
    ) = Field(
        description=(
            "XLA compiler flags passed NATIVELY to every jit's `compiler_options` — no "
            "`XLA_FLAGS` env round-trip, and (unlike env) they ARE in the compile-cache key, "
            "so changing one actually recompiles. REQUIRED, no default and no merge: every "
            "run's flags trace to a visible authored token. `tuned-v2` = the frozen "
            "production set (`TUNED_V2_COMPILER_OPTIONS`), safe for every placement, "
            "resident included; `tuned-v2-autotune1` = tuned-v2 plus "
            "`xla_gpu_autotune_level: 1` — the fast-iteration set (shorter first "
            "compile; kernel picks and arena may differ from tuned-v2); `bare` = {} (true XLA "
            "defaults — the debugging baseline); or an explicit dict, used VERBATIM as the complete "
            "flag set the run compiles with. Explicit dicts: full `xla_*` flag names, typed "
            "values (True/int/str, not 'true'); keys outside `xla_*` refuse. "
            "`xla_disable_hlo_passes: rematerialization` opts into the disable-XLA-remat "
            "win (validate save/resume first). `xla_gpu_memory_limit_slop_factor` is the "
            "memory-vs-wall dial of the latency-hiding scheduler (which deliberately "
            "spends memory for overlap): a percent scaling of the scheduler's memory "
            "budget, so it moves the COMPILED arena — the fit check's DEMANDED — not the "
            "runtime BFC pool. Memory-tight cells author it per cell in an explicit "
            "dict. On CPU (toys/tests) the GPU flags are ignored."
        ),
    )

    @field_validator("compiler_options", mode="before")
    @classmethod
    def _dead_preset_spellings_refuse_with_the_successor(cls, options: object) -> object:
        """jaxlib 0.11 removed the pipelined-collective trio the tuned-v1 presets froze,
        so those tokens can never again mean their exact flag sets; each refuses naming
        its successor rather than silently resolving to different flags."""
        successor_of_retired = {
            "tuned-v1": "tuned-v2",
            "tuned-v1-resident": "tuned-v2",
            "tuned-v1-resident-autotune1": "tuned-v2-autotune1",
        }
        if isinstance(options, str) and options in successor_of_retired:
            raise ValueError(
                f"compiler_options {options!r} is retired: jaxlib 0.11 removed the "
                f"pipelined-collective compile options it froze. Author "
                f"{successor_of_retired[options]!r}"
            )
        return options

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

    @model_validator(mode="before")
    @classmethod
    def _flat_mesh_keys_refuse_with_the_nested_spelling(cls, data: object) -> object:
        """`extra="forbid"` already rejects the retired flat axis keys; this exists only
        to say the nested spelling in the refusal."""
        if isinstance(data, dict):
            flat = [key for key in ("replicate", "fsdp", "data", "tp") if key in data]
            if flat:
                raise ValueError(
                    f"mesh axes are nested under `runtime.mesh:` — spell "
                    f"`mesh: {{replicate: R, fsdp: F, tp: T}}` (or, for a "
                    f"`*-replicated-resident` run, `mesh: {{data: D, tp: T}}`); "
                    f"got flat keys {flat}"
                )
        return data

    @property
    def resolved_compiler_options(self) -> dict[str, bool | int | str]:
        """The concrete flag map every jit receives — the presets resolve here, nowhere else."""
        match self.compiler_options:
            case "tuned-v2":
                return dict(TUNED_V2_COMPILER_OPTIONS)
            case "tuned-v2-autotune1":
                return dict(TUNED_V2_AUTOTUNE1_COMPILER_OPTIONS)
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
    `nsys` (machine-specific executable resolution stays outside the library; the profiler
    and its version remain pinned here)."""

    @field_validator("mesh")
    @classmethod
    def _mesh_has_supported_world_size(cls, mesh: MeshShape) -> MeshShape:
        world_size_from_device_count(mesh.world_size)
        return mesh

    @property
    def world_size(self) -> WorldSize:
        return world_size_from_device_count(self.mesh.world_size)

    @property
    def data_parallel_size(self) -> int:
        """Distinct batch shards after carving TP groups from the device world."""
        match self.mesh:
            case HsdpMeshShape(replicate=replicate, fsdp=fsdp):
                return replicate * fsdp
            case ResidentMeshShape(data=data):
                return data
