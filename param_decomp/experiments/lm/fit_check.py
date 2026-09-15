"""LM entry for the AOT GPU-fit check (`core/tools/fit_check.py`): resolve a run YAML,
describe the GPU topology, compile the real jit_step AND every authored scalar-tier
jit_eval_step devicelessly, print one verdict per program. The eval boundary is its own
compiled program with its own arena — a run whose train step fits can still OOM at the
first eval pass, so the receipt covers both.

    JAX_PLATFORMS=cpu python -m param_decomp.experiments.lm.fit_check <launch_config.yaml> \
        --data-root <root> --pool-gib 73.6 \
        [--target-config <gpu_target_config.pbtxt>] [--gpus-per-node 8] \
        [--dump <dir> | --ledger <dir>] [--graph-html]

`--ledger <dir>` is the memory-ledger mode: each program dumps (with HLO protos) to
`<dir>/<program>/dump/` and gets `<dir>/<program>/{ledger.csv,meta.json,peak_report.txt}`
— every assigned buffer with size, arena offset, live range, and jax provenance (op path
+ source file:line); see `core/tools/memory_ledger.py`. Compare two ledger dirs with
`python -m param_decomp.core.tools.ledger_diff`. The kept `dump/*after_optimizations.hlo.pb`
is also the profiler memory viewer's input (module + buffer assignment).

`--graph-html` (composes with either dump mode) adds XLA's HTML renders per program:
`*.top_level.html` (the entry computation's graph) and `*_fusion.html` neighborhoods
(interactive SVG; graphviz-wasm + svg-pan-zoom load from gstatic.com, so viewing needs
network). XLA's renderer caps graphs at a few thousand nodes and falls back to
neighborhood views around each fusion — at large model scale expect MANY per-fusion files and
a multi-GB dump dir, and prefer opening one fusion's file over the top-level render. For
whole-module browsing at scale, Google's model-explorer (`pip install ai-edge-model-explorer`,
`model-explorer <dump>/module_*_after_optimizations.txt`) handles large HLO text better.

Runs on a CPU-only box with the CUDA jaxlib installed (`--extra cuda` env; no GPUs, no
driver needed): the topology is described, not attached. `JAX_PLATFORMS=cpu` is load-
bearing on such a box — the cuda plugin's import-time CUDA version check fails without
the driver stack and then nothing registers the compile-only cuda topology factory;
with platforms pinned away from cuda the plugin skips the check and registers as a
pure compile target. `--target-config` is the XLA `GpuTargetConfigProto` text for the
target device (an `xla_dump_to` dump of any real run
writes one as `module_*.jit_step.gpu_target_config.pbtxt`); omit it only when real GPUs
are attached, in which case they are used directly. The frozen target's weights ARE read
(the loader is the trainer's own), so run it where the HF/pretrain cache lives, with
host RAM for the full checkpoint plus assembly copies — budget ~3-4x the checkpoint's
bytes (a 35B bf16 target wanted ~250-300G)."""

import argparse
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import numpy as np
import yaml
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.core import placement
from param_decomp.core.built_run import BuiltRun
from param_decomp.core.ci_fn import PlacedCIFn
from param_decomp.core.configs import (
    CI_L0Config,
    CIHistogramsConfig,
    CIMeanPerComponentConfig,
    ComponentActivationDensityConfig,
    IdentityCIErrorConfig,
    PermutedCIPlotsConfig,
    PGDReconLossConfig,
    UVPlotsConfig,
)
from param_decomp.core.hardware_utilization import checked_device_kind
from param_decomp.core.model import PlacedModel, Positioned
from param_decomp.core.placement import batch_axes
from param_decomp.core.sharding import mesh_over_devices
from param_decomp.core.tools.fit_check import (
    DumpConfig,
    FitReport,
    abstract_placed_model,
    aot_fit_check,
    aot_targeted_fit_check,
    argument_audit,
    declared_run,
    fit_report_of_compiled,
)
from param_decomp.core.tools.memory_ledger import emit_program_ledger, render_peak_report
from param_decomp.experiments.eval_config import EvalConfig
from param_decomp.experiments.lm.config import (
    LMTargetedExperimentConfig,
    build_targeted_experiment_config,
    load_config,
)
from param_decomp.experiments.lm.eval_config import (
    ArithmeticCIGridConfig,
    CEandKLLossesConfig,
    CIMaskedAttnPatternsReconLossConfig,
    StochasticAttnPatternsReconLossConfig,
    WellTemperednessConfig,
)
from param_decomp.experiments.lm.load_run import load_target
from param_decomp.experiments.lm.scalar_eval_operations import scalar_step_for
from param_decomp.experiments.lm.targeted_data import build_prompt_pool
from param_decomp.experiments.lm.training_targeted import (
    load_pool_tokenizer,
    pool_tokenizer_source,
)
from param_decomp.infra.dataset_store import read_dataset_meta
from param_decomp.targets.lm_output import LMOutput


def _topology_devices(world_size: int, gpus_per_node: int, target_config: Path | None) -> list[Any]:
    """Attached GPUs when present, else compile-only devices from the described topology
    (`jax.Device` is not a static type in jax 0.10 — hence the loose element type). Either
    way the device set's kind is checked against the trainer's enumeration
    (`hardware_utilization.DeviceKind`) — the pre-allocation twin of the launch boundary's
    check, so an unlisted kind refuses in the receipt, not on the allocated nodes."""
    attached = [d for d in jax.devices() if d.platform == "gpu"] if target_config is None else []
    if attached:
        assert len(attached) == world_size, (len(attached), world_size)
        checked_device_kind(attached)
        return attached
    assert target_config is not None, (
        "no attached GPUs: pass --target-config <gpu_target_config.pbtxt> for the "
        "deviceless compile"
    )
    from jax.experimental import topologies

    assert world_size % gpus_per_node == 0, (world_size, gpus_per_node)
    try:
        topo = topologies.get_topology_desc(
            platform="cuda",
            topology=f"1x{world_size // gpus_per_node}x{gpus_per_node}",
            target_config=target_config.read_text(),
        )
    except NotImplementedError as e:
        # "topology not implemented for cuda" = the cuda PJRT plugin never registered
        # its topology factory. On a GPU-less box the plugin's import-time CUDA version
        # check fails (cuPTI needs the driver stack) and registration is abandoned —
        # unless JAX_PLATFORMS names only non-cuda platforms, in which case the plugin
        # skips the check and registers as a compile-only target.
        raise RuntimeError(
            "the cuda topology factory is not registered — on a GPU-less box start the "
            "process with JAX_PLATFORMS=cpu so the cuda plugin registers as a "
            "compile-only target (JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1 is the blunter "
            "override; a node with attached GPUs and the driver present needs neither)"
        ) from e
    assert len(topo.devices) == world_size, (len(topo.devices), world_size)
    checked_device_kind(topo.devices)
    return topo.devices


def scalar_eval_fit_reports(
    built: BuiltRun[Any, Any, Any],
    eval_cfg: EvalConfig,
    model: PlacedModel[LMOutput],
    mesh: Mesh,
    seq_len: int,
    *,
    compiler_options: dict[str, bool | int | str],
    pool_gib: float,
    dump_for: Callable[[str], DumpConfig | None],
) -> dict[str, FitReport]:
    """Compile every authored scalar-tier metric's jit_eval_step AOT, one `FitReport`
    per metric, keyed by its logged identity (`name or type`).

    Same assembly as the engine's eval boundary (`run._run_due_evaluation` +
    `scalar_step_for`): the declared-sharding decomposition as the live state, the CI fn
    paired with its resolved placement, the eval batch on the batch axes. The plot-tier
    metrics have no jitted step of this signature and are not compiled here."""
    jax.set_mesh(mesh)
    declared = declared_run(built.pd, built.ci_fn, model, Positioned(n_positions=seq_len))
    placed_ci_fn = PlacedCIFn(
        fn=declared.state.decomposition.ci_fn, placement=declared.ci_placement
    )
    components = declared.state.decomposition.components
    tokens = jax.ShapeDtypeStruct(
        (eval_cfg.batch_size, seq_len),
        np.int32,
        sharding=NamedSharding(mesh, P(batch_axes(mesh), None)),
    )
    key_struct = jax.eval_shape(lambda: random.PRNGKey(0))

    argument_audit((model, components, placed_ci_fn, tokens), pool_gib)
    reports: dict[str, FitReport] = {}
    for metric in eval_cfg.metrics:
        match metric:
            case CEandKLLossesConfig() | CI_L0Config() | PGDReconLossConfig():
                pass
            case (
                ArithmeticCIGridConfig()
                | CIHistogramsConfig()
                | CIMaskedAttnPatternsReconLossConfig()
                | CIMeanPerComponentConfig()
                | ComponentActivationDensityConfig()
                | IdentityCIErrorConfig()
                | PermutedCIPlotsConfig()
                | StochasticAttnPatternsReconLossConfig()
                | UVPlotsConfig()
                | WellTemperednessConfig()
            ):
                # a plot/diagnostic-tier metric: no scalar jit_eval_step to compile
                continue
        step_fn = scalar_step_for(metric, model, built.ci_fn.capture_keys, mesh, None)

        # The inner eqx filter_jit inlines under this outer trace; compiler options must
        # be restated here to reach the compile (the train check's nested-jit rule). The
        # wrapper's name makes the dump module `jit_eval_step`, the production name.
        def eval_step(
            m: PlacedModel[LMOutput],
            c: Any,
            f: PlacedCIFn,
            t: Any,
            k: Any,
            step_fn: Any = step_fn,
        ) -> Any:
            return step_fn(m, c, f, t, k)

        # The metric's logged identity (`validate_eval_metrics`): only the
        # `LossMetricConfig` descendants carry a `name`.
        label = (
            (metric.name or metric.type) if isinstance(metric, PGDReconLossConfig) else metric.type
        )
        options: dict[str, bool | int | str] = dict(compiler_options)
        dump = dump_for(label)
        if dump is not None:
            options |= dump.compiler_options()
        print(f"compiling jit_eval_step ({label}) AOT ...", flush=True)
        outer = jax.jit(eval_step, compiler_options=options)
        compiled = outer.lower(model, components, placed_ci_fn, tokens, key_struct).compile()
        assert label not in reports, f"duplicate scalar metric identity {label!r}"
        reports[label] = fit_report_of_compiled(compiled, pool_gib)
    return reports


def _eval_program_slug(label: str) -> str:
    return "jit_eval_step__" + re.sub(r"[^\w.-]+", "_", label)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--pool-gib",
        type=float,
        required=True,
        help="usable per-device HBM pool to judge against (GiB)",
    )
    parser.add_argument("--target-config", type=Path, default=None)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument(
        "--dump",
        type=Path,
        default=None,
        help="one shared xla_dump_to dir (text artifacts, memreport-decodable)",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=None,
        help="memory-ledger mode: per-program proto dump + ledger.csv / meta.json /"
        " peak_report.txt (module docstring)",
    )
    parser.add_argument(
        "--graph-html",
        action="store_true",
        help="add XLA's HTML graph renders to each program's dump (module docstring)",
    )
    args = parser.parse_args()
    assert args.dump is None or args.ledger is None, (
        "--dump and --ledger both claim xla_dump_to — pass one"
    )

    def dump_config_for(program: str) -> DumpConfig | None:
        if args.ledger is not None:
            return DumpConfig(
                dir=args.ledger / program / "dump", hlo_protos=True, graph_html=args.graph_html
            )
        if args.dump is not None:
            return DumpConfig(dir=args.dump, hlo_protos=False, graph_html=args.graph_html)
        return None

    def emit_ledger(program: str) -> None:
        if args.ledger is None:
            return
        ledger = emit_program_ledger(args.ledger / program / "dump", args.ledger / program)
        print(
            f"\n-- ledger: {args.ledger / program}\n{render_peak_report(ledger, top=20)}",
            flush=True,
        )

    # Any well-formed p-<8hex> id satisfies the run-identity gate; the fit check never
    # writes into the run dir it names. A targeted (tPD) seat is its own schema,
    # dispatched on the presence of `nontarget:` exactly as the parse and trace gates do.
    raw = yaml.safe_load(args.config.read_text())
    targeted: LMTargetedExperimentConfig | None = None
    if "nontarget" in raw:
        targeted = LMTargetedExperimentConfig.model_validate(raw)
        built = build_targeted_experiment_config(targeted, "p-00000000", args.data_root)
        authored: Any = targeted
    else:
        built, authored = load_config(args.config, "p-00000000", args.data_root)
    runtime = authored.runtime
    train_meta = read_dataset_meta(built.data.dir)
    seq_len = train_meta.seq_len

    devices = _topology_devices(runtime.world_size, args.gpus_per_node, args.target_config)
    mesh = mesh_over_devices(runtime.mesh, np.array(devices))

    print(f"loading frozen target weights ({type(built.target).__name__}) ...", flush=True)
    model = load_target(built.target, args.data_root)
    rules = placement.from_config(
        runtime.sharding, mesh, model.sites, sequence_sharding=runtime.sequence_sharding
    )
    abstract_model = abstract_placed_model(model, rules)
    del model

    def token_batch(batch_size: int, extent: int) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(
            (batch_size, extent),
            np.int32,
            sharding=NamedSharding(mesh, P(batch_axes(mesh), None)),
        )

    # Each compile entry runs under the target device's identity: the routed-experts
    # interpret gate and tokamax's triton arm key on the default DEVICE
    # (`jax.extend.backend.get_default_device()`), so this context makes the deviceless
    # compile lower the real GPU kernels rather than the interpret/densifying
    # fallbacks. Scoped to lower/compile only — eager array creation on a compile-only
    # device must keep failing loudly.
    if targeted is not None:
        # The receipt prices the TARGET stream at the pool's true geometry (unpadded
        # at its own prompt length), so the pool tokenizer IS loaded here — unlike the
        # trace gate, whose extent is sharding-irrelevant.
        tokenizer = load_pool_tokenizer(
            pool_tokenizer_source(built.target, train_meta.tokenizer_name)
        )
        pool = build_prompt_pool(targeted.prompts, tokenizer)
        n_prompts, prompt_len = pool.tokens.shape
        with jax.default_device(devices[0]):
            report = aot_targeted_fit_check(
                built,
                targeted.nontarget,
                abstract_model,
                Positioned(n_positions=prompt_len),
                token_batch(built.pd.batch_size, prompt_len),
                token_batch(targeted.nontarget.batch_size, seq_len),
                remat_recon_forwards=runtime.remat_recon_forwards,
                remat_ci_fn=runtime.remat_ci_fn,
                compiler_options=runtime.resolved_compiler_options,
                pool_gib=args.pool_gib,
                dump=dump_config_for("jit_step"),
            )
        cell = (
            f"{runtime.world_size}dev ({runtime.mesh}) "
            f"{runtime.sharding if isinstance(runtime.sharding, str) else 'table'} "
            f"targetB{built.pd.batch_size}x{prompt_len} (pool {n_prompts}) "
            f"nontargetB{targeted.nontarget.batch_size} seq{seq_len}"
        )
    else:
        with jax.default_device(devices[0]):
            report = aot_fit_check(
                built,
                abstract_model,
                Positioned(n_positions=seq_len),
                token_batch(built.pd.batch_size, seq_len),
                remat_recon_forwards=runtime.remat_recon_forwards,
                remat_ci_fn=runtime.remat_ci_fn,
                compiler_options=runtime.resolved_compiler_options,
                pool_gib=args.pool_gib,
                dump=dump_config_for("jit_step"),
            )
        cell = (
            f"{runtime.world_size}dev ({runtime.mesh}) "
            f"{runtime.sharding if isinstance(runtime.sharding, str) else 'table'} "
            f"B{built.pd.batch_size} seq{seq_len}"
        )
    print(f"\n== fit check: jit_step {cell} ==\n{report.render()}", flush=True)
    emit_ledger("jit_step")

    if authored.eval is not None:
        with jax.default_device(devices[0]):
            eval_reports = scalar_eval_fit_reports(
                built,
                authored.eval,
                abstract_model,
                mesh,
                seq_len,
                compiler_options=runtime.resolved_compiler_options,
                pool_gib=args.pool_gib,
                dump_for=lambda label: dump_config_for(_eval_program_slug(label)),
            )
        for label, eval_report in eval_reports.items():
            print(
                f"\n== fit check: jit_eval_step [{label}] eval_B{authored.eval.batch_size} "
                f"seq{seq_len} ==\n{eval_report.render()}",
                flush=True,
            )
            emit_ledger(_eval_program_slug(label))


if __name__ == "__main__":
    main()
