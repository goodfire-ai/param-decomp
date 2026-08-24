"""LM entry for the AOT GPU-fit check (`core/tools/fit_check.py`): resolve a run YAML,
describe the GPU topology, compile the real jit_step AND every authored scalar-tier
jit_eval_step devicelessly, print one verdict per program. The eval boundary is its own
compiled program with its own arena — a run whose train step fits can still OOM at the
first eval pass, so the receipt covers both.

    python -m param_decomp.experiments.lm.fit_check <launch_config.yaml> \
        --data-root <root> --pool-gib 73.6 \
        [--target-config <gpu_target_config.pbtxt>] [--gpus-per-node 8] [--dump <dir>]

Runs on a CPU-only box with the CUDA jaxlib installed (`--extra cuda` env; no GPUs, no
driver needed): the topology is described, not attached. `--target-config` is the XLA
`GpuTargetConfigProto` text for the fleet's device (an `xla_dump_to` dump of any real run
writes one as `module_*.jit_step.gpu_target_config.pbtxt`); omit it only when real GPUs
are attached, in which case they are used directly. The frozen target's weights ARE read
(the loader is the trainer's own), so run it where the HF/pretrain cache lives."""

import argparse
from pathlib import Path
from typing import Any

import jax
import numpy as np
from jax import random
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.core import placement
from param_decomp.core.built_run import BuiltRun
from param_decomp.core.ci_fn import PlacedCIFn
from param_decomp.core.configs import CI_L0Config, PGDReconLossConfig
from param_decomp.core.model import BATCH_AXES, PlacedModel, Positioned
from param_decomp.core.sharding import HSDP_MESH_AXES
from param_decomp.core.tools.fit_check import (
    FitReport,
    abstract_placed_model,
    aot_fit_check,
    argument_audit,
    declared_run,
    fit_report_of_compiled,
)
from param_decomp.experiments.eval_config import EvalConfig
from param_decomp.experiments.lm.config import load_config
from param_decomp.experiments.lm.eval_config import CEandKLLossesConfig
from param_decomp.experiments.lm.load_run import load_target
from param_decomp.experiments.lm.scalar_eval_operations import scalar_step_for
from param_decomp.infra.dataset_store import read_dataset_meta


def _topology_devices(world_size: int, gpus_per_node: int, target_config: Path | None) -> list[Any]:
    """Attached GPUs when present, else compile-only devices from the described topology
    (`jax.Device` is not a static type in jax 0.10 — hence the loose element type)."""
    attached = [d for d in jax.devices() if d.platform == "gpu"] if target_config is None else []
    if attached:
        assert len(attached) == world_size, (len(attached), world_size)
        return attached
    assert target_config is not None, (
        "no attached GPUs: pass --target-config <gpu_target_config.pbtxt> for the "
        "deviceless compile"
    )
    from jax.experimental import topologies

    assert world_size % gpus_per_node == 0, (world_size, gpus_per_node)
    topo = topologies.get_topology_desc(
        platform="cuda",
        topology=f"1x{world_size // gpus_per_node}x{gpus_per_node}",
        target_config=target_config.read_text(),
    )
    assert len(topo.devices) == world_size, (len(topo.devices), world_size)
    return topo.devices


def scalar_eval_fit_reports(
    built: BuiltRun[Any, Any, Any],
    eval_cfg: EvalConfig,
    model: PlacedModel,
    mesh: Mesh,
    seq_len: int,
    *,
    compiler_options: dict[str, bool | int | str],
    pool_gib: float,
    dump_dir: Path | None,
) -> dict[str, FitReport]:
    """Compile every authored scalar-tier metric's jit_eval_step AOT, one `FitReport`
    per metric, keyed by its logged identity (`name or type`).

    Same assembly as the engine's eval boundary (`run._run_due_evaluation` +
    `scalar_step_for`): the declared-sharding decomposition as the live state, the CI fn
    paired with its resolved placement, the eval batch on the batch axes. The plot-tier
    metrics have no jitted step of this signature and are not compiled here."""
    jax.set_mesh(mesh)
    declared = declared_run(built, model, Positioned(n_positions=seq_len))
    placed_ci_fn = PlacedCIFn(
        fn=declared.state.decomposition.ci_fn, placement=declared.ci_placement
    )
    components = declared.state.decomposition.components
    tokens = jax.ShapeDtypeStruct(
        (eval_cfg.batch_size, seq_len),
        np.int32,
        sharding=NamedSharding(mesh, P(BATCH_AXES, None)),
    )
    key_struct = jax.eval_shape(lambda: random.PRNGKey(0))

    options: dict[str, bool | int | str] = dict(compiler_options)
    if dump_dir is not None:
        options["xla_dump_to"] = str(dump_dir)

    argument_audit((model, components, placed_ci_fn, tokens), pool_gib)
    reports: dict[str, FitReport] = {}
    for metric in eval_cfg.metrics:
        match metric:
            case CEandKLLossesConfig() | CI_L0Config() | PGDReconLossConfig():
                pass
            case _:  # a plot/diagnostic-tier metric: no scalar jit_eval_step to compile
                continue
        step_fn = scalar_step_for(metric, model, built.ci_fn.capture_keys, mesh, None)

        # The inner eqx filter_jit inlines under this outer trace; compiler options must
        # be restated here to reach the compile (the train check's nested-jit rule). The
        # wrapper's name makes the dump module `jit_eval_step`, the production name.
        def eval_step(
            m: PlacedModel, c: Any, f: PlacedCIFn, t: Any, k: Any, step_fn: Any = step_fn
        ) -> Any:
            return step_fn(m, c, f, t, k)

        # The metric's logged identity (`validate_eval_metrics`): only the
        # `LossMetricConfig` descendants carry a `name`.
        label = (
            (metric.name or metric.type) if isinstance(metric, PGDReconLossConfig) else metric.type
        )
        print(f"compiling jit_eval_step ({label}) AOT ...", flush=True)
        outer = jax.jit(eval_step, compiler_options=options)
        compiled = outer.lower(model, components, placed_ci_fn, tokens, key_struct).compile()
        assert label not in reports, f"duplicate scalar metric identity {label!r}"
        reports[label] = fit_report_of_compiled(compiled, pool_gib)
    return reports


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
        help="xla_dump_to dir for memreport-decodable buffer assignment",
    )
    args = parser.parse_args()

    # Any well-formed p-<8hex> id satisfies the run-identity gate; the fit check never
    # writes into the run dir it names.
    built, authored = load_config(args.config, "p-00000000", args.data_root)
    runtime = authored.runtime
    seq_len = read_dataset_meta(built.data.dir).seq_len

    devices = _topology_devices(runtime.world_size, args.gpus_per_node, args.target_config)
    mesh = Mesh(
        np.array(devices).reshape(runtime.replicate, runtime.fsdp, runtime.tp),
        axis_names=HSDP_MESH_AXES,
        axis_types=(AxisType.Explicit,) * 3,
    )

    print(f"loading frozen target weights ({type(built.target).__name__}) ...", flush=True)
    model = load_target(built.target, args.data_root)
    rules = placement.from_config(runtime.sharding, mesh, model.sites)
    abstract_model = abstract_placed_model(model, rules)
    del model

    batch = jax.ShapeDtypeStruct(
        (built.pd.batch_size, seq_len),
        np.int32,
        sharding=NamedSharding(mesh, P(BATCH_AXES, None)),
    )
    report = aot_fit_check(
        built,
        abstract_model,
        Positioned(n_positions=seq_len),
        batch,
        remat_recon_forwards=runtime.remat_recon_forwards,
        remat_ci_fn=runtime.remat_ci_fn,
        compiler_options=runtime.resolved_compiler_options,
        pool_gib=args.pool_gib,
        dump_dir=args.dump,
    )
    cell = (
        f"{runtime.world_size}dev ({runtime.replicate},{runtime.fsdp},{runtime.tp}) "
        f"{runtime.sharding if isinstance(runtime.sharding, str) else 'table'} "
        f"B{built.pd.batch_size} seq{seq_len}"
    )
    print(f"\n== fit check: jit_step {cell} ==\n{report.render()}", flush=True)

    if authored.eval is not None:
        eval_reports = scalar_eval_fit_reports(
            built,
            authored.eval,
            abstract_model,
            mesh,
            seq_len,
            compiler_options=runtime.resolved_compiler_options,
            pool_gib=args.pool_gib,
            dump_dir=args.dump,
        )
        for label, eval_report in eval_reports.items():
            print(
                f"\n== fit check: jit_eval_step [{label}] eval_B{authored.eval.batch_size} "
                f"seq{seq_len} ==\n{eval_report.render()}",
                flush=True,
            )


if __name__ == "__main__":
    main()
