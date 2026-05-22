"""Per-point assembly: render run.yaml + topology.yaml + sbatch, then sbatch it.

``submit_point`` is the unit of work — given a :class:`SweepPoint` and a
:class:`SweepConfig`, it computes the world topology, validates the rendered
run yaml against ``LMRunConfig``, writes the three files into
``{GEN_ROOT}/<name>/``, and shells out to ``sbatch``. Pure-orchestration —
no rendering logic lives here.
"""

import subprocess
from pathlib import Path

from param_decomp.scripts.two_pool_benchmark.submit_sweep.paths import GEN_ROOT, SLURM_LOGS
from param_decomp.scripts.two_pool_benchmark.submit_sweep.render_run import (
    render_run_yaml,
    validate_run_yaml,
)
from param_decomp.scripts.two_pool_benchmark.submit_sweep.render_sbatch import render_sbatch
from param_decomp.scripts.two_pool_benchmark.submit_sweep.schema import SweepConfig, SweepPoint
from param_decomp.scripts.two_pool_benchmark.submit_sweep.topology import (
    render_topology,
    topology_label,
)


def point_name(prefix: str, point: SweepPoint) -> str:
    ci = point.ci.resolved()
    return (
        f"{prefix}-b{point.batch}-s{point.seq}-"
        f"ci_d{ci.d}n{ci.n_blocks}-{topology_label(point.topology)}"
    )


def submit_point(
    point: SweepPoint,
    cfg: SweepConfig,
    master_port: int,
    do_submit: bool,
) -> tuple[str, str | None]:
    """Materialize a single sweep point and (optionally) submit it.

    Returns ``(point_name, slurm_job_id_or_None)``. If ``do_submit=False`` only
    prints the resolved plan and returns ``(name, None)``.
    """
    name = point_name(cfg.name_prefix, point)
    job_dir = Path(GEN_ROOT) / name
    label = topology_label(point.topology)

    topology_yaml, world, n_nodes, pool_b = render_topology(point.topology, cfg.model.n_layers)
    assert point.batch % pool_b == 0, (
        f"batch={point.batch} not divisible by pool_b={pool_b} (topology={label})"
    )
    assert point.batch % point.topology.ddp == 0, (
        f"batch={point.batch} not divisible by ddp={point.topology.ddp}"
    )

    run_yaml = render_run_yaml(name=name, model=cfg.model, point=point, topology_label=label)
    sbatch_text = render_sbatch(
        name=name,
        n_nodes=n_nodes,
        job_dir=job_dir,
        runtime=cfg.runtime,
        master_port=master_port,
    )

    ci = point.ci.resolved()
    print(
        f"[sweep] {name}  world={world}  nodes={n_nodes}  pool_b={pool_b}  "
        f"batch={point.batch}  seq={point.seq}  ci=(d={ci.d}, n={ci.n_blocks})"
    )
    if not do_submit:
        return name, None

    # Validate before writing so a schema mismatch fails fast (and we don't
    # leave a stale job.sbatch on disk pointing at a broken run.yaml).
    validate_run_yaml(run_yaml)

    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "run.yaml").write_text(run_yaml)
    (job_dir / "topology.yaml").write_text(topology_yaml)
    sbatch_path = job_dir / "job.sbatch"
    sbatch_path.write_text(sbatch_text)
    sbatch_path.chmod(0o755)
    Path(SLURM_LOGS).mkdir(parents=True, exist_ok=True)
    out = subprocess.run(["sbatch", str(sbatch_path)], capture_output=True, text=True, check=True)
    job_id = out.stdout.strip().split()[-1]
    print(f"  → {out.stdout.strip()}")
    return name, job_id
